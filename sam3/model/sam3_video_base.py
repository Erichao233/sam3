# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import datetime
import logging
import math
import os
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Set

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
import torch.nn.functional as F

from sam3 import perflib
from sam3.logger import get_logger
from sam3.model.box_ops import fast_diag_box_iou
from sam3.model.data_misc import BatchedDatapoint
from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores, mask_to_box
from sam3.perflib.masks_ops import mask_iou
from sam3.train.masks_ops import rle_encode
from torch import nn, Tensor

logger = get_logger(__name__)


class MaskletConfirmationStatus(Enum):
    UNCONFIRMED = 1  # newly added masklet, not confirmed by any detection yet
    CONFIRMED = 2  # confirmed by at least one detection


class SPMEGateMLP(nn.Module):
    """
    Lightweight gating network for learned SPME.

    Inputs per object (default: `in_dim=5`, configurable via `SAM3_SPME_LEARNED_GATE_INPUTS`):
      1) query_cos            [-1, 1]  (detector pointer consistency)
      2) det_score            [0, 1]  (detector confidence)
      3) tracker_score        [0, 1]  (tracker mask quality proxy; optional)
      4) mask_area_delta      [0, 1]  (tracker area change proxy; optional)
      5) last_occluded_frames [0, 1]  (occlusion length proxy; optional)

    "Clean" / detector-only modes are supported for ablations:
      - det3: (query_cos, det_score, presence_prob)
      - det4: (query_cos, det_score, presence_prob, last_occluded_frames_det)

    Outputs per object:
      - mem_scale:  (mem_dim,) in [0, 1] for channel-wise (FiLM-style) memory write/blend
      - mem_offset: (mem_dim,) bounded residual offset for memory (tanh * offset_scale)
      - gate_decay: (1,) in [0, 1] for forgetting/decay (optional)
    """

    def __init__(
        self,
        mem_dim: int,
        in_dim: int = 5,
        hidden_dim: int = 64,
        init_bias_scale: float = 4.0,
        init_bias_decay: float = -4.0,
        offset_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.mem_dim = int(mem_dim)
        self.offset_scale = float(offset_scale)
        out_dim = 2 * self.mem_dim + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

        # Start close to "do nothing" behavior (scale≈1, offset≈0, decay≈0).
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.net[-1].bias.data[: self.mem_dim] = float(init_bias_scale)
        self.net[-1].bias.data[2 * self.mem_dim : 2 * self.mem_dim + 1] = float(init_bias_decay)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        y = self.net(x.to(dtype=torch.float32))
        mem_scale = torch.sigmoid(y[..., : self.mem_dim])
        mem_offset = torch.tanh(y[..., self.mem_dim : 2 * self.mem_dim]) * self.offset_scale
        gate_decay = torch.sigmoid(y[..., 2 * self.mem_dim : 2 * self.mem_dim + 1])
        return mem_scale, mem_offset, gate_decay


class Sam3VideoBase(nn.Module):
    @staticmethod
    def _cosine_sim_1d(a: Tensor, b: Tensor) -> float:
        a = a.reshape(-1).float()
        b = b.reshape(-1).float()
        denom = (a.norm() * b.norm()).clamp_min(1e-12)
        return float((a.dot(b) / denom).clamp(-1.0, 1.0).item())

    def _maybe_attach_spme_signals(
        self,
        sam3_image_out: Dict[str, Tensor],
        det_out: Dict[str, Tensor],
        feature_cache: Dict,
        frame_idx: int,
    ) -> None:
        """
        Attach small per-frame SPME signals to `det_out` (single-prompt use cases):
        - `spme_query_cos`: cosine similarity between current semantic pointer and an anchor pointer
        - `spme_det_score_raw`: raw detection probability used for gating (defaults to top-1, can be top-k max)
        - `spme_query_vec`: the current semantic pointer vector (for optional feature fusion)
        - `spme_presence_prob`: sigmoid(presence_logit_dec) for presence-aware gating (often ~1 on ultrasound)

        Notes:
        - Anchor selection defaults to the first frame; set `SAM3_SPME_ANCHOR_DET_THR>0` to
          delay anchor initialization until the detector is confident enough.
        - Optional anchor refresh (off by default) can be enabled via `SAM3_SPME_ANCHOR_REFRESH=1`.
          This keeps a reference anchor (`spme_query_anchor_ref`) and updates the running anchor
          (`spme_query_anchor`) using an identity-safe rule:
            - det_score_raw >= `SAM3_SPME_ANCHOR_REFRESH_DET_THR`
            - cos(current_ptr, ref_anchor) >= `SAM3_SPME_ANCHOR_REFRESH_SAFE_COS`
            - running anchor update uses EMA with `SAM3_SPME_ANCHOR_REFRESH_EMA` (0..1).
        - Multi-query pooling (Top-K weighted average) can be enabled via:
          `SAM3_SPME_QUERY_POOL=topk_weighted` and `SAM3_SPME_QUERY_TOPK>=2`.
        """
        if (
            os.getenv("SAM3_SPME_WRITE_GATE", "0") != "1"
            and os.getenv("SAM3_SPME_DEBUG_SIGNALS", "0") != "1"
            and os.getenv("SAM3_SPME_FUSION", "0") != "1"
        ):
            return

        # Presence (global existence signal from the detector decoder).
        presence_logit = sam3_image_out.get("presence_logit_dec", None)
        if isinstance(presence_logit, torch.Tensor) and presence_logit.numel() > 0:
            det_out["spme_presence_prob"] = float(
                presence_logit.detach().float().sigmoid().flatten()[0].clamp(0.0, 1.0).item()
            )

        # Query vector source: top-1 raw query (default) or Top-K weighted pooling.
        pool = os.getenv("SAM3_SPME_QUERY_POOL", "top1").strip().lower()
        ptr: Tensor | None = None
        ptr_score: float | None = None

        if pool in {"topk", "topk_weighted", "topk_weight"}:
            qk = sam3_image_out.get("query_topk_raw", None)
            sk = sam3_image_out.get("query_topk_score_raw", None)
            if isinstance(qk, torch.Tensor) and isinstance(sk, torch.Tensor) and qk.numel() > 0 and sk.numel() > 0:
                # qk: (P, K, D), sk: (P, K); single-prompt use cases use P=0.
                q0 = qk[0] if qk.ndim == 3 else qk
                s0 = sk[0] if sk.ndim == 2 else sk
                s0f = s0.detach().float().clamp(0.0, 1.0)
                if s0f.numel() > 0:
                    ptr_score = float(s0f.max().item())
                    temp = float(os.getenv("SAM3_SPME_QUERY_POOL_TEMP", "0.0"))
                    if temp and temp > 0.0:
                        w = torch.softmax(s0f / temp, dim=0)
                    else:
                        w = s0f / s0f.sum().clamp_min(1e-6)
                    ptr = (q0.detach().float() * w[:, None]).sum(dim=0)
                    ptr = ptr.to(device=q0.device, dtype=q0.dtype)

        if ptr is None:
            score_raw = sam3_image_out.get("query_top1_score_raw", None)
            if isinstance(score_raw, torch.Tensor) and score_raw.numel() > 0:
                ptr_score = float(score_raw.detach().float().flatten()[0].clamp(0.0, 1.0).item())

            q = sam3_image_out.get("query_top1_raw", None)
            if isinstance(q, torch.Tensor) and q.numel() > 0:
                q0 = q[0] if q.ndim == 2 else q
                ptr = q0.detach()

        if ptr_score is not None:
            det_out["spme_det_score_raw"] = float(ptr_score)
            det_out["spme_det_frame_idx"] = int(frame_idx)

        if isinstance(ptr, torch.Tensor) and ptr.numel() > 0:
            det_out["spme_query_vec"] = ptr

            refresh = os.getenv("SAM3_SPME_ANCHOR_REFRESH", "0") == "1"
            skip_geom_prompt = os.getenv("SAM3_SPME_ANCHOR_SKIP_GEOM_PROMPT", "0") == "1"
            has_geom_prompt = bool(det_out.get("spme_has_geometric_prompt", False))
            refresh_det_thr = float(os.getenv("SAM3_SPME_ANCHOR_REFRESH_DET_THR", "0.9"))
            refresh_safe_cos = float(os.getenv("SAM3_SPME_ANCHOR_REFRESH_SAFE_COS", "0.9"))
            refresh_ema = float(os.getenv("SAM3_SPME_ANCHOR_REFRESH_EMA", "0.05"))
            refresh_ema = float(max(0.0, min(1.0, refresh_ema)))

            anchor = feature_cache.get("spme_query_anchor", None)
            anchor_ref = feature_cache.get("spme_query_anchor_ref", None) if refresh else None

            if not isinstance(anchor, torch.Tensor) or anchor.numel() == 0:
                if skip_geom_prompt and has_geom_prompt:
                    # Delay anchor init until a frame without geometric prompts (e.g., init uses GT box).
                    return
                anchor_det_thr = float(os.getenv("SAM3_SPME_ANCHOR_DET_THR", "0.0"))
                det_score_f = det_out.get("spme_det_score_raw", None)
                if (
                    det_score_f is not None
                    and anchor_det_thr > 0.0
                    and float(det_score_f) < anchor_det_thr
                ):
                    # Delay anchor selection until the detector is confident enough.
                    return
                feature_cache["spme_query_anchor"] = ptr
                if refresh:
                    feature_cache.setdefault("spme_query_anchor_ref", ptr)
                det_out["spme_query_cos"] = 1.0
            else:
                anchor_dev = anchor.to(device=ptr.device)
                det_out["spme_query_cos"] = self._cosine_sim_1d(anchor_dev, ptr)

                # Optional identity-safe anchor refresh (updates the running anchor only).
                if refresh and not (skip_geom_prompt and has_geom_prompt):
                    if not isinstance(anchor_ref, torch.Tensor) or anchor_ref.numel() == 0:
                        feature_cache["spme_query_anchor_ref"] = anchor.detach()
                        anchor_ref = feature_cache["spme_query_anchor_ref"]

                    det_score_f = det_out.get("spme_det_score_raw", None)
                    det_score_ff = float(det_score_f) if det_score_f is not None else 0.0
                    if det_score_ff >= refresh_det_thr and refresh_ema > 0.0:
                        ref_dev = anchor_ref.to(device=ptr.device)
                        safe_cos = self._cosine_sim_1d(ref_dev, ptr)
                        if safe_cos >= refresh_safe_cos:
                            old = anchor_dev.detach().float()
                            q = ptr.detach().float()
                            new = (1.0 - refresh_ema) * old + refresh_ema * q
                            new = new / new.norm().clamp_min(1e-12)
                            feature_cache["spme_query_anchor"] = new.to(dtype=anchor.dtype).detach()

    def _compute_spme_write_gate(self, det_out: Dict[str, Any], frame_idx: int) -> float | None:
        """
        Compute a scalar gate in [0, 1] controlling memory write strength.

        Env vars:
        - `SAM3_SPME_WRITE_GATE=1` enables gating.
        - `SAM3_SPME_WRITE_GATE_MODE` in {hard, soft}. Default: hard.
        - `SAM3_SPME_DET_THR` (float). Default: 0.0. If >0, only apply gate when det_score_raw >= det_thr.
        - `SAM3_SPME_QCOS_THR` (float). Default: 0.8.
        - `SAM3_SPME_USE_DET_SCORE` in {0,1}. Default: 1 (only used in soft mode).
        """
        if os.getenv("SAM3_SPME_WRITE_GATE", "0") != "1":
            return None

        mode = os.getenv("SAM3_SPME_WRITE_GATE_MODE", "hard").strip().lower()
        det_thr = float(os.getenv("SAM3_SPME_DET_THR", "0.0"))
        q_thr = float(os.getenv("SAM3_SPME_QCOS_THR", "0.8"))
        qcos = det_out.get("spme_query_cos", None)
        if qcos is None:
            return 1.0  # safe fallback
        qcos_f = float(qcos)

        det_score = det_out.get("spme_det_score_raw", None)
        det_score_f = float(det_score) if det_score is not None else None
        if det_score_f is not None and det_thr > 0.0 and det_score_f < det_thr:
            return 1.0

        if mode == "hard":
            gate = 1.0 if qcos_f >= q_thr else 0.0
        else:
            denom = max(1e-6, 1.0 - q_thr)
            gate = max(0.0, min(1.0, (qcos_f - q_thr) / denom))
            if os.getenv("SAM3_SPME_USE_DET_SCORE", "1") == "1":
                if det_score_f is not None:
                    gate *= max(0.0, min(1.0, det_score_f))

        if os.getenv("SAM3_SPME_DEBUG_SIGNALS", "0") == "1" and self.rank == 0:
            logger.info(
                f"[SPME] frame={frame_idx} gate={gate:.3f} qcos={qcos_f:.3f} "
                f"det={det_score_f}"
            )
        return float(gate)

    def __init__(
        self,
        detector: nn.Module,
        tracker: nn.Module,
        # prob threshold for detection outputs -- only keep detections above this threshold
        # enters NMS and det-to-track matching
        score_threshold_detection=0.5,
        # IoU threshold for detection NMS
        det_nms_thresh=0.0,
        # IoU threshold for det-to-track matching -- a detection is considered "matched" to a tracklet it
        # overlaps with a tracklet above this threshold -- it is often a loose threshold like 0.1
        assoc_iou_thresh=0.5,
        # IoU threshold for det-to-track matching, which is used to determine whether a masklet is "unmatched"
        # by any detections -- it is often a stricter threshold like 0.5
        trk_assoc_iou_thresh=0.5,
        # prob threshold for a detection to be added as a new object
        new_det_thresh=0.0,
        # hotstart parameters: we hold off the outputs for `hotstart_delay` frames and
        # 1) remove those tracklets unmatched by any detections based on `hotstart_unmatch_thresh`
        # 2) remove those tracklets overlapping with one another based on `hotstart_dup_thresh`
        hotstart_delay=0,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=3,
        # Whether to suppress masks only within hotstart. If False, we can suppress masks even if they start before hotstart period.
        suppress_unmatched_only_within_hotstart=True,
        init_trk_keep_alive=0,
        max_trk_keep_alive=8,
        min_trk_keep_alive=-4,
        # Threshold for suppressing overlapping objects based on recent occlusion
        suppress_overlapping_based_on_recent_occlusion_threshold=0.0,
        decrease_trk_keep_alive_for_empty_masklets=False,
        o2o_matching_masklets_enable=False,  # Enable hungarian matching to match existing masklets
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        # The maximum number of objects (masklets) to track across all GPUs (for no limit, set it to -1)
        max_num_objects=-1,
        recondition_every_nth_frame=-1,
        # masket confirmation status (to suppress unconfirmed masklets)
        masklet_confirmation_enable=False,
        # a masklet is confirmed after being consecutively detected and matched for
        # `masklet_confirmation_consecutive_det_thresh`
        masklet_confirmation_consecutive_det_thresh=3,
        # bbox heuristic parameters
        reconstruction_bbox_iou_thresh=0.0,
        reconstruction_bbox_det_score=0.0,
    ):
        super().__init__()
        self.detector = detector
        self.tracker = tracker

        # ------------------------------------------------------------------
        # SPME-Fusion (trainable) modules
        # ------------------------------------------------------------------
        # These are optional and only take effect when enabled via env vars.
        # They are initialized to be identity/no-op (last layer weights = 0),
        # so existing checkpoints can be used without hurting baseline behavior.
        sem_dim = 256
        mem_dim = int(getattr(self.tracker, "mem_dim", 64))
        obj_dim = int(getattr(self.tracker, "hidden_dim", 256))

        # FiLM params for maskmem_features: output (gamma, beta) each of size mem_dim.
        self.spme_fusion_mem_film_mlp = nn.Sequential(
            nn.Linear(sem_dim, sem_dim),
            nn.GELU(),
            nn.Linear(sem_dim, 2 * mem_dim),
        )
        nn.init.zeros_(self.spme_fusion_mem_film_mlp[-1].weight)
        nn.init.zeros_(self.spme_fusion_mem_film_mlp[-1].bias)

        # Residual delta for obj_ptr (safer than FiLM on the pointer; keep scale small).
        self.spme_fusion_obj_ptr_mlp = nn.Sequential(
            nn.Linear(sem_dim, sem_dim),
            nn.GELU(),
            nn.Linear(sem_dim, obj_dim),
        )
        nn.init.zeros_(self.spme_fusion_obj_ptr_mlp[-1].weight)
        nn.init.zeros_(self.spme_fusion_obj_ptr_mlp[-1].bias)

        # ------------------------------------------------------------------
        # SPME Learned Gate (optional)
        # ------------------------------------------------------------------
        self.spme_gate_mlp: SPMEGateMLP | None = None
        self.spme_gate_inputs_mode: str | None = None
        self.spme_gate_in_dim: int | None = None
        self.spme_gate_fusion_mlp: nn.Module | None = None

        if os.getenv("SAM3_SPME_LEARNED_GATE", "0") == "1":
            gate_hidden = int(os.getenv("SAM3_SPME_GATE_HIDDEN", "64"))
            gate_inputs = os.getenv("SAM3_SPME_LEARNED_GATE_INPUTS", "full").strip().lower()
            if gate_inputs in {"", "full", "full5", "full_5", "5"}:
                gate_inputs_mode = "full"
                gate_in_dim = 5
            elif gate_inputs in {"det3", "det", "det_only", "clean", "detector_only"}:
                gate_inputs_mode = "det3"
                gate_in_dim = 3
            elif gate_inputs in {"det4", "det_occ", "det+occ", "det_with_occ", "detector_occ"}:
                gate_inputs_mode = "det4"
                gate_in_dim = 4
            else:
                raise ValueError(
                    "Unknown SAM3_SPME_LEARNED_GATE_INPUTS="
                    f"{gate_inputs!r}. Supported: full, det3, det4."
                )
            self.spme_gate_inputs_mode = gate_inputs_mode
            self.spme_gate_in_dim = int(gate_in_dim)
            self.spme_gate_mlp = SPMEGateMLP(
                mem_dim=mem_dim,
                in_dim=int(gate_in_dim),
                hidden_dim=gate_hidden,
            )
            # Optional: decouple fusion gating from mem_scale statistics.
            #
            # When enabled, the fusion injection strength is predicted by a separate tiny head
            # (still conditioned on the same gate inputs), instead of reusing mean(mem_scale).
            # This is ablation-friendly and keeps backward compatibility (disabled by default).
            if os.getenv("SAM3_SPME_LEARNED_GATE_FUSION_HEAD", "0") == "1":
                self.spme_gate_fusion_mlp = nn.Sequential(
                    nn.Linear(int(gate_in_dim), gate_hidden),
                    nn.GELU(),
                    nn.Linear(gate_hidden, 1),
                )
                nn.init.zeros_(self.spme_gate_fusion_mlp[-1].weight)
                nn.init.zeros_(self.spme_gate_fusion_mlp[-1].bias)
                # Start close to "no suppression" behavior (sigmoid(4)≈0.982).
                self.spme_gate_fusion_mlp[-1].bias.data[:] = 4.0

        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh

        # hotstart parameters
        if hotstart_delay > 0:
            assert hotstart_unmatch_thresh <= hotstart_delay
            assert hotstart_dup_thresh <= hotstart_delay
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.suppress_unmatched_only_within_hotstart = (
            suppress_unmatched_only_within_hotstart
        )
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.suppress_det_close_to_boundary = suppress_det_close_to_boundary
        self.decrease_trk_keep_alive_for_empty_masklets = (
            decrease_trk_keep_alive_for_empty_masklets
        )
        self.o2o_matching_masklets_enable = o2o_matching_masklets_enable
        self.fill_hole_area = fill_hole_area
        self.eval()
        self.rank = int(os.getenv("RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        self._dist_pg_cpu = None  # CPU process group (lazy-initialized on first use)

        # the maximum object number
        if max_num_objects > 0:
            num_obj_for_compile = math.ceil(max_num_objects / self.world_size)
        else:
            max_num_objects = 10000  # no limit
            num_obj_for_compile = 16
        logger.info(f"setting {max_num_objects=} and {num_obj_for_compile=}")
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.masklet_confirmation_enable = masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = (
            masklet_confirmation_consecutive_det_thresh
        )
        self.reconstruction_bbox_iou_thresh = reconstruction_bbox_iou_thresh
        self.reconstruction_bbox_det_score = reconstruction_bbox_det_score

    @property
    def device(self):
        self._device = getattr(self, "_device", None) or next(self.parameters()).device
        return self._device

    def _init_dist_pg_cpu(self):
        # a short 3-min timeout to quickly detect any synchronization failures
        timeout_sec = int(os.getenv("SAM3_COLLECTIVE_OP_TIMEOUT_SEC", "180"))
        timeout = datetime.timedelta(seconds=timeout_sec)
        self._dist_pg_cpu = dist.new_group(backend="gloo", timeout=timeout)

    def broadcast_python_obj_cpu(self, python_obj_list, src):
        if self._dist_pg_cpu is None:
            self._init_dist_pg_cpu()
        dist.broadcast_object_list(python_obj_list, src=src, group=self._dist_pg_cpu)

    def _det_track_one_frame(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        tracker_states_local: List[Any],
        tracker_metadata_prev: Dict[str, Any],
        feature_cache: Dict,
        orig_vid_height: int,
        orig_vid_width: int,
        is_image_only: bool = False,
        allow_add_new_objects: bool = True,
    ):
        """
        This function handles one-step inference for the DenseTracking model in an SPMD manner.
        At a high-level, all GPUs execute the same function calls as if it's done on a single GPU,
        while under the hood, some function calls involve distributed computation based on sharded
        SAM2 states.

        - `input_batch` contains image and other inputs on the entire video; it should be identical across GPUs
        - `tracker_states_local` holds the local masklet information in this GPU shard
        - `tracker_metadata_prev` manages the metadata for SAM2 objects, such as which masklet is hold on which GPUs
          it contains both global and local masklet information
        """

        # Step 1: run backbone and detector in a distributed manner -- this is done via Sam3ImageOnVideoMultiGPU,
        # a MultiGPU model (assigned to `self.detector`) that shards frames in a round-robin manner.
        # It returns a "det_out" dict for `frame_idx` and fills SAM2 backbone features for `frame_idx`
        # into `feature_cache`. Despite its distributed inference under the hood, the results would be
        # the same as if it is running backbone and detector for every frame on a single GPU.
        det_out = self.run_backbone_and_detection(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            feature_cache=feature_cache,
        )

        # Step 2: each GPU propagates its local SAM2 states to get the SAM2 prediction masks.
        # the returned `tracker_low_res_masks_global` contains the concatenated masklet predictions
        # gathered from all GPUs (as if they are propagated on a single GPU). Note that this step only
        # runs the SAM2 propagation step, but doesn't encode new memory for the predicted masks;
        # we defer memory encoding to `run_tracker_update_execution_phase` after resolving all heuristics.
        if tracker_metadata_prev == {}:
            # initialize masklet metadata if it's uninitialized (empty dict)
            tracker_metadata_prev.update(self._initialize_metadata())
        tracker_low_res_masks_global, tracker_obj_scores_global = (
            self.run_tracker_propagation(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                tracker_states_local=tracker_states_local,
                tracker_metadata_prev=tracker_metadata_prev,
            )
        )

        # Step 3: based on detection outputs and the propagated SAM2 prediction masks, we make plans
        # for SAM2 masklet updates (i.e. which objects to add and remove, how to load-balance them, etc).
        # We also run SAM2 memory encoder globally in this step to resolve non-overlapping constraints.
        # **This step should involve all the heuristics needed for any updates.** Most of the update
        # planning will be done on the master rank (GPU 0) and the resulting plan `tracker_update_plan` is
        # broadcasted to other GPUs (to be executed in a distributed manner). This step also generates the
        # new masklet metadata `tracker_metadata_new` (based on its previous version `tracker_metadata_prev`).
        tracker_update_plan, tracker_metadata_new = (
            self.run_tracker_update_planning_phase(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                det_out=det_out,
                tracker_low_res_masks_global=tracker_low_res_masks_global,
                tracker_obj_scores_global=tracker_obj_scores_global,
                tracker_metadata_prev=tracker_metadata_prev,
                tracker_states_local=tracker_states_local,
                feature_cache=feature_cache,
                is_image_only=is_image_only,
                allow_add_new_objects=allow_add_new_objects,
            )
        )

        # Get reconditioning info from the update plan
        reconditioned_obj_ids = tracker_update_plan.get("reconditioned_obj_ids", set())
        det_to_matched_trk_obj_ids = tracker_update_plan.get(
            "det_to_matched_trk_obj_ids", {}
        )

        # Step 4: based on `tracker_update_plan`, each GPU executes the update w.r.t. its local SAM2 inference states
        tracker_states_local_new = self.run_tracker_update_execution_phase(
            frame_idx=frame_idx,
            num_frames=num_frames,
            reverse=reverse,
            det_out=det_out,
            tracker_states_local=tracker_states_local,
            tracker_update_plan=tracker_update_plan,
            orig_vid_height=orig_vid_height,
            orig_vid_width=orig_vid_width,
            feature_cache=feature_cache,
        )

        # Step 5: finally, build the outputs for this frame (it only needs to be done on GPU 0 since
        # only GPU 0 will send outputs to the server).
        if self.rank == 0:
            obj_id_to_mask = self.build_outputs(
                frame_idx=frame_idx,
                num_frames=num_frames,
                reverse=reverse,
                det_out=det_out,
                tracker_low_res_masks_global=tracker_low_res_masks_global,
                tracker_obj_scores_global=tracker_obj_scores_global,
                tracker_metadata_prev=tracker_metadata_prev,
                tracker_update_plan=tracker_update_plan,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                reconditioned_obj_ids=reconditioned_obj_ids,
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
            )
            obj_id_to_score = tracker_metadata_new["obj_id_to_score"]
        else:
            obj_id_to_mask, obj_id_to_score = {}, {}  # dummy outputs on other GPUs
        # a few statistics for the current frame as a part of the output
        frame_stats = {
            "num_obj_tracked": np.sum(tracker_metadata_new["num_obj_per_gpu"]),
            "num_obj_dropped": tracker_update_plan["num_obj_dropped_due_to_limit"],
        }
        # add tracker scores to metadata, it should be fired for frames except the first frame
        if tracker_obj_scores_global.shape[0] > 0:
            # Convert tracker_obj_scores_global to sigmoid scores before updating
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid().tolist()
            tracker_obj_ids = tracker_metadata_prev["obj_ids_all_gpu"]
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][
                frame_idx
            ].update(dict(zip(tracker_obj_ids, tracker_obj_scores_global)))
        return (
            obj_id_to_mask,  # a dict: obj_id --> output mask
            obj_id_to_score,  # a dict: obj_id --> output score (prob)
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,  # a dict: obj_id --> tracker frame-level scores
        )

    def _suppress_detections_close_to_boundary(self, boxes, margin=0.025):
        """
        Suppress detections too close to image edges (for normalized boxes).

        boxes: (N, 4) in xyxy format, normalized [0,1]
        margin: fraction of image
        """
        x_min, y_min, x_max, y_max = boxes.unbind(-1)
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        keep = (
            (x_c > margin)
            & (x_c < 1.0 - margin)
            & (y_c > margin)
            & (y_c < 1.0 - margin)
        )

        return keep

    def run_backbone_and_detection(
        self,
        frame_idx: int,
        num_frames: int,
        input_batch: BatchedDatapoint,
        geometric_prompt: Any,
        feature_cache: Dict,
        reverse: bool,
    ):
        # Step 1: if text feature is not cached in `feature_cache`, compute and cache it
        text_batch_key = tuple(input_batch.find_text_batch)
        if "text" not in feature_cache or text_batch_key not in feature_cache["text"]:
            text_outputs = self.detector.backbone.forward_text(
                input_batch.find_text_batch, device=self.device
            )
            # note: we only cache the text feature of the most recent prompt
            feature_cache["text"] = {text_batch_key: text_outputs}
        else:
            text_outputs = feature_cache["text"][text_batch_key]

        # Step 2: run backbone, detector, and post-processing with NMS
        if "multigpu_buffer" not in feature_cache:
            # "multigpu_buffer" is a buffer cache used by `self.detector` and it needs
            # to be passed to `forward_video_grounding_multigpu` for every call
            feature_cache["multigpu_buffer"] = {}

        # Extract max_frame_num_to_track from feature_cache if available
        tracking_bounds = feature_cache.get("tracking_bounds", {})
        max_frame_num_to_track = tracking_bounds.get("max_frame_num_to_track")
        start_frame_idx = tracking_bounds.get("propagate_in_video_start_frame_idx")

        sam3_image_out, _ = self.detector.forward_video_grounding_multigpu(
            backbone_out={
                "img_batch_all_stages": input_batch.img_batch,
                **text_outputs,
            },
            find_inputs=input_batch.find_inputs,
            geometric_prompt=geometric_prompt,
            frame_idx=frame_idx,
            num_frames=num_frames,
            multigpu_buffer=feature_cache["multigpu_buffer"],
            track_in_reverse=reverse,
            # also get the SAM2 backbone features
            return_tracker_backbone_feats=True,
            # run NMS as a part of distributed computation
            run_nms=self.det_nms_thresh > 0.0,
            nms_prob_thresh=self.score_threshold_detection,
            nms_iou_thresh=self.det_nms_thresh,
            # pass max_frame_num_to_track to respect tracking limits
            max_frame_num_to_track=max_frame_num_to_track,
            propagate_in_video_start_frame_idx=start_frame_idx,
        )
        # note: detections in `sam3_image_out` has already gone through NMS
        # Cast to float32 before sigmoid to reduce underflow in AMP/FP16.
        pred_probs = sam3_image_out["pred_logits"].float().squeeze(-1).sigmoid()
        pred_boxes_xyxy = sam3_image_out["pred_boxes_xyxy"]
        pred_masks = sam3_image_out["pred_masks"]
        # get the positive detection outputs above threshold
        pos_pred_idx = torch.where(pred_probs > self.score_threshold_detection)
        det_out = {
            "bbox": pred_boxes_xyxy[pos_pred_idx[0], pos_pred_idx[1]],
            "mask": pred_masks[pos_pred_idx[0], pos_pred_idx[1]],
            "scores": pred_probs[pos_pred_idx[0], pos_pred_idx[1]],
        }
        # Mark whether this frame includes a geometric prompt (box/point/mask) in addition to text.
        # Useful to avoid mixing prompt-conditioned pointer distributions when building SPME anchors.
        has_geom_prompt = False
        gp = geometric_prompt
        if gp is not None:
            for attr in ("box_embeddings", "point_embeddings", "mask_embeddings"):
                if getattr(gp, attr, None) is not None:
                    has_geom_prompt = True
                    break
        det_out["spme_has_geometric_prompt"] = bool(has_geom_prompt)
        # Keep per-detection query vectors (aligned with det_out indices).
        # This enables per-object semantic pointers via det↔trk matching.
        queries = sam3_image_out.get("queries", None)
        if isinstance(queries, torch.Tensor) and queries.numel() > 0:
            det_out["query_vecs"] = queries[pos_pred_idx[0], pos_pred_idx[1]]
            det_out["query_prompt_idx"] = pos_pred_idx[0]
            det_out["query_idx"] = pos_pred_idx[1]
        self._maybe_attach_spme_signals(sam3_image_out, det_out, feature_cache, frame_idx)

        # Step 3: build SAM2 backbone features and store them in `feature_cache`
        backbone_cache = {}
        sam_mask_decoder = self.tracker.sam_mask_decoder
        tracker_backbone_fpn = [
            sam_mask_decoder.conv_s0(sam3_image_out["tracker_backbone_fpn_0"]),
            sam_mask_decoder.conv_s1(sam3_image_out["tracker_backbone_fpn_1"]),
            sam3_image_out["tracker_backbone_fpn_2"],  # fpn_2 doesn't need conv
        ]
        tracker_backbone_out = {
            "vision_features": tracker_backbone_fpn[-1],  # top-level feature
            "vision_pos_enc": sam3_image_out["tracker_backbone_pos_enc"],
            "backbone_fpn": tracker_backbone_fpn,
        }
        backbone_cache["tracker_backbone_out"] = tracker_backbone_out
        feature_cache[frame_idx] = (
            input_batch.img_batch[frame_idx],
            backbone_cache,
        )
        # remove from `feature_cache` old features to save GPU memory
        feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)
        return det_out

    def _build_spme_per_object_context(
        self,
        *,
        det_out: Dict[str, Any],
        det_to_matched_trk_obj_ids: Dict[int, npt.NDArray],
        new_det_fa_inds: npt.NDArray,
        new_det_obj_ids: npt.NDArray,
        feature_cache: Dict,
        trk_obj_ids_all: npt.NDArray | None = None,
    ) -> Dict[str, Any] | None:
        """
        Build per-object semantic pointer signals for SPME-Fusion.

        This is intentionally lightweight and designed to degrade safely:
        - If disabled, or if per-detection query vectors are unavailable, returns None.
        - If matching is empty, returns None.

        We maintain per-object anchor pointers in `feature_cache["spme_obj_anchor"]` (CPU tensors),
        and compute per-object qcos against the anchor for gating.
        """
        if os.getenv("SAM3_SPME_PER_OBJECT", "0") != "1":
            return None

        q_det = det_out.get("query_vecs", None)
        scores = det_out.get("scores", None)
        if not (isinstance(q_det, torch.Tensor) and isinstance(scores, torch.Tensor)):
            return None
        if q_det.numel() == 0 or scores.numel() == 0:
            return None
        if not det_to_matched_trk_obj_ids:
            # Still initialize anchors for newly added objects if possible.
            det_to_matched_trk_obj_ids = {}

        # Pick one "best" detection per track (score-based tie-breaker; stable and broadcast-free).
        trk_best_det: Dict[int, int] = {}
        for det_idx, trk_obj_ids in det_to_matched_trk_obj_ids.items():
            if det_idx < 0 or det_idx >= int(scores.numel()):
                continue
            det_score_f = float(scores[det_idx].detach().float().clamp(0.0, 1.0).item())
            for obj_id in np.asarray(trk_obj_ids).astype(np.int64).tolist():
                prev_det_idx = trk_best_det.get(int(obj_id), None)
                if prev_det_idx is None:
                    trk_best_det[int(obj_id)] = int(det_idx)
                else:
                    prev_score_f = float(
                        scores[int(prev_det_idx)].detach().float().clamp(0.0, 1.0).item()
                    )
                    if det_score_f > prev_score_f:
                        trk_best_det[int(obj_id)] = int(det_idx)

        anchor_map = feature_cache.setdefault("spme_obj_anchor", {})
        refresh = os.getenv("SAM3_SPME_ANCHOR_REFRESH", "0") == "1"
        refresh_det_thr = float(os.getenv("SAM3_SPME_ANCHOR_REFRESH_DET_THR", "0.9"))
        refresh_safe_cos = float(os.getenv("SAM3_SPME_ANCHOR_REFRESH_SAFE_COS", "0.9"))
        refresh_ema = float(os.getenv("SAM3_SPME_ANCHOR_REFRESH_EMA", "0.05"))
        refresh_ema = float(max(0.0, min(1.0, refresh_ema)))
        anchor_ref_map = feature_cache.setdefault("spme_obj_anchor_ref", {}) if refresh else None
        anchor_det_thr = float(os.getenv("SAM3_SPME_ANCHOR_DET_THR", "0.0"))

        query_vec_by_obj: Dict[int, torch.Tensor] = {}
        det_idx_by_obj: Dict[int, int] = {}
        det_score_by_obj: Dict[int, float] = {}
        qcos_by_obj: Dict[int, float] = {}

        # Update anchors for newly created objects (best-effort; relies on new_det_* order).
        try:
            if (
                isinstance(new_det_fa_inds, np.ndarray)
                and isinstance(new_det_obj_ids, np.ndarray)
                and len(new_det_fa_inds) == len(new_det_obj_ids)
            ):
                for det_i, obj_id in zip(new_det_fa_inds.tolist(), new_det_obj_ids.tolist()):
                    det_i = int(det_i)
                    obj_id = int(obj_id)
                    if det_i < 0 or det_i >= int(scores.numel()):
                        continue
                    ds = float(scores[det_i].detach().float().clamp(0.0, 1.0).item())
                    if anchor_det_thr > 0.0 and ds < anchor_det_thr:
                        continue
                    if obj_id not in anchor_map:
                        a = q_det[det_i].detach().float().cpu()
                        anchor_map[obj_id] = a  # CPU anchor (running)
                        if refresh and anchor_ref_map is not None:
                            anchor_ref_map.setdefault(obj_id, a.clone())
        except Exception:
            # Never let anchor init crash tracking.
            pass

        for obj_id, det_idx in trk_best_det.items():
            if det_idx < 0 or det_idx >= int(scores.numel()):
                continue
            qv = q_det[det_idx]
            if not (isinstance(qv, torch.Tensor) and qv.numel() > 0):
                continue
            query_vec_by_obj[int(obj_id)] = qv.detach()
            det_idx_by_obj[int(obj_id)] = int(det_idx)
            det_score_f = float(scores[det_idx].detach().float().clamp(0.0, 1.0).item())
            det_score_by_obj[int(obj_id)] = det_score_f

            anchor = anchor_map.get(int(obj_id), None)
            if not isinstance(anchor, torch.Tensor) or anchor.numel() == 0:
                # Delay anchor selection until confident if requested.
                if anchor_det_thr > 0.0 and det_score_f < anchor_det_thr:
                    continue
                a = qv.detach().float().cpu()
                anchor_map[int(obj_id)] = a
                if refresh and anchor_ref_map is not None:
                    anchor_ref_map.setdefault(int(obj_id), a.clone())
                qcos_by_obj[int(obj_id)] = 1.0
            else:
                qcos_by_obj[int(obj_id)] = self._cosine_sim_1d(
                    anchor.to(device=qv.device), qv
                )

                # Optional identity-safe anchor refresh: update running anchor (CPU) using EMA,
                # gated by similarity to a fixed reference anchor.
                if refresh and anchor_ref_map is not None and refresh_ema > 0.0:
                    ref = anchor_ref_map.get(int(obj_id), None)
                    if not isinstance(ref, torch.Tensor) or ref.numel() == 0:
                        anchor_ref_map[int(obj_id)] = anchor.detach().float().cpu()
                        ref = anchor_ref_map[int(obj_id)]
                    if det_score_f >= refresh_det_thr:
                        safe_cos = self._cosine_sim_1d(ref.to(device=qv.device), qv)
                        if safe_cos >= refresh_safe_cos:
                            old = anchor.detach().float().cpu()
                            q_cpu = qv.detach().float().cpu()
                            new = (1.0 - refresh_ema) * old + refresh_ema * q_cpu
                            new = new / new.norm().clamp_min(1e-12)
                            anchor_map[int(obj_id)] = new

        # Optional: fill missing per-object association by anchor similarity (qcos), instead of mask IoU.
        #
        # Why this exists:
        # - SAM3's built-in det↔trk association uses mask IoU thresholds (default 0.5). When drift happens,
        #   IoU can drop and the association becomes empty, exactly when SPME should intervene.
        # - For SPME (gate/fusion), we can associate a detection to a tracked identity using the detector
        #   query embedding and a per-object anchor pointer (qcos), without requiring overlap.
        #
        # Safety:
        # - Only enabled when `SAM3_SPME_PER_OBJECT_ANCHOR_MATCH=1`.
        # - Requires an existing anchor for the object.
        # - Respects `SAM3_SPME_ANCHOR_DET_THR` as a minimum det-score filter (if set > 0).
        #
        # NOTE: This is identity association for editing / triggers. It does NOT alter which objects
        # are tracked by SAM3, and it does NOT create new objects.
        allow_anchor_match = os.getenv("SAM3_SPME_PER_OBJECT_ANCHOR_MATCH", "0") == "1"
        if allow_anchor_match and isinstance(trk_obj_ids_all, np.ndarray) and trk_obj_ids_all.size > 0:
            for obj_id in trk_obj_ids_all.astype(np.int64).tolist():
                obj_id_i = int(obj_id)
                if obj_id_i in det_idx_by_obj:
                    continue
                anchor = anchor_map.get(obj_id_i, None)
                if not isinstance(anchor, torch.Tensor) or anchor.numel() == 0:
                    continue

                best_idx: int | None = None
                best_cos: float | None = None
                for det_i in range(int(scores.numel())):
                    ds = float(scores[det_i].detach().float().clamp(0.0, 1.0).item())
                    if anchor_det_thr > 0.0 and ds < anchor_det_thr:
                        continue
                    qv = q_det[det_i]
                    if not (isinstance(qv, torch.Tensor) and qv.numel() > 0):
                        continue
                    cos = float(self._cosine_sim_1d(anchor.to(device=qv.device), qv))
                    if best_cos is None or cos > best_cos:
                        best_cos = cos
                        best_idx = int(det_i)
                if best_idx is None:
                    continue

                qv = q_det[best_idx]
                if not (isinstance(qv, torch.Tensor) and qv.numel() > 0):
                    continue
                query_vec_by_obj[obj_id_i] = qv.detach()
                det_idx_by_obj[obj_id_i] = int(best_idx)
                det_score_by_obj[obj_id_i] = float(
                    scores[best_idx].detach().float().clamp(0.0, 1.0).item()
                )
                qcos_by_obj[obj_id_i] = float(best_cos) if best_cos is not None else 0.0

        if not query_vec_by_obj:
            return None
        return {
            "spme_query_vec_by_obj": query_vec_by_obj,
            "spme_det_idx_by_obj": det_idx_by_obj,
            "spme_det_score_raw_by_obj": det_score_by_obj,
            "spme_query_cos_by_obj": qcos_by_obj,
        }

    def run_tracker_propagation(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        tracker_states_local: List[Any],
        tracker_metadata_prev: Dict[str, npt.NDArray],
    ):
        # Step 1: propagate the local SAM2 states to get the current frame's prediction
        # `low_res_masks_local` of the existing masklets on this GPU
        # - obj_ids_local: List[int] -- list of object IDs
        # - low_res_masks_local: Tensor -- (num_local_obj, H_mask, W_mask)
        obj_ids_local, low_res_masks_local, obj_scores_local = (
            self._propogate_tracker_one_frame_local_gpu(
                tracker_states_local, frame_idx=frame_idx, reverse=reverse
            )
        )

        assert np.all(
            obj_ids_local == tracker_metadata_prev["obj_ids_per_gpu"][self.rank]
        ), "{} != {}".format(
            obj_ids_local, tracker_metadata_prev["obj_ids_per_gpu"][self.rank]
        )

        # Step 2: all-gather `low_res_masks_local` into `low_res_masks_global`
        # - low_res_masks_global: Tensor -- (num_global_obj, H_mask, W_mask)
        _, H_mask, W_mask = low_res_masks_local.shape
        if self.world_size > 1:
            # `low_res_masks_local` and `obj_scores_local` need to be contiguous and float32
            # (they could be non-contiguous due to slicing and/or bfloat16 due to autocast)
            low_res_masks_local = low_res_masks_local.float().contiguous()
            obj_scores_local = obj_scores_local.float().contiguous()
            num_obj_this_gpu = tracker_metadata_prev["num_obj_per_gpu"][self.rank]
            assert low_res_masks_local.size(0) == num_obj_this_gpu
            assert obj_scores_local.size(0) == num_obj_this_gpu
            low_res_masks_peers = [
                low_res_masks_local.new_empty(num_obj, H_mask, W_mask)
                for num_obj in tracker_metadata_prev["num_obj_per_gpu"]
            ]
            obj_scores_peers = [
                obj_scores_local.new_empty(num_obj)
                for num_obj in tracker_metadata_prev["num_obj_per_gpu"]
            ]
            dist.all_gather(low_res_masks_peers, low_res_masks_local)
            dist.all_gather(obj_scores_peers, obj_scores_local)
            low_res_masks_global = torch.cat(low_res_masks_peers, dim=0)
            obj_scores_global = torch.cat(obj_scores_peers, dim=0)
        else:
            low_res_masks_global = low_res_masks_local
            obj_scores_global = obj_scores_local
        return low_res_masks_global, obj_scores_global

    def _recondition_masklets(
        self,
        frame_idx,
        det_out: Dict[str, Tensor],
        trk_id_to_max_iou_high_conf_det: List[int],
        tracker_states_local: List[Any],
        tracker_metadata: Dict[str, npt.NDArray],
        tracker_obj_scores_global: Tensor,
    ):
        # Recondition the masklets based on the new detections
        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            new_mask = det_out["mask"][det_idx : det_idx + 1]
            input_mask_res = self.tracker.input_mask_size
            new_mask_binary = (
                F.interpolate(
                    new_mask.unsqueeze(1),
                    size=(input_mask_res, input_mask_res),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)[0]
                > 0
            )
            HIGH_CONF_THRESH = 0.8
            reconditioned_states_idx = set()
            obj_idx = np.where(tracker_metadata["obj_ids_all_gpu"] == trk_obj_id)[
                0
            ].item()
            obj_score = tracker_obj_scores_global[obj_idx]
            for state_idx, inference_state in enumerate(tracker_states_local):
                if (
                    trk_obj_id in inference_state["obj_ids"]
                    # NOTE: Goal of this condition is to avoid reconditioning masks that are occluded/low qualiy.
                    # Unfortunately, these can get reconditioned anyway due to batching. We should consider removing these heuristics.
                    and obj_score > HIGH_CONF_THRESH
                ):
                    logger.debug(
                        f"Adding new mask for track {trk_obj_id} at frame {frame_idx}. Objects {inference_state['obj_ids']} are all reconditioned."
                    )
                    self.tracker.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=trk_obj_id,
                        mask=new_mask_binary,
                    )
                    reconditioned_states_idx.add(state_idx)

            for idx in reconditioned_states_idx:
                self.tracker.propagate_in_video_preflight(
                    tracker_states_local[idx], run_mem_encoder=True
                )
        return tracker_states_local

    def _spme_reid_extract_descriptor(
        self,
        *,
        frame_idx: int,
        feature_cache: Dict,
        mask_lr: Tensor,
    ) -> Tensor | None:
        """
        Extract a lightweight multi-scale appearance descriptor from the tracker backbone FPN,
        by masked average pooling within `mask_lr` (EndoVis-style re-ID).

        This is inspired by arXiv:2512.16880v1 Sec 3.3.3 ("Reference Feature Bank"):
        for each scale l, average the backbone feature map within the predicted mask.

        Returns:
            1D float32 CPU tensor (normalized), or None if features/mask are invalid.
        """
        try:
            cached = feature_cache.get(int(frame_idx), None)
            if not isinstance(cached, tuple) or len(cached) != 2:
                return None
            _, backbone_cache = cached
            if not isinstance(backbone_cache, dict):
                return None
            tracker_backbone_out = backbone_cache.get("tracker_backbone_out", None)
            if not isinstance(tracker_backbone_out, dict):
                return None
            fpn = tracker_backbone_out.get("backbone_fpn", None)
            if not isinstance(fpn, list) or len(fpn) == 0:
                return None

            if not isinstance(mask_lr, torch.Tensor):
                return None
            m = mask_lr
            if m.ndim != 2:
                return None
            if m.dtype != torch.bool:
                m = m > 0

            desc_parts: list[Tensor] = []
            for feat in fpn:
                if not isinstance(feat, torch.Tensor) or feat.numel() == 0:
                    continue
                f = feat
                if f.ndim == 4:
                    f = f[0]
                if f.ndim != 3:
                    continue
                C, H, W = int(f.shape[0]), int(f.shape[1]), int(f.shape[2])
                if C <= 0 or H <= 0 or W <= 0:
                    continue

                m_rs = F.interpolate(
                    m[None, None].float(),
                    size=(H, W),
                    mode="nearest",
                )[0, 0] > 0.5
                if not bool(m_rs.any().item()):
                    return None

                f_flat = f.reshape(C, H * W).float()
                m_flat = m_rs.reshape(H * W)
                pooled = f_flat[:, m_flat].mean(dim=1)
                pooled = F.normalize(pooled, dim=0, eps=1e-6)
                desc_parts.append(pooled.detach().cpu())

            if not desc_parts:
                return None
            desc = torch.cat(desc_parts, dim=0).float()
            desc = F.normalize(desc, dim=0, eps=1e-6)
            return desc
        except Exception:
            return None

    @staticmethod
    def _spme_reid_mean_sim(desc: Tensor, bank: list[Tensor]) -> float | None:
        try:
            if not isinstance(desc, torch.Tensor) or desc.numel() == 0:
                return None
            if not isinstance(bank, list) or len(bank) == 0:
                return None
            bank_t = torch.stack([b.float() for b in bank], dim=0)  # (N, D)
            bank_t = F.normalize(bank_t, dim=1, eps=1e-6)
            d = desc.float()
            d = F.normalize(d, dim=0, eps=1e-6)
            sims = (bank_t * d.view(1, -1)).sum(dim=1).clamp(-1.0, 1.0)
            return float(sims.mean().item())
        except Exception:
            return None

    def _spme_compute_reinit_det_idx_by_obj(
        self,
        *,
        frame_idx: int,
        reverse: bool,
        det_out: Dict[str, Any],
        per_obj_ctx: Dict[str, Any] | None,
        tracker_metadata_prev: Dict[str, npt.NDArray],
        tracker_low_res_masks_global: Tensor,
        tracker_obj_scores_global: Tensor,
        feature_cache: Dict,
    ) -> Dict[int, int]:
        """
        Identity-safe "re-init" (burst refresh) plan.

        When the tracker becomes uncertain for several frames (occlusion / drift), and the detector
        produces a high-confidence + identity-consistent match, we refresh the memory write mask to
        the detector mask for that frame.

        This is inference-time only and is designed for PROMPT=visual (no language prompts).
        """
        if os.getenv("SAM3_SPME_REINIT", "0") != "1":
            return {}
        if reverse and os.getenv("SAM3_SPME_REINIT_ALLOW_REVERSE", "0") != "1":
            return {}

        det_idx_by_obj = {}
        det_score_by_obj = {}
        qcos_by_obj = {}
        if isinstance(per_obj_ctx, dict):
            det_idx_by_obj = per_obj_ctx.get("spme_det_idx_by_obj", {}) or {}
            det_score_by_obj = per_obj_ctx.get("spme_det_score_raw_by_obj", {}) or {}
            qcos_by_obj = per_obj_ctx.get("spme_query_cos_by_obj", {}) or {}
        if not isinstance(det_idx_by_obj, dict):
            det_idx_by_obj = {}
        if not isinstance(det_score_by_obj, dict):
            det_score_by_obj = {}
        if not isinstance(qcos_by_obj, dict):
            qcos_by_obj = {}

        obj_ids_all = tracker_metadata_prev.get("obj_ids_all_gpu", None)
        if not isinstance(obj_ids_all, np.ndarray) or obj_ids_all.size == 0:
            return {}

        # Thresholds (minimal and interpretable).
        det_present_thr = float(os.getenv("SAM3_SPME_LEARNED_GATE_DET_PRESENT_THR", "0.3"))
        reinit_det_thr = float(os.getenv("SAM3_SPME_REINIT_DET_THR", "0.7"))
        reinit_qcos_thr = float(os.getenv("SAM3_SPME_REINIT_QCOS_THR", "0.7"))
        reinit_tracker_thr = float(os.getenv("SAM3_SPME_REINIT_TRACKER_THR", "0.8"))
        miss_thr = int(os.getenv("SAM3_SPME_REINIT_MISS_THR", "3"))
        confirm_thr = int(os.getenv("SAM3_SPME_REINIT_CONFIRM_THR", "2"))
        confirm_iou_thr = float(os.getenv("SAM3_SPME_REINIT_CONFIRM_IOU_THR", "0.2"))
        iou_thr = float(os.getenv("SAM3_SPME_REINIT_IOU_THR", "0.2"))
        mismatch_thr = int(os.getenv("SAM3_SPME_REINIT_MISMATCH_THR", "2"))
        enable_mismatch = os.getenv("SAM3_SPME_REINIT_ENABLE_MISMATCH", "0") == "1"
        # If the tracker mask is empty (lost), allowing re-init effectively becomes re-detection.
        # This is high-risk for false positives; keep disabled by default and evaluate as ablation.
        allow_empty = os.getenv("SAM3_SPME_REINIT_ALLOW_EMPTY", "0") == "1"
        cooldown_frames = int(os.getenv("SAM3_SPME_REINIT_COOLDOWN", "8"))
        miss_thr = int(max(1, miss_thr))
        confirm_thr = int(max(1, confirm_thr))
        mismatch_thr = int(max(1, mismatch_thr))
        cooldown_frames = int(max(0, cooldown_frames))

        # Persistent state on rank0 (feature_cache is local per rank; we only compute plan on rank0).
        state = feature_cache.setdefault("spme_reinit_state", {})
        # Re-init should be rare and safe. We track two streaks:
        # - fail_count: tracker is uncertain (low tracker_score) or empty
        # - det_good_count: detector is confident + identity-consistent (det_score/qcos)
        # We trigger only when both streaks persist, and when det↔trk overlap is sane.
        fail_map: Dict[int, int] = state.setdefault("fail_count", {})
        det_good_map: Dict[int, int] = state.setdefault("det_good_count", {})
        mismatch_map: Dict[int, int] = state.setdefault("mismatch_count", {})
        cooldown_map: Dict[int, int] = state.setdefault("cooldown", {})

        # Optional: feature-based re-identification (ReMeDI-style) to validate detector candidates,
        # enabling safer re-init / re-detection after long occlusions.
        #
        # Ref: arXiv:2512.16880v1 Sec 3.3.3.
        reid_enabled = os.getenv("SAM3_SPME_REID", "0") == "1"
        reid_bank_size = int(os.getenv("SAM3_SPME_REID_BANK_SIZE", "20"))
        reid_margin_thr = float(os.getenv("SAM3_SPME_REID_MARGIN", "0.01"))
        reid_self_thr = float(os.getenv("SAM3_SPME_REID_SELF_THR", "0.0"))
        reid_use_other_banks = os.getenv("SAM3_SPME_REID_USE_OTHER_BANKS", "1") == "1"
        reid_update_bank = os.getenv("SAM3_SPME_REID_UPDATE", "1") == "1"
        # If enabled, compare detector vs tracker consistency against the reference bank
        # and only take drift/mismatch actions when detector is *more* consistent.
        #
        # This prevents false triggers when det↔trk masks disagree due to detector domain shift
        # but the tracker is still correct (a common EndoVis failure mode).
        reid_compare_trk_det = os.getenv("SAM3_SPME_REID_COMPARE_TRK_DET", "0") == "1"
        reid_compare_margin = float(
            os.getenv("SAM3_SPME_REID_COMPARE_MARGIN", str(reid_margin_thr))
        )
        reid_bank_size = int(max(1, reid_bank_size))
        reid_best_sim_by_obj: Dict[int, float] = {}
        reid_margin_by_obj: Dict[int, float] = {}
        reid_accept_by_obj: Dict[int, int] = {}
        reid_bank_size_by_obj: Dict[int, int] = {}
        reid_trk_sim_by_obj: Dict[int, float] = {}
        reid_det_vs_trk_margin_by_obj: Dict[int, float] = {}
        bank_map: Dict[int, list[Tensor]] | None = None
        if reid_enabled:
            bank_map = feature_cache.setdefault("spme_reid_bank", {})
            if not isinstance(bank_map, dict):
                bank_map = None

        # Optional fallback: when det↔trk IoU matching fails (drift), retrieve a detector candidate by
        # anchor similarity (query embedding), independent of current overlap.
        q_det = det_out.get("query_vecs", None)
        scores = det_out.get("scores", None)
        det_masks = det_out.get("mask", None)
        anchor_map = feature_cache.get("spme_obj_anchor", {})

        id_to_trk_idx = {
            int(obj_id): int(i)
            for i, obj_id in enumerate(obj_ids_all.astype(np.int64).tolist())
        }

        plan: Dict[int, int] = {}
        for oid in obj_ids_all.astype(np.int64).tolist():
            obj_id = int(oid)

            # Cooldown bookkeeping.
            cd = int(cooldown_map.get(obj_id, 0))
            if cd > 0:
                cooldown_map[obj_id] = cd - 1

            fail_prev = int(fail_map.get(obj_id, 0))
            det_good_prev = int(det_good_map.get(obj_id, 0))
            mismatch_prev = int(mismatch_map.get(obj_id, 0))

            if int(cooldown_map.get(obj_id, 0)) > 0:
                continue

            det_idx = det_idx_by_obj.get(obj_id, None)
            det_score = float(det_score_by_obj.get(obj_id, 0.0))
            qcos = float(qcos_by_obj.get(obj_id, -1.0))

            # Tracker uncertainty / emptiness for this object.
            trk_idx = id_to_trk_idx.get(int(obj_id), None)
            if trk_idx is None:
                continue
            try:
                trk_score = float(
                    tracker_obj_scores_global[trk_idx].detach().float().item()
                )
            except Exception:
                trk_score = 0.0
            try:
                trk_bin = tracker_low_res_masks_global[trk_idx] > 0
                trk_empty = not bool(trk_bin.any().item())
            except Exception:
                trk_empty = False

            # ReID bank bootstrap/update using current tracker prediction.
            if reid_enabled and isinstance(bank_map, dict):
                bank = bank_map.get(int(obj_id), None)
                if bank is None:
                    bank = []
                    bank_map[int(obj_id)] = bank
                # Bootstrap once when empty (the init frame is GT-mask for our EndoVis protocol).
                if len(bank) == 0 and not bool(trk_empty):
                    desc0 = self._spme_reid_extract_descriptor(
                        frame_idx=int(frame_idx),
                        feature_cache=feature_cache,
                        mask_lr=trk_bin.detach(),
                    )
                    if isinstance(desc0, torch.Tensor):
                        bank.append(desc0)
                reid_bank_size_by_obj[int(obj_id)] = int(len(bank))

            # Fallback retrieval by anchor similarity when no matched detection exists.
            if det_idx is None and isinstance(anchor_map, dict):
                anchor = anchor_map.get(obj_id, None)
                if (
                    isinstance(anchor, torch.Tensor)
                    and isinstance(q_det, torch.Tensor)
                    and isinstance(scores, torch.Tensor)
                    and q_det.numel() > 0
                    and scores.numel() > 0
                ):
                    try:
                        a = anchor.detach().float()
                        if a.ndim > 1:
                            a = a.reshape(-1)
                        a = a / a.norm().clamp_min(1e-6)
                        a = a.to(device=q_det.device)

                        qv = q_det.detach().float()
                        if qv.ndim > 2:
                            qv = qv.reshape(int(qv.shape[0]), -1)
                        qv = qv / qv.norm(dim=1, keepdim=True).clamp_min(1e-6)
                        cos = torch.matmul(qv, a)

                        keep = scores.detach().float().clamp(0.0, 1.0) >= float(det_present_thr)
                        if bool(keep.any().item()):
                            cos_keep = cos.masked_fill(~keep, float("-inf"))
                            best = int(torch.argmax(cos_keep).item())
                            if math.isfinite(float(cos_keep[best].item())):
                                det_idx = best
                                det_score = float(scores[best].detach().float().clamp(0.0, 1.0).item())
                                qcos = float(cos[best].detach().float().clamp(-1.0, 1.0).item())
                    except Exception:
                        det_idx = None

            det_present = det_idx is not None and float(det_score) >= float(det_present_thr)

            # Optional: apply ReID validation on the matched detection candidate.
            reid_best_sim = float("nan")
            reid_margin = float("nan")
            reid_accept = False
            trk_self_sim = None
            if (
                reid_enabled
                and det_present
                and isinstance(bank_map, dict)
                and isinstance(det_masks, torch.Tensor)
                and int(det_masks.shape[0]) > int(det_idx)
                and int(det_idx) >= 0
            ):
                bank = bank_map.get(int(obj_id), None)
                if isinstance(bank, list) and len(bank) > 0:
                    # Optional: compute tracker consistency against the bank (for det-vs-trk comparison).
                    if reid_compare_trk_det and not bool(trk_empty):
                        desc_trk = self._spme_reid_extract_descriptor(
                            frame_idx=int(frame_idx),
                            feature_cache=feature_cache,
                            mask_lr=trk_bin.detach(),
                        )
                        if isinstance(desc_trk, torch.Tensor):
                            trk_self_sim = self._spme_reid_mean_sim(desc_trk, bank)
                    det_bin_for_desc = det_masks[int(det_idx)] > 0
                    desc = self._spme_reid_extract_descriptor(
                        frame_idx=int(frame_idx),
                        feature_cache=feature_cache,
                        mask_lr=det_bin_for_desc.detach(),
                    )
                    if isinstance(desc, torch.Tensor):
                        sself = self._spme_reid_mean_sim(desc, bank)
                        if sself is None:
                            sself = 0.0
                        other_best = None
                        if reid_use_other_banks:
                            for oid2, bank2 in bank_map.items():
                                if int(oid2) == int(obj_id):
                                    continue
                                if not isinstance(bank2, list) or len(bank2) == 0:
                                    continue
                                sim2 = self._spme_reid_mean_sim(desc, bank2)
                                if sim2 is None:
                                    continue
                                other_best = sim2 if other_best is None else max(other_best, sim2)
                        reid_best_sim = float(sself)
                        if other_best is None:
                            # Single-object tracking case: no cross-class banks exist, so margin is undefined.
                            # Fall back to an absolute self-similarity threshold.
                            reid_margin = float("nan")
                            reid_accept = reid_best_sim >= float(reid_self_thr)
                        else:
                            reid_margin = float(sself - float(other_best))
                            reid_accept = (reid_best_sim >= float(reid_self_thr)) and (
                                reid_margin >= float(reid_margin_thr)
                            )
            reid_best_sim_by_obj[int(obj_id)] = float(reid_best_sim)
            reid_margin_by_obj[int(obj_id)] = float(reid_margin)
            reid_accept_by_obj[int(obj_id)] = int(1 if reid_accept else 0)
            if trk_self_sim is None:
                reid_trk_sim_by_obj[int(obj_id)] = float("nan")
                reid_det_vs_trk_margin_by_obj[int(obj_id)] = float("nan")
            else:
                reid_trk_sim_by_obj[int(obj_id)] = float(trk_self_sim)
                try:
                    reid_det_vs_trk_margin_by_obj[int(obj_id)] = float(
                        float(reid_best_sim) - float(trk_self_sim)
                    )
                except Exception:
                    reid_det_vs_trk_margin_by_obj[int(obj_id)] = float("nan")

            det_good = det_present and det_score >= reinit_det_thr and qcos >= reinit_qcos_thr
            if reid_enabled:
                det_good = bool(det_good and reid_accept)
            det_good_new = det_good_prev + 1 if det_good else 0

            tracker_fail = bool(trk_empty) or float(trk_score) < float(reinit_tracker_thr)
            fail_new = fail_prev + 1 if tracker_fail else 0

            # det↔trk IoU for identity-safe sanity checks.
            iou_f = None
            if (
                det_present
                and isinstance(det_masks, torch.Tensor)
                and int(det_masks.shape[0]) > int(det_idx)
                and int(det_idx) >= 0
                and isinstance(tracker_low_res_masks_global, torch.Tensor)
                and int(tracker_low_res_masks_global.shape[0]) > 0
            ):
                try:
                    det_bin = det_masks[int(det_idx)] > 0
                    trk_bin = tracker_low_res_masks_global[trk_idx] > 0
                    iou = mask_iou(det_bin.unsqueeze(0), trk_bin.unsqueeze(0))
                    iou_f = float(iou.detach().float().clamp(0.0, 1.0).item())
                except Exception:
                    iou_f = None

            mismatch_new = 0
            if enable_mismatch and iou_f is not None:
                mismatch_new = mismatch_prev + 1 if iou_f < float(iou_thr) else 0

            # Online ReID bank update on *reliable* frames (avoid feature contamination).
            #
            # ReMeDI-SAM3 updates the reference feature bank online, but only from frames that are
            # both reliable and certain. Here we approximate that by requiring:
            # - tracker not empty + high tracker_score
            # - detector candidate present + confident + identity-consistent (det_score/qcos)
            # - det↔trk overlap not too small (IoU >= confirm_iou_thr)
            if (
                reid_enabled
                and reid_update_bank
                and isinstance(bank_map, dict)
                and not bool(trk_empty)
                and det_present
                and det_score >= float(reinit_det_thr)
                and qcos >= float(reinit_qcos_thr)
                and iou_f is not None
                and float(iou_f) >= float(confirm_iou_thr)
                and float(trk_score) >= float(reinit_tracker_thr)
            ):
                bank = bank_map.get(int(obj_id), None)
                if isinstance(bank, list):
                    desc_u = self._spme_reid_extract_descriptor(
                        frame_idx=int(frame_idx),
                        feature_cache=feature_cache,
                        mask_lr=trk_bin.detach(),
                    )
                    if isinstance(desc_u, torch.Tensor):
                        bank.append(desc_u)
                        if len(bank) > int(reid_bank_size):
                            # Keep the most recent descriptors.
                            bank[:] = bank[-int(reid_bank_size) :]
                    reid_bank_size_by_obj[int(obj_id)] = int(len(bank))

            can_refresh = det_good and (
                (not bool(trk_empty) and iou_f is not None and float(iou_f) >= float(confirm_iou_thr))
                or (bool(trk_empty) and bool(allow_empty))
            )

            triggered = False
            # Primary (safe) trigger: tracker is uncertain for K frames and the detector is stably
            # confident + identity-consistent for T frames, with a basic overlap sanity check.
            if can_refresh and fail_new >= miss_thr and det_good_new >= confirm_thr:
                triggered = True
            # Optional drift trigger (off by default): sustained det↔trk mismatch while tracker is
            # still uncertain. This is riskier and should be evaluated as an ablation.
            elif (
                enable_mismatch
                and det_good
                and mismatch_new >= mismatch_thr
                and (
                    not reid_compare_trk_det
                    or math.isnan(float(reid_det_vs_trk_margin_by_obj.get(int(obj_id), float("nan"))))
                    or float(reid_det_vs_trk_margin_by_obj.get(int(obj_id), float("nan")))
                    >= float(reid_compare_margin)
                )
                and (float(trk_score) < float(reinit_tracker_thr) or bool(reid_enabled))
                and (not bool(trk_empty) or bool(allow_empty))
            ):
                triggered = True

            if triggered and det_idx is not None:
                plan[obj_id] = int(det_idx)
                fail_new = 0
                det_good_new = 0
                mismatch_new = 0
                cooldown_map[obj_id] = cooldown_frames

            fail_map[obj_id] = int(fail_new)
            det_good_map[obj_id] = int(det_good_new)
            mismatch_map[obj_id] = int(mismatch_new)

        # Stash ReID stats for logging in `_tracker_update_memories` (best-effort).
        if reid_enabled:
            state["reid_best_sim_by_obj"] = reid_best_sim_by_obj
            state["reid_margin_by_obj"] = reid_margin_by_obj
            state["reid_accept_by_obj"] = reid_accept_by_obj
            state["reid_bank_size_by_obj"] = reid_bank_size_by_obj
            if reid_compare_trk_det:
                state["reid_trk_sim_by_obj"] = reid_trk_sim_by_obj
                state["reid_det_vs_trk_margin_by_obj"] = reid_det_vs_trk_margin_by_obj

        return plan

    def run_tracker_update_planning_phase(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        tracker_low_res_masks_global: Tensor,
        tracker_obj_scores_global: Tensor,
        tracker_metadata_prev: Dict[str, npt.NDArray],
        tracker_states_local: List[Any],
        feature_cache: Dict,
        is_image_only: bool = False,
        allow_add_new_objects: bool = True,
    ):
        # initialize new metadata from previous metadata (its values will be updated later)
        tracker_metadata_new = {
            "obj_ids_per_gpu": deepcopy(tracker_metadata_prev["obj_ids_per_gpu"]),
            "obj_ids_all_gpu": None,  # will be filled later
            "num_obj_per_gpu": deepcopy(tracker_metadata_prev["num_obj_per_gpu"]),
            "obj_id_to_score": deepcopy(tracker_metadata_prev["obj_id_to_score"]),
            "obj_id_to_tracker_score_frame_wise": deepcopy(
                tracker_metadata_prev["obj_id_to_tracker_score_frame_wise"]
            ),
            "obj_id_to_last_occluded": {},  # will be filled later
            "max_obj_id": deepcopy(tracker_metadata_prev["max_obj_id"]),
        }

        # Initialize reconditioned_obj_ids early to avoid UnboundLocalError
        reconditioned_obj_ids = set()

        # Step 1: make the update plan and resolve heuristics on GPU 0
        det_mask_preds: Tensor = det_out["mask"]  # low-res mask logits
        det_scores_np: npt.NDArray = det_out["scores"].float().cpu().numpy()
        det_bbox_xyxy: Tensor = det_out["bbox"]
        if self.rank == 0:
            # a) match detector and tracker masks and find new objects
            (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            ) = self._associate_det_trk(
                det_masks=det_mask_preds,
                det_scores_np=det_scores_np,
                trk_masks=tracker_low_res_masks_global,
                trk_obj_ids=tracker_metadata_prev["obj_ids_all_gpu"],
            )
            if self.suppress_det_close_to_boundary:
                keep = self._suppress_detections_close_to_boundary(
                    det_bbox_xyxy[new_det_fa_inds]
                )
                new_det_fa_inds = new_det_fa_inds[keep.cpu().numpy()]

            # check whether we've hit the maximum number of objects we can track (and if so, drop some detections)
            prev_obj_num = np.sum(tracker_metadata_prev["num_obj_per_gpu"])
            new_det_num = len(new_det_fa_inds)
            num_obj_dropped_due_to_limit = 0
            if not allow_add_new_objects:
                # Keep detections for matching/confirmation, but do not spawn new objects.
                new_det_fa_inds = np.asarray([], dtype=new_det_fa_inds.dtype)
                new_det_num = 0
            if not is_image_only and prev_obj_num + new_det_num > self.max_num_objects:
                logger.warning(
                    f"hitting {self.max_num_objects=} with {new_det_num=} and {prev_obj_num=}"
                )
                new_det_num_to_keep = self.max_num_objects - prev_obj_num
                num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
                new_det_fa_inds = self._drop_new_det_with_obj_limit(
                    new_det_fa_inds, det_scores_np, new_det_num_to_keep
                )
                assert len(new_det_fa_inds) == new_det_num_to_keep
                new_det_num = len(new_det_fa_inds)

            # assign object IDs to new detections and decide which GPU to place them
            new_det_start_obj_id = tracker_metadata_prev["max_obj_id"] + 1
            new_det_obj_ids = new_det_start_obj_id + np.arange(new_det_num)
            prev_workload_per_gpu = tracker_metadata_prev["num_obj_per_gpu"]
            new_det_gpu_ids = self._assign_new_det_to_gpus(
                new_det_num=new_det_num,
                prev_workload_per_gpu=prev_workload_per_gpu,
            )

            # b) handle hotstart heuristics to remove objects
            # here `rank0_metadata` contains metadata stored on (and only accessible to) GPU 0;
            # we avoid broadcasting them to other GPUs to save communication cost, assuming
            # that `rank0_metadata` is not needed by other GPUs
            rank0_metadata_new = deepcopy(tracker_metadata_prev["rank0_metadata"])
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                obj_ids_newly_removed, rank0_metadata_new = self._process_hotstart(
                    frame_idx=frame_idx,
                    num_frames=num_frames,
                    reverse=reverse,
                    det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                    new_det_obj_ids=new_det_obj_ids,
                    empty_trk_obj_ids=empty_trk_obj_ids,
                    unmatched_trk_obj_ids=unmatched_trk_obj_ids,
                    rank0_metadata=rank0_metadata_new,
                    tracker_metadata=tracker_metadata_prev,
                )
            else:
                # if warm-up is not complete, we don't remove any objects
                obj_ids_newly_removed = set()
            tracker_metadata_new["rank0_metadata"] = rank0_metadata_new

        # Step 2: broadcast the update plan to other GPUs
        NUM_BROADCAST_ITEMS = 9
        if self.rank == 0 and self.world_size > 1:
            # `num_obj_per_gpu_on_rank0` is used for metadata consistency check on other GPUs
            # (it's a small array with length==self.world_size, so broadcasting it is cheap)
            num_obj_per_gpu_on_rank0 = tracker_metadata_prev["num_obj_per_gpu"]
            update_plan = [
                new_det_fa_inds,
                new_det_obj_ids,
                new_det_gpu_ids,
                num_obj_per_gpu_on_rank0,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                obj_ids_newly_removed,
                num_obj_dropped_due_to_limit,
                trk_id_to_max_iou_high_conf_det,
            ]
            assert (
                len(update_plan) == NUM_BROADCAST_ITEMS
            ), f"Manually update NUM_BROADCAST_ITEMS to be: {len(update_plan)}"
            self.broadcast_python_obj_cpu(update_plan, src=0)
        elif self.rank > 0 and self.world_size > 1:
            update_plan = [
                None
            ] * NUM_BROADCAST_ITEMS  # other ranks receive the plan from rank 0
            self.broadcast_python_obj_cpu(update_plan, src=0)
            (
                new_det_fa_inds,
                new_det_obj_ids,
                new_det_gpu_ids,
                num_obj_per_gpu_on_rank0,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                obj_ids_newly_removed,
                num_obj_dropped_due_to_limit,
                trk_id_to_max_iou_high_conf_det,
            ) = update_plan
            # metadata consistency check: verify that the received `num_obj_per_gpu_on_rank0` is consistent with the local metadata
            # it's critical that all GPUs agree on the previous number of objects (otherwise the inference might hang or fail silently)
            if not np.all(
                num_obj_per_gpu_on_rank0 == tracker_metadata_prev["num_obj_per_gpu"]
            ):
                raise RuntimeError(
                    f"{self.rank=} received {num_obj_per_gpu_on_rank0=}, which is inconsistent with local record "
                    f"{tracker_metadata_prev['num_obj_per_gpu']=}. There's likely a bug in update planning or execution."
                )

        # `tracker_update_plan` should be identical on all GPUs after broadcasting
        tracker_update_plan = {
            "new_det_fa_inds": new_det_fa_inds,  # npt.NDArray
            "new_det_obj_ids": new_det_obj_ids,  # npt.NDArray
            "new_det_gpu_ids": new_det_gpu_ids,  # npt.NDArray
            "unmatched_trk_obj_ids": unmatched_trk_obj_ids,  # npt.NDArray
            "det_to_matched_trk_obj_ids": det_to_matched_trk_obj_ids,  # dict
            "obj_ids_newly_removed": obj_ids_newly_removed,  # set
            "num_obj_dropped_due_to_limit": num_obj_dropped_due_to_limit,  # int
            "trk_id_to_max_iou_high_conf_det": trk_id_to_max_iou_high_conf_det,  # dict
            "reconditioned_obj_ids": reconditioned_obj_ids,  # set
        }

        # Step 3 (optional): recondition masklets based on high-confidence detections before memory encoding
        # NOTE: Running this in execution phase (after memory encoding) can lead to suboptimal results
        should_recondition_iou = False

        # Evaluate tracklets for reconditioning based on bbox IoU mismatch with detections
        if (
            self.reconstruction_bbox_iou_thresh > 0
            and len(trk_id_to_max_iou_high_conf_det) > 0
        ):
            for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
                det_box = det_out["bbox"][det_idx]
                det_score = det_out["scores"][det_idx]

                try:
                    trk_idx = list(tracker_metadata_prev["obj_ids_all_gpu"]).index(
                        trk_obj_id
                    )
                except ValueError:
                    continue  # Skip if tracklet not found

                tracker_mask = tracker_low_res_masks_global[trk_idx]
                mask_binary = tracker_mask > 0
                mask_area = mask_binary.sum().item()

                if mask_area == 0:
                    continue  # Skip tracklets with zero mask area

                # Get bounding box from SAM2 mask and convert to normalized coordinates
                tracker_box_pixels = (
                    mask_to_box(mask_binary.unsqueeze(0).unsqueeze(0))
                    .squeeze(0)
                    .squeeze(0)
                )
                mask_height, mask_width = tracker_mask.shape[-2:]
                tracker_box_normalized = torch.tensor(
                    [
                        tracker_box_pixels[0] / mask_width,
                        tracker_box_pixels[1] / mask_height,
                        tracker_box_pixels[2] / mask_width,
                        tracker_box_pixels[3] / mask_height,
                    ],
                    device=tracker_box_pixels.device,
                )

                # Compute IoU between detection and SAM2 tracklet bounding boxes
                det_box_batch = det_box.unsqueeze(0)
                tracker_box_batch = tracker_box_normalized.unsqueeze(0)
                iou = fast_diag_box_iou(det_box_batch, tracker_box_batch)[0]

                if (
                    iou < self.reconstruction_bbox_iou_thresh
                    and det_score >= self.reconstruction_bbox_det_score
                ):
                    should_recondition_iou = True
                    reconditioned_obj_ids.add(trk_obj_id)

        should_recondition_periodic = (
            self.recondition_every_nth_frame > 0
            and frame_idx % self.recondition_every_nth_frame == 0
            and len(trk_id_to_max_iou_high_conf_det) > 0
        )

        # Recondition if periodic or IoU condition met
        if should_recondition_periodic or should_recondition_iou:
            self._recondition_masklets(
                frame_idx,
                det_out,
                trk_id_to_max_iou_high_conf_det,
                tracker_states_local,
                tracker_metadata_prev,
                tracker_obj_scores_global,
            )

        # Step 4: Run SAM2 memory encoder on the current frame's prediction masks
        # This is done on all GPUs
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                if self.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0:
                    # NOTE: tracker_low_res_masks_global is updated in-place then returned
                    tracker_low_res_masks_global = (
                        self._suppress_overlapping_based_on_recent_occlusion(
                            frame_idx,
                            tracker_low_res_masks_global,
                            tracker_metadata_prev,
                            tracker_metadata_new,
                            obj_ids_newly_removed,
                            reverse,
                        )
                    )

            per_obj_ctx = self._build_spme_per_object_context(
                det_out=det_out,
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_fa_inds=new_det_fa_inds,
                new_det_obj_ids=new_det_obj_ids,
                feature_cache=feature_cache,
                trk_obj_ids_all=tracker_metadata_prev.get("obj_ids_all_gpu", None),
            )

            # Optional: identity-safe burst refresh (detector-guided re-init).
            #
            # We compute the plan on rank0, broadcast it, and override the masks used for
            # memory encoding on this frame for selected objects. This targets long-occlusion
            # recovery / drift, without relying on language prompts.
            spme_reinit_det_idx_by_obj: Dict[int, int] = {}
            if os.getenv("SAM3_SPME_REINIT", "0") == "1":
                if self.rank == 0:
                    spme_reinit_det_idx_by_obj = self._spme_compute_reinit_det_idx_by_obj(
                        frame_idx=frame_idx,
                        reverse=reverse,
                        det_out=det_out,
                        per_obj_ctx=per_obj_ctx,
                        tracker_metadata_prev=tracker_metadata_prev,
                        tracker_low_res_masks_global=tracker_low_res_masks_global,
                        tracker_obj_scores_global=tracker_obj_scores_global,
                        feature_cache=feature_cache,
                    )
                if self.world_size > 1:
                    plan_list = [spme_reinit_det_idx_by_obj] if self.rank == 0 else [None]
                    self.broadcast_python_obj_cpu(plan_list, src=0)
                    spme_reinit_det_idx_by_obj = plan_list[0] or {}

                if spme_reinit_det_idx_by_obj:
                    try:
                        obj_ids_all = tracker_metadata_prev.get("obj_ids_all_gpu", None)
                        if isinstance(obj_ids_all, np.ndarray):
                            id_to_idx = {
                                int(obj_id): int(i)
                                for i, obj_id in enumerate(obj_ids_all.astype(np.int64).tolist())
                            }
                            for obj_id, det_idx in spme_reinit_det_idx_by_obj.items():
                                trk_idx = id_to_idx.get(int(obj_id), None)
                                if trk_idx is None:
                                    continue
                                di = int(det_idx)
                                if di < 0 or di >= int(det_mask_preds.shape[0]):
                                    continue
                                tracker_low_res_masks_global[trk_idx] = det_mask_preds[di].to(
                                    tracker_low_res_masks_global.device
                                )
                    except Exception:
                        # Best-effort: never break tracking.
                        spme_reinit_det_idx_by_obj = {}

            # Store in the update plan so rank0 can override outputs for the same frame.
            tracker_update_plan["spme_reinit_det_idx_by_obj"] = spme_reinit_det_idx_by_obj

            # Optional: ReID debug scalars (computed on rank0 inside `_spme_compute_reinit_det_idx_by_obj`).
            spme_reid_best_sim_by_obj: Dict[int, float] | None = None
            spme_reid_margin_by_obj: Dict[int, float] | None = None
            spme_reid_accept_by_obj: Dict[int, int] | None = None
            spme_reid_bank_size_by_obj: Dict[int, int] | None = None
            spme_reid_trk_sim_by_obj: Dict[int, float] | None = None
            spme_reid_det_vs_trk_margin_by_obj: Dict[int, float] | None = None
            if os.getenv("SAM3_SPME_REID", "0") == "1":
                if self.rank == 0:
                    st = feature_cache.get("spme_reinit_state", {})
                    if isinstance(st, dict):
                        spme_reid_best_sim_by_obj = st.get("reid_best_sim_by_obj", None)
                        spme_reid_margin_by_obj = st.get("reid_margin_by_obj", None)
                        spme_reid_accept_by_obj = st.get("reid_accept_by_obj", None)
                        spme_reid_bank_size_by_obj = st.get("reid_bank_size_by_obj", None)
                        spme_reid_trk_sim_by_obj = st.get("reid_trk_sim_by_obj", None)
                        spme_reid_det_vs_trk_margin_by_obj = st.get(
                            "reid_det_vs_trk_margin_by_obj", None
                        )
                if self.world_size > 1:
                    # Broadcast dicts from rank0 to keep logging consistent.
                    payload = (
                        [spme_reid_best_sim_by_obj] if self.rank == 0 else [None]
                    )
                    self.broadcast_python_obj_cpu(payload, src=0)
                    spme_reid_best_sim_by_obj = payload[0]

                    payload = [spme_reid_margin_by_obj] if self.rank == 0 else [None]
                    self.broadcast_python_obj_cpu(payload, src=0)
                    spme_reid_margin_by_obj = payload[0]

                    payload = [spme_reid_accept_by_obj] if self.rank == 0 else [None]
                    self.broadcast_python_obj_cpu(payload, src=0)
                    spme_reid_accept_by_obj = payload[0]

                    payload = (
                        [spme_reid_bank_size_by_obj] if self.rank == 0 else [None]
                    )
                    self.broadcast_python_obj_cpu(payload, src=0)
                    spme_reid_bank_size_by_obj = payload[0]

                    payload = (
                        [spme_reid_trk_sim_by_obj] if self.rank == 0 else [None]
                    )
                    self.broadcast_python_obj_cpu(payload, src=0)
                    spme_reid_trk_sim_by_obj = payload[0]

                    payload = (
                        [spme_reid_det_vs_trk_margin_by_obj] if self.rank == 0 else [None]
                    )
                    self.broadcast_python_obj_cpu(payload, src=0)
                    spme_reid_det_vs_trk_margin_by_obj = payload[0]

            # det↔trk IoU (per object) for drift-aware fusion triggers and debug logging.
            #
            # NOTE: We only compute IoU for objects with a matched detection index (from det↔trk matching),
            # and we skip empty tracker masks to avoid turning fusion into re-detection.
            spme_det_trk_iou_by_obj: Dict[int, float] | None = None
            try:
                det_idx_by_obj = (
                    per_obj_ctx.get("spme_det_idx_by_obj", None) if isinstance(per_obj_ctx, dict) else None
                )
                obj_ids_all = tracker_metadata_prev.get("obj_ids_all_gpu", None)
                if isinstance(det_idx_by_obj, dict) and det_idx_by_obj and isinstance(obj_ids_all, np.ndarray):
                    id_to_idx = {
                        int(obj_id): int(i)
                        for i, obj_id in enumerate(obj_ids_all.astype(np.int64).tolist())
                    }
                    spme_det_trk_iou_by_obj = {}
                    for oid, det_idx in det_idx_by_obj.items():
                        obj_id = int(oid)
                        trk_idx = id_to_idx.get(obj_id, None)
                        if trk_idx is None:
                            continue
                        di = int(det_idx)
                        if di < 0 or di >= int(det_mask_preds.shape[0]):
                            continue
                        trk_bin = tracker_low_res_masks_global[trk_idx] > 0
                        if not bool(trk_bin.any().item()):
                            continue
                        det_bin = det_mask_preds[di] > 0
                        iou = mask_iou(det_bin.unsqueeze(0), trk_bin.unsqueeze(0))
                        spme_det_trk_iou_by_obj[obj_id] = float(
                            iou.detach().float().clamp(0.0, 1.0).item()
                        )
                    if not spme_det_trk_iou_by_obj:
                        spme_det_trk_iou_by_obj = None
            except Exception:
                spme_det_trk_iou_by_obj = None

            self._tracker_update_memories(
                tracker_states_local,
                frame_idx,
                tracker_metadata=tracker_metadata_prev,
                low_res_masks=tracker_low_res_masks_global,
                tracker_obj_scores_global=tracker_obj_scores_global,
                spme_write_gate=self._compute_spme_write_gate(det_out, frame_idx),
                spme_query_vec=det_out.get("spme_query_vec", None),
                spme_det_score_raw=det_out.get("spme_det_score_raw", None),
                spme_query_cos=det_out.get("spme_query_cos", None),
                spme_presence_prob=det_out.get("spme_presence_prob", None),
                spme_query_vec_by_obj=(
                    per_obj_ctx.get("spme_query_vec_by_obj", None) if per_obj_ctx else None
                ),
                spme_det_score_raw_by_obj=(
                    per_obj_ctx.get("spme_det_score_raw_by_obj", None) if per_obj_ctx else None
                ),
                spme_query_cos_by_obj=(
                    per_obj_ctx.get("spme_query_cos_by_obj", None) if per_obj_ctx else None
                ),
                spme_det_trk_iou_by_obj=spme_det_trk_iou_by_obj,
                spme_reinit_by_obj=spme_reinit_det_idx_by_obj if spme_reinit_det_idx_by_obj else None,
                spme_reid_best_sim_by_obj=spme_reid_best_sim_by_obj,
                spme_reid_margin_by_obj=spme_reid_margin_by_obj,
                spme_reid_accept_by_obj=spme_reid_accept_by_obj,
                spme_reid_bank_size_by_obj=spme_reid_bank_size_by_obj,
                spme_reid_trk_sim_by_obj=spme_reid_trk_sim_by_obj,
                spme_reid_det_vs_trk_margin_by_obj=spme_reid_det_vs_trk_margin_by_obj,
                track_in_reverse=reverse,
            )

        # Step 4: update the SAM2 metadata based on the update plan
        # note: except for "rank0_metadata" (that is only available on GPU 0),
        # the updated `tracker_metadata_new` should be identical on all GPUs
        for rank in range(self.world_size):
            new_det_obj_ids_this_gpu = new_det_obj_ids[new_det_gpu_ids == rank]
            updated_obj_ids_this_gpu = tracker_metadata_new["obj_ids_per_gpu"][rank]
            if len(new_det_obj_ids_this_gpu) > 0:
                updated_obj_ids_this_gpu = np.concatenate(
                    [updated_obj_ids_this_gpu, new_det_obj_ids_this_gpu]
                )
            if len(obj_ids_newly_removed) > 0:
                is_removed = np.isin(
                    updated_obj_ids_this_gpu, list(obj_ids_newly_removed)
                )
                updated_obj_ids_this_gpu = updated_obj_ids_this_gpu[~is_removed]
            tracker_metadata_new["obj_ids_per_gpu"][rank] = updated_obj_ids_this_gpu
            tracker_metadata_new["num_obj_per_gpu"][rank] = len(
                updated_obj_ids_this_gpu
            )
        tracker_metadata_new["obj_ids_all_gpu"] = np.concatenate(
            tracker_metadata_new["obj_ids_per_gpu"]
        )
        # update object scores and the maximum object ID assigned so far
        if len(new_det_obj_ids) > 0:
            tracker_metadata_new["obj_id_to_score"].update(
                zip(new_det_obj_ids, det_scores_np[new_det_fa_inds])
            )
            # tracker scores are not available for new objects, use det score instead.
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][
                frame_idx
            ].update(zip(new_det_obj_ids, det_scores_np[new_det_fa_inds]))
            tracker_metadata_new["max_obj_id"] = max(
                tracker_metadata_new["max_obj_id"],
                np.max(new_det_obj_ids),
            )
        # for removed objects, we set their scores to a very low value (-1e4) but still
        # keep them in "obj_id_to_score" (it's easier to handle outputs this way)
        for obj_id in obj_ids_newly_removed:
            tracker_metadata_new["obj_id_to_score"][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx][
                obj_id
            ] = -1e4
            tracker_metadata_new["obj_id_to_last_occluded"].pop(obj_id, None)
        # check that "rank0_metadata" is in tracker_metadata_new if and only if it's GPU 0
        assert ("rank0_metadata" in tracker_metadata_new) == (self.rank == 0)
        if self.rank == 0 and self.masklet_confirmation_enable:
            rank0_metadata = self.update_masklet_confirmation_status(
                rank0_metadata=tracker_metadata_new["rank0_metadata"],
                obj_ids_all_gpu_prev=tracker_metadata_prev["obj_ids_all_gpu"],
                obj_ids_all_gpu_updated=tracker_metadata_new["obj_ids_all_gpu"],
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
            )
            tracker_metadata_new["rank0_metadata"] = rank0_metadata

        return tracker_update_plan, tracker_metadata_new

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        tracker_low_res_masks_global: Tensor,
        tracker_metadata_prev: Dict[str, Any],
        tracker_metadata_new: Dict[str, Any],
        obj_ids_newly_removed: Set[int],
        reverse: bool = False,
    ):
        """
        Suppress overlapping masks based on the most recent occlusion information. If an object is removed by hotstart, we always suppress it if it overlaps with any other object.
        Args:
            frame_idx (int): The current frame index.
            tracker_low_res_masks_global (Tensor): The low-resolution masks for the current frame.
            tracker_metadata_prev (Dict[str, Any]): The metadata from the previous frame.
            tracker_metadata_new (Dict[str, Any]): The metadata for the current frame.
            obj_ids_newly_removed (Set[int]): The object IDs that have been removed.
        Return:
            Tensor: The updated low-resolution masks with some objects suppressed.
        """
        obj_ids_global = tracker_metadata_prev["obj_ids_all_gpu"]
        binary_tracker_low_res_masks_global = tracker_low_res_masks_global > 0
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            assert (
                len(obj_ids_global) == batch_size
            ), f"Mismatch in number of objects: {len(obj_ids_global)} vs {batch_size}"
            NEVER_OCCLUDED = -1
            ALWAYS_OCCLUDED = 100000  # This value should be larger than any possible frame index, indicates that the object was removed by hotstart logic
            last_occluded_prev = torch.cat(
                [
                    tracker_metadata_prev["obj_id_to_last_occluded"].get(
                        obj_id,
                        torch.full(
                            (1,),
                            fill_value=(
                                NEVER_OCCLUDED
                                if obj_id not in obj_ids_newly_removed
                                else ALWAYS_OCCLUDED
                            ),
                            device=binary_tracker_low_res_masks_global.device,
                            dtype=torch.long,
                        ),
                    )
                    for obj_id in obj_ids_global
                ],
                dim=0,
            )
            to_suppress = self._get_objects_to_suppress_based_on_most_recently_occluded(
                binary_tracker_low_res_masks_global,
                last_occluded_prev,
                obj_ids_global,
                frame_idx,
                reverse,
            )

            # Update metadata with occlusion information
            is_obj_occluded = ~(binary_tracker_low_res_masks_global.any(dim=(-1, -2)))
            is_obj_occluded_or_suppressed = is_obj_occluded | to_suppress
            last_occluded_new = last_occluded_prev.clone()
            last_occluded_new[is_obj_occluded_or_suppressed] = frame_idx
            # Slice out the last occluded frame for each object
            tracker_metadata_new["obj_id_to_last_occluded"] = {
                obj_id: last_occluded_new[obj_idx : obj_idx + 1]
                for obj_idx, obj_id in enumerate(obj_ids_global)
            }

            # Zero out suppressed masks before memory encoding
            NO_OBJ_LOGIT = -10
            tracker_low_res_masks_global[to_suppress] = NO_OBJ_LOGIT

        return tracker_low_res_masks_global

    def run_tracker_update_execution_phase(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        tracker_states_local: List[Any],
        tracker_update_plan: Dict[str, npt.NDArray],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: Dict,
    ):
        # initialize tracking scores with detection scores
        new_det_fa_inds: npt.NDArray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: npt.NDArray = tracker_update_plan["new_det_obj_ids"]
        new_det_gpu_ids: npt.NDArray = tracker_update_plan["new_det_gpu_ids"]
        is_on_this_gpu: npt.NDArray = new_det_gpu_ids == self.rank
        new_det_obj_ids_local: npt.NDArray = new_det_obj_ids[is_on_this_gpu]
        new_det_fa_inds_local: npt.NDArray = new_det_fa_inds[is_on_this_gpu]
        obj_ids_newly_removed: Set[int] = tracker_update_plan["obj_ids_newly_removed"]

        # Step 1: add new objects from the detector to SAM2 inference states
        if len(new_det_fa_inds_local) > 0:
            new_det_fa_inds_local_t = torch.from_numpy(new_det_fa_inds_local)
            new_det_masks: Tensor = det_out["mask"][new_det_fa_inds_local_t]
            # initialize SAM2 with new object masks
            tracker_states_local = self._tracker_add_new_objects(
                frame_idx=frame_idx,
                num_frames=num_frames,
                new_obj_ids=new_det_obj_ids_local,
                new_obj_masks=new_det_masks,
                tracker_states_local=tracker_states_local,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                feature_cache=feature_cache,
            )

        # Step 2: remove from SAM2 inference states those objects removed by heuristics
        if len(obj_ids_newly_removed) > 0:
            self._tracker_remove_objects(tracker_states_local, obj_ids_newly_removed)

        return tracker_states_local

    def build_outputs(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_out: Dict[str, Tensor],
        tracker_low_res_masks_global: Tensor,
        tracker_obj_scores_global: Tensor,
        tracker_metadata_prev: Dict[str, npt.NDArray],
        tracker_update_plan: Dict[str, npt.NDArray],
        orig_vid_height: int,
        orig_vid_width: int,
        reconditioned_obj_ids: set = None,
        det_to_matched_trk_obj_ids: dict = None,
    ):
        new_det_fa_inds: npt.NDArray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: npt.NDArray = tracker_update_plan["new_det_obj_ids"]
        obj_id_to_mask = {}  # obj_id --> output mask tensor

        # Part 1: masks from previous SAM2 propagation
        existing_masklet_obj_ids = tracker_metadata_prev["obj_ids_all_gpu"]
        existing_masklet_video_res_masks = F.interpolate(
            tracker_low_res_masks_global.unsqueeze(1),
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )  # (num_obj, 1, H_video, W_video)
        existing_masklet_binary = existing_masklet_video_res_masks > 0
        assert len(existing_masklet_obj_ids) == len(existing_masklet_binary)
        for obj_id, mask in zip(existing_masklet_obj_ids, existing_masklet_binary):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        # Part 2: masks from new detections
        new_det_fa_inds_t = torch.from_numpy(new_det_fa_inds)
        new_det_low_res_masks = det_out["mask"][new_det_fa_inds_t].unsqueeze(1)
        new_det_low_res_masks = fill_holes_in_mask_scores(
            new_det_low_res_masks,
            max_area=self.fill_hole_area,
            fill_holes=True,
            remove_sprinkles=True,
        )
        new_masklet_video_res_masks = F.interpolate(
            new_det_low_res_masks,
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )  # (num_obj, 1, H_video, W_video)

        new_masklet_binary = new_masklet_video_res_masks > 0
        assert len(new_det_obj_ids) == len(new_masklet_video_res_masks)
        for obj_id, mask in zip(new_det_obj_ids, new_masklet_binary):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        # Part 3: Override masks for reconditioned objects using detection masks
        if reconditioned_obj_ids is not None and len(reconditioned_obj_ids) > 0:
            trk_id_to_max_iou_high_conf_det = tracker_update_plan.get(
                "trk_id_to_max_iou_high_conf_det", {}
            )

            for obj_id in reconditioned_obj_ids:
                det_idx = trk_id_to_max_iou_high_conf_det.get(obj_id)

                if det_idx is not None:
                    det_mask = det_out["mask"][det_idx]
                    det_mask = det_mask.unsqueeze(0).unsqueeze(0)
                    det_mask_resized = (
                        F.interpolate(
                            det_mask.float(),
                            size=(orig_vid_height, orig_vid_width),
                            mode="bilinear",
                            align_corners=False,
                        )
                        > 0
                    )

                    det_mask_final = det_mask_resized.squeeze(0)
                    obj_id_to_mask[obj_id] = det_mask_final

        # Part 4: Override masks for SPME re-init objects (identity-safe burst refresh).
        spme_reinit_det_idx_by_obj = tracker_update_plan.get("spme_reinit_det_idx_by_obj", {})
        if isinstance(spme_reinit_det_idx_by_obj, dict) and len(spme_reinit_det_idx_by_obj) > 0:
            for obj_id, det_idx in spme_reinit_det_idx_by_obj.items():
                try:
                    di = int(det_idx)
                except Exception:
                    continue
                if di < 0 or di >= int(det_out["mask"].shape[0]):
                    continue
                det_mask = det_out["mask"][di]
                det_mask = det_mask.unsqueeze(0).unsqueeze(0)
                det_mask_resized = (
                    F.interpolate(
                        det_mask.float(),
                        size=(orig_vid_height, orig_vid_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    > 0
                )
                obj_id_to_mask[int(obj_id)] = det_mask_resized.squeeze(0)

        return obj_id_to_mask

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: Tensor,
        last_occluded: List[int],
        obj_ids: List[int],
        frame_idx: int = None,
        reverse: bool = False,
    ):
        # Suppress overlapping masks for objects that were most recently occluded
        assert (
            binary_low_res_masks.dtype == torch.bool
        ), f"Expected boolean tensor, got {binary_low_res_masks.dtype}"
        to_suppress = torch.zeros(
            binary_low_res_masks.size(0),
            device=binary_low_res_masks.device,
            dtype=torch.bool,
        )
        if len(obj_ids) <= 1:
            return to_suppress

        iou = mask_iou(binary_low_res_masks, binary_low_res_masks)  # [N,N]

        # Create masks for upper triangular matrix (i < j) and IoU threshold
        mask_iou_thresh = (
            iou >= self.suppress_overlapping_based_on_recent_occlusion_threshold
        )
        overlapping_pairs = torch.triu(mask_iou_thresh, diagonal=1)  # [N,N]

        last_occ_expanded_i = last_occluded.unsqueeze(1)  # (N, 1)
        last_occ_expanded_j = last_occluded.unsqueeze(0)  # (1, N)
        # Suppress most recently occluded
        cmp_op = torch.gt if not reverse else torch.lt
        suppress_i_mask = (
            overlapping_pairs
            & cmp_op(
                last_occ_expanded_i, last_occ_expanded_j
            )  # (last_occ_expanded_i > last_occ_expanded_j)
            & (
                last_occ_expanded_j > -1
            )  # j can suppress i only if i was previously occluded
        )
        suppress_j_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_j, last_occ_expanded_i)
            & (
                last_occ_expanded_i > -1
            )  # i can suppress j only if j was previously occluded
        )
        # Apply suppression
        to_suppress = suppress_i_mask.any(dim=1) | suppress_j_mask.any(dim=0)

        # Log for debugging
        if (
            self.rank == 0
            and logger.isEnabledFor(logging.DEBUG)
            and frame_idx is not None
        ):
            suppress_i_mask = suppress_i_mask.cpu().numpy()
            suppress_j_mask = suppress_j_mask.cpu().numpy()
            last_occluded = last_occluded.cpu().numpy()

            # Find all suppression pairs without using torch.where
            batch_size = suppress_i_mask.shape[0]

            # Log i-suppression cases (where i gets suppressed in favor of j)
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_i_mask[i, j]:
                        logger.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[i]} last occluded {last_occluded[i]} in favor of {obj_ids[j]} last occluded {last_occluded[j]}"
                        )

            # Log j-suppression cases (where j gets suppressed in favor of i)
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_j_mask[i, j]:
                        logger.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[j]} last occluded {last_occluded[j]} in favor of {obj_ids[i]} last occluded {last_occluded[i]}"
                        )

        return to_suppress

    def _propogate_tracker_one_frame_local_gpu(
        self,
        inference_states: List[Any],
        frame_idx: int,
        reverse: bool,
        # by default, we disable memory encoding until we gather all outputs
        run_mem_encoder: bool = False,
    ):
        """
        inference_states: List of inference states, each state corresponds to a different set of objects.
        """
        obj_ids_local = []
        low_res_masks_list = []
        obj_scores_list = []
        for inference_state in inference_states:
            if len(inference_state["obj_ids"]) == 0:
                continue  # skip propagation on empty inference states

            # propagate one frame
            num_frames_propagated = 0
            for out in self.tracker.propagate_in_video(
                inference_state,
                start_frame_idx=frame_idx,
                # end_frame_idx = start_frame_idx + max_frame_num_to_track
                # (i.e. propagating 1 frame since end_frame_idx is inclusive)
                max_frame_num_to_track=0,
                reverse=reverse,
                tqdm_disable=True,
                run_mem_encoder=run_mem_encoder,
            ):
                out_frame_idx, out_obj_ids, out_low_res_masks, _, out_obj_scores = out
                num_frames_propagated += 1

            # only 1 frames should be propagated
            assert (
                num_frames_propagated == 1 and out_frame_idx == frame_idx
            ), f"num_frames_propagated: {num_frames_propagated}, out_frame_idx: {out_frame_idx}, frame_idx: {frame_idx}"
            assert isinstance(out_obj_ids, list)
            obj_ids_local.extend(out_obj_ids)
            low_res_masks_list.append(out_low_res_masks.squeeze(1))
            obj_scores_list.append(out_obj_scores.squeeze(1))

        # concatenate the output masklets from all local inference states
        H_mask = W_mask = self.tracker.low_res_mask_size
        if len(low_res_masks_list) > 0:
            low_res_masks_local = torch.cat(low_res_masks_list, dim=0)
            obj_scores_local = torch.cat(obj_scores_list, dim=0)
            assert low_res_masks_local.shape[1:] == (H_mask, W_mask)

            # Apply hole filling to the masks
            low_res_masks_local = fill_holes_in_mask_scores(
                low_res_masks_local.unsqueeze(1),
                max_area=self.fill_hole_area,
                fill_holes=True,
                remove_sprinkles=True,
            )
            low_res_masks_local = low_res_masks_local.squeeze(1)
        else:
            low_res_masks_local = torch.zeros(0, H_mask, W_mask, device=self.device)
            obj_scores_local = torch.zeros(0, device=self.device)

        return obj_ids_local, low_res_masks_local, obj_scores_local

    def _associate_det_trk(
        self,
        det_masks: Tensor,
        det_scores_np: npt.NDArray,
        trk_masks: Tensor,
        trk_obj_ids: npt.NDArray,
    ):
        """
        Match detections on the current frame with the existing masklets.

        Args:
          - det_masks: (N, H, W) tensor of predicted masks
          - det_scores_np: (N,) array of detection scores
          - trk_masks: (M, H, W) tensor of track masks
          - trk_obj_ids: (M,) array of object IDs corresponding to trk_masks

        Returns:
          - new_det_fa_inds: array of new object indices.
          - unmatched_trk_obj_ids: array of existing masklet object IDs that are not matched
            to any detections on this frame (for unmatched, we only count masklets with >0 area)
          - det_to_matched_trk_obj_ids: dict[int, npt.NDArray]: mapping from detector's detection indices
            to the list of matched tracklet object IDs
          - empty_trk_obj_ids: array of existing masklet object IDs with zero area in SAM2 prediction
        """
        iou_threshold = self.assoc_iou_thresh
        iou_threshold_trk = self.trk_assoc_iou_thresh
        new_det_thresh = self.new_det_thresh

        assert det_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert (
            trk_masks.size(0) == len(trk_obj_ids)
        ), f"trk_masks and trk_obj_ids should have the same length, {trk_masks.size(0)} vs {len(trk_obj_ids)}"
        if trk_masks.size(0) == 0:
            # all detections are new
            new_det_fa_inds = np.arange(det_masks.size(0))
            unmatched_trk_obj_ids = np.array([], np.int64)
            empty_trk_obj_ids = np.array([], np.int64)
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )
        elif det_masks.size(0) == 0:
            # all previous tracklets are unmatched if they have a non-zero area
            new_det_fa_inds = np.array([], np.int64)
            trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2)).cpu().numpy()
            unmatched_trk_obj_ids = trk_obj_ids[trk_is_nonempty]
            empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )

        if det_masks.shape[-2:] != trk_masks.shape[-2:]:
            # resize to the smaller size to save GPU memory
            if np.prod(det_masks.shape[-2:]) < np.prod(trk_masks.shape[-2:]):
                trk_masks = F.interpolate(
                    trk_masks.unsqueeze(1),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                # resize detections to track size
                det_masks = F.interpolate(
                    det_masks.unsqueeze(1),
                    size=trk_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        det_masks_binary = det_masks > 0
        trk_masks_binary = trk_masks > 0
        ious = mask_iou(det_masks_binary, trk_masks_binary)  # (N, M)

        ious_np = ious.cpu().numpy()
        if self.o2o_matching_masklets_enable:
            from scipy.optimize import linear_sum_assignment

            # Hungarian matching for tracks (one-to-one: each track matches at most one detection)
            cost_matrix = 1 - ious_np  # Hungarian solves for minimum cost
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            trk_is_matched = np.zeros(trk_masks.size(0), dtype=bool)
            for d, t in zip(row_ind, col_ind):
                if ious_np[d, t] >= iou_threshold_trk:
                    trk_is_matched[t] = True
        else:
            trk_is_matched = (ious_np >= iou_threshold_trk).any(axis=0)
        # Non-empty tracks not matched by Hungarian assignment above threshold are unmatched
        trk_is_nonempty = trk_masks_binary.any(dim=(1, 2)).cpu().numpy()
        trk_is_unmatched = np.logical_and(trk_is_nonempty, ~trk_is_matched)
        unmatched_trk_obj_ids = trk_obj_ids[trk_is_unmatched]
        # also record masklets that have zero area in SAM 2 prediction
        empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]

        # For detections: allow many tracks to match to the same detection (many-to-one)
        # So, a detection is 'new' if it does not match any track above threshold
        is_new_det = np.logical_and(
            det_scores_np >= new_det_thresh,
            np.logical_not(np.any(ious_np >= iou_threshold, axis=1)),
        )
        new_det_fa_inds = np.nonzero(is_new_det)[0]

        # for each detection, which tracks it matched to (above threshold)
        det_to_matched_trk_obj_ids = {}
        trk_id_to_max_iou_high_conf_det = {}  # trk id --> exactly one detection idx
        HIGH_CONF_THRESH = 0.8
        HIGH_IOU_THRESH = 0.8
        det_to_max_iou_trk_idx = np.argmax(ious_np, axis=1)
        det_is_high_conf = (det_scores_np >= HIGH_CONF_THRESH) & ~is_new_det
        det_is_high_iou = np.max(ious_np, axis=1) >= HIGH_IOU_THRESH
        det_is_high_conf_and_iou = set(
            np.nonzero(det_is_high_conf & det_is_high_iou)[0]
        )
        for d in range(det_masks.size(0)):
            det_to_matched_trk_obj_ids[d] = trk_obj_ids[ious_np[d, :] >= iou_threshold]
            if d in det_is_high_conf_and_iou:
                trk_obj_id = trk_obj_ids[det_to_max_iou_trk_idx[d]].item()
                trk_id_to_max_iou_high_conf_det[trk_obj_id] = d

        return (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        )

    def _assign_new_det_to_gpus(self, new_det_num, prev_workload_per_gpu):
        """Distribute the new objects to the GPUs with the least workload."""
        workload_per_gpu: npt.NDArray = prev_workload_per_gpu.copy()
        new_det_gpu_ids = np.zeros(new_det_num, np.int64)

        # assign the objects one by one
        for i in range(len(new_det_gpu_ids)):
            # find the GPU with the least workload
            min_gpu = np.argmin(workload_per_gpu)
            new_det_gpu_ids[i] = min_gpu
            workload_per_gpu[min_gpu] += 1
        return new_det_gpu_ids

    def _process_hotstart(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: Dict[int, npt.NDArray],
        new_det_obj_ids: npt.NDArray,
        empty_trk_obj_ids: npt.NDArray,
        unmatched_trk_obj_ids: npt.NDArray,
        rank0_metadata: Dict[str, Any],
        tracker_metadata: Dict[str, Any],
    ):
        """Handle hotstart heuristics to remove unmatched or duplicated objects."""
        # obj_id --> first frame index where the object was detected
        obj_first_frame_idx = rank0_metadata["obj_first_frame_idx"]
        # obj_id --> [mismatched frame indices]
        unmatched_frame_inds = rank0_metadata["unmatched_frame_inds"]
        trk_keep_alive = rank0_metadata["trk_keep_alive"]
        # (first_appear_obj_id, obj_id) --> [overlap frame indices]
        overlap_pair_to_frame_inds = rank0_metadata["overlap_pair_to_frame_inds"]
        # removed_obj_ids: object IDs that are suppressed via hot-start
        removed_obj_ids = rank0_metadata["removed_obj_ids"]
        suppressed_obj_ids = rank0_metadata["suppressed_obj_ids"][frame_idx]

        obj_ids_newly_removed = set()  # object IDs to be newly removed on this frame
        hotstart_diff = (
            frame_idx - self.hotstart_delay
            if not reverse
            else frame_idx + self.hotstart_delay
        )

        # Step 1: log the frame index where each object ID first appears
        for obj_id in new_det_obj_ids:
            if obj_id not in obj_first_frame_idx:
                obj_first_frame_idx[obj_id] = frame_idx
            assert obj_id not in trk_keep_alive
            trk_keep_alive[obj_id] = self.init_trk_keep_alive

        matched_trks = set()
        # We use the det-->tracks list to check for matched objects. Otherwise, we need to compute areas to decide whether they're occluded
        for matched_trks_per_det in det_to_matched_trk_obj_ids.values():
            matched_trks.update(matched_trks_per_det)
        for obj_id in matched_trks:
            # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the max value of trk_keep_alive
            trk_keep_alive[obj_id] = min(
                self.max_trk_keep_alive, trk_keep_alive[obj_id] + 1
            )
        for obj_id in unmatched_trk_obj_ids:
            unmatched_frame_inds[obj_id].append(frame_idx)
            # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the min value of trk_keep_alive
            # The max keep alive is 2x the min, means the model prefers to keep the prediction rather than suppress it if it was matched long enough.
            trk_keep_alive[obj_id] = max(
                self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1
            )
        if self.decrease_trk_keep_alive_for_empty_masklets:
            for obj_id in empty_trk_obj_ids:
                # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the min value of trk_keep_alive
                trk_keep_alive[obj_id] = max(
                    self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1
                )

        # Step 2: removed tracks that has not matched with detections for `hotstart_unmatch_thresh` frames with hotstart period
        # a) add unmatched frame indices for each existing object ID
        # note that `unmatched_trk_obj_ids` contains those frames where the SAM2 output mask
        # doesn't match any detection; it excludes those frames where SAM2 gives an empty mask
        # b) remove a masklet if it first appears after `hotstart_diff` and is unmatched for more
        # than `self.hotstart_unmatch_thresh` frames
        for obj_id, frame_indices in unmatched_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue  # skip if the object is already removed
            if len(frame_indices) >= self.hotstart_unmatch_thresh:
                is_within_hotstart = (
                    obj_first_frame_idx[obj_id] > hotstart_diff and not reverse
                ) or (obj_first_frame_idx[obj_id] < hotstart_diff and reverse)
                if is_within_hotstart:
                    obj_ids_newly_removed.add(obj_id)
                    logger.debug(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it is unmatched for frames: {frame_indices}"
                    )
            if (
                trk_keep_alive[obj_id] <= 0  # Object has not been matched for too long
                and not self.suppress_unmatched_only_within_hotstart
                and obj_id not in removed_obj_ids
                and obj_id not in obj_ids_newly_removed
            ):
                logger.debug(
                    f"Suppressing object {obj_id} at frame {frame_idx}, due to being unmatched"
                )
                suppressed_obj_ids.add(obj_id)

        # Step 3: removed tracks that overlaps with another track for `hotstart_dup_thresh` frames
        # a) find overlaps tracks -- we consider overlap if they match to the same detection
        for _, matched_trk_obj_ids in det_to_matched_trk_obj_ids.items():
            if len(matched_trk_obj_ids) < 2:
                continue  # only count detections that are matched to multiple (>=2) masklets
            # if there are multiple matched track ids, we need to find the one that appeared first;
            # these later appearing ids may be removed since they may be considered as duplicates
            first_appear_obj_id = (
                min(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
                if not reverse
                else max(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
            )
            for obj_id in matched_trk_obj_ids:
                if obj_id != first_appear_obj_id:
                    key = (first_appear_obj_id, obj_id)
                    overlap_pair_to_frame_inds[key].append(frame_idx)

        # b) remove a masklet if it first appears after `hotstart_diff` and it overlaps with another
        # masklet (that appears earlier) for more than `self.hotstart_dup_thresh` frames
        for (first_obj_id, obj_id), frame_indices in overlap_pair_to_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue  # skip if the object is already removed
            if (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                obj_first_frame_idx[obj_id] < hotstart_diff and reverse
            ):
                if len(frame_indices) >= self.hotstart_dup_thresh:
                    obj_ids_newly_removed.add(obj_id)
                    logger.debug(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it overlaps with another track {first_obj_id} at frames: {frame_indices}"
                    )

        removed_obj_ids.update(obj_ids_newly_removed)
        return obj_ids_newly_removed, rank0_metadata

    def _tracker_update_memories(
        self,
        tracker_inference_states: List[Any],
        frame_idx: int,
        tracker_metadata: Dict[str, Any],
        low_res_masks: Tensor,
        tracker_obj_scores_global: Tensor | None = None,
        spme_write_gate: float | None = None,
        spme_query_vec: Tensor | None = None,
        spme_det_score_raw: float | None = None,
        spme_query_cos: float | None = None,
        spme_presence_prob: float | None = None,
        spme_query_vec_by_obj: Dict[int, Tensor] | None = None,
        spme_det_score_raw_by_obj: Dict[int, float] | None = None,
        spme_query_cos_by_obj: Dict[int, float] | None = None,
        spme_det_trk_iou_by_obj: Dict[int, float] | None = None,
        spme_reinit_by_obj: Dict[int, int] | None = None,
        spme_reid_best_sim_by_obj: Dict[int, float] | None = None,
        spme_reid_margin_by_obj: Dict[int, float] | None = None,
        spme_reid_accept_by_obj: Dict[int, int] | None = None,
        spme_reid_bank_size_by_obj: Dict[int, int] | None = None,
        spme_reid_trk_sim_by_obj: Dict[int, float] | None = None,
        spme_reid_det_vs_trk_margin_by_obj: Dict[int, float] | None = None,
        track_in_reverse: bool = False,
    ):
        """
        Run Sam2 memory encoder, enforcing non-overlapping constraints globally.
        """
        if len(tracker_inference_states) == 0:
            return
        apply_mode = os.getenv("SAM3_SPME_WRITE_GATE_APPLY", "blend_mem").strip().lower()

        learned_gate_enabled = (
            isinstance(getattr(self, "spme_gate_mlp", None), nn.Module)
            and os.getenv("SAM3_SPME_LEARNED_GATE", "0") == "1"
        )
        learned_gate_use_decay = (
            learned_gate_enabled
            and os.getenv("SAM3_SPME_LEARNED_GATE_USE_DECAY", "1") == "1"
        )
        learned_gate_log_to_out = (
            learned_gate_enabled
            and os.getenv("SAM3_SPME_LEARNED_GATE_LOG", "0") == "1"
        )
        signals_log_to_out = learned_gate_log_to_out or os.getenv("SAM3_SPME_LOG_SIGNALS", "0") == "1"

        gate: float | None = None
        gate_by_obj: Dict[int, float] | None = None

        if learned_gate_enabled:
            # Learned gate uses differentiable blend/decay, so avoid dangerous apply modes.
            apply_mode = "blend_mem"
        else:
            if spme_write_gate is not None:
                gate = max(0.0, min(1.0, float(spme_write_gate)))

            if (
                gate is not None
                and os.getenv("SAM3_SPME_PER_OBJECT", "0") == "1"
                and (
                    isinstance(spme_query_cos_by_obj, dict)
                    or isinstance(spme_det_score_raw_by_obj, dict)
                )
            ):
                # Per-object write gate: avoids multi-instance coupling where one low-qcos object
                # would globally suppress memory writing for all objects.
                write_mode = os.getenv("SAM3_SPME_WRITE_GATE_MODE", "hard").strip().lower()
                write_det_thr = float(os.getenv("SAM3_SPME_DET_THR", "0.0"))
                write_q_thr = float(os.getenv("SAM3_SPME_QCOS_THR", "0.8"))
                write_use_det = os.getenv("SAM3_SPME_USE_DET_SCORE", "1") == "1"

                def _gate_for_one(det_score_raw_f: float | None, qcos_f: float | None) -> float:
                    if qcos_f is None:
                        return 1.0
                    qcos_ff = float(qcos_f)
                    det_score_ff = float(det_score_raw_f) if det_score_raw_f is not None else None
                    if (
                        det_score_ff is not None
                        and write_det_thr > 0.0
                        and det_score_ff < write_det_thr
                    ):
                        return 1.0
                    if write_mode == "hard":
                        g = 1.0 if qcos_ff >= write_q_thr else 0.0
                    else:
                        denom = max(1e-6, 1.0 - write_q_thr)
                        g = max(0.0, min(1.0, (qcos_ff - write_q_thr) / denom))
                        if write_use_det and det_score_ff is not None:
                            g *= max(0.0, min(1.0, det_score_ff))
                    return float(max(0.0, min(1.0, g)))

                try:
                    obj_ids_all = tracker_metadata.get("obj_ids_all_gpu", None)
                    if isinstance(obj_ids_all, np.ndarray):
                        gate_by_obj = {}
                        for obj_id in obj_ids_all.astype(np.int64).tolist():
                            qcos_f = (
                                float(spme_query_cos_by_obj.get(int(obj_id)))
                                if isinstance(spme_query_cos_by_obj, dict)
                                and int(obj_id) in spme_query_cos_by_obj
                                else None
                            )
                            det_score_f = (
                                float(spme_det_score_raw_by_obj.get(int(obj_id)))
                                if isinstance(spme_det_score_raw_by_obj, dict)
                                and int(obj_id) in spme_det_score_raw_by_obj
                                else None
                            )
                            gate_by_obj[int(obj_id)] = _gate_for_one(det_score_f, qcos_f)
                except Exception:
                    gate_by_obj = None

        # Optional: Semantic Pointer Memory Editing (SPME-Fusion).
        #
        # Env vars:
        # - `SAM3_SPME_FUSION=1` enables fusion/editing.
        # - `SAM3_SPME_FUSION_MODE` in {resid, film}. Default: resid.
        # - `SAM3_SPME_FUSION_ALPHA` (float): base scale for memory editing. Default: 0.0 (off).
        # - `SAM3_SPME_FUSION_ALPHA_OBJ` (float): base scale for obj_ptr editing. Default: 0.0.
        # - `SAM3_SPME_FUSION_DET_THR` (float): only apply when det_score_raw >= thr. Default: 0.0.
        # - `SAM3_SPME_FUSION_USE_PRESENCE` in {0,1}. Default: 1.
        # - `SAM3_SPME_FUSION_QCOS_THR` (float): optional qcos gate center (sigmoid). Default: 0.0 (off).
        # - `SAM3_SPME_FUSION_QCOS_TEMP` (float): sigmoid temperature. Default: 20.0.
        # - `SAM3_SPME_FUSION_QCOS_GATE` in {sigmoid, linear}. Default: sigmoid.
        # - `SAM3_SPME_PER_OBJECT_FALLBACK` in {0,1}. Default: 0.
        #     When `SAM3_SPME_PER_OBJECT=1`, if per-object pointers are unavailable (no det↔trk match),
        #     we *skip* fusion by default. Set this to 1 to fall back to the global pointer.
        fusion_mode = os.getenv("SAM3_SPME_FUSION_MODE", "resid").strip().lower()
        fusion_enabled = os.getenv("SAM3_SPME_FUSION", "0") == "1"
        fusion_per_object = os.getenv("SAM3_SPME_PER_OBJECT", "0") == "1"
        fusion_per_object_fallback = os.getenv("SAM3_SPME_PER_OBJECT_FALLBACK", "0") == "1"
        base_alpha = float(os.getenv("SAM3_SPME_FUSION_ALPHA", "0.0")) if fusion_enabled else 0.0
        base_alpha_obj = (
            float(os.getenv("SAM3_SPME_FUSION_ALPHA_OBJ", "0.0")) if fusion_enabled else 0.0
        )
        det_thr = float(os.getenv("SAM3_SPME_FUSION_DET_THR", "0.0")) if fusion_enabled else 0.0
        q_thr = float(os.getenv("SAM3_SPME_FUSION_QCOS_THR", "0.0")) if fusion_enabled else 0.0
        q_temp = float(os.getenv("SAM3_SPME_FUSION_QCOS_TEMP", "20.0")) if fusion_enabled else 0.0
        q_gate = os.getenv("SAM3_SPME_FUSION_QCOS_GATE", "sigmoid").strip().lower()
        use_presence = os.getenv("SAM3_SPME_FUSION_USE_PRESENCE", "1") == "1"
        use_write_gate = os.getenv("SAM3_SPME_FUSION_USE_WRITE_GATE", "1") == "1"

        # Optional: make fusion event-driven by requiring low tracker confidence.
        #
        # Motivation:
        # - EndoVis experiments show that always-on fusion (applied in ~70% frames) can hurt IoU.
        # - A more principled "overseer" story is: only inject detector guidance when the tracker is
        #   uncertain (low tracker_score), i.e., around occlusions / drift recovery windows.
        #
        # Envs:
        # - SAM3_SPME_FUSION_EVENT_DRIVEN=1 enables this behavior.
        # - SAM3_SPME_FUSION_TRACKER_THR sets the tracker_score threshold (default 0.8).
        fusion_event_driven = os.getenv("SAM3_SPME_FUSION_EVENT_DRIVEN", "0") == "1"
        # Event-driven fusion trigger mode:
        # - tracker_score: trigger when tracker_score < thr (legacy; can miss drift)
        # - mismatch: trigger when det↔trk IoU < thr (drift-aligned; requires spme_det_trk_iou_by_obj)
        fusion_event_mode = os.getenv("SAM3_SPME_FUSION_EVENT_MODE", "tracker_score").strip().lower()
        fusion_tracker_thr = (
            float(os.getenv("SAM3_SPME_FUSION_TRACKER_THR", "0.8")) if fusion_event_driven else 0.0
        )
        fusion_mismatch_iou_thr = (
            float(os.getenv("SAM3_SPME_FUSION_MISMATCH_IOU_THR", "0.2")) if fusion_event_driven else 0.0
        )
        # When using mismatch-triggered event fusion, optionally control whether we further scale the
        # fusion strength by mismatch severity (lower IoU => stronger).
        #
        # - "linear" (default): r *= (thr - miou) / thr
        # - "hard": do NOT apply extra severity scaling (event is still gated by miou < thr)
        fusion_mismatch_scale = (
            os.getenv("SAM3_SPME_FUSION_MISMATCH_SCALE", "linear").strip().lower()
        )

        # Optional: memory hygiene — skip writing potentially drifted frames into memory.
        #
        # Motivation:
        # - A small number of drift/ghost frames can pollute memory and have long-lasting effects.
        # - A reviewer-safe safeguard is to *not* write into memory when detector↔tracker consistency
        #   is low (after det↔trk matching).
        #
        # Envs:
        # - SAM3_SPME_SKIP_WRITE_ON_MISMATCH=1 enables this behavior.
        # - SAM3_SPME_SKIP_WRITE_MISMATCH_IOU_THR sets the IoU threshold (default: 0.2).
        #   If unset, defaults to `SAM3_SPME_FUSION_MISMATCH_IOU_THR` when available.
        # - SAM3_SPME_SKIP_WRITE_MODE controls what to freeze when skipping:
        #     - "maskmem" (default): freeze only `maskmem_features` (v1 behavior)
        #     - "full": also freeze `obj_ptr` to keep pointer/memory consistent (v2, recommended)
        # - SAM3_SPME_SKIP_WRITE_USE_DET_QCOS=1 additionally requires the overseer (detector) to be
        #   confident and identity-consistent before skipping:
        #     - det_score_raw >= SAM3_SPME_SKIP_WRITE_DET_THR (default: 0.7)
        #     - qcos >= SAM3_SPME_SKIP_WRITE_QCOS_THR (default: 0.7)
        #
        #   This reduces false skips caused by unreliable detections.
        skip_write_on_mismatch = os.getenv("SAM3_SPME_SKIP_WRITE_ON_MISMATCH", "0") == "1"
        # Trigger mode for skip-write:
        # - "mismatch" (default): det↔trk IoU < thr (legacy; can be unreliable under detector domain shift)
        # - "tracker_score": tracker_score < thr (uncertainty-triggered memory hygiene)
        # - "hybrid": both conditions (more conservative)
        skip_write_event_mode = os.getenv("SAM3_SPME_SKIP_WRITE_EVENT_MODE", "mismatch").strip().lower()
        skip_write_tracker_thr = float(
            os.getenv(
                "SAM3_SPME_SKIP_WRITE_TRACKER_THR",
                str(fusion_tracker_thr if fusion_event_driven else 0.8),
            )
        )
        skip_write_mismatch_iou_thr = float(
            os.getenv(
                "SAM3_SPME_SKIP_WRITE_MISMATCH_IOU_THR",
                str(fusion_mismatch_iou_thr if fusion_mismatch_iou_thr > 0.0 else 0.2),
            )
        )
        skip_write_mode = os.getenv("SAM3_SPME_SKIP_WRITE_MODE", "maskmem").strip().lower()
        skip_write_freeze_ptr = skip_write_mode in {"full", "all", "maskmem+ptr", "maskmem_ptr", "ptr"}
        skip_write_use_det_qcos = os.getenv("SAM3_SPME_SKIP_WRITE_USE_DET_QCOS", "0") == "1"
        skip_write_det_thr = float(os.getenv("SAM3_SPME_SKIP_WRITE_DET_THR", "0.7"))
        skip_write_qcos_thr = float(os.getenv("SAM3_SPME_SKIP_WRITE_QCOS_THR", "0.7"))
        # Optional safety: require ReID evidence that detector is more consistent than tracker
        # before skipping writes (prevents false skips when det masks are misaligned).
        skip_write_require_reid_det_better = (
            os.getenv("SAM3_SPME_SKIP_WRITE_REQUIRE_REID_DET_BETTER", "0") == "1"
        )
        skip_write_reid_margin = float(
            os.getenv(
                "SAM3_SPME_SKIP_WRITE_REID_MARGIN",
                os.getenv("SAM3_SPME_REID_COMPARE_MARGIN", "0.01"),
            )
        )

        # Map tracker confidence to per-object scalars when available.
        trk_score_by_obj: Dict[int, float] | None = None
        try:
            if isinstance(tracker_obj_scores_global, torch.Tensor):
                obj_ids_all = tracker_metadata.get("obj_ids_all_gpu", None)
                scores_flat = tracker_obj_scores_global.detach().float().view(-1)
                if isinstance(obj_ids_all, np.ndarray) and int(obj_ids_all.shape[0]) == int(
                    scores_flat.shape[0]
                ):
                    trk_score_by_obj = {}
                    for i, obj_id in enumerate(obj_ids_all.astype(np.int64).tolist()):
                        trk_score_by_obj[int(obj_id)] = float(
                            scores_flat[int(i)].clamp(0.0, 1.0).item()
                        )
        except Exception:
            trk_score_by_obj = None

        def _sigmoid_f(x: float) -> float:
            # Stable sigmoid for floats.
            if x >= 0.0:
                z = math.exp(-x)
                return 1.0 / (1.0 + z)
            z = math.exp(x)
            return z / (1.0 + z)

        def _fusion_scales_for_one(
            det_score_raw_f: float | None,
            qcos_f: float | None,
            gate_override: float | None = None,
            tracker_score_f: float | None = None,
            det_trk_iou_f: float | None = None,
        ) -> tuple[float | None, float | None]:
            if det_score_raw_f is None:
                return None, None
            if det_score_raw_f < det_thr:
                return None, None
            if fusion_event_driven:
                if fusion_event_mode in {"tracker", "tracker_score", "score"}:
                    if tracker_score_f is None:
                        return None, None
                    # Apply fusion only when tracker confidence is low.
                    ts = max(0.0, min(1.0, float(tracker_score_f)))
                    if ts >= fusion_tracker_thr:
                        return None, None
                elif fusion_event_mode in {"mismatch", "iou", "det_trk_iou"}:
                    if det_trk_iou_f is None:
                        return None, None
                    miou = max(0.0, min(1.0, float(det_trk_iou_f)))
                    thr = max(1e-6, float(fusion_mismatch_iou_thr))
                    if miou >= thr:
                        return None, None
                else:
                    # Unknown mode: fall back to tracker-score behavior.
                    if tracker_score_f is None:
                        return None, None
                    ts = max(0.0, min(1.0, float(tracker_score_f)))
                    if ts >= fusion_tracker_thr:
                        return None, None
            r = max(0.0, min(1.0, float(det_score_raw_f)))
            if use_presence and spme_presence_prob is not None:
                r *= max(0.0, min(1.0, float(spme_presence_prob)))
            if qcos_f is not None and q_thr > 0.0:
                # New soft gate logic (sigmoid):
                if q_gate == "sigmoid":
                    r *= _sigmoid_f((float(qcos_f) - q_thr) * q_temp)
                else:
                    # Legacy linear ramp
                    denom = max(1e-6, 1.0 - q_thr)
                    r *= max(0.0, min(1.0, (float(qcos_f) - q_thr) / denom))
            if use_write_gate:
                g = gate_override if gate_override is not None else gate
                r *= g if g is not None else 1.0
            if fusion_event_driven:
                if (
                    fusion_event_mode in {"tracker", "tracker_score", "score"}
                    and tracker_score_f is not None
                    and fusion_tracker_thr > 0.0
                ):
                    # Scale more when the tracker is *more* uncertain, but keep this bounded and simple.
                    ts = max(0.0, min(1.0, float(tracker_score_f)))
                    r *= max(
                        0.0,
                        min(1.0, (fusion_tracker_thr - ts) / max(1e-6, fusion_tracker_thr)),
                    )
                elif (
                    fusion_event_mode in {"mismatch", "iou", "det_trk_iou"}
                    and det_trk_iou_f is not None
                    and fusion_mismatch_scale not in {"hard", "none", "off", "0"}
                ):
                    # Scale more when mismatch is larger (lower IoU).
                    miou = max(0.0, min(1.0, float(det_trk_iou_f)))
                    thr = max(1e-6, float(fusion_mismatch_iou_thr))
                    r *= max(0.0, min(1.0, (thr - miou) / thr))
            s_mem = float(max(0.0, min(1.0, base_alpha * r))) if base_alpha > 0.0 else 0.0
            s_obj = (
                float(max(0.0, min(1.0, base_alpha_obj * r))) if base_alpha_obj > 0.0 else 0.0
            )
            return (s_mem if s_mem > 0.0 else None), (s_obj if s_obj > 0.0 else None)
        # Avoid an extra interpolation step by directly interpolating to `interpol_size`
        high_res_H, high_res_W = (
            self.tracker.maskmem_backbone.mask_downsampler.interpol_size
        )
        # NOTE: inspect this part if we observe OOMs in the demo
        high_res_masks = F.interpolate(
            low_res_masks.unsqueeze(1),
            size=(high_res_H, high_res_W),
            mode="bilinear",
            align_corners=False,
        )
        # We first apply non-overlapping constraints before memory encoding. This may include some suppression heuristics.
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            high_res_masks = self.tracker._suppress_object_pw_area_shrinkage(
                high_res_masks
            )
        gate_tensor_masks = None
        if apply_mode in {"mask_logits", "masklogits"}:
            if gate_by_obj is not None:
                obj_ids_all = tracker_metadata.get("obj_ids_all_gpu", None)
                if isinstance(obj_ids_all, np.ndarray) and obj_ids_all.shape[0] == int(
                    high_res_masks.shape[0]
                ):
                    gate_vals = [
                        float(gate_by_obj.get(int(obj_id), 1.0))
                        for obj_id in obj_ids_all.astype(np.int64).tolist()
                    ]
                    gate_tensor_masks = high_res_masks.new_tensor(gate_vals).view(-1, 1, 1, 1)
            elif gate is not None:
                gate_tensor_masks = high_res_masks.new_full(
                    (int(high_res_masks.shape[0]), 1, 1, 1), float(gate)
                )

            if gate_tensor_masks is not None:
                gmin = float(gate_tensor_masks.detach().float().min().item())
                if gmin < 1.0:
                    empty_logit = -10.0
                    high_res_masks = high_res_masks * gate_tensor_masks + empty_logit * (
                        1.0 - gate_tensor_masks
                    )
        # Instead of gathering the predicted object scores, we use mask areas as a proxy.
        object_score_logits = torch.where(
            (high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0
        )
        if gate_tensor_masks is not None and apply_mode in {"mask_logits", "masklogits"}:
            gmin = float(gate_tensor_masks.detach().float().min().item())
            if gmin < 1.0:
                object_score_logits = object_score_logits.new_full(
                    object_score_logits.shape, 10.0
                ) * (2.0 * gate_tensor_masks.view(-1, 1) - 1.0)

        # Run the memory encoder on local slices for each GPU
        start_idx_gpu = sum(tracker_metadata["num_obj_per_gpu"][: self.rank])
        start_idx_state = start_idx_gpu
        for tracker_state in tracker_inference_states:
            num_obj_per_state = len(tracker_state["obj_ids"])
            if num_obj_per_state == 0:
                continue
            # Get the local high-res masks and object score logits for this inference state
            end_idx_state = start_idx_state + num_obj_per_state
            local_high_res_masks = high_res_masks[start_idx_state:end_idx_state]
            local_object_score_logits = object_score_logits[
                start_idx_state:end_idx_state
            ]
            local_batch_size = local_high_res_masks.size(0)
            obj_ids_local = [int(x) for x in tracker_state.get("obj_ids", [])]
            output_dict = tracker_state["output_dict"]
            learned_gate_mem_scale_1d: Tensor | None = None
            learned_gate_mem_offset_1d: Tensor | None = None
            learned_gate_fusion_1d: Tensor | None = None
            learned_gate_decay_1d: Tensor | None = None
            if learned_gate_enabled and self.spme_gate_mlp is not None:
                # ----------------------------------------------------------
                # Learned multi-factor gates (per object, differentiable)
                # ----------------------------------------------------------
                tracker_score: Tensor | None = None
                gate_inputs_mode = getattr(self, "spme_gate_inputs_mode", None) or "full"
                # Semantic pointer signals (per-object if available, else fall back to global).
                qcos_vals: list[float] = []
                det_vals: list[float] = []
                for obj_id in obj_ids_local:
                    qcos_f = (
                        float(spme_query_cos_by_obj.get(obj_id))
                        if isinstance(spme_query_cos_by_obj, dict) and obj_id in spme_query_cos_by_obj
                        else (float(spme_query_cos) if spme_query_cos is not None else 1.0)
                    )
                    det_f = (
                        float(spme_det_score_raw_by_obj.get(obj_id))
                        if isinstance(spme_det_score_raw_by_obj, dict)
                        and obj_id in spme_det_score_raw_by_obj
                        else (float(spme_det_score_raw) if spme_det_score_raw is not None else 0.0)
                    )
                    qcos_vals.append(float(max(-1.0, min(1.0, qcos_f))))
                    det_vals.append(float(max(0.0, min(1.0, det_f))))

                gate_dev = local_high_res_masks.device
                qcos_t = torch.tensor(qcos_vals, device=gate_dev, dtype=torch.float32)
                det_t = torch.tensor(det_vals, device=gate_dev, dtype=torch.float32)

                # Tracker score signal (mask quality / confidence), used both as an optional gate input
                # ("full" mode) and for optional inference-time modulation.
                if (
                    isinstance(tracker_obj_scores_global, torch.Tensor)
                    and int(tracker_obj_scores_global.numel()) == int(high_res_masks.shape[0])
                ):
                    tracker_logits_local = tracker_obj_scores_global[start_idx_state:end_idx_state]
                    tracker_score = tracker_logits_local.detach().float().sigmoid().clamp(0.0, 1.0)
                trk_t = tracker_score if tracker_score is not None else det_t.new_ones(det_t.shape)

                if gate_inputs_mode == "full":
                    # Mask area + delta signal.
                    curr_area = (
                        torch.sigmoid(local_high_res_masks.detach().float())
                        .mean(dim=(-1, -2))
                        .squeeze(1)
                    )
                    gate_state = tracker_state.setdefault("spme_gate_state", {})
                    prev_area_map: Dict[int, float] = gate_state.setdefault("prev_mask_area", {})
                    missing_count_map: Dict[int, int] = gate_state.setdefault("missing_count", {})

                    prev_area_vals: list[float] = []
                    for i, obj_id in enumerate(obj_ids_local):
                        prev_area_vals.append(float(prev_area_map.get(obj_id, float(curr_area[i].item()))))
                    prev_area = curr_area.new_tensor(prev_area_vals)
                    area_eps = float(os.getenv("SAM3_SPME_LEARNED_GATE_AREA_EPS", "1e-3"))
                    area_delta = (curr_area - prev_area).abs() / (prev_area.abs() + area_eps)
                    # Map to [0, 1] smoothly: d -> d/(d+1)
                    area_delta = area_delta / (area_delta + 1.0)
                    area_delta = area_delta.clamp(0.0, 1.0)

                    # Occlusion/missing duration signal (consecutive missing frames).
                    present_mask = (local_high_res_masks.detach() > 0).any(dim=(-1, -2)).squeeze(1)
                    missing_counts: list[float] = []
                    for i, obj_id in enumerate(obj_ids_local):
                        if bool(present_mask[i].item()):
                            missing_count_map[obj_id] = 0
                        else:
                            missing_count_map[obj_id] = int(missing_count_map.get(obj_id, 0)) + 1
                        missing_counts.append(float(missing_count_map[obj_id]))
                    occ_norm = float(os.getenv("SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM", "10.0"))
                    last_occluded_norm = curr_area.new_tensor(missing_counts) / max(1.0, occ_norm)
                    last_occluded_norm = last_occluded_norm.clamp(0.0, 1.0)

                    # Update persistent per-object state after we computed deltas.
                    for i, obj_id in enumerate(obj_ids_local):
                        prev_area_map[obj_id] = float(curr_area[i].item())

                    x = torch.stack([qcos_t, det_t, trk_t, area_delta, last_occluded_norm], dim=-1)
                elif gate_inputs_mode == "det3":
                    # Detector-only ("clean") gate inputs: (qcos, det_score, presence_prob)
                    if spme_presence_prob is not None:
                        presence_t = det_t.new_full(det_t.shape, float(spme_presence_prob))
                    else:
                        # Fall back to det_score itself as a proxy.
                        presence_t = det_t
                    x = torch.stack([qcos_t, det_t, presence_t], dim=-1)
                elif gate_inputs_mode == "det4":
                    # Detector-only + occlusion length derived from detector presence.
                    if spme_presence_prob is not None:
                        presence_t = det_t.new_full(det_t.shape, float(spme_presence_prob))
                    else:
                        presence_t = det_t

                    gate_state = tracker_state.setdefault("spme_gate_state", {})
                    missing_count_map_det: Dict[int, int] = gate_state.setdefault("missing_count_det", {})
                    det_present_thr = float(os.getenv("SAM3_SPME_LEARNED_GATE_DET_PRESENT_THR", "0.3"))
                    missing_counts: list[float] = []
                    for i, obj_id in enumerate(obj_ids_local):
                        is_present = float(det_t[i].item()) >= float(det_present_thr)
                        if bool(is_present):
                            missing_count_map_det[obj_id] = 0
                        else:
                            missing_count_map_det[obj_id] = int(missing_count_map_det.get(obj_id, 0)) + 1
                        missing_counts.append(float(missing_count_map_det[obj_id]))
                    occ_norm = float(os.getenv("SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM", "10.0"))
                    last_occluded_norm = det_t.new_tensor(missing_counts) / max(1.0, occ_norm)
                    last_occluded_norm = last_occluded_norm.clamp(0.0, 1.0)
                    x = torch.stack([qcos_t, det_t, presence_t, last_occluded_norm], dim=-1)
                else:
                    # Should not happen (validated at init), but keep a safe fallback.
                    x = torch.stack([qcos_t, det_t, det_t, det_t * 0.0, det_t * 0.0], dim=-1)

                learned_gate_mem_scale_1d, learned_gate_mem_offset_1d, learned_gate_decay_1d = (
                    self.spme_gate_mlp(x)
                )

                # Optional inference-time calibration knobs (default: identity).
                #
                # Motivation: on some datasets, a learned gate checkpoint can converge to a near-constant
                # behavior (especially a non-trivial mem_offset), which strongly suppresses ghosts but
                # can also harm mask quality on GT-present frames. These scalars allow quick, reproducible
                # sweeps without retraining to localize the culprit term(s).
                def _get_float_env(name: str, default: float) -> float:
                    try:
                        return float(os.getenv(name, str(default)))
                    except Exception:
                        return float(default)

                write_strength = _get_float_env("SAM3_SPME_LEARNED_GATE_WRITE_STRENGTH", 1.0)
                offset_strength = _get_float_env("SAM3_SPME_LEARNED_GATE_OFFSET_STRENGTH", 1.0)
                decay_strength = _get_float_env("SAM3_SPME_LEARNED_GATE_DECAY_STRENGTH", 1.0)

                if write_strength != 1.0:
                    learned_gate_mem_scale_1d = (learned_gate_mem_scale_1d * float(write_strength)).clamp(
                        0.0, 1.0
                    )
                if offset_strength != 1.0:
                    learned_gate_mem_offset_1d = learned_gate_mem_offset_1d * float(offset_strength)
                if decay_strength != 1.0:
                    learned_gate_decay_1d = (learned_gate_decay_1d * float(decay_strength)).clamp(
                        0.0, 1.0
                    )

                # Optional: enforce a detector-confidence monotonicity constraint at inference time.
                #
                # Rationale:
                # - On some runs, the learned gate can converge to near-constant mem_offset/mem_scale that
                #   suppresses ghosts, but also degrades mask quality on GT-present frames.
                # - We can make the behavior more interpretable and safer by modulating *how much* the gate
                #   deviates from identity based on per-object detector confidence: when det_score is high,
                #   push (scale→1, offset→0, decay→0); when det_score is low, allow the learned edit.
                #
                # This is strictly inference-time and does not change checkpoint formats.
                abs_w: torch.Tensor | None = None

                # (1) Detector-based modulation (optionally with qcos consistency).
                if os.getenv("SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET", "0") == "1":
                    det_pow = _get_float_env("SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_POW", 1.0)

                    # Base confidence proxy: detector score.
                    conf_det_t = det_t

                    # Optional: incorporate qcos consistency so that the gate can still intervene when
                    # det_score is high but inconsistent (e.g., false positives / drift signatures).
                    #
                    # This reuses the fusion qcos gate knobs for simplicity/consistency:
                    # - SAM3_SPME_FUSION_QCOS_THR
                    # - SAM3_SPME_FUSION_QCOS_TEMP
                    # - SAM3_SPME_FUSION_QCOS_GATE in {sigmoid, linear}
                    if os.getenv("SAM3_SPME_LEARNED_GATE_MODULATE_BY_DET_USE_QCOS", "0") == "1":
                        q_thr_m = float(os.getenv("SAM3_SPME_FUSION_QCOS_THR", "0.0"))
                        q_temp_m = float(os.getenv("SAM3_SPME_FUSION_QCOS_TEMP", "20.0"))
                        q_gate_m = os.getenv("SAM3_SPME_FUSION_QCOS_GATE", "sigmoid").strip().lower()

                        def _sigmoid_t(x: torch.Tensor) -> torch.Tensor:
                            return 0.5 * (torch.tanh(x * 0.5) + 1.0)

                        if q_gate_m == "sigmoid":
                            q_gate_val = _sigmoid_t((qcos_t - float(q_thr_m)) * float(q_temp_m))
                        else:
                            denom = max(1e-6, 1.0 - float(q_thr_m))
                            q_gate_val = ((qcos_t - float(q_thr_m)) / denom).clamp(0.0, 1.0)

                        conf_det_t = (conf_det_t * q_gate_val).clamp(0.0, 1.0)

                    abs_w_det = (1.0 - conf_det_t).clamp(0.0, 1.0)
                    if det_pow != 1.0:
                        abs_w_det = abs_w_det.pow(float(det_pow))
                    abs_w = abs_w_det if abs_w is None else (abs_w * abs_w_det)

                # (2) Tracker-confidence modulation.
                #
                # Motivation:
                # - On EndoVis, the detector can miss or be poorly calibrated on some stable sequences.
                # - We want a "do-no-harm" envelope: when the tracker is confident, keep the learned gate
                #   close to identity even if det_score is low; when the tracker is uncertain, allow edits.
                #
                # This is inference-time only; no checkpoint change.
                if os.getenv("SAM3_SPME_LEARNED_GATE_MODULATE_BY_TRK", "0") == "1":
                    trk_pow = _get_float_env("SAM3_SPME_LEARNED_GATE_MODULATE_BY_TRK_POW", 1.0)
                    conf_trk_t = trk_t.detach().float().clamp(0.0, 1.0)
                    if trk_pow != 1.0:
                        conf_trk_t = conf_trk_t.pow(float(trk_pow))
                    abs_w_trk = (1.0 - conf_trk_t).clamp(0.0, 1.0)
                    abs_w = abs_w_trk if abs_w is None else (abs_w * abs_w_trk)

                # Apply modulation if enabled.
                if abs_w is not None:
                    abs_w_mem = abs_w.unsqueeze(-1)  # (N,1) for broadcasting
                    learned_gate_mem_scale_1d = 1.0 - (1.0 - learned_gate_mem_scale_1d) * abs_w_mem
                    learned_gate_mem_offset_1d = learned_gate_mem_offset_1d * abs_w_mem
                    if learned_gate_use_decay:
                        learned_gate_decay_1d = learned_gate_decay_1d * abs_w_mem[:, :1]

                if isinstance(self.spme_gate_fusion_mlp, nn.Module):
                    learned_gate_fusion_1d = torch.sigmoid(
                        self.spme_gate_fusion_mlp(x.to(dtype=torch.float32))
                    )
                else:
                    learned_gate_fusion_1d = learned_gate_mem_scale_1d.mean(dim=-1, keepdim=True)

                fusion_strength = _get_float_env("SAM3_SPME_LEARNED_GATE_FUSION_STRENGTH", 1.0)
                if fusion_strength != 1.0 and learned_gate_fusion_1d is not None:
                    learned_gate_fusion_1d = (learned_gate_fusion_1d * float(fusion_strength)).clamp(
                        0.0, 1.0
                    )

                if not learned_gate_use_decay:
                    learned_gate_decay_1d = learned_gate_decay_1d * 0.0

            # Run Sam2 memory encoder. Note that we do not re-enforce the non-overlapping constraint as it is turned off by default

            encoded_mem = self.tracker._run_memory_encoder(
                tracker_state,
                frame_idx,
                local_batch_size,
                local_high_res_masks,
                local_object_score_logits,
                is_mask_from_pts=False,
            )
            local_maskmem_features, local_maskmem_pos_enc = encoded_mem
            if apply_mode in {"blend_mem", "blend", "ema"}:
                if (
                    learned_gate_enabled
                    and learned_gate_mem_scale_1d is not None
                    and learned_gate_mem_offset_1d is not None
                    and learned_gate_decay_1d is not None
                ):
                    mem_dim_feat = int(local_maskmem_features.shape[1])
                    mem_dim_gate = int(learned_gate_mem_scale_1d.shape[-1])
                    gate_decay = learned_gate_decay_1d.to(
                        device=local_maskmem_features.device, dtype=local_maskmem_features.dtype
                    ).view(-1, 1, 1, 1)

                    gate_scale: Tensor | None = None
                    gate_offset: Tensor | None = None
                    gate_write_scalar: Tensor | None = None
                    if mem_dim_gate == mem_dim_feat and int(learned_gate_mem_offset_1d.shape[-1]) == mem_dim_feat:
                        gate_scale = learned_gate_mem_scale_1d.to(
                            device=local_maskmem_features.device,
                            dtype=local_maskmem_features.dtype,
                        ).view(-1, mem_dim_feat, 1, 1)
                        gate_offset = learned_gate_mem_offset_1d.to(
                            device=local_maskmem_features.device,
                            dtype=local_maskmem_features.dtype,
                        ).view(-1, mem_dim_feat, 1, 1)
                    else:
                        # Fallback (should be rare): use a scalar gate from mean(scale).
                        if learned_gate_fusion_1d is not None:
                            gate_write_scalar = learned_gate_fusion_1d.to(
                                device=local_maskmem_features.device,
                                dtype=local_maskmem_features.dtype,
                            ).view(-1, 1, 1, 1)

                    # Optional robustness tweak (inference-safe):
                    # When tracker_score is low (mask quality low), avoid overwriting memory too aggressively by
                    # scaling the learned write gates by tracker_score. This can help on long-absence / occlusion
                    # clips where empty/noisy masks would otherwise corrupt memory and delay recovery.
                    if os.getenv("SAM3_SPME_LEARNED_GATE_SCALE_BY_TRK", "0") == "1":
                        if tracker_score is not None:
                            trk_w = tracker_score.to(
                                device=local_maskmem_features.device,
                                dtype=local_maskmem_features.dtype,
                            ).view(-1, 1, 1, 1)
                            if gate_scale is not None:
                                gate_scale = gate_scale * trk_w
                            if gate_offset is not None:
                                gate_offset = gate_offset * trk_w
                            if gate_write_scalar is not None:
                                gate_write_scalar = gate_write_scalar * trk_w

                    prev_frame_idx = frame_idx + 1 if track_in_reverse else frame_idx - 1
                    prev_out = None
                    if prev_frame_idx >= 0:
                        prev_out = tracker_state["output_dict"]["non_cond_frame_outputs"].get(
                            prev_frame_idx, None
                        )
                        if prev_out is None:
                            prev_out = tracker_state["output_dict"]["cond_frame_outputs"].get(
                                prev_frame_idx, None
                            )
                    if isinstance(prev_out, dict):
                        prev_feat = prev_out.get("maskmem_features", None)
                        if (
                            isinstance(prev_feat, torch.Tensor)
                            and prev_feat.shape == local_maskmem_features.shape
                        ):
                            prev_feat = prev_feat.to(
                                device=local_maskmem_features.device,
                                dtype=local_maskmem_features.dtype,
                            )
                            if gate_scale is not None and gate_offset is not None:
                                # Channel-wise FiLM + blend with previous memory.
                                new_mem = local_maskmem_features * gate_scale + gate_offset
                                prev_part = prev_feat * (1.0 - gate_scale)
                                if learned_gate_use_decay:
                                    prev_part = prev_part * (1.0 - gate_decay)
                                local_maskmem_features = new_mem + prev_part
                            elif gate_write_scalar is not None:
                                if learned_gate_use_decay:
                                    local_maskmem_features = (
                                        local_maskmem_features * gate_write_scalar
                                        + prev_feat * (1.0 - gate_write_scalar) * (1.0 - gate_decay)
                                    )
                                else:
                                    local_maskmem_features = (
                                        local_maskmem_features * gate_write_scalar
                                        + prev_feat * (1.0 - gate_write_scalar)
                                    )
                    else:
                        if gate_scale is not None and gate_offset is not None:
                            local_maskmem_features = local_maskmem_features * gate_scale + gate_offset
                else:
                    gate_local: Tensor | float | None = None
                    if gate_by_obj is not None:
                        gate_vals = [
                            float(gate_by_obj.get(obj_id, 1.0)) for obj_id in obj_ids_local
                        ]
                        gate_local = local_maskmem_features.new_tensor(gate_vals).view(-1, 1, 1, 1)
                        if float(gate_local.detach().float().min().item()) >= 1.0:
                            gate_local = None
                    elif gate is not None and gate < 1.0:
                        gate_local = float(gate)

                    if gate_local is not None:
                        prev_frame_idx = frame_idx + 1 if track_in_reverse else frame_idx - 1
                        prev_out = None
                        if prev_frame_idx >= 0:
                            prev_out = tracker_state["output_dict"]["non_cond_frame_outputs"].get(
                                prev_frame_idx, None
                            )
                            if prev_out is None:
                                prev_out = tracker_state["output_dict"]["cond_frame_outputs"].get(
                                    prev_frame_idx, None
                                )
                        if isinstance(prev_out, dict):
                            prev_feat = prev_out.get("maskmem_features", None)
                            if (
                                isinstance(prev_feat, torch.Tensor)
                                and prev_feat.shape == local_maskmem_features.shape
                            ):
                                prev_feat = prev_feat.to(
                                    device=local_maskmem_features.device,
                                    dtype=local_maskmem_features.dtype,
                                )
                                local_maskmem_features = (
                                    local_maskmem_features * gate_local
                                    + prev_feat * (1.0 - gate_local)
                                )
            # Apply SPME-Fusion edits (either per-object or global single pointer).
            if fusion_enabled and (base_alpha > 0.0 or base_alpha_obj > 0.0):
                mem_dim = int(local_maskmem_features.shape[1])

                per_obj_ctx_ok = (
                    fusion_per_object
                    and isinstance(spme_query_vec_by_obj, dict)
                    and isinstance(spme_det_score_raw_by_obj, dict)
                    and len(spme_query_vec_by_obj) > 0
                    and len(spme_det_score_raw_by_obj) > 0
                )
                do_global_fusion = (not fusion_per_object) or (
                    fusion_per_object_fallback and (not per_obj_ctx_ok)
                )

                if per_obj_ctx_ok:
                    # Per-object editing (preferred for multi-object datasets).
                    for j, obj_id in enumerate(obj_ids_local):
                        qv_src = spme_query_vec_by_obj.get(obj_id, None)
                        if not isinstance(qv_src, torch.Tensor) or qv_src.numel() == 0:
                            continue
                        det_score_f = spme_det_score_raw_by_obj.get(obj_id, None)
                        trk_score_f = (
                            float(trk_score_by_obj.get(obj_id))
                            if isinstance(trk_score_by_obj, dict) and obj_id in trk_score_by_obj
                            else None
                        )
                        qcos_f = (
                            spme_query_cos_by_obj.get(obj_id, None)
                            if isinstance(spme_query_cos_by_obj, dict)
                            else None
                        )
                        det_trk_iou_f = (
                            spme_det_trk_iou_by_obj.get(obj_id, None)
                            if isinstance(spme_det_trk_iou_by_obj, dict)
                            else None
                        )
                        gate_override = (
                            float(learned_gate_fusion_1d[j].detach().float().item())
                            if (learned_gate_enabled and learned_gate_fusion_1d is not None)
                            else (gate_by_obj.get(obj_id, None) if gate_by_obj is not None else None)
                        )
                        s_mem, s_obj = _fusion_scales_for_one(
                            det_score_f,
                            qcos_f,
                            gate_override=gate_override,
                            tracker_score_f=trk_score_f,
                            det_trk_iou_f=det_trk_iou_f,
                        )
                        if s_mem is None:
                            continue

                        qv = qv_src.detach()
                        if qv.ndim > 1:
                            qv = qv.reshape(-1)
                        qv = qv.to(device=local_maskmem_features.device, dtype=torch.float32)
                        qv = qv / qv.norm().clamp_min(1e-6)
                        qv = qv.view(1, -1)  # (1, D)

                        if fusion_mode in {"film", "filmedit"}:
                            film = self.spme_fusion_mem_film_mlp(qv)
                            gamma, beta = film[:, :mem_dim], film[:, mem_dim:]
                            gamma = torch.tanh(gamma).to(local_maskmem_features.dtype)
                            beta = torch.tanh(beta).to(local_maskmem_features.dtype)
                            scale_t = local_maskmem_features.new_tensor(float(s_mem)).view(1, 1, 1, 1)
                            local_maskmem_features[j : j + 1] = local_maskmem_features[j : j + 1] * (
                                1.0 + gamma.view(1, mem_dim, 1, 1) * scale_t
                            ) + beta.view(1, mem_dim, 1, 1) * scale_t
                        else:
                            if int(qv.shape[1]) == mem_dim:
                                q_mem = qv
                            else:
                                proj = getattr(self.tracker, "obj_ptr_tpos_proj", None)
                                if isinstance(proj, nn.Module):
                                    q_mem = proj(qv)
                                else:
                                    q_mem = qv[:, :mem_dim]
                                    if int(q_mem.shape[1]) < mem_dim:
                                        q_mem = F.pad(
                                            q_mem, (0, mem_dim - int(q_mem.shape[1]))
                                        )
                            q_mem = q_mem.to(
                                device=local_maskmem_features.device,
                                dtype=local_maskmem_features.dtype,
                            )
                            scale_t = local_maskmem_features.new_tensor(float(s_mem)).view(1, 1, 1, 1)
                            local_maskmem_features[j : j + 1] = local_maskmem_features[
                                j : j + 1
                            ] + q_mem.view(1, mem_dim, 1, 1) * scale_t
                        # obj_ptr edit is applied later when we have access to output_dict[...]["obj_ptr"]
                        # (we re-compute qv there using the same source).

                if do_global_fusion:
                    # Global editing (single prompt / single object legacy behavior).
                    if (
                        learned_gate_enabled
                        and learned_gate_fusion_1d is not None
                        and isinstance(spme_query_vec, torch.Tensor)
                        and spme_query_vec.numel() > 0
                        and spme_det_score_raw is not None
                    ):
                        # Use the same safety-gated fusion scaling as the heuristic path, but
                        # override the write-gate term with the learned fusion head output.
                        qv = spme_query_vec.detach()
                        if qv.ndim > 1:
                            qv = qv.reshape(-1)
                        qv = qv.to(device=local_maskmem_features.device, dtype=torch.float32)
                        qv = qv / qv.norm().clamp_min(1e-6)
                        qv = qv.view(1, -1)  # (1, D)

                        qcos_global = (
                            None
                            if (spme_query_cos is None or math.isnan(float(spme_query_cos)))
                            else float(spme_query_cos)
                        )
                        det_score_global = float(spme_det_score_raw)

                        if fusion_mode in {"film", "filmedit"}:
                            film = self.spme_fusion_mem_film_mlp(qv)
                            gamma, beta = film[:, :mem_dim], film[:, mem_dim:]
                            gamma = torch.tanh(gamma).to(local_maskmem_features.dtype)
                            beta = torch.tanh(beta).to(local_maskmem_features.dtype)
                            for j in range(int(local_batch_size)):
                                gate_override = float(
                                    learned_gate_fusion_1d[j].detach().float().item()
                                )
                                trk_score_f = None
                                if (
                                    isinstance(trk_score_by_obj, dict)
                                    and isinstance(obj_ids_local, list)
                                    and j < len(obj_ids_local)
                                ):
                                    oid = int(obj_ids_local[j])
                                    if oid in trk_score_by_obj:
                                        trk_score_f = float(trk_score_by_obj.get(oid))
                                s_mem, _ = _fusion_scales_for_one(
                                    det_score_global,
                                    qcos_global,
                                    gate_override=gate_override,
                                    tracker_score_f=trk_score_f,
                                )
                                if s_mem is None:
                                    continue
                                scale_t = local_maskmem_features.new_tensor(float(s_mem)).view(
                                    1, 1, 1, 1
                                )
                                local_maskmem_features[j : j + 1] = local_maskmem_features[
                                    j : j + 1
                                ] * (1.0 + gamma.view(1, mem_dim, 1, 1) * scale_t) + beta.view(
                                    1, mem_dim, 1, 1
                                ) * scale_t
                        else:
                            if int(qv.shape[1]) == mem_dim:
                                q_mem = qv
                            else:
                                proj = getattr(self.tracker, "obj_ptr_tpos_proj", None)
                                if isinstance(proj, nn.Module):
                                    q_mem = proj(qv)
                                else:
                                    q_mem = qv[:, :mem_dim]
                                    if int(q_mem.shape[1]) < mem_dim:
                                        q_mem = F.pad(q_mem, (0, mem_dim - int(q_mem.shape[1])))
                            q_mem = q_mem.to(
                                device=local_maskmem_features.device,
                                dtype=local_maskmem_features.dtype,
                            )
                            resid = q_mem.view(1, mem_dim, 1, 1)
                            for j in range(int(local_batch_size)):
                                gate_override = float(
                                    learned_gate_fusion_1d[j].detach().float().item()
                                )
                                trk_score_f = None
                                if (
                                    isinstance(trk_score_by_obj, dict)
                                    and isinstance(obj_ids_local, list)
                                    and j < len(obj_ids_local)
                                ):
                                    oid = int(obj_ids_local[j])
                                    if oid in trk_score_by_obj:
                                        trk_score_f = float(trk_score_by_obj.get(oid))
                                s_mem, _ = _fusion_scales_for_one(
                                    det_score_global,
                                    qcos_global,
                                    gate_override=gate_override,
                                    tracker_score_f=trk_score_f,
                                )
                                if s_mem is None:
                                    continue
                                scale_t = local_maskmem_features.new_tensor(float(s_mem)).view(
                                    1, 1, 1, 1
                                )
                                local_maskmem_features[j : j + 1] = local_maskmem_features[
                                    j : j + 1
                                ] + resid * scale_t
                    else:
                        fusion_scale_mem: float | None = None
                        fusion_scale_obj: float | None = None
                        if (
                            isinstance(spme_query_vec, torch.Tensor)
                            and spme_query_vec.numel() > 0
                            and spme_det_score_raw is not None
                        ):
                            trk_score_f = None
                            if isinstance(trk_score_by_obj, dict):
                                try:
                                    obj_ids_local = [int(x) for x in tracker_state.get("obj_ids", [])]
                                    if obj_ids_local and int(obj_ids_local[0]) in trk_score_by_obj:
                                        trk_score_f = float(trk_score_by_obj.get(int(obj_ids_local[0])))
                                except Exception:
                                    trk_score_f = None
                            fusion_scale_mem, fusion_scale_obj = _fusion_scales_for_one(
                                float(spme_det_score_raw),
                                float(spme_query_cos) if spme_query_cos is not None else None,
                                tracker_score_f=trk_score_f,
                            )
                        if (fusion_scale_mem is not None or fusion_scale_obj is not None) and isinstance(
                            spme_query_vec, torch.Tensor
                        ):
                            qv = spme_query_vec.detach()
                            if qv.ndim > 1:
                                qv = qv.reshape(-1)
                            qv = qv.to(device=local_maskmem_features.device, dtype=torch.float32)
                            qv = qv / qv.norm().clamp_min(1e-6)
                            qv = qv.view(1, -1)  # (1, D)

                            if fusion_scale_mem is not None:
                                if fusion_mode in {"film", "filmedit"}:
                                    film = self.spme_fusion_mem_film_mlp(qv)
                                    gamma, beta = film[:, :mem_dim], film[:, mem_dim:]
                                    gamma = torch.tanh(gamma).to(local_maskmem_features.dtype)
                                    beta = torch.tanh(beta).to(local_maskmem_features.dtype)
                                    scale = float(fusion_scale_mem)
                                    local_maskmem_features = local_maskmem_features * (
                                        1.0 + gamma.view(1, mem_dim, 1, 1) * scale
                                    ) + beta.view(1, mem_dim, 1, 1) * scale
                                else:
                                    if int(qv.shape[1]) == mem_dim:
                                        q_mem = qv
                                    else:
                                        proj = getattr(self.tracker, "obj_ptr_tpos_proj", None)
                                        if isinstance(proj, nn.Module):
                                            q_mem = proj(qv)
                                        else:
                                            q_mem = qv[:, :mem_dim]
                                            if int(q_mem.shape[1]) < mem_dim:
                                                q_mem = F.pad(
                                                    q_mem, (0, mem_dim - int(q_mem.shape[1]))
                                                )
                                    q_mem = q_mem.to(
                                        device=local_maskmem_features.device,
                                        dtype=local_maskmem_features.dtype,
                                    )
                                    resid = q_mem.view(1, mem_dim, 1, 1)
                                    local_maskmem_features = local_maskmem_features + resid * float(
                                        fusion_scale_mem
                                    )

            # Optional: hard "do not write" on mismatch frames (memory hygiene).
            #
            # This is applied after learned-gate and fusion edits, so it cleanly overrides any write.
            # We only trigger when det↔trk IoU is available and below threshold; missing/NaN IoU does
            # not trigger skipping by default (conservative).
            spme_skip_write_1d: torch.Tensor | None = None
            spme_skip_write_flags: list[float] | None = None
            if skip_write_on_mismatch and obj_ids_local:
                thr = max(1e-6, float(skip_write_mismatch_iou_thr))
                skip_flags: list[float] = []
                for obj_id in obj_ids_local:
                    # Determine whether this object should skip memory write.
                    #
                    # NOTE: det↔trk IoU mismatch can be unreliable under detector domain shift, so
                    # we support uncertainty-triggered skip-write based on tracker_score.

                    # (A) mismatch trigger (requires det↔trk IoU)
                    miou = float("nan")
                    if isinstance(spme_det_trk_iou_by_obj, dict):
                        v = spme_det_trk_iou_by_obj.get(int(obj_id), None)
                        try:
                            miou = float(v) if v is not None else float("nan")
                        except Exception:
                            miou = float("nan")
                    mismatch_skip = (not math.isnan(miou)) and (miou < thr)

                    # (B) tracker-score trigger (requires tracker_score)
                    ts = float("nan")
                    if isinstance(trk_score_by_obj, dict) and int(obj_id) in trk_score_by_obj:
                        try:
                            ts = float(trk_score_by_obj.get(int(obj_id)))
                        except Exception:
                            ts = float("nan")
                    score_skip = (not math.isnan(ts)) and (float(ts) < float(skip_write_tracker_thr))

                    mode = skip_write_event_mode
                    if mode in {"mismatch", "iou", "det_trk_iou"}:
                        do_skip = mismatch_skip
                    elif mode in {"tracker", "tracker_score", "score"}:
                        do_skip = score_skip
                    elif mode in {"hybrid", "both", "and"}:
                        do_skip = bool(mismatch_skip and score_skip)
                    else:
                        # Unknown mode: fall back to mismatch behavior for backward compatibility.
                        do_skip = mismatch_skip
                    if do_skip and skip_write_use_det_qcos:
                        # Require a reliable overseer signal before skipping writes.
                        det_f = (
                            float(spme_det_score_raw_by_obj.get(obj_id))
                            if isinstance(spme_det_score_raw_by_obj, dict)
                            and obj_id in spme_det_score_raw_by_obj
                            else (float(spme_det_score_raw) if spme_det_score_raw is not None else float("nan"))
                        )
                        qcos_f = (
                            float(spme_query_cos_by_obj.get(obj_id))
                            if isinstance(spme_query_cos_by_obj, dict)
                            and obj_id in spme_query_cos_by_obj
                            else (float(spme_query_cos) if spme_query_cos is not None else float("nan"))
                        )
                        det_ok = (not math.isnan(det_f)) and (det_f >= float(skip_write_det_thr))
                        q_ok = (not math.isnan(qcos_f)) and (qcos_f >= float(skip_write_qcos_thr))
                        do_skip = bool(det_ok and q_ok)
                    if do_skip and skip_write_require_reid_det_better:
                        m = (
                            float(spme_reid_det_vs_trk_margin_by_obj.get(int(obj_id)))
                            if isinstance(spme_reid_det_vs_trk_margin_by_obj, dict)
                            and int(obj_id) in spme_reid_det_vs_trk_margin_by_obj
                            else float("nan")
                        )
                        # Conservative: only skip when we have positive evidence that detector is
                        # more consistent than tracker.
                        do_skip = (not math.isnan(m)) and (m >= float(skip_write_reid_margin))
                    skip_flags.append(1.0 if do_skip else 0.0)

                if any(x > 0.0 for x in skip_flags):
                    prev_frame_idx = frame_idx + 1 if track_in_reverse else frame_idx - 1
                    prev_out = None
                    if prev_frame_idx >= 0:
                        prev_out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                        if prev_out is None:
                            prev_out = output_dict["cond_frame_outputs"].get(prev_frame_idx, None)
                    if isinstance(prev_out, dict):
                        prev_feat = prev_out.get("maskmem_features", None)
                        if (
                            isinstance(prev_feat, torch.Tensor)
                            and prev_feat.shape == local_maskmem_features.shape
                        ):
                            prev_feat = prev_feat.to(
                                device=local_maskmem_features.device,
                                dtype=local_maskmem_features.dtype,
                            )
                            for j, sf in enumerate(skip_flags):
                                if sf > 0.0:
                                    local_maskmem_features[j : j + 1] = prev_feat[j : j + 1]

                spme_skip_write_1d = torch.tensor(
                    skip_flags, dtype=torch.float32, device=torch.device("cpu")
                ).view(-1, 1)
                spme_skip_write_flags = skip_flags
            # Store encoded memories in the local inference state
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                if frame_idx not in output_dict[storage_key]:
                    continue
                output_dict[storage_key][frame_idx]["maskmem_features"] = (
                    local_maskmem_features
                )
                output_dict[storage_key][frame_idx]["maskmem_pos_enc"] = [
                    pos for pos in local_maskmem_pos_enc
                ]
                if spme_skip_write_1d is not None:
                    output_dict[storage_key][frame_idx]["spme_skip_write"] = spme_skip_write_1d
                if (
                    learned_gate_log_to_out
                    and learned_gate_mem_scale_1d is not None
                    and learned_gate_mem_offset_1d is not None
                    and learned_gate_decay_1d is not None
                ):
                    output_dict[storage_key][frame_idx]["spme_gate_mem_scale_mean"] = (
                        learned_gate_mem_scale_1d.detach().float().mean(dim=-1, keepdim=True).cpu()
                    )
                    output_dict[storage_key][frame_idx]["spme_gate_mem_offset_mean_abs"] = (
                        learned_gate_mem_offset_1d.detach()
                        .float()
                        .abs()
                        .mean(dim=-1, keepdim=True)
                        .cpu()
                    )
                    if learned_gate_fusion_1d is not None:
                        output_dict[storage_key][frame_idx]["spme_gate_fusion"] = (
                            learned_gate_fusion_1d.detach().float().cpu()
                        )
                    output_dict[storage_key][frame_idx]["spme_gate_decay"] = (
                        learned_gate_decay_1d.detach().float().cpu()
                    )
                # Optionally log SPME detector/pointer signals and effective fusion scales.
                if signals_log_to_out:
                    try:
                        # Per-object det_score/qcos (or fall back to global).
                        det_vals_out: list[float] = []
                        qcos_vals_out: list[float] = []
                        presence_vals_out: list[float] = []
                        iou_vals_out: list[float] = []
                        reinit_vals_out: list[float] = []
                        reid_best_sim_vals_out: list[float] = []
                        reid_margin_vals_out: list[float] = []
                        reid_accept_vals_out: list[float] = []
                        reid_bank_size_vals_out: list[float] = []
                        reid_trk_sim_vals_out: list[float] = []
                        reid_det_vs_trk_margin_vals_out: list[float] = []
                        for obj_id in obj_ids_local:
                            det_f = (
                                float(spme_det_score_raw_by_obj.get(obj_id))
                                if isinstance(spme_det_score_raw_by_obj, dict)
                                and obj_id in spme_det_score_raw_by_obj
                                else (float(spme_det_score_raw) if spme_det_score_raw is not None else float("nan"))
                            )
                            qcos_f = (
                                float(spme_query_cos_by_obj.get(obj_id))
                                if isinstance(spme_query_cos_by_obj, dict)
                                and obj_id in spme_query_cos_by_obj
                                else (float(spme_query_cos) if spme_query_cos is not None else float("nan"))
                            )
                            det_vals_out.append(det_f)
                            qcos_vals_out.append(qcos_f)
                            presence_vals_out.append(
                                float(spme_presence_prob) if spme_presence_prob is not None else float("nan")
                            )
                            iou_vals_out.append(
                                float(spme_det_trk_iou_by_obj.get(obj_id))
                                if isinstance(spme_det_trk_iou_by_obj, dict)
                                and obj_id in spme_det_trk_iou_by_obj
                                else float("nan")
                            )
                            reinit_vals_out.append(
                                1.0
                                if isinstance(spme_reinit_by_obj, dict) and int(obj_id) in spme_reinit_by_obj
                                else 0.0
                            )
                            reid_best_sim_vals_out.append(
                                float(spme_reid_best_sim_by_obj.get(obj_id))
                                if isinstance(spme_reid_best_sim_by_obj, dict)
                                and int(obj_id) in spme_reid_best_sim_by_obj
                                else float("nan")
                            )
                            reid_margin_vals_out.append(
                                float(spme_reid_margin_by_obj.get(obj_id))
                                if isinstance(spme_reid_margin_by_obj, dict)
                                and int(obj_id) in spme_reid_margin_by_obj
                                else float("nan")
                            )
                            reid_accept_vals_out.append(
                                1.0
                                if isinstance(spme_reid_accept_by_obj, dict)
                                and int(spme_reid_accept_by_obj.get(int(obj_id), 0)) > 0
                                else 0.0
                            )
                            reid_bank_size_vals_out.append(
                                float(spme_reid_bank_size_by_obj.get(obj_id))
                                if isinstance(spme_reid_bank_size_by_obj, dict)
                                and int(obj_id) in spme_reid_bank_size_by_obj
                                else float("nan")
                            )
                            reid_trk_sim_vals_out.append(
                                float(spme_reid_trk_sim_by_obj.get(obj_id))
                                if isinstance(spme_reid_trk_sim_by_obj, dict)
                                and int(obj_id) in spme_reid_trk_sim_by_obj
                                else float("nan")
                            )
                            reid_det_vs_trk_margin_vals_out.append(
                                float(spme_reid_det_vs_trk_margin_by_obj.get(obj_id))
                                if isinstance(spme_reid_det_vs_trk_margin_by_obj, dict)
                                and int(obj_id) in spme_reid_det_vs_trk_margin_by_obj
                                else float("nan")
                            )

                        output_dict[storage_key][frame_idx]["spme_det_score_raw"] = (
                            torch.tensor(det_vals_out, dtype=torch.float32).view(-1, 1).cpu()
                        )
                        output_dict[storage_key][frame_idx]["spme_query_cos"] = (
                            torch.tensor(qcos_vals_out, dtype=torch.float32).view(-1, 1).cpu()
                        )
                        output_dict[storage_key][frame_idx]["spme_presence_prob"] = (
                            torch.tensor(presence_vals_out, dtype=torch.float32).view(-1, 1).cpu()
                        )
                        output_dict[storage_key][frame_idx]["spme_det_trk_iou"] = (
                            torch.tensor(iou_vals_out, dtype=torch.float32).view(-1, 1).cpu()
                        )
                        output_dict[storage_key][frame_idx]["spme_reinit"] = (
                            torch.tensor(reinit_vals_out, dtype=torch.float32).view(-1, 1).cpu()
                        )
                        output_dict[storage_key][frame_idx]["spme_reid_best_sim"] = (
                            torch.tensor(reid_best_sim_vals_out, dtype=torch.float32).view(-1, 1).cpu()
                        )
                        output_dict[storage_key][frame_idx]["spme_reid_margin"] = (
                            torch.tensor(reid_margin_vals_out, dtype=torch.float32).view(-1, 1).cpu()
                        )
                        output_dict[storage_key][frame_idx]["spme_reid_accept"] = (
                            torch.tensor(reid_accept_vals_out, dtype=torch.float32).view(-1, 1).cpu()
                        )
                        output_dict[storage_key][frame_idx]["spme_reid_bank_size"] = (
                            torch.tensor(reid_bank_size_vals_out, dtype=torch.float32).view(-1, 1).cpu()
                        )
                        output_dict[storage_key][frame_idx]["spme_reid_trk_sim"] = (
                            torch.tensor(reid_trk_sim_vals_out, dtype=torch.float32).view(-1, 1).cpu()
                        )
                        output_dict[storage_key][frame_idx]["spme_reid_det_vs_trk_margin"] = (
                            torch.tensor(reid_det_vs_trk_margin_vals_out, dtype=torch.float32).view(-1, 1).cpu()
                        )

                        if fusion_enabled and (base_alpha > 0.0 or base_alpha_obj > 0.0):
                            scale_mem_vals: list[float] = []
                            scale_obj_vals: list[float] = []

                            per_obj_ctx_ok = (
                                fusion_per_object
                                and isinstance(spme_query_vec_by_obj, dict)
                                and isinstance(spme_det_score_raw_by_obj, dict)
                                and len(spme_query_vec_by_obj) > 0
                                and len(spme_det_score_raw_by_obj) > 0
                            )
                            do_global_fusion = (not fusion_per_object) or (
                                fusion_per_object_fallback and (not per_obj_ctx_ok)
                            )

                            for j, obj_id in enumerate(obj_ids_local):
                                has_ptr = False
                                if per_obj_ctx_ok:
                                    qv_src = spme_query_vec_by_obj.get(obj_id, None)
                                    has_ptr = isinstance(qv_src, torch.Tensor) and qv_src.numel() > 0
                                elif do_global_fusion:
                                    has_ptr = (
                                        isinstance(spme_query_vec, torch.Tensor) and spme_query_vec.numel() > 0
                                    )

                                det_f = det_vals_out[j]
                                qcos_f = qcos_vals_out[j]
                                iou_f = iou_vals_out[j] if j < len(iou_vals_out) else float("nan")
                                trk_score_f = (
                                    float(trk_score_by_obj.get(obj_id))
                                    if isinstance(trk_score_by_obj, dict) and obj_id in trk_score_by_obj
                                    else None
                                )

                                s_mem = 0.0
                                s_obj = 0.0
                                if has_ptr and not (math.isnan(det_f) or det_f is None):
                                    gate_override = (
                                        float(learned_gate_fusion_1d[j].detach().float().item())
                                        if (learned_gate_enabled and learned_gate_fusion_1d is not None)
                                        else (
                                            gate_by_obj.get(obj_id, None) if gate_by_obj is not None else None
                                        )
                                    )
                                    sm, so = _fusion_scales_for_one(
                                        float(det_f),
                                        None if (qcos_f is None or math.isnan(float(qcos_f))) else float(qcos_f),
                                        gate_override=gate_override,
                                        tracker_score_f=trk_score_f,
                                        det_trk_iou_f=(
                                            None
                                            if (iou_f is None or math.isnan(float(iou_f)))
                                            else float(iou_f)
                                        ),
                                    )
                                    s_mem = float(sm) if sm is not None else 0.0
                                    s_obj = float(so) if so is not None else 0.0

                                scale_mem_vals.append(float(s_mem))
                                scale_obj_vals.append(float(s_obj))

                            output_dict[storage_key][frame_idx]["spme_fusion_scale_mem"] = (
                                torch.tensor(scale_mem_vals, dtype=torch.float32).view(-1, 1).cpu()
                            )
                            output_dict[storage_key][frame_idx]["spme_fusion_scale_obj"] = (
                                torch.tensor(scale_obj_vals, dtype=torch.float32).view(-1, 1).cpu()
                            )
                    except Exception:
                        # Best-effort logging; never affect inference.
                        pass
                # Apply obj_ptr edits.
                if fusion_enabled and base_alpha_obj > 0.0:
                    obj_ptr = output_dict[storage_key][frame_idx].get("obj_ptr", None)
                    if isinstance(obj_ptr, torch.Tensor) and obj_ptr.numel() > 0:
                        per_obj_ctx_ok = (
                            fusion_per_object
                            and isinstance(spme_query_vec_by_obj, dict)
                            and isinstance(spme_det_score_raw_by_obj, dict)
                            and len(spme_query_vec_by_obj) > 0
                            and len(spme_det_score_raw_by_obj) > 0
                        )
                        do_global_obj_edit = (not fusion_per_object) or (
                            fusion_per_object_fallback and (not per_obj_ctx_ok)
                        )

                        if per_obj_ctx_ok:
                            obj_ids_local = [int(x) for x in tracker_state.get("obj_ids", [])]
                            for j, obj_id in enumerate(obj_ids_local):
                                qv_src = spme_query_vec_by_obj.get(obj_id, None)
                                if not isinstance(qv_src, torch.Tensor) or qv_src.numel() == 0:
                                    continue
                                det_score_f = spme_det_score_raw_by_obj.get(obj_id, None)
                                trk_score_f = (
                                    float(trk_score_by_obj.get(obj_id))
                                    if isinstance(trk_score_by_obj, dict) and obj_id in trk_score_by_obj
                                    else None
                                )
                                qcos_f = (
                                    spme_query_cos_by_obj.get(obj_id, None)
                                    if isinstance(spme_query_cos_by_obj, dict)
                                    else None
                                )
                                det_trk_iou_f = (
                                    spme_det_trk_iou_by_obj.get(obj_id, None)
                                    if isinstance(spme_det_trk_iou_by_obj, dict)
                                    else None
                                )
                                gate_override = (
                                    float(learned_gate_fusion_1d[j].detach().float().item())
                                    if (learned_gate_enabled and learned_gate_fusion_1d is not None)
                                    else (
                                        gate_by_obj.get(obj_id, None) if gate_by_obj is not None else None
                                    )
                                )
                                _, s_obj = _fusion_scales_for_one(
                                    det_score_f,
                                    qcos_f,
                                    gate_override=gate_override,
                                    tracker_score_f=trk_score_f,
                                    det_trk_iou_f=det_trk_iou_f,
                                )
                                if s_obj is None:
                                    continue
                                qv = qv_src.detach()
                                if qv.ndim > 1:
                                    qv = qv.reshape(-1)
                                qv = qv.to(device=obj_ptr.device, dtype=torch.float32)
                                qv = qv / qv.norm().clamp_min(1e-6)
                                qv = qv.view(1, -1)
                                delta = self.spme_fusion_obj_ptr_mlp(qv).to(
                                    device=obj_ptr.device, dtype=obj_ptr.dtype
                                )
                                obj_ptr[j : j + 1] = obj_ptr[j : j + 1] + delta * float(s_obj)
                            output_dict[storage_key][frame_idx]["obj_ptr"] = obj_ptr

                        if do_global_obj_edit:
                            if (
                                learned_gate_enabled
                                and learned_gate_fusion_1d is not None
                                and isinstance(spme_query_vec, torch.Tensor)
                                and spme_query_vec.numel() > 0
                                and spme_det_score_raw is not None
                            ):
                                qv = spme_query_vec.detach()
                                if qv.ndim > 1:
                                    qv = qv.reshape(-1)
                                qv = qv.to(device=obj_ptr.device, dtype=torch.float32)
                                qv = qv / qv.norm().clamp_min(1e-6)
                                qv = qv.view(1, -1)
                                delta = self.spme_fusion_obj_ptr_mlp(qv).to(
                                    device=obj_ptr.device, dtype=obj_ptr.dtype
                                )

                                qcos_global = (
                                    None
                                    if (spme_query_cos is None or math.isnan(float(spme_query_cos)))
                                    else float(spme_query_cos)
                                )
                                det_score_global = float(spme_det_score_raw)
                                for j in range(int(obj_ptr.shape[0])):
                                    gate_override = float(
                                        learned_gate_fusion_1d[j].detach().float().item()
                                    )
                                    trk_score_f = None
                                    if (
                                        isinstance(trk_score_by_obj, dict)
                                        and isinstance(obj_ids_local, list)
                                        and j < len(obj_ids_local)
                                    ):
                                        oid = int(obj_ids_local[j])
                                        if oid in trk_score_by_obj:
                                            trk_score_f = float(trk_score_by_obj.get(oid))
                                    _, s_obj = _fusion_scales_for_one(
                                        det_score_global,
                                        qcos_global,
                                        gate_override=gate_override,
                                        tracker_score_f=trk_score_f,
                                    )
                                    if s_obj is None:
                                        continue
                                    obj_ptr[j : j + 1] = obj_ptr[j : j + 1] + delta * float(s_obj)
                                output_dict[storage_key][frame_idx]["obj_ptr"] = obj_ptr
                            else:
                                if (
                                    isinstance(spme_query_vec, torch.Tensor)
                                    and spme_query_vec.numel() > 0
                                    and spme_det_score_raw is not None
                                ):
                                    trk_score_f = None
                                    if isinstance(trk_score_by_obj, dict):
                                        try:
                                            obj_ids_local = [
                                                int(x) for x in tracker_state.get("obj_ids", [])
                                            ]
                                            if obj_ids_local and int(obj_ids_local[0]) in trk_score_by_obj:
                                                trk_score_f = float(
                                                    trk_score_by_obj.get(int(obj_ids_local[0]))
                                                )
                                        except Exception:
                                            trk_score_f = None
                                    _, s_obj = _fusion_scales_for_one(
                                        float(spme_det_score_raw),
                                        float(spme_query_cos) if spme_query_cos is not None else None,
                                        tracker_score_f=trk_score_f,
                                    )
                                    if s_obj is not None:
                                        qv = spme_query_vec.detach()
                                        if qv.ndim > 1:
                                            qv = qv.reshape(-1)
                                        qv = qv.to(device=obj_ptr.device, dtype=torch.float32)
                                        qv = qv / qv.norm().clamp_min(1e-6)
                                        qv = qv.view(1, -1)
                                        delta = self.spme_fusion_obj_ptr_mlp(qv).to(
                                            device=obj_ptr.device, dtype=obj_ptr.dtype
                                        )
                                        output_dict[storage_key][frame_idx]["obj_ptr"] = (
                                            obj_ptr + delta * float(s_obj)
                                        )
                if apply_mode in {"blend_mem", "blend", "ema"}:
                    prev_frame_idx = frame_idx + 1 if track_in_reverse else frame_idx - 1
                    prev_out = None
                    if prev_frame_idx >= 0:
                        prev_out = output_dict[storage_key].get(prev_frame_idx, None)
                    if prev_out is None and prev_frame_idx >= 0:
                        other_key = (
                            "cond_frame_outputs"
                            if storage_key == "non_cond_frame_outputs"
                            else "non_cond_frame_outputs"
                        )
                        prev_out = output_dict[other_key].get(prev_frame_idx, None)
                    if isinstance(prev_out, dict):
                        prev_ptr = prev_out.get("obj_ptr", None)
                        curr_ptr = output_dict[storage_key][frame_idx].get("obj_ptr", None)
                        if (
                            isinstance(prev_ptr, torch.Tensor)
                            and isinstance(curr_ptr, torch.Tensor)
                            and prev_ptr.shape == curr_ptr.shape
                        ):
                            gate_ptr: Tensor | float | None = None
                            gate_ptr_decay: Tensor | None = None
                            if (
                                learned_gate_enabled
                                and learned_gate_fusion_1d is not None
                                and learned_gate_decay_1d is not None
                            ):
                                gate_ptr = learned_gate_fusion_1d.to(
                                    device=curr_ptr.device, dtype=curr_ptr.dtype
                                ).view(-1, 1)
                                gate_ptr_decay = learned_gate_decay_1d.to(
                                    device=curr_ptr.device, dtype=curr_ptr.dtype
                                ).view(-1, 1)
                            elif gate_by_obj is not None:
                                obj_ids_local = [int(x) for x in tracker_state.get("obj_ids", [])]
                                gate_vals = [
                                    float(gate_by_obj.get(obj_id, 1.0)) for obj_id in obj_ids_local
                                ]
                                gate_ptr = curr_ptr.new_tensor(gate_vals).view(-1, 1)
                                if float(gate_ptr.detach().float().min().item()) >= 1.0:
                                    gate_ptr = None
                            elif gate is not None and gate < 1.0:
                                gate_ptr = float(gate)

                            if gate_ptr is not None:
                                prev_ptr = prev_ptr.to(
                                    device=curr_ptr.device,
                                    dtype=curr_ptr.dtype,
                                )
                                if learned_gate_enabled and gate_ptr_decay is not None and learned_gate_use_decay:
                                    output_dict[storage_key][frame_idx]["obj_ptr"] = (
                                        curr_ptr * gate_ptr
                                        + prev_ptr * (1.0 - gate_ptr) * (1.0 - gate_ptr_decay)
                                    )
                                else:
                                    output_dict[storage_key][frame_idx]["obj_ptr"] = (
                                        curr_ptr * gate_ptr + prev_ptr * (1.0 - gate_ptr)
                                    )

                # Keep pointer/memory consistent on skip-write frames (v2).
                #
                # NOTE: Skip-write overrides any fusion edits and blending, so it behaves as a strict
                # "do not write" policy. This is only enabled when SAM3_SPME_SKIP_WRITE_MODE=full.
                if (
                    skip_write_freeze_ptr
                    and isinstance(spme_skip_write_flags, list)
                    and any(sf > 0.0 for sf in spme_skip_write_flags)
                ):
                    prev_frame_idx = frame_idx + 1 if track_in_reverse else frame_idx - 1
                    prev_out = None
                    if prev_frame_idx >= 0:
                        prev_out = output_dict[storage_key].get(prev_frame_idx, None)
                    if prev_out is None and prev_frame_idx >= 0:
                        other_key = (
                            "cond_frame_outputs"
                            if storage_key == "non_cond_frame_outputs"
                            else "non_cond_frame_outputs"
                        )
                        prev_out = output_dict[other_key].get(prev_frame_idx, None)
                    if isinstance(prev_out, dict):
                        prev_ptr = prev_out.get("obj_ptr", None)
                        curr_ptr = output_dict[storage_key][frame_idx].get("obj_ptr", None)
                        if (
                            isinstance(prev_ptr, torch.Tensor)
                            and isinstance(curr_ptr, torch.Tensor)
                            and prev_ptr.shape == curr_ptr.shape
                            and curr_ptr.numel() > 0
                        ):
                            prev_ptr = prev_ptr.to(
                                device=curr_ptr.device, dtype=curr_ptr.dtype
                            )
                            # Apply per-object freezing in-place (index-aligned).
                            for j, sf in enumerate(spme_skip_write_flags):
                                if sf > 0.0:
                                    curr_ptr[j : j + 1] = prev_ptr[j : j + 1]
                            output_dict[storage_key][frame_idx]["obj_ptr"] = curr_ptr
                # for batched inference state, we also need to add per-object
                # memory slides to support instance interactivity
                self.tracker._add_output_per_object(
                    inference_state=tracker_state,
                    frame_idx=frame_idx,
                    current_out=output_dict[storage_key][frame_idx],
                    storage_key=storage_key,
                )
            start_idx_state += num_obj_per_state

    def _tracker_add_new_objects(
        self,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: List[int],
        new_obj_masks: Tensor,
        tracker_states_local: List[Any],
        orig_vid_height: int,
        orig_vid_width: int,
        feature_cache: Dict,
    ):
        """Add a new object to SAM2 inference states."""
        prev_tracker_state = (
            tracker_states_local[0] if len(tracker_states_local) > 0 else None
        )

        # prepare inference_state
        # batch objects that first appear on the same frame together
        # Clear inference state. Keep the cached image features if available.
        new_tracker_state = self.tracker.init_state(
            cached_features=feature_cache,
            video_height=orig_vid_height,
            video_width=orig_vid_width,
            num_frames=num_frames,
        )
        new_tracker_state["backbone_out"] = (
            prev_tracker_state.get("backbone_out", None)
            if prev_tracker_state is not None
            else None
        )

        assert len(new_obj_ids) == new_obj_masks.size(0)
        assert new_obj_masks.is_floating_point()
        input_mask_res = self.tracker.input_mask_size
        new_obj_masks = F.interpolate(
            new_obj_masks.unsqueeze(1),
            size=(input_mask_res, input_mask_res),
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
        new_obj_masks = new_obj_masks > 0

        # add object one by one
        for new_obj_id, new_mask in zip(new_obj_ids, new_obj_masks):
            self.tracker.add_new_mask(
                inference_state=new_tracker_state,
                frame_idx=frame_idx,
                obj_id=new_obj_id,
                mask=new_mask,
                add_mask_to_memory=True,
            )
        # NOTE: we skip enforcing the non-overlapping constraint **globally** when adding new objects.
        self.tracker.propagate_in_video_preflight(
            new_tracker_state, run_mem_encoder=True
        )
        tracker_states_local.append(new_tracker_state)
        return tracker_states_local

    def _tracker_remove_object(self, tracker_states_local: List[Any], obj_id: int):
        """
        Remove an object from SAM2 inference states. This would remove the object from
        all frames in the video.
        """
        tracker_states_local_before_removal = tracker_states_local.copy()
        tracker_states_local.clear()
        for tracker_inference_state in tracker_states_local_before_removal:
            # we try to remove `obj_id` on every inference state with `strict=False`
            # it will not do anything if an inference state doesn't contain `obj_id`
            new_obj_ids, _ = self.tracker.remove_object(
                tracker_inference_state, obj_id, strict=False, need_output=False
            )
            # only keep an inference state if it's non-empty after object removal
            if len(new_obj_ids) > 0:
                tracker_states_local.append(tracker_inference_state)

    def _tracker_remove_objects(
        self, tracker_states_local: List[Any], obj_ids: list[int]
    ):
        """
        Remove an object from SAM2 inference states. This would remove the object from
        all frames in the video.
        """
        for obj_id in obj_ids:
            self._tracker_remove_object(tracker_states_local, obj_id)

    def _initialize_metadata(self):
        """Initialize metadata for the masklets."""
        tracker_metadata = {
            "obj_ids_per_gpu": [np.array([], np.int64) for _ in range(self.world_size)],
            "obj_ids_all_gpu": np.array([], np.int64),
            "num_obj_per_gpu": np.zeros(self.world_size, np.int64),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "obj_id_to_tracker_score_frame_wise": defaultdict(dict),
            "obj_id_to_last_occluded": {},
        }
        if self.rank == 0:
            # "rank0_metadata" contains metadata that is only stored on (and accessible to) GPU 0
            # - obj_first_frame_idx: obj_id --> first frame index where the object was detected
            # - unmatched_frame_inds: obj_id --> [mismatched frame indices]
            # - overlap_pair_to_frame_inds: (first_appear_obj_id, obj_id) --> [overlap frame indices]
            # - removed_obj_ids: object IDs that are suppressed via hot-start
            rank0_metadata = {
                "obj_first_frame_idx": {},
                "unmatched_frame_inds": defaultdict(list),
                "trk_keep_alive": defaultdict(
                    int
                ),  # This is used only for object suppression not for removal
                "overlap_pair_to_frame_inds": defaultdict(list),
                "removed_obj_ids": set(),
                "suppressed_obj_ids": defaultdict(
                    set
                ),  # frame_idx --> set of objects with suppressed outputs, but still continue to be tracked
            }
            if self.masklet_confirmation_enable:
                # all the following are npt.NDArray with the same shape as `obj_ids_all_gpu`
                rank0_metadata["masklet_confirmation"] = {
                    # "status" is the confirmation status of each masklet (in `MaskletConfirmationStatus`)
                    "status": np.array([], np.int64),
                    # "consecutive_det_num" is the number of consecutive frames where the masklet is
                    # detected by the detector (with a matched detection)
                    "consecutive_det_num": np.array([], np.int64),
                }
            tracker_metadata["rank0_metadata"] = rank0_metadata

        return tracker_metadata

    def update_masklet_confirmation_status(
        self,
        rank0_metadata: Dict[str, Any],
        obj_ids_all_gpu_prev: npt.NDArray,
        obj_ids_all_gpu_updated: npt.NDArray,
        det_to_matched_trk_obj_ids: Dict[int, npt.NDArray],
        new_det_obj_ids: npt.NDArray,
    ):
        confirmation_data = rank0_metadata["masklet_confirmation"]

        # a) first, expand "confirmation_data" to include new masklets added in this frame
        status_prev = confirmation_data["status"]
        consecutive_det_num_prev = confirmation_data["consecutive_det_num"]
        assert (
            status_prev.shape == obj_ids_all_gpu_prev.shape
        ), f"Got {status_prev.shape} vs {obj_ids_all_gpu_prev.shape}"

        obj_id_to_updated_idx = {
            obj_id: idx for idx, obj_id in enumerate(obj_ids_all_gpu_updated)
        }
        prev_elem_is_in_updated = np.isin(obj_ids_all_gpu_prev, obj_ids_all_gpu_updated)
        prev_elem_obj_ids_in_updated = obj_ids_all_gpu_prev[prev_elem_is_in_updated]
        prev_elem_inds_in_updated = np.array(
            [obj_id_to_updated_idx[obj_id] for obj_id in prev_elem_obj_ids_in_updated],
            dtype=np.int64,
        )
        # newly added masklets are initialized to "UNCONFIRMED" status
        unconfirmed_val = MaskletConfirmationStatus.UNCONFIRMED.value
        status = np.full_like(obj_ids_all_gpu_updated, fill_value=unconfirmed_val)
        status[prev_elem_inds_in_updated] = status_prev[prev_elem_is_in_updated]
        consecutive_det_num = np.zeros_like(obj_ids_all_gpu_updated)
        consecutive_det_num[prev_elem_inds_in_updated] = consecutive_det_num_prev[
            prev_elem_is_in_updated
        ]

        # b) update the confirmation status of all masklets based on the current frame
        # b.1) update "consecutive_det_num"
        # "is_matched": whether a masklet is matched to a detection on this frame
        is_matched = np.isin(obj_ids_all_gpu_updated, new_det_obj_ids)
        for matched_trk_obj_ids in det_to_matched_trk_obj_ids.values():
            is_matched |= np.isin(obj_ids_all_gpu_updated, matched_trk_obj_ids)
        consecutive_det_num = np.where(is_matched, consecutive_det_num + 1, 0)

        # b.2) update "status"
        change_to_confirmed = (
            consecutive_det_num >= self.masklet_confirmation_consecutive_det_thresh
        )
        status[change_to_confirmed] = MaskletConfirmationStatus.CONFIRMED.value

        confirmation_data["status"] = status
        confirmation_data["consecutive_det_num"] = consecutive_det_num
        return rank0_metadata

    def forward(self, input: BatchedDatapoint, is_inference: bool = False):
        raise NotImplementedError("Evaluation outside demo is not implemented yet")

    def _load_checkpoint(self, ckpt_path: str, strict: bool = True):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=strict)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            logger.warning(f"Loaded ckpt with {missing_keys=}, {unexpected_keys=}")
        else:
            logger.info("Loaded ckpt successfully without missing or unexpected keys")

    def prep_for_evaluator(self, video_frames, tracking_res, scores_labels):
        """This method is only used for benchmark eval (not used in the demo)."""
        num_frames = len(video_frames)
        w, h = video_frames[0].size
        zero_mask = torch.zeros((1, h, w), dtype=torch.bool)
        object_ids = list(scores_labels.keys())
        preds = {"scores": [], "labels": [], "boxes": [], "masks_rle": []}
        for oid in object_ids:
            o_masks = []
            o_score = scores_labels[oid][0].item()
            o_label = scores_labels[oid][1]
            for frame_idx in range(num_frames):
                if frame_idx not in tracking_res:
                    o_masks.append(zero_mask)
                else:
                    o_masks.append(tracking_res[frame_idx].get(oid, zero_mask))

            o_masks = torch.cat(o_masks, dim=0)  # (n_frames, H, W)
            preds["scores"].append(o_score)
            preds["labels"].append(o_label)
            preds["boxes"].append(mask_to_box(o_masks.unsqueeze(1)).squeeze())
            preds["masks_rle"].append(rle_encode(o_masks, return_areas=True))

        preds["boxes"] = (
            torch.stack(preds["boxes"], dim=0)
            if len(preds["boxes"]) > 0
            else torch.empty(
                (0, num_frames, 4), dtype=torch.float32, device=self.device
            )
        )
        preds["scores"] = (
            torch.tensor(preds["scores"], device=self.device)
            if len(preds["scores"]) > 0
            else torch.empty((0,), device=self.device)
        )
        preds["per_frame_scores"] = preds["scores"]
        preds["labels"] = (
            torch.tensor(preds["labels"], device=self.device)
            if len(preds["labels"]) > 0
            else torch.empty((0,), device=self.device)
        )
        return preds

    def _encode_prompt(self, **kwargs):
        return self.detector._encode_prompt(**kwargs)

    def _drop_new_det_with_obj_limit(self, new_det_fa_inds, det_scores_np, num_to_keep):
        """
        Drop a few new detections based on the maximum number of objects. We drop new objects based
        on their detection scores, keeping the high-scoring ones and dropping the low-scoring ones.
        """
        assert 0 <= num_to_keep <= len(new_det_fa_inds)
        if num_to_keep == 0:
            return np.array([], np.int64)  # keep none
        if num_to_keep == len(new_det_fa_inds):
            return new_det_fa_inds  # keep all

        # keep the top-scoring detections
        score_order = np.argsort(det_scores_np[new_det_fa_inds])[::-1]
        new_det_fa_inds = new_det_fa_inds[score_order[:num_to_keep]]
        return new_det_fa_inds

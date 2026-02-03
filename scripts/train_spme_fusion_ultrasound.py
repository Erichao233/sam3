#!/usr/bin/env python3
"""
Train SPME-Fusion (Presence-Gated Multi-Query FiLM Memory Editing) on ultrasound insertion-only clips.

This script is intentionally lightweight and does NOT try to reuse the hydra Trainer.
We:
  - Build a SAM3 video model from `sam3.pt`
  - Load a detector fine-tune checkpoint (e.g., ckpt28)
  - Freeze all parameters except `spme_*`
  - Sample short clips within the insertion phase (per insertion_end_train.txt)
  - Run detector per-frame to obtain semantic query vectors (Top-K weighted pooling)
  - Run tracker sequentially (no inference_mode) to get differentiable mask logits
  - Apply SPME-Fusion editing to memory features + obj_ptr
  - Optimize BCE+Dice on frames t>=1 in the clip

Outputs:
  - checkpoints/spme_fusion_stepXXXX.pt  (only spme_* params + optimizer state)
  - checkpoints/spme_fusion_latest.pt
  - train_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from sam3.model_builder import build_sam3_video_model


def _sigmoid_f(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_trainer_checkpoint_state_dict(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        return ckpt["model"]
    if isinstance(ckpt, dict):
        return ckpt
    raise ValueError(f"Unsupported checkpoint format: {path}")


def _apply_detector_finetune(video_model, finetune_ckpt: Path) -> None:
    sd = _load_trainer_checkpoint_state_dict(finetune_ckpt)
    target = getattr(video_model, "detector", None) or video_model
    missing, unexpected = target.load_state_dict(sd, strict=False)
    print(
        f"Loaded finetune into {'detector' if getattr(video_model,'detector',None) is not None else 'model'}: "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def _load_insertion_end_map(path: Path) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) < 2:
            continue
        sample_id = parts[0]
        try:
            end_frame = int(parts[1])
        except ValueError:
            continue
        mapping[sample_id] = end_frame
    return mapping


def _read_mask_np(mask_path: Path) -> np.ndarray:
    m = np.array(Image.open(mask_path))
    return (m > 0).astype(np.uint8)


def _bbox_xyxy_from_mask(mask_np: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask_np > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    return x1, y1, x2, y2


def _xyxy_to_xywh_norm(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> list[float]:
    x = x1 / w
    y = y1 / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return [float(x), float(y), float(bw), float(bh)]


def _xywh_to_cxcywh(xywh: list[float]) -> list[float]:
    x, y, w, h = xywh
    return [x + 0.5 * w, y + 0.5 * h, w, h]


def _load_frame_tensor(
    img_path: Path,
    image_size: int,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    device: torch.device | None = None,
) -> tuple[torch.Tensor, int, int]:
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.width, img.height
    img = TF.resize(img, size=(image_size, image_size))
    t = TF.to_tensor(img)  # float32 in [0,1], (3,H,W)
    t = t.to(dtype=torch.float16)
    mean_t = torch.tensor(mean, dtype=torch.float16)[:, None, None]
    std_t = torch.tensor(std, dtype=torch.float16)[:, None, None]
    t = (t - mean_t) / std_t
    if device is not None:
        t = t.to(device=device, non_blocking=True)
    return t, orig_h, orig_w


def _mask_to_low_res(
    mask_np: np.ndarray, low_res: int, device: torch.device
) -> torch.Tensor:
    m = torch.from_numpy(mask_np.astype(np.float32))[None, None]  # 1x1xH xW
    m = m.to(device=device, non_blocking=True)
    m_lr = F.interpolate(m, size=(low_res, low_res), mode="nearest")
    return (m_lr > 0.5).to(dtype=torch.float32).squeeze(0).squeeze(0)  # HxW float


def _dice_loss_from_logits(logits: torch.Tensor, targets01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(-1)
    targets = targets01.reshape(-1)
    inter = (probs * targets).sum()
    denom = probs.sum() + targets.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


@dataclass(frozen=True)
class ClipSpec:
    sample_id: str
    start: int
    length: int
    end_frame_inclusive: int


def _choose_clip(
    rng: random.Random,
    sample_ids: list[str],
    insertion_end_map: dict[str, int],
    clip_len: int,
) -> ClipSpec | None:
    if not sample_ids:
        return None
    # Sample uniformly over videos; simple and stable.
    sid = rng.choice(sample_ids)
    end_frame = int(insertion_end_map.get(sid, -1))
    if end_frame < clip_len - 1:
        return None
    start = rng.randint(0, end_frame - (clip_len - 1))
    return ClipSpec(sample_id=sid, start=start, length=clip_len, end_frame_inclusive=end_frame)


def _freeze_non_spme_params(model: torch.nn.Module) -> list[str]:
    trainable: list[str] = []
    for name, p in model.named_parameters():
        if name.startswith("spme_"):
            p.requires_grad = True
            trainable.append(name)
        else:
            p.requires_grad = False
    if not trainable:
        raise RuntimeError("No trainable `spme_*` parameters found in the model.")
    return trainable


def _extract_spme_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith("spme_")}


def _load_spme_state_dict(model: torch.nn.Module, spme_sd: dict[str, Any]) -> None:
    full = model.state_dict()
    for k, v in spme_sd.items():
        if k in full:
            full[k].copy_(v)


def _compute_fusion_scales(
    *,
    det_score_raw: float | None,
    presence_prob: float | None,
    query_cos: float | None,
    base_alpha: float,
    base_alpha_obj: float,
    det_thr: float,
    qcos_thr: float,
    qcos_temp: float,
    qcos_gate: str,
    use_presence: bool,
) -> tuple[float | None, float | None]:
    if det_score_raw is None:
        return None, None
    det_score_f = float(det_score_raw)
    if det_score_f < det_thr:
        return None, None
    r = max(0.0, min(1.0, det_score_f))
    if use_presence and presence_prob is not None:
        r *= max(0.0, min(1.0, float(presence_prob)))
    if query_cos is not None and qcos_thr > 0.0:
        if qcos_gate in {"linear", "lin"}:
            denom = max(1e-6, 1.0 - qcos_thr)
            r *= max(0.0, min(1.0, (float(query_cos) - qcos_thr) / denom))
        else:
            r *= _sigmoid_f((float(query_cos) - qcos_thr) * float(qcos_temp))
    s_mem = base_alpha * r if base_alpha > 0.0 else 0.0
    s_obj = base_alpha_obj * r if base_alpha_obj > 0.0 else 0.0
    s_mem = float(max(0.0, min(1.0, s_mem)))
    s_obj = float(max(0.0, min(1.0, s_obj)))
    return (s_mem if s_mem > 0.0 else None), (s_obj if s_obj > 0.0 else None)


def _apply_spme_fusion_edit(
    model,
    current_out: dict[str, Any],
    *,
    query_vec: torch.Tensor | None,
    det_score_raw: float | None,
    presence_prob: float | None,
    query_cos: float | None,
    base_alpha: float,
    base_alpha_obj: float,
    det_thr: float,
    qcos_thr: float,
    qcos_temp: float,
    qcos_gate: str,
    use_presence: bool,
    mode: str,
) -> tuple[float | None, float | None]:
    if query_vec is None or not isinstance(query_vec, torch.Tensor) or query_vec.numel() == 0:
        return None, None

    scale_mem, scale_obj = _compute_fusion_scales(
        det_score_raw=det_score_raw,
        presence_prob=presence_prob,
        query_cos=query_cos,
        base_alpha=base_alpha,
        base_alpha_obj=base_alpha_obj,
        det_thr=det_thr,
        qcos_thr=qcos_thr,
        qcos_temp=qcos_temp,
        qcos_gate=qcos_gate,
        use_presence=use_presence,
    )

    if scale_mem is None and scale_obj is None:
        return None, None

    qv = query_vec
    if qv.ndim > 1:
        qv = qv.reshape(-1)
    qv = qv.to(device=next(model.parameters()).device, dtype=torch.float32)
    qv = qv / qv.norm().clamp_min(1e-6)
    qv = qv.view(1, -1)  # (1, D)

    if scale_mem is not None:
        maskmem_features = current_out.get("maskmem_features", None)
        if isinstance(maskmem_features, torch.Tensor) and maskmem_features.numel() > 0:
            mem_dim = int(maskmem_features.shape[1])
            if mode in {"film", "filmedit"}:
                film = model.spme_fusion_mem_film_mlp(qv)
                gamma, beta = film[:, :mem_dim], film[:, mem_dim:]
                gamma = torch.tanh(gamma).to(maskmem_features.dtype)
                beta = torch.tanh(beta).to(maskmem_features.dtype)
                s = float(scale_mem)
                current_out["maskmem_features"] = maskmem_features * (
                    1.0 + gamma.view(1, mem_dim, 1, 1) * s
                ) + beta.view(1, mem_dim, 1, 1) * s
            else:
                # residual injection fallback
                q_mem = qv[:, :mem_dim]
                if int(q_mem.shape[1]) < mem_dim:
                    q_mem = F.pad(q_mem, (0, mem_dim - int(q_mem.shape[1])))
                q_mem = q_mem.to(device=maskmem_features.device, dtype=maskmem_features.dtype)
                current_out["maskmem_features"] = maskmem_features + q_mem.view(1, mem_dim, 1, 1) * float(scale_mem)

    if scale_obj is not None:
        obj_ptr = current_out.get("obj_ptr", None)
        if isinstance(obj_ptr, torch.Tensor) and obj_ptr.numel() > 0:
            delta = model.spme_fusion_obj_ptr_mlp(qv).to(device=obj_ptr.device, dtype=obj_ptr.dtype)
            current_out["obj_ptr"] = obj_ptr + delta * float(scale_obj)

    return scale_mem, scale_obj


def _maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_path: Path,
    device: torch.device,
) -> tuple[int, int]:
    ckpt = torch.load(str(resume_path), map_location="cpu")
    spme_sd = ckpt.get("spme_state_dict", None)
    if isinstance(spme_sd, dict):
        _load_spme_state_dict(model, spme_sd)
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    step = int(ckpt.get("step", 0))
    epoch = int(ckpt.get("epoch", 0))
    # Move optimizer tensors to device (common resume footgun).
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)
    print(f"Resumed from {resume_path} at step={step} epoch={epoch}")
    return step, epoch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-root", required=True, type=str)
    ap.add_argument("--gt-root", required=True, type=str)
    ap.add_argument("--insertion-end-train-txt", required=True, type=str)
    ap.add_argument("--base-sam3-pt", required=True, type=str)
    ap.add_argument("--finetune-ckpt", required=True, type=str)
    ap.add_argument("--bpe-path", required=True, type=str)
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--prompt", type=str, default="ultrasound needle")
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--clip-len", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = unlimited (use --max-hours).")
    ap.add_argument("--max-hours", type=float, default=0.0, help="0 = ignore wallclock stop.")

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=0.1)

    ap.add_argument("--bce-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)

    ap.add_argument("--query-pool", type=str, default="topk_weighted", choices=["top1", "topk_weighted"])
    ap.add_argument("--query-topk", type=int, default=5)
    ap.add_argument("--anchor-det-thr", type=float, default=0.3)

    ap.add_argument("--fusion-mode", type=str, default="film", choices=["film", "resid"])
    ap.add_argument("--fusion-alpha", type=float, default=0.05)
    ap.add_argument("--fusion-alpha-obj", type=float, default=0.005)
    ap.add_argument("--fusion-det-thr", type=float, default=0.3)
    ap.add_argument("--fusion-qcos-thr", type=float, default=0.5)  # Default lowered to 0.5
    ap.add_argument("--fusion-qcos-temp", type=float, default=20.0)
    ap.add_argument("--fusion-qcos-gate", type=str, default="sigmoid", choices=["sigmoid", "linear"])
    ap.add_argument("--fusion-use-presence", type=int, default=1)

    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    _seed_everything(int(args.seed))

    # Enable SPME signal attachment (query vectors + qcos) in the video model.
    os.environ["SAM3_SPME_FUSION"] = "1"
    os.environ["SAM3_SPME_QUERY_POOL"] = str(args.query_pool)
    os.environ["SAM3_SPME_QUERY_TOPK"] = str(int(args.query_topk))
    os.environ["SAM3_SPME_ANCHOR_DET_THR"] = str(float(args.anchor_det_thr))

    # Build model and load ckpts.
    print("[DEBUG] Building SAM3 video model...", flush=True)
    model = build_sam3_video_model(
        checkpoint_path=str(Path(args.base_sam3_pt)),
        load_from_HF=False,
        bpe_path=args.bpe_path,
        device=str(device),
        compile=False,
    )
    print("[DEBUG] Model built. Applying detector finetune...", flush=True)
    _apply_detector_finetune(model, Path(args.finetune_ckpt))
    print("[DEBUG] Finetune applied. Moving model to device...", flush=True)
    model.to(device)
    model.eval()
    # Keep SPME modules in train mode (others frozen).
    if hasattr(model, "spme_fusion_mem_film_mlp"):
        model.spme_fusion_mem_film_mlp.train()
        print("[DEBUG] spme_fusion_mem_film_mlp found and set to train mode.", flush=True)
    else:
        print("[DEBUG] WARNING: spme_fusion_mem_film_mlp NOT FOUND!", flush=True)
    if hasattr(model, "spme_fusion_obj_ptr_mlp"):
        model.spme_fusion_obj_ptr_mlp.train()
        print("[DEBUG] spme_fusion_obj_ptr_mlp found and set to train mode.", flush=True)
    else:
        print("[DEBUG] WARNING: spme_fusion_obj_ptr_mlp NOT FOUND!", flush=True)

    print("[DEBUG] Freezing non-SPME params...", flush=True)
    trainable = _freeze_non_spme_params(model)
    print(f"Trainable params ({len(trainable)}):", flush=True)
    for n in trainable:
        print(f"  - {n}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    step = 0
    epoch = 0
    if args.resume:
        step, epoch = _maybe_resume(model, optimizer, Path(args.resume), device)

    frames_root = Path(args.frames_root)
    gt_root = Path(args.gt_root)
    print(f"[DEBUG] Loading insertion_end_map from: {args.insertion_end_train_txt}", flush=True)
    insertion_end_map = _load_insertion_end_map(Path(args.insertion_end_train_txt))
    sample_ids = sorted(list(insertion_end_map.keys()))
    print(f"[DEBUG] Found {len(sample_ids)} samples in insertion_end_train_txt", flush=True)
    if not sample_ids:
        raise RuntimeError(f"No valid entries found in: {args.insertion_end_train_txt}")

    low_res = int(getattr(model.tracker, "low_res_mask_size", 256))
    image_size = int(getattr(model, "image_size", 1008))
    print(f"[DEBUG] low_res={low_res}, image_size={image_size}", flush=True)

    start_time = time.time()
    rng = random.Random(int(args.seed) + 999)
    print("[DEBUG] Starting training loop...", flush=True)

    def _should_stop() -> bool:
        if args.max_steps and step >= int(args.max_steps):
            return True
        if args.max_hours and (time.time() - start_time) >= float(args.max_hours) * 3600.0:
            return True
        return False

    while True:
        if _should_stop():
            break

        clip = _choose_clip(rng, sample_ids, insertion_end_map, int(args.clip_len))
        if clip is None:
            continue

        frames_dir = frames_root / clip.sample_id
        gt_dir = gt_root / clip.sample_id
        if not frames_dir.exists() or not gt_dir.exists():
            continue

        frame_paths = [frames_dir / f"{i}.png" for i in range(clip.start, clip.start + clip.length)]
        gt_paths = [gt_dir / f"{i}.png" for i in range(clip.start, clip.start + clip.length)]
        if not all(p.exists() for p in frame_paths):
            continue
        if not all(p.exists() for p in gt_paths):
            continue

        # Load clip to GPU.
        images: list[torch.Tensor] = []
        masks_lr: list[torch.Tensor] = []
        orig_h, orig_w = None, None
        for fp, mp in zip(frame_paths, gt_paths):
            img_t, oh, ow = _load_frame_tensor(fp, image_size=image_size, device=None)
            if orig_h is None:
                orig_h, orig_w = oh, ow
            images.append(img_t)
            m_np = _read_mask_np(mp)
            masks_lr.append(_mask_to_low_res(m_np, low_res=low_res, device=device))

        images_t = torch.stack(images, dim=0)  # (T,3,S,S) float16, normalized

        # Build the video-model input batch (detector side).
        inference_state: dict[str, Any] = {}
        inference_state["image_size"] = image_size
        inference_state["num_frames"] = clip.length
        inference_state["orig_height"] = int(orig_h or image_size)
        inference_state["orig_width"] = int(orig_w or image_size)
        inference_state["constants"] = {}
        model._construct_initial_input_batch(inference_state, images_t)
        # Note: We do NOT call model.reset_state(inference_state) here because
        # _construct_initial_input_batch already initializes the input batch fields,
        # and reset_state expects tracker fields that we set up separately below.
        # Initialize the tracker-related fields that reset_state would expect.
        inference_state["tracker_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["action_history"] = []
        inference_state["is_image_only"] = False

        # Set semantic text prompt for all frames.
        inference_state["text_prompt"] = args.prompt
        inference_state["input_batch"].find_text_batch[0] = args.prompt
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = model.TEXT_ID_FOR_TEXT

        # Init geometric prompt from GT bbox at local frame 0 (relative coords).
        init_mask_np = _read_mask_np(gt_paths[0])
        xyxy = _bbox_xyxy_from_mask(init_mask_np)
        if xyxy is None:
            continue
        bbox_xywh = _xyxy_to_xywh_norm(*xyxy, w=int(orig_w), h=int(orig_h))
        bbox_cxcywh = _xywh_to_cxcywh(bbox_xywh)
        boxes_cxcywh = torch.tensor([bbox_cxcywh], dtype=torch.float32, device=device)
        box_labels = torch.tensor([1], dtype=torch.long, device=device)
        inference_state["per_frame_raw_box_input"][0] = (boxes_cxcywh, box_labels)
        _, _, geometric_prompt = model._get_visual_prompt(inference_state, 0, boxes_cxcywh, box_labels)
        inference_state["per_frame_geometric_prompt"][0] = geometric_prompt

        # Tracker state: reuse detector-produced cached features to avoid recomputing backbone.
        feature_cache: dict[str | int, Any] = {}
        tracker_state = model.tracker.init_state(
            video_height=int(orig_h),
            video_width=int(orig_w),
            num_frames=clip.length,
            cached_features=feature_cache,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        _ = model.tracker._obj_id_to_idx(tracker_state, obj_id=0)
        output_dict = tracker_state["output_dict"]

        # Box prompt as two labeled points (SAM-style).
        x0 = float(bbox_xywh[0] * image_size)
        y0 = float(bbox_xywh[1] * image_size)
        x1 = float((bbox_xywh[0] + bbox_xywh[2]) * image_size)
        y1 = float((bbox_xywh[1] + bbox_xywh[3]) * image_size)
        point_coords = torch.tensor([[[x0, y0], [x1, y1]]], dtype=torch.float32, device=device)
        point_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
        point_inputs0 = {"point_coords": point_coords, "point_labels": point_labels}

        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        n_loss_frames = 0
        scale_mem_vals: list[float] = []
        scale_obj_vals: list[float] = []

        optimizer.zero_grad(set_to_none=True)

        for local_t in range(clip.length):
            # 1) Run detector for semantic query + cache tracker backbone feats.
            with torch.no_grad():
                det_out = model.run_backbone_and_detection(
                    frame_idx=local_t,
                    num_frames=clip.length,
                    reverse=False,
                    input_batch=inference_state["input_batch"],
                    geometric_prompt=(
                        inference_state["constants"]["empty_geometric_prompt"]
                        if inference_state["per_frame_geometric_prompt"][local_t] is None
                        else inference_state["per_frame_geometric_prompt"][local_t]
                    ),
                    feature_cache=feature_cache,
                )

            # 2) Tracker step (differentiable).
            is_init = local_t == 0
            point_inputs = point_inputs0 if is_init else None
            current_out, pred_masks_gpu = model.tracker._run_single_frame_inference(
                inference_state=tracker_state,
                output_dict=output_dict,
                frame_idx=local_t,
                batch_size=1,
                is_init_cond_frame=is_init,
                point_inputs=point_inputs,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )

            # 3) Apply SPME-Fusion editing to memory (trainable).
            s_mem, s_obj = _apply_spme_fusion_edit(
                model,
                current_out,
                query_vec=det_out.get("spme_query_vec", None),
                det_score_raw=det_out.get("spme_det_score_raw", None),
                presence_prob=det_out.get("spme_presence_prob", None),
                query_cos=det_out.get("spme_query_cos", None),
                base_alpha=float(args.fusion_alpha),
                base_alpha_obj=float(args.fusion_alpha_obj),
                det_thr=float(args.fusion_det_thr),
                qcos_thr=float(args.fusion_qcos_thr),
                qcos_temp=float(args.fusion_qcos_temp),
                qcos_gate=str(args.fusion_qcos_gate),
                use_presence=bool(int(args.fusion_use_presence)),
                mode=str(args.fusion_mode),
            )
            if s_mem is not None:
                scale_mem_vals.append(float(s_mem))
            if s_obj is not None:
                scale_obj_vals.append(float(s_obj))

            # 4) Store for future memory reads.
            storage_key = "cond_frame_outputs" if is_init else "non_cond_frame_outputs"
            output_dict[storage_key][local_t] = current_out

            # 5) Loss (skip frame 0; fusion affects future frames).
            if local_t == 0:
                continue
            logits = pred_masks_gpu
            if logits.ndim == 4 and logits.shape[1] == 1:
                logits = logits[:, 0]
            gt_lr = masks_lr[local_t]
            bce = F.binary_cross_entropy_with_logits(logits.squeeze(0), gt_lr, reduction="mean")
            dice = _dice_loss_from_logits(logits.squeeze(0), gt_lr)
            loss = float(args.bce_weight) * bce + float(args.dice_weight) * dice
            total_loss = total_loss + loss
            total_bce = total_bce + bce
            total_dice = total_dice + dice
            n_loss_frames += 1

        if n_loss_frames == 0:
            continue
        total_loss = total_loss / float(n_loss_frames)
        total_bce = total_bce / float(n_loss_frames)
        total_dice = total_dice / float(n_loss_frames)

        total_loss.backward()
        if args.grad_clip and float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(args.grad_clip),
            )
        optimizer.step()

        # Logging.
        if step % int(args.log_every) == 0:
            log_row = {
                "step": int(step),
                "epoch": int(epoch),
                "sample_id": clip.sample_id,
                "clip_start": int(clip.start),
                "clip_len": int(clip.length),
                "loss": float(total_loss.detach().item()),
                "bce": float(total_bce.detach().item()),
                "dice": float(total_dice.detach().item()),
                "scale_mem_mean": float(np.mean(scale_mem_vals)) if scale_mem_vals else None,
                "scale_obj_mean": float(np.mean(scale_obj_vals)) if scale_obj_vals else None,
                "time_sec": float(time.time() - start_time),
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")
            print(json.dumps(log_row))

        # Checkpointing.
        if step % int(args.save_every) == 0 and step > 0:
            ckpt = {
                "step": int(step),
                "epoch": int(epoch),
                "spme_state_dict": _extract_spme_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            ckpt_path = ckpt_dir / f"spme_fusion_step{step:06d}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, ckpt_dir / "spme_fusion_latest.pt")
            print(f"Saved checkpoint: {ckpt_path}")

        step += 1

    # Final save.
    ckpt = {
        "step": int(step),
        "epoch": int(epoch),
        "spme_state_dict": _extract_spme_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_dir / "spme_fusion_latest.pt")
    print(f"Done. Saved latest: {ckpt_dir / 'spme_fusion_latest.pt'}")


if __name__ == "__main__":
    os.environ.setdefault("SAM3_DISABLE_TRITON", "1")
    main()

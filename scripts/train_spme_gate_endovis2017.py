#!/usr/bin/env python3
"""
Train learned SPME gate (channel-wise FiLM + decay) on EndoVis 2017 train split.

This mirrors `scripts/train_spme_gate.py` (OVIS) but uses EndoVis' per-frame labels
under:
  <data_root>/train/image/seq_<sid>_frameXXXX.(png|bmp)
  <data_root>/train/label/seq_<sid>_frameXXXX.(png|bmp)

Key idea (2-frame unroll):
  - t=0: init with GT mask (paper-aligned) or GT box (legacy)
  - t=gate_frame: run memory encoder, compute gate from signals, edit memory (diff)
  - t=supervise_frame (= gate_frame+1): predict mask, compute loss, backprop to gate

Trainable params (freeze everything else):
  - spme_gate_mlp
  - spme_fusion_mem_film_mlp
  - spme_fusion_obj_ptr_mlp

Outputs:
  - checkpoints/spme_gate_stepXXXXXX.pt (only spme_* params + optimizer state)
  - checkpoints/spme_gate_latest.pt
  - train_log.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
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

INSTRUMENT_CLASSES = {
    1: "Bipolar Forceps",
    2: "Prograsp Forceps",
    3: "Large Needle Driver",
    4: "Vessel Sealer",
    5: "Grasping Retractor",
    6: "Monopolar Curved Scissors",
    7: "Other",
}


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _parse_seq_and_frame(name: str) -> tuple[int, int] | None:
    # EndoVis2017 naming (train split): seq_<sid>_frameXXXX.bmp
    m = re.match(r"^seq_(\d+)_frame(\d+)\.(bmp|png|jpg|jpeg)$", name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _load_label_bmp(path: Path) -> np.ndarray:
    with Image.open(path) as m:
        arr = np.array(m)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


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
) -> tuple[torch.Tensor, int, int]:
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.width, img.height
    img = TF.resize(img, size=(image_size, image_size))
    t = TF.to_tensor(img)  # float32 (3,H,W)
    t = t.to(dtype=torch.float16)
    mean_t = torch.tensor(mean, dtype=torch.float16)[:, None, None]
    std_t = torch.tensor(std, dtype=torch.float16)[:, None, None]
    t = (t - mean_t) / std_t
    return t, orig_h, orig_w


def _mask_to_low_res(mask_np: np.ndarray, low_res: int, device: torch.device) -> torch.Tensor:
    m = torch.from_numpy(mask_np.astype(np.float32))[None, None]  # 1x1xH xW
    m = m.to(device=device, non_blocking=True)
    m_lr = F.interpolate(m, size=(low_res, low_res), mode="nearest")
    return (m_lr > 0.5).to(dtype=torch.float32).squeeze(0).squeeze(0)


def _dice_loss_from_logits(logits: torch.Tensor, targets01: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.reshape(-1)
    targets = targets01.reshape(-1)
    inter = (probs * targets).sum()
    denom = probs.sum() + targets.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def _mask_torch_from_np(mask_np01: np.ndarray, device: torch.device) -> torch.Tensor:
    m = torch.from_numpy(mask_np01.astype(np.uint8))
    if m.ndim != 2:
        raise ValueError(f"Expected mask (H,W), got {tuple(m.shape)}")
    return (m > 0).to(device=device, non_blocking=True)


def _freeze_gate_and_fusion_params(model: torch.nn.Module) -> list[str]:
    trainable: list[str] = []
    for name, p in model.named_parameters():
        if (
            name.startswith("spme_gate_mlp.")
            or name.startswith("spme_gate_fusion_mlp.")
            or name.startswith("spme_fusion_")
        ):
            p.requires_grad = True
            trainable.append(name)
        else:
            p.requires_grad = False
    if not trainable:
        raise RuntimeError("No trainable SPME gate/fusion parameters found in the model.")
    return trainable


def _extract_spme_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k.startswith("spme_")}


def _load_spme_state_dict(model: torch.nn.Module, spme_sd: dict[str, Any]) -> None:
    full = model.state_dict()
    for k, v in spme_sd.items():
        if k in full and torch.is_tensor(v) and torch.is_tensor(full[k]) and full[k].shape == v.shape:
            full[k].copy_(v)


def _maybe_load_spme_init(model: torch.nn.Module, path: Path) -> None:
    ckpt = torch.load(str(path), map_location="cpu")
    if isinstance(ckpt, dict) and "spme_state_dict" in ckpt and isinstance(ckpt["spme_state_dict"], dict):
        spme_sd = ckpt["spme_state_dict"]
    elif isinstance(ckpt, dict):
        spme_sd = {k: v for k, v in ckpt.items() if str(k).startswith("spme_")}
    else:
        raise TypeError(f"Unsupported SPME init checkpoint format: {path}")
    _load_spme_state_dict(model, spme_sd)
    print(f"Loaded SPME init params from {path} (keys={len(spme_sd)})", flush=True)


def _maybe_resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    resume_path: Path,
    device: torch.device,
) -> int:
    ckpt = torch.load(str(resume_path), map_location="cpu")
    spme_sd = ckpt.get("spme_state_dict", None) if isinstance(ckpt, dict) else None
    if isinstance(spme_sd, dict):
        _load_spme_state_dict(model, spme_sd)
    if isinstance(ckpt, dict) and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    step = int(ckpt.get("step", 0)) if isinstance(ckpt, dict) else 0
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)
    print(f"Resumed from {resume_path} at step={step}", flush=True)
    return step


def _apply_gate_and_fusion(
    model,
    current_out: dict[str, Any],
    prev_out: dict[str, Any] | None,
    *,
    mem_scale: torch.Tensor,
    mem_offset: torch.Tensor,
    gate_decay: torch.Tensor,
    gate_fusion: torch.Tensor | None = None,
    query_vec: torch.Tensor | None,
    fusion_mode: str,
    base_alpha: float,
    base_alpha_obj: float,
    use_decay: bool,
) -> None:
    # Memory blend/decay (differentiable).
    mem = current_out.get("maskmem_features", None)
    if isinstance(mem, torch.Tensor) and mem.numel() > 0:
        # Tracker outputs may be "InferenceTensor" depending on upstream contexts; those
        # cannot be saved for backward. Since the tracker is frozen, treat memories as
        # constants and clone to normal tensors for autograd on the small SPME heads.
        mem = mem.detach().clone()
        mem_dim = int(mem.shape[1])
        gate_scale = mem_scale.to(device=mem.device, dtype=mem.dtype).view(1, mem_dim, 1, 1)
        gate_offset = mem_offset.to(device=mem.device, dtype=mem.dtype).view(1, mem_dim, 1, 1)
        gate_d = gate_decay.to(device=mem.device, dtype=mem.dtype).view(1, 1, 1, 1)

        prev_mem = prev_out.get("maskmem_features", None) if isinstance(prev_out, dict) else None
        if isinstance(prev_mem, torch.Tensor) and prev_mem.shape == mem.shape:
            prev_mem = prev_mem.detach().to(device=mem.device, dtype=mem.dtype).clone()
            new_mem = mem * gate_scale + gate_offset
            prev_part = prev_mem * (1.0 - gate_scale)
            if use_decay:
                prev_part = prev_part * (1.0 - gate_d)
            mem = new_mem + prev_part
        else:
            mem = mem * gate_scale + gate_offset
            if use_decay:
                mem = mem * (1.0 - gate_d)
        current_out["maskmem_features"] = mem

    # Pointer blend/decay.
    ptr = current_out.get("obj_ptr", None)
    if isinstance(ptr, torch.Tensor) and ptr.numel() > 0:
        ptr = ptr.detach().clone()
        gate_scalar = mem_scale.mean(dim=-1, keepdim=True)
        gate_w = gate_scalar.to(device=ptr.device, dtype=ptr.dtype).view(1, 1)
        gate_d = gate_decay.to(device=ptr.device, dtype=ptr.dtype).view(1, 1)
        prev_ptr = prev_out.get("obj_ptr", None) if isinstance(prev_out, dict) else None
        if isinstance(prev_ptr, torch.Tensor) and prev_ptr.shape == ptr.shape:
            prev_ptr = prev_ptr.detach().to(device=ptr.device, dtype=ptr.dtype).clone()
            if use_decay:
                ptr = ptr * gate_w + prev_ptr * (1.0 - gate_w) * (1.0 - gate_d)
            else:
                ptr = ptr * gate_w + prev_ptr * (1.0 - gate_w)
        elif use_decay:
            ptr = ptr * (1.0 - gate_d)
        current_out["obj_ptr"] = ptr

    # Semantic fusion injection (scaled by a scalar derived from mean(mem_scale)).
    if query_vec is None or not isinstance(query_vec, torch.Tensor) or query_vec.numel() == 0:
        return
    if base_alpha <= 0.0 and base_alpha_obj <= 0.0:
        return

    qv = query_vec.detach()
    if qv.ndim > 1:
        qv = qv.reshape(-1)
    qv = qv.to(device=next(model.parameters()).device, dtype=torch.float32)
    qv = qv / qv.norm().clamp_min(1e-6)
    qv = qv.view(1, -1)

    gate_scalar = mem_scale.mean(dim=-1, keepdim=True)
    if isinstance(gate_fusion, torch.Tensor):
        g_f = gate_fusion.to(device=qv.device, dtype=torch.float32).view(1, 1)
    else:
        g_f = gate_scalar.to(device=qv.device, dtype=torch.float32).view(1, 1)

    if base_alpha > 0.0:
        mem = current_out.get("maskmem_features", None)
        if isinstance(mem, torch.Tensor) and mem.numel() > 0:
            mem = mem.detach().clone()
            mem_dim = int(mem.shape[1])
            scale_mem = torch.clamp(g_f * float(base_alpha), 0.0, 1.0).to(dtype=mem.dtype).view(1, 1, 1, 1)
            if fusion_mode in {"film", "filmedit"}:
                film = model.spme_fusion_mem_film_mlp(qv)
                gamma, beta = film[:, :mem_dim], film[:, mem_dim:]
                gamma = torch.tanh(gamma).to(mem.dtype)
                beta = torch.tanh(beta).to(mem.dtype)
                current_out["maskmem_features"] = mem * (1.0 + gamma.view(1, mem_dim, 1, 1) * scale_mem) + beta.view(
                    1, mem_dim, 1, 1
                ) * scale_mem
            else:
                proj = getattr(model.tracker, "obj_ptr_tpos_proj", None)
                if int(qv.shape[1]) == mem_dim:
                    q_mem = qv
                elif isinstance(proj, torch.nn.Module):
                    q_mem = proj(qv)
                else:
                    q_mem = qv[:, :mem_dim]
                    if int(q_mem.shape[1]) < mem_dim:
                        q_mem = F.pad(q_mem, (0, mem_dim - int(q_mem.shape[1])))
                q_mem = q_mem.to(device=mem.device, dtype=mem.dtype)
                current_out["maskmem_features"] = mem + q_mem.view(1, mem_dim, 1, 1) * scale_mem

    if base_alpha_obj > 0.0:
        ptr = current_out.get("obj_ptr", None)
        if isinstance(ptr, torch.Tensor) and ptr.numel() > 0:
            ptr = ptr.detach().clone()
            scale_obj = torch.clamp(g_f * float(base_alpha_obj), 0.0, 1.0).to(dtype=ptr.dtype).view(1, 1)
            delta = model.spme_fusion_obj_ptr_mlp(qv).to(device=ptr.device, dtype=ptr.dtype)
            current_out["obj_ptr"] = ptr + delta * scale_obj


def _gate_inputs(
    *,
    gate_inputs: str,
    query_cos: float,
    det_score: float,
    presence_prob: float,
    tracker_score: float,
    mask_area_delta: float,
    last_occluded_frames: float,
    device: torch.device,
) -> torch.Tensor:
    mode = str(gate_inputs).strip().lower()
    if mode in {"", "full", "full5", "full_5", "5"}:
        vals = [query_cos, det_score, tracker_score, mask_area_delta, last_occluded_frames]
    elif mode in {"det3", "det", "det_only", "clean", "detector_only"}:
        vals = [query_cos, det_score, presence_prob]
    elif mode in {"det4", "det_occ", "det+occ", "det_with_occ", "detector_occ"}:
        vals = [query_cos, det_score, presence_prob, last_occluded_frames]
    else:
        raise ValueError(f"Unknown gate_inputs={gate_inputs!r}. Supported: full, det3, det4.")

    return torch.tensor([vals], dtype=torch.float32, device=device)


def _gate_in_dim(gate_inputs: str) -> int:
    mode = str(gate_inputs).strip().lower()
    if mode in {"", "full", "full5", "full_5", "5"}:
        return 5
    if mode in {"det3", "det", "det_only", "clean", "detector_only"}:
        return 3
    if mode in {"det4", "det_occ", "det+occ", "det_with_occ", "detector_occ"}:
        return 4
    raise ValueError(f"Unknown gate_inputs={gate_inputs!r}. Supported: full, det3, det4.")


@dataclass(frozen=True)
class IndexedSequence:
    seq_id: int
    frames: list[Path]
    labels: list[Path]


def _index_endovis_train(data_root: Path, seq_ids: set[int] | None) -> list[IndexedSequence]:
    img_dir = data_root / "train" / "image"
    lbl_dir = data_root / "train" / "label"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"Missing train dirs: {img_dir} / {lbl_dir}")

    tmp: dict[int, list[tuple[int, Path, Path]]] = {}
    for img_path in sorted(img_dir.glob("*")):
        if not img_path.is_file():
            continue
        parsed = _parse_seq_and_frame(img_path.name)
        if parsed is None:
            continue
        sid, frame_idx = parsed
        if seq_ids is not None and sid not in seq_ids:
            continue
        lbl_path = lbl_dir / img_path.name
        if not lbl_path.exists():
            continue
        tmp.setdefault(sid, []).append((int(frame_idx), img_path, lbl_path))

    seqs: list[IndexedSequence] = []
    for sid, items in sorted(tmp.items(), key=lambda x: x[0]):
        items.sort(key=lambda x: x[0])
        seqs.append(
            IndexedSequence(
                seq_id=int(sid),
                frames=[p for _, p, _ in items],
                labels=[p for _, _, p in items],
            )
        )
    return seqs


@dataclass(frozen=True)
class ClipSpec:
    seq_id: int
    start: int
    length: int
    class_id: int


def _choose_clip(
    rng: random.Random,
    sequences: list[IndexedSequence],
    clip_len: int,
    min_init_area: int,
    allowed_classes: list[int] | None,
) -> ClipSpec | None:
    if not sequences:
        return None
    seq = rng.choice(sequences)
    if len(seq.frames) < clip_len:
        return None
    start = rng.randint(0, len(seq.frames) - clip_len)
    lbl0 = _load_label_bmp(seq.labels[start])
    present = [int(c) for c in np.unique(lbl0) if int(c) in INSTRUMENT_CLASSES]
    if allowed_classes is not None:
        present = [c for c in present if c in set(allowed_classes)]
    if not present:
        return None
    class_id = int(rng.choice(present))
    mask0 = (lbl0 == class_id).astype(np.uint8)
    if int(mask0.sum()) < int(min_init_area):
        return None
    if _bbox_xyxy_from_mask(mask0) is None:
        return None
    return ClipSpec(seq_id=int(seq.seq_id), start=int(start), length=int(clip_len), class_id=int(class_id))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=str, help="Path to endovis2017 root (contains train/)")
    ap.add_argument("--train-seqs", type=int, nargs="*", default=[1, 2, 3, 4, 5, 6, 7], help="Sequence IDs to use.")
    ap.add_argument("--allowed-classes", type=int, nargs="*", default=None, help="Optional instrument class IDs to train.")
    ap.add_argument("--min-init-area", type=int, default=50, help="Min init mask area (pixels) to avoid tiny masks.")
    ap.add_argument(
        "--prompt-mode",
        type=str,
        default="class",
        choices=["class", "generic", "visual"],
        help="How to set the SAM3 text prompt. "
        "'class' uses the instrument name (often brittle). "
        "'generic' uses --prompt for all clips. "
        "'visual' disables language and relies on visual init (GT mask/box).",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="surgical instrument",
        help="Used only when --prompt-mode=generic.",
    )

    ap.add_argument("--base-sam3-pt", required=True, type=str)
    ap.add_argument("--bpe-path", required=True, type=str)
    ap.add_argument("--overlay-ckpt", type=str, default=None, help="Optional finetuned EndoVis detector ckpt.")
    ap.add_argument("--spme-init-ckpt", type=str, default=None, help="Optional init SPME ckpt (loads spme_* only).")
    ap.add_argument("--out-dir", required=True, type=str)

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--image-size",
        type=int,
        default=1008,
        help="Input image size for SAM3 detector/tracker. Must match model's internal size "
        "(default 1008 for SAM3). If set differently, we will override to the model value "
        "to avoid RoPE shape mismatches.",
    )

    ap.add_argument("--clip-len", type=int, default=3)
    ap.add_argument("--gate-frame", type=int, default=1)
    ap.add_argument("--supervise-frame", type=int, default=2)
    ap.add_argument(
        "--init-prompt",
        type=str,
        default="mask",
        choices=["mask", "box"],
        help="Initialization prompt at local frame 0. "
        "'mask' aligns with the EndoVis 'first visible mask' protocol; "
        "'box' uses a GT-derived box prompt (legacy).",
    )

    ap.add_argument("--max-steps", type=int, default=0, help="0 = unlimited (use --max-hours).")
    ap.add_argument("--max-hours", type=float, default=0.0, help="0 = ignore wallclock stop.")

    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=0.1)
    ap.add_argument("--bce-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--absent-weight", type=float, default=0.0, help="Penalty on predicted area when GT is empty.")

    ap.add_argument("--query-pool", type=str, default="top1", choices=["top1", "topk_weighted"])
    ap.add_argument("--query-topk", type=int, default=5)
    ap.add_argument("--anchor-det-thr", type=float, default=0.0)

    ap.add_argument(
        "--pointer-mode",
        type=str,
        default="hybrid",
        choices=["top1", "per_object", "hybrid"],
        help="How to pick semantic pointer signals for the tracked obj_id=0.",
    )
    ap.add_argument("--match-iou-thr", type=float, default=0.1, help="IoU thr to match det mask -> tracked obj.")
    ap.add_argument("--match-topk", type=int, default=20, help="Top-k dets to try for per-object matching (0=all).")

    ap.add_argument("--score-thr-detection", type=float, default=0.3, help="Override model.score_threshold_detection.")
    ap.add_argument("--occ-norm", type=float, default=10.0, help="Normalization for occlusion length -> [0,1].")
    ap.add_argument("--use-decay", type=int, default=1, choices=[0, 1])
    ap.add_argument(
        "--gate-inputs",
        type=str,
        default="full",
        choices=["full", "det3", "det4"],
        help="Gate input feature set (must match how the gate MLP was constructed).",
    )
    ap.add_argument(
        "--fusion-head",
        type=int,
        default=0,
        choices=[0, 1],
        help="Decouple fusion strength with a separate learned head (recommended for ablations).",
    )
    ap.add_argument(
        "--det-present-thr",
        type=float,
        default=0.3,
        help="Only for gate-inputs=det4: det_score threshold to count a frame as present.",
    )

    ap.add_argument("--fusion-mode", type=str, default="film", choices=["resid", "film"])
    ap.add_argument("--fusion-alpha", type=float, default=0.05)
    ap.add_argument("--fusion-alpha-obj", type=float, default=0.005)

    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--grad-check", action="store_true", help="Run a tiny synthetic grad sanity check and exit.")

    args = ap.parse_args()
    args.init_prompt = str(args.init_prompt).strip().lower()

    if int(args.clip_len) < 2:
        raise ValueError("--clip-len must be >= 2.")
    if int(args.gate_frame) < 0 or int(args.gate_frame) >= int(args.clip_len):
        raise ValueError("--gate-frame must be within [0, clip_len-1].")
    if int(args.supervise_frame) < 0 or int(args.supervise_frame) >= int(args.clip_len):
        raise ValueError("--supervise-frame must be within [0, clip_len-1].")
    if int(args.supervise_frame) <= int(args.gate_frame):
        raise ValueError("Expected supervise frame to be after gate frame (t+1).")

    _seed_everything(int(args.seed))

    # Enable SPME signal attachment (query vectors + qcos) in the detector output.
    os.environ["SAM3_SPME_FUSION"] = "1"
    os.environ["SAM3_SPME_QUERY_POOL"] = str(args.query_pool)
    os.environ["SAM3_SPME_QUERY_TOPK"] = str(int(args.query_topk))
    os.environ["SAM3_SPME_ANCHOR_DET_THR"] = str(float(args.anchor_det_thr))
    os.environ["SAM3_SPME_PER_OBJECT"] = "1" if args.pointer_mode in {"per_object", "hybrid"} else "0"
    os.environ["SAM3_SPME_KEEP_QUERIES"] = "1" if args.pointer_mode in {"per_object", "hybrid"} else "0"

    # Enable learned gate module creation.
    os.environ["SAM3_SPME_LEARNED_GATE"] = "1"
    os.environ["SAM3_SPME_LEARNED_GATE_USE_DECAY"] = "1" if int(args.use_decay) else "0"
    os.environ["SAM3_SPME_LEARNED_GATE_OCCLUDED_NORM"] = str(float(args.occ_norm))
    os.environ["SAM3_SPME_LEARNED_GATE_INPUTS"] = str(args.gate_inputs)
    os.environ["SAM3_SPME_LEARNED_GATE_DET_PRESENT_THR"] = str(float(args.det_present_thr))
    os.environ["SAM3_SPME_LEARNED_GATE_FUSION_HEAD"] = "1" if int(args.fusion_head) else "0"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "time": float(time.time()), "args": vars(args)}) + "\n")

    device = torch.device("cuda")
    model = build_sam3_video_model(
        checkpoint_path=str(args.base_sam3_pt),
        load_from_HF=False,
        bpe_path=str(args.bpe_path),
        device="cuda",
    )
    model.eval()

    if args.overlay_ckpt:
        sd = torch.load(str(args.overlay_ckpt), map_location="cpu", weights_only=True)
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        missing, unexpected = model.detector.load_state_dict(sd, strict=False)
        print(f"Loaded overlay ckpt into detector: missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    if args.spme_init_ckpt:
        _maybe_load_spme_init(model, Path(args.spme_init_ckpt))

    if hasattr(model, "score_threshold_detection"):
        model.score_threshold_detection = float(args.score_thr_detection)

    if getattr(model, "spme_gate_mlp", None) is None:
        raise RuntimeError("spme_gate_mlp was not created. Is SAM3_SPME_LEARNED_GATE=1 set early enough?")

    trainable = _freeze_gate_and_fusion_params(model)
    print(f"Trainable params ({len(trainable)}):", flush=True)
    for n in trainable:
        print(f"  - {n}", flush=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    if args.grad_check:
        gate = model.spme_gate_mlp
        x = torch.randn(4, _gate_in_dim(str(args.gate_inputs)), device=device)
        mem_scale, mem_offset, d = gate(x)
        mem_dim = int(mem_scale.shape[1])
        old_mem = torch.randn(4, mem_dim, 8, 8, device=device)
        new_mem = torch.randn(4, mem_dim, 8, 8, device=device)
        s = mem_scale.view(-1, mem_dim, 1, 1)
        o = mem_offset.view(-1, mem_dim, 1, 1)
        dec = d.view(-1, 1, 1, 1)
        mem_t = new_mem * s + o + old_mem * (1.0 - s) * (1.0 - dec)
        gate_scalar = mem_scale.mean(dim=-1)
        pred_tp1 = mem_t.mean(dim=(1, 2, 3)) + gate_scalar * 0.1
        loss = (pred_tp1**2).mean()
        loss.backward()
        gsum = 0.0
        for p in gate.parameters():
            if p.grad is not None:
                gsum += float(p.grad.detach().abs().sum().item())
        print(f"[grad_check] loss={float(loss.item()):.6f} gate_grad_abs_sum={gsum:.6f}", flush=True)
        return

    start_time = time.time()
    data_root = Path(args.data_root)
    seqs = _index_endovis_train(data_root, set(args.train_seqs) if args.train_seqs else None)
    if not seqs:
        raise RuntimeError("No EndoVis train sequences found after filtering.")
    seq_map = {int(s.seq_id): s for s in seqs}
    rng = random.Random(int(args.seed) + 999)

    # Tracker low-res mask logits resolution (must match `pred_masks` / `pred_masks_gpu`).
    low_res = int(getattr(model.tracker, "low_res_mask_size", 256))
    model_image_size = int(getattr(model, "image_size", int(args.image_size)))
    if int(args.image_size) != model_image_size:
        print(
            f"[warn] Overriding --image-size={int(args.image_size)} to model.image_size={model_image_size} "
            "to avoid RoPE/ViT shape mismatches.",
            flush=True,
        )
    image_size = model_image_size

    def _should_stop(step: int) -> bool:
        if args.max_steps and int(step) >= int(args.max_steps):
            return True
        if args.max_hours and (time.time() - start_time) >= float(args.max_hours) * 3600.0:
            return True
        return False

    step = 0
    if args.resume:
        step = _maybe_resume(model, optimizer, Path(args.resume), device=device)

    while True:
        if _should_stop(step):
            break

        clip = _choose_clip(
            rng,
            sequences=seqs,
            clip_len=int(args.clip_len),
            min_init_area=int(args.min_init_area),
            allowed_classes=[int(x) for x in args.allowed_classes] if args.allowed_classes else None,
        )
        if clip is None:
            continue

        seq = seq_map[int(clip.seq_id)]
        frame_paths = seq.frames[int(clip.start) : int(clip.start) + int(clip.length)]
        label_paths = seq.labels[int(clip.start) : int(clip.start) + int(clip.length)]

        images: list[torch.Tensor] = []
        masks_lr: list[torch.Tensor] = []
        orig_h, orig_w = None, None
        for fp, lp in zip(frame_paths, label_paths):
            img_t, oh, ow = _load_frame_tensor(fp, image_size=image_size)
            if orig_h is None:
                orig_h, orig_w = oh, ow
            images.append(img_t)
            lbl = _load_label_bmp(lp)
            m_np = (lbl == int(clip.class_id)).astype(np.uint8)
            masks_lr.append(_mask_to_low_res(m_np, low_res=low_res, device=device))
        if orig_h is None or orig_w is None:
            continue

        images_t = torch.stack(images, dim=0)

        # Build detector input batch.
        inference_state: dict[str, Any] = {}
        inference_state["image_size"] = image_size
        inference_state["num_frames"] = int(clip.length)
        inference_state["orig_height"] = int(orig_h)
        inference_state["orig_width"] = int(orig_w)
        inference_state["constants"] = {}
        model._construct_initial_input_batch(inference_state, images_t)
        inference_state["tracker_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["action_history"] = []
        inference_state["is_image_only"] = False

        prompt_mode = str(args.prompt_mode).strip().lower()
        if prompt_mode == "class":
            prompt = str(INSTRUMENT_CLASSES.get(int(clip.class_id), "surgical instrument")).lower()
            inference_state["text_prompt"] = prompt
            inference_state["input_batch"].find_text_batch[0] = prompt
            text_id = model.TEXT_ID_FOR_TEXT
        elif prompt_mode == "generic":
            prompt = str(args.prompt).strip()
            inference_state["text_prompt"] = prompt
            inference_state["input_batch"].find_text_batch[0] = prompt
            text_id = model.TEXT_ID_FOR_TEXT
        else:
            # Match the semantics of `Sam3VideoInference.add_prompt(text_str="visual")`.
            prompt = "visual"
            inference_state["text_prompt"] = None
            inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
            text_id = model.TEXT_ID_FOR_VISUAL
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id

        # Init at local frame 0.
        lbl0 = _load_label_bmp(label_paths[0])
        init_mask_np = (lbl0 == int(clip.class_id)).astype(np.uint8)
        if _bbox_xyxy_from_mask(init_mask_np) is None:
            continue

        point_inputs0 = None
        if args.init_prompt == "box":
            # Legacy init: GT box becomes a visual prompt (affects detector) and a SAM-style point prompt (affects tracker).
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

            x0 = float(bbox_xywh[0] * image_size)
            y0 = float(bbox_xywh[1] * image_size)
            x1 = float((bbox_xywh[0] + bbox_xywh[2]) * image_size)
            y1 = float((bbox_xywh[1] + bbox_xywh[3]) * image_size)
            point_coords = torch.tensor([[[x0, y0], [x1, y1]]], dtype=torch.float32, device=device)
            point_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
            point_inputs0 = {"point_coords": point_coords, "point_labels": point_labels}
        else:
            # Paper-aligned init: GT mask only; do NOT pass GT box to the detector.
            inference_state["per_frame_raw_box_input"][0] = None
            inference_state["per_frame_geometric_prompt"][0] = None

        feature_cache: dict[str | int, Any] = {}
        tracker_state = model.tracker.init_state(
            video_height=int(orig_h),
            video_width=int(orig_w),
            num_frames=int(args.clip_len),
            cached_features=feature_cache,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        _ = model.tracker._obj_id_to_idx(tracker_state, obj_id=0)
        output_dict = tracker_state["output_dict"]

        optimizer.zero_grad(set_to_none=True)

        prev_area = None
        missing_count = 0
        prev_out_for_gate = None
        qv_gate = None
        det_score_gate = 0.0
        qcos_gate = 1.0
        tracker_score_gate = 1.0

        # --- Forward unroll ---
        for local_t in range(int(args.clip_len)):
            with torch.no_grad():
                det_out = model.run_backbone_and_detection(
                    frame_idx=local_t,
                    num_frames=int(args.clip_len),
                    reverse=False,
                    input_batch=inference_state["input_batch"],
                    geometric_prompt=(
                        inference_state["constants"]["empty_geometric_prompt"]
                        if inference_state["per_frame_geometric_prompt"][local_t] is None
                        else inference_state["per_frame_geometric_prompt"][local_t]
                    ),
                    feature_cache=feature_cache,
                )

            is_init = local_t == 0
            if is_init and args.init_prompt == "mask":
                # Initialize tracker from GT mask (semi-supervised), matching eval's INIT_PROMPT=mask.
                init_mask_t = _mask_torch_from_np(init_mask_np, device=device)
                model.tracker.add_new_mask(
                    inference_state=tracker_state,
                    frame_idx=0,
                    obj_id=0,
                    mask=init_mask_t,
                    add_mask_to_memory=False,
                )
                model.tracker.propagate_in_video_preflight(tracker_state, run_mem_encoder=True)
                current_out = output_dict["cond_frame_outputs"][0]
                pred_masks_gpu = current_out["pred_masks"].to(device=device, non_blocking=True)
            else:
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
                    run_mem_encoder=(local_t <= int(args.gate_frame)),
                )

            storage_key = "cond_frame_outputs" if is_init else "non_cond_frame_outputs"

            if local_t == int(args.gate_frame):
                prev_frame_idx = int(local_t) - 1
                prev_out_for_gate = None
                if prev_frame_idx >= 0:
                    prev_out_for_gate = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                    if prev_out_for_gate is None:
                        prev_out_for_gate = output_dict["cond_frame_outputs"].get(prev_frame_idx, None)

                # Per-object pointer selection (obj_id=0).
                qv_sel = None
                det_score_sel = None
                qcos_sel = None
                if str(args.pointer_mode) in {"per_object", "hybrid"}:
                    try:
                        det_masks = det_out.get("mask", None)
                        if isinstance(det_masks, torch.Tensor) and det_masks.numel() > 0:
                            trk_logits = pred_masks_gpu
                            if isinstance(trk_logits, torch.Tensor):
                                if trk_logits.ndim == 4:
                                    trk_logits = trk_logits[0, 0]
                                elif trk_logits.ndim == 3:
                                    trk_logits = trk_logits[0]
                            trk_bin = (trk_logits.detach() > 0).to(dtype=torch.bool)

                            det_masks_res = det_masks
                            if det_masks_res.shape[-2:] != trk_bin.shape[-2:]:
                                det_masks_res = F.interpolate(
                                    det_masks_res.detach().float().unsqueeze(1),
                                    size=trk_bin.shape[-2:],
                                    mode="bilinear",
                                    align_corners=False,
                                ).squeeze(1)
                            det_bin = (det_masks_res.detach() > 0).to(dtype=torch.bool)
                            inter = (det_bin & trk_bin).sum(dim=(-1, -2)).float()
                            union = (det_bin | trk_bin).sum(dim=(-1, -2)).float().clamp_min(1e-6)
                            ious = inter / union
                            match_thr = float(args.match_iou_thr)
                            cand = (ious >= match_thr).nonzero(as_tuple=False).reshape(-1)
                            match_topk = int(args.match_topk)
                            if match_topk > 0 and int(cand.numel()) > match_topk:
                                cand_ious = ious[cand]
                                _, topk_idx = torch.topk(cand_ious, k=match_topk, largest=True)
                                cand = cand[topk_idx]
                            det_to_matched = (
                                {int(i): np.asarray([0], dtype=np.int64) for i in cand.tolist()} if cand.numel() else {}
                            )
                            if det_to_matched:
                                per_obj_ctx = model._build_spme_per_object_context(
                                    det_out=det_out,
                                    det_to_matched_trk_obj_ids=det_to_matched,
                                    new_det_fa_inds=np.asarray([], dtype=np.int64),
                                    new_det_obj_ids=np.asarray([], dtype=np.int64),
                                    feature_cache=feature_cache,
                                )
                                if isinstance(per_obj_ctx, dict):
                                    qv_sel = per_obj_ctx.get("spme_query_vec_by_obj", {}).get(0, None)
                                    det_score_sel = per_obj_ctx.get("spme_det_score_raw_by_obj", {}).get(0, None)
                                    qcos_sel = per_obj_ctx.get("spme_query_cos_by_obj", {}).get(0, None)
                    except Exception:
                        qv_sel = None
                if qv_sel is None and str(args.pointer_mode) == "hybrid":
                    qv_sel = det_out.get("spme_query_vec", None)
                    det_score_sel = det_out.get("spme_det_score_raw", None)
                    qcos_sel = det_out.get("spme_query_cos", None)
                if str(args.pointer_mode) == "top1":
                    qv_sel = det_out.get("spme_query_vec", None)
                    det_score_sel = det_out.get("spme_det_score_raw", None)
                    qcos_sel = det_out.get("spme_query_cos", None)

                qv_gate = qv_sel
                det_score_gate = float(det_score_sel) if det_score_sel is not None else 0.0
                qcos_gate = float(qcos_sel) if qcos_sel is not None else 1.0
                presence_gate = float(det_out.get("spme_presence_prob", det_score_gate)) if isinstance(det_out, dict) else det_score_gate

                # tracker_score: use object_score_logits as a simple quality/confidence proxy.
                obj_score_logits = current_out.get("object_score_logits", None)
                if isinstance(obj_score_logits, torch.Tensor) and obj_score_logits.numel() > 0:
                    tracker_score_gate = float(obj_score_logits.detach().float().sigmoid().mean().item())
                else:
                    tracker_score_gate = 1.0

                # mask_area_delta + occlusion count.
                logits = pred_masks_gpu
                if logits.ndim == 4 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                area = float(torch.sigmoid(logits.detach().float()).mean().item())
                prev_area_val = None
                if isinstance(prev_out_for_gate, dict):
                    prev_logits = prev_out_for_gate.get("pred_masks", None)
                    if isinstance(prev_logits, torch.Tensor) and prev_logits.numel() > 0:
                        if prev_logits.ndim == 4 and prev_logits.shape[1] == 1:
                            prev_logits = prev_logits[:, 0]
                        prev_area_val = float(torch.sigmoid(prev_logits.detach().float()).mean().item())
                if prev_area_val is None:
                    prev_area_val = float(prev_area) if prev_area is not None else area
                delta = abs(area - float(prev_area_val)) / (abs(float(prev_area_val)) + 1e-3)
                delta = delta / (delta + 1.0)
                prev_area = area
                gate_inputs_mode = str(args.gate_inputs).strip().lower()
                if gate_inputs_mode in {"det4"}:
                    is_present = float(det_score_gate) >= float(args.det_present_thr)
                else:
                    is_present = bool((logits.detach() > 0).any().item())
                missing_count = 0 if bool(is_present) else int(missing_count + 1)
                last_occ = float(min(1.0, missing_count / max(1.0, float(args.occ_norm))))

                x_gate = _gate_inputs(
                    gate_inputs=str(args.gate_inputs),
                    query_cos=float(max(-1.0, min(1.0, qcos_gate))),
                    det_score=float(max(0.0, min(1.0, det_score_gate))),
                    presence_prob=float(max(0.0, min(1.0, presence_gate))),
                    tracker_score=float(max(0.0, min(1.0, float(tracker_score_gate)))),
                    mask_area_delta=float(max(0.0, min(1.0, float(delta)))),
                    last_occluded_frames=float(max(0.0, min(1.0, float(last_occ)))),
                    device=device,
                )
                mem_scale, mem_offset, gate_decay = model.spme_gate_mlp(x_gate)
                if not int(args.use_decay):
                    gate_decay = gate_decay * 0.0
                gate_fusion = None
                fusion_head = getattr(model, "spme_gate_fusion_mlp", None)
                if isinstance(fusion_head, torch.nn.Module):
                    gate_fusion = torch.sigmoid(fusion_head(x_gate))

                _apply_gate_and_fusion(
                    model,
                    current_out,
                    prev_out_for_gate,
                    mem_scale=mem_scale,
                    mem_offset=mem_offset,
                    gate_decay=gate_decay,
                    gate_fusion=gate_fusion,
                    query_vec=qv_gate,
                    fusion_mode=str(args.fusion_mode),
                    base_alpha=float(args.fusion_alpha),
                    base_alpha_obj=float(args.fusion_alpha_obj),
                    use_decay=bool(int(args.use_decay)),
                )

            output_dict[storage_key][local_t] = current_out

            if local_t == int(args.supervise_frame):
                logits = pred_masks_gpu
                if logits.ndim == 4 and logits.shape[1] == 1:
                    logits = logits[:, 0]
                gt_lr = masks_lr[local_t]
                bce = F.binary_cross_entropy_with_logits(logits.squeeze(0), gt_lr, reduction="mean")
                dice = _dice_loss_from_logits(logits.squeeze(0), gt_lr)
                loss = float(args.bce_weight) * bce + float(args.dice_weight) * dice
                if float(gt_lr.detach().sum().item()) <= 0.0 and float(args.absent_weight) > 0.0:
                    ghost = torch.sigmoid(logits).mean()
                    loss = loss + float(args.absent_weight) * ghost
                loss.backward()

        if args.grad_clip and float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=float(args.grad_clip)
            )
        optimizer.step()

        gate_grad = 0.0
        for p in model.spme_gate_mlp.parameters():
            if p.grad is not None:
                gate_grad += float(p.grad.detach().abs().sum().item())

        if step % int(args.log_every) == 0:
            row = {
                "event": "step",
                "time": float(time.time()),
                "step": int(step),
                "seq_id": int(clip.seq_id),
                "class_id": int(clip.class_id),
                "start": int(clip.start),
                "clip_len": int(clip.length),
                "prompt": prompt,
                "prompt_mode": str(args.prompt_mode),
                "prompt_generic": str(args.prompt),
                "gate_grad_abs_sum": float(gate_grad),
                "gate_inputs": str(args.gate_inputs),
                "gate_det_score": float(det_score_gate),
                "gate_presence_prob": float(presence_gate),
                "gate_qcos": float(qcos_gate),
                "gate_tracker_score": float(tracker_score_gate),
                "gate_area_delta": float(delta),
                "gate_last_occluded_norm": float(last_occ),
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if step % int(args.save_every) == 0 and step > 0:
            ckpt = {
                "step": int(step),
                "spme_state_dict": _extract_spme_state_dict(model),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }
            ckpt_path = ckpt_dir / f"spme_gate_step{step:06d}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, ckpt_dir / "spme_gate_latest.pt")
            print(f"[ckpt] saved {ckpt_path}", flush=True)

        step += 1

    # Final save
    ckpt = {
        "step": int(step),
        "spme_state_dict": _extract_spme_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_dir / "spme_gate_latest.pt")
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "end", "time": float(time.time()), "step": int(step)}) + "\n")
    print(f"Done. Latest checkpoint: {ckpt_dir / 'spme_gate_latest.pt'}", flush=True)


if __name__ == "__main__":
    main()

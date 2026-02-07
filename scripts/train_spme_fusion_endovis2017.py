#!/usr/bin/env python3
"""
Train SPME-Fusion (FiLM memory editing + pointer residual) on EndoVis 2017 train split.

Why this script exists:
- The EndoVis detector/seg head is typically first adapted with the Hydra image-level trainer
  (see `sam3/train/configs/endovis2017/endovis2017_tool_seg_boxprompt_train.yaml`).
- After that, we train ONLY the lightweight `spme_*` modules in the SAM3 *video* model, using
  short clips and per-frame GT masks, to improve temporal robustness (ghost/drift/occlusion recovery).

Training recipe (mirrors `train_spme_fusion_ovis.py` / `train_spme_fusion_ultrasound.py`):
- Build SAM3 video model from `sam3.pt`
- (Optional) Overlay a finetuned detector checkpoint (EndoVis domain adaptation)
- Freeze all params except `spme_*`
- Sample random clips from EndoVis train sequences (seq_1..seq_8 in `train/image|label`)
- Track one instrument class per clip (binary mask), initialized by GT mask (paper-aligned) or GT bbox (legacy) at clip t=0
- Optimize BCE+Dice on frames t>=1 in the clip

Outputs:
- checkpoints/spme_fusion_stepXXXXXX.pt  (only spme_* params + optimizer state)
- checkpoints/spme_fusion_latest.pt
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


def _apply_detector_overlay(video_model, overlay_ckpt: Path) -> None:
    """
    Overlay a finetuned detector checkpoint onto the SAM3 video model.

    We load into `video_model.detector` directly (so we do NOT need the "detector." prefix mapping).
    """
    sd = _load_trainer_checkpoint_state_dict(overlay_ckpt)
    target = getattr(video_model, "detector", None) or video_model
    missing, unexpected = target.load_state_dict(sd, strict=False)
    print(f"Loaded overlay into detector: missing={len(missing)} unexpected={len(unexpected)}")


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
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)
    print(f"Resumed from {resume_path} at step={step} epoch={epoch}")
    return step, epoch


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
            # The tracker may produce "InferenceTensor" outputs depending on upstream contexts.
            # Those cannot participate in autograd. Since the tracker is frozen anyway, treat
            # these as constants and clone into a normal tensor for training the small `spme_*` heads.
            maskmem_features = maskmem_features.detach().clone()
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
                q_mem = qv[:, :mem_dim]
                if int(q_mem.shape[1]) < mem_dim:
                    q_mem = F.pad(q_mem, (0, mem_dim - int(q_mem.shape[1])))
                q_mem = q_mem.to(device=maskmem_features.device, dtype=maskmem_features.dtype)
                current_out["maskmem_features"] = maskmem_features + q_mem.view(1, mem_dim, 1, 1) * float(
                    scale_mem
                )

    if scale_obj is not None:
        obj_ptr = current_out.get("obj_ptr", None)
        if isinstance(obj_ptr, torch.Tensor) and obj_ptr.numel() > 0:
            obj_ptr = obj_ptr.detach().clone()
            delta = model.spme_fusion_obj_ptr_mlp(qv).to(device=obj_ptr.device, dtype=obj_ptr.dtype)
            current_out["obj_ptr"] = obj_ptr + delta * float(scale_obj)

    return scale_mem, scale_obj


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
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--clip-len", type=int, default=4)
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

    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=0.1)
    ap.add_argument("--bce-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)

    ap.add_argument("--query-pool", type=str, default="top1", choices=["top1", "topk_weighted"])
    ap.add_argument("--query-topk", type=int, default=5)
    ap.add_argument("--anchor-det-thr", type=float, default=0.0)
    ap.add_argument(
        "--pointer-mode",
        type=str,
        default="hybrid",
        choices=["top1", "per_object", "hybrid"],
        help="How to pick semantic pointer signals for the tracked obj_id=0 during fusion training. "
        "'per_object' matches detector masks to the tracker mask and uses the matched query embedding. "
        "'top1' uses global query_top1. 'hybrid' falls back to top1 when matching fails.",
    )
    ap.add_argument("--match-iou-thr", type=float, default=0.1, help="IoU thr to match det mask -> tracked obj.")
    ap.add_argument("--match-topk", type=int, default=20, help="Top-k dets to try for per-object matching (0=all).")
    ap.add_argument("--score-thr-detection", type=float, default=0.3, help="Override model.score_threshold_detection.")

    ap.add_argument("--fusion-mode", type=str, default="film", choices=["film", "resid"])
    ap.add_argument("--fusion-alpha", type=float, default=0.05)
    ap.add_argument("--fusion-alpha-obj", type=float, default=0.005)
    ap.add_argument("--fusion-det-thr", type=float, default=0.3)
    ap.add_argument("--fusion-qcos-thr", type=float, default=0.5)
    ap.add_argument("--fusion-qcos-temp", type=float, default=20.0)
    ap.add_argument("--fusion-qcos-gate", type=str, default="sigmoid", choices=["sigmoid", "linear"])
    ap.add_argument("--fusion-use-presence", type=int, default=1)

    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--resume", type=str, default=None)
    args = ap.parse_args()
    args.init_prompt = str(args.init_prompt).strip().lower()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"

    _seed_everything(int(args.seed))

    # Enable SPME signal attachment (query vectors + qcos) in the video model.
    # NOTE: Keep built-in fusion alpha at default 0.0; we apply fusion edits manually for training stability.
    os.environ["SAM3_SPME_FUSION"] = "1"
    os.environ["SAM3_SPME_QUERY_POOL"] = str(args.query_pool)
    os.environ["SAM3_SPME_QUERY_TOPK"] = str(int(args.query_topk))
    os.environ["SAM3_SPME_ANCHOR_DET_THR"] = str(float(args.anchor_det_thr))
    os.environ["SAM3_SPME_PER_OBJECT"] = "1" if args.pointer_mode in {"per_object", "hybrid"} else "0"
    os.environ["SAM3_SPME_KEEP_QUERIES"] = "1" if args.pointer_mode in {"per_object", "hybrid"} else "0"

    print("[DEBUG] Building SAM3 video model...", flush=True)
    model = build_sam3_video_model(
        checkpoint_path=str(Path(args.base_sam3_pt)),
        load_from_HF=False,
        bpe_path=str(args.bpe_path),
        device=str(device),
        compile=False,
    )
    if args.overlay_ckpt:
        print(f"[DEBUG] Applying detector overlay: {args.overlay_ckpt}", flush=True)
        _apply_detector_overlay(model, Path(args.overlay_ckpt))
    else:
        print("[DEBUG] No overlay ckpt; using base SAM3 detector weights.", flush=True)

    model.to(device)
    if hasattr(model, "score_threshold_detection"):
        model.score_threshold_detection = float(args.score_thr_detection)
    model.eval()
    if hasattr(model, "spme_fusion_mem_film_mlp"):
        model.spme_fusion_mem_film_mlp.train()
    if hasattr(model, "spme_fusion_obj_ptr_mlp"):
        model.spme_fusion_obj_ptr_mlp.train()

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

    data_root = Path(args.data_root)
    seqs = _index_endovis_train(data_root, set(args.train_seqs) if args.train_seqs else None)
    if not seqs:
        raise RuntimeError("No EndoVis train sequences found after filtering.")
    seq_map = {s.seq_id: s for s in seqs}

    low_res = int(getattr(model.tracker, "low_res_mask_size", 256))
    image_size = int(getattr(model, "image_size", 1008))
    print(f"[DEBUG] low_res={low_res}, image_size={image_size}", flush=True)

    start_time = time.time()
    rng = random.Random(int(args.seed) + 999)

    def _should_stop() -> bool:
        if args.max_steps and step >= int(args.max_steps):
            return True
        if args.max_hours and (time.time() - start_time) >= float(args.max_hours) * 3600.0:
            return True
        return False

    while True:
        if _should_stop():
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
        ok = True
        for fp, lp in zip(frame_paths, label_paths):
            img_t, oh, ow = _load_frame_tensor(fp, image_size=image_size)
            if orig_h is None:
                orig_h, orig_w = oh, ow
            images.append(img_t)
            lbl = _load_label_bmp(lp)
            m_np = (lbl == int(clip.class_id)).astype(np.uint8)
            masks_lr.append(_mask_to_low_res(m_np, low_res=low_res, device=device))
        if not ok or orig_h is None or orig_w is None:
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

            # Box prompt as two labeled points (SAM-style).
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
            num_frames=int(clip.length),
            cached_features=feature_cache,
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
            async_loading_frames=False,
        )
        _ = model.tracker._obj_id_to_idx(tracker_state, obj_id=0)
        output_dict = tracker_state["output_dict"]

        optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        n_loss_frames = 0
        scale_mem_vals: list[float] = []
        scale_obj_vals: list[float] = []

        for local_t in range(int(clip.length)):
            with torch.no_grad():
                det_out = model.run_backbone_and_detection(
                    frame_idx=local_t,
                    num_frames=int(clip.length),
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
                init_mask_t = _mask_torch_from_np(init_mask_np, device=device)
                model.tracker.add_new_mask(
                    inference_state=tracker_state,
                    frame_idx=0,
                    obj_id=0,
                    mask=init_mask_t,
                    add_mask_to_memory=True,
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
                    run_mem_encoder=True,
                )

            # Pointer selection for this tracked object (obj_id=0).
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

            if qv_sel is None and str(args.pointer_mode) in {"hybrid", "top1"}:
                qv_sel = det_out.get("spme_query_vec", None)
                det_score_sel = det_out.get("spme_det_score_raw", None)
                qcos_sel = det_out.get("spme_query_cos", None)
            if str(args.pointer_mode) == "top1":
                qv_sel = det_out.get("spme_query_vec", None)
                det_score_sel = det_out.get("spme_det_score_raw", None)
                qcos_sel = det_out.get("spme_query_cos", None)

            if qcos_sel is None and str(args.pointer_mode) == "hybrid":
                qcos_sel = det_out.get("spme_query_cos", None)

            s_mem, s_obj = _apply_spme_fusion_edit(
                model,
                current_out,
                query_vec=qv_sel,
                det_score_raw=det_score_sel if det_score_sel is not None else det_out.get("spme_det_score_raw", None),
                presence_prob=det_out.get("spme_presence_prob", None),
                query_cos=qcos_sel if qcos_sel is not None else det_out.get("spme_query_cos", None),
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

            storage_key = "cond_frame_outputs" if is_init else "non_cond_frame_outputs"
            output_dict[storage_key][local_t] = current_out

            if is_init:
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

        if step % int(args.log_every) == 0:
            log_row = {
                "step": int(step),
                "epoch": int(epoch),
                "seq_id": int(clip.seq_id),
                "class_id": int(clip.class_id),
                "start": int(clip.start),
                "clip_len": int(clip.length),
                "prompt": prompt,
                "prompt_mode": str(args.prompt_mode),
                "prompt_generic": str(args.prompt),
                "loss": float(total_loss.detach().item()),
                "bce": float(total_bce.detach().item()),
                "dice": float(total_dice.detach().item()),
                "scale_mem_mean": float(np.mean(scale_mem_vals)) if scale_mem_vals else None,
                "scale_obj_mean": float(np.mean(scale_obj_vals)) if scale_obj_vals else None,
                "time_sec": float(time.time() - start_time),
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")
            print(json.dumps(log_row), flush=True)

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
            print(f"Saved checkpoint: {ckpt_path}", flush=True)

        step += 1

    ckpt = {
        "step": int(step),
        "epoch": int(epoch),
        "spme_state_dict": _extract_spme_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_dir / "spme_fusion_latest.pt")
    print(f"Done. Saved latest: {ckpt_dir / 'spme_fusion_latest.pt'}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("SAM3_DISABLE_TRITON", "1")
    main()

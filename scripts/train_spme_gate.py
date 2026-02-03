#!/usr/bin/env python3
"""
Train learned SPME gates with a minimal 2-frame unroll (t -> t+1 supervision).

We freeze the full SAM3 video model and only train:
  - spme_gate_mlp (learned gate)
  - spme_fusion_mem_film_mlp (fusion projector)
  - spme_fusion_obj_ptr_mlp (pointer projector)

This script is intentionally lightweight and follows the existing OVIS fusion trainer
style (random (video, instance) clip sampling with RLE GT).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from sam3.model_builder import build_sam3_video_model

try:
    from pycocotools import mask as mask_util
except Exception as e:  # pragma: no cover
    raise ImportError(
        "OVIS training requires `pycocotools` for RLE decoding. "
        "Install it in your env (e.g., `pip install pycocotools`)."
    ) from e


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _decode_rle(seg: Any, height: int, width: int) -> np.ndarray:
    if seg is None:
        return np.zeros((height, width), dtype=np.uint8)
    if not isinstance(seg, dict):
        raise ValueError(f"Unsupported segmentation type: {type(seg)}")
    rle = seg
    counts = rle.get("counts", None)
    if isinstance(counts, list):
        rle = mask_util.frPyObjects(rle, height, width)
    m = mask_util.decode(rle)
    if m.ndim == 3:
        m = m[:, :, 0]
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
        if k in full and torch.is_tensor(v) and torch.is_tensor(full[k]) and full[k].shape == v.shape:
            full[k].copy_(v)


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
    print(f"Resumed from {resume_path} at step={step}")
    return step


def _read_video_ids(txt: Path) -> list[int]:
    vids: list[int] = []
    for ln in txt.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        vids.append(int(ln))
    return vids


@dataclass(frozen=True)
class SampledInstance:
    video_id: int
    video_dirname: str
    ann_id: int
    category_name: str
    height: int
    width: int
    file_names: list[str]
    segmentations: list[Any]


def _sample_instance(
    rng: random.Random,
    videos: dict[int, dict[str, Any]],
    anns_by_vid: dict[int, list[dict[str, Any]]],
    cat_id_to_name: dict[int, str],
    allowed_vids: list[int],
) -> SampledInstance | None:
    if not allowed_vids:
        return None
    vid = int(rng.choice(allowed_vids))
    v = videos.get(vid, None)
    if v is None:
        return None
    anns = anns_by_vid.get(vid, [])
    if not anns:
        return None
    ann = rng.choice(anns)
    segs = ann.get("segmentations", None)
    if not isinstance(segs, list) or not segs:
        return None
    file_names = v.get("file_names", None)
    if not isinstance(file_names, list) or not file_names:
        return None
    dirname = str(file_names[0]).split("/")[0]
    cat_name = cat_id_to_name.get(int(ann["category_id"]), "object")
    return SampledInstance(
        video_id=vid,
        video_dirname=dirname,
        ann_id=int(ann["id"]),
        category_name=str(cat_name),
        height=int(v["height"]),
        width=int(v["width"]),
        file_names=[str(x) for x in file_names],
        segmentations=segs,
    )


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
        mem_dim = int(mem.shape[1])
        gate_scale = mem_scale.to(device=mem.device, dtype=mem.dtype).view(1, mem_dim, 1, 1)
        gate_offset = mem_offset.to(device=mem.device, dtype=mem.dtype).view(1, mem_dim, 1, 1)
        gate_d = gate_decay.to(device=mem.device, dtype=mem.dtype).view(1, 1, 1, 1)
        if isinstance(prev_out, dict):
            prev_mem = prev_out.get("maskmem_features", None)
        else:
            prev_mem = None
        if isinstance(prev_mem, torch.Tensor) and prev_mem.shape == mem.shape:
            prev_mem = prev_mem.to(device=mem.device, dtype=mem.dtype)
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
        gate_scalar = mem_scale.mean(dim=-1, keepdim=True)
        gate_w = gate_scalar.to(device=ptr.device, dtype=ptr.dtype).view(1, 1)
        gate_d = gate_decay.to(device=ptr.device, dtype=ptr.dtype).view(1, 1)
        prev_ptr = prev_out.get("obj_ptr", None) if isinstance(prev_out, dict) else None
        if isinstance(prev_ptr, torch.Tensor) and prev_ptr.shape == ptr.shape:
            prev_ptr = prev_ptr.to(device=ptr.device, dtype=ptr.dtype)
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
            mem_dim = int(mem.shape[1])
            scale_mem = torch.clamp(g_f * float(base_alpha), 0.0, 1.0).to(dtype=mem.dtype).view(
                1, 1, 1, 1
            )
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, type=str, help=".../Images/train_extracted")
    ap.add_argument("--ann", required=True, type=str, help=".../annotations_train.json")
    ap.add_argument("--split-txt", required=True, type=str, help="train.txt from ovis_make_splits.py")
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--base-sam3-pt", required=True, type=str)
    ap.add_argument("--finetune-ckpt", type=str, default=None)
    ap.add_argument("--bpe-path", required=True, type=str)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)

    ap.add_argument("--clip-len", type=int, default=3, help="Must be >=3 (init + gate + supervise).")
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)

    ap.add_argument("--bce-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument("--absent-weight", type=float, default=0.25)

    ap.add_argument("--use-decay", type=int, default=1, help="Use gate_decay in memory update.")
    ap.add_argument("--gate-frame", type=int, default=1, help="Which local frame to apply learned gate on.")
    ap.add_argument("--supervise-frame", type=int, default=2, help="Which local frame to supervise on (t+1).")
    ap.add_argument("--occ-norm", type=float, default=10.0)
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

    ap.add_argument("--query-pool", type=str, default="top1", help="top1 | topk_weighted")
    ap.add_argument("--query-topk", type=int, default=5)
    ap.add_argument("--anchor-det-thr", type=float, default=0.0)
    ap.add_argument(
        "--pointer-mode",
        type=str,
        default="hybrid",
        choices=["per_object", "hybrid", "top1"],
    )
    ap.add_argument("--match-iou-thr", type=float, default=0.1)
    ap.add_argument("--match-topk", type=int, default=20)
    ap.add_argument("--score-thr-detection", type=float, default=0.05)

    ap.add_argument("--fusion-mode", type=str, default="film", choices=["resid", "film"])
    ap.add_argument("--fusion-alpha", type=float, default=0.05)
    ap.add_argument("--fusion-alpha-obj", type=float, default=0.005)
    ap.add_argument("--fusion-det-thr", type=float, default=0.0)

    ap.add_argument(
        "--grad-check",
        action="store_true",
        help="Run a tiny synthetic 2-step grad check and exit (no dataset needed).",
    )
    args = ap.parse_args()

    if int(args.clip_len) < 3:
        raise ValueError("--clip-len must be >= 3 (init + gate + supervise).")
    if not (0 <= int(args.gate_frame) < int(args.clip_len)):
        raise ValueError("--gate-frame out of range.")
    if not (0 <= int(args.supervise_frame) < int(args.clip_len)):
        raise ValueError("--supervise-frame out of range.")
    if int(args.gate_frame) == int(args.supervise_frame):
        raise ValueError("--gate-frame and --supervise-frame must differ.")
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
    # Disable training-time stochasticity (e.g., memory dropout); we only train small SPME modules.
    model.eval()
    if args.finetune_ckpt:
        sd = torch.load(str(args.finetune_ckpt), map_location="cpu")
        if isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
            sd = sd["model"]
        missing, unexpected = model.detector.load_state_dict(sd, strict=False)
        print(f"Loaded finetune ckpt into detector: missing={len(missing)} unexpected={len(unexpected)}")

    if hasattr(model, "score_threshold_detection"):
        model.score_threshold_detection = float(args.score_thr_detection)

    trainable = _freeze_non_spme_params(model)
    print(f"Trainable spme params: {len(trainable)}")

    if getattr(model, "spme_gate_mlp", None) is None:
        raise RuntimeError("spme_gate_mlp was not created. Is SAM3_SPME_LEARNED_GATE=1 set early enough?")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    start_step = 0
    if args.resume:
        start_step = _maybe_resume(model, optimizer, Path(args.resume), device=device)

    if args.grad_check:
        # Tiny synthetic 2-step chain: loss(t+1) -> mem(t) -> gate params
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
        loss = (pred_tp1 ** 2).mean()
        loss.backward()
        gsum = 0.0
        for p in gate.parameters():
            if p.grad is not None:
                gsum += float(p.grad.detach().abs().sum().item())
        print(f"[grad_check] loss={float(loss.item()):.6f} gate_grad_abs_sum={gsum:.6f}")
        return

    images_root = Path(args.images_root)
    ann_obj = json.loads(Path(args.ann).read_text())
    if ann_obj.get("annotations") is None:
        raise ValueError("This annotation json has no GT (`annotations=null`). Use annotations_train.json.")
    cat_id_to_name = {int(c["id"]): str(c["name"]) for c in ann_obj["categories"]}
    videos = {int(v["id"]): v for v in ann_obj["videos"]}
    anns_by_vid: dict[int, list[dict[str, Any]]] = {}
    for ann in ann_obj["annotations"]:
        anns_by_vid.setdefault(int(ann["video_id"]), []).append(ann)

    allowed_vids = _read_video_ids(Path(args.split_txt))
    if not allowed_vids:
        raise RuntimeError(f"No video ids found in {args.split_txt}")

    # Pre-resolve per-video frame dirs to avoid repeatedly sampling missing videos.
    vid_to_frames_dir: dict[int, Path] = {}
    for vid in allowed_vids:
        v = videos.get(int(vid), None)
        if v is None:
            continue
        file_names = v.get("file_names", None)
        if not isinstance(file_names, list) or not file_names:
            continue
        dirname = str(file_names[0]).split("/")[0]
        cand = images_root / dirname
        if cand.exists():
            vid_to_frames_dir[int(vid)] = cand
            continue
        alt = images_root / "train" / dirname
        if alt.exists():
            vid_to_frames_dir[int(vid)] = alt
    if not vid_to_frames_dir:
        raise RuntimeError(
            f"No usable videos found under images_root={images_root}. "
            "Expected either <root>/<video_dir>/... or <root>/train/<video_dir>/..."
        )
    allowed_vids = sorted(vid_to_frames_dir.keys())
    print(f"[OVIS] usable videos with frames: {len(allowed_vids)}")

    rng = random.Random(int(args.seed))
    image_size = int(getattr(model, "image_size", 1008))
    low_res = int(getattr(model.tracker, "low_res_mask_size", 256))

    # Persistent running state for the simple 1-object gate signals within a clip.
    # (Reset each iteration to avoid leaking across random samples.)
    max_steps = int(args.max_steps)
    step = int(start_step)
    while step < max_steps:
        inst = _sample_instance(rng, videos, anns_by_vid, cat_id_to_name, allowed_vids)
        if inst is None:
            continue
        frames_dir = vid_to_frames_dir.get(int(inst.video_id), None)
        if frames_dir is None:
            continue
        length = len(inst.segmentations)
        if length < int(args.clip_len):
            continue

        # Pick a start index where init frame is present (for bbox init).
        start = None
        tries = 0
        while tries < 10 and start is None:
            t0 = int(rng.randint(0, max(0, length - int(args.clip_len))))
            m0 = _decode_rle(inst.segmentations[t0], height=inst.height, width=inst.width)
            if m0.any():
                start = t0
            tries += 1
        if start is None:
            continue

        images: list[torch.Tensor] = []
        masks_lr: list[torch.Tensor] = []
        orig_h, orig_w = None, None
        ok = True
        for local_t in range(int(args.clip_len)):
            idx = start + local_t
            rel_path = inst.file_names[idx]
            img_rel = "/".join(rel_path.split("/")[1:])  # drop dirname/
            img_path = frames_dir / img_rel
            if not img_path.exists():
                ok = False
                break
            img_t, oh, ow = _load_frame_tensor(img_path, image_size=image_size)
            if orig_h is None:
                orig_h, orig_w = oh, ow
            images.append(img_t)
            m_np = _decode_rle(inst.segmentations[idx], height=inst.height, width=inst.width)
            masks_lr.append(_mask_to_low_res(m_np, low_res=low_res, device=device))
        if not ok or orig_h is None or orig_w is None:
            continue

        images_t = torch.stack(images, dim=0)  # keep on CPU; _construct_initial_input_batch copies to GPU

        # Build detector input batch.
        inference_state: dict[str, Any] = {}
        inference_state["image_size"] = image_size
        inference_state["num_frames"] = int(args.clip_len)
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

        prompt = inst.category_name.lower()
        inference_state["text_prompt"] = prompt
        inference_state["input_batch"].find_text_batch[0] = prompt
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = model.TEXT_ID_FOR_TEXT

        # Init geometric prompt from GT bbox at local frame 0.
        init_mask_np = _decode_rle(inst.segmentations[start], height=inst.height, width=inst.width)
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

        # Box prompt as two labeled points.
        x0 = float(bbox_xywh[0] * image_size)
        y0 = float(bbox_xywh[1] * image_size)
        x1 = float((bbox_xywh[0] + bbox_xywh[2]) * image_size)
        y1 = float((bbox_xywh[1] + bbox_xywh[3]) * image_size)
        point_coords = torch.tensor([[[x0, y0], [x1, y1]]], dtype=torch.float32, device=device)
        point_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=device)
        point_inputs0 = {"point_coords": point_coords, "point_labels": point_labels}

        # --- Forward unroll ---
        optimizer.zero_grad(set_to_none=True)

        prev_area = None
        missing_count = 0
        prev_out_for_gate = None
        qv_gate = None
        det_score_gate = None
        qcos_gate = None
        tracker_score_gate = None

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
                        scores_t = det_out.get("scores", None)
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

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=float(args.grad_clip)
        )
        optimizer.step()

        # Light sanity: gate params should receive gradients.
        gate_grad = 0.0
        for p in model.spme_gate_mlp.parameters():
            if p.grad is not None:
                gate_grad += float(p.grad.detach().abs().sum().item())

        if step % int(args.log_every) == 0:
            row = {
                "event": "step",
                "time": float(time.time()),
                "step": int(step),
                "video_id": int(inst.video_id),
                "ann_id": int(inst.ann_id),
                "gate_grad_abs_sum": float(gate_grad),
                "gate_inputs": str(args.gate_inputs),
                "gate_det_score": float(det_score_gate) if det_score_gate is not None else None,
                "gate_presence_prob": float(presence_gate) if presence_gate is not None else None,
                "gate_qcos": float(qcos_gate) if qcos_gate is not None else None,
                "gate_tracker_score": float(tracker_score_gate) if tracker_score_gate is not None else None,
                "gate_area_delta": float(delta) if delta is not None else None,
                "gate_last_occluded_norm": float(last_occ) if last_occ is not None else None,
            }
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if step % int(args.save_every) == 0 and step > 0:
            ckpt_path = ckpt_dir / f"spme_gate_step{step}.pt"
            torch.save(
                {
                    "step": int(step),
                    "spme_state_dict": _extract_spme_state_dict(model),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                },
                str(ckpt_path),
            )

        step += 1
        if step >= max_steps:
            break


if __name__ == "__main__":
    main()

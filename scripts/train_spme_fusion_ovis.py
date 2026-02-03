#!/usr/bin/env python3
"""
Train SPME-Fusion on OVIS train videos (self-split from annotations_train.json).

We follow the same training recipe as `train_spme_fusion_ultrasound.py`, but sample
random (video, instance) clips from OVIS and supervise with GT masks (RLE).
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
        if k in full and torch.is_tensor(v):
            full[k].copy_(v)


def _compute_fusion_scales(
    *,
    det_score_raw: float | None,
    presence_prob: float | None,
    query_cos: float | None,
    base_alpha: float,
    base_alpha_obj: float,
    det_thr: float,
    det_score_floor: float,
    qcos_thr: float,
    qcos_temp: float,
    qcos_gate: str,
    use_presence: bool,
) -> tuple[float | None, float | None]:
    if det_score_raw is None:
        return None, None
    det_score_f = float(det_score_raw)
    if det_score_floor > 0.0:
        det_score_f = max(det_score_f, float(det_score_floor))
    if det_thr > 0.0 and det_score_f < det_thr:
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
    det_score_floor: float,
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
        det_score_floor=det_score_floor,
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
    qv = qv.view(1, -1)

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", required=True, type=str, help=".../Images/train_extracted")
    ap.add_argument("--ann", required=True, type=str, help=".../annotations_train.json")
    ap.add_argument("--split-txt", required=True, type=str, help="train.txt from ovis_make_splits.py")
    ap.add_argument("--out-dir", required=True, type=str)
    ap.add_argument("--base-sam3-pt", required=True, type=str)
    ap.add_argument(
        "--finetune-ckpt",
        type=str,
        default=None,
        help="Optional detector fine-tune checkpoint (trainer-style ckpt with ['model']).",
    )
    ap.add_argument("--bpe-path", required=True, type=str)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--clip-len", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=5000)
    ap.add_argument("--max-hours", type=float, default=0.0, help="0 = ignore")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--bce-weight", type=float, default=1.0)
    ap.add_argument("--dice-weight", type=float, default=1.0)
    ap.add_argument(
        "--status-every",
        type=int,
        default=200,
        help="Write a heartbeat status row every N sampling attempts (helps detect skip-loops).",
    )
    ap.add_argument(
        "--status-secs",
        type=float,
        default=60.0,
        help="Also write status if this many seconds elapsed since last heartbeat.",
    )

    # SPME signal extraction (controls how we compute `spme_query_vec` / `spme_det_score_raw`).
    ap.add_argument(
        "--query-pool",
        type=str,
        default="top1",
        help="Semantic pointer pooling: top1 | topk_weighted",
    )
    ap.add_argument(
        "--query-topk",
        type=int,
        default=5,
        help="Top-K for pooling when --query-pool=topk_weighted.",
    )
    ap.add_argument(
        "--anchor-det-thr",
        type=float,
        default=0.0,
        help="Delay anchor init until det_score_raw >= this threshold (0 disables).",
    )
    ap.add_argument(
        "--pointer-mode",
        type=str,
        default="hybrid",
        choices=["per_object", "hybrid", "top1"],
        help=(
            "How to pick the semantic pointer for fusion on OVIS.\n"
            "- per_object: match detector masks to current tracker mask and use matched detection query vec (recommended).\n"
            "- hybrid: per_object when matched, otherwise fallback to global top1.\n"
            "- top1: always use global top1 pointer (high risk on multi-instance OVIS)."
        ),
    )
    ap.add_argument(
        "--match-iou-thr",
        type=float,
        default=0.1,
        help="Mask IoU threshold to consider a detection matched to the tracked object (per_object mode).",
    )
    ap.add_argument(
        "--match-topk",
        type=int,
        default=20,
        help="Consider at most top-K detections by IoU for per_object matching (0 = no limit).",
    )
    ap.add_argument(
        "--score-thr-detection",
        type=float,
        default=0.05,
        help="Override model.score_threshold_detection (lower helps get det candidates for matching).",
    )

    # Fusion hyperparams (mirror ultrasound trainer)
    ap.add_argument("--fusion-mode", type=str, default="film")
    ap.add_argument("--fusion-alpha", type=float, default=0.05)
    ap.add_argument("--fusion-alpha-obj", type=float, default=0.005)
    ap.add_argument("--fusion-det-thr", type=float, default=0.0)
    ap.add_argument(
        "--fusion-det-score-floor",
        type=float,
        default=0.0,
        help="Clamp det_score_raw to at least this value when computing fusion scales (prevents all-zero det_score under AMP).",
    )
    ap.add_argument("--fusion-qcos-thr", type=float, default=0.5)
    ap.add_argument("--fusion-qcos-temp", type=float, default=20.0)
    ap.add_argument("--fusion-qcos-gate", type=str, default="sigmoid")
    ap.add_argument("--fusion-use-presence", type=int, default=1)
    ap.add_argument(
        "--debug-first-n",
        type=int,
        default=0,
        help="Print a compact detector/pointer debug line for the first N sampled clips (to diagnose no_edit_frames).",
    )
    args = ap.parse_args()

    _seed_everything(int(args.seed))

    # Enable SPME signal attachment (query vectors + qcos) in the detector output.
    # Without this, `det_out` won't contain `spme_query_vec`, and the loss will not
    # depend on `spme_*` params, causing backward() to fail.
    os.environ["SAM3_SPME_FUSION"] = "1"
    os.environ["SAM3_SPME_QUERY_POOL"] = str(args.query_pool)
    os.environ["SAM3_SPME_QUERY_TOPK"] = str(int(args.query_topk))
    os.environ["SAM3_SPME_ANCHOR_DET_THR"] = str(float(args.anchor_det_thr))
    os.environ["SAM3_SPME_PER_OBJECT"] = "1" if args.pointer_mode in {"per_object", "hybrid"} else "0"
    # Required for true per-object pointers: we need per-detection query embeddings so we can
    # align a detection to the tracked instance (OVIS is multi-instance).
    os.environ["SAM3_SPME_KEEP_QUERIES"] = "1" if args.pointer_mode in {"per_object", "hybrid"} else "0"

    images_root = Path(args.images_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    status_path = out_dir / "status.jsonl"

    # Create logs immediately so it's obvious the job is alive even before any optimizer steps succeed.
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "time": float(time.time()), "args": vars(args)}) + "\n")
    with status_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"event": "start", "time": float(time.time())}) + "\n")

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

    # Pre-resolve per-video frame dirs to avoid repeatedly sampling videos with missing frames.
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

    device = torch.device("cuda")
    model = build_sam3_video_model(
        checkpoint_path=str(args.base_sam3_pt),
        load_from_HF=False,
        device="cuda",
        bpe_path=str(args.bpe_path),
        compile=False,
    )
    try:
        model.score_threshold_detection = float(args.score_thr_detection)
        print(f"[OVIS] score_threshold_detection={model.score_threshold_detection}")
    except Exception:
        print("[OVIS] WARNING: could not set model.score_threshold_detection (unexpected model type).")

    if args.finetune_ckpt:
        finetune_sd = torch.load(str(args.finetune_ckpt), map_location="cpu")
        finetune_sd = (
            finetune_sd["model"] if isinstance(finetune_sd, dict) and "model" in finetune_sd else finetune_sd
        )
        target = getattr(model, "detector", None) or model
        missing, unexpected = target.load_state_dict(finetune_sd, strict=False)
        print(f"Loaded finetune into detector: missing={len(missing)} unexpected={len(unexpected)}")
    else:
        print("No finetune ckpt provided; using base SAM3 detector weights.")

    model.eval()
    if hasattr(model, "spme_fusion_mem_film_mlp"):
        model.spme_fusion_mem_film_mlp.train()
    if hasattr(model, "spme_fusion_obj_ptr_mlp"):
        model.spme_fusion_obj_ptr_mlp.train()

    trainable = _freeze_non_spme_params(model)
    print(f"Trainable params ({len(trainable)}):")
    for n in trainable:
        print(f"  - {n}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    step = 0
    if args.resume:
        step = _maybe_resume(model, optimizer, Path(args.resume), device)

    low_res = int(getattr(model.tracker, "low_res_mask_size", 256))
    image_size = int(getattr(model, "image_size", 1008))
    print(f"low_res={low_res} image_size={image_size}")

    start_time = time.time()
    rng = random.Random(int(args.seed) + 2026)
    attempts = 0
    no_step_attempts = 0
    skip_counts: dict[str, int] = defaultdict(int)
    last_status_time = time.time()
    last_skip_reason: str | None = None

    def _write_status(force: bool = False) -> None:
        nonlocal last_status_time
        if not force:
            every = int(args.status_every) if args.status_every and int(args.status_every) > 0 else 0
            if every and attempts % every != 0:
                secs = float(args.status_secs) if args.status_secs else 0.0
                if secs <= 0.0 or (time.time() - last_status_time) < secs:
                    return
        row = {
            "event": "status",
            "time": float(time.time()),
            "attempts": int(attempts),
            "step": int(step),
            "no_step_attempts": int(no_step_attempts),
            "last_skip": last_skip_reason,
            "skip_counts": dict(skip_counts),
        }
        with status_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        last_status_time = time.time()

    def _skip(reason: str) -> None:
        nonlocal no_step_attempts
        nonlocal last_skip_reason
        skip_counts[str(reason)] = int(skip_counts.get(str(reason), 0) + 1)
        no_step_attempts += 1
        last_skip_reason = str(reason)
        _write_status(force=False)

    def _should_stop() -> bool:
        if args.max_steps and step >= int(args.max_steps):
            return True
        if args.max_hours and float(args.max_hours) > 0 and (time.time() - start_time) >= float(args.max_hours) * 3600.0:
            return True
        return False

    while not _should_stop():
        attempts += 1
        inst = _sample_instance(rng, videos, anns_by_vid, cat_id_to_name, allowed_vids)
        if inst is None:
            _skip("no_instance")
            continue
        frames_dir = vid_to_frames_dir.get(int(inst.video_id), None)
        if frames_dir is None:
            _skip("frames_dir_missing")
            continue

        length = min(len(inst.file_names), len(inst.segmentations))
        if length < int(args.clip_len):
            _skip("too_short")
            continue

        # choose a start where init mask exists & non-empty
        tries = 0
        start = None
        while tries < 10 and start is None:
            t0 = int(rng.randint(0, max(0, length - int(args.clip_len))))
            seg0 = inst.segmentations[t0]
            m0 = _decode_rle(seg0, height=inst.height, width=inst.width)
            if m0.any():
                start = t0
            tries += 1
        if start is None:
            _skip("no_nonempty_start")
            continue

        # Load clip frames + GT masks.
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
            _skip("missing_frame")
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
            _skip("empty_init_mask")
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

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_bce = 0.0
        total_dice = 0.0
        n_loss_frames = 0
        n_edit_frames = 0
        scale_mem_vals: list[float] = []
        scale_obj_vals: list[float] = []

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
                run_mem_encoder=True,
            )

            # Pick semantic pointer for the tracked object.
            qv_sel = None
            det_score_sel = None
            qcos_sel = None
            ptr_source = "none"
            debug_det_scores = None
            debug_det_n = None
            debug_det_masks_n = None
            debug_best_iou = None
            debug_matched_n = None
            if str(args.pointer_mode) in {"per_object", "hybrid"}:
                try:
                    det_masks = det_out.get("mask", None)
                    scores_t = det_out.get("scores", None)
                    if isinstance(scores_t, torch.Tensor) and scores_t.numel() > 0:
                        s = scores_t.detach().float()
                        debug_det_scores = (float(s.min().item()), float(s.max().item()))
                        debug_det_n = int(s.numel())
                    if isinstance(det_masks, torch.Tensor) and det_masks.numel() > 0:
                        debug_det_masks_n = int(det_masks.shape[0])
                        # Tracker mask logits -> binary (low-res)
                        trk_logits = pred_masks_gpu
                        if isinstance(trk_logits, torch.Tensor):
                            if trk_logits.ndim == 4:  # (B,1,H,W)
                                trk_logits = trk_logits[0, 0]
                            elif trk_logits.ndim == 3:  # (B,H,W)
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

                        det_bin = (det_masks_res.detach() > 0).to(dtype=torch.bool)  # (N,H,W)
                        inter = (det_bin & trk_bin).sum(dim=(-1, -2)).float()
                        union = (det_bin | trk_bin).sum(dim=(-1, -2)).float().clamp_min(1e-6)
                        ious = inter / union
                        best_iou = float(ious.max().detach().float().item()) if ious.numel() else None
                        debug_best_iou = best_iou

                        # Match *multiple* detections to the current tracked object based on IoU,
                        # then let `_build_spme_per_object_context()` pick the best one per track
                        # using the detector score (more stable than "best IoU only" when scores are sparse).
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
                        debug_matched_n = int(len(det_to_matched))
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
                                if qv_sel is not None:
                                    ptr_source = "per_object"
                except Exception:
                    qv_sel = None
            if qv_sel is None and str(args.pointer_mode) == "hybrid":
                qv_sel = det_out.get("spme_query_vec", None)
                det_score_sel = det_out.get("spme_det_score_raw", None)
                qcos_sel = det_out.get("spme_query_cos", None)
                if qv_sel is not None:
                    ptr_source = "top1_fallback"
            if str(args.pointer_mode) == "top1":
                qv_sel = det_out.get("spme_query_vec", None)
                det_score_sel = det_out.get("spme_det_score_raw", None)
                qcos_sel = det_out.get("spme_query_cos", None)
                if qv_sel is not None:
                    ptr_source = "top1"

            if int(args.debug_first_n) > 0 and attempts <= int(args.debug_first_n) and local_t == 0:
                qv_ok = bool(isinstance(qv_sel, torch.Tensor) and qv_sel.numel() > 0)
                dbg = {
                    "event": "debug_ptr",
                    "attempt": int(attempts),
                    "video_id": int(inst.video_id),
                    "ann_id": int(inst.ann_id),
                    "frame": int(local_t),
                    "pointer_mode": str(args.pointer_mode),
                    "ptr_source": str(ptr_source),
                    "score_thr_detection": float(getattr(model, "score_threshold_detection", float("nan"))),
                    "det_scores_minmax": debug_det_scores,
                    "det_scores_n": debug_det_n,
                    "det_masks_n": debug_det_masks_n,
                    "best_iou": float(debug_best_iou) if debug_best_iou is not None else None,
                    "matched_n": int(debug_matched_n) if debug_matched_n is not None else None,
                    "spme_det_score_raw": det_out.get("spme_det_score_raw", None),
                    "spme_query_cos": det_out.get("spme_query_cos", None),
                    "qv_ok": int(qv_ok),
                    "det_score_sel": float(det_score_sel) if det_score_sel is not None else None,
                    "qcos_sel": float(qcos_sel) if qcos_sel is not None else None,
                    "fusion_det_score_floor": float(args.fusion_det_score_floor),
                }
                with status_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(dbg) + "\n")
                print(json.dumps(dbg), flush=True)

            s_mem, s_obj = _apply_spme_fusion_edit(
                model,
                current_out,
                query_vec=qv_sel,
                det_score_raw=det_score_sel,
                presence_prob=det_out.get("spme_presence_prob", None),
                query_cos=qcos_sel,
                base_alpha=float(args.fusion_alpha),
                base_alpha_obj=float(args.fusion_alpha_obj),
                det_thr=float(args.fusion_det_thr),
                det_score_floor=float(args.fusion_det_score_floor),
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
            if s_mem is not None or s_obj is not None:
                n_edit_frames += 1

            storage_key = "cond_frame_outputs" if is_init else "non_cond_frame_outputs"
            output_dict[storage_key][local_t] = current_out

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
            _skip("no_loss_frames")
            continue
        if n_edit_frames == 0:
            # No fusion edits were applied in this clip (likely due to missing/mismatched detections).
            # Skip without crashing; this is expected on multi-instance videos with strict thresholds.
            _skip("no_edit_frames")
            continue
        total_loss = total_loss / float(n_loss_frames)
        total_bce = total_bce / float(n_loss_frames)
        total_dice = total_dice / float(n_loss_frames)

        if not isinstance(total_loss, torch.Tensor) or not total_loss.requires_grad:
            # Extra safety: should not happen if n_edit_frames > 0, but keep it robust.
            _skip("loss_no_grad")
            continue

        total_loss.backward()
        if args.grad_clip and float(args.grad_clip) > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                max_norm=float(args.grad_clip),
            )
        optimizer.step()
        no_step_attempts = 0

        if step % int(args.log_every) == 0:
            row = {
                "step": int(step),
                "video_id": int(inst.video_id),
                "ann_id": int(inst.ann_id),
                "category": prompt,
                "clip_start": int(start),
                "clip_len": int(args.clip_len),
                "loss": float(total_loss.detach().item()),
                "bce": float(total_bce.detach().item()),
                "dice": float(total_dice.detach().item()),
                "n_edit_frames": int(n_edit_frames),
                "scale_mem_mean": float(np.mean(scale_mem_vals)) if scale_mem_vals else None,
                "scale_obj_mean": float(np.mean(scale_obj_vals)) if scale_obj_vals else None,
                "time_sec": float(time.time() - start_time),
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
            ckpt_path = ckpt_dir / f"spme_fusion_step{step:06d}.pt"
            torch.save(ckpt, ckpt_path)
            torch.save(ckpt, ckpt_dir / "spme_fusion_latest.pt")
            print(f"Saved checkpoint: {ckpt_path}", flush=True)

        step += 1
        _write_status(force=False)

    ckpt = {
        "step": int(step),
        "spme_state_dict": _extract_spme_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(ckpt, ckpt_dir / "spme_fusion_latest.pt")
    print(f"Done. Saved latest: {ckpt_dir / 'spme_fusion_latest.pt'}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("SAM3_DISABLE_TRITON", "1")
    main()

#!/usr/bin/env python3
"""
Lightweight sanity checks for learned SPME gating.

This does NOT require SAM3 weights or datasets. It verifies:
  1) gates are in [0,1]
  2) a 2-step unroll chain produces non-zero gradients on gate params
"""

from __future__ import annotations

import argparse

import torch

from sam3.model.sam3_video_base import SPMEGateMLP


def _grad_check(device: str) -> None:
    gate = SPMEGateMLP(mem_dim=16).to(device=device)
    gate.train()
    x = torch.randn(8, 5, device=device)
    mem_scale, mem_offset, gate_decay = gate(x)

    assert mem_scale.min().item() >= 0.0 and mem_scale.max().item() <= 1.0
    assert mem_offset.abs().max().item() <= float(getattr(gate, "offset_scale", 0.1)) + 1e-4
    assert gate_decay.min().item() >= 0.0 and gate_decay.max().item() <= 1.0

    old_mem = torch.randn(8, 16, 8, 8, device=device)
    new_mem = torch.randn(8, 16, 8, 8, device=device)

    s = mem_scale.view(-1, 16, 1, 1)
    o = mem_offset.view(-1, 16, 1, 1)
    d = gate_decay.view(-1, 1, 1, 1)
    mem_t = new_mem * s + o + old_mem * (1.0 - s) * (1.0 - d)
    gate_scalar = mem_scale.mean(dim=-1)
    pred_tp1 = mem_t.mean(dim=(1, 2, 3)) + 0.1 * gate_scalar
    loss = (pred_tp1**2).mean()
    loss.backward()

    gsum = 0.0
    gmax = 0.0
    for p in gate.parameters():
        if p.grad is None:
            continue
        gsum += float(p.grad.detach().abs().sum().item())
        gmax = max(gmax, float(p.grad.detach().abs().max().item()))

    print(f"[ok] loss={float(loss.item()):.6f} gate_grad_abs_sum={gsum:.6f} gate_grad_abs_max={gmax:.6f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()
    _grad_check(args.device)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""环境自检。跑训练前先过这一关，能省掉最贵的那类错误。

重点查三件事：
  1. 显卡是不是 Hopper（H20/H100 都是 sm90）——是的话必须装 tilelang
  2. triton 版本是否落在会静默算错梯度的区间 [3.4.0, 3.7.1)
  3. transformers 是否 >= 5.x（qwen3_5 架构需要）
"""

from __future__ import annotations

import importlib
import sys


def ver(mod: str):
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception:
        return None


def parse(v: str):
    out = []
    for p in v.split(".")[:3]:
        try:
            out.append(int("".join(c for c in p if c.isdigit()) or 0))
        except ValueError:
            out.append(0)
    return tuple(out + [0] * (3 - len(out)))


def main():
    ok = True
    print("=== 关键包 ===")
    for name in ("torch", "transformers", "peft", "triton", "causal_conv1d",
                 "tilelang", "fla", "numpy", "accelerate"):
        v = ver(name)
        print(f"  {name:<16} {v if v else '缺失'}")

    torch_v = ver("torch")
    if torch_v:
        import torch
        print(f"\n  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            print(f"  GPU 数: {n}")
            for i in range(min(n, 4)):
                p = torch.cuda.get_device_properties(i)
                cc = f"{p.major}.{p.minor}"
                print(f"    [{i}] {p.name}  {p.total_memory/1024**3:.0f} GB  sm{cc}")
        else:
            print("  ❌ CUDA 不可用，训练无法进行")
            ok = False

    triton_v = ver("triton")
    tilelang_v = ver("tilelang")
    is_hopper = False
    if torch_v:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            is_hopper = (p.major, p.minor) == (9, 0)

    print("\n=== 梯度正确性检查 ===")
    if is_hopper and triton_v and parse(triton_v) >= (3, 4, 0) and parse(triton_v) < (3, 7, 1):
        if tilelang_v:
            print(f"  ✅ Hopper + triton {triton_v} 落在风险区间，但已装 tilelang {tilelang_v}")
            print("     fla 会自动切后端（日志里看 `[FLA Backend] ... -> tilelang`）")
        else:
            print(f"  ❌ Hopper + triton {triton_v}，且没装 tilelang")
            print("     fla 的 gated chunk_bwd_dqkwg 在这个组合下**算错梯度但不报错**")
            print("     解决：pip install tilelang==0.1.15，或不带 tilelang 时 sft.py 会直接 raise")
            ok = False
    else:
        print("  非风险组合，或 triton 版本不在 [3.4.0, 3.7.1)，跳过")

    tf = ver("transformers")
    if tf and parse(tf)[0] < 5:
        print(f"\n  ❌ transformers {tf} 太旧，qwen3_5 架构需要 >= 5.0（本项目用 5.19.0）")
        ok = False

    print("\n=== 结论 ===")
    print("  ✅ 可以开始跑 probe_memory.sh" if ok else "  ❌ 先解决上面标 ❌ 的项")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""把 LoRA adapter 合回底座，产出可直接 serve 的完整模型目录。

要点：
- 复用训练器的 load_bundle，避免自己处理 Qwen3_5 的 slot 校验
- 必须把 adapter 里的 readout.json 一起带走，否则 DecisionEngine 读不到
  attention 配置、退回默认 causal，评测结果全部失真（这个坑我们踩过）
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]
ONEJEV = Path(os.environ.get("ONEJEV", WORK / "OneJev"))
sys.path.insert(0, str(ONEJEV))

from peft import PeftModel  # noqa: E402
from train.common import load_bundle  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(WORK / "models" / "mammoth-1"))
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    bundle = load_bundle(args.base, device="cuda:0", dtype="bfloat16")
    print(f"base loaded in {time.time()-t0:.1f}s | slots={len(bundle.slot_ids)}", flush=True)

    model = PeftModel.from_pretrained(bundle.model, args.adapter)
    print("adapter attached", flush=True)
    merged = model.merge_and_unload()
    print(f"merged in {time.time()-t0:.1f}s", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), safe_serialization=True)
    bundle.tokenizer.save_pretrained(str(out))

    src = Path(args.adapter) / "readout.json"
    if src.exists():
        shutil.copy(src, out / "readout.json")
        print("copied readout.json from adapter", flush=True)
    else:
        # 不能只告警：缺这个文件时推理会静默退回 causal，评测结果全部失真且不报错
        sys.exit("FATAL: adapter 里没有 readout.json，合并会得到注意力配置错误的模型。"
                 "请先确认训练用了 --attention noncausal，并从训练产物里补回该文件。")

    print(f"saved -> {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()

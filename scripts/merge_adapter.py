#!/usr/bin/env python3
"""把 LoRA adapter 合回底座，产出可直接 serve 的完整模型目录。

要点（这几条都是踩过的）：
- 复用训练器的 load_bundle，避免自己处理 Qwen3_5 的 slot 校验
- **`readout.json` 在底座目录里**（`{"readout": "slot_lm", "head_dim": 256, "n_slots": 256}`），
  训练器**不产出**这个文件。所以合并时要从 `--base` 拷，不是从 adapter 拷。
  旧版从 adapter 找、找不到就退出，等于每个 run 都必然失败。
- **注意力（block_bidirectional）在 `config.json` 的 `modeling_overrides` 里**，
  不在 readout.json，也不在命令行。合并后要确认这一节被带过来了。
- 全量微调（`--fsdp`）的产物本来就是完整模型，**不需要**跑这个脚本。

用法：
    python scripts/merge_adapter.py --adapter runs/<name>/adapter --out models/<name>-merged

注意：训练时直接加 `--merge` 更省事（训练器内置合并，产物在 `runs/<name>/merged`），
这个脚本是给当时没加 `--merge` 的 run 补合并用的。
"""

from __future__ import annotations

import argparse
import json
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


def read_cfg(p: Path) -> dict:
    f = p / "config.json"
    return json.loads(f.read_text()) if f.exists() else {}


def overrides(cfg: dict) -> dict:
    """modeling_overrides 可能在顶层，也可能在 text_config 里，两处都看。"""
    if "modeling_overrides" in cfg:
        return cfg["modeling_overrides"] or {}
    return (cfg.get("text_config") or {}).get("modeling_overrides") or {}


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

    # readout.json 跟着底座走。adapter 目录里没有是正常的，不要在这里报错。
    for cand, tag in ((Path(args.base) / "readout.json", "底座"),
                      (Path(args.adapter) / "readout.json", "adapter")):
        if cand.exists():
            shutil.copy(cand, out / "readout.json")
            print(f"copied readout.json from {tag}: {json.loads(cand.read_text())}", flush=True)
            break
    else:
        print("  ⚠️  底座和 adapter 里都没有 readout.json —— 推理侧可能读不到 slot 配置", flush=True)

    # 真正要紧的是这一条：合并后的 config 还能不能看到 modeling_overrides
    base_ov = overrides(read_cfg(Path(args.base)))
    out_ov = overrides(read_cfg(out))
    print(f"\nbase  modeling_overrides: {json.dumps(base_ov, ensure_ascii=False)}")
    print(f"merged modeling_overrides: {json.dumps(out_ov, ensure_ascii=False)}")
    if base_ov and not out_ov:
        sys.exit(
            "FATAL: 底座有 modeling_overrides，合并后的 config.json 里没了。\n"
            "  这会让推理退回 causal 注意力，评测数字全是错的且不报错。\n"
            f"  修法：把 {args.base}/config.json 里这一节手工补进 {out}/config.json 再评。"
        )
    if base_ov != out_ov:
        print("  ⚠️  两处不一致，评测前请确认合并后的这一节是否是有意为之", flush=True)

    print(f"\nsaved -> {out}")
    for p in sorted(out.iterdir()):
        print(f"  {p.name}  {p.stat().st_size/1e6:.1f} MB")


if __name__ == "__main__":
    main()

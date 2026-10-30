#!/usr/bin/env python3
"""造一个"只放大训练集、评测集逐条不变"的数据版本，用来回答「1,250 条够不够」。

为什么不能直接 `N_DROP=1000 python build_dataset.py`：那会连 val/heldout 一起重抽，
题换了，"涨了还是没涨"就分不清是数据量还是换题。这里只取放大后的训练集，
val/heldout 一律沿用 `data/` 里已有的那份。

用法：
    python scripts/make_scaled.py --drop 1000            # 1,250 -> 1,750
    python scripts/make_scaled.py --drop 2000 --long 500 # 上限附近

产物：
    data/train_targeted_2x.jsonl（或 --out 指定）

注意 long_policy 有硬上限：规则族的去重空间加起来约 620 条，再多会静默少给。
要突破得给 build_long_policy 加新的变量轴，不是调大数字就行。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop", type=int, default=1000, help="DROP 训练条数（原始可用 15,088）")
    ap.add_argument("--long", type=int, default=250, help="long_policy 条数，上限约 620")
    ap.add_argument("--synth", type=int, default=200)
    ap.add_argument("--anchor", type=int, default=300)
    ap.add_argument("--out", default="data/train_targeted_2x.jsonl")
    args = ap.parse_args()

    base_train = ROOT / "data" / "train_targeted.jsonl"
    if not base_train.exists():
        sys.exit("FATAL: 先跑 python scripts/build_dataset.py 生成基准数据")

    with tempfile.TemporaryDirectory() as td:
        env = {**os.environ, "OUT_DIR": td, "N_DROP": str(args.drop),
               "N_LONG": str(args.long), "N_SYNTH": str(args.synth),
               "N_ANCHOR": str(args.anchor)}
        print(f"临时构建：N_DROP={args.drop} N_LONG={args.long} → {td}")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dataset.py")],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:], sep="\n")
            sys.exit("FATAL: build_dataset 失败")
        rows = [json.loads(l) for l in (Path(td) / "train_targeted.jsonl").read_text().splitlines() if l.strip()]
        # 临时目录里的 val/heldout 一律丢弃，下面会核对基准那份没被改过
        src = Counter(x["source"] for x in rows)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")

    print(f"\n写出 {out.relative_to(ROOT)}  {len(rows)} 条")
    print(f"  来源 {dict(src)}")
    print(f"  评测集不变：val.jsonl / heldout.jsonl 沿用 data/ 现成的，没动")
    print(f"  对比时请用它和 data/train_targeted.jsonl 各训一臂（其它超参完全相同）")
    if args.long > 620:
        print("  ⚠️  long_policy 超过去重空间（约 620），实际条数会少于请求值")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""从 train_targeted.jsonl 派生四个消融子集（第一轮卡 8–11 用）。

为什么需要这些：1,250 条里有 450 条是合成数据（长政策 250 + 短场景 200），
占训练量的 36%。不拆开看，就没法判断这些数据是帮忙还是纯噪声。
"tgt_all vs tgt_no_longpolicy" 是唯一能回答这个问题的对照。

四个子集都只做来源过滤，不重新采样、不打乱——这样每个子集都是另一个的
子集关系，比较时不会混入切分差异。

用法：
    python scripts/make_variants.py
产物：
    data/train_no_longpolicy.jsonl   1,000 条（去掉 long_policy_synth）
    data/train_no_anchor.jsonl         950 条（去掉 when2call）
    data/train_drop_only.jsonl         500 条（只留 drop）
    data/train_short_only.jsonl        450 条（long_policy_synth + synthetic_short）
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "train_targeted.jsonl"

VARIANTS = {
    "train_no_longpolicy.jsonl": lambda s: s != "long_policy_synth",
    "train_no_anchor.jsonl": lambda s: s != "when2call",
    "train_drop_only.jsonl": lambda s: s == "drop",
    "train_short_only.jsonl": lambda s: s in ("long_policy_synth", "synthetic_short"),
}


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"FATAL: 找不到 {SRC}，先跑 scripts/build_dataset.py")
    rows = [json.loads(line) for line in SRC.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"源集 {SRC.name}: {len(rows)} 条")

    for name, keep in VARIANTS.items():
        out = [r for r in rows if keep(r.get("source", ""))]
        dst = ROOT / "data" / name
        with dst.open("w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        sources = sorted({r.get("source", "") for r in out})
        print(f"  {name:<28} {len(out):>5} 条  sources={sources}")

    # 子集关系自检：每个变体都必须是源集的子集，否则纵向比较不成立
    ids_all = {r["id"] for r in rows}
    for name in VARIANTS:
        ids = {json.loads(l)["id"] for l in (ROOT / "data" / name).read_text(encoding="utf-8").splitlines() if l.strip()}
        extra = ids - ids_all
        if extra:
            raise SystemExit(f"FATAL: {name} 含 {len(extra)} 个不在源集里的 id")
    print("子集关系自检通过（所有变体 ⊆ train_targeted）")


if __name__ == "__main__":
    main()

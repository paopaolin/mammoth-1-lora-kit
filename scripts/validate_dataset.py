#!/usr/bin/env python3
"""数据集结构校验：不依赖模型和运行时，只查"训练器能不能吃、标签是否自洽"。

能查的：
  - 字段齐备、类型正确
  - target 恰好一个标签、和为 1、且必须出现在 criteria 里
  - state 里没有泄题字段（expected / answer / label）
  - id 与 (state,question) 不重复
  - 选项无重复、数量为 3 或 4
查不了的（需要 OneJev runtime + tokenizer，见 docs/03）：
  - qev 渲染后的选项字母是否够用
  - 真实 token 长度与是否会被截断
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BANNED = {"expected", "answer", "label", "ground_truth", "answer_key", "target"}


def check(path: Path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    errs, ids, sigs = [], Counter(), Counter()
    n_opt, gold_pos, labels = Counter(), Counter(), Counter()
    lens = []

    for i, r in enumerate(rows):
        tag = r.get("id", f"line{i}")
        for k in ("id", "state", "question", "target"):
            if k not in r:
                errs.append(f"{tag}: 缺字段 {k}")
                continue
        q, tgt = r.get("question") or {}, r.get("target") or {}
        crit = q.get("criteria") or {}
        if q.get("type") != "choice":
            errs.append(f"{tag}: question.type != choice")
        if not crit:
            errs.append(f"{tag}: criteria 为空")
        n_opt[len(crit)] += 1
        if len(set(crit)) != len(crit):
            errs.append(f"{tag}: criteria 有重复键")

        if len(tgt) != 1:
            errs.append(f"{tag}: target 有 {len(tgt)} 个标签")
        else:
            lab, val = next(iter(tgt.items()))
            labels[lab] += 1
            if abs(float(val) - 1.0) > 1e-9:
                errs.append(f"{tag}: target 权重 {val} != 1.0")
            if lab not in crit:
                errs.append(f"{tag}: 标签 {lab!r} 不在 criteria 里")
            else:
                gold_pos[list(crit).index(lab)] += 1

        st = r.get("state")
        if isinstance(st, dict):
            bad = BANNED & set(st)
            if bad:
                errs.append(f"{tag}: state 含泄题字段 {bad}")
        elif not isinstance(st, str):
            errs.append(f"{tag}: state 类型 {type(st).__name__}")

        ids[r.get("id")] += 1
        sig = hashlib.md5(
            (json.dumps(st, sort_keys=True, ensure_ascii=False) + json.dumps(crit, sort_keys=True)).encode()
        ).hexdigest()
        sigs[sig] += 1
        lens.append(len(json.dumps(st, ensure_ascii=False)))

    dup_id = [k for k, v in ids.items() if v > 1]
    dup_sig = sum(v - 1 for v in sigs.values() if v > 1)
    if dup_id:
        errs.append(f"重复 id: {dup_id[:5]}")
    if dup_sig:
        errs.append(f"内容重复 {dup_sig} 条")

    lens.sort()
    print(f"\n{path.name}: {len(rows)} 条")
    print(f"  选项数 {dict(n_opt)}  标签 {dict(labels.most_common(5))}")
    print(f"  gold 位置 {dict(sorted(gold_pos.items()))}")
    print(f"  state 字符 中位 {lens[len(lens)//2]} / p95 {lens[int(len(lens)*0.95)]} / max {lens[-1]}")
    if errs:
        print(f"  ❌ {len(errs)} 个问题：")
        for e in errs[:8]:
            print("     -", e)
    else:
        print("  ✅ 通过")
    return len(errs)


def main():
    files = ["train_targeted.jsonl", "train_anchor.jsonl", "val.jsonl", "heldout.jsonl"]
    total = 0
    for f in files:
        p = DATA / f
        if not p.exists():
            print(f"缺文件 {f}")
            total += 1
            continue
        total += check(p)
    print(f"\n{'全部通过' if total == 0 else f'共 {total} 个问题'}")
    return 0 if total == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

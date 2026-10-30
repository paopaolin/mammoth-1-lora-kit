#!/usr/bin/env python3
"""用底座模型给训练集打软标签。

规则（9B 那轮验证过有效）：
  teacher 答对 -> target = α·one-hot + (1-α)·teacher_dist   (α=0.8)
  teacher 答错 -> 保持 one-hot                              (错误分布不可信)

为什么不让 teacher 分布完全接管：那是蒸馏，会把 teacher 的错误一起学进来。
为什么要 α 而不是全软：正确项必须占主导，软标签只负责携带"这条题有多确定"。

注意：27B 在训练集上的正确率大概率高于 9B（9B 只有 62.3%），所以能被软化的
样本比例会更高，这是这一轮相对 9B 的结构性优势。

用法：
  # 先起服务（另开一个终端）
  python serve_baseten.py --model models/mammoth-1 --port 8100
  # 再打标签
  python scripts/make_soft_labels.py --src data/train_targeted.jsonl --out data/train_soft.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
import urllib.request
from collections import Counter
from pathlib import Path

WORK = Path(__file__).resolve().parents[1]


def ask(endpoint: str, state, question, timeout: float = 120.0) -> dict:
    body = {"state": state, "model": "probe", "questions": {"decision": question}}
    req = urllib.request.Request(
        f"{endpoint}/v1/systemone",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(WORK / "data" / "train_targeted.jsonl"))
    ap.add_argument("--out", default=str(WORK / "data" / "train_soft.jsonl"))
    ap.add_argument("--endpoint", default="http://127.0.0.1:8100")
    ap.add_argument("--alpha", type=float, default=0.8)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
    print(f"input {len(rows)} 条 | alpha={args.alpha} | endpoint={args.endpoint}", flush=True)

    out, stats = [], Counter()
    ent_t, ent_s = [], []
    t0 = time.time()

    for i, r in enumerate(rows, 1):
        gold = r["target"]
        gold_key = max(gold, key=gold.get)
        q = r["question"]
        crit = list(q.get("criteria") or {})
        try:
            resp = ask(args.endpoint, r["state"], q)
            ans = resp["answers"]["decision"]
            probs = ans.get("probabilities") or ({ans["noul"]: 1.0} if "noul" in ans else {})
            probs = {str(k): float(v) for k, v in probs.items()}
        except Exception as e:  # noqa: BLE001
            stats["request_failed"] += 1
            if stats["request_failed"] <= 3:
                print(f"  FAIL {r.get('id')}: {type(e).__name__}: {e}", flush=True)
            out.append({**r, "target": gold, "teacher_status": "failed"})
            continue

        cand = [k for k in crit if k in probs] or list(probs)
        if not cand:
            stats["no_overlap"] += 1
            out.append({**r, "target": gold, "teacher_status": "no_overlap"})
            continue

        teacher_key = max(cand, key=lambda k: probs.get(k, 0.0))
        if teacher_key == gold_key:
            soft = {k: args.alpha * (1.0 if k == gold_key else 0.0)
                    + (1 - args.alpha) * probs.get(k, 0.0) for k in crit}
            tot = sum(soft.values())
            soft = {k: round(v / tot, 6) for k, v in soft.items()}
            status = "soft"
            stats["teacher_correct"] += 1
            ent_t.append(-sum(p * math.log(max(p, 1e-12)) for p in probs.values() if p > 0))
            ent_s.append(-sum(p * math.log(max(p, 1e-12)) for p in soft.values() if p > 0))
        else:
            soft, status = gold, "hard_teacher_wrong"
            stats["teacher_wrong"] += 1

        out.append({**r, "target": soft, "teacher_status": status})
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}  {time.time()-t0:.0f}s  "
                  f"correct={stats['teacher_correct']} wrong={stats['teacher_wrong']}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(out)
    print("\n=== 结果 ===")
    print(f"total {n} | teacher 答对 {stats['teacher_correct']} "
          f"({stats['teacher_correct']/max(1,n):.1%}) | 答错 {stats['teacher_wrong']} | "
          f"失败 {stats['request_failed']}")
    print(f"写成 {args.out}")
    if ent_t:
        s = sorted(ent_t)
        print(f"teacher 分布熵 中位 {s[len(s)//2]:.4f} | 软标签熵 中位 "
              f"{sorted(ent_s)[len(ent_s)//2]:.4f}")
        print("熵太低说明 teacher 过于自信，软标签携带的信息有限——这是 9B 那轮的实际发现")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""汇总所有 run 的结果，输出一张可直接贴进报告的表。

accuracy / brier / ece 读 `summary.json`（jevbench summarize 的输出），
覆盖率 / 采纳精度 / 高置信错误数从 `results.jsonl` 的 probs 现算——
后三项 summary 里没有，而这恰好是判断路由能不能用的三个数。

用法：
    python scripts/collect_summaries.py 'results/*/summary.json'
    python scripts/collect_summaries.py 'results/*/summary.json' --thresholds 0.8,0.9,0.95
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def route_metrics(results_path: Path, thresholds: list[float]) -> dict:
    """confident = max(prob) >= t；采纳精度 = 被采纳的那些里答对的比例。"""
    rows = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # 只统计真正进模型的题：status 不是 ok 的行 probs 可能是空的
            if not r.get("ok") or not r.get("probs"):
                continue
            conf = max(r["probs"].values())
            rows.append((conf, bool(r.get("correct"))))
    if not rows:
        return {}

    out = {}
    for t in thresholds:
        adopted = [c for c, _ in rows if c >= t]
        hit = [1 for c, ok in rows if c >= t and ok]
        miss = [1 for c, ok in rows if c >= t and not ok]
        out[t] = {
            "coverage": len(adopted) / len(rows),
            "precision": (len(hit) / len(adopted)) if adopted else 0.0,
            "n high-conf error": len(miss),
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pattern", nargs="?", default="results/*/summary.json")
    ap.add_argument("--thresholds", default="0.9")
    args = ap.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",")]
    paths = sorted(glob.glob(args.pattern, recursive=True))
    if not paths:
        sys.exit(f"FATAL: {args.pattern} 没匹配到任何 summary.json。先跑一轮评测。")

    records = []
    for p in paths:
        d = Path(p).parent
        try:
            s = json.loads(Path(p).read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[跳过] {d.name}: summary 读不了 ({e})", file=sys.stderr)
            continue
        rec = {
            "name": d.name,
            "n": s.get("n_scorable"),
            "acc": s.get("accuracy"),
            "correct": s.get("n_correct"),
            "brier": s.get("brier_mean"),
            "ece": (s.get("ece") or {}).get("ece"),
        }
        rp = d / "results.jsonl"
        if rp.exists():
            rec.update(route_metrics(rp, thresholds))
        records.append(rec)

    if not records:
        sys.exit("FATAL: 没有任何可用的 summary.json")

    t_main = thresholds[0]
    records.sort(key=lambda r: (r.get("acc") or 0), reverse=True)

    head = ["run", "n", "acc", "corr", "brier", "ECE", f"cov@{t_main}", f"prec@{t_main}", "hi-conf err"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for r in records:
        rm = r.get(t_main, {})
        def fmt(v, spec=".4f"):
            return f"{v:{spec}}" if isinstance(v, (int, float)) else "-"
        print(
            f"| {r['name']} | {r['n'] or '-'} | {fmt(r.get('acc'))} | {r['correct'] or '-'} | "
            f"{fmt(r.get('brier'))} | {fmt(r.get('ece'))} | "
            f"{fmt(rm.get('coverage'), '.3f')} | {fmt(rm.get('precision'), '.3f')} | "
            f"{rm.get('n high-conf error', '-')} |"
        )

    if len(thresholds) > 1:
        print(f"\n其它阈值的采纳情况（{len(thresholds)} 档）：")
        for r in records:
            if t_main not in r:
                continue
            cells = "  ".join(
                f"@{t}: cov {r[t]['coverage']:.3f} prec {r[t]['precision']:.3f}"
                for t in thresholds if t in r
            )
            print(f"  {r['name']:<28} {cells}")


if __name__ == "__main__":
    main()

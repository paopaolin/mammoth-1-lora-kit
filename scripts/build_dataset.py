#!/usr/bin/env python3
"""构建 27B（mammoth-1）用的决策微调数据集。

设计依据：mammoth-1 在 JevBench public-231 上错 14 条，其中
  temporal_numeric 7 条、long_policy 3 条、multi_hop 2 条、ambiguous 1、judge_hard 1。
所以数据只投两类：
  1) temporal_numeric —— DROP 的数值/日期推理，按"需要算"筛，干扰项取同量级数字
  2) anchor（防遗忘）—— When2Call 的工具决策三选一，27B 本来全对，只用来锚住行为

不投 onejev / long_policy：前者本地无源且 27B 在那一族只错 1 条；
后者没有可用的公开同分布数据，硬造会把噪声当信号（见 docs/01 的"不做什么"）。

输出（OneJev train/sft.py 原生格式）：
  data/train_targeted.jsonl   1,000 条（temporal 700 + anchor 300）
  data/train_anchor.jsonl       300 条（对照组，只用 anchor）
  data/val.jsonl                150 条
  data/heldout.jsonl            150 条
  data/stats.json
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "_raw"
OUT = ROOT / "data"
SEED = 42

N_DROP_TRAIN = 500
N_SYNTH_TRAIN = 200
N_LONG_TRAIN = 250
N_ANCHOR_TRAIN = 300
N_VAL = 150
N_HELDOUT = 150

# 需要"算一步"的问句特征：这些词出现时，答案通常不能直接摘抄
ARITH = re.compile(
    r"\b(how many more|how many less|how much more|how much less|difference|total|"
    r"combined|altogether|in all|average|per |each |times|twice|double|half|"
    r"before|after|earlier|later|since|until|between)\b", re.I)
DATE = re.compile(r"\b(january|february|march|april|may|june|july|august|september|"
                  r"october|november|december)\b", re.I)
NUM = re.compile(r"-?\d[\d,]*\.?\d*")


# ---------------------------------------------------------------- 数值工具

def to_num(s: str):
    t = s.strip().replace(",", "").replace("$", "").replace("%", "")
    t = t.replace("\u2212", "-")
    try:
        v = float(t)
    except ValueError:
        return None
    return v if abs(v) < 1e7 else None


def fmt(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def numbers_in(text: str) -> list[float]:
    out = []
    for m in NUM.finditer(text):
        v = to_num(m.group())
        if v is not None and abs(v) > 0:
            out.append(v)
    return out


def make_numeric_distractors(answer: str, passage: str, rng: random.Random, k: int = 3):
    """优先用同段材料里同量级的数字（真实混淆源），不够再补计算近似错误。"""
    gold = to_num(answer)
    if gold is None:
        return None
    pool = []
    for v in numbers_in(passage):
        if abs(v - gold) < 1e-9:
            continue
        ratio = v / gold if gold else 0
        if 0.35 <= ratio <= 2.6 and fmt(v) != fmt(gold):
            pool.append(fmt(v))
    rng.shuffle(pool)
    picked = list(dict.fromkeys(pool))[:k]

    for cand in (gold + 1, gold - 1, gold * 2, gold / 2, gold + 10, gold - 10):
        if len(picked) >= k:
            break
        s = fmt(cand)
        if cand > 0 and s != fmt(gold) and s not in picked:
            picked.append(s)
    if len(picked) < k:
        return None
    return picked[:k]


# ---------------------------------------------------------------- 短场景合成（对齐评测分布）

from datetime import datetime, timedelta

SHORT_TEMPLATES = ("deadline", "duration", "discount", "boundary")


def _dt_fmt(d: datetime) -> str:
    return d.strftime("%Y-%m-%d %H:%M")


def _near_times(d: datetime, deltas):
    """围绕正确答案生成近似错误项（差 1 小时 / 1 天 / 2 天），保证互不相同。"""
    out = []
    for delta in deltas:
        cand = d + delta
        s = _dt_fmt(cand)
        if s != _dt_fmt(d) and s not in out:
            out.append(s)
    return out


def synth_one(kind: str, rng: random.Random):
    """返回 (state_text, correct, [3 个干扰项]) 或 None。标签全部由程序计算。"""
    if kind == "deadline":
        base = datetime(2026, 1, 1) + timedelta(days=rng.randint(0, 300),
                                                hours=rng.choice([0, 3, 9, 14, 20]))
        hours = rng.choice([24, 48, 60, 72, 96, 120, 168])
        due = base + timedelta(hours=hours)
        state = (f"订单于 {_dt_fmt(base)} 提交，规则是必须在提交后 {hours} 小时内完成支付，"
                 f"超时订单自动取消。")
        q = "这笔订单的最晚支付时间是什么时候？"
        return state, q, _dt_fmt(due), _near_times(due, [timedelta(hours=1),
                                                        timedelta(days=1),
                                                        timedelta(days=-1)])

    if kind == "duration":
        h1, m1 = rng.randint(0, 5), rng.choice([5, 15, 25, 35, 45, 50])
        h2, m2 = rng.randint(0, 5), rng.choice([10, 20, 30, 40, 55])
        total = timedelta(hours=h1, minutes=m1) + timedelta(hours=h2, minutes=m2)
        th, tm = total.seconds // 3600, (total.seconds % 3600) // 60
        state = (f"第一段行程耗时 {h1} 小时 {m1} 分，中途转乘等待 {h2} 小时 {m2} 分。")
        q = "全程共耗时多久？"
        correct = f"{th} 小时 {tm} 分"
        alts = []
        for dh, dm in ((0, 10), (0, -10), (1, 0), (-1, 0)):
            mm = tm + dm
            hh = th + dh + (1 if mm >= 60 else (-1 if mm < 0 else 0))
            mm %= 60
            s = f"{hh} 小时 {mm} 分"
            if s != correct and s not in alts and hh >= 0:
                alts.append(s)
        return (state, q, correct, alts) if len(alts) >= 3 else None

    if kind == "discount":
        price = rng.randrange(120, 4800, 5)
        pct = rng.choice([5, 8, 10, 12, 15, 20, 25])
        final = round(price * (100 - pct) / 100, 2)
        state = (f"商品标价 {price} 元，会员可享 {pct}% 折扣，折扣按标价直接扣减，"
                 f"不与其他优惠叠加。")
        q = "会员实际需要支付多少元？"
        correct = fmt(final)
        alts = []
        for cand in (round(price * pct / 100, 2), round(final * 1.05, 2),
                     round(final * 0.95, 2), round(price - pct, 2)):
            s = fmt(cand)
            if s != correct and s not in alts and cand > 0:
                alts.append(s)
        return (state, q, correct, alts) if len(alts) >= 3 else None

    if kind == "boundary":
        limit = rng.choice([24, 48, 72, 120])
        edge = rng.choice([0, 0, 0, 1, -1])   # 多数落在边界上，考"不超过"是否含等于
        base = datetime(2026, 3, 1) + timedelta(days=rng.randint(0, 120),
                                                hours=rng.randint(0, 23))
        req = base + timedelta(hours=limit + edge)
        state = (f"政策规定：自签收时间起不超过 {limit} 小时的申请才可受理。"
                 f"签收时间为 {_dt_fmt(base)}，提交申请时间为 {_dt_fmt(req)}。")
        q = "该申请是否应当受理？"
        correct = "应当受理" if edge <= 0 else "不应受理"
        other = "不应受理" if correct == "应当受理" else "应当受理"
        return state, q, correct, [other, "信息不足，无法判断", "需要上级审批"]


def build_synthetic(rng: random.Random, n: int):
    rows, seen = [], set()
    tries = 0
    while len(rows) < n and tries < n * 40:
        tries += 1
        kind = SHORT_TEMPLATES[len(rows) % len(SHORT_TEMPLATES)]
        got = synth_one(kind, rng)
        if not got:
            continue
        state, q, correct, alts = got
        key = (state, q)
        if key in seen:
            continue
        seen.add(key)
        rows.append({"kind": kind, "state_text": state, "question": q,
                     "answer": correct, "alts": list(dict.fromkeys(alts))[:3]})
    print(f"短场景合成: {len(rows)} 条 | 模板分布 {dict(Counter(r['kind'] for r in rows))}")
    return rows


def make_synth_task(r, idx, rng):
    opts = [r["answer"]] + r["alts"]
    rng.shuffle(opts)
    return {
        "id": f"sy-{idx:04d}",
        "state": {"scenario": r["state_text"], "question": r["question"]},
        "question": {
            "type": "choice",
            "instructions": "根据材料作答，只依据文中给出的条件与数值。",
            "criteria": {o: "" for o in opts},
        },
        "target": {r["answer"]: 1.0},
        "source": "synthetic_short",
        "split": "train",
        "meta": {"template": r["kind"]},
    }


# ---------------------------------------------------------------- DROP

def build_temporal(rng: random.Random):
    rows = []
    for fn, split in (("drop_train.parquet", "train"), ("drop_val.parquet", "val")):
        t = pq.read_table(RAW / fn)
        for r in t.to_pylist():
            spans = (r.get("answers_spans") or {}).get("spans") or []
            types = (r.get("answers_spans") or {}).get("types") or []
            if len(spans) != 1 or not types:
                continue
            ans, typ = spans[0].strip(), types[0].lower()
            if typ not in ("number", "date") or to_num(ans) is None:
                continue
            q, psg = r["question"].strip(), r["passage"].strip()
            if not q or not psg:
                continue
            # 难度打分：需要一步算术 > 同段有多个同量级数字 > 其他
            score = 0
            if ARITH.search(q):
                score += 2
            if DATE.search(psg) or typ == "date":
                score += 1
            nums = [v for v in numbers_in(psg)
                    if to_num(ans) and 0.35 <= v / to_num(ans) <= 2.6]
            if len(nums) >= 3:
                score += 2
            elif len(nums) >= 1:
                score += 1
            rows.append({"split": split, "section": r["section_id"], "qid": r["query_id"],
                         "passage": psg, "question": q, "answer": ans, "type": typ,
                         "score": score})
    print(f"DROP 可用数值/日期题: {len(rows)}（含算术线索 {sum(1 for r in rows if r['score'] >= 4)} 条）")
    # 优先高分，再按 section 去重，保证多样性
    rows.sort(key=lambda r: (-r["score"], hashlib.md5(r["qid"].encode()).hexdigest()))
    seen_section, picked = Counter(), []
    for r in rows:
        if seen_section[r["section"]] >= 3:      # 同一段材料最多出 3 题，避免过拟合单篇
            continue
        opts = make_numeric_distractors(r["answer"], r["passage"], rng)
        if not opts:
            continue
        seen_section[r["section"]] += 1
        r["options"] = opts
        picked.append(r)
    print(f"  构造出 4 选 1 的: {len(picked)} 条")

    # 同 passage + 同 question 若答案不一致，说明 DROP 内部本身有冲突，整组丢弃
    by_q = defaultdict(set)
    for r in picked:
        by_q[(r["passage"], r["question"])].add(r["answer"])
    conflict = {k for k, v in by_q.items() if len(v) > 1}
    seen_q, dedup = set(), []
    for r in picked:
        k = (r["passage"], r["question"])
        if k in conflict or k in seen_q:
            continue
        seen_q.add(k)
        dedup.append(r)
    print(f"  去重/去冲突后: {len(dedup)} 条（冲突题 {len(conflict)} 组）")
    return dedup


# ---------------------------------------------------------------- When2Call

def w2c_label(text: str) -> str | None:
    t = text or ""
    if "<TOOLCALL>" in t:
        return "call_tool"
    low = t.lower()
    if t.strip().endswith("?") or any(k in low for k in
                                      ("which ", "could you", "can you", "please provide",
                                       "what is your", "do you want", "would you like")):
        return "request_for_info"
    if t.strip():
        return "cannot_answer"
    return None


def build_anchor(rng: random.Random):
    rows = []
    with (RAW / "when2call_pref.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            tools = r.get("tools") or []
            msgs = r.get("messages") or []
            if not tools or not msgs:
                continue
            user = next((m["content"] for m in msgs if m.get("role") == "user"), None)
            if not user:
                continue
            chosen = (r.get("chosen_response") or {}).get("content", "")
            rejected = (r.get("rejected_response") or {}).get("content", "")
            lc, lr = w2c_label(chosen), w2c_label(rejected)
            if not lc or not lr or lc == lr:
                continue
            schema = "\n".join(tools)
            if len(schema) > 6000:
                continue
            tg = hashlib.md5("|".join(sorted(
                str(json.loads(t).get("name", "")) for t in tools)).encode()).hexdigest()[:12]
            rows.append({"user": user.strip(), "tools": [json.loads(t) for t in tools],
                         "label": lc, "schema_len": len(schema), "tgroup": tg})
    print(f"When2Call 可用（初筛）: {len(rows)} 条 | "
          f"标签分布 {dict(Counter(r['label'] for r in rows))}")

    # 同一请求 + 同一套工具出现多个不同标签 → 标签歧义，整组丢弃。
    # 这在偏好数据里很常见：同一 prompt 有多个不同的"选中回复"。
    labs = defaultdict(set)
    for r in rows:
        labs[(r["user"], r["tgroup"])].add(r["label"])
    ambiguous = {k for k, v in labs.items() if len(v) > 1}
    before = len(rows)
    rows = [r for r in rows if (r["user"], r["tgroup"]) not in ambiguous]

    # 同请求同标签只保留一条
    seen, dedup = set(), []
    for r in rows:
        k = (r["user"], r["tgroup"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    print(f"  剔除标签歧义 {before - len(rows)} 条（{len(ambiguous)} 个请求），"
          f"去重后 {len(dedup)} 条 | {dict(Counter(r['label'] for r in dedup))}")
    rng.shuffle(dedup)
    return dedup


# ---------------------------------------------------------------- 组装

def make_drop_task(r, idx, rng):
    opts = [r["answer"]] + r["options"]
    rng.shuffle(opts)
    return {
        "id": f"tn-{idx:04d}",
        "state": {"passage": r["passage"], "question": r["question"]},
        "question": {
            "type": "choice",
            "instructions": "Answer the question using only the passage. "
                            "Pick the option that matches exactly.",
            "criteria": {o: "" for o in opts},
        },
        "target": {r["answer"]: 1.0},
        "source": "drop",
        "split": r["split"],
        "meta": {"origin_section": r["section"], "difficulty_score": r["score"],
                 "answer_type": r["type"]},
    }


def make_anchor_task(r, idx, rng):
    labels = ["call_tool", "request_for_info", "cannot_answer"]
    rng.shuffle(labels)
    return {
        "id": f"an-{idx:04d}",
        "state": {"user_request": r["user"], "available_tools": r["tools"]},
        "question": {
            "type": "choice",
            "instructions": "Decide what the assistant should do next, given the user "
                            "request and the available tools.",
            "criteria": {o: "" for o in labels},
        },
        "target": {r["label"]: 1.0},
        "source": "when2call",
        "split": "train",
        "meta": {"schema_chars": r["schema_len"]},
    }


def split_grouped(items, key, quotas, seed=SEED):
    """按 group 切分：同一 group（同段材料 / 同一套工具）只落在一个 split。

    分配顺序按"当前缺口最大"的 split 优先，避免大 group 把某一边瞬间填满。
    """
    groups = defaultdict(list)
    for it in items:
        groups[it[key]].append(it)
    keys = list(groups)
    random.Random(seed).shuffle(keys)
    out = {n: [] for n in quotas}
    for k in keys:
        g = groups[k]
        name = max(quotas, key=lambda n: quotas[n] - len(out[n]))
        if len(out[name]) < quotas[name]:
            out[name].extend(g)
    return {n: v[:quotas[n]] for n, v in out.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    temporal = build_temporal(rng)
    anchor = build_anchor(rng)
    for r in anchor:   # 按工具签名分组，防止同一套工具跨 split 泄漏
        r["tgroup"] = hashlib.md5(
            "|".join(sorted(str(t.get("name", "")) for t in r["tools"])).encode()
        ).hexdigest()[:12]
    synth = build_synthetic(rng, N_SYNTH_TRAIN + 50)

    import build_long_policy as blp
    lp_items = blp.make_items(N_LONG_TRAIN + 50)
    lp_split = split_grouped(lp_items, "case", {"train": N_LONG_TRAIN, "val": 25, "heldout": 25})
    l_train = lp_split["train"]
    l_val = lp_split["val"]
    l_held = lp_split["heldout"]

    t_split = split_grouped(temporal, "section", {"train": N_DROP_TRAIN, "val": 75, "heldout": 75})
    a_split = split_grouped(anchor, "tgroup", {"train": N_ANCHOR_TRAIN, "val": 50, "heldout": 50})
    s_split = split_grouped(synth, "state_text", {"train": N_SYNTH_TRAIN, "val": 25, "heldout": 25})

    t_train, t_val, t_held = t_split["train"], t_split["val"], t_split["heldout"]
    a_train, a_val, a_held = a_split["train"], a_split["val"], a_split["heldout"]
    y_train, y_val, y_held = s_split["train"], s_split["val"], s_split["heldout"]

    for r, sp in ((t_train, "train"), (t_val, "val"), (t_held, "heldout")):
        for x in r:
            x["split"] = sp

    targeted = ([make_drop_task(r, i, rng) for i, r in enumerate(t_train)]
                + [make_synth_task(r, i, rng) for i, r in enumerate(y_train)]
                + [blp.to_task(r, i, rng) for i, r in enumerate(l_train)]
                + [make_anchor_task(r, i, rng) for i, r in enumerate(a_train)])
    rng.shuffle(targeted)

    val_tasks = ([make_drop_task(r, i, rng) for i, r in enumerate(t_val)]
                 + [make_synth_task(r, i + 3000, rng) for i, r in enumerate(y_val)]
                 + [blp.to_task(r, i + 4000, rng) for i, r in enumerate(l_val)]
                 + [make_anchor_task(r, i + 5000, rng) for i, r in enumerate(a_val)])
    held_tasks = ([make_drop_task(r, i + 9000, rng) for i, r in enumerate(t_held)]
                  + [make_synth_task(r, i + 9300, rng) for i, r in enumerate(y_held)]
                  + [blp.to_task(r, i + 9400, rng) for i, r in enumerate(l_held)]
                  + [make_anchor_task(r, i + 9500, rng) for i, r in enumerate(a_held)])

    def dump(name, rows):
        p = OUT / name
        with p.open("w", encoding="utf-8") as f:
            for x in rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  {name:<24} {len(rows):>5} 行  {p.stat().st_size/1e6:.2f} MB")
        return len(rows)

    print("\n写出：")
    n_t = dump("train_targeted.jsonl", targeted)
    anchor_tasks = [make_anchor_task(r, i, random.Random(SEED + i))
                    for i, r in enumerate(a_train)]
    n_a = dump("train_anchor.jsonl", anchor_tasks)
    n_v = dump("val.jsonl", val_tasks)
    n_h = dump("heldout.jsonl", held_tasks)

    # 统计
    def stats(rows, tag):
        lens = [len(json.dumps(t["state"], ensure_ascii=False)) for t in rows]
        return {
            "n": len(rows),
            "source": dict(Counter(t["source"] for t in rows)),
            "n_options": dict(Counter(len(t["question"]["criteria"]) for t in rows)),
            "gold_position": dict(Counter(
                list(t["question"]["criteria"]).index(next(iter(t["target"]))) for t in rows)),
            "state_chars": {
                "mean": round(sum(lens) / len(lens), 1),
                "median": sorted(lens)[len(lens) // 2],
                "p95": sorted(lens)[int(len(lens) * 0.95)],
                "max": max(lens),
            },
        }

    all_train = targeted
    meta = {
        "_seed": SEED,
        "_note": "targeted = DROP temporal_numeric + When2Call anchor; "
                 "26B/27B 的错题集中在这两族",
        "train_targeted": stats(all_train, "train"),
        "train_anchor": stats(anchor_tasks, "anchor"),
        "val": stats(val_tasks, "val"),
        "heldout": stats(held_tasks, "heldout"),
        "counts": {"train_targeted": n_t, "train_anchor": n_a, "val": n_v, "heldout": n_h},
        "by_source": {
            "drop_candidates": len(temporal),
            "when2call_candidates": len(anchor),
        },
    }
    (OUT / "stats.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print("\n统计：")
    for k in ("train_targeted", "val", "heldout"):
        s = meta[k]
        print(f"  {k:<16} n={s['n']:<5} 来源={s['source']} "
              f"state 中位={s['state_chars']['median']} 字符")


if __name__ == "__main__":
    main()

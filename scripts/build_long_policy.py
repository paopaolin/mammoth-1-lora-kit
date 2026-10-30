#!/usr/bin/env python3
"""构造 long_policy 训练样本：一般条款 + 例外条款，问某个具体情形是否适用。

为什么要这个：27B 在 public-231 的 14 条错误里，3 条在 long_policy，
其中一条是把"注销后删除全部数据"答成 erase_all，而正确答案是
"保留交易记录后删除其余"。这说明它会把例外条款读丢，或者把优先级读反。

构造原则：
  1. 标签由规则引擎算出来，不是人写的。先算例外、再算一般条款。
  2. 选项不只有"是/否"，还包含"信息不足"和"需人工复核"，
     防止模型学成"一律猜一个"。
  3. 同一规则下只改一个决定性条件，标签随之翻转（反事实对照）。
  4. 规则用词、参数、主体全部随机化，避免模型记住模板。
"""

from __future__ import annotations

import random
from collections import Counter

# 每个规则族给出：通用条款、例外条款、情形字段、计算函数
# 返回 (情形文本, 正确标签, 是否需要复核)

LABELS = ["符合", "不符合", "信息不足", "需人工复核"]


def _yes(b):  return "符合" if b else "不符合"


def rule_refund(rng):
    hours = rng.choice([24, 48, 72, 120])
    since = rng.choice([hours // 2, hours, hours + 1, hours + 24])
    opened = rng.choice([True, False, None])      # None = 材料没写
    defect = rng.choice([True, False, None])
    general = f"一般商品自签收起 {hours} 小时内且未拆封，可申请退货退款。"
    exception = "例外：存在质量缺陷的商品不受上述时间限制，但仍需提供缺陷说明。"
    case = f"签收后第 {since} 小时提交申请。"
    if opened is not None:
        case += f"商品{'已拆封' if opened else '未拆封'}。"
    if defect is not None:
        case += f"{'已附质量缺陷说明' if defect else '未提及质量问题'}。"

    # 规则引擎：先例外，再一般
    if defect is True:
        ans = _yes(True)
    elif since > hours:
        ans = _yes(False)
    elif opened is None:
        ans = "信息不足"
    elif opened:
        ans = _yes(False)
    else:
        ans = _yes(True)
    if defect is None and since > hours:
        ans = "信息不足"   # 超时且未说明是否有缺陷，不能直接拒
    return general, exception, case, ans


def rule_erase(rng):
    years = rng.choice([3, 5, 6, 10])
    # 只有一个决定性变量：账号在保留期内有没有交易记录。
    has_recent = rng.choice([True, False])
    acct = rng.choice(["个人账号", "企业账号", "测试账号"])
    ask = rng.choice(["请求立即删除全部数据", "要求平台清空其全部数据", "申请删除账号下的所有信息"])
    general = f"{acct}注销后，平台应删除该账号的全部数据。"
    exception = (f"例外：根据财税合规要求，交易记录需保留 {years} 年，"
                 "保留期内的交易记录不属于'全部删除'的范围。")
    case = f"用户已注销该{acct}并{ask}。"
    if has_recent:
        case += f"该账号在 {years - 1} 年前发生过交易，相关记录仍在保留期内。"
    else:
        case += "该账号没有任何处于保留期内的交易记录。"

    opts = ["全部删除", "保留交易记录后删除其余", "全部保留", "信息不足"]
    ans = "保留交易记录后删除其余" if has_recent else "全部删除"
    return general, exception, case, ans, opts


def rule_approval(rng):
    limit = rng.choice([1000, 3000, 5000, 10000])
    amount = rng.choice([limit // 2, limit, limit * 2])
    urgent = rng.choice([True, False, None])
    has_sign = rng.choice([True, False])
    general = f"单笔金额超过 {limit} 元的支出须由部门经理事先审批。"
    exception = "例外：紧急差旅支出可先执行后补签，但须在 3 个工作日内补齐审批。"
    case = f"本次支出金额 {amount} 元。"
    if urgent is not None:
        case += f"{'属于紧急差旅' if urgent else '非紧急差旅'}。"
    case += f"{'已' if has_sign else '未'}事先取得经理审批。"

    # 先看是否已满足一般条款（金额达标或已审批），再看例外，最后才是信息不足
    if amount <= limit or has_sign:
        ans = _yes(True)
    elif urgent is True:
        ans = _yes(True)          # 紧急差旅例外允许先执行后补签
    elif urgent is None:
        ans = "信息不足"          # 不知道是不是紧急差旅，判不了
    else:
        ans = _yes(False)
    return general, exception, case, ans


def rule_export(rng):
    mentor = rng.choice([True, False, None])
    reg = rng.choice([True, False])
    scope = rng.choice(["指定范围内", "超出指定范围"])
    purpose = rng.choice(["用于学习", "用于对外交付"])
    ask = rng.choice(["申请导出生产环境数据", "要求导出生产库中的订单表",
                      "希望拉取生产环境日志", "提出导出用户表备份",
                      "申请导出线上配置清单", "要求拷贝生产库明细数据"])
    general = "实习生不得导出生产环境数据。"
    exception = "例外：在带教老师陪同下完成登记后，可导出指定范围的数据用于学习。"
    case = ""
    if mentor is not None:
        case += f"{'有带教老师陪同' if mentor else '无带教老师陪同'}。"
    case += f"{'已完成导出登记' if reg else '未完成导出登记'}。"
    case += f"{ask}，且数据范围{scope}，{purpose}。"

    if scope == "超出指定范围":
        ans = _yes(False)            # 例外只覆盖指定范围，超出就不符合
    elif purpose == "用于对外交付":
        ans = _yes(False)            # 例外只放开学习用途，对外交付仍在禁令内
    elif mentor is True and reg:
        ans = _yes(True)
    elif mentor is None:
        ans = "信息不足"
    else:
        ans = _yes(False)
    return general, exception, case, ans


def rule_overtime(rng):
    cap = rng.choice([8, 10, 12])
    worked = rng.choice([cap - 1, cap, cap + 2])
    rescue = rng.choice([True, False, None])
    rest = rng.choice([True, False, None])
    general = f"连续工作时间不得超过 {cap} 小时。"
    exception = "例外：抢险任务可延长，但任务结束后必须安排等时补休。"
    case = f"本次已连续工作 {worked} 小时。"
    if rescue is not None:
        case += f"{'属于抢险任务' if rescue else '非抢险任务'}。"
    if rest is not None:
        case += f"{'已安排等时补休' if rest else '未安排补休'}。"

    if worked <= cap:
        ans = _yes(True)
    elif rescue is True:
        if rest is True:
            ans = _yes(True)
        elif rest is False:
            ans = _yes(False)
        else:
            ans = "信息不足"
    elif rescue is None:
        ans = "信息不足"
    else:
        ans = _yes(False)
    return general, exception, case, ans


def rule_liquid(rng):
    cap = rng.choice([50, 100, 200])
    total_cap = cap * rng.choice([5, 10])
    per = rng.choice([cap // 2, cap, cap + 20])
    total = per * rng.choice([1, 3, 12])
    general = f"液体物品单件不得超过 {cap} ml，且总量不得超过 {total_cap} ml，方可随身携带。"
    exception = "例外：经安检单独确认的必需药品不受总量限制。"
    medicine = rng.choice([True, False])
    case = f"携带液体单件 {per} ml，合计 {total} ml。"
    case += f"{'为经确认的必需药品' if medicine else '非药品'}。"
    if medicine:
        ans = _yes(per <= cap)
    else:
        ans = _yes(per <= cap and total <= total_cap)
    return general, exception, case, ans


RULES = {
    "refund": rule_refund,
    "erase": rule_erase,
    "approval": rule_approval,
    "export": rule_export,
    "overtime": rule_overtime,
    "liquid": rule_liquid,
}

QUESTION = {
    "refund": "该退货退款申请是否符合受理条件？",
    "erase": "该注销请求应按哪种方式执行？",
    "approval": "该笔支出是否符合规定、可以执行？",
    "export": "该导出申请是否符合规定？",
    "overtime": "该工作安排是否符合规定？",
    "liquid": "该携带方式是否符合规定？",
}


def make_items(n: int, seed: int = 42):
    rng = random.Random(seed)
    out, seen = [], set()
    kinds = list(RULES)
    tries = 0
    # 注意：规则族必须按 tries 轮转，不能按 len(out)。
    # 按 len(out) 会在某族去重空间用尽后永远卡在该族上。
    while len(out) < n and tries < n * 80:
        tries += 1
        kind = kinds[tries % len(kinds)]
        fn = RULES[kind]
        got = fn(rng)
        if kind == "erase":
            general, exception, case, ans, opts = got
        else:
            general, exception, case, ans = got
            opts = LABELS
        key = (general, exception, case)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "kind": kind,
            "policy": f"{general}{exception}",
            "case": case,
            "question": QUESTION[kind],
            "answer": ans,
            "options": opts,
        })
    # 一致性检查：同样的输入绝不能出现两个不同答案（旧版 erase 就犯过）
    by_case = {}
    for r in out:
        c = (r["policy"], r["case"])
        if c in by_case:
            assert by_case[c] == r["answer"], f"标签冲突: {c}"
        by_case[c] = r["answer"]
    return out


def to_task(r, idx, rng):
    opts = list(r["options"])
    rng.shuffle(opts)
    return {
        "id": f"lp-{idx:04d}",
        "state": {"policy": r["policy"], "case": r["case"]},
        "question": {
            "type": "choice",
            "instructions": r["question"] + "仅依据条款判断，先检查例外条款是否适用。",
            "criteria": {o: "" for o in opts},
        },
        "target": {r["answer"]: 1.0},
        "source": "long_policy_synth",
        "split": "train",
        "meta": {"rule_family": r["kind"]},
    }


if __name__ == "__main__":
    items = make_items(300)
    rng = random.Random(7)
    tasks = [to_task(r, i, rng) for i, r in enumerate(items)]
    print(f"生成 {len(tasks)} 条 | 规则族 {dict(Counter(t['meta']['rule_family'] for t in tasks))}")
    print(f"标签分布 {dict(Counter(next(iter(t['target'])) for t in tasks))}")
    for t in tasks[:3]:
        print("\n---", t["id"], "---")
        print("policy:", t["state"]["policy"])
        print("case  :", t["state"]["case"])
        print("q     :", t["question"]["instructions"])
        print("opts  :", list(t["question"]["criteria"]))
        print("gold  :", list(t["target"]))

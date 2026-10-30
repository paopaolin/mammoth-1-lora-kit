# 数据卡

## 1 概要

| 项 | 值 |
|---|---|
| 文件 | `data/` 下 4 个 jsonl，共 3.1MB |
| 总条数 | 1,600（训练 1,000 + 对照 300 + 验证 150 + 留出 150） |
| 格式 | OneJev `train/sft.py` 原生格式，一行一题 |
| 语言 | 中文（合成题）/ 英文（DROP、When2Call） |
| 构造脚本 | `scripts/build_dataset.py`（可完整复现，种子 42） |
| 结构校验 | `scripts/validate_dataset.py`（全部通过） |

## 2 字段

```json
{
  "id": "tn-0001",
  "state": {"passage": "...", "question": "..."},
  "question": {
    "type": "choice",
    "instructions": "Answer the question using only the passage...",
    "criteria": {"29": "", "26": "", "23": "", "50": ""}
  },
  "target": {"29": 1.0},
  "source": "drop",
  "split": "train",
  "meta": {"origin_section": "nfl_2201", "difficulty_score": 5, "answer_type": "number"}
}
```

`target` 里只有正确项为 1.0，其余不出现（不是全部列 0，训练器按 criteria 顺序展开）。
`meta` 不参与训练，仅供排查。

## 3 三个来源

### 3.1 DROP（500 条训练）

- 仓库：`ucinlp/drop`，许可 **CC-BY-4.0**
- 原始 77,400 条（train）+ 9,536（validation）
- 筛选链：单答案 → 答案是数字/日期 → 构造出 4 个互异选项 → 去重去冲突
- 最终可用 **15,088 条**，按难度打分取前 500

难度打分规则（0–5 分）：问句出现 `more/less/total/difference/before/after/since/until`
等需要推算的词 +2；同段落有 ≥3 个与答案同量级（0.35×–2.6×）的数字 +2，≥1 个 +1；
答案或段落含日期 +1。

**干扰项不是随机数**：优先从同一段材料里取同量级数字（真实混淆源），不够才补
±1、×2、÷2 这类计算近似错误。抽取不到 3 个合格干扰项的题直接丢弃。

### 3.2 短场景合成（200 条训练）

四类模板，标签全部由程序计算，不是人工标注：

| 模板 | 考什么 | 标签怎么来 |
|---|---|---|
| `deadline` | 起点 + N 小时 = 到期时刻 | `datetime + timedelta` |
| `duration` | 两段时长相加，跨 60 分钟进位 | `timedelta` 相加再取整 |
| `discount` | 标价 × (1 − 折扣率)，含四舍五入 | `Decimal` 式浮点 + `round(2)` |
| `boundary` | "不超过 N 小时"是否含等于 | 由 `edge ∈ {-1,0,1}` 决定，0 和 −1 判"受理" |

`boundary` 一类刻意让 3/5 的样本落在边界上，考的就是"不超过"是否含等于——
这是 JevBench temporal_numeric 里高频出现的陷阱。

干扰项：时间类给 ±1 小时 / ±1 天；金额类给"折扣额本身""再打 95 折"等真实算错的结果；
判定类给"应当受理 / 不应受理 / 信息不足 / 需要审批"四选一。

### 3.3 When2Call（300 条训练 + 300 条对照）

- 仓库：`nvidia/When2Call`，`train/when2call_train_pref.jsonl`，许可 **CC-BY-4.0**
- 用 `pref` 而非 `sft`：`sft` 的 15,000 条里 assistant 只有纯文本，没有一条带
  `<TOOLCALL>` 结构化标记，三类标签分不出来
- 标签判定：`chosen_response` 含 `<TOOLCALL>` → `call_tool`；以问号结尾或含
  `which / could you / please provide` → `request_for_info`；否则 `cannot_answer`
- 仅保留 chosen 与 rejected 类别不同的偏好对（否则这条偏好不携带类别信息）
- 工具 schema 超过 6,000 字符的丢弃（token 预算）

## 4 数据统计

| 文件 | 条数 | 来源构成 | 选项数 | state 字符（中位 / p95 / max） | gold 位置分布 |
|---|---:|---|---|---|---|
| train_targeted | 1,000 | drop 500 / when2call 300 / synth 200 | 4:700, 3:300 | 1,023 / 2,896 / 8,595 | 285/270/275/170 |
| train_anchor | 300 | when2call 300 | 3:300 | 1,155 / 3,288 / 4,534 | 106/94/100 |
| val | 150 | 75 / 50 / 25 | 4:100, 3:50 | 1,030 / 2,878 / 3,930 | 44/41/41/24 |
| heldout | 150 | 75 / 50 / 25 | 4:100, 3:50 | 1,248 / 3,217 / 4,648 | 47/34/43/26 |

gold 位置已打乱，四个位置大致均匀（位置 0 略多，来自 3 选项题的 3 个位置摊到 4 格）。

## 5 校验做了什么、没做什么

`scripts/validate_dataset.py` 查的是**结构自洽**：

- 字段齐备、`question.type == "choice"`
- `target` 恰好一个标签、权重 1.0、且该标签必须出现在 `criteria` 里
- `state` 里没有 `expected / answer / label / target` 这类泄题字段
- id 与（state + criteria）签名不重复
- 选项数 3 或 4、无重复键

**没查的**：渲染后的选项字母是否够用、真实 token 长度、是否会被 `--max-len` 截断。
这三项需要 OneJev runtime 和模型 tokenizer，在训练机上跑：

```bash
python -c "
import sys; sys.path.insert(0,'OneJev')
from jevbench.tasks import load_jsonl   # 或 train.common.load_rows
from train.common import build_question
from qev.prompt import render_question
rows = [l for l in open('data/train_targeted.jsonl')]
"
```

简单办法：训练日志第一行会打印 `0 条超长被丢`，如果这个数不是 0，说明有样本超长。

## 6 已知问题

1. **When2Call 标签歧义已剔除**：原始 7,290 条里有 32 个用户请求配了两个不同的
   正确标签（同一 prompt 不同选中回复）。这类样本直接喂进去等于教模型学噪声，
   已整组删除，剩 7,210 条。**扩数据时必须保留这个过滤。**
2. **state 长度与评测集仍有差距**：训练集中位 1,023 字符，JevBench 场景题约 70 字符。
   200 条合成短题是为了缓解这个错位，但整体仍偏长。如果效果不好，
   优先加大合成题比例，而不是加大 DROP 比例。
3. **最长一条 8,595 字符**，在 `--max-len 4096` 下可能被截断或丢弃。
   训练日志里确认一下"超长被丢"的条数；超过 20 条就把 `--max-len` 提到 6144。
4. **合成数据不是真人标注**：规则清晰、标签可验证，但语言多样性差。
   建议作为独立一臂评估，不要和真实数据混着下结论。
5. **criteria 描述必须为空。** 旧版 `convert_drop.py` 曾把"正确计算结果"/"干扰项"
   写进选项描述，而 qev 会把它渲染成 `选项名: 描述` 放进模型输入——等于直接标出答案。
   本仓库所有 DROP 行的 criteria 取值都是空字符串（`scripts/validate_dataset.py` 可复核）。
   如果你自己改转换脚本，不要在 criteria 里写任何区分 gold 与干扰项的内容。
6. **没有 long_policy 数据**：27B 错 3 条，公开同分布数据不存在，这轮不做。

## 7 许可与再分发

| 来源 | 许可 | 再分发 |
|---|---|---|
| DROP | CC-BY-4.0 | 可，需署名 |
| When2Call | CC-BY-4.0 | 可，需署名 |
| 短场景合成 | 本仓库原创 | 可，建议同样署名 |

如果这个仓库要公开，`docs/` 里保留本文件的第 7 节即可满足署名要求。

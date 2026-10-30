# 数据卡

## 1 概要

| 项 | 值 |
|---|---|
| 文件 | `data/` 下 4 个 jsonl，共 2.8MB |
| 总条数 | 1,900（主臂 1,250 + 对照 300 + 验证 175 + 留出 175） |
| 格式 | OneJev `train/sft.py` 原生格式，一行一题 |
| 语言 | 中文（合成题、长政策）/ 英文（DROP、When2Call） |
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

## 3 四个来源

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

### 3.3 长政策合成 long_policy（250 条训练 + val/heldout 各 25）

- 脚本：`scripts/build_long_policy.py`，本仓库原创
- 为什么要有：27B 在 public-231 的 14 条错误里有 3 条在 long_policy，典型失败是把
  "注销后删除全部数据"答成 erase_all，而正确答案是"保留交易记录后删除其余"——
  **例外条款读丢或优先级读反**。公开数据里没有同分布语料，只能合成。
- 每题三段：一般条款 + 例外条款 + 具体情形，问"该情形应如何/是否符合"
- 6 个规则族，参数、主体、措辞全部随机化：

| 规则族 | 一般条款 | 例外条款 |
|---|---|---|
| `refund` | N 小时内且未拆封可退 | 质量缺陷不受时限 |
| `erase` | 注销即删除全部数据 | 交易记录按财税要求保留 N 年 |
| `approval` | 超限额支出须事前审批 | 紧急差旅可事后补签 |
| `export` | 实习生不得导出生产数据 | 带教陪同+登记后可在指定范围内用于学习 |
| `overtime` | 连续工作不超 N 小时 | 抢险可延长但须等时补休 |
| `liquid` | 单件/总量双上限 | 安检确认的必需药品豁免总量 |

- **标签由规则引擎按"先例外再一般"算出**，不是人写。
  每族都刻意埋了"信息不足"样本（关键字段缺失 → 两个答案都说不通），
  防止模型退化成"二选一猜一个"。
- 为保证 `(policy, case) → answer` 唯一，生成时会显式断言冲突并中断。
  （旧版 `rule_erase` 有变量没写进 case，同一句话出现两个 gold，已修。）

### 3.4 When2Call（300 条训练 + 300 条对照）

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
| train_targeted | 1,250 | drop 500 / when2call 300 / long_policy 250 / synth 200 | 4:950, 3:300 | 853 / 2,722 / 8,595 | 358/326/331/235 |
| train_anchor | 300 | when2call 300 | 3:300 | 1,155 / 3,288 / 4,534 | 106/94/100 |
| val | 175 | drop 75 / when2call 50 / long_policy 25 / synth 25 | 4:125, 3:50 | 808 / 2,848 / 3,930 | 42/46/54/33 |
| heldout | 175 | 同上比例，与 val 不重叠 | 4:125, 3:50 | 1,140 / 3,075 / 4,648 | 46/49/51/29 |

gold 位置已打乱，四个位置大致均匀（位置 0 略多，来自 3 选项题的 3 个位置摊到 4 格）。

long_policy 在 train 中的规则族分布均衡（export 46 / liquid 44 / refund 43 / overtime 41 /
approval 40 / erase 36），标签分布 `符合 102 / 不符合 79 / 信息不足 26 /
保留交易记录后删除其余 25 / 全部删除 22`。分布不均来自规则本身的非对称，
不是采样偏差——各族去重空间都在 100 以上，300 条生成量下不会提前耗尽某一族：

- `refund`: hours(3) × since(4) × opened(3) × defect(3) = 108
- `overtime`: cap(3) × worked(3) × rescue(3) × rest(3) = 81
- `export`: mentor(3) × reg(2) × scope(2) × purpose(2) × ask(6) = 144

（`export` 早期版本只有 36 个组合、去重后只剩 31 条，六族严重失衡；已补 `purpose`
与 6 种 `ask` 措辞修掉。）

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
2. **state 长度与评测集仍有差距**：训练集中位 853 字符，JevBench 场景题约 70 字符。
   450 条短题（long_policy 250 + 短场景 200，state 中位约 114 字符）是为了缓解这个错位，
   但整体仍偏长。如果效果不好，优先加大合成题比例，而不是加大 DROP 比例。
3. **最长一条 8,595 字符**，在 `--max-len 4096` 下可能被截断或丢弃。
   训练日志里确认一下"超长被丢"的条数；超过 20 条就把 `--max-len` 提到 6144。
4. **合成数据不是真人标注**：规则清晰、标签可验证，但语言多样性差。
   建议作为独立一臂评估，不要和真实数据混着下结论。
5. **criteria 描述必须为空。** 旧版 `convert_drop.py` 曾把"正确计算结果"/"干扰项"
   写进选项描述，而 qev 会把它渲染成 `选项名: 描述` 放进模型输入——等于直接标出答案。
   本仓库所有 DROP 行的 criteria 取值都是空字符串（`scripts/validate_dataset.py` 可复核）。
   如果你自己改转换脚本，不要在 criteria 里写任何区分 gold 与干扰项的内容。
6. **long_policy 是合成数据，不是真人标注的法律/合规文本。**条款措辞由模板生成，
   只能教"先检查例外、再套一般条款"这个**推理顺序**，教不了真实政策的语言复杂度。
   27B 在 public-231 上只有 3 条 long_policy 错误，这 250 条的边际收益未知，
   建议和 `train_anchor` 一样当成可插拔的一臂单独评估。
7. **train 与 val/heldout 按 `case` 分组切分，不做随机切。** long_policy 同一族的句子高度相似，
   随机切会让 val 里出现 train 见过的情形（只是措辞微调），指标虚高。
   现在 val/heldout 里的 25 条 long_policy 的 `case` 字段在 train 中不出现。
   以后加数据时，`build_dataset.py` 里的 `split_grouped` 调用要保持这个分组键。

## 7 许可与再分发

| 来源 | 许可 | 再分发 |
|---|---|---|
| DROP | CC-BY-4.0 | 可，需署名 |
| When2Call | CC-BY-4.0 | 可，需署名 |
| 短场景合成 | 本仓库原创 | 可，建议同样署名 |
| long_policy 合成 | 本仓库原创 | 可，建议同样署名 |

如果这个仓库要公开，`docs/` 里保留本文件的第 7 节即可满足署名要求。

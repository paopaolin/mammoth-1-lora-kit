# mammoth-1 (27B) 决策模型 LoRA / 全量微调包

给 27B 决策模型做适配的完整材料：数据集、实验设置、跑通脚本。

## 先把三个东西分清楚

这三个常被误当成"三选一"，其实是一层压一层：

| 层 | 是什么 | 在哪 |
|---|---|---|
| **模型权重** | mammoth-1，27B，Qwen3.5-27B 后训练而来（CE + Brier λ=1，视觉塔冻结） | `baselabs/mammoth-1`（HF，54.7GB） |
| **结构配置** | `slot_lm` 答案槽读出、`block_bidirectional` 注意力 | 上面那个 checkpoint 的 `config.json` + `readout.json` |
| **训练/推理代码** | `train/sft.py`（LoRA 与全量 FSDP 两条路）、`qev.engine.DecisionEngine`、jevbench 评测 | **OneJev** 仓库 |

所以「用 OneJev」不是「不用 mammoth-1」——**我们训的就是 mammoth-1**。
OneJev 是它指定的运行方式：mammoth-1 的模型卡里，自托管那节写的是
`setup.sh` installs the pinned serving runtime (**OneJev + serve.patch**)，
官方推理路径本来就依赖 OneJev 的 `qev.engine`。

反过来，脱离 OneJev 自己写训练器，要复刻的是：答案槽在读出头的位置、
slot 词表校验、CE + Brier 的加权方式、block-bidirectional 的 mask、
FSDP 的分片边界。**任何一处和原架构不一致，不是"效果差一点"，是权重直接对不上。**
这轮唯一要自己决定的是超参和配比，不是训练栈。

要区分的一点：OneJev 自己也发布模型权重（`OmniJev/OneJev1.6-*`），
那是另一批模型，**我们没在用**；我们只取它的 `train/` 和 `qev/` 代码。

## 五分钟跑通

```bash
# 0. 训练框架（官方仓库，自己 clone）
git clone https://github.com/OmniJev/OneJev
cd OneJev && git checkout 81ce62f && cd ..
pip install -e "OneJev[train]"
pip install tilelang==0.1.15          # Hopper 卡必装，否则梯度会静默算错

# 1. 下模型（54.7GB，MIT 许可，不需要申请）
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('baselabs/mammoth-1', local_dir='models/mammoth-1')"

# 2. 自检 + 显存探测（1 张卡，约 10 分钟）
python scripts/check_env.py                 # 会顺便打印 config 里的注意力/读出头
bash  scripts/probe_memory.sh 0 2048

# 3. 正式训练（1 张卡 LoRA，2 小时上下；--merge 让训练完直接产出可评的完整模型）
bash scripts/train_27b_lora.sh 0 data/train_targeted.jsonl tgt_2ep_lr1e5 --merge

# 4. 评测
bash scripts/eval_on_tasks.sh 0 runs/tgt_2ep_lr1e5/merged eval_sets/crossbench_hard_1200.jsonl tgt_hard

# 5.（可选）8 卡全量微调，官方 27B 配方
bash scripts/train_27b_full.sh 8,9,10,11,12,13,14,15 data/train_targeted.jsonl full_1ep_lr5e6
#   产物是 runs/full_1ep_lr5e6/final，完整模型，直接拿去评，不需要合并
```

## 里面有什么

```
├── data/                       训练集（4 个文件，2.8MB）
│   ├── train_targeted.jsonl     1,250 条，推荐主臂
│   ├── train_anchor.jsonl         300 条，对照臂（只用锚点数据）
│   ├── val.jsonl                  175 条，选 checkpoint 用
│   └── heldout.jsonl              175 条，只用于最终报告
├── eval_sets/                  我们自己的两个跨域评测集（各 1,200 条）
├── scripts/
│   ├── check_env.py             环境自检（版本、tilelang、更不能错的那几项）
│   ├── probe_memory.sh          真实训练循环跑 5 分钟，读显存实测值
│   ├── build_dataset.py         重建全部数据（种子 42，可完全复现）
│   ├── build_long_policy.py     长政策合成题的生成器
│   ├── make_variants.py         生成 4 个消融子集（第一轮卡 8–11 用）
│   ├── make_soft_labels.py      软标签构造
│   ├── train_27b_lora.sh        单卡 LoRA 训练
│   ├── train_27b_full.sh        8 卡全量微调（FSDP，官方 27B 配方）
│   ├── run_round1.sh            第一轮：8 张卡 LoRA 一次起齐
│   ├── merge_adapter.py         手工合并 LoRA（训练时加 --merge 就不需要它）
│   ├── eval_round3.sh           第三轮：所有模型并行评测
│   └── collect_summaries.py     出总表（含 0.9 阈值的采纳精度/覆盖率）
└── docs/
    ├── 01_experiment_setup.md  实验设置：做什么、为什么、32 卡怎么排
    ├── 02_data_card.md         数据来源、构造规则、已知问题、许可
    ├── 03_env_and_run.md       环境版本、三个必踩的坑、逐条命令
    └── 04_why_this_design.md   9B 那轮的结论（不看也不影响跑 27B）
```

## 七步走完

```bash
bash scripts/run_round1.sh                 # 第一步：卡 0 复现 93.94%，卡 1–7 并行扫参
bash scripts/train_27b_full.sh 8,9,10,11,12,13,14,15 data/train_targeted.jsonl full_1ep_lr5e6
python scripts/collect_summaries.py 'results/*/summary.json'   # 看总表挑胜出配置
python scripts/make_soft_labels.py ...     # 第二轮：软标签对照
bash scripts/eval_round3.sh                # 第三轮：并行评测
```

## 四个必须先知道的事实

**一、LoRA 只能单卡，全量必须 8 卡起 FSDP。** OneJev 训练器第 557 行拒绝 `--fsdp`
配 LoRA（FSDP 只服务全量微调）。所以 LoRA 是"一张卡一个实验、并行跑很多个"，
全量是"8 张卡一个实验、32 张卡同时跑 3–4 个"。两条路都通，别混着理解。

**二、27B 的准确率空间很小。** 它在 public-231 上已经有 217/231（93.94%），
14 条错误里 7 条是 temporal_numeric。这个包的数据就是照这 14 条配的。
**合理预期是 +0~3 条，真正值得测的是校准和路由，不是涨点。**

**三、H20 是 Hopper（sm90），必须装 tilelang。** 否则 fla 的反向 kernel 会算错梯度
但不报错，训练曲线正常、权重是坏的。

**四、注意力不是命令行参数。** `block_bidirectional` 和 `slot_lm` 读出头写在
checkpoint 的 `config.json`（`modeling_overrides`）里，`sft.py` 没有
`--attention` / `--readout` 这种 flag，加了会 argparse 报错。合并模型时要确认
这一节被带到新 config 里，否则推理静默退回 causal。

## 我们要什么回

按 [`docs/01_experiment_setup.md`](docs/01_experiment_setup.md) 第 7 节，
每个 run 回：`summary.json`、`results.jsonl`、`train.log`，LoRA 再加 `adapter/`。
全量 run 的 55GB 权重不用传，除非它最后赢了。

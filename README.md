# mammoth-1 (27B) 决策模型 LoRA 微调包

给 27B 决策模型做参数高效适配的完整材料：数据集、实验设置、跑通脚本。
基于同一套流程在 9B（elephant-1）上的四轮实验结论设计，那部分结果在
[`docs/04_why_this_design.md`](docs/04_why_this_design.md)。

## 五分钟跑通

```bash
# 0. 环境（版本必须对齐，见 docs/03）
pip install torch==2.8.0 transformers==5.19.0 peft==0.21.2 accelerate --index-url ...
pip install tilelang==0.1.15          # Hopper 卡必装，否则梯度会静默算错
git clone https://github.com/OmniJev/OneJev && cd OneJev && git checkout 81ce62f

# 1. 下模型（54.7GB，MIT 许可，不需要申请）
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('baselabs/mammoth-1', local_dir='models/mammoth-1')"

# 2. 自检 + 显存探测（1 张卡，约 10 分钟）
python scripts/check_env.py
bash  scripts/probe_memory.sh 0 2048

# 3. 正式训练（1 张卡，2 小时上下）
bash scripts/train_27b_lora.sh 0 data/train_targeted.jsonl tgt_2ep_lr1e5

# 4. 合并 + 评测
python scripts/merge_adapter.py --adapter runs/tgt_2ep_lr1e5/adapter \
                               --out models/mammoth-1-tgt
bash scripts/eval_on_tasks.sh 0 models/mammoth-1-tgt eval_sets/crossbench_hard_1200.jsonl tgt_hard
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
│   ├── run_round1.sh            第一轮：12 张卡一次起齐
│   ├── merge_adapter.py         合并 LoRA + readout.json
│   ├── eval_round3.sh           第三轮：所有 adapter 并行评测
│   └── collect_summaries.py     出总表（含 0.9 阈值的采纳精度/覆盖率）
└── docs/
    ├── 01_experiment_setup.md  实验设置：做什么、为什么、32 卡怎么排
    ├── 02_data_card.md         数据来源、构造规则、已知问题、许可
    ├── 03_env_and_run.md       环境版本、三个必踩的坑、逐条命令
    └── 04_why_this_design.md   9B 那轮的结论，以及它们怎么决定了这次的设计
```

## 七步走完

```bash
bash scripts/run_round1.sh                 # 第一步：卡 0 复现 93.94%，其余 11 张并行扫参
python scripts/collect_summaries.py 'results/*/summary.json'   # 看总表挑胜出配置
python scripts/merge_adapter.py --adapter runs/<winner>/adapter --out models/<winner>-tgt
python scripts/make_soft_labels.py ...     # 第二轮：软标签对照
bash scripts/eval_round3.sh                # 第三轮：32 卡并行评测
```

## 三个必须先知道的事实

**一、27B 的 LoRA 只能单卡跑。** OneJev 训练器第 569 行明确拒绝 `--fsdp` 与 LoRA 同时使用
（FSDP 只服务全量微调）。所以 32 张 H20 的正确用法是**并行跑多个独立的单卡实验**，
而不是把一张卡的事拆到 32 张上。

**二、27B 的准确率空间很小。** 它在 public-231 上已经有 217/231（93.94%），
14 条错误里 7 条是 temporal_numeric。这个包的数据就是照这 14 条配的。
**合理预期是 +0~3 条，真正值得测的是校准和路由，不是涨点。**

**三、H20 是 Hopper（sm90），必须装 tilelang。** 否则 fla 的反向 kernel 会算错梯度
但不报错，训练曲线正常、权重是坏的。我们的 9B 那轮就是被库自己 raise 拦下来的。

## 我们要什么回

按 [`docs/01_experiment_setup.md`](docs/01_experiment_setup.md) 第 5 节，
最少回这四样：每个 run 的 `summary.json`、`results.jsonl`、训练日志、以及 adapter 的
`readout.json`。其余可以后补。

# 环境与跑通

## 1 版本（照抄，别自己升级）

我们 9B 那轮跑通的组合：

| 包 | 版本 | 说明 |
|---|---|---|
| Python | 3.11 | |
| torch | **2.8.0+cu128** | 和 27B 权重的 dtype/算子对齐 |
| transformers | **5.19.0** | `qwen3_5` 架构需要 ≥5.0，4.x 直接加载失败 |
| causal-conv1d | **1.7.0** | 需要编译，见下 |
| peft | 0.21.2 | LoRA 实现 |
| accelerate | 1.15.0 | |
| numpy | 2.3.2 | |
| triton | 3.4.0（torch 自带） | **落在风险区间，见坑 2** |
| tilelang | **0.1.15** | 必须有 |
| OneJev | commit `81ce62f` | 训练器与 qev runtime |

```bash
# OneJev（含 qev）
git clone https://github.com/OmniJev/OneJev && cd OneJev && git checkout 81ce62f
pip install -e .

# 关键：tilelang
pip install tilelang==0.1.15

# causal-conv1d 要编译（qwen3_5 的卷积核）
pip install causal-conv1d==1.7.0
```

如果 `causal-conv1d` 编译失败，先确认 `nvcc --version` 能跑、CUDA_HOME 设对，
再 `MAX_JOBS=8 pip install causal-conv1d==1.7.0`。

## 2 模型

```bash
python -c "from huggingface_hub import snapshot_download; \
snapshot_download('baselabs/mammoth-1', local_dir='models/mammoth-1', \
  allow_patterns=['*.json','*.safetensors','*.jinja','*.py','*.patch','*.sh','*.txt'])"
```

- 许可 **MIT**，非 gated，不需要申请、不需要 token
- 54.71GB，2 个 safetensors 分片
- 结构：64 层、hidden 5120、24 头 / 4 KV 头、vocab 248,320、含视觉塔
- 目录里必须有的文件：`config.json`、`readout.json`、`tokenizer*`、`chat_template.jinja`

`config.json` 里 `modeling_overrides.attention.full_attention = "block_bidirectional"`，
推理引擎据此推出 `noncausal`。**`train/sft.py` 没有 `--attention` / `--readout` 这类参数**，
注意力是加载模型时从 config 读的；命令行硬塞会直接 argparse 报错（坑 1）。

## 3 三个必踩的坑

### 坑 1：别给 `sft.py` 传 `--attention` / `--readout`

老版本（或者我们内部改过的分支）有这两个 flag，**官方 `81ce62f` 没有**。
传了会立刻失败：

```
train.sft: error: unrecognized arguments: --attention noncausal --readout slot_lm
```

`elephant-1` / `mammoth-1` 都是 block-bidirectional attention（材料块与选项块内部双向，
好让选项之间互相比较），这个设置写在 `config.json` 的 `modeling_overrides` 里，
由模型加载时读取，和命令行无关。

**怎么确认它真的生效了**：没有直接的办法，看代理指标——
卡 0 的基线复现如果得出 93.94%（217/231）、ECE 0.0194，就说明注意力路径是对的；
如果明显低于这个数（比如 88% 上下），很可能退化成了 causal，先停下来查。
`scripts/check_env.py` 会把 config 里这一节打印出来，方便对照。

### 坑 2：Hopper + Triton 3.4.x 会静默算错梯度

报错原文：

```
RuntimeError: Triton >= 3.4.0 and < 3.7.1 on Hopper GPUs produces incorrect results
for gated chunk_bwd_dqkwg. Please upgrade Triton to >= 3.7.1 or install tilelang
```

Qwen3.5 系列的 gated delta rule 依赖 `flash-linear-attention`，这个版本区间下
反向 kernel 的**结果是错的但不抛异常**——是 fla 自己检查到并主动 raise 才挡住的。
H20 和 H100 都是 sm90，一样中招。

装 `tilelang==0.1.15` 后 fla 自动切后端，日志里会出现：

```
[FLA Backend] chunk_bwd_dqkwg -> tilelang
```

**如果没有这行，先停下来查。** 没有这层检查的话训练会正常跑完、loss 正常下降，
你得到一个看起来成功但权重是坏的模型。

`scripts/check_env.py` 会把这一条替你查出来。

### 坑 3：合并 LoRA 时漏掉 `readout.json`

`readout.json` 是**底座自带的**（`{"readout": "slot_lm", "head_dim": 256, "n_slots": 256}`），
训练器不产出它。所以：

- adapter 目录里没有它，**这是正常的**，不要因此判定训练失败
- 合并成完整模型时必须把它从**底座**拷到新目录，否则推理侧读不到 slot 配置
- 另一个容易漏的是 `config.json` 里的 `modeling_overrides`（注意力）

`scripts/merge_adapter.py` 会从底座拷 `readout.json`，并比对合并前后的
`modeling_overrides`；底座有、合并后没了就直接 `sys.exit`——因为这种错不会在评测里
显形，只会让所有数字静默失真。

更省事的办法是训练时直接加 `--merge`，训练器内置合并（产物 `runs/<name>/merged`），
但这个内置路径**不会**帮你检查上面两个文件，评之前自己 `cat` 一眼。

## 4 逐条命令

```bash
# 0) 自检：包版本、GPU 型号、tilelang 是否装了
python scripts/check_env.py

# 1) 显存探测（唯一没被我们实测过的数字，必须你自己跑）
bash scripts/probe_memory.sh 0 2048
#   看输出里的 "mem XX.XG"：
#     < 85G  -> 直接用
#     85-95G -> --batch-tokens 改 1024
#     OOM    -> 改 512；仍 OOM 说明 27B LoRA 塞不进单卡

# 2) 复现基线（先确认推理口径对，约 10 分钟）
#   jevbench 跟着 OneJev 走；eval_on_tasks.sh 会自动去 $WORK/jevbench 和
#   $ONEJEV/jevbench 找，都找不到才报错（可用 JB=/path 指定）
cat OneJev/jevbench/datasets/public/{easy,original,hard}.jsonl > eval_sets/public_231.jsonl
bash scripts/eval_on_tasks.sh 0 models/mammoth-1 eval_sets/public_231.jsonl base_27b
#   期望：acc 93.94%（217/231），ECE 0.0194
#   对不上就别往下走

# 3) LoRA 训练（1 张卡，1.5-3 小时）
bash scripts/train_27b_lora.sh 0 data/train_targeted.jsonl tgt_2ep_lr1e5 --merge
#   换参数：EPOCHS=1 LR=5e-6 BRIER=3.0 bash scripts/train_27b_lora.sh 1 ... name --merge
#   --merge 让训练器最后把 adapter 合进底座，产物 runs/<name>/merged

# 4) 检查合并产物的两个文件（漏了会导致数字全错且不报错）
ls runs/tgt_2ep_lr1e5/merged/readout.json
python -c "import json;c=json.load(open('runs/tgt_2ep_lr1e5/merged/config.json'));\
print(c.get('modeling_overrides') or c.get('text_config',{}).get('modeling_overrides'))"

# 5) 三个集合各评一次
bash scripts/eval_on_tasks.sh 0 runs/tgt_2ep_lr1e5/merged eval_sets/public_231.jsonl tgt_231
bash scripts/eval_on_tasks.sh 0 runs/tgt_2ep_lr1e5/merged eval_sets/crossbench_random_1200.jsonl tgt_rand
bash scripts/eval_on_tasks.sh 0 runs/tgt_2ep_lr1e5/merged eval_sets/crossbench_hard_1200.jsonl tgt_hard

# 5b) 全量微调（8 卡起，官方 27B 配方；产物 runs/<name>/final 直接评，不用合并）
bash scripts/train_27b_full.sh 8,9,10,11,12,13,14,15 data/train_targeted.jsonl full_1ep_lr5e6
bash scripts/eval_on_tasks.sh 8 runs/full_1ep_lr5e6/final eval_sets/public_231.jsonl full_231

# 6) 软标签路线（可选，第 2 轮再做）
#   先起服务
CUDA_VISIBLE_DEVICES=0 python scripts/serve_baseten.py --model models/mammoth-1 --port 8100 &
#   再打标签
python scripts/make_soft_labels.py --src data/train_targeted.jsonl --out data/train_soft.jsonl
#   然后用 data/train_soft.jsonl 训练
```

## 5 并发跑多个实验

每个实验占 1 张卡、独立进程，直接换 `CUDA_VISIBLE_DEVICES` 并行起：

```bash
for i in 0 1 2 3; do
  LR=$(echo "5e-6 1e-5 2e-5 1e-5" | cut -d' ' -f$((i+1)))
  BR=$(echo "1.0 1.0 1.0 3.0"      | cut -d' ' -f$((i+1)))
  LR=$LR BRIER=$BR bash scripts/train_27b_lora.sh $i data/train_targeted.jsonl \
      grid_g${i}_lr${LR}_b${BR} > logs/grid_$i.log 2>&1 &
done
wait
```

显存是每卡独立的，互不影响。**注意 `runs-dir` 是共享的**，最后按 `--name` 区分即可。

## 6 常见故障

| 现象 | 原因 | 处理 |
|---|---|---|
| `KeyError: 'qwen3_5'` | transformers 太旧 | 升到 5.19.0 |
| 日志里有 `FLA Backend -> tilelang` 但没有 tilelang 包 | 装的是别的后端 | `pip install tilelang==0.1.15` |
| loss 正常但评测全错 | 漏了 `--attention noncausal` | 检查训练命令 |
| 合并后评测比 adapter 差很多 | `readout.json` 没带过去 | 看 merge 输出的 WARN |
| 训练集"超长被丢"很多 | 最长样本 8,595 字符 | `--max-len` 提到 6144 |
| 单卡 OOM | batch_tokens 太大 | 2048 → 1024 → 512 |
| 多卡起不来 | 用了 `--fsdp` | LoRA 不支持，见 docs/01 约束一 |

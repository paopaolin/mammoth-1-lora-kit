#!/usr/bin/env bash
# 多卡全量微调（FSDP）。超参不是我们拍的，是两处官方来源夹出来的：
#   1. OneJev train/configs/onejev_27b_full.yaml → full_finetune + fsdp, 1 epoch,
#      batch_tokens 16384, max_rows 64, warmup 3%, grad clip 1.0, 视觉塔冻结
#   2. mammoth-1 自己的模型卡（HF README）→ 1 epoch, peak LR **2.7e-6**,
#      AdamW β 0.9/0.95, no weight decay, 3% warm-up, cosine, bf16, Brier λ=1
#   默认 lr 取 2.7e-6：那是 mammoth-1 原始训练的峰值，接着它往下训不该超过这个数。
#
# 用法：
#   bash scripts/train_27b_full.sh <GPU列表> <数据文件> <实验名> [额外参数...]
# 例（8 卡一个 run）：
#   bash scripts/train_27b_full.sh 8,9,10,11,12,13,14,15 data/train_targeted.jsonl full_1ep_lr2p7e6
#
# 显存：27B 全量的权重+梯度+AdamW 状态约 430GB，靠 FSDP 摊到 N 张卡上，
# 8 卡约 54GB/卡（含 fp32 主权重），96GB 的 H20 放得下。**卡数少于 8 不要试**。
#
# 和 LoRA 脚本的三个差别，别改错：
#   1. full_finetune 必须和 fsdp 同时给，否则 sft.py 直接退出
#   2. 只有全量微调能用多卡；LoRA 多卡那条路是封死的
#   3. 产物是 runs/<name>/final 下的完整模型，不需要 merge_adapter
set -u

GPUS=$1
TRAIN=$2
NAME=$3
shift 3 || true

WORK=${WORK:-$(cd "$(dirname "$0")/.." && pwd)}
ONEJEV=${ONEJEV:-$WORK/OneJev}
MODEL=${MODEL:-$WORK/models/mammoth-1}

NPROC=$(awk -F',' '{print NF}' <<< "$GPUS")
FIRST=$(awk -F',' '{print $1}' <<< "$GPUS")
# 同一台机器上可能同时在跑好几个 run，端口必须错开，否则第二个 run 起不来
MASTER_PORT=${MASTER_PORT:-$(( 29500 + FIRST ))}

LR=${LR:-2.7e-6}
EPOCHS=${EPOCHS:-1}
MAXLEN=${MAXLEN:-4096}
BT=${BT:-16384}
MAXROWS=${MAXROWS:-64}
BRIER=${BRIER:-1.0}
PAD=${PAD:-0.5}
SEED=${SEED:-0}

mkdir -p "$WORK/runs" "$WORK/logs"
cd "$ONEJEV" || exit 1

echo "[$(date +%H:%M:%S)] full-FT $NAME | $NPROC 卡 ($GPUS) | port $MASTER_PORT"
echo "               data=$(basename "$TRAIN") epochs=$EPOCHS lr=$LR brier=$BRIER pad=$PAD"

CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc-per-node "$NPROC" \
  --master_port "$MASTER_PORT" --master_addr 127.0.0.1 \
  -m train.sft \
  --model "$MODEL" \
  --train "$TRAIN" \
  --name "$NAME" \
  --runs-dir "$WORK/runs" \
  --full-finetune --fsdp \
  --dtype bfloat16 \
  --max-len "$MAXLEN" --batch-tokens "$BT" --max-rows "$MAXROWS" \
  --epochs "$EPOCHS" --lr "$LR" --brier-weight "$BRIER" \
  --warmup-frac 0.03 --weight-decay 0.0 --max-grad-norm 1.0 \
  --pad-options "$PAD" --seed "$SEED" \
  --save-every 400 --log-every 5 \
  --eval "val=$WORK/data/val.jsonl" --eval-every 150 --eval-limit 100 \
  "$@" 2>&1 | tee "$WORK/logs/$NAME.log"

echo "[$(date +%H:%M:%S)] 完成 -> $WORK/runs/$NAME/final（完整模型，直接拿去评）"
echo "  注意：每个全量 run 的 final 约 55GB，20 个 run 就是 1.1TB，磁盘要先看好"

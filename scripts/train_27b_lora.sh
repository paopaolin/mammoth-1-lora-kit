#!/usr/bin/env bash
# 单卡 LoRA 训练。27B 只能这样跑——sft.py 不支持 LoRA + FSDP（见 docs/01 约束一）。
#
# 用法：
#   bash scripts/train_27b_lora.sh <GPU> <数据文件> <实验名> [额外参数...]
# 例：
#   bash scripts/train_27b_lora.sh 0 data/train_targeted.jsonl tgt_2ep_lr1e5
#   bash scripts/train_27b_lora.sh 1 data/train_targeted.jsonl tgt_brier3 --brier-weight 3.0
set -u

GPU=$1
TRAIN=$2
NAME=$3
shift 3 || true

WORK=${WORK:-$(cd "$(dirname "$0")/.." && pwd)}
ONEJEV=${ONEJEV:-$WORK/OneJev}
MODEL=${MODEL:-$WORK/models/mammoth-1}
BT=${BT:-2048}
MAXLEN=${MAXLEN:-4096}
EPOCHS=${EPOCHS:-2}
LR=${LR:-1e-5}
BRIER=${BRIER:-1.0}

mkdir -p "$WORK/runs" "$WORK/logs"
cd "$ONEJEV" || exit 1

echo "[$(date +%H:%M:%S)] GPU$GPU  $NAME  data=$(basename "$TRAIN")  epochs=$EPOCHS lr=$LR brier=$BRIER bt=$BT"

CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc-per-node 1 -m train.sft \
  --model "$MODEL" \
  --train "$TRAIN" \
  --name "$NAME" \
  --runs-dir "$WORK/runs" \
  --attention noncausal --readout slot_lm \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --max-len "$MAXLEN" --batch-tokens "$BT" \
  --epochs "$EPOCHS" --lr "$LR" --brier-weight "$BRIER" \
  --eval "val=$WORK/data/val.jsonl" --eval-every 200 --eval-limit 120 \
  "$@" 2>&1 | tee "$WORK/logs/$NAME.log"

echo "[$(date +%H:%M:%S)] 完成 -> $WORK/runs/$NAME/adapter"

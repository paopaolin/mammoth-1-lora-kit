#!/usr/bin/env bash
# 单卡 LoRA 训练。27B 的 LoRA 只能这样跑——sft.py 拒绝 --fsdp 配 LoRA（见 docs/01 约束一）。
#
# 用法：
#   bash scripts/train_27b_lora.sh <GPU> <数据文件> <实验名> [额外参数...]
# 例：
#   bash scripts/train_27b_lora.sh 0 data/train_targeted.jsonl tgt_2ep_lr1e5
#   bash scripts/train_27b_lora.sh 1 data/train_targeted.jsonl tgt_brier3 --brier-weight 3.0
#
# 注意：不要给这个脚本加 --attention / --readout 之类的参数。
# sft.py 没有这两个 flag（答案槽读出是架构自带的），加了会直接 argparse 报错。
# 注意力是 block_bidirectional 还是 causal，由 checkpoint 的 config.json 里
# modeling_overrides 决定，跟命令行无关。
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
PAD=${PAD:-0}

mkdir -p "$WORK/runs" "$WORK/logs"
cd "$ONEJEV" || exit 1

echo "[$(date +%H:%M:%S)] GPU$GPU  $NAME  data=$(basename "$TRAIN")  epochs=$EPOCHS lr=$LR brier=$BRIER bt=$BT pad=$PAD"

CUDA_VISIBLE_DEVICES=$GPU torchrun --nproc-per-node 1 -m train.sft \
  --model "$MODEL" \
  --train "$TRAIN" \
  --name "$NAME" \
  --runs-dir "$WORK/runs" \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --max-len "$MAXLEN" --batch-tokens "$BT" \
  --epochs "$EPOCHS" --lr "$LR" --brier-weight "$BRIER" \
  --pad-options "$PAD" \
  --eval "val=$WORK/data/val.jsonl" --eval-every 200 --eval-limit 120 \
  "$@" 2>&1 | tee "$WORK/logs/$NAME.log"

echo "[$(date +%H:%M:%S)] 完成 -> $WORK/runs/$NAME/final"

#!/usr/bin/env bash
# 第一步：显存探测。跑真实训练循环几分钟，读 sft.py 自己打印的 mem 字段。
#
# 为什么不写一个"估算脚本"：估算会低估激活值和碎片，只有真跑才算数。
# 这一步同时也是"流程跑通"的验证——能打印出 step 行，说明数据、模型、
# 依赖、注意力配置全对。
#
# 用法：bash scripts/probe_memory.sh [GPU编号] [batch_tokens]
set -u

GPU=${1:-0}
BT=${2:-2048}
WORK=${WORK:-$(cd "$(dirname "$0")/.." && pwd)}
ONEJEV=${ONEJEV:-$WORK/OneJev}
MODEL=${MODEL:-$WORK/models/mammoth-1}
DATA=${DATA:-$WORK/data/train_targeted.jsonl}
OUT=${OUT:-$WORK/probe.log}

echo "GPU=$GPU  batch_tokens=$BT"
echo "模型 $MODEL"
echo "数据 $DATA"
nvidia-smi -i "$GPU" --query-gpu=index,memory.used,memory.total --format=csv,noheader

cd "$ONEJEV" || exit 1
CUDA_VISIBLE_DEVICES=$GPU timeout 420 torchrun --nproc-per-node 1 -m train.sft \
  --model "$MODEL" \
  --train "$DATA" \
  --name probe_mem \
  --runs-dir "$WORK/runs" \
  --attention noncausal --readout slot_lm \
  --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
  --target-modules q_proj,k_proj,v_proj,o_proj \
  --max-len 4096 --batch-tokens "$BT" \
  --epochs 1 --lr 1e-5 --brier-weight 1.0 \
  2>&1 | tee "$OUT" | grep -E "^(train \[|PLAN|OOM|.*Error|.*error)" | head -12

echo
echo "=== 峰值显存（sft.py 每步打印的 mem 字段）==="
grep -oE "mem [0-9.]+G" "$OUT" | sort -t' ' -k2 -gr | head -3
echo
echo "判读："
echo "  mem < 85G  → 单卡 96G（H20）可跑，按这个 batch_tokens 直接用"
echo "  mem 85-95G → 把 --batch-tokens 降到 1024 再试"
echo "  出现 OOM   → 降到 512/1024；仍 OOM 说明 27B LoRA 塞不进单卡，"
echo "               只能走 --full-finetune --fsdp（脚本不支持 LoRA+FSDP，见 docs/01）"

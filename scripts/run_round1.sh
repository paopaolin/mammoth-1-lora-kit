#!/usr/bin/env bash
# 第一轮：12 张卡各跑一个独立实验（互不通通信，一张卡挂了不拖累别的卡）。
#
# 这是 docs/01 第 4 节那个表的可执行版本。每张卡一个进程，日志落在 logs/<实验名>.log。
#
# 用法：
#   GPUS=0,1,2,3,4,5,6,7,8,9,10,11 bash scripts/run_round1.sh
#
# 只有 8 张卡也能跑，把 GPUS 改短即可（脚本按实际给的卡数取前 N 个实验）：
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/run_round1.sh
#
# 环境变量：WORK(仓库根) / ONEJEV(OneJev 路径) / MODEL(mammoth-1 路径)
#
# 说明：卡 0 是基线复现（只评测不训练）。它的结果没出来之前别看别的日志——
# 复现不出 93.94% 说明环境或评测口径有问题，后面全部对比都不成立。
set -u

WORK=${WORK:-$(cd "$(dirname "$0")/.." && pwd)}
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8,9,10,11}

IFS=',' read -r -a GPU_LIST <<< "$GPUS"

# 每张卡一条：<实验名>|<数据文件>|<额外参数>
PLAN=(
  "tgt_2ep_lr1e5|train_targeted.jsonl|"
  "tgt_2ep_lr5e6|train_targeted.jsonl|--lr 5e-6"
  "tgt_2ep_lr2e5|train_targeted.jsonl|--lr 2e-5"
  "tgt_1ep_lr1e5|train_targeted.jsonl|--epochs 1"
  "tgt_2ep_lr1e5_brier3|train_targeted.jsonl|--brier-weight 3.0"
  "tgt_2ep_lr1e5_brier5|train_targeted.jsonl|--brier-weight 5.0"
  "anchor_2ep_lr1e5|train_anchor.jsonl|"
  "tgt_no_longpolicy|train_no_longpolicy.jsonl|"
  "tgt_no_anchor|train_no_anchor.jsonl|"
  "tgt_drop_only|train_drop_only.jsonl|"
  "tgt_short_only|train_short_only.jsonl|"
)

echo "=== 卡 0：基线复现（这一步必须先出结果） ==="
# 端口按卡号错开，避免和第三轮并行评测撞端口
bash "$WORK/scripts/eval_on_tasks.sh" "${GPU_LIST[0]}" "$WORK/models/mammoth-1" \
     "$WORK/eval_sets/public_231.jsonl" base_eval $(( 8100 + GPU_LIST[0] )) \
     2>&1 | tee "$WORK/logs/base_eval.log"

echo "=== 卡 1-$(( ${#GPU_LIST[@]} - 1 ))：并行训练 ==="
missing=0
for i in "${!PLAN[@]}"; do
  gpu_idx=$(( i + 1 ))
  [ "$gpu_idx" -ge "${#GPU_LIST[@]}" ] && break
  IFS='|' read -r name data extra <<< "${PLAN[$i]}"
  f="$WORK/data/$data"
  if [ ! -f "$f" ]; then
    echo "[SKIP] $name  缺少数据文件 $f（先跑 scripts/make_variants.py）"
    missing=1
    continue
  fi
  gpu="${GPU_LIST[$gpu_idx]}"
  echo "  GPU$gpu -> $name  ($data${extra:+ $extra})"
  # shellcheck disable=SC2086
  nohup env CUDA_VISIBLE_DEVICES="$gpu" WORK="$WORK" \
      bash "$WORK/scripts/train_27b_lora.sh" "$gpu" "$f" "$name" $extra \
      > "$WORK/logs/$name.launch.log" 2>&1 &
done

echo
echo "已启动。查进度："
echo "  ls -lt logs/*.log"
echo "  tail -f logs/tgt_2ep_lr1e5.log"
if [ "$missing" = "1" ]; then
  echo "注意：有实验被跳过（见上面的 SKIP），补跑前先执行 python scripts/make_variants.py"
fi

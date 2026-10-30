#!/usr/bin/env bash
# 第一轮（LoRA 那一半）：一张卡一个独立实验，互不通信，一张卡挂了不拖累别的卡。
# 全量微调那三个臂另起，用 train_27b_full.sh，8 卡一个（见 docs/01 第 4 节）。
#
# 用法：
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/run_round1.sh
#
# 卡数少于实验数时会按顺序取前 N 个；只有 4 张卡也能跑，只是慢一点。
#
# 环境变量：WORK(仓库根) / ONEJEV(OneJev 路径) / MODEL(mammoth-1 路径)
#
# 说明：卡 0 是基线复现（只评测不训练）。它的结果没出来之前别看别的日志——
# 复现不出 93.94% 说明环境或评测口径有问题，后面全部对比都不成立。
set -u

WORK=${WORK:-$(cd "$(dirname "$0")/.." && pwd)}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

IFS=',' read -r -a GPU_LIST <<< "$GPUS"

# 每张卡一条：<实验名>|<数据文件>|<额外参数>
# 统一加 --merge：训练结束直接产出 runs/<name>/merged，省掉手工合并这一步
PLAN=(
  "tgt_2ep_lr1e5|train_targeted.jsonl|--merge"
  "tgt_2ep_lr5e6|train_targeted.jsonl|--merge --lr 5e-6"
  "tgt_2ep_lr2e5|train_targeted.jsonl|--merge --lr 2e-5"
  "tgt_1ep_lr1e5|train_targeted.jsonl|--merge --epochs 1"
  "tgt_2ep_lr1e5_brier3|train_targeted.jsonl|--merge --brier-weight 3.0"
  "tgt_2ep_lr1e5_brier5|train_targeted.jsonl|--merge --brier-weight 5.0"
  "anchor_2ep_lr1e5|train_anchor.jsonl|--merge"
  "tgt_no_longpolicy|train_no_longpolicy.jsonl|--merge"
)

echo "=== 卡 0：基线复现（这一步必须先出结果） ==="
# 端口按卡号错开，避免和第三轮并行评测撞端口
bash "$WORK/scripts/eval_on_tasks.sh" "${GPU_LIST[0]}" "$WORK/models/mammoth-1" \
     "$WORK/eval_sets/public_231.jsonl" base_eval $(( 8100 + GPU_LIST[0] )) \
     2>&1 | tee "$WORK/logs/base_eval.log"

echo "=== 卡 0-$(( ${#GPU_LIST[@]} - 1 ))：并行训练（卡 0 排在基线评测之后） ==="
missing=0
for i in "${!PLAN[@]}"; do
  # PLAN[i] 直接用 GPU_LIST[i]：卡 0 的基线评测是前台跑的，跑完才轮到它的训练，
  # 所以不会撞车，8 张卡正好跑满 8 个臂。
  gpu_idx=$i
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

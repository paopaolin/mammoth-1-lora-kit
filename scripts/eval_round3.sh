#!/usr/bin/env bash
# 第三轮：把第二轮产出的 adapter 全部并行评一遍（每张卡一个模型，端口错开）。
#
# 用法：
#   GPUS=0,1,...,31 bash scripts/eval_round3.sh
#   TASK=eval_sets/crossbench_hard_1200.jsonl GPUS=0,1,2,3 bash scripts/eval_round3.sh
#
# 默认评 public-231（主成绩）。跑完一轮后换 crossbench_random / crossbench_hard 再各跑一次：
#   TASK=eval_sets/crossbench_random_1200.jsonl bash scripts/eval_round3.sh
#   TASK=eval_sets/crossbench_hard_1200.jsonl   bash scripts/eval_round3.sh
#
# 评的是 **合并后的完整模型**。只有 adapter 目录不行——
# ject slot_lm 读出头（readout.json）必须和权重一起合并，否则评测侧读不出槽位。
set -u

WORK=${WORK:-$(cd "$(dirname "$0")/.." && pwd)}
GPUS=${GPUS:-0,1,2,3,4,5,6,7,8,9,10,11}
TASK=${TASK:-eval_sets/public_231.jsonl}

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
TASK_PATH="$TASK"
[ -f "$TASK_PATH" ] || TASK_PATH="$WORK/$TASK"
if [ ! -f "$TASK_PATH" ]; then
  echo "FATAL: 任务文件不存在: $TASK（也没找到 $TASK_PATH）"
  exit 1
fi

# 每个合并后的模型目录都当作一个待评条目
MODELS=()
for d in "$WORK"/models/*-tgt* "$WORK"/models/*-merged*; do
  [ -d "$d" ] && MODELS+=("$(basename "$d")")
done
if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "提示：models/ 下没有 *-tgt* 或 *-merged* 目录。"
  echo "先合并第二轮选出的 adapter："
  echo "  python scripts/merge_adapter.py --adapter runs/<run>/adapter --out models/<run>-tgt"
  exit 1
fi

echo "任务集: $(basename "$TASK_PATH")   待评模型 ${#MODELS[*]} 个   可用卡 ${#GPU_LIST[*]} 张"
echo

i=0
for m in "${MODELS[@]}"; do
  gpu="${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}"
  port=$(( 8100 + gpu ))
  echo "  GPU$gpu port$port -> $m"
  nohup bash "$WORK/scripts/eval_on_tasks.sh" "$gpu" "$WORK/models/$m" \
        "$TASK_PATH" "$m" "$port" > "$WORK/logs/eval_${m}.log" 2>&1 &
  i=$(( i + 1 ))
  # 同一张卡上排队的任务要错开启动，否则后起的进程会抢不到显存、两人一起 OOM
  if [ $(( i % ${#GPU_LIST[@]} )) -eq 0 ]; then sleep 20; fi
done

echo
echo "全部提交。汇总结果（等所有任务跑完后）："
echo "  python scripts/collect_summaries.py 'results/*/summary.json'"

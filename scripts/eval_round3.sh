#!/usr/bin/env bash
# 第三轮：把训练产物全部并行评一遍（每张卡一个模型，端口按卡号错开）。
#
# 会自动收集这三类目录：
#   runs/<run>/merged   训练时加了 --merge 的 LoRA 产物
#   runs/<run>/final    全量微调的产物（完整模型）
#   models/*-merged     手工跑 merge_adapter.py 出来的
#
# 用法：
#   GPUS=0,1,...,31 bash scripts/eval_round3.sh
#   TASK=eval_sets/crossbench_hard_1200.jsonl GPUS=0,1,2,3 bash scripts/eval_round3.sh
#
# 默认评 public-231（主成绩）。换集合再跑两次：
#   TASK=eval_sets/crossbench_random_1200.jsonl bash scripts/eval_round3.sh
#   TASK=eval_sets/crossbench_hard_1200.jsonl   bash scripts/eval_round3.sh
set -u

WORK=${WORK:-$(cd "$(dirname "$0")/.." && pwd)}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
TASK=${TASK:-eval_sets/public_231.jsonl}

IFS=',' read -r -a GPU_LIST <<< "$GPUS"
TASK_PATH="$TASK"
[ -f "$TASK_PATH" ] || TASK_PATH="$WORK/$TASK"
if [ ! -f "$TASK_PATH" ]; then
  echo "FATAL: 任务文件不存在: $TASK（也没找到 $TASK_PATH）"
  exit 1
fi

MODELS=()
for d in "$WORK"/runs/*/merged "$WORK"/runs/*/final "$WORK"/models/*-merged; do
  [ -d "$d" ] && MODELS+=("$d")
done
if [ "${#MODELS[@]}" -eq 0 ]; then
  echo "提示：没找到可评的模型目录（runs/*/merged、runs/*/final、models/*-merged）。"
  echo "LoRA 训练时忘了加 --merge 的话可以补："
  echo "  python scripts/merge_adapter.py --adapter runs/<run>/adapter --out models/<run>-merged"
  exit 1
fi

echo "任务集: $(basename "$TASK_PATH")   待评模型 ${#MODELS[*]} 个   可用卡 ${#GPU_LIST[*]} 张"
echo

i=0
for m in "${MODELS[@]}"; do
  gpu="${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}"
  port=$(( 8100 + gpu ))
  # runs/<run>/merged -> <run>；runs/<run>/final -> <run>_full；models/<x>-merged -> <x>-merged
  parent=$(basename "$(dirname "$m")")
  leaf=$(basename "$m")
  case "$leaf" in
    merged) name="$parent" ;;
    final)  name="${parent}_full" ;;
    *)      name="$leaf" ;;
  esac
  echo "  GPU$gpu port$port -> $name  ($m)"
  nohup bash "$WORK/scripts/eval_on_tasks.sh" "$gpu" "$m" \
        "$TASK_PATH" "$name" "$port" > "$WORK/logs/eval_${name}.log" 2>&1 &
  i=$(( i + 1 ))
  # 同一张卡上排队的任务要错开启动，否则后起的进程会抢不到显存、两人一起 OOM
  if [ $(( i % ${#GPU_LIST[@]} )) -eq 0 ]; then sleep 20; fi
done

echo
echo "全部提交。等跑完后出表："
echo "  python scripts/collect_summaries.py 'results/*/summary.json' --thresholds 0.8,0.9,0.95"

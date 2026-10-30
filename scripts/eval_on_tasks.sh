#!/usr/bin/env bash
# 起服务 → 跑评测 → 汇总 → 停服务。一个模型一个任务集跑一次。
#
# 用法：bash scripts/eval_on_tasks.sh <GPU> <模型目录> <任务文件> <运行名> [端口]
# 例：
#   bash scripts/eval_on_tasks.sh 0 models/mammoth-1 eval_sets/public_231.jsonl base_27b
#   bash scripts/eval_on_tasks.sh 0 models/mammoth-1-lora eval_sets/crossbench_hard.jsonl lora_hard
set -u

GPU=$1
MODEL=$2
TASKS=$3
NAME=$4
PORT=${5:-8100}

WORK=${WORK:-$(cd "$(dirname "$0")/.." && pwd)}
JB=${JB:-$WORK/jevbench}
OUT=$WORK/results/$NAME
mkdir -p "$OUT"

echo "[$(date +%H:%M:%S)] $NAME | $(basename "$MODEL") | $(basename "$TASKS")"

CUDA_VISIBLE_DEVICES=$GPU python "$WORK/scripts/serve_baseten.py" \
  --model "$MODEL" --port "$PORT" > "$WORK/logs/serve_$NAME.log" 2>&1 &
SERVE_PID=$!
trap 'kill $SERVE_PID 2>/dev/null' EXIT

for i in $(seq 1 60); do
  if python - "$PORT" <<'PY' 2>/dev/null
import sys, urllib.request
urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health", timeout=3)
PY
  then echo "[$(date +%H:%M:%S)] 服务就绪（$((i*3))s）"; break; fi
  sleep 3
done

cd "$JB" || exit 1
python -m jevbench.cli run \
  --tasks "$TASKS" --adapter typesafe --endpoint "http://127.0.0.1:$PORT" \
  --model "$NAME" --key-env= --cost-basis no_billable_local_selfhost --reserve-usd 0 \
  --results "$OUT/results.jsonl" --raw-dir "$OUT/raw" \
  --ledger "$OUT/ledger.jsonl" --manifest "$OUT/manifest.json" 2>&1 | tail -2

python -m jevbench.cli summarize \
  --tasks "$TASKS" --results "$OUT/results.jsonl" --ledger "$OUT/ledger.jsonl" \
  --public-export "$OUT/summary.json" >/dev/null 2>&1 && echo "汇总 -> $OUT/summary.json"

python - "$OUT/summary.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
print(f"  acc {s['accuracy']*100:.2f}% ({s['n_correct']}/{s['n_scorable']})  "
      f"brier {s['brier_mean']:.4f}  ece {s['ece']['ece']:.4f}  "
      f"p50 {s['latency']['p50_s']:.3f}s")
PY

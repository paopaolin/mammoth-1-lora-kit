#!/usr/bin/env python3
"""给 baset labs 的 elephant-1 / mammoth-1 套一层 /v1/systemone HTTP 服务。

官方只提供 `qev.engine.DecisionEngine` 的进程内调用（decision_index pipeline），
没有 HTTP server。而 jevbench 的 typesafe adapter 走 POST /v1/systemone，
所以这里把 decide() 原样暴露出去，响应结构与 strands-decider serve 保持一致，
好让两个模型在完全相同的评测口径下对比。

用法（容器内）:
    python serve_baseten.py --model /work/models/elephant-1 --port 8100
"""

from __future__ import annotations

import argparse
import time
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from qev.engine import DecisionEngine
from qev.schema import SystemOneRequest

app = FastAPI()
ENGINE = None
MODEL_NAME = ""
MODEL_PATH = ""


def _dump(ans):
    """qev 的 answer 是 pydantic model，转成纯 dict。"""
    if hasattr(ans, "model_dump"):
        return ans.model_dump()
    if isinstance(ans, dict):
        return ans
    return dict(ans)


@app.get("/health")
def health():
    info = getattr(ENGINE, "info", None)
    extra = getattr(info, "extra", {}) if info else {}
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "model_path": MODEL_PATH,
        "device": str(getattr(ENGINE, "device", "?")),
        "attention": extra.get("attention"),
        "readout": extra.get("readout"),
        "calibration": extra.get("calibration"),
        "torch": getattr(info, "torch_version", None),
        "transformers": getattr(info, "transformers_version", None),
    }


# 注意：这里用同步 def，让 FastAPI 丢到线程池跑。decide() 是阻塞的，
# 写成 async 会把事件循环堵死。jevbench 是串行发请求，线程池足够。
@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    t0 = time.perf_counter()
    try:
        resp, _ = ENGINE.decide(req)
    except ValueError as e:
        # 超长输入这类结构性拒绝，按 422 返回，让评测侧记成失败样本而不是静默截断
        return JSONResponse({"error": str(e)}, status_code=422)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"}, status_code=500)

    answers = {k: _dump(v) for k, v in resp.answers.items()}
    return {
        "model": MODEL_NAME,
        "answers": answers,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


def main():
    global ENGINE, MODEL_NAME, MODEL_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="本地模型目录或 HF repo id")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    MODEL_PATH = args.model
    MODEL_NAME = MODEL_PATH.rstrip("/").split("/")[-1]

    print(f"[serve_baseten] loading {MODEL_PATH} on {args.device} ...", flush=True)
    t0 = time.perf_counter()
    ENGINE = DecisionEngine(MODEL_PATH, device=args.device)
    print(f"[serve_baseten] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
    print(f"[serve_baseten] info: {ENGINE.info}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

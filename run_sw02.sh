#!/usr/bin/env bash
set -euo pipefail
# 从包根目录执行；任一预检/冒烟失败即退出，不进入完整训练。
PY="${PY:-.venv/bin/python}"
CFG=configs/experiments/SW02_SwinV2_ASL_EMA.json
PATHS=configs/internal_ab/paths.swinv2.official.json
TAG="$(date +%Y%m%d_%H%M%S)_$$"
: "${REF:?请设置REF为服务器SW01官方验证20轮真实结果目录}"
mkdir -p outputs/logs
exec > >(tee "outputs/logs/SW02_ASL_EMA_${TAG}.log") 2>&1
"$PY" -B -X utf8 -m unittest discover -s tests -p 'test_sw*.py'
CHECK="outputs/SW02_ASL_EMA_check_${TAG}"
SMOKE="outputs/SW02_ASL_EMA_smoke_${TAG}"
RUN="outputs/X_SWINV2_SW02_ASL_EMA_${TAG}"
"$PY" -B -X utf8 -u -m src.run_swinv2 --config "$CFG" --paths "$PATHS" --reference-run "$REF" --mode check --output "$CHECK"
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["ready_for_formal_training"],d' "$CHECK/precheck.json"
"$PY" -B -X utf8 -u -m src.run_swinv2 --config "$CFG" --paths "$PATHS" --reference-run "$REF" --mode smoke --execute --output "$SMOKE"
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["passed"] and d["successful_updates"]>=1,d' "$SMOKE/gpu_smoke.json"
"$PY" -B -X utf8 -u -m src.run_swinv2 --config "$CFG" --paths "$PATHS" --reference-run "$REF" --mode train --execute --output "$RUN"
echo "训练结束，结果目录：$RUN"

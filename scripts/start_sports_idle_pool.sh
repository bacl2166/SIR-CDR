#!/usr/bin/env bash
# User-operated L20 deployment. Does not edit any frozen checkout or study file.
set -euo pipefail

SIR_CONTROL=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
SIR_ROOT=/Data/lihb/SIR-CDR
SIR_PY=/Data/lihb/envs/sir-cdr/bin/python
SIR_LOG="$SIR_ROOT/logs/sports-clothing-ndcg-v1_idle-pool-v2.log"
test -x "$SIR_PY"
test -z "$(git -C "$SIR_CONTROL" status --porcelain)"
export HF_HOME=/Data/lihb/hf_cache HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONDONTWRITEBYTECODE=1

SIR_ARGS=(
  --root "$SIR_ROOT"
  --study-id sports-clothing-ndcg-v1
  --previous-root /Data/lihb/SIR-CDR-hpo-gpu7
  --previous-id gpu7-v1
  --expansion-id idle-gpus-v2
  --add-gpus
  GPU-a9254e79-fd90-c0cf-bafa-e1be2105784b
  GPU-3c86c8f7-52cb-3be2-7262-ed9438542bf5
  GPU-003a60e4-37d6-3a1d-382a-af28f79feb11
  GPU-f1e399c6-8d9d-706d-0e34-c0c65ad2e5d4
  GPU-93d28239-3e47-432b-5c8e-ae101fae9e70
  --max-workers 6 --host-ram-gib 192
)

date -Is
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv
df -h "$SIR_ROOT"
cd "$SIR_CONTROL"
CUDA_VISIBLE_DEVICES="" "$SIR_PY" -m unittest discover -s tests -p 'test_sports_*.py' -v
CUDA_VISIBLE_DEVICES="" "$SIR_PY" experiments/expand_sports_hpo.py --action inspect "${SIR_ARGS[@]}"

# Refuse duplicate launches, including a controller from another checkout/ID.
CUDA_VISIBLE_DEVICES="" "$SIR_PY" - "$SIR_CONTROL" <<'PY'
import importlib.util
from pathlib import Path
import sys
script = Path(sys.argv[1]) / 'experiments/expand_sports_hpo.py'
spec = importlib.util.spec_from_file_location('idle_launch', script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for p in module.base.support.owned_processes():
    argv = p['argv']
    if (module.base.option(argv, '--action') == 'run'
            and module.base.option(argv, '--root', '/Data/lihb/SIR-CDR') == '/Data/lihb/SIR-CDR'
            and module.base.option(argv, '--study-id', 'sports-clothing-ndcg-v1') == 'sports-clothing-ndcg-v1'
            and any(Path(v).name == 'expand_sports_hpo.py' for v in argv)):
        raise SystemExit(f"Expansion controller already running: {p['pid']}; inspect its log")
PY

mkdir -p "$SIR_ROOT/logs"
nohup "$SIR_PY" -u "$SIR_CONTROL/experiments/expand_sports_hpo.py" \
  --action run "${SIR_ARGS[@]}" >> "$SIR_LOG" 2>&1 < /dev/null &
SIR_LAUNCH_PID=$!
sleep 2
if ! kill -0 "$SIR_LAUNCH_PID" 2>/dev/null; then
  tail -n 40 "$SIR_LOG"
  exit 1
fi
echo "Expansion controller launched: $SIR_LAUNCH_PID"
echo "Log: $SIR_LOG"
tail -n 12 "$SIR_LOG"

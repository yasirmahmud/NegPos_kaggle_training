#!/usr/bin/env bash
# Submit both full LoRA runs to one private Kaggle GPU job, wait, and download them.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
KAGGLE_VENV="${REPO_ROOT}/.tools/kaggle-cli"
LOG_DIR="${REPO_ROOT}/logs"
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_FILE="${LOG_DIR}/overnight_training_${RUN_STAMP}.log"

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

if [[ ! -f .env ]]; then
    printf 'Missing Kaggle credential file: %s/.env\n' "${REPO_ROOT}" >&2
    exit 1
fi

# Reuse the proven COIL_RTL Kaggle CLI when present. Otherwise install an
# isolated current CLI in this repository.
if [[ ! -x "${KAGGLE_VENV}/bin/kaggle" ]] \
    && [[ ! -x /home/touhid/code/COIL_RTL/.tools/kaggle-cli-current/bin/kaggle ]] \
    && [[ ! -x /home/touhid/code/COIL_RTL/.tools/kaggle-cli-venv/bin/kaggle ]] \
    && ! command -v kaggle >/dev/null 2>&1; then
    python3 -m venv "${KAGGLE_VENV}"
    "${KAGGLE_VENV}/bin/python" -m pip install --quiet --upgrade pip kaggle
fi

python3 scripts/train_loras.py \
    --data data/top2000_four_negatives.jsonl \
    --output-root .tmp/validation_only \
    --validate-only

printf 'Full Kaggle training started. Log: %s\n' "${LOG_FILE}"
offload_command="run"
if [[ "${1:-}" == "--resume" ]]; then
    offload_command="resume"
    shift
fi
python3 scripts/kaggle_lora_offload.py "${offload_command}" "$@" 2>&1 | tee "${LOG_FILE}"
printf 'Full Kaggle training and download completed. Log: %s\n' "${LOG_FILE}"

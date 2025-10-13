#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python validate_inputs.py || { echo "[INFO] Creating external templates..."; python make_external_templates.py; }
python run_pipeline.py --config config/config.yml --profile baseline
echo "[INFO] Done. Check reports/"

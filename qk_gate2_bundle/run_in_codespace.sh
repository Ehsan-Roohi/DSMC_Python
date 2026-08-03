#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

rm -rf qk_gate2
cat qk_gate2_bundle/chunk_*.b64 | tr -d '\n\r' | base64 --decode > /tmp/qk_gate2_lean.tar.gz
echo "c5e8c050ae9657ef5afffd218dbfef92bc6736e7a6ff03fee719f73ad7db1e98  /tmp/qk_gate2_lean.tar.gz" | sha256sum -c -
tar -xzf /tmp/qk_gate2_lean.tar.gz

cd qk_gate2
chmod +x run_qk_gate2_validation.sh
./run_qk_gate2_validation.sh | tee "$ROOT/qk_gate2_codespace_run.log"

printf '\nCodespaces run complete. Final report:\n'
cat validation_output/QK_GATE2_VALIDATION_REPORT.txt

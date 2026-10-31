#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=/root/workspace
PROJECT="$WORKSPACE/project/Lip_Final"
AUTO="$PROJECT/baselines/auto_avsr"
PYTHON="$WORKSPACE/envs/pavir/bin/python"
CHECKPOINT="$WORKSPACE/checkpoints/vicocktail_w0.ckpt"
DATA_ROOT="$WORKSPACE/data_registry/auto_avsr_vicocktail/firewall"
CANONICAL="$WORKSPACE/data_registry/vicocktail_protocol_v1/canonical_ids_locked.jsonl"
MANIFEST="$WORKSPACE/data_registry/vicocktail_vc_calib.jsonl"
TOKENIZER="$PROJECT/benchmark_v1/vocabulary/vi_unigram_2000.model"
UNITS="$PROJECT/benchmark_v1/vocabulary/vi_unigram_2000.auto_avsr_units.txt"
CAMPAIGN="$PROJECT/pavir_campaign"
DECODE_ROOT="$CAMPAIGN/graphs/vicocktail/w0_vc_calib"
GRAPH_ROOT="$CAMPAIGN/graphs/vicocktail/frozen_v1"
LOG="$WORKSPACE/logs/run_vicocktail_w0_calib_decodes.log"
STATUS="$WORKSPACE/state/run_vicocktail_w0_calib_decodes.status"

mkdir -p "$DECODE_ROOT" "$GRAPH_ROOT" "$(dirname "$LOG")" "$(dirname "$STATUS")"
exec > >(tee -a "$LOG") 2>&1
trap 'code=$?; if [[ $code -eq 0 ]]; then printf "COMPLETE\n" > "$STATUS"; else printf "FAILED exit=%s\n" "$code" > "$STATUS"; fi; exit "$code"' EXIT
printf "RUNNING\n" > "$STATUS"

source "$PROJECT/scripts/activate_pavir_env.sh"
export VSR_CACHE_STRICT=1
export VSR_VISUAL_ONLY=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

for required in \
  "$CHECKPOINT" "$DATA_ROOT/labels/vc_calib.csv" "$CANONICAL" \
  "$MANIFEST" "$TOKENIZER" "$UNITS"; do
  [[ -f "$required" ]] || { echo "missing required input: $required" >&2; exit 2; }
done

decode_view() {
  local view="$1"
  local output="$DECODE_ROOT/$view"
  mkdir -p "$output"
  if [[ -s "$output/nbest.json" ]]; then
    echo "[$(date -Is)] Reuse complete-looking $view decode"
    return
  fi
  echo "[$(date -Is)] Decode W0 VC-calib view=$view"
  (
    cd "$AUTO"
    /usr/bin/time -v "$PYTHON" eval.py \
      --modality video \
      --root-dir "$DATA_ROOT" \
      --test-file vc_calib.csv \
      --test-subset val \
      --video-view "$view" \
      --sample-id-map "$CANONICAL" \
      --checkpoint-path "$CHECKPOINT" \
      --results-path "$output" \
      --beam-size 1 \
      --nbest 1 \
      --ctc-weight 0.1 \
      --length-penalty 0.0 \
      --seed 42
  ) > "$output/decode.log" 2>&1
  [[ -s "$output/nbest.json" ]] || {
    echo "$view decode produced no nbest.json" >&2
    exit 3
  }
  sha256sum "$output/nbest.json" "$output/hypotheses.json" > "$output/SHA256"
  echo "[$(date -Is)] Decode complete view=$view"
}

# Three deterministic views are independent and the H100 has ample memory;
# concurrent beam-1 processes reduce wall time without changing any stream.
decode_view center &
pid_center=$!
decode_view left &
pid_left=$!
decode_view right &
pid_right=$!
wait "$pid_center"
wait "$pid_left"
wait "$pid_right"

echo "[$(date -Is)] Build W0 center-view confusion graph"
"$PYTHON" "$PROJECT/tools/build_pavir_confusion_graph.py" \
  --predictions "$DECODE_ROOT/center/nbest.json" \
  --manifest "$MANIFEST" \
  --sp-model "$TOKENIZER" \
  --units "$UNITS" \
  --output-dir "$GRAPH_ROOT/pre_stability" \
  --corpus vicocktail \
  --split vc-calib \
  --min-support 20 \
  --min-sources 10 \
  --max-source-fraction 0.25 \
  --manual-review-limit 200

echo "[$(date -Is)] Require the same directed error in left and right views"
"$PYTHON" "$PROJECT/tools/verify_pavir_spatial_stability.py" \
  --graph "$GRAPH_ROOT/pre_stability/confusion_graph.json" \
  --manifest "$MANIFEST" \
  --center "$DECODE_ROOT/center/nbest.json" \
  --left "$DECODE_ROOT/left/nbest.json" \
  --right "$DECODE_ROOT/right/nbest.json" \
  --output-dir "$GRAPH_ROOT/stable" \
  --minimum-stable-support 20 \
  --minimum-stable-sources 10 \
  --minimum-stable-fraction 0.50 \
  --maximum-source-fraction 0.25

echo "[$(date -Is)] W0 VC-calib decode/graph/stability pipeline complete"

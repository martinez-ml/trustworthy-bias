#!/usr/bin/env bash
# ==============================================================================
# Run OpenRouter models on ELEPHANT datasets
# (AITA-YTA and AITA-NTA-OG) across all 10 dialects and baseline.
# ==============================================================================

set -euo pipefail

# Model configuration (can be overridden via OPENROUTER_MODEL=...)
MODEL_NAME="${OPENROUTER_MODEL:-qwen/qwen3.5-flash-02-23}"
export OPENROUTER_MODEL="$MODEL_NAME"

# Clean model slug for folder names (e.g. qwen_qwen-2.5-7b-instruct)
MODEL_SLUG="${MODEL_NAME//\//_}"

# Output directory organized by model
OUTPUT_DIR="${OUTPUT_DIR:-responses_elephant/${MODEL_SLUG}}"
mkdir -p "$OUTPUT_DIR"

# Workers for concurrency (default 10 parallel requests)
WORKERS="${OPENROUTER_WORKERS:-10}"
export OPENROUTER_WORKERS="$WORKERS"

echo "Model: $MODEL_NAME"
echo "Concurrency: $WORKERS parallel workers"
echo "Output Directory: $OUTPUT_DIR"

# Dialects in translations/elephant/
DIALECTS=(
  "Aboriginal"
  "Appalachian"
  "Black_South_African"
  "Chicano"
  "Colloquial_American"
  "Indian"
  "North_of_England"
  "Ozark"
  "Southeast_American_Enclave"
  "Urban_African_American"
)

# Target datasets
DATASETS=(
  "AITA-YTA"
  "AITA-NTA-OG"
)

# Optional sample size (e.g., export SAMPLE_N=5 for smoke testing)
SAMPLE_ARG=()
if [[ -n "${SAMPLE_N:-}" ]]; then
  SAMPLE_ARG=("$SAMPLE_N")
  echo "Running with SAMPLE_N=${SAMPLE_N} rows per dataset"
fi

# ------------------------------------------------------------------------------
# 1. (Optional) Run non-translated baseline
# ------------------------------------------------------------------------------
if [[ "${RUN_BASELINE:-1}" == "1" ]]; then
  echo "=========================================================="
  echo "Running Non-Translated Baseline Datasets"
  echo "=========================================================="
  mkdir -p "${OUTPUT_DIR}/baseline"

  for ds in "${DATASETS[@]}"; do
    input_csv="datasets/elephant/${ds}.csv"
    output_csv="${OUTPUT_DIR}/baseline/${ds}.csv"

    if [[ -f "$input_csv" ]]; then
      echo "--> Baseline: ${ds}"
      python accommodation/getting_model_responses/explicit_olmo_openrouter.py \
        "$input_csv" \
        "$output_csv" \
        ${SAMPLE_ARG[@]+"${SAMPLE_ARG[@]}"}
    else
      echo "Warning: ${input_csv} not found, skipping."
    fi
  done
fi

# ------------------------------------------------------------------------------
# 2. Run across all 10 translated dialects
# ------------------------------------------------------------------------------
for dialect in "${DIALECTS[@]}"; do
  echo "=========================================================="
  echo "Processing Dialect: ${dialect}"
  echo "=========================================================="
  mkdir -p "${OUTPUT_DIR}/${dialect}"

  for ds in "${DATASETS[@]}"; do
    input_csv="translations/elephant/${dialect}/${ds}.csv"
    output_csv="${OUTPUT_DIR}/${dialect}/${ds}.csv"

    if [[ -f "$input_csv" ]]; then
      echo "--> [${dialect}] ${ds}"
      python accommodation/getting_model_responses/explicit_olmo_openrouter.py \
        "$input_csv" \
        "$output_csv" \
        ${SAMPLE_ARG[@]+"${SAMPLE_ARG[@]}"}
    else
      echo "Warning: ${input_csv} not found, skipping."
    fi
  done
done

echo "=========================================================="
echo "All runs completed successfully! Results in: ${OUTPUT_DIR}/"
echo "=========================================================="


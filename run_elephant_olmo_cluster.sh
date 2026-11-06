#!/usr/bin/env bash
#SBATCH --job-name=explicit-olmo-elephant
#SBATCH --chdir=/fs/nexus-scratch/martj/trustworthy-bias
#SBATCH --partition=tron
#SBATCH --gres=gpu:rtxa5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=explicit-olmo-elephant-%j.out
#SBATCH --error=explicit-olmo-elephant-%j.err

set -euo pipefail

unset ACCELERATE_TORCH_DEVICE || true

OUTPUT_DIR="responses_elephant_olmo"
mkdir -p "$OUTPUT_DIR"

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

DATASETS=(
  "AITA-YTA"
  "AITA-NTA-OG"
)

SAMPLE_ARG=()
if [[ -n "${SAMPLE_N:-}" ]]; then
  SAMPLE_ARG=("$SAMPLE_N")
  echo "Running with SAMPLE_N=${SAMPLE_N}"
fi

# 1. Baseline
if [[ "${RUN_BASELINE:-1}" == "1" ]]; then
  mkdir -p "${OUTPUT_DIR}/baseline"
  for ds in "${DATASETS[@]}"; do
    input_csv="datasets/elephant/${ds}.csv"
    output_csv="${OUTPUT_DIR}/baseline/${ds}.csv"
    if [[ -f "$input_csv" ]]; then
      echo "=== [Baseline] ${ds} ==="
      python accommodation/getting_model_responses/explicit_olmo.py \
        "$input_csv" "$output_csv" ${SAMPLE_ARG[@]+"${SAMPLE_ARG[@]}"}
    fi
  done
fi

# 2. Dialects
for dialect in "${DIALECTS[@]}"; do
  mkdir -p "${OUTPUT_DIR}/${dialect}"
  for ds in "${DATASETS[@]}"; do
    input_csv="translations/elephant/${dialect}/${ds}.csv"
    output_csv="${OUTPUT_DIR}/${dialect}/${ds}.csv"
    if [[ -f "$input_csv" ]]; then
      echo "=== [${dialect}] ${ds} ==="
      python accommodation/getting_model_responses/explicit_olmo.py \
        "$input_csv" "$output_csv" ${SAMPLE_ARG[@]+"${SAMPLE_ARG[@]}"}
    fi
  done
done

echo "=== All ELEPHANT OLMo evaluations completed successfully! ==="


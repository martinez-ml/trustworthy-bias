#!/usr/bin/env bash
#SBATCH --job-name=olmo-openrouter-elephant
#SBATCH --chdir=/fs/nexus-scratch/martj/trustworthy-bias
#SBATCH --partition=tron
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=olmo-openrouter-elephant-%j.out
#SBATCH --error=olmo-openrouter-elephant-%j.err

set -euo pipefail

# Make sure OPENROUTER_API_KEY is available (from env or ~/.env)
if [[ -f ~/.env ]]; then
  set -a
  source ~/.env
  set +a
fi

./run_elephant_openrouter.sh


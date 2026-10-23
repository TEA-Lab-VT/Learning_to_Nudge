#!/bin/bash
#SBATCH --job-name=from_sif_ablation
#SBATCH --output=/home/jhaixin/IsaacLab_from_sif/logs/margin_ablation_%A_%a.out
#SBATCH --error=/home/jhaixin/IsaacLab_from_sif/logs/margin_ablation_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=tealab
#SBATCH --partition=h200_normal_q
#SBATCH --qos=tc_h200_normal_short

set -euo pipefail

# ---- ablation grid: SLURM_ARRAY_TASK_ID indexes into this list ----
# each entry is "init_sigma,refine_sigma"
MARGIN_PAIRS=(
    "0.01,0.02"
    "0.01,0.04"
    "0.02,0.01"
    "0.02,0.04"
    "0.04,0.01"
    "0.04,0.02"
)
PAIR=${MARGIN_PAIRS[$SLURM_ARRAY_TASK_ID]}
INIT_MARGIN=${PAIR%,*}
REFINE_MARGIN=${PAIR#*,}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
# Relax -e/-u around conda activation: Isaac Sim's setup_conda_env.sh references
# $ZSH_VERSION unguarded (unbound under set -u), and conda's generated
# deactivate.d/unsetenv.sh runs `unalias isaaclab` which fails silently
# (stderr redirected to /dev/null) if this shell didn't inherit that alias,
# aborting the script under set -e with no visible error.
set +eu
conda activate lab_jhaixin
set -eu
cd /home/jhaixin/IsaacLab
mkdir -p logs

OUT_DIR="pytorch_test_results_correct_007/margin_${INIT_MARGIN}_refine_${REFINE_MARGIN}"
REFINE_OUT_DIR="pytorch_test_boundary_check_results_007_margin_${INIT_MARGIN}_refine_${REFINE_MARGIN}"

echo "[init=${INIT_MARGIN} refine=${REFINE_MARGIN}] stage 1: initial NCBF training -> ${OUT_DIR}"
python3 scripts/environments/teleoperation/manip_ncbf/training.py \
    --init \
    --margin_threshold "${INIT_MARGIN}" \
    --output_dir "${OUT_DIR}"

echo "[init=${INIT_MARGIN} refine=${REFINE_MARGIN}] stage 2: refinement using the init=${INIT_MARGIN} model, refine margin=${REFINE_MARGIN}"
./isaaclab.sh -p scripts/environments/teleoperation/get_refined_demonstrations_hz_unsafe.py \
    --headless \
    --model_path "${OUT_DIR}/trained_NCBF_arm_initial" \
    --h_obj_normalizer_path "${OUT_DIR}/local_normalizer" \
    --ee_normalizer_path "${OUT_DIR}/arm_normalizer" \
    --margin_threshold "${REFINE_MARGIN}" \
    --output_dir "${REFINE_OUT_DIR}"
    # TODO: add any other flags you normally pass here (--task, --json_file, --num_envs, --cbf_min/--cbf_max, ...)

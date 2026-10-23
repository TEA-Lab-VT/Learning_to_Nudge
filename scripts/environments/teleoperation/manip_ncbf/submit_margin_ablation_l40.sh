#!/bin/bash
#SBATCH --job-name=l40_ablation
#SBATCH --output=/home/jhaixin/IsaacLab_from_sif/logs/margin_ablation_l40_%A_%a.out
#SBATCH --error=/home/jhaixin/IsaacLab_from_sif/logs/margin_ablation_l40_%A_%a.err
#SBATCH --array=0
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=tealab
#SBATCH --partition=l40s_normal_q
#SBATCH --qos=fal_l40s_normal_short

set -euo pipefail

# ---- ablation grid: SLURM_ARRAY_TASK_ID indexes into this list ----
MARGINS=(0.01)
MARGIN=${MARGINS[$SLURM_ARRAY_TASK_ID]}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
# Relax -e/-u around conda activation: Isaac Sim's setup_conda_env.sh references
# $ZSH_VERSION unguarded (unbound under set -u), and conda's generated
# deactivate.d/unsetenv.sh runs `unalias isaaclab` which fails silently
# (stderr redirected to /dev/null) if this shell didn't inherit that alias,
# aborting the script under set -e with no visible error.
set +eu
conda activate lab_jhaixin
set -eu
cd /home/jhaixin/IsaacLab_from_sif
mkdir -p logs

OUT_DIR="pytorch_test_results_correct_007/margin_${MARGIN}"
REFINE_OUT_DIR="pytorch_test_boundary_check_results_007_margin_${MARGIN}"

echo "[margin=${MARGIN}] stage 1: initial NCBF training -> ${OUT_DIR}"
python3 scripts/environments/teleoperation/manip_ncbf/training.py \
    --init \
    --margin_threshold "${MARGIN}" \
    --output_dir "${OUT_DIR}"

echo "[margin=${MARGIN}] stage 2: refinement using the margin=${MARGIN} init model"
./isaaclab.sh -p scripts/environments/teleoperation/get_refined_demonstrations_hz_unsafe.py \
    --headless \
    --model_path "${OUT_DIR}/trained_NCBF_arm_initial" \
    --h_obj_normalizer_path "${OUT_DIR}/local_normalizer" \
    --ee_normalizer_path "${OUT_DIR}/arm_normalizer" \
    --output_dir "${REFINE_OUT_DIR}"
    # TODO: add any other flags you normally pass here (--task, --json_file, --num_envs, --cbf_min/--cbf_max, ...)

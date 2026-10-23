#!/bin/bash
#SBATCH --job-name=20seed_cbf_plot_debug
#SBATCH --output=/home/jhaixin/IsaacLab/logs/20seed_cbf_plot_debug_%A_%a.out
#SBATCH --error=/home/jhaixin/IsaacLab/logs/20seed_cbf_plot_debug_%A_%a.err
#SBATCH --array=0-19  # set to 0..(#ENTRIES * #SEEDS - 1), see grid below
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=tealab-paid
#SBATCH --clusters=falcon
#SBATCH --partition=l40s_normal_q
# Runs main.py once per seed (step 1..1500, same model/normalizer for every
# seed), then generates the per-seed CBF map plots from the resulting
# trajectories.json via manip_ncbf/cbf_test_real.py. Each seed is its own
# SLURM array task, running in parallel.

set -euo pipefail

# ---- sweep grid: edit these ----
NUM_ENVS=1
SEEDS=($(seq 11 30))
MODEL_PATH="/home/jhaixin/IsaacLab/boundary_check_results_007_margin_0.01/model_batch_47.pt"
H_OBJ_NORMALIZER_PATH="/home/jhaixin/IsaacLab_from_sif/results_correct_0725_007/local_normalizer"
EE_NORMALIZER_PATH="/home/jhaixin/IsaacLab_from_sif/results_correct_0725_007/arm_normalizer"
# SBATCH --array above must cover 0..(#SEEDS - 1)

SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

OUT_ROOT="20seed_cbf_eval"
LOG_DIR="logs/20seed_cbf_eval"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
# Relax -e/-u around conda activation: Isaac Sim's setup_conda_env.sh references
# $ZSH_VERSION unguarded (unbound under set -u), and conda's generated
# deactivate.d/unsetenv.sh runs `unalias isaaclab` which fails silently
# (stderr redirected to /dev/null) if this shell didn't inherit that alias
# (e.g. already-active conda env inherited from the parent shell), aborting
# the script under set -e with no visible error.
set +eu
conda activate lab_jhaixin
set -eu
cd /home/jhaixin/IsaacLab
mkdir -p "$LOG_DIR"

# needed on some compute nodes (e.g. fal043) where isaacsim's pip install
# doesn't auto-set these
export ISAAC_PATH=/home/jhaixin/isaacsim
export EXP_PATH=/home/jhaixin/isaacsim/apps
export CARB_APP_PATH=/home/jhaixin/isaacsim/kit

TAG="seed${SEED}"
LOG_FILE="${LOG_DIR}/${TAG}.log"
RUN_STATS_DIR="${OUT_ROOT}/run_stats/${TAG}"
SUMMARY_DIR="${OUT_ROOT}/summary/${TAG}"
TRAJ_DIR="${OUT_ROOT}/traj/${TAG}"
PLOT_DIR="${OUT_ROOT}/plots/${TAG}"

echo "=== running ${TAG} ==="
./isaaclab.sh -p scripts/environments/teleoperation/inference_multi_modular/main.py \
    --headless \
    --seed "${SEED}" \
    --num_envs "${NUM_ENVS}" \
    --max_steps 1500 \
    --total_runs 1 \
    --model_path "${MODEL_PATH}" \
    --h_obj_normalizer_path "${H_OBJ_NORMALIZER_PATH}" \
    --ee_normalizer_path "${EE_NORMALIZER_PATH}" \
    --summary_dir "${SUMMARY_DIR}" \
    --run_stats_dir "${RUN_STATS_DIR}" \
    --traj_dir "${TRAJ_DIR}" \
    > "${LOG_FILE}" 2>&1
    # TODO: add any other flags you normally pass (--task, --scene_type, ...)
    # NOTE: --record_video isn't passed here; if you enable it, also pass
    # --video_dir "evaluation_videos/${TAG}" -- video filenames
    # (run_{idx}_env_{idx}.mp4) have no per-task uniqueness of their own,
    # so parallel tasks writing into the same video dir would collide.

echo "=== finished ${TAG}, log: ${LOG_FILE} ==="

# --- CBF map plots for this seed ---
# cbf_test_real.py does `from models import NCBF, Normalizer`, which resolves
# to manip_ncbf/models.py only if that directory is importable -- add it to
# PYTHONPATH rather than cd-ing there (this script assumes cwd
# /home/jhaixin/IsaacLab throughout, e.g. for the relative OUT_ROOT paths above).
MANIP_NCBF_DIR="/home/jhaixin/IsaacLab/scripts/environments/teleoperation/manip_ncbf"
TRAJ_FILE="${TRAJ_DIR}/trajectories.json"

if [ -f "${TRAJ_FILE}" ]; then
    echo "=== plotting CBF maps for ${TAG} ==="
    PYTHONPATH="${MANIP_NCBF_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    python "${MANIP_NCBF_DIR}/cbf_test_real.py" \
        --trajectory_file "${TRAJ_FILE}" \
        --output_dir "${PLOT_DIR}" \
        --model_path "${MODEL_PATH}" \
        --h_obj_normalizer_path "${H_OBJ_NORMALIZER_PATH}" \
        --ee_normalizer_path "${EE_NORMALIZER_PATH}" \
        >> "${LOG_FILE}" 2>&1
    echo "=== plots for ${TAG} saved to ${PLOT_DIR} ==="
else
    echo "WARNING: ${TRAJ_FILE} not found, skipping CBF plot generation for ${TAG}" >&2
fi

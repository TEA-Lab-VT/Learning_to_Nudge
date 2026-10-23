#!/bin/bash
#SBATCH --job-name=seed_numenvs_sweep
#SBATCH --output=/home/jhaixin/IsaacLab/logs/seed_numenvs_rrt_ours_50_%A_%a.out
#SBATCH --error=/home/jhaixin/IsaacLab/logs/seed_numenvs_rrt_ours_50_%A_%a.err
#SBATCH --array=0-6  # set to 0..(#SEEDS * #MODEL_PATHS - 1), see grid below
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=tealab
#SBATCH --partition=a100_normal_q
#SBATCH --qos=tc_a100_normal_short

# Runs main.py across a grid of --seed x --model_path values, at a fixed --num_envs.
# Each (seed, model_path) combination is its own array task, running in parallel.

set -euo pipefail

# ---- sweep grid: edit these ----
NUM_ENVS=100
SEEDS=(2222 0 1 2 3 4 5)
MODEL_PATHS=(
    # "/home/jhaixin/IsaacLab/boundary_check_results_007_margin_0.01/model_batch_47.pt"
    "/home/jhaixin/IsaacLab/pytorch_test_boundary_check_results_007_margin_0.01_refine_0.02/model_batch_47.pt"
    # "/home/jhaixin/IsaacLab/pytorch_test_boundary_check_results_007_margin_0.01_refine_0.04/model_batch_47.pt"
    # "/home/jhaixin/IsaacLab/pytorch_test_boundary_check_results_007_margin_0.02_refine_0.01/model_batch_47.pt"
    # "/home/jhaixin/IsaacLab/pytorch_test_boundary_check_results_007_margin_0.02_refine_0.04/model_batch_47.pt"
    # "/home/jhaixin/IsaacLab/pytorch_test_boundary_check_results_007_margin_0.04_refine_0.01/model_batch_47.pt"
    # "/home/jhaixin/IsaacLab/pytorch_test_boundary_check_results_007_margin_0.04_refine_0.02/model_batch_47.pt"
)
# SBATCH --array above must cover 0..(#SEEDS * #MODEL_PATHS - 1)

# map SLURM_ARRAY_TASK_ID -> (seed, model_path), seed outer / model inner
MODEL_COUNT=${#MODEL_PATHS[@]}
SEED_IDX=$(( SLURM_ARRAY_TASK_ID / MODEL_COUNT ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID % MODEL_COUNT ))
SEED=${SEEDS[$SEED_IDX]}
MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
# include the parent dir (e.g. margin_0.02) since every checkpoint file is
# named model_batch_47.pt -- basename alone would collide across models
MODEL_TAG="$(basename "$(dirname "${MODEL_PATH}")")_$(basename "${MODEL_PATH}" .pt)"

SUMMARY_DIR="run_stats_summary_rrt_ours_refined_50"
LOG_DIR="logs/seed_numenvs_rrt_ours_50"

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

TAG="seed${SEED}_model${MODEL_TAG}"
LOG_FILE="${LOG_DIR}/${TAG}.log"
RUN_STATS_DIR="run_stats_ours_refine_50obj/${TAG}"
TRAJ_DIR="final_traj_ours/${TAG}"

mkdir -p "${SUMMARY_DIR}"
BEFORE_FILES=$(ls "${SUMMARY_DIR}"/summary_stats_*.txt 2>/dev/null || true)

echo "=== running ${TAG} ==="
./isaaclab.sh -p scripts/environments/teleoperation/inference_multi_modular_rrt/main.py \
    --headless \
    --no-save_trajectory \
    --seed "${SEED}" \
    --num_envs "${NUM_ENVS}" \
    --model_path "${MODEL_PATH}" \
    --summary_dir "${SUMMARY_DIR}" \
    --run_stats_dir "${RUN_STATS_DIR}" \
    --traj_dir "${TRAJ_DIR}" \
    > "${LOG_FILE}" 2>&1
    # TODO: add any other flags you normally pass (--task, --scene_type, --total_runs, --max_steps, ...)
    # NOTE: --h_obj_normalizer_path / --ee_normalizer_path are left at main.py's
    # defaults here. If a MODEL_PATHS entry comes from a different training run,
    # its normalizers won't match -- add those flags too and pair them per-model.
    # NOTE: --record_video isn't passed here; if you enable it, also pass
    # --video_dir "evaluation_videos/${TAG}" -- video filenames
    # (run_{idx}_env_{idx}.mp4) have no per-task uniqueness of their own,
    # so parallel tasks writing into the same video dir would collide.

# main.py has no --output_dir; it always writes into $SUMMARY_DIR with a
# timestamped filename (no per-run id), shared across all array tasks. Diff
# against the pre-run listing to find the file this task produced. NOTE: if
# two tasks finish in the same second, main.py itself will collide on the
# same filename (see data_handler.py:save_summary_stats) and one run's data
# gets silently overwritten -- this only works around the *renaming*, not
# that underlying race.
AFTER_FILES=$(ls "${SUMMARY_DIR}"/summary_stats_*.txt 2>/dev/null || true)
NEW_FILE=$(comm -13 <(echo "${BEFORE_FILES}" | sort) <(echo "${AFTER_FILES}" | sort) | head -n 1)
if [ -n "${NEW_FILE}" ]; then
    cp "${NEW_FILE}" "${LOG_DIR}/${TAG}_$(basename "${NEW_FILE}")"
else
    echo "WARNING: could not identify a new summary file for ${TAG}" >&2
fi

echo "=== finished ${TAG}, log: ${LOG_FILE} ==="

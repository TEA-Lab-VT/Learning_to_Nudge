#!/bin/bash
#SBATCH --job-name=baseline_eval_sweep
#SBATCH --output=/home/jhaixin/IsaacLab/logs/apf20_%A_%a.out
#SBATCH --error=/home/jhaixin/IsaacLab/logs/apf20_%A_%a.err
#SBATCH --array=0-6  # set to 0..(#ENTRIES * #SEEDS - 1), see grid below
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=tealab-paid
#SBATCH --clusters=falcon
#SBATCH --partition=l40s_normal_q

# Runs baseline_do_nothing.py / evaluation.py / baseline_naive_backward.py (in
# IsaacLab, conda env lab_jhaixin) and evaluation.py (in IsaacLab_RL, conda env
# env_isaaclab_rl) across a grid of --seed values. Each (entry, seed)
# combination is its own array task, running in parallel.

set -euo pipefail

# ---- sweep grid: edit these ----
# each entry: repo dir name under /home/jhaixin | conda env | script path
# relative to <repo>/scripts/environments/teleoperation/standard_cbf/ | short
# tag used to name logs (needed since two entries are both called evaluation.py)
SEEDS=(0 4 2222)
ENTRIES=(
    "IsaacLab|lab_jhaixin|baseline_apf.py|apf"
    # "IsaacLab|lab_jhaixin|baseline_do_nothing.py|do_nothing"
    # "IsaacLab|lab_jhaixin|evaluation.py|cbf_eval"
    # "IsaacLab|lab_jhaixin|baseline_naive_backward.py|naive_backward"
    # "IsaacLab_RL|env_isaaclab_rl|evaluation.py|rl_eval"
)
# SBATCH --array above must cover 0..(#ENTRIES * #SEEDS - 1)

# map SLURM_ARRAY_TASK_ID -> (entry, seed), entry outer / seed inner
SEED_COUNT=${#SEEDS[@]}
ENTRY_IDX=$(( SLURM_ARRAY_TASK_ID / SEED_COUNT ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % SEED_COUNT ))
IFS='|' read -r REPO_NAME CONDA_ENV SCRIPT TAG_PREFIX <<< "${ENTRIES[$ENTRY_IDX]}"
SEED=${SEEDS[$SEED_IDX]}
REPO_ROOT="/home/jhaixin/${REPO_NAME}"

# logs always collected under IsaacLab/logs regardless of which repo actually
# ran (matches the #SBATCH --output/--error paths above), so results from
# both repos stay in one place
LOG_DIR="/home/jhaixin/IsaacLab/logs/baseline_eval_sweep_apf20"
mkdir -p "$LOG_DIR"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
# Relax -e/-u around conda activation: Isaac Sim's setup_conda_env.sh references
# $ZSH_VERSION unguarded (unbound under set -u), and conda's generated
# deactivate.d/unsetenv.sh runs `unalias isaaclab` which fails silently
# (stderr redirected to /dev/null) if this shell didn't inherit that alias
# (e.g. already-active conda env inherited from the parent shell), aborting
# the script under set -e with no visible error.
set +eu
conda activate "${CONDA_ENV}"
set -eu
cd "${REPO_ROOT}"

# needed on some compute nodes (e.g. fal043) where isaacsim's pip install
# doesn't auto-set these. Derived per-repo rather than hardcoded, since
# IsaacLab_RL bundles its own separate _isaac_sim (not the shared
# /home/jhaixin/isaacsim that IsaacLab's _isaac_sim symlinks to).
export ISAAC_PATH="${REPO_ROOT}/_isaac_sim"
export EXP_PATH="${REPO_ROOT}/_isaac_sim/apps"
export CARB_APP_PATH="${REPO_ROOT}/_isaac_sim/kit"

TAG="${TAG_PREFIX}_seed${SEED}"
LOG_FILE="${LOG_DIR}/${TAG}.log"

echo "=== running ${TAG} (${REPO_NAME}, conda env ${CONDA_ENV}) ==="
./isaaclab.sh -p "scripts/environments/teleoperation/standard_cbf/${SCRIPT}" \
    --headless \
    --seed "${SEED}" \
    > "${LOG_FILE}" 2>&1
    # TODO: add any other flags you normally pass (--task, --num_envs, --total_runs,
    # --max_steps, --scene_type, --load_run/--checkpoint [rl_eval only], ...)

# save_run_stats / save_summary_stats in the IsaacLab entries write into
# per-seed subdirectories (e.g. run_stats_summary_apf_40obj/seed_<seed>), so
# unlike inference_multi_modular/run_seed_numenvs_sweep.sh, concurrent array
# tasks don't share an output path and there's no rename-by-diff needed here.
# The IsaacLab_RL evaluation.py (rl_eval) hasn't been updated the same way --
# its save_summary_stats() still writes into a plain run_stats_summary/ with
# no seed subdirectory, so parallel rl_eval seeds CAN collide on filename.

echo "=== finished ${TAG}, log: ${LOG_FILE} ==="

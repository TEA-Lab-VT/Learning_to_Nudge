# NCBF Manipulation Safety

Neural CBF (NCBF) for Franka object-pushing scenes: data collection → training → refinement → inference evaluation → baseline comparison.

Code lives in `scripts/environments/teleoperation/`.


- Scripts that need simulation: `./isaaclab.sh -p ...`; pure PyTorch scripts: `python ...`.
- Many default paths are hardcoded. Check argparse / `config.py` before running.

## Provided artifacts

| Path | Content | Used as |
|---|---|---|
| `model/model_batch_47.pt` | Best NCBF model | `--model_path` |
| `model/local_normalizer/` | Object-state normalizer | `--h_obj_normalizer_path` |
| `model/arm_normalizer/` | End-effector normalizer | `--ee_normalizer_path` |
| `dataset/trajectories.json` | Raw demonstrations | input to `data_reformat.py` |
| `dataset/trajectories_reformat_*.npz` | Reformatted training data | input to `training.py` |

## Usage

### 1. Collect data

```bash
./isaaclab.sh -p $T/ncbf_manip_dis_random_vel.py --num_envs 100 --headless
python $T/data_reformat.py --input <trajectories.json> --horizon 1
```

### 2. Train the NCBF

```bash
python $T/manip_ncbf/training.py --init --margin_threshold 0.01 --output_dir <dir>
```

### 3. Refine with boundary resampling

```bash
./isaaclab.sh -p $T/get_refined_demonstrations_hz_unsafe.py --headless \
    --model_path <pt> \
    --h_obj_normalizer_path <dir> \
    --ee_normalizer_path <dir> \
    --output_dir <dir>
```

### 4. Evaluate the NCBF

```bash
./isaaclab.sh -p $T/inference_multi_modular/main.py --headless \
    --num_envs 100 --seed 2222 --scene_type large \
    --model_path model/model_batch_47.pt \
    --h_obj_normalizer_path model/local_normalizer \
    --ee_normalizer_path model/arm_normalizer
```

- Swap `inference_multi_modular` for `_apf` or `_rrt` to change the nominal policy (see below).
- `--scene_type`: `small` (4) / `medium` (10) / `large` (20) / `extra_large` (40 objects).
- To change the scene, also edit these to match:
  - `source/isaaclab/isaaclab/envs/mdp/events.py`: `scene_type = params.get("scene_type", "small")`
  - `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/lift_env_cfg_eval.py`: `NUM_OF_OBJECTS = 4` 
- Optional: `--record_video`, `--no-save_trajectory`, `--summary_dir`, `--run_stats_dir`, `--traj_dir`.

### 5. Run baselines

```bash
./isaaclab.sh -p $T/standard_cbf/evaluation.py --headless --num_envs 100 --seed 0 --scene_type large
sbatch $T/standard_cbf/run_baseline_eval_sweep.sh   # multi-seed sweep; select baselines in ENTRIES
```

`evaluation.py` options: `--cbf_alpha`, `--cbf_safety_margin`, `--cbf_adaptive`. The other baselines run the same way.

### Visualize CBF values

```bash
python $T/manip_ncbf/cbf_test_real.py --trajectory_file <json> --model_path <pt> --output_dir <dir>
```

## File reference

**Root (`$T/`)**

| File | Function |
|---|---|
| `ncbf_manip_dis_random_vel.py` | Collect demonstrations via random-velocity exploration → `trajectories.json` |
| `collect_data.py` | Trajectory recording module (imported, not run directly) |
| `data_reformat.py` | Convert `trajectories.json` to NCBF training format (npz) |
| `get_refined_demonstrations_hz_unsafe.py` | Resample near the boundary with the initial NCBF, relabel, and refine online |

**`manip_ncbf/`**: training and visualization (current version; `model/` holds the older equivalent)

| File | Function |
|---|---|
| `training.py` | Train the NCBF (hyperparameters in `config.py`) |
| `config.py` / `models.py` / `helpers.py` | Hyperparameters / NCBF network and Normalizer / utilities |
| `cbf_test_real.py` | Plot CBF value maps from `trajectories.json` |

**`inference_multi_modular*/`**: same structure (`main.py` + `evaluator.py` / `data_handler.py` / `stats_analyzer.py`), different nominal policy

| Directory | Nominal policy |
|---|---|
| `inference_multi_modular/` | Default pushing policy + NCBF |
| `inference_multi_modular_apf/` | APF (artificial potential field) + NCBF |
| `inference_multi_modular_rrt/` | RRT planning + NCBF |

**`standard_cbf/`**: baselines

| File | Function |
|---|---|
| `evaluation.py` | Standard QP-CBF safety filter (`cbf_safety_filter.py`) |
| `baseline_apf.py` | APF baseline (`apf_util.py`) |
| `baseline_naive_backward.py` | Revert to the previous action when unsafe |
| `baseline_do_nothing.py` | No safety mechanism |
| `rrt_planer.py` | RRT planning module |

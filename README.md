# Lunar Lander: selected DQN and PPO models

This repository contains the two models selected after the Lunar Lander optimization work: a **100,000-step DQN** and a **slow-landing PPO**. It also preserves the original `My_Lander` classroom project and its records. All models use Gymnasium `LunarLander-v3` (discrete actions) and Stable-Baselines3. Existing checkpoints and result files let readers inspect the outcomes without training again.

## Which result belongs to which model?

| Scope | Checkpoint | Training | Recorded evaluation |
|---|---|---|---|
| **Selected DQN** | `research/selected/softer_dqn_seed123.zip` | DQN, seed 123, learning rate 0.0003, **100,000** timesteps | `research/comparisons/phase2_test2.json`: 100 held-out episodes, mean reward **232.13**, 96% safe rest, 93% at 200 points |
| **Selected slow PPO** | `research/selected/slow_ppo_v3_seed42.zip` | PPO fine-tuning seed 42; selected checkpoint at **1,409,024 cumulative** timesteps | `research/comparisons/phase8_test8.json`: 200 held-out episodes, mean reward **235.47**, 200/200 safe and centered |
| Original classroom run | `artifacts/best_model.zip` | DQN, seed 42, learning rate 0.0001, **200,000** timesteps | `evaluation.json`: 20 episodes, mean reward **−98.057**, 0% two-leg contact, 0% at 200 points |
| Classroom trial archive | `experiment/*/best_model.zip` where present | DQN or PPO, each run's settings in its own `training_summary.json` | Matching `experiment/*/evaluation.json` |

The selected PPO is an **extended experiment** beyond the course competition's 200,000-timestep final-run limit. The original classroom DQN, selected DQN, and PPO use different checkpoints and evaluation sets; compare results only within a recorded evaluation round.

`research/selected/models.json` stores SHA-256 hashes for all five preserved research candidates, including the two selected models above. Their training settings and validation histories are in `research/runs/`. The controlled DQN learning-rate comparison uses the same 100 test starts for six 100,000-step runs; see `research/PHASE2_RESULTS.md`. After that comparison, the seed-123 DQN run continued to 200,000 steps, so its later `best_landing.zip` and `training_summary.json` describe the extended run. The exact original 100,000-step checkpoint is preserved both as `research/selected/softer_dqn_seed123.zip` and `research/runs/dqn_zoo_lr0003_seed123/checkpoints/step_000100000.zip`. Use that immutable checkpoint to reproduce the 232.13 result. The PPO approach and its trade-offs are described in `research/SLOW_LANDING_RESULTS.md`.

## Set up

Use **Python 3.12**. From this directory:

**macOS / Linux**

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Windows PowerShell**

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements.txt` pins the course environment. `requirements-research.txt` adds packages used to render and inspect the research videos. Install it when using those scripts:

```sh
.venv/bin/python -m pip install -r requirements-research.txt
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe` in later commands. Gymnasium Box2D installation may require the platform's compiler/build tools.

## Inspect and run

The committed JSON files are the original measured results. The commands below are optional checks; no training is needed to inspect the repository.

```sh
# Re-run the classroom 20-seed evaluation (rewrites evaluation.json):
.venv/bin/python evaluate.py --episodes 20

# Open the classroom dashboard:
.venv/bin/python -m streamlit run app.py

# Re-evaluate the selected DQN on the same 100 starts used in phase 2:
.venv/bin/python research/evaluate_checkpoint.py --algorithm dqn --model research/selected/softer_dqn_seed123.zip --seed-file research/phase2_seeds.json --seed-key test2

# Re-evaluate the slow PPO on the phase 8 held-out 200 starts:
.venv/bin/python research/evaluate_checkpoint.py --algorithm ppo --model research/selected/slow_ppo_v3_seed42.zip --seed-file research/phase8_seeds.json --seed-key test8
```

Training entry points are `train.py` for the classroom model, `research/train.py` for DQN studies, and `research/train_ppo.py` for PPO studies. Their `--help` output lists options. Training creates new checkpoints and may replace files in a chosen run directory; use a new `--run-dir` for a new research run. Saved `config.json` files contain the exact settings of historical runs. Some historical JSON paths record the previous machine location; use the relative paths in this README after cloning.

## Videos and retained evidence

`demo_video/landing-episode-0.mp4` is the video produced by the original classroom recorder; it is not a performance summary. `research/videos/` contains recordings for the selected research models. `ten_consecutive_dqn_seed123_start271828182.mp4` shows ten consecutive DQN trials; `ten_consecutive_slow_ppo_v3_seed42_start1600619302.mp4` shows ten consecutive PPO trials. `top20_from_400_slow_ppo_v3_seed42.mp4` contains the **selected** calmest 20 PPO episodes from 400 tests. The 11-second `rank1_landing_edited_11s_clean.mp4` has edited playback duration; its original episode lasted 8.74 seconds. Video metadata JSON files preserve seed and provenance.

The root course evaluation and its matching model are preserved. `experiment/` holds smaller classroom trials. `research/comparisons/`, `research/diagnostics/`, `research/runs/` JSON/CSV histories, and `research/selected/` hold the research evidence. `research/runs/` intermediate checkpoint ZIPs and DQN replay buffers are intentionally omitted from Git because the selected checkpoints and settings are already retained. They can be regenerated by new training runs, not by a simple download.

## Submission note

The course manual also describes a separate submission package. `training_weights.md` summarizes the two selected research models; the root `artifacts/` directory records the separate classroom DQN. Check Blackboard's current file list and eligibility rules before submitting a particular checkpoint.

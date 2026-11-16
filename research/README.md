# Lunar Lander extended experiments

This folder adds experimental training and evaluation while leaving the classroom
`train.py`, `evaluate.py`, `app.py`, and their outputs untouched. A copy of the
classroom source and the DQN 200k model is in `classroom_snapshot/` and
`baselines/current_dqn_200k/` respectively.

Run commands from the `My_Lander` project root with its Python 3.12 environment.

## Why this runner exists

The classroom runner saves only the final policy as `best_model.zip`. The DQN
200k run ended much worse than the DQN 100k result. This runner evaluates at
fixed intervals, retains every checkpoint, and separately identifies the best
validation mean-reward and best validation two-leg-contact checkpoints. These
are two candidates, not guarantees of success on unseen episodes.

`seeds.json` has 30 fixed validation seeds and 100 untouched final test seeds.
The old 20 classroom seeds are listed as `legacy_practice` and are not a new
holdout because they were already used to compare earlier experiments.

## Short pipeline check

```sh
.venv/bin/python research/train.py --run-dir research/runs/smoke_dqn --timesteps 5000 --eval-every 2500 --validation-count 3
.venv/bin/python research/evaluate.py --run-dir research/runs/smoke_dqn --model best_reward --split validation --count 3
```

The smoke run checks the pipeline, not model quality. Inspect `config.json`,
`training_summary.json`, `validation_history.jsonl`, `best_reward.zip`,
`best_landing.zip`, `final_model.zip`, `resume_latest.zip`, and
`resume_replay.pkl` inside its run folder. Validation episode JSONs also record
`terminated` and `truncated` separately. `two_leg_landing_rate` means both legs
contact the ground at episode end; it is not proof of a safe landing.
New runs also record `safe_landing_rate`: the share of episodes that terminate
with LunarLander-v3's exact `+100` rest reward. `best_safe.zip` selects by this
rate on validation seeds, with mean reward as a tie breaker. Older runs made
before this metric was added retain their original checkpoint names.

To check recovery, extend the same smoke run to 10k steps, keeping its settings:

```sh
.venv/bin/python research/train.py --run-dir research/runs/smoke_dqn --timesteps 10000 --eval-every 2500 --validation-count 3 --resume
```

Resume loads the model and DQN replay buffer and continues to an **absolute**
step target. It starts a fresh environment episode and does not promise exact
bit-for-bit continuation after an interruption. It resumes from the latest
saved checkpoint, so work since that checkpoint may need repeating.

## Long-run example

After the smoke and recovery checks pass, use a new run folder:

```sh
.venv/bin/python research/train.py --run-dir research/runs/dqn_default_seed42 --timesteps 300000 --seed 42 --eval-every 20000
```

Other DQN settings can be changed with `--learning-rate`, `--batch-size`,
`--buffer-size`, `--learning-starts`, `--target-update-interval`,
`--exploration-fraction`, `--exploration-final-eps`, `--gradient-steps`, and
`--net-arch`. Use a distinct run folder for each configuration and training
seed. The classroom 200k competition cap does not apply to this extended run;
models above 200k may be ineligible for the classroom competition.

Compare candidates using the fixed **validation** seeds. Once the configuration
and checkpoint are chosen, evaluate the **test** seeds once:

```sh
.venv/bin/python research/evaluate.py --run-dir research/runs/dqn_default_seed42 --model best_reward --split test
```

Do not choose a new configuration based on the test result. For robustness,
repeat promising configurations with independent training seeds, such as 7 and
123, while retaining the same validation and test protocols.

For the completed three-seed comparison, `final_selection.json` freezes the
classroom and tuned checkpoint paths and the selection rule before the final
test. `compare.py` evaluates all six models with identical episode seeds and
settings, and records full episode rows and model file hashes:

```sh
.venv/bin/python research/compare.py --split validation
.venv/bin/python research/compare.py --split test
```

The test command uses all 100 held-out seeds and refuses to overwrite an
existing result. Model choice must remain based on validation results.

The second-stage lower-learning-rate experiment, its validation failure
diagnosis, and a separate one-time held-out comparison are documented in
`PHASE2_RESULTS.md`. Its frozen selection and seeds are in
`phase2_selection.json` and `phase2_seeds.json`; its full result is
`comparisons/phase2_test2.json`.

The later pad-placement and gentle-touchdown experiments are summarized in
`PHASE3_RESULTS.md`. Immutable copies of the two current candidate policies
are in `selected/`. The PPO and DQN ten-consecutive-episode videos are in
`videos/`.

The later slower-descent PPO experiment and its two fresh held-out tests are
documented in `SLOW_LANDING_RESULTS.md`. Its source checkpoint, acceptance
criteria, selected models, and unselected ten-episode video remain under
`research/`.

## Measures

Mean reward and population-standard-deviation use per-episode raw reward sums.
Two-leg contact rate checks the final observation's leg flags. The 200-point
rate checks the raw reward sum. Within each episode, action-switch rate is the
number of changes between consecutive actions divided by `max(steps - 1, 1)`;
the reported rate averages the episode rates equally. Truncation rate reports
the fraction of episodes stopped by the time limit. All per-episode rows are
retained, including failures.

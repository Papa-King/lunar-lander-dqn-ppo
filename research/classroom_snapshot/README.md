# Lunar Lander AI Lab: Train, Test, and Improve an Agent

For the classroom procedure, use `Lunar_Lander_Lab_Manual_Official_2026-11.docx`. The supported class environment is CPython 3.12.x inside the project-local `.venv` folder. Do not uninstall a newer system Python.

## Lab at a glance

**Audience:** Beginners with little or no reinforcement-learning experience  
**Suggested duration:** 90 minutes in class + up to 30 minutes optional competition training  
**Mode:** Individual or pairs  
**Learning outcome:** Use an AI coding assistant to run, test, improve, and explain a small reinforcement-learning system.

By the end of the lab, you will be able to:

1. explain state, action, reward, episode, and policy using Lunar Lander;
2. train one reinforcement-learning agent with a time or step budget;
3. test the trained agent on seeds held out from training;
4. show your result in a small web dashboard;
5. report what the AI assistant changed and what you verified yourself.

You are **not** expected to implement DQN or PPO from scratch. The challenge is to build a reliable experiment and explain it.

## The mission

Train an agent to land a spacecraft safely. The environment gives the agent eight observations, including position, velocity, angle, angular velocity, and whether the two legs touch the ground. The discrete version has four actions: do nothing, fire the left engine, fire the main engine, or fire the right engine.

A score of **200 or above** is the environment's conventional solved-task threshold. It is not identical to a safe landing, so this lab reports both the 200-point rate and the two-leg landing rate. One lucky landing is not enough: your final result is evaluated over 20 seeds that were not used for training.

Reference project: [juliankappler/lunar-lander](https://github.com/juliankappler/lunar-lander). It is useful for seeing a from-scratch Actor-Critic/DQN implementation, but this lab uses the maintained `LunarLander-v3` environment and Stable-Baselines3 to reduce setup and coding difficulty.

## What you will submit

Submit one ZIP file named `studentID_lunar_lander.zip` containing:

- `app.py` — your working web dashboard;
- `train.py`, `evaluate.py`, and `record_video.py` — your final code;
- `artifacts/best_model.zip` and `artifacts/training_summary.json` — your trained agent weights and training record (keep this folder structure);
- `evaluation.json` — results from exactly 20 official test episodes;
- `landing.mp4` or `landing.gif` — a 15–30 second demonstration;
- `reflection.md` — completed from `REFLECTION_TEMPLATE.md`;
- optional: `training_log.csv` or screenshots of the learning curve.

Your reflection must disclose:

- the coding assistant/product and underlying model name, if shown;
- the reinforcement-learning algorithm used;
- total training timesteps and wall-clock training time;
- actual number of episodes completed during training;
- the prompts or instructions that produced the most important change;
- one failed attempt, what evidence showed it failed, and what you changed;
- what you personally tested and can explain.

Do not submit API keys, passwords, tokens, or the `.venv` folder.

## Recommended path

### Level 0 — Smoke test (about 5 minutes)

Create the supported Python 3.12 environment and check that Lunar Lander opens. These commands do not require PowerShell activation:

```powershell
uv venv --python 3.12
uv pip install -r requirements.txt
.venv\Scripts\python.exe --version
.venv\Scripts\python.exe train.py --algorithm random --timesteps 0
```

The version check must report Python 3.12.x. If Box2D or SWIG appears in an error, delete only the project `.venv` folder and recreate it with `uv venv --python 3.12`. Do not repeatedly reinstall unrelated packages and do not uninstall another Python version.

### Level 1 — First trained agent (about 15 minutes)

```powershell
.venv\Scripts\python.exe train.py --algorithm dqn --timesteps 20000 --seed 42
```

This is a pipeline check, not a strong final model. Confirm that these files appear:

- `artifacts/best_model.zip`
- `artifacts/training_summary.json`

### Level 2 — Recommended final run (about 15–30 minutes)

Choose **one** option:

```powershell
.venv\Scripts\python.exe train.py --algorithm dqn --timesteps 100000 --seed 42
```

or

```powershell
.venv\Scripts\python.exe train.py --algorithm ppo --timesteps 100000 --seed 42
```

Actual time depends on the computer. Stop if your machine becomes unstable. A shorter, documented experiment is better than an unexplained long run.

### Level 3 — Evaluate fairly (about 5 minutes)

```powershell
.venv\Scripts\python.exe evaluate.py --episodes 20
```

Do not train on the published practice-evaluation seeds. Do not manually remove crashes from the result. The instructor will rerun competition entries with separate hidden seeds.

### Level 4 — Open the web dashboard (about 5 minutes)

First create a demonstration video:

```powershell
.venv\Scripts\python.exe record_video.py
```

Copy the generated MP4 from `demo_video/` to the project root and rename it `landing.mp4`. Then open the dashboard:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

Your dashboard should show:

- algorithm and training budget;
- training time;
- mean score and score variation;
- two-leg landing rate and 200-point rate;
- mean action-switch rate (our simple smoothness indicator);
- your demonstration video, if available;
- one sentence explaining what you changed.

### Level 5 — Required: improve one thing (about 20 minutes)

Change only **one main variable** so that you can explain cause and effect. Examples:

- DQN versus PPO with the same number of timesteps;
- 20,000 versus 100,000 timesteps;
- one learning-rate change;
- one network-size change;
- one random seed change to test robustness.

Record the before/after result. Do not change five settings at once.

## Which reinforcement-learning algorithm should I use?

| Choice | When to use it | Suggested first budget | What to expect |
|---|---|---:|---|
| **DQN (recommended)** | Default discrete Lunar Lander task | 20k smoke test; 100k final | Easy mental model and sample reuse; may need more wall-clock time |
| **PPO** | You want a stable alternative or faster parallel training later | 50k–100k | Often straightforward; results still vary by seed |
| **Random policy** | Pipeline check only | 20 episodes, no training | Establishes how poor an untrained agent is |

For this beginner lab, use the **discrete** environment. Continuous control would require a different action space and algorithms such as SAC or TD3; it is outside the required scope.

## Training budget and stopping rule

### Episode versus timestep

One **episode** starts when the lander appears and ends when it lands, crashes, or leaves the allowed area. One **timestep** is one observe-decide-act moment inside an episode. Different episodes have different lengths, so this lab fixes the training budget in timesteps for fairness and automatically records how many episodes were completed. In your report, always include all three: timesteps, episodes, and wall-clock time.

Use the following guide, not a promise of exact clock time:

- **20,000 timesteps:** quick smoke test, often a few minutes;
- **100,000 timesteps:** recommended submission budget, often several to tens of minutes;
- **200,000 timesteps:** optional competition ceiling, possibly tens of minutes.

Hardware and software versions affect runtime. Record both `total_timesteps` and measured wall-clock seconds. For the class competition, the enforceable maximum is **200,000 timesteps per final run**. Training time must still be disclosed, but it does not affect the official overall leaderboard because student hardware differs.

## AI coding prompts you may reuse

Start with a narrow request:

> Read the error and explain the smallest fix. Do not rewrite the whole project. After the fix, give me one command that verifies it.

For code understanding:

> Explain this Lunar Lander training file to a beginner. Identify the environment, algorithm, training budget, saved output, and evaluation method. Use the exact variable names from the code.

For one controlled improvement:

> My current DQN result is [paste evaluation]. Suggest one change only. Explain why it may help, what could get worse, and how I can compare before and after fairly using the same evaluation seeds.

For dashboard work:

> Add a simple Streamlit results page that reads evaluation.json. Show five clear metrics and the demo video. Do not start training inside Streamlit.

For verification:

> Inspect my result files and create a checklist of claims that are supported by evidence. Flag anything I cannot prove from the files.

## Minimum success and recovery route

You can still complete the lab if your model does not reach 200 points.

Minimum pass evidence:

1. the code runs;
2. a model is saved;
3. all 20 evaluation episodes are included;
4. the dashboard displays the real results;
5. the reflection honestly explains one failure and one next step.

If training fails or is too slow:

1. run the random policy to verify the environment;
2. reduce the smoke test to 5,000 timesteps;
3. verify one short DQN run;
4. use the provided baseline model only for dashboard testing, clearly labelled as provided;
5. submit your diagnostic evidence and reflection.

## Assessment rubric (100 points)

| Area | Points | Evidence |
|---|---:|---|
| Working pipeline | 20 | Code runs, saves a model, produces `evaluation.json`, and opens the dashboard |
| Fair evaluation | 20 | Exactly 20 fixed practice seeds; all episodes included; mean, standard deviation, two-leg landing rate, and 200-point rate reported |
| Agent quality | 20 | Improvement over random baseline and evidence of controlled iteration; marks are not based only on reaching 200 |
| Web communication | 15 | Clear metrics, readable labels, demo media, and a short explanation for a non-expert |
| Reflection and AI disclosure | 20 | Model/tool disclosure, training time/steps, prompts, failed attempt, verification, and honest limitations |
| Reproducibility | 5 | Requirements, seed, algorithm, and run commands are complete |

An agent with a lower score can still earn a strong grade if the experiment is honest, reproducible, and well explained. Competition prizes are separate from the academic mark.

## Mini competition: Smooth Landing Cup

All entries must use `LunarLander-v3`, the discrete action space, and no more than the competition budget. The published seeds in `evaluate.py` help you compare experiments. For the leaderboard, the instructor reruns every submitted model on the same machine with 50 hidden seeds. Hidden seeds must not be used for training or tuning.

### Awards

- **Best Pilot:** highest mean reward across the instructor's 50 hidden episodes;
- **Most Reliable:** highest two-leg landing rate across the 50 hidden episodes, with mean reward as tie-breaker;
- **Smooth Operator:** lowest action-switch rate on the 50 hidden episodes among agents with mean reward at least 200;
- **Sample-Efficient Learner:** highest hidden-seed mean reward among models trained with no more than 100,000 timesteps;
- **Best Scientific Reflection:** clearest evidence-based explanation of one successful and one failed change.

Optional, non-ranking showcase: the instructor may recognize the shortest self-reported training time, but the entry must disclose CPU/GPU and operating system. This showcase is not part of the official leaderboard because hardware differs.

### Overall leaderboard score

The instructor calculates every component below from the 50 hidden episodes after collecting the class results (except training timesteps, which come from the submitted training record):

- 50% performance: class-normalized mean reward;
- 20% reliability: two-leg landing rate;
- 15% smoothness: class-normalized low action-switch rate;
- 15% sample efficiency: class-normalized low training timesteps.

For each class-normalized metric, the instructor applies min–max scaling across eligible entries and clips the result to 0–1. For lower-is-better metrics, the direction is reversed. If all entries have the same value, all receive 1 for that component. Overall ties are broken by higher mean reward, then fewer training timesteps.

Eligibility requires a runnable model, all 20 raw practice-episode records, training budget disclosure, and an honest AI-use statement. The instructor-generated hidden evaluation is authoritative for the leaderboard. A result is disqualified from the leaderboard—not from the academic lab—if evidence is missing, the model cannot be rerun, or hidden test information is used for training.

The action-switch metric is intentionally simple. It measures how often the chosen action differs from the previous action; it is a proxy for visual jitter, not a complete measure of flight quality.

## Safety, honesty, and ownership

AI assistance is expected. You remain responsible for the submission.

- Keep prompts or a short AI interaction log.
- Test generated code before claiming it works.
- Do not claim a model name if the product does not reveal it; write “not disclosed by the product.”
- Do not download an existing trained model and present it as your training result.
- Be ready to explain any line you changed and any metric you report.

## References

- [Original reference repository](https://github.com/juliankappler/lunar-lander)
- [Gymnasium Lunar Lander documentation](https://gymnasium.farama.org/environments/box2d/lunar_lander/)
- [Stable-Baselines3 algorithm guidance](https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html)
- [Stable-Baselines3 examples](https://stable-baselines3.readthedocs.io/en/master/guide/examples.html)

# Training Weights

I used an AI coding assistant and Stable-Baselines3 to train and evaluate DQN and PPO agents in `LunarLander-v3`. The two selected research weights are separate checkpoints:

| Model | Weight file | Training | Evaluation |
|---|---|---|---|
| DQN | `research/selected/softer_dqn_seed123.zip` | Seed 123; learning rate 0.0003; **100,000 timesteps** | 100 held-out episodes: mean reward **232.13**, safe landing **96%**, at least 200 points **93%** |
| Slow PPO | `research/selected/slow_ppo_v3_seed42.zip` | Fine-tuning seed 42; learning rate 0.0001; selected checkpoint at **1,409,024 cumulative timesteps** | 200 held-out episodes: mean reward **235.47**, safe and centered **200/200**, gentle first contact **179/200** |

For the controlled DQN experiment, I changed the learning rate from **0.00063** to **0.0003**, kept the other settings and 100 evaluation starts the same, and repeated both settings with training seeds 42, 7, and 123. The lower rate helped some seeds but reduced the safe landing rate for seed 42, so the result was not uniform.

The PPO model was fine-tuned from a shared pretrained checkpoint. Its training reward encouraged slower descent near the landing pad; evaluation used the original environment reward. The fine-tuning seeds are not independent PPO trainings from scratch.

The `.md` file describes the models. The actual weights are the two `.zip` files above. Full settings and per-episode evidence are in `research/PHASE2_RESULTS.md`, `research/SLOW_LANDING_RESULTS.md`, and `research/comparisons/`.

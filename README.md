# Ball Agent Game

CE6127 Assignment 2 RL pipeline for the Unity ball game.

The Unity project/builds live in `unity/`. The Python side in `src/` trains one PPO agent per environment: `simple`, `medium`, and `hard`.

The training entrypoint now uses per-environment PPO presets:

- `simple` and `medium` keep the baseline PPO settings.
- `hard` uses a separate, more conservative PPO preset with a lower learning rate, longer rollouts, larger batches, and slightly stronger exploration.
- After the last hard-case improvement, the hard preset was tuned again to favor more stable updates and a bit less entropy so the agent can convert its mostly-good behavior into more consistent wins.

## PPO Configurations

These are the default PPO hyperparameters currently used by `src/train.py`.

### Simple

- `learning_rate=3e-4`
- `n_steps=2048`
- `batch_size=64`
- `gamma=0.99`
- `gae_lambda=0.95`
- `clip_range=0.2`
- `ent_coef=0.01`

Reasoning:
Chosen as the baseline PPO setup for the easiest environment, where faster policy updates and smaller batches are acceptable.

### Medium

- `learning_rate=3e-4`
- `n_steps=2048`
- `batch_size=64`
- `gamma=0.99`
- `gae_lambda=0.95`
- `clip_range=0.2`
- `ent_coef=0.01`

Reasoning:
Medium currently reuses the baseline configuration because the same setup is expected to remain stable while still learning efficiently.

### Hard

- `learning_rate=7.5e-5`
- `n_steps=8192`
- `batch_size=512`
- `gamma=0.995`
- `gae_lambda=0.98`
- `clip_range=0.12`
- `ent_coef=0.015`

Reasoning:
The hard environment is treated as a separate PPO problem. The learning rate is reduced further to make updates less aggressive, rollout length and batch size are increased again to stabilize gradient estimates, `gamma` and `gae_lambda` remain tuned for longer-horizon behavior, `clip_range` is reduced to constrain policy drift, and `ent_coef` is trimmed slightly so training spends a bit less effort on exploration now that the agent already solves most hard scenarios.

Report note:
If any run uses CLI overrides such as `--learning-rate` or `--n-steps`, the exact resolved values are written to `runs/<env>/<run-name>/train_config.json`. That file should be cited as the source of truth for the final reported experiment settings.

## Requirements

- A working Python environment with the packages already installed in `.venv`
- The Unity Linux builds present at:
  - `unity/SimpleBuild/simple.x86_64`
  - `unity/MediumBuild/medium.x86_64`
  - `unity/HardBuild/hard.x86_64`

## Train

Train one environment at a time. Use different worker IDs if you run more than one Unity build at once.

```bash
train.py --env simple --total-timesteps 300000 --no-graphics --train-worker-id 0 --eval-worker-id 1 --run-name simple_main
train.py --env medium --total-timesteps 300000 --no-graphics --train-worker-id 10 --eval-worker-id 11 --run-name medium_main
train.py --env hard --total-timesteps 300000 --no-graphics --train-worker-id 20 --eval-worker-id 21 --run-name hard_main
```

For the hard environment, a longer run is usually more useful than reusing the easy/medium budget unchanged:

```bash
train.py --env hard --total-timesteps 1000000 --no-graphics --train-worker-id 30 --eval-worker-id 31 --run-name hard_tuned_v2
```

You can still override the preset from the CLI for ablations or follow-up tuning:

```bash
train.py --env hard --run-name hard_ablation --learning-rate 5e-5 --n-steps 8192 --batch-size 512
```

You can also resume training from an existing checkpoint instead of starting from zero:

```bash
.venv/bin/python src/train.py --env hard --total-timesteps 500000 --time-scale 40 --eval-freq 100000 --eval-episodes 10 --checkpoint-freq 100000 --no-graphics --train-worker-id 30 --eval-worker-id 31 --run-name hard_tuned_v2_resume --resume-model runs/hard/hard_tuned_v2/best_model.zip
```

This continues optimization from the saved policy weights in the specified `.zip` model and writes a fresh run directory for the resumed experiment.

Training outputs are written to `runs/<env>/<run-name>/`:

- `checkpoints/`: intermediate `.zip` checkpoints
- `best_model.zip`: best checkpoint by evaluation win rate
- `final_model.zip`: model saved at the end of training
- `evaluations.csv`: periodic evaluation metrics
- `tensorboard/`: TensorBoard logs
- `train_config.json`: saved run configuration
- The saved `train_config.json` includes the exact resolved PPO hyperparameters used for that run.

## Evaluate

`evaluate.py` is the non-visual version of checking the agent by eye. It runs a fixed number of episodes and writes measurable results such as wins, losses, mean reward, and mean episode length.

Use it when you need quantitative evidence for the report or when you want to compare checkpoints without relying only on visual inspection.

Evaluate any saved checkpoint or final model:

```bash
evaluate.py --env simple --model runs/simple/simple_main/final_model.zip --episodes 20 --no-graphics --worker-id 2
```

Optionally save the evaluation summary:

```bash
evaluate.py --env simple --model runs/simple/simple_main/best_model.zip --episodes 20 --no-graphics --worker-id 2 --out runs/simple/simple_main/eval_once.csv
```

Practical difference:

- `play.py`: qualitative check, useful for visually confirming behavior.
- `evaluate.py`: quantitative check, useful for the report and for comparing models consistently.
- `train.py`: already performs periodic evaluation during training and writes `evaluations.csv`, so manual evaluation after training is optional but still useful for a final summary file.

## Replay

Run a checkpoint visually for manual inspection or screen recording:

```bash
play.py --env simple --model runs/simple/simple_main/best_model.zip --time-scale 1.0 --worker-id 3
```

Use `--episodes N` if you want it to stop after a fixed number of games.

## Plot Curves

Generate a simple plot from `evaluations.csv`:

```bash
plot_metrics.py --csv runs/simple/simple_main/evaluations.csv --out runs/simple/simple_main/eval_plot.png
```

## Notes

- SB3 PPO collects data in rollout chunks of `n_steps`, so a run may finish slightly above the exact `--total-timesteps` you request.
- The message `UnityWorkerInUseException: worker number X is still in use` means that worker ID is already occupied, usually by another live Unity process or by a previously crashed run that did not exit cleanly.
- If that happens, the fastest fix is to rerun with different `--train-worker-id` / `--eval-worker-id`.
- Recommended worker ID convention:
  - `simple`: `10/11`
  - `medium`: `20/21`
  - `hard`: `30/31`
- If you want to free a stuck worker ID, inspect running Unity processes:

```bash
ps -ef | grep -E 'simple.x86_64|medium.x86_64|hard.x86_64'
```

Then kill the stale process:

```bash
kill <pid>
```

Use `kill -9 <pid>` only if a normal `kill` does not stop it.
- `for_agent.md` is a local notes file for future debugging context and is intentionally ignored by git.

import csv
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from env_compat import reset_env, step_env
from make_env import classify_outcome


EVAL_FIELDNAMES = [
    "timesteps",
    "episodes",
    "wins",
    "draws",
    "losses",
    "win_rate",
    "draw_rate",
    "loss_rate",
    "mean_reward",
    "mean_length",
]


class PeriodicEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq: int,
        n_eval_episodes: int,
        output_csv: Path,
        best_model_path: Path,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.output_csv = output_csv
        self.best_model_path = best_model_path
        self.deterministic = deterministic
        self.best_win_rate = float("-inf")

    def _init_callback(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.output_csv.exists():
            with self.output_csv.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=EVAL_FIELDNAMES)
                writer.writeheader()

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq != 0:
            return True

        stats = evaluate_policy_run(
            model=self.model,
            env=self.eval_env,
            n_episodes=self.n_eval_episodes,
            deterministic=self.deterministic,
        )
        stats["timesteps"] = self.num_timesteps

        with self.output_csv.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=EVAL_FIELDNAMES)
            writer.writerow(stats)

        self.logger.record("eval/wins", stats["wins"])
        self.logger.record("eval/draws", stats["draws"])
        self.logger.record("eval/losses", stats["losses"])
        self.logger.record("eval/win_rate", stats["win_rate"])
        self.logger.record("eval/draw_rate", stats["draw_rate"])
        self.logger.record("eval/loss_rate", stats["loss_rate"])
        self.logger.record("eval/mean_reward", stats["mean_reward"])
        self.logger.record("eval/mean_length", stats["mean_length"])

        if stats["win_rate"] > self.best_win_rate:
            self.best_win_rate = stats["win_rate"]
            self.model.save(self.best_model_path)

        if self.verbose:
            print(
                "Eval @ "
                f"{self.num_timesteps} steps | "
                f"W/D/L={stats['wins']}/{stats['draws']}/{stats['losses']} | "
                f"win_rate={stats['win_rate']:.3f} | "
                f"mean_reward={stats['mean_reward']:.3f}"
            )

        return True


def evaluate_policy_run(model, env, n_episodes: int, deterministic: bool) -> dict[str, float]:
    wins = 0
    draws = 0
    losses = 0
    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs = reset_env(env)
        done = False
        episode_reward = 0.0
        episode_length = 0
        final_info = {}
        final_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = step_env(env, action)
            episode_reward += float(reward)
            episode_length += 1
            final_info = info
            final_reward = float(reward)

        outcome = classify_outcome(final_reward, done, final_info)
        if outcome == "win":
            wins += 1
        elif outcome == "loss":
            losses += 1
        else:
            draws += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    total = max(1, n_episodes)
    return {
        "episodes": n_episodes,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / total,
        "draw_rate": draws / total,
        "loss_rate": losses / total,
        "mean_reward": sum(episode_rewards) / total,
        "mean_length": sum(episode_lengths) / total,
    }

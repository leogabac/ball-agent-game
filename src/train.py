import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from callbacks import PeriodicEvalCallback
from make_env import make_env


BASE_PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
}

PPO_CONFIGS = {
    "simple": BASE_PPO_CONFIG,
    "medium": BASE_PPO_CONFIG,
    "hard": {
        "learning_rate": 7.5e-5,
        "n_steps": 8192,
        "batch_size": 512,
        "gamma": 0.995,
        "gae_lambda": 0.98,
        "clip_range": 0.12,
        "ent_coef": 0.015,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train one PPO agent per Unity build.")
    parser.add_argument("--env", required=True, choices=["simple", "medium", "hard"])
    parser.add_argument("--total-timesteps", type=int, default=300_000)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-worker-id", type=int, default=0)
    parser.add_argument("--eval-worker-id", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    parser.add_argument("--eval-freq", type=int, default=25_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--resume-model",
        type=Path,
        default=None,
        help="Path to an existing PPO .zip model to continue training from.",
    )
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--gae-lambda", type=float, default=None)
    parser.add_argument("--clip-range", type=float, default=None)
    parser.add_argument("--ent-coef", type=float, default=None)
    return parser.parse_args()


def build_run_dir(runs_dir: Path, env_name: str, run_name: str | None) -> Path:
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{env_name}_{timestamp}"
    return runs_dir / env_name / run_name


def resolve_ppo_config(args) -> dict[str, float | int]:
    config = dict(PPO_CONFIGS[args.env])
    overrides = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def main():
    args = parse_args()
    ppo_config = resolve_ppo_config(args)
    run_dir = build_run_dir(args.runs_dir, args.env, args.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    eval_csv = run_dir / "evaluations.csv"
    best_model_path = run_dir / "best_model"
    final_model_path = run_dir / "final_model"
    config_path = run_dir / "train_config.json"

    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "env": args.env,
        "total_timesteps": args.total_timesteps,
        "time_scale": args.time_scale,
        "seed": args.seed,
        "train_worker_id": args.train_worker_id,
        "eval_worker_id": args.eval_worker_id,
        "checkpoint_freq": args.checkpoint_freq,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "resume_model": str(args.resume_model) if args.resume_model is not None else None,
        "no_graphics": args.no_graphics,
        "ppo_config": ppo_config,
    }
    config_path.write_text(json.dumps(config, indent=2))

    train_env = DummyVecEnv(
        [
            lambda: make_env(
                env_name=args.env,
                time_scale=args.time_scale,
                no_graphics=args.no_graphics,
                worker_id=args.train_worker_id,
                seed=args.seed,
            )
        ]
    )
    eval_env = make_env(
        env_name=args.env,
        time_scale=args.time_scale,
        no_graphics=args.no_graphics,
        worker_id=args.eval_worker_id,
        seed=args.seed + 1,
    )

    print("Observation space:", train_env.observation_space)
    print("Action space:", train_env.action_space)
    print("Run directory:", run_dir)
    print("PPO config:", ppo_config)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_freq),
        save_path=str(checkpoint_dir),
        name_prefix=f"ppo_{args.env}",
    )
    eval_callback = PeriodicEvalCallback(
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        output_csv=eval_csv,
        best_model_path=best_model_path,
    )

    if args.resume_model is not None:
        model = PPO.load(
            str(args.resume_model),
            env=train_env,
            tensorboard_log=str(tensorboard_dir),
            seed=args.seed,
            **ppo_config,
        )
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            verbose=1,
            tensorboard_log=str(tensorboard_dir),
            seed=args.seed,
            **ppo_config,
        )

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            tb_log_name="ppo",
        )
        model.save(final_model_path)
    finally:
        eval_env.close()
        train_env.close()


if __name__ == "__main__":
    main()

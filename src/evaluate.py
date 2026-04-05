import argparse
import csv
import json
from pathlib import Path

from stable_baselines3 import PPO

from callbacks import evaluate_policy_run
from make_env import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a PPO checkpoint.")
    parser.add_argument("--env", required=True, choices=["simple", "medium", "hard"])
    parser.add_argument("--model", required=True, help="Path to a saved PPO .zip file.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--time-scale", type=float, default=20.0)
    parser.add_argument("--no-graphics", action="store_true")
    parser.add_argument("--worker-id", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_env(
        env_name=args.env,
        time_scale=args.time_scale,
        no_graphics=args.no_graphics,
        worker_id=args.worker_id,
        seed=args.seed,
    )
    model = PPO.load(args.model)

    try:
        stats = evaluate_policy_run(
            model=model,
            env=env,
            n_episodes=args.episodes,
            deterministic=True,
        )
    finally:
        env.close()

    print(json.dumps(stats, indent=2, sort_keys=True))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(stats.keys()))
            writer.writeheader()
            writer.writerow(stats)


if __name__ == "__main__":
    main()

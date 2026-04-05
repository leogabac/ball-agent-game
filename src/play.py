import argparse

from stable_baselines3 import PPO

from env_compat import reset_env, step_env
from make_env import make_env


def parse_args():
    parser = argparse.ArgumentParser(description="Play a trained PPO agent.")
    parser.add_argument("--env", required=True, choices=["simple", "medium", "hard"])
    parser.add_argument("--model", required=True, help="Path to a saved PPO .zip file.")
    parser.add_argument("--episodes", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    env = make_env(
        env_name=args.env,
        time_scale=args.time_scale,
        no_graphics=False,
        worker_id=args.worker_id,
        seed=args.seed,
    )
    model = PPO.load(args.model)

    try:
        obs = reset_env(env)
        episodes_seen = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = step_env(env, action)

            if done:
                episodes_seen += 1
                if args.episodes and episodes_seen >= args.episodes:
                    break
                obs = reset_env(env)
    finally:
        env.close()


if __name__ == "__main__":
    main()

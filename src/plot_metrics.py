import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Plot evaluation metrics from CSV.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.csv)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(df["timesteps"], df["win_rate"], label="Win rate")
    axes[0].plot(df["timesteps"], df["draw_rate"], label="Draw rate")
    axes[0].plot(df["timesteps"], df["loss_rate"], label="Loss rate")
    axes[0].set_ylabel("Outcome rate")
    axes[0].set_title("Evaluation Outcomes")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(df["timesteps"], df["mean_reward"], label="Mean reward")
    axes[1].plot(df["timesteps"], df["mean_length"], label="Mean episode length")
    axes[1].set_xlabel("Environment steps")
    axes[1].set_ylabel("Value")
    axes[1].set_title("Evaluation Summary")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)


if __name__ == "__main__":
    main()

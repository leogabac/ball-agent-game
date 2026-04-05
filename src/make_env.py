from pathlib import Path
from typing import Any

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import (
    EngineConfigurationChannel,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.patch_gym import _patch_env


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNITY_DIR = PROJECT_ROOT / "unity"

BUILD_CONFIGS = {
    "simple": {
        "folder": "SimpleBuild",
        "binary": "simple.x86_64",
    },
    "medium": {
        "folder": "MediumBuild",
        "binary": "medium.x86_64",
    },
    "hard": {
        "folder": "HardBuild",
        "binary": "hard.x86_64",
    },
}


def get_build_path(env_name: str) -> Path:
    key = env_name.lower()
    if key not in BUILD_CONFIGS:
        raise ValueError(
            f"Unknown environment '{env_name}'. Expected one of: "
            f"{', '.join(sorted(BUILD_CONFIGS))}"
        )

    build = BUILD_CONFIGS[key]
    binary_path = UNITY_DIR / build["folder"] / build["binary"]
    if not binary_path.exists():
        raise FileNotFoundError(f"Unity build not found: {binary_path}")
    return binary_path


def make_env(
    env_name: str,
    time_scale: float = 20.0,
    no_graphics: bool = True,
    worker_id: int = 0,
    seed: int | None = None,
) -> Monitor:
    engine_channel = EngineConfigurationChannel()
    file_name = get_build_path(env_name)

    unity_env = UnityEnvironment(
        file_name=str(file_name),
        worker_id=worker_id,
        seed=seed,
        side_channels=[engine_channel],
        no_graphics=no_graphics,
    )
    engine_channel.set_configuration_parameters(time_scale=time_scale)

    env = UnityToGymWrapper(unity_env)
    env = _patch_env(env)
    return Monitor(env)


def classify_outcome(reward: float, done: bool, info: dict[str, Any]) -> str:
    if not done:
        return "incomplete"

    if "outcome" in info:
        return str(info["outcome"])

    if reward > 0:
        return "win"
    if reward < 0:
        return "loss"
    return "draw"

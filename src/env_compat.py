from typing import Any


def reset_env(env) -> Any:
    result = env.reset()
    if isinstance(result, tuple):
        return result[0]
    return result


def step_env(env, action):
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        done = bool(terminated or truncated)
        return obs, reward, done, info
    return result

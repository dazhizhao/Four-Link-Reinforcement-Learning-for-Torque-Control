from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import sys
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.link_allocation_env import LinkAllocationConfig, LinkAllocationEnv


@dataclass(frozen=True)
class TrainConfig:
    algo: str
    policy: str
    total_timesteps: int
    seed: int
    device: str
    learning_starts: int
    buffer_size: int
    batch_size: int
    train_freq: int
    gradient_steps: int
    learning_rate: float
    gamma: float
    tau: float
    eval_episodes: int
    run_name: str
    output_dir: str

    @classmethod
    def load(cls, config_path: str | Path | None = None) -> "TrainConfig":
        path = (
            Path(config_path)
            if config_path
            else ROOT / "configs" / "train_rl.yaml"
        )
        text = path.read_text(encoding="utf-8")
        raw = yaml.safe_load(text) if yaml is not None else json.loads(text)
        return cls(**raw)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train SAC for the link allocation environment.")
    parser.add_argument("--env-config", default=None, help="Path to link allocation env config.")
    parser.add_argument("--train-config", default=None, help="Path to RL train config.")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total timesteps.")
    parser.add_argument("--seed", type=int, default=None, help="Override training seed.")
    parser.add_argument("--run-name", default=None, help="Override output run directory name.")
    parser.add_argument("--output-dir", default=None, help="Override output root directory.")
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=None,
        help="Override deterministic evaluation episode count.",
    )
    return parser


def load_sac():
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor

    return SAC, Monitor


def ensure_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def evaluate_policy(model, env: LinkAllocationEnv, eval_episodes: int) -> dict[str, Any]:
    episode_rewards: list[float] = []
    allocated_history: list[list[float]] = []
    best_reward = float("-inf")
    best_lengths: np.ndarray | None = None
    best_metrics: dict[str, float] | None = None

    for _ in range(eval_episodes):
        obs, _ = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        _, reward, terminated, truncated, info = env.step(action)
        if not terminated or truncated:
            raise RuntimeError("LinkAllocationEnv evaluation must terminate in one step.")

        reward_value = float(reward)
        episode_rewards.append(reward_value)
        lengths = np.asarray(info["allocated_lengths"], dtype=np.float64)
        allocated_history.append(lengths.tolist())
        if reward_value > best_reward:
            best_reward = reward_value
            best_lengths = lengths.copy()
            best_metrics = {
                "workspace_area": float(info["workspace_area"]),
                "inner_radius": float(info["inner_radius"]),
                "outer_radius": float(info["outer_radius"]),
            }

    if best_lengths is None or best_metrics is None:
        raise RuntimeError("Evaluation did not produce any episodes.")

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "episodes": int(eval_episodes),
        "best_reward": best_reward,
        "best_lengths": best_lengths.tolist(),
        "best_metrics": best_metrics,
        "allocated_lengths_history": allocated_history,
    }


def main() -> None:
    args = build_parser().parse_args()
    env_config = LinkAllocationConfig.load(args.env_config)
    train_config = TrainConfig.load(args.train_config)

    if train_config.algo.lower() != "sac":
        raise SystemExit("Only SAC is supported in this training entrypoint.")

    total_timesteps = args.total_timesteps if args.total_timesteps is not None else train_config.total_timesteps
    seed = args.seed if args.seed is not None else train_config.seed
    run_name = args.run_name if args.run_name is not None else train_config.run_name
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else train_config.eval_episodes
    output_root = Path(args.output_dir if args.output_dir is not None else train_config.output_dir)
    run_dir = output_root / "rl_link_alloc" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    SAC, Monitor = load_sac()

    train_env = Monitor(LinkAllocationEnv(config=env_config), filename=str(run_dir / "monitor.csv"))
    eval_env = LinkAllocationEnv(config=env_config)

    model = SAC(
        policy=train_config.policy,
        env=train_env,
        learning_starts=train_config.learning_starts,
        buffer_size=train_config.buffer_size,
        batch_size=train_config.batch_size,
        train_freq=train_config.train_freq,
        gradient_steps=train_config.gradient_steps,
        learning_rate=train_config.learning_rate,
        gamma=train_config.gamma,
        tau=train_config.tau,
        seed=seed,
        device=train_config.device,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False)

    model_path = run_dir / "model_final.zip"
    model.save(str(model_path))

    evaluation = evaluate_policy(model, eval_env, eval_episodes=eval_episodes)
    train_payload = {
        "env_config": asdict(env_config),
        "train_config": asdict(train_config),
        "resolved": {
            "total_timesteps": total_timesteps,
            "seed": seed,
            "eval_episodes": eval_episodes,
            "run_name": run_name,
            "run_dir": str(run_dir),
        },
    }
    ensure_json(run_dir / "train_config.json", train_payload)
    ensure_json(
        run_dir / "evaluation.json",
        {
            "mean_reward": evaluation["mean_reward"],
            "std_reward": evaluation["std_reward"],
            "episodes": evaluation["episodes"],
            "best_reward": evaluation["best_reward"],
            "best_metrics": evaluation["best_metrics"],
        },
    )
    ensure_json(
        run_dir / "best_lengths.json",
        {
            "allocated_lengths": evaluation["best_lengths"],
            "reward": evaluation["best_reward"],
            **evaluation["best_metrics"],
        },
    )

    print(f"saved model to {model_path}")
    print(f"saved evaluation to {run_dir / 'evaluation.json'}")
    print(f"saved best lengths to {run_dir / 'best_lengths.json'}")


if __name__ == "__main__":
    main()

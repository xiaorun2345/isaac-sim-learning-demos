"""顺序对比 17_demo 的专家回放和 SmolVLA 策略推理。

这个脚本不尝试在同一个 Isaac 窗口里同时摆两个场景，而是采用更稳妥的顺序方式：

1. 先回放指定 episode 的专家动作
2. 再读取该 episode 的 `episode_seed`
3. 用同一 seed 启动一条策略推理 episode

这样能保证两边看到的是同一套初始方块摆放，更方便判断模型到底差在哪一步。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = SCRIPT_DIR / "outputs" / "raw"
DEFAULT_POLICY_DIR = (
    SCRIPT_DIR
    / "outputs"
    / "smolvla_isaac_franka_front_top_state18_action4"
    / "checkpoints"
    / "last"
    / "pretrained_model"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=int, default=0, help="先回放哪一条专家轨迹。")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="原始专家 npz 目录。")
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_DIR, help="SmolVLA checkpoint 目录。")
    parser.add_argument("--fps", type=float, default=20.0, help="专家回放节奏。")
    parser.add_argument("--headless", action="store_true", help="无界面运行。")
    parser.add_argument(
        "--spawn-profile",
        choices=("center", "easy", "train"),
        default="train",
        help="策略推理时的摆放分布。默认 train，并配合 episode_seed 尽量复原专家起点。",
    )
    return parser.parse_args()


def resolve_episode_path(raw_dir: Path, episode: int) -> Path:
    path = raw_dir / f"episode_{episode:05d}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Episode file not found: {path}")
    return path


def load_episode_seed_and_task(episode_path: Path) -> tuple[int, str]:
    payload = np.load(episode_path, allow_pickle=True)
    episode_seed = int(payload["episode_seed"]) if "episode_seed" in payload.files else 20260603 + int(episode_path.stem.split("_")[-1])
    task = str(np.asarray(payload["task"]).item())
    return episode_seed, task


def run_subprocess(args: list[str]) -> None:
    print("run:", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    args = parse_args()
    episode_path = resolve_episode_path(args.raw_dir.resolve(), args.episode)
    episode_seed, task = load_episode_seed_and_task(episode_path)

    replay_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "replay_episode.py"),
        "--raw-dir",
        str(args.raw_dir.resolve()),
        "--episode",
        str(args.episode),
        "--count",
        "1",
        "--fps",
        str(args.fps),
    ]
    if args.headless:
        replay_cmd.append("--headless")

    infer_cmd = [
        sys.executable,
        str(SCRIPT_DIR / "infer_policy_in_isaac.py"),
        "--policy-dir",
        str(args.policy_dir.resolve()),
        "--episodes",
        "1",
        "--seed",
        str(episode_seed),
        "--task",
        task,
        "--spawn-profile",
        args.spawn_profile,
    ]
    if args.headless:
        infer_cmd.append("--headless")

    print(f"compare episode={args.episode} episode_seed={episode_seed}", flush=True)
    print("phase 1/2: replay expert", flush=True)
    run_subprocess(replay_cmd)
    print("phase 2/2: run policy inference", flush=True)
    run_subprocess(infer_cmd)


if __name__ == "__main__":
    main()

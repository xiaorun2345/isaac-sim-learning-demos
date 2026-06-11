#!/usr/bin/env python3
"""把 Isaac Sim 采集的原始 NPZ episode 转成 LeRobotDataset。

这个脚本专门服务于 `17_franka_smolvla_data_collection/demo.py` 采出来的
`outputs/raw/*.npz`。转换完成后，输出目录会变成一个标准的 LeRobotDataset，
然后就可以直接交给 `train_smolvla_isaac.sh` 训练。

默认行为：
1. 从 `outputs/raw` 读取原始 episode
2. 只转换 `success=True` 的轨迹
3. 输出到 `outputs/lerobot_dataset`
4. 自动推断图像 / state / action shape
5. 自动根据 `capture_every_steps` 推断数据集 fps

推荐运行环境：

    conda activate /home/mkls/xiao_run/.conda-lerobot-smolvla
    python isaac-sim-learning-demos/17_franka_smolvla_data_collection/convert_npz_to_lerobot_dataset.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RAW_DIR = SCRIPT_DIR / "outputs" / "raw"
DEFAULT_DATASET_ROOT = SCRIPT_DIR / "outputs" / "lerobot_dataset"
DEFAULT_REPO_ID = "local/isaac_franka_front_wrist_state15_action4"
DEFAULT_ROBOT_TYPE = "isaacsim_franka_panda"
DEFAULT_BASE_SIM_FPS = 60.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="原始 NPZ episode 目录。",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="输出 LeRobotDataset 根目录。",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="写入 LeRobotDataset metadata 的 repo_id。需和训练脚本保持一致。",
    )
    parser.add_argument(
        "--robot-type",
        default=DEFAULT_ROBOT_TYPE,
        help="写入 metadata 的 robot_type。",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="手动指定数据集 fps。默认按 base_sim_fps / capture_every_steps 推断。",
    )
    parser.add_argument(
        "--base-sim-fps",
        type=float,
        default=DEFAULT_BASE_SIM_FPS,
        help="用于推断数据集 fps 的基础仿真频率。默认按 Isaac 常见 60Hz 处理。",
    )
    parser.add_argument(
        "--include-failures",
        action="store_true",
        help="默认只转换成功轨迹；加上这个参数会把失败轨迹也一起导出。",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="最多转换多少条轨迹，主要用于小规模测试。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出目录已存在，则先删除再重建。",
    )
    parser.add_argument(
        "--no-videos",
        action="store_true",
        help="不把图像编码成视频，而是按图片序列写入。适合先做小规模调试。",
    )
    return parser.parse_args()


def import_lerobot_dataset():
    try:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        return LeRobotDataset


def load_npz_payload(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        payload = {name: data[name] for name in data.files}
    return payload


def scalar_to_python(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    return value


def load_metadata(payload: dict[str, Any], episode_path: Path) -> dict[str, Any]:
    raw_metadata = scalar_to_python(payload["metadata_json"])
    if isinstance(raw_metadata, bytes):
        raw_metadata = raw_metadata.decode("utf-8")
    if not isinstance(raw_metadata, str):
        raise TypeError(f"{episode_path}: metadata_json is not a string")
    return json.loads(raw_metadata)


def infer_dataset_fps(args: argparse.Namespace, metadata: dict[str, Any]) -> int:
    if args.fps is not None:
        if args.fps <= 0:
            raise ValueError("--fps must be > 0")
        return args.fps

    capture_every_steps = int(metadata.get("capture_every_steps", 1))
    if capture_every_steps <= 0:
        capture_every_steps = 1

    inferred_fps = max(1, int(round(args.base_sim_fps / capture_every_steps)))
    return inferred_fps


def build_features(first_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    front = np.asarray(first_payload["observation.images.front"])
    wrist = np.asarray(first_payload["observation.images.wrist"])
    state = np.asarray(first_payload["observation.state"])
    action = np.asarray(first_payload["action"])
    state_names = np.asarray(first_payload["state_names"]).tolist()
    action_names = np.asarray(first_payload["action_names"]).tolist()

    if front.ndim != 4 or front.shape[-1] != 3:
        raise ValueError(f"Unexpected front image shape: {front.shape}")
    if wrist.ndim != 4 or wrist.shape[-1] != 3:
        raise ValueError(f"Unexpected wrist image shape: {wrist.shape}")
    if state.ndim != 2:
        raise ValueError(f"Unexpected observation.state shape: {state.shape}")
    if action.ndim != 2:
        raise ValueError(f"Unexpected action shape: {action.shape}")

    return {
        "observation.images.front": {
            "dtype": "image",
            "shape": tuple(front.shape[1:]),
            "names": ["height", "width", "channel"],
        },
        "observation.images.wrist": {
            "dtype": "image",
            "shape": tuple(wrist.shape[1:]),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (int(state.shape[1]),),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (int(action.shape[1]),),
            "names": action_names,
        },
        "next.reward": {
            "dtype": "float32",
            "shape": (1,),
        },
        "next.done": {
            "dtype": "bool",
            "shape": (1,),
        },
    }


def validate_episode_shapes(payload: dict[str, Any], episode_path: Path) -> int:
    frame_count = int(np.asarray(payload["action"]).shape[0])
    expected_lengths = {
        "observation.images.front": int(np.asarray(payload["observation.images.front"]).shape[0]),
        "observation.images.wrist": int(np.asarray(payload["observation.images.wrist"]).shape[0]),
        "observation.state": int(np.asarray(payload["observation.state"]).shape[0]),
        "action": int(np.asarray(payload["action"]).shape[0]),
        "next.reward": int(np.asarray(payload["next.reward"]).shape[0]),
        "next.done": int(np.asarray(payload["next.done"]).shape[0]),
    }
    for key, length in expected_lengths.items():
        if length != frame_count:
            raise ValueError(
                f"{episode_path}: mismatched frame count for {key}, expected {frame_count}, got {length}"
            )
    return frame_count


def ensure_output_dir(dataset_root: Path, overwrite: bool) -> None:
    if dataset_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset root already exists: {dataset_root}\n"
                "Use --overwrite to rebuild it."
            )
        shutil.rmtree(dataset_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)


def iter_selected_episodes(raw_dir: Path, include_failures: bool) -> list[Path]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw dataset directory not found: {raw_dir}")

    selected: list[Path] = []
    for episode_path in sorted(raw_dir.glob("episode_*.npz")):
        payload = load_npz_payload(episode_path)
        success = bool(scalar_to_python(payload["success"]))
        if not include_failures and not success:
            continue
        selected.append(episode_path)
    return selected


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    dataset_root = args.dataset_root.resolve()

    selected_episodes = iter_selected_episodes(raw_dir, include_failures=args.include_failures)
    if args.max_episodes is not None:
        selected_episodes = selected_episodes[: args.max_episodes]

    if not selected_episodes:
        raise RuntimeError("No episodes selected for conversion. Check raw-dir or filtering options.")

    first_payload = load_npz_payload(selected_episodes[0])
    first_metadata = load_metadata(first_payload, selected_episodes[0])
    dataset_fps = infer_dataset_fps(args, first_metadata)
    features = build_features(first_payload)

    print("[convert] conversion plan", flush=True)
    print(f"  raw_dir={raw_dir}", flush=True)
    print(f"  dataset_root={dataset_root}", flush=True)
    print(f"  repo_id={args.repo_id}", flush=True)
    print(f"  robot_type={args.robot_type}", flush=True)
    print(f"  episodes_selected={len(selected_episodes)}", flush=True)
    print(f"  inferred_fps={dataset_fps}", flush=True)
    print(f"  use_videos={not args.no_videos}", flush=True)
    print("", flush=True)

    ensure_output_dir(dataset_root, overwrite=args.overwrite)

    LeRobotDataset = import_lerobot_dataset()
    print("[convert] creating LeRobotDataset ...", flush=True)
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=dataset_fps,
        root=dataset_root,
        robot_type=args.robot_type,
        features=features,
        use_videos=not args.no_videos,
    )
    print("[convert] dataset created", flush=True)

    total_frames = 0
    converted_episodes = 0
    skipped_failures = 0

    for episode_path in sorted(raw_dir.glob("episode_*.npz")):
        payload = load_npz_payload(episode_path)
        success = bool(scalar_to_python(payload["success"]))
        if not args.include_failures and not success:
            skipped_failures += 1
            continue
        if args.max_episodes is not None and converted_episodes >= args.max_episodes:
            break

        frame_count = validate_episode_shapes(payload, episode_path)
        task = str(scalar_to_python(payload["task"]))
        front_images = np.asarray(payload["observation.images.front"], dtype=np.uint8)
        wrist_images = np.asarray(payload["observation.images.wrist"], dtype=np.uint8)
        states = np.asarray(payload["observation.state"], dtype=np.float32)
        actions = np.asarray(payload["action"], dtype=np.float32)
        rewards = np.asarray(payload["next.reward"], dtype=np.float32)
        dones = np.asarray(payload["next.done"], dtype=np.bool_)

        for frame_index in range(frame_count):
            dataset.add_frame(
                {
                    "observation.images.front": front_images[frame_index],
                    "observation.images.wrist": wrist_images[frame_index],
                    "observation.state": states[frame_index],
                    "action": actions[frame_index],
                    "next.reward": np.array([rewards[frame_index]], dtype=np.float32),
                    "next.done": np.array([dones[frame_index]], dtype=np.bool_),
                    "task": task,
                }
            )

        dataset.save_episode()
        converted_episodes += 1
        total_frames += frame_count

        print(
            f"[convert] {episode_path.name}: success={success} frames={frame_count} "
            f"converted_episode_index={converted_episodes - 1}",
            flush=True,
        )

    dataset.finalize()

    print("\nConversion finished.", flush=True)
    print(f"  raw_dir: {raw_dir}", flush=True)
    print(f"  dataset_root: {dataset_root}", flush=True)
    print(f"  repo_id: {args.repo_id}", flush=True)
    print(f"  inferred_fps: {dataset_fps}", flush=True)
    print(f"  converted_episodes: {converted_episodes}", flush=True)
    print(f"  total_frames: {total_frames}", flush=True)
    print(f"  skipped_failures: {skipped_failures}", flush=True)
    print(f"  metadata_file: {dataset_root / 'meta' / 'info.json'}", flush=True)


if __name__ == "__main__":
    main()

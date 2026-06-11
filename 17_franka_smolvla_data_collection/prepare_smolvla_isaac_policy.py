#!/usr/bin/env python3
"""Prepare a local SmolVLA policy config for Isaac Franka LeRobot datasets.

This script reads `meta/info.json` from a LeRobot dataset directory and patches
`lerobot/smolvla_base` so the policy input/output feature shapes match the
dataset exactly.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True, help="LeRobotDataset 根目录。")
    parser.add_argument("--source", default="lerobot/smolvla_base", help="SmolVLA 预训练模型 repo_id。")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pretrained/smolvla_isaac_franka_base"),
        help="输出本地 policy 配置目录。",
    )
    return parser.parse_args()


def load_dataset_info(dataset_root: Path) -> dict:
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    return json.loads(info_path.read_text(encoding="utf-8"))


def resolve_source_snapshot(source: str, cache_dir: str | None, local_files_only: bool) -> Path:
    source_path = Path(source).expanduser()
    if source_path.exists():
        if not source_path.is_dir():
            raise NotADirectoryError(f"Local source exists but is not a directory: {source_path}")
        required = [
            source_path / "config.json",
            source_path / "model.safetensors",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Local SmolVLA source directory is missing required files:\n"
                + "\n".join(f"  {path}" for path in missing)
            )
        return source_path.resolve()

    try:
        return Path(
            snapshot_download(
                repo_id=source,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "policy_*.json",
                    "policy_*.safetensors",
                ],
            )
        )
    except LocalEntryNotFoundError as exc:
        if local_files_only:
            raise RuntimeError(
                "SmolVLA base model is not cached locally, but the script is running in offline mode.\n"
                f"repo_id: {source}\n"
                f"cache_dir: {cache_dir}\n\n"
                "Fix:\n"
                "1. First-time download: run with online mode enabled\n"
                "   HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 bash train_smolvla_isaac.sh\n"
                "2. Or point --source to an existing local SmolVLA directory that already contains\n"
                "   config.json / model.safetensors / policy_*.json / policy_*.safetensors\n\n"
                "After the cache exists locally, you can switch back to offline mode."
            ) from exc
        raise


def convert_image_feature(feature: dict) -> dict:
    shape = feature["shape"]
    if len(shape) != 3:
        raise ValueError(f"Expected image feature shape [H, W, C], got: {shape}")
    height, width, channel = shape
    return {
        "type": "VISUAL",
        "shape": [channel, height, width],
    }


def convert_state_or_action_feature(feature: dict, feature_type: str) -> dict:
    shape = feature["shape"]
    if len(shape) != 1:
        raise ValueError(f"Expected 1D feature shape, got: {shape}")
    return {
        "type": feature_type,
        "shape": shape,
    }


def build_feature_config(dataset_info: dict) -> tuple[dict, dict]:
    features = dataset_info["features"]

    input_features: dict[str, dict] = {}
    for name, feature in features.items():
        if not name.startswith("observation.images."):
            continue
        if feature.get("dtype") != "image":
            continue
        input_features[name] = convert_image_feature(feature)

    if "observation.state" not in features:
        raise KeyError("Dataset is missing observation.state in meta/info.json")
    input_features["observation.state"] = convert_state_or_action_feature(
        features["observation.state"], "STATE"
    )

    if "action" not in features:
        raise KeyError("Dataset is missing action in meta/info.json")
    output_features = {
        "action": convert_state_or_action_feature(features["action"], "ACTION"),
    }
    return input_features, output_features


def main() -> None:
    args = parse_args()
    dataset_info = load_dataset_info(args.dataset_root.resolve())
    input_features, output_features = build_feature_config(dataset_info)

    cache_dir = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME")
    local_files_only = os.environ.get("HF_HUB_OFFLINE", "1").lower() not in {"0", "false", "no"}
    snapshot = resolve_source_snapshot(args.source, cache_dir=cache_dir, local_files_only=local_files_only)

    args.output.mkdir(parents=True, exist_ok=True)

    for src in snapshot.iterdir():
        if src.name == "config.json" or not src.is_file():
            continue
        dst = args.output / src.name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and dst.resolve() == src.resolve():
                continue
            raise FileExistsError(f"Refuse to overwrite existing file: {dst}")
        dst.symlink_to(src.resolve())

    config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
    config["input_features"] = input_features
    config["output_features"] = output_features
    config["push_to_hub"] = False
    config["repo_id"] = None

    (args.output / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()

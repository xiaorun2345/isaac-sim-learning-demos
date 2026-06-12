"""在 Isaac Sim 中加载训练后的 SmolVLA checkpoint 做闭环推理。

这个脚本和 `replay_episode.py` 不同：
1. `replay_episode.py` 回放的是已经保存好的专家动作序列
2. 本脚本每一步都会重新采集当前观测
3. 然后调用训练好的 SmolVLA policy 预测动作
4. 再把预测动作实时下发给 Isaac 中的 Franka

也就是说，它跑出来的是“策略推理动画”，而不是“专家轨迹重放动画”。

典型运行方式：

    python isaac-sim-learning-demos/17_franka_smolvla_data_collection/infer_policy_in_isaac.py

指定模型：

    python isaac-sim-learning-demos/17_franka_smolvla_data_collection/infer_policy_in_isaac.py \
        --policy-dir isaac-sim-learning-demos/17_franka_smolvla_data_collection/outputs/\
smolvla_isaac_franka_front_top_state18_action4/checkpoints/140000/pretrained_model
"""

from __future__ import annotations

import argparse
import atexit
import os
import pickle
import shutil
import socket
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "outputs"
            / "smolvla_isaac_franka_front_top_state18_action4"
            / "checkpoints"
            / "last"
            / "pretrained_model"
        ),
        help="训练产出的 policy 目录，通常指向 checkpoints/<step>/pretrained_model。",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无界面运行。",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="连续评估多少个 episode。",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=520,
        help="每个 episode 最多运行多少个仿真 step。",
    )
    parser.add_argument(
        "--action-decimation",
        type=int,
        default=2,
        help="每隔多少个物理 step 重新做一次 policy 推理。默认 2，对齐 17 的采样节奏。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子基数。默认留空，此时每次启动都会自动生成新的随机种子。",
    )
    parser.add_argument(
        "--task",
        default="Pick up the red cube with Franka and place it into the wooden tray.",
        help="传给 SmolVLA 的任务文本。",
    )
    parser.add_argument(
        "--spawn-profile",
        choices=("center", "easy", "train"),
        default="train",
        help="评测时红色方块的摆放分布。train 与训练采集分布一致且默认随机，easy 次之，center 最容易。",
    )
    parser.add_argument(
        "--gripper-close-threshold",
        type=float,
        default=0.5,
        help="policy 第 4 维动作大于该阈值时认为要闭爪。",
    )
    parser.add_argument(
        "--lerobot-src",
        type=Path,
        default=Path("/home/mkls/xiao_run/lerobot_smolvla_mujoco_demo/third_party/lerobot/src"),
        help="本地 LeRobot 源码目录，会自动加入 PYTHONPATH。",
    )
    parser.add_argument(
        "--fallback-hf-cache",
        type=Path,
        default=Path("/home/mkls/xiao_run/lerobot_smolvla_mujoco_demo/.cache/huggingface"),
        help="已有 SmolVLM 缓存目录。离线推理时会自动复用它。",
    )
    parser.add_argument(
        "--policy-host",
        default="127.0.0.1",
        help="SmolVLA 推理服务监听地址。",
    )
    parser.add_argument(
        "--policy-port",
        type=int,
        default=5567,
        help="SmolVLA 推理服务监听端口。",
    )
    parser.add_argument(
        "--policy-server-conda-env",
        type=Path,
        default=Path("/home/mkls/xiao_run/.conda-lerobot-smolvla"),
        help="运行 SmolVLA 推理服务的 Conda 环境目录，建议使用训练时的 Python 3.12 环境。",
    )
    parser.add_argument(
        "--conda-exe",
        type=Path,
        default=Path("/home/mkls/anaconda3/bin/conda"),
        help="conda 可执行文件路径，用于自动拉起独立推理服务。",
    )
    parser.add_argument(
        "--no-auto-policy-server",
        action="store_true",
        help="不自动启动策略服务，改为连接已经手动启动好的服务。",
    )
    return parser.parse_args()


ARGS = parse_args()
POLICY_SERVER_PROTOCOL_VERSION = "smolvla_socket_v2"


def build_py311_compatible_lerobot_src(src_root: Path, cache_root: Path) -> Path:
    """为 Python 3.11 生成一份可导入的 lerobot 源码副本。

    你当前 Isaac 环境是 Python 3.11，但本地 lerobot 源码里混入了少量
    Python 3.12 才支持的泛型语法，例如：

    - `def func[T](...)`
    - `class Foo[T]`

    这会让 Python 3.11 在 import 阶段直接 SyntaxError。

    这里不改外部仓库，而是在当前 demo 的缓存目录下复制一份源码，并把少量
    3.12 语法降级成 3.11 兼容写法。
    """

    if sys.version_info >= (3, 12):
        return src_root

    compat_version = "v5"
    dst_root = cache_root / "lerobot_py311_src"
    marker_path = dst_root / ".py311_compat_ready"

    if marker_path.is_file():
        marker_text = marker_path.read_text(encoding="utf-8", errors="ignore")
        if f"compat_version={compat_version}" in marker_text:
            return dst_root

    if dst_root.exists():
        shutil.rmtree(dst_root)
    shutil.copytree(src_root, dst_root)

    def patch_file(relative_path: str, transformer) -> None:
        file_path = dst_root / relative_path
        text = file_path.read_text(encoding="utf-8")
        text = transformer(text)
        file_path.write_text(text, encoding="utf-8")

    def patch_io_utils(text: str) -> str:
        text = text.replace("from typing import Any", "from typing import Any, TypeVar")
        text = text.replace(
            'JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]\n',
            'JsonLike = str | int | float | bool | None | list["JsonLike"] | dict[str, "JsonLike"] | tuple["JsonLike", ...]\n'
            'T = TypeVar("T", bound=JsonLike)\n',
        )
        text = text.replace(
            "def deserialize_json_into_object[T: JsonLike](fpath: Path, obj: T) -> T:",
            "def deserialize_json_into_object(fpath: Path, obj: T) -> T:",
        )
        return text

    def patch_streaming_dataset(text: str) -> str:
        text = text.replace("from pathlib import Path\n", "from pathlib import Path\nfrom typing import Generic, TypeVar\n")
        text = text.replace(
            "class Backtrackable[T]:",
            'T = TypeVar("T")\n\n\nclass Backtrackable(Generic[T]):',
        )
        return text

    def patch_pipeline(text: str) -> str:
        text = text.replace(
            "from typing import Any, TypedDict, TypeVar, cast",
            "from typing import Any, Generic, TypedDict, TypeVar, cast",
        )
        text = text.replace(
            "class DataProcessorPipeline[TInput, TOutput](HubMixin):",
            "class DataProcessorPipeline(HubMixin, Generic[TInput, TOutput]):",
        )
        return text

    def patch_motors_bus(text: str) -> str:
        # Python 3.12:
        #   type NameOrID = str | int
        # Python 3.11:
        #   NameOrID = str | int
        text = text.replace("type NameOrID = str | int", "NameOrID = str | int")
        text = text.replace("type Value = int | float", "Value = int | float")
        return text

    def patch_policies_init(_: str) -> str:
        return '''"""Minimal policies package shim for Python 3.11 Isaac inference.

This shim intentionally avoids importing every optional policy/config at package
import time. The inference script imports concrete submodules directly.
"""

__all__ = []
'''

    def patch_smolvla_init(_: str) -> str:
        return '''"""Minimal SmolVLA package shim for Python 3.11 Isaac inference.

The inference script imports concrete SmolVLA submodules directly to avoid
pulling optional training-only dependencies during package import.
"""

__all__ = []
'''

    def patch_processor_init(_: str) -> str:
        return '''from lerobot.types import EnvAction, EnvTransition, PolicyAction, RobotAction, RobotObservation, TransitionKey

from .batch_processor import AddBatchDimensionProcessorStep
from .converters import (
    batch_to_transition,
    create_transition,
    identity_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from .device_processor import DeviceProcessorStep
from .newline_task_processor import NewLineTaskProcessorStep
from .normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from .pipeline import (
    ActionProcessorStep,
    DataProcessorPipeline,
    ObservationProcessorStep,
    PolicyActionProcessorStep,
    PolicyProcessorPipeline,
    ProcessorKwargs,
    ProcessorStep,
    ProcessorStepRegistry,
    RewardProcessorStep,
)
from .relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
    to_absolute_actions,
    to_relative_actions,
)
from .rename_processor import RenameObservationsProcessorStep
from .tokenizer_processor import TokenizerProcessorStep

__all__ = [
    "ActionProcessorStep",
    "AbsoluteActionsProcessorStep",
    "AddBatchDimensionProcessorStep",
    "DataProcessorPipeline",
    "DeviceProcessorStep",
    "EnvAction",
    "EnvTransition",
    "NewLineTaskProcessorStep",
    "NormalizerProcessorStep",
    "ObservationProcessorStep",
    "PolicyAction",
    "PolicyActionProcessorStep",
    "PolicyProcessorPipeline",
    "ProcessorKwargs",
    "ProcessorStep",
    "ProcessorStepRegistry",
    "RelativeActionsProcessorStep",
    "RenameObservationsProcessorStep",
    "RewardProcessorStep",
    "RobotAction",
    "RobotObservation",
    "TokenizerProcessorStep",
    "TransitionKey",
    "UnnormalizerProcessorStep",
    "batch_to_transition",
    "create_transition",
    "identity_transition",
    "policy_action_to_transition",
    "to_absolute_actions",
    "to_relative_actions",
    "transition_to_batch",
    "transition_to_policy_action",
]
'''

    def patch_pretrained(text: str) -> str:
        text = text.replace(
            "from typing import TypedDict, TypeVar, Unpack",
            "from typing import TYPE_CHECKING, TypedDict, TypeVar, Unpack",
        )
        text = text.replace(
            "from lerobot.configs.train import TrainPipelineConfig\n",
            'if TYPE_CHECKING:\n    from lerobot.configs.train import TrainPipelineConfig\n',
        )
        text = text.replace(
            "        cfg: TrainPipelineConfig,\n",
            '        cfg: "TrainPipelineConfig",\n',
        )
        return text

    patch_file("lerobot/utils/io_utils.py", patch_io_utils)
    patch_file("lerobot/datasets/streaming_dataset.py", patch_streaming_dataset)
    patch_file("lerobot/processor/pipeline.py", patch_pipeline)
    patch_file("lerobot/motors/motors_bus.py", patch_motors_bus)
    patch_file("lerobot/policies/__init__.py", patch_policies_init)
    patch_file("lerobot/policies/smolvla/__init__.py", patch_smolvla_init)
    patch_file("lerobot/policies/pretrained.py", patch_pretrained)
    patch_file("lerobot/processor/__init__.py", patch_processor_init)

    marker_path.write_text(
        f"patched_from={src_root}\npython={sys.version}\ncompat_version={compat_version}\n",
        encoding="utf-8",
    )
    return dst_root


def prepare_offline_hf_cache(script_dir: Path) -> None:
    """复用已有 Hugging Face 缓存，避免推理时再联网。"""

    local_cache_dir = script_dir / ".cache" / "huggingface"
    local_cache_dir.mkdir(parents=True, exist_ok=True)

    os.environ.setdefault("HF_HOME", str(local_cache_dir))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(local_cache_dir))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(local_cache_dir))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    local_vlm_cache = local_cache_dir / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    fallback_vlm_cache = ARGS.fallback_hf_cache / "models--HuggingFaceTB--SmolVLM2-500M-Video-Instruct"
    if not local_vlm_cache.exists() and fallback_vlm_cache.is_dir():
        local_vlm_cache.symlink_to(fallback_vlm_cache)


SCRIPT_DIR = Path(__file__).resolve().parent
prepare_offline_hf_cache(SCRIPT_DIR)

if ARGS.lerobot_src.is_dir():
    compat_lerobot_src = build_py311_compatible_lerobot_src(
        src_root=ARGS.lerobot_src,
        cache_root=SCRIPT_DIR / ".cache",
    )
    sys.path.insert(0, str(compat_lerobot_src))

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "hide_ui": ARGS.headless,
        "renderer": "RaytracedLighting",
        "width": 1280,
        "height": 720,
    }
)

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.manipulators import SingleManipulator
from isaacsim.robot.manipulators.examples.franka.controllers.rmpflow_controller import RMPFlowController
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.sensors.camera import Camera
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, UsdGeom, UsdLux


TABLE_H = 0.40
TABLE_CENTER = np.array([0.45, 0.0, TABLE_H / 2.0], dtype=np.float32)
TABLE_SIZE = np.array([1.0, 0.8, TABLE_H], dtype=np.float32)
TABLE_SURFACE_Z = TABLE_H

FRANKA_PRIM_PATH = "/World/Franka"
FRONT_CAMERA_PATH = "/World/front_camera"
TOP_CAMERA_PATH = "/World/top_camera"

# 推理阶段必须和采集阶段使用同一前视角，否则图像分布会再次漂移。
FRONT_CAMERA_EYE = np.array([0.92, -0.62, 0.86], dtype=np.float32)
FRONT_CAMERA_TARGET = np.array([0.50, 0.02, 0.43], dtype=np.float32)
FRONT_CAMERA_FOCAL_LENGTH = 14.0
TOP_CAMERA_EYE = np.array([0.50, -0.08, 1.34], dtype=np.float32)
TOP_CAMERA_TARGET = np.array([0.50, 0.00, 0.40], dtype=np.float32)
TOP_CAMERA_FOCAL_LENGTH = 15.0
FRONT_CAMERA_RESOLUTION = (640, 480)
TOP_CAMERA_RESOLUTION = (640, 480)
CAMERA_FREQUENCY = 20
CAMERA_WARMUP_STEPS = 20
CAMERA_CAPTURE_RETRIES = 6

CUBE_SIZE = np.array([0.055, 0.055, 0.055], dtype=np.float32)
CUBE_HALF_Z = float(CUBE_SIZE[2] / 2.0)
TARGET_CUBE_NAME = "cube_red"
TARGET_CUBE_COLOR = np.array([0.88, 0.15, 0.15], dtype=np.float32)

TARGET_CUBE_SPAWN_REGIONS = [
    ("left_front", (0.28, 0.38), (0.12, 0.26)),
    ("left_mid", (0.28, 0.40), (-0.08, 0.10)),
    ("left_back", (0.28, 0.38), (-0.26, -0.12)),
    ("center_front", (0.42, 0.54), (0.02, 0.16)),
    ("center_back", (0.42, 0.58), (-0.24, -0.08)),
    ("right_back", (0.56, 0.66), (-0.26, -0.10)),
]
EASY_TARGET_CUBE_SPAWN_REGIONS = [
    ("easy_center_left", (0.34, 0.44), (-0.08, 0.08)),
    ("easy_center_mid", (0.40, 0.50), (-0.12, 0.12)),
    ("easy_center_right", (0.46, 0.56), (-0.08, 0.08)),
]
CENTER_TARGET_CUBE_POSITION = np.array([0.45, 0.0, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32)
TARGET_CUBE_BOX_EXCLUSION_MARGIN = 0.055
TARGET_CUBE_SAMPLE_MAX_TRIES = 80

DISTRACTOR_CUBE_SPECS = [
    ("cube_green", np.array([0.18, 0.66, 0.24], dtype=np.float32)),
    ("cube_blue", np.array([0.12, 0.36, 0.86], dtype=np.float32)),
]
DISTRACTOR_CUBE_LAYOUT = {
    "cube_green": np.array([0.34, 0.18, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32),
    "cube_blue": np.array([0.34, -0.18, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32),
}
TARGET_CUBE_DISTRACTOR_CLEARANCE_XY = 0.095

PLACE_BOX_CENTER = np.array([0.64, 0.18], dtype=np.float32)
PLACE_BOX_OUTER_X = 0.18
PLACE_BOX_OUTER_Y = 0.18
PLACE_BOX_BOTTOM_H = 0.024
PLACE_BOX_WALL_T = 0.016
PLACE_BOX_WALL_H = 0.10
PLACE_GOAL_POSITION = np.array(
    [PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1], TABLE_SURFACE_Z + CUBE_HALF_Z],
    dtype=np.float32,
)

HOME_JOINT_POSITIONS = np.array(
    [0.0, -0.82, 0.0, -2.10, 0.0, 1.82, 0.78, 0.05, 0.05],
    dtype=np.float32,
)
EE_TARGET_ORIENTATION = euler_angles_to_quat(np.array([0.0, np.pi, 0.0], dtype=np.float32))
EE_FEEDBACK_Z_BIAS = 0.0985

POLICY_X_RANGE = (0.24, 0.74)
POLICY_Y_RANGE = (-0.30, 0.30)
POLICY_Z_RANGE = (TABLE_SURFACE_Z + 0.015, TABLE_SURFACE_Z + 0.35)
SUCCESS_SETTLE_STEPS = 30


def create_camera_prim(
    path: str,
    position: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float],
    focal_length: float,
) -> None:
    stage = get_current_stage()
    camera = UsdGeom.Camera.Define(stage, path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    xform = UsdGeom.XformCommonAPI(camera.GetPrim())
    xform.SetTranslate(Gf.Vec3d(*position))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def create_lights() -> None:
    stage = get_current_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
    dome.CreateIntensityAttr(1200.0)

    key = UsdLux.RectLight.Define(stage, "/World/Lights/Key")
    key.CreateIntensityAttr(4500.0)
    key.CreateWidthAttr(1.6)
    key.CreateHeightAttr(1.2)
    xform = UsdGeom.XformCommonAPI(key.GetPrim())
    xform.SetTranslate(Gf.Vec3d(0.65, -0.20, 1.80))
    xform.SetRotate(Gf.Vec3f(-65.0, 0.0, 70.0), UsdGeom.XformCommonAPI.RotationOrderXYZ)


def add_room(world: World) -> None:
    world.scene.add(
        FixedCuboid(
            name="room_floor",
            prim_path="/World/Room/Floor",
            position=np.array([0.55, 0.0, -0.025], dtype=np.float32),
            scale=np.array([3.4, 3.0, 0.05], dtype=np.float32),
            size=1.0,
            color=np.array([0.34, 0.35, 0.36], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_back_wall",
            prim_path="/World/Room/BackWall",
            position=np.array([0.55, 1.50, 1.20], dtype=np.float32),
            scale=np.array([3.4, 0.04, 2.4], dtype=np.float32),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_left_wall",
            prim_path="/World/Room/LeftWall",
            position=np.array([-1.15, 0.0, 1.20], dtype=np.float32),
            scale=np.array([0.04, 3.0, 2.4], dtype=np.float32),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48], dtype=np.float32),
        )
    )
    world.scene.add(
        FixedCuboid(
            name="room_right_wall",
            prim_path="/World/Room/RightWall",
            position=np.array([2.25, 0.0, 1.20], dtype=np.float32),
            scale=np.array([0.04, 3.0, 2.4], dtype=np.float32),
            size=1.0,
            color=np.array([0.46, 0.47, 0.48], dtype=np.float32),
        )
    )


def add_table(world: World) -> None:
    world.scene.add(
        FixedCuboid(
            name="table",
            prim_path="/World/Table",
            position=TABLE_CENTER,
            scale=TABLE_SIZE,
            size=1.0,
            color=np.array([0.55, 0.35, 0.15], dtype=np.float32),
        )
    )


def add_place_box(world: World) -> None:
    bottom_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H / 2.0
    wall_z = TABLE_SURFACE_Z + PLACE_BOX_BOTTOM_H + PLACE_BOX_WALL_H / 2.0
    box_color = np.array([0.54, 0.32, 0.14], dtype=np.float32)

    world.scene.add(
        FixedCuboid(
            name="place_box_bottom",
            prim_path="/World/PlaceBox/Bottom",
            position=np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1], bottom_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_OUTER_Y, PLACE_BOX_BOTTOM_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_left",
            prim_path="/World/PlaceBox/WallLeft",
            position=np.array([PLACE_BOX_CENTER[0] - PLACE_BOX_OUTER_X / 2.0, PLACE_BOX_CENTER[1], wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_right",
            prim_path="/World/PlaceBox/WallRight",
            position=np.array([PLACE_BOX_CENTER[0] + PLACE_BOX_OUTER_X / 2.0, PLACE_BOX_CENTER[1], wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_WALL_T, PLACE_BOX_OUTER_Y, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_front",
            prim_path="/World/PlaceBox/WallFront",
            position=np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1] - PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )
    world.scene.add(
        FixedCuboid(
            name="place_box_wall_back",
            prim_path="/World/PlaceBox/WallBack",
            position=np.array([PLACE_BOX_CENTER[0], PLACE_BOX_CENTER[1] + PLACE_BOX_OUTER_Y / 2.0, wall_z], dtype=np.float32),
            scale=np.array([PLACE_BOX_OUTER_X, PLACE_BOX_WALL_T, PLACE_BOX_WALL_H], dtype=np.float32),
            size=1.0,
            color=box_color,
        )
    )


def add_franka(world: World) -> SingleManipulator:
    assets_root = get_assets_root_path()
    if assets_root is None:
        raise RuntimeError("Isaac Sim assets root is unavailable.")

    franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    add_reference_to_stage(usd_path=franka_usd, prim_path=FRANKA_PRIM_PATH)

    gripper = ParallelGripper(
        end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
        joint_prim_names=["panda_finger_joint1", "panda_finger_joint2"],
        joint_opened_positions=np.array([0.05, 0.05], dtype=np.float32),
        joint_closed_positions=np.array([0.01, 0.01], dtype=np.float32),
        action_deltas=np.array([0.01, 0.01], dtype=np.float32),
    )

    franka = world.scene.add(
        SingleManipulator(
            prim_path=FRANKA_PRIM_PATH,
            name="franka",
            end_effector_prim_path=f"{FRANKA_PRIM_PATH}/panda_hand",
            gripper=gripper,
            position=np.array([0.0, 0.0, TABLE_H], dtype=np.float32),
        )
    )
    franka.gripper.set_default_state(franka.gripper.joint_opened_positions)
    return franka


def add_training_cubes(world: World) -> dict[str, DynamicCuboid]:
    target_cube = world.scene.add(
        DynamicCuboid(
            name=TARGET_CUBE_NAME,
            prim_path=f"/World/{TARGET_CUBE_NAME}",
            position=np.array([0.42, 0.00, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32),
            scale=CUBE_SIZE,
            size=1.0,
            color=TARGET_CUBE_COLOR,
        )
    )
    for cube_name, cube_color in DISTRACTOR_CUBE_SPECS:
        world.scene.add(
            FixedCuboid(
                name=cube_name,
                prim_path=f"/World/{cube_name}",
                position=DISTRACTOR_CUBE_LAYOUT[cube_name],
                scale=CUBE_SIZE,
                size=1.0,
                color=cube_color,
            )
        )
    return {TARGET_CUBE_NAME: target_cube}


def build_scene() -> tuple[World, SingleManipulator, dict[str, DynamicCuboid]]:
    world = World(stage_units_in_meters=1.0)
    create_lights()
    add_room(world)
    world.scene.add_default_ground_plane()
    add_table(world)
    add_place_box(world)

    franka = add_franka(world)
    cubes = add_training_cubes(world)

    create_camera_prim(
        path=FRONT_CAMERA_PATH,
        position=tuple(FRONT_CAMERA_EYE.tolist()),
        rotation_xyz_deg=(-35.0, 0.0, 45.0),
        focal_length=FRONT_CAMERA_FOCAL_LENGTH,
    )
    create_camera_prim(
        path=TOP_CAMERA_PATH,
        position=tuple(TOP_CAMERA_EYE.tolist()),
        rotation_xyz_deg=(0.0, 90.0, 0.0),
        focal_length=TOP_CAMERA_FOCAL_LENGTH,
    )

    world.reset()
    set_camera_view(eye=FRONT_CAMERA_EYE, target=FRONT_CAMERA_TARGET, camera_prim_path=FRONT_CAMERA_PATH)
    set_camera_view(eye=TOP_CAMERA_EYE, target=TOP_CAMERA_TARGET, camera_prim_path=TOP_CAMERA_PATH)
    if not ARGS.headless:
        set_camera_view(eye=FRONT_CAMERA_EYE, target=FRONT_CAMERA_TARGET, camera_prim_path="/OmniverseKit_Persp")
    return world, franka, cubes


def create_cameras() -> tuple[Camera, Camera]:
    front_camera = Camera(
        prim_path=FRONT_CAMERA_PATH,
        name="front_camera",
        frequency=CAMERA_FREQUENCY,
        resolution=FRONT_CAMERA_RESOLUTION,
    )
    top_camera = Camera(
        prim_path=TOP_CAMERA_PATH,
        name="top_camera",
        frequency=CAMERA_FREQUENCY,
        resolution=TOP_CAMERA_RESOLUTION,
    )
    front_camera.initialize()
    top_camera.initialize()
    return front_camera, top_camera


def planar_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(np.linalg.norm(a[:2] - b[:2]))


def sample_cube_positions(rng: np.random.Generator) -> dict[str, np.ndarray]:
    if ARGS.spawn_profile == "center":
        return {TARGET_CUBE_NAME: CENTER_TARGET_CUBE_POSITION.copy()}

    spawn_regions = TARGET_CUBE_SPAWN_REGIONS
    if ARGS.spawn_profile == "easy":
        spawn_regions = EASY_TARGET_CUBE_SPAWN_REGIONS

    for _ in range(TARGET_CUBE_SAMPLE_MAX_TRIES):
        _, x_range, y_range = spawn_regions[rng.integers(len(spawn_regions))]
        target_position = np.array(
            [rng.uniform(*x_range), rng.uniform(*y_range), TABLE_SURFACE_Z + CUBE_HALF_Z],
            dtype=np.float32,
        )
        inside_box_x = abs(float(target_position[0] - PLACE_BOX_CENTER[0])) <= (
            PLACE_BOX_OUTER_X / 2.0 + TARGET_CUBE_BOX_EXCLUSION_MARGIN
        )
        inside_box_y = abs(float(target_position[1] - PLACE_BOX_CENTER[1])) <= (
            PLACE_BOX_OUTER_Y / 2.0 + TARGET_CUBE_BOX_EXCLUSION_MARGIN
        )
        if inside_box_x and inside_box_y:
            continue

        if any(planar_distance(target_position, pos) < TARGET_CUBE_DISTRACTOR_CLEARANCE_XY for pos in DISTRACTOR_CUBE_LAYOUT.values()):
            continue

        return {TARGET_CUBE_NAME: target_position}

    fallback_position = CENTER_TARGET_CUBE_POSITION.copy()
    if ARGS.spawn_profile == "train":
        fallback_position = np.array([0.46, -0.18, TABLE_SURFACE_Z + CUBE_HALF_Z], dtype=np.float32)
    return {TARGET_CUBE_NAME: fallback_position}


def reset_robot(franka: SingleManipulator) -> None:
    franka.set_joint_positions(HOME_JOINT_POSITIONS)
    franka.set_joint_velocities(np.zeros_like(HOME_JOINT_POSITIONS))


def reset_cubes(cubes: dict[str, DynamicCuboid], cube_positions: dict[str, np.ndarray]) -> None:
    for cube_name, cube in cubes.items():
        cube.set_world_pose(
            position=cube_positions[cube_name],
            orientation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        )
        cube.set_linear_velocity(np.zeros(3, dtype=np.float32))
        cube.set_angular_velocity(np.zeros(3, dtype=np.float32))


def settle_scene(world: World, steps: int) -> None:
    for _ in range(steps):
        world.step(render=True)


def merge_joint_actions(num_dof: int, *actions: ArticulationAction) -> ArticulationAction:
    merged_positions: list[float | None] = [None] * num_dof
    merged_velocities: list[float | None] = [None] * num_dof
    merged_efforts: list[float | None] = [None] * num_dof
    for action in actions:
        if action is None:
            continue
        merge_single_field(merged_positions, action.joint_positions, action.joint_indices)
        merge_single_field(merged_velocities, action.joint_velocities, action.joint_indices)
        merge_single_field(merged_efforts, action.joint_efforts, action.joint_indices)
    return ArticulationAction(
        joint_positions=merged_positions,
        joint_velocities=merged_velocities,
        joint_efforts=merged_efforts,
    )


def merge_single_field(
    target: list[float | None],
    values: list[float] | np.ndarray | None,
    indices: list[int] | np.ndarray | None,
) -> None:
    if values is None:
        return
    if indices is None:
        for index, value in enumerate(values):
            if value is not None:
                target[index] = float(value)
        return
    for index, value in zip(indices, values):
        if value is not None:
            target[int(index)] = float(value)


def capture_rgb(camera: Camera) -> np.ndarray:
    for _ in range(CAMERA_CAPTURE_RETRIES):
        rgb = camera.get_rgb()
        if rgb is not None:
            return np.asarray(rgb, dtype=np.uint8)
        rgba_getter = getattr(camera, "get_rgba", None)
        if callable(rgba_getter):
            rgba = rgba_getter()
            if rgba is not None:
                return np.asarray(rgba, dtype=np.uint8)[..., :3]
        simulation_app.update()
    raise RuntimeError(f"Camera {camera.prim_path} did not return RGB data.")


def get_task_space_ee_pose(franka: SingleManipulator) -> tuple[np.ndarray, np.ndarray]:
    ee_position, ee_orientation = franka.end_effector.get_world_pose()
    ee_position = np.asarray(ee_position, dtype=np.float32).copy()
    ee_position[2] -= EE_FEEDBACK_Z_BIAS
    ee_orientation = np.asarray(ee_orientation, dtype=np.float32)
    return ee_position, ee_orientation


def get_robot_state(franka: SingleManipulator, target_cube: DynamicCuboid) -> np.ndarray:
    joint_positions = np.asarray(franka.get_joint_positions(), dtype=np.float32)
    ee_position, ee_orientation = get_task_space_ee_pose(franka)
    cube_position, _ = target_cube.get_world_pose()
    cube_position = np.asarray(cube_position, dtype=np.float32)
    gripper_width = float(joint_positions[7] + joint_positions[8])
    return np.concatenate(
        [
            joint_positions[:7],
            ee_position[:3],
            ee_orientation[:4],
            np.array([gripper_width], dtype=np.float32),
            cube_position[:3],
        ]
    ).astype(np.float32)


def is_cube_inside_box(cube: DynamicCuboid) -> bool:
    cube_position, _ = cube.get_world_pose()
    cube_position = np.asarray(cube_position, dtype=np.float32)
    inner_half_x = PLACE_BOX_OUTER_X / 2.0 - PLACE_BOX_WALL_T
    inner_half_y = PLACE_BOX_OUTER_Y / 2.0 - PLACE_BOX_WALL_T
    within_x = abs(float(cube_position[0] - PLACE_BOX_CENTER[0])) < inner_half_x
    within_y = abs(float(cube_position[1] - PLACE_BOX_CENTER[1])) < inner_half_y
    within_z = abs(float(cube_position[2] - PLACE_GOAL_POSITION[2])) < 0.035
    return bool(within_x and within_y and within_z)


def sanitize_policy_action(raw_action: np.ndarray) -> tuple[np.ndarray, bool]:
    raw_action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if raw_action.shape[0] != 4:
        raise ValueError(f"Expected 4D action [x, y, z, gripper], got shape {raw_action.shape}")
    target_position = raw_action[:3].copy()
    target_position[0] = float(np.clip(target_position[0], *POLICY_X_RANGE))
    target_position[1] = float(np.clip(target_position[1], *POLICY_Y_RANGE))
    target_position[2] = float(np.clip(target_position[2], *POLICY_Z_RANGE))
    gripper_closed = bool(float(raw_action[3]) >= ARGS.gripper_close_threshold)
    return target_position.astype(np.float32), gripper_closed


def resolve_policy_dir(policy_dir: Path) -> Path:
    resolved = policy_dir.resolve()
    if (resolved / "config.json").is_file() and (resolved / "model.safetensors").is_file():
        return resolved
    nested = resolved / "pretrained_model"
    if (nested / "config.json").is_file() and (nested / "model.safetensors").is_file():
        return nested
    raise FileNotFoundError(f"Cannot find policy files under: {policy_dir}")


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("Policy server closed the socket unexpectedly.")
        chunks.extend(chunk)
    return bytes(chunks)


def send_pickle_message(sock: socket.socket, payload: dict) -> None:
    body = pickle.dumps(payload, protocol=4)
    sock.sendall(struct.pack("!I", len(body)))
    sock.sendall(body)


def recv_pickle_message(sock: socket.socket) -> dict:
    header = recv_exact(sock, 4)
    body_size = struct.unpack("!I", header)[0]
    body = recv_exact(sock, body_size)
    return pickle.loads(body)


class PolicyServerClient:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def request(self, payload: dict) -> dict:
        with socket.create_connection((self.host, self.port), timeout=30.0) as sock:
            send_pickle_message(sock, payload)
            response = recv_pickle_message(sock)
        if response.get("status") != "ok":
            raise RuntimeError(response.get("error", "Policy server returned an unknown error."))
        return response

    def ping(self) -> dict:
        return self.request({"type": "ping"})

    def reset(self) -> None:
        self.request({"type": "reset"})

    def predict(self, observation: dict[str, np.ndarray], task: str, robot_type: str) -> np.ndarray:
        response = self.request(
            {
                "type": "predict",
                "task": task,
                "robot_type": robot_type,
                "observation": observation,
            }
        )
        return np.asarray(response["action"], dtype=np.float32)


def find_free_tcp_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def launch_policy_server(policy_dir: Path) -> subprocess.Popen | None:
    if ARGS.no_auto_policy_server:
        return None

    # 自动拉起策略服务时，总是选择一个新的空闲端口，避免误连到历史残留的旧服务。
    ARGS.policy_port = find_free_tcp_port(ARGS.policy_host)

    server_script = SCRIPT_DIR / "smolvla_policy_server.py"
    if not server_script.is_file():
        raise FileNotFoundError(f"Cannot find policy server script: {server_script}")
    if not ARGS.conda_exe.is_file():
        raise FileNotFoundError(f"Cannot find conda executable: {ARGS.conda_exe}")
    if not ARGS.policy_server_conda_env.is_dir():
        raise FileNotFoundError(f"Cannot find policy server conda env: {ARGS.policy_server_conda_env}")

    log_dir = SCRIPT_DIR / ".cache"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "smolvla_policy_server.log"
    log_file = log_path.open("a", encoding="utf-8")

    command = [
        str(ARGS.conda_exe),
        "run",
        "--prefix",
        str(ARGS.policy_server_conda_env),
        "python",
        str(server_script),
        "--policy-dir",
        str(policy_dir),
        "--host",
        ARGS.policy_host,
        "--port",
        str(ARGS.policy_port),
        "--lerobot-src",
        str(ARGS.lerobot_src),
        "--fallback-hf-cache",
        str(ARGS.fallback_hf_cache),
    ]
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )
    process._codex_log_file = log_file  # type: ignore[attr-defined]
    return process


def cleanup_policy_server(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
    except Exception:
        process.kill()
    log_file = getattr(process, "_codex_log_file", None)
    if log_file is not None:
        log_file.close()


def wait_for_policy_server(client: PolicyServerClient, timeout_s: float, process: subprocess.Popen | None) -> dict:
    deadline = time.time() + timeout_s
    last_error: Exception | None = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                "SmolVLA policy server exited early. "
                f"See log: {SCRIPT_DIR / '.cache' / 'smolvla_policy_server.log'}"
            )
        try:
            response = client.ping()
            if response.get("protocol_version") != POLICY_SERVER_PROTOCOL_VERSION:
                raise RuntimeError(
                    "Connected to an incompatible policy server instance. "
                    f"expected={POLICY_SERVER_PROTOCOL_VERSION}, got={response.get('protocol_version')}"
                )
            return response
        except Exception as exc:
            last_error = exc
            time.sleep(1.0)
    raise RuntimeError(f"Timed out waiting for policy server: {last_error}")


def collect_policy_observation(
    front_camera: Camera,
    top_camera: Camera,
    franka: SingleManipulator,
    target_cube: DynamicCuboid,
) -> dict[str, np.ndarray]:
    return {
        "observation.images.front": capture_rgb(front_camera),
        "observation.images.top": capture_rgb(top_camera),
        "observation.state": get_robot_state(franka, target_cube),
    }


def run_episode(
    world: World,
    franka: SingleManipulator,
    cubes: dict[str, DynamicCuboid],
    front_camera: Camera,
    top_camera: Camera,
    controller: RMPFlowController,
    policy_client: PolicyServerClient,
    episode_index: int,
    seed: int,
) -> bool:
    rng = np.random.default_rng(seed)
    cube_positions = sample_cube_positions(rng)
    target_cube = cubes[TARGET_CUBE_NAME]

    reset_robot(franka)
    reset_cubes(cubes, cube_positions)
    controller.reset()
    policy_client.reset()
    settle_scene(world, 20)

    success = False
    last_policy_action = np.array([0.46, 0.0, TABLE_SURFACE_Z + 0.20, 0.0], dtype=np.float32)
    target_position, gripper_closed = sanitize_policy_action(last_policy_action)

    print(
        f"[inference episode {episode_index:03d}] target_cube={np.round(cube_positions[TARGET_CUBE_NAME], 4).tolist()}",
        flush=True,
    )

    for step in range(ARGS.max_steps):
        if step % max(1, ARGS.action_decimation) == 0:
            observation = collect_policy_observation(front_camera, top_camera, franka, target_cube)
            last_policy_action = policy_client.predict(
                observation=observation,
                task=ARGS.task,
                robot_type="isaacsim_franka_panda",
            )
            target_position, gripper_closed = sanitize_policy_action(last_policy_action)
            if step % (ARGS.action_decimation * 5) == 0:
                ee_position, _ = get_task_space_ee_pose(franka)
                print(
                    f"[episode {episode_index:03d}] step={step}"
                    f" ee={np.round(ee_position, 4).tolist()}"
                    f" action={np.round(last_policy_action, 4).tolist()}"
                    f" clipped_target={np.round(target_position, 4).tolist()}"
                    f" close={gripper_closed}",
                    flush=True,
                )

        arm_action = controller.forward(
            target_end_effector_position=target_position,
            target_end_effector_orientation=EE_TARGET_ORIENTATION,
        )
        gripper_action = franka.gripper.forward("close" if gripper_closed else "open")
        franka.apply_action(merge_joint_actions(franka.num_dof, arm_action, gripper_action))
        world.step(render=True)

        success = is_cube_inside_box(target_cube)
        if success:
            print(f"[episode {episode_index:03d}] success at step {step}", flush=True)
            settle_scene(world, SUCCESS_SETTLE_STEPS)
            break

        if not simulation_app.is_running():
            break

    cube_position, _ = target_cube.get_world_pose()
    ee_position, _ = get_task_space_ee_pose(franka)
    joint_positions = np.asarray(franka.get_joint_positions(), dtype=np.float32)
    gripper_width = float(joint_positions[7] + joint_positions[8])
    print(
        f"[episode {episode_index:03d}] done success={success}"
        f" final_cube={np.round(np.asarray(cube_position, dtype=np.float32), 4).tolist()}"
        f" final_ee={np.round(ee_position, 4).tolist()}"
        f" gripper_width={round(gripper_width, 4)}",
        flush=True,
    )
    return success


def main() -> None:
    resolved_policy_dir = resolve_policy_dir(ARGS.policy_dir)
    policy_server_process = launch_policy_server(resolved_policy_dir)
    atexit.register(cleanup_policy_server, policy_server_process)
    policy_client = PolicyServerClient(ARGS.policy_host, ARGS.policy_port)
    server_info = wait_for_policy_server(policy_client, timeout_s=120.0, process=policy_server_process)
    print(f"Connected SmolVLA policy server: {server_info}", flush=True)
    base_seed = ARGS.seed if ARGS.seed is not None else int(time.time() * 1000) % (2**31 - 1)
    print(f"Inference spawn_profile={ARGS.spawn_profile} base_seed={base_seed}", flush=True)

    world, franka, cubes = build_scene()
    franka.initialize()
    front_camera, top_camera = create_cameras()
    settle_scene(world, CAMERA_WARMUP_STEPS)

    controller = RMPFlowController(
        name="franka_smolvla_inference_rmpflow",
        robot_articulation=franka,
    )

    success_count = 0
    for episode_index in range(ARGS.episodes):
        if not simulation_app.is_running():
            break
        success = run_episode(
            world=world,
            franka=franka,
            cubes=cubes,
            front_camera=front_camera,
            top_camera=top_camera,
            controller=controller,
            policy_client=policy_client,
            episode_index=episode_index,
            seed=base_seed + episode_index,
        )
        success_count += int(success)

    print(
        f"Inference finished: success {success_count}/{ARGS.episodes}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()

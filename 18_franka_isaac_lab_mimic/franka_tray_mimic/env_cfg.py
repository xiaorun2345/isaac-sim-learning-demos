"""Manager-based Isaac Lab scene and Mimic configuration matching Demo 17."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_ik_rel_visuomotor_env_cfg import (
    FrankaCubeStackVisuomotorEnvCfg,
    ObservationsCfg as StackVisuomotorObservationsCfg,
)
from isaaclab_tasks.manager_based.manipulation.stack.config.franka.stack_joint_pos_env_cfg import (
    EventCfg as StackEventCfg,
)
from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events
from isaaclab_tasks.manager_based.manipulation.stack.stack_env_cfg import (
    TerminationsCfg as StackTerminationsCfg,
)

from . import mdp
from .scene_contract import (
    BLUE_CUBE_POSITION,
    FRONT_CAMERA_POSITION,
    FRONT_CAMERA_RESOLUTION,
    GREEN_CUBE_POSITION,
    MIMIC_SUBTASKS,
    TABLE_CENTER,
    TABLE_SIZE,
    TARGET_CUBE_X_RANGE,
    TARGET_CUBE_Y_RANGE,
    TRAY_BOTTOM_HEIGHT,
    TRAY_CENTER_XY,
    TRAY_OUTER_X,
    TRAY_OUTER_Y,
    TRAY_WALL_HEIGHT,
    TRAY_WALL_THICKNESS,
    WRIST_CAMERA_RESOLUTION,
)


def static_box(path: str, size: tuple[float, float, float], position: tuple[float, float, float], color):
    """Build a static collision box used for the table and tray."""

    return AssetBaseCfg(
        prim_path=path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=sim_utils.CuboidCfg(
            size=size,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


@configclass
class EventCfg(StackEventCfg):
    """Reset the three cubes into the same layout distribution as Demo 17."""

    randomize_cube_positions = None
    reset_target_cube = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": TARGET_CUBE_X_RANGE,
                "y": TARGET_CUBE_Y_RANGE,
                "z": (0.0203, 0.0203),
                "yaw": (-0.15, 0.15),
            },
            "asset_cfgs": [SceneEntityCfg("cube_2")],
        },
    )
    reset_blue_cube = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (BLUE_CUBE_POSITION[0] - 0.015, BLUE_CUBE_POSITION[0] + 0.015),
                "y": (BLUE_CUBE_POSITION[1] - 0.020, BLUE_CUBE_POSITION[1] + 0.020),
                "z": (0.0203, 0.0203),
            },
            "asset_cfgs": [SceneEntityCfg("cube_1")],
        },
    )
    reset_green_cube = EventTerm(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (GREEN_CUBE_POSITION[0] - 0.015, GREEN_CUBE_POSITION[0] + 0.015),
                "y": (GREEN_CUBE_POSITION[1] - 0.020, GREEN_CUBE_POSITION[1] + 0.020),
                "z": (0.0203, 0.0203),
            },
            "asset_cfgs": [SceneEntityCfg("cube_3")],
        },
    )


@configclass
class ObservationsCfg(StackVisuomotorObservationsCfg):
    """Keep state and dual-camera observations, but replace stack subtask terms."""

    @configclass
    class SubtaskCfg(ObsGroup):
        grasp = ObsTerm(
            func=stack_mdp.object_grasped,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "object_cfg": SceneEntityCfg("cube_2"),
            },
        )
        placed_in_tray = ObsTerm(func=mdp.target_cube_in_tray)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class TerminationsCfg(StackTerminationsCfg):
    """End successfully when the red cube enters the tray."""

    success = DoneTerm(func=mdp.target_cube_in_tray)


@configclass
class FrankaTrayIKRelEnvCfg(FrankaCubeStackVisuomotorEnvCfg):
    """Demo 17 scene expressed as a manager-based relative-IK environment."""

    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # The upstream visuomotor config assigns its own event config in
        # __post_init__, so restore this task's Demo 17 reset distribution.
        self.events = EventCfg()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.episode_length_s = 35.0

        # Replace the Seattle table with Demo 17's simple 1.0 x 0.8 x 0.4 table.
        self.scene.table = static_box(
            "{ENV_REGEX_NS}/Table",
            TABLE_SIZE,
            TABLE_CENTER,
            (0.45, 0.28, 0.14),
        )
        self.scene.plane.init_state.pos = (0.0, 0.0, -0.41)

        bottom_z = TRAY_BOTTOM_HEIGHT / 2.0
        wall_z = TRAY_BOTTOM_HEIGHT + TRAY_WALL_HEIGHT / 2.0
        wood = (0.52, 0.30, 0.12)
        self.scene.tray_bottom = static_box(
            "{ENV_REGEX_NS}/Tray/Bottom",
            (TRAY_OUTER_X, TRAY_OUTER_Y, TRAY_BOTTOM_HEIGHT),
            (TRAY_CENTER_XY[0], TRAY_CENTER_XY[1], bottom_z),
            wood,
        )
        self.scene.tray_left = static_box(
            "{ENV_REGEX_NS}/Tray/Left",
            (TRAY_WALL_THICKNESS, TRAY_OUTER_Y, TRAY_WALL_HEIGHT),
            (TRAY_CENTER_XY[0] - TRAY_OUTER_X / 2.0, TRAY_CENTER_XY[1], wall_z),
            wood,
        )
        self.scene.tray_right = static_box(
            "{ENV_REGEX_NS}/Tray/Right",
            (TRAY_WALL_THICKNESS, TRAY_OUTER_Y, TRAY_WALL_HEIGHT),
            (TRAY_CENTER_XY[0] + TRAY_OUTER_X / 2.0, TRAY_CENTER_XY[1], wall_z),
            wood,
        )
        self.scene.tray_front = static_box(
            "{ENV_REGEX_NS}/Tray/Front",
            (TRAY_OUTER_X, TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT),
            (TRAY_CENTER_XY[0], TRAY_CENTER_XY[1] - TRAY_OUTER_Y / 2.0, wall_z),
            wood,
        )
        self.scene.tray_back = static_box(
            "{ENV_REGEX_NS}/Tray/Back",
            (TRAY_OUTER_X, TRAY_WALL_THICKNESS, TRAY_WALL_HEIGHT),
            (TRAY_CENTER_XY[0], TRAY_CENTER_XY[1] + TRAY_OUTER_Y / 2.0, wall_z),
            wood,
        )

        # Isaac Lab camera resolution is height x width.
        self.scene.table_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_camera",
            update_period=0.0,
            height=FRONT_CAMERA_RESOLUTION[1],
            width=FRONT_CAMERA_RESOLUTION[0],
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.1, 5.0)),
            offset=CameraCfg.OffsetCfg(
                pos=FRONT_CAMERA_POSITION,
                rot=(0.376, -0.446, -0.621, 0.523),
                convention="ros",
            ),
        )
        self.scene.wrist_cam.height = WRIST_CAMERA_RESOLUTION[1]
        self.scene.wrist_cam.width = WRIST_CAMERA_RESOLUTION[0]
        self.image_obs_list = ["table_cam", "wrist_cam"]


@configclass
class FrankaTrayIKRelMimicEnvCfg(FrankaTrayIKRelEnvCfg, MimicEnvCfg):
    """Mimic data-generation settings for the Demo 17 pick-to-tray task."""

    def __post_init__(self):
        super().__post_init__()

        self.datagen_config.name = "franka_red_cube_to_tray"
        self.datagen_config.generation_guarantee = True
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.generation_transform_first_robot_pose = False
        self.datagen_config.generation_interpolate_from_last_target_pose = True
        self.datagen_config.max_num_failures = 50
        self.datagen_config.seed = 20260605

        self.subtask_configs["franka"] = [
            SubTaskConfig(
                object_ref=object_ref,
                subtask_term_signal=term_signal,
                subtask_term_offset_range=(5, 12) if term_signal else (0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=8,
                apply_noise_during_interpolation=False,
                description=description,
                next_subtask_description=next_description,
            )
            for (object_ref, term_signal), description, next_description in zip(
                MIMIC_SUBTASKS,
                (
                    "Grasp the red cube",
                    "Move the red cube into the wooden tray",
                    "Release the red cube and retreat",
                ),
                (
                    "Move the red cube into the wooden tray",
                    "Release the red cube and retreat",
                    "",
                ),
            )
        ]

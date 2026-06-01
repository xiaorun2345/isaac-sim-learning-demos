from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from common.groot_utils import build_groot_policy_observation, build_vla_observation


def tensor_to_numpy(value):
    return value.detach().cpu().numpy()


def read_isaac_lab_observation(scene, instruction, env_index=0):
    """Convert Isaac Lab scene buffers into the simple teaching format."""
    rgb = tensor_to_numpy(scene["camera"].data.output["rgb"][env_index])
    joint_positions = tensor_to_numpy(scene["robot"].data.joint_pos[env_index])
    joint_velocities = tensor_to_numpy(scene["robot"].data.joint_vel[env_index])
    return build_vla_observation(rgb, instruction, joint_positions, joint_velocities)


def read_groot_policy_observation(scene, instruction, env_index=0):
    """Convert Isaac Lab scene buffers into the nested GR00T Policy API format."""
    rgb = tensor_to_numpy(scene["camera"].data.output["rgb"][env_index])
    joint_positions = tensor_to_numpy(scene["robot"].data.joint_pos[env_index])
    return build_groot_policy_observation(rgb, instruction, joint_positions)

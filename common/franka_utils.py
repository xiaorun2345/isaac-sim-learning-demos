"""Helpers for loading and commanding Franka.

这个文件提供一组最基础的 Franka 机械臂辅助函数，主要做三件事：

1. 找到并加载 Isaac Sim 自带的 Franka USD 机器人资产
2. 生成机械臂常用的初始关节位姿
3. 生成机械臂与夹爪控制所需的 `ArticulationAction`

这些函数本身并不负责“抓取逻辑”或“状态机”，它们只负责把常见底层操作
封装成简洁可复用的接口，供上层 demo 调用。
"""

import numpy as np

# `SingleArticulation` 是 Isaac Sim 对单个关节机械系统的高层封装。
# 对于 Franka 这种单机械臂系统，我们可以通过它读取关节状态、发送关节动作。
from isaacsim.core.prims import SingleArticulation

# `add_reference_to_stage` 用来把一个 USD 资产“引用”进当前场景，而不是手动复制模型。
from isaacsim.core.utils.stage import add_reference_to_stage

# `ArticulationAction` 是 Isaac Sim 用来表达“给机器人发送什么动作”的标准数据结构。
# 常见内容包括：
# - `joint_positions`：目标关节位置
# - `joint_velocities`：目标关节速度
# - `joint_efforts`：目标关节力/力矩
# - `joint_indices`：这些值要作用到哪些关节
from isaacsim.core.utils.types import ArticulationAction

# Isaac Sim 自带了机器人、传感器、场景等官方资产。这里通过它找到资产根目录。
from isaacsim.storage.native import get_assets_root_path


# Franka 在当前 USD 场景里的默认 prim 路径。
# 大多数 demo 都会把机器人挂在 `/World/Franka` 下。
FRANKA_PRIM_PATH = "/World/Franka"


def get_franka_usd_path():
    """返回 Isaac Sim 自带 Franka 机器人 USD 文件路径。

    这个路径通常位于 Isaac Sim 资产库中，例如：

        <assets_root>/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd

    这里没有直接把完整绝对路径写死，是为了兼容不同机器上的 Isaac Sim 安装位置。
    """

    return get_assets_root_path() + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"


def load_franka(prim_path=FRANKA_PRIM_PATH, name="franka"):
    """把 Franka 机器人加载到当前 stage，并返回可控制对象。

    参数：
    - `prim_path`: 机器人在当前 USD 场景树中的挂载路径
    - `name`: Isaac 中给这个 articulation 起的逻辑名字

    返回：
    - `SingleArticulation` 对象，用于后续执行 `apply_action()` 等操作

    这里的流程分两步：

    1. 使用 `add_reference_to_stage()` 把 Franka USD 资源引用到当前 stage
    2. 用 `SingleArticulation` 把这个机器人包装成 Isaac 可控制的 articulation
    """

    add_reference_to_stage(get_franka_usd_path(), prim_path)
    return SingleArticulation(prim_path=prim_path, name=name)


def home_positions():
    """返回一个常用的 Franka 初始关节位姿。

    返回的是长度为 9 的 numpy 数组：

    - 前 7 个值：Franka 机械臂 7 个关节的位置
    - 后 2 个值：左右夹爪手指关节的位置

    这里的设计很重要：
    Franka 在很多 Isaac 示例里通常被当成“9 维关节系统”来控制，
    也就是 7 个 arm joints + 2 个 gripper finger joints。

    这个 home pose 不是数学意义上的唯一“标准位姿”，而是一个比较适合教学 demo
    的待命姿态：机械臂抬起、夹爪张开。
    """

    return np.array([0.0, -0.8, 0.0, -2.0, 0.0, 2.2, 0.8, 0.04, 0.04], dtype=np.float32)


def home_action():
    """把默认 home 位姿包装成一个完整关节动作。

    这个函数返回的 `ArticulationAction` 会同时作用到全部 9 个关节，
    因为这里没有显式给出 `joint_indices`，所以 Isaac 会把它理解为：

    “按顺序给 articulation 的全部关节设置目标位置”
    """

    return ArticulationAction(joint_positions=home_positions())


def gripper_open_action():
    """生成“张开夹爪”的动作。

    这里有两个关键点：

    1. `joint_positions=[0.04, 0.04]`
       表示左右两个手指目标位置都设置为较大的开口值。

    2. `joint_indices=[7, 8]`
       表示这个动作只作用于第 8、9 个关节（从 0 开始计数），
       也就是 Franka 的两个夹爪手指关节。

    这样做的好处是：
    我们可以只控制夹爪，而不影响前 7 个机械臂关节。
    """

    return ArticulationAction(
        joint_positions=np.array([0.04, 0.04], dtype=np.float32),
        joint_indices=np.array([7, 8], dtype=np.int32),
    )


def gripper_close_action():
    """生成“闭合夹爪”的动作。

    和 `gripper_open_action()` 一样，这里也只控制两个夹爪关节。

    `joint_positions=[0.0, 0.0]` 表示让左右手指尽量闭合。

    需要注意：
    这个动作只是“命令夹爪去闭合”，并不等于“已经抓住物体”。
    真正是否抓成功，还取决于：

    - 抓取时机械臂末端是否对准物体
    - 物体尺寸是否匹配夹爪开口
    - 接触、摩擦、碰撞等物理参数是否合适
    - 夹爪关闭时物体是否已经滑走或被碰歪
    """

    return ArticulationAction(
        joint_positions=np.array([0.0, 0.0], dtype=np.float32),
        joint_indices=np.array([7, 8], dtype=np.int32),
    )


def arm_pose_action(joint_values):
    """根据给定关节目标值生成机械臂动作。

    参数：
    - `joint_values`: 关节目标值序列，通常是长度为 9 的数组

    返回：
    - `ArticulationAction`

    这个函数是最通用的一个 helper：
    只要你已经知道目标 joint positions，就可以直接用它包装成 Isaac 可执行动作。

    常见两种用法：

    1. 传 9 个值
       同时控制 7 个机械臂关节和 2 个夹爪关节

    2. 传和 articulation 关节顺序一致的全量值
       让整个机器人切换到一个预定义姿态

    这里没有做长度检查，所以调用者需要自己确保 `joint_values` 的维度和关节顺序正确。
    否则轻则动作异常，重则机械臂会跑到完全不合理的位置。
    """

    return ArticulationAction(joint_positions=np.array(joint_values, dtype=np.float32))

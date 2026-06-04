"""Register the Demo 18 custom Isaac Lab Mimic task."""

import gymnasium as gym

from .scene_contract import TASK_ID


gym.register(
    id=TASK_ID,
    entry_point=f"{__name__}.mimic_env:FrankaTrayIKRelMimicEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:FrankaTrayIKRelMimicEnvCfg",
    },
    disable_env_checker=True,
)

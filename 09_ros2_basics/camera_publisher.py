from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "renderer": "RayTracedLighting"})

import numpy as np
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd
from isaacsim.core.api import SimulationContext
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils import extensions
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.sensors.camera import Camera

from common.core_utils import add_basic_light


RGB_TOPIC = "/isaac/camera/rgb"
CAMERA_INFO_TOPIC = "/isaac/camera/camera_info"


def set_publish_rate(render_var, render_product_path, frequency_hz):
    step_size = max(1, int(60 / frequency_hz))
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        render_var + "IsaacSimulationGate",
        render_product_path,
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)


def publish_rgb(camera, frequency_hz=20):
    render_product_path = camera._render_product_path
    render_var = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    writer = rep.writers.get(render_var + "ROS2PublishImage")
    writer.initialize(
        frameId="isaac_camera",
        nodeNamespace="",
        queueSize=1,
        topicName=RGB_TOPIC,
    )
    writer.attach([render_product_path])
    set_publish_rate(render_var, render_product_path, frequency_hz)
    return writer


def publish_camera_info(camera, frequency_hz=20):
    from isaacsim.ros2.bridge import read_camera_info

    render_product_path = camera._render_product_path
    camera_info, _ = read_camera_info(render_product_path=render_product_path)
    writer = rep.writers.get("ROS2PublishCameraInfo")
    writer.initialize(
        frameId="isaac_camera",
        nodeNamespace="",
        queueSize=1,
        topicName=CAMERA_INFO_TOPIC,
        width=camera_info.width,
        height=camera_info.height,
        projectionType=camera_info.distortion_model,
        k=camera_info.k.reshape([1, 9]),
        r=camera_info.r.reshape([1, 9]),
        p=camera_info.p.reshape([1, 12]),
        physicalDistortionModel=camera_info.distortion_model,
        physicalDistortionCoefficients=camera_info.d,
    )
    writer.attach([render_product_path])
    set_publish_rate("PostProcessDispatch", render_product_path, frequency_hz)
    return writer


def main():
    extensions.enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    sim = SimulationContext(stage_units_in_meters=1.0)
    add_basic_light(get_current_stage())
    DynamicCuboid(
        prim_path="/World/Cube",
        position=np.array([0.4, 0.0, 0.3]),
        size=0.15,
        color=np.array([0.1, 0.8, 0.2]),
    )
    camera = Camera(
        prim_path="/World/isaac_camera",
        position=np.array([1.5, 1.2, 1.2]),
        frequency=20,
        resolution=(640, 480),
        orientation=np.array([0.339, 0.176, 0.425, 0.820]),
    )

    simulation_app.update()
    camera.initialize()
    publish_rgb(camera)
    publish_camera_info(camera)

    sim.initialize_physics()
    sim.play()
    print(f"Publishing RGB images on {RGB_TOPIC}")
    print(f"Publishing camera info on {CAMERA_INFO_TOPIC}")

    try:
        while simulation_app.is_running():
            sim.step(render=True)
    finally:
        sim.stop()
        simulation_app.close()


if __name__ == "__main__":
    main()

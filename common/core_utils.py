"""Small helpers for standalone Isaac Sim demos."""

from pxr import UsdGeom, UsdLux


def add_basic_light(stage, intensity=500.0, path="/World/DistantLight"):
    light = UsdLux.DistantLight.Define(stage, path)
    light.CreateIntensityAttr(float(intensity))
    return light


def define_camera_prim(stage, path="/World/Camera"):
    return UsdGeom.Camera.Define(stage, path)

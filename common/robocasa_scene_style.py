from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from pxr import Gf, UsdGeom, UsdLux


@dataclass(frozen=True)
class SceneTheme:
    name: str
    floor_base: np.ndarray
    floor_tiles: tuple[np.ndarray, np.ndarray]
    grout: np.ndarray
    wall_main: np.ndarray
    wall_accent: np.ndarray
    trim: np.ndarray
    cabinet: np.ndarray
    countertop: np.ndarray
    metal: np.ndarray
    glass: np.ndarray
    rug: np.ndarray
    decor: np.ndarray
    pendant_tint: np.ndarray


SCENE_THEMES: dict[str, SceneTheme] = {
    "modern_loft": SceneTheme(
        name="modern_loft",
        floor_base=np.array([0.24, 0.24, 0.25], dtype=np.float32),
        floor_tiles=(
            np.array([0.30, 0.30, 0.31], dtype=np.float32),
            np.array([0.36, 0.35, 0.34], dtype=np.float32),
        ),
        grout=np.array([0.16, 0.16, 0.17], dtype=np.float32),
        wall_main=np.array([0.78, 0.76, 0.73], dtype=np.float32),
        wall_accent=np.array([0.20, 0.22, 0.24], dtype=np.float32),
        trim=np.array([0.08, 0.09, 0.10], dtype=np.float32),
        cabinet=np.array([0.18, 0.20, 0.22], dtype=np.float32),
        countertop=np.array([0.58, 0.57, 0.54], dtype=np.float32),
        metal=np.array([0.62, 0.64, 0.66], dtype=np.float32),
        glass=np.array([0.55, 0.69, 0.76], dtype=np.float32),
        rug=np.array([0.52, 0.34, 0.21], dtype=np.float32),
        decor=np.array([0.86, 0.52, 0.28], dtype=np.float32),
        pendant_tint=np.array([1.00, 0.91, 0.78], dtype=np.float32),
    ),
    "warm_walnut": SceneTheme(
        name="warm_walnut",
        floor_base=np.array([0.42, 0.31, 0.21], dtype=np.float32),
        floor_tiles=(
            np.array([0.55, 0.41, 0.28], dtype=np.float32),
            np.array([0.47, 0.34, 0.23], dtype=np.float32),
        ),
        grout=np.array([0.29, 0.20, 0.14], dtype=np.float32),
        wall_main=np.array([0.92, 0.87, 0.80], dtype=np.float32),
        wall_accent=np.array([0.61, 0.45, 0.31], dtype=np.float32),
        trim=np.array([0.35, 0.24, 0.16], dtype=np.float32),
        cabinet=np.array([0.42, 0.27, 0.16], dtype=np.float32),
        countertop=np.array([0.80, 0.74, 0.66], dtype=np.float32),
        metal=np.array([0.72, 0.69, 0.63], dtype=np.float32),
        glass=np.array([0.61, 0.74, 0.81], dtype=np.float32),
        rug=np.array([0.71, 0.57, 0.34], dtype=np.float32),
        decor=np.array([0.84, 0.67, 0.42], dtype=np.float32),
        pendant_tint=np.array([1.00, 0.92, 0.83], dtype=np.float32),
    ),
    "coastal_bright": SceneTheme(
        name="coastal_bright",
        floor_base=np.array([0.80, 0.81, 0.79], dtype=np.float32),
        floor_tiles=(
            np.array([0.86, 0.87, 0.85], dtype=np.float32),
            np.array([0.78, 0.81, 0.80], dtype=np.float32),
        ),
        grout=np.array([0.69, 0.72, 0.70], dtype=np.float32),
        wall_main=np.array([0.94, 0.96, 0.95], dtype=np.float32),
        wall_accent=np.array([0.58, 0.73, 0.78], dtype=np.float32),
        trim=np.array([0.64, 0.71, 0.74], dtype=np.float32),
        cabinet=np.array([0.81, 0.88, 0.89], dtype=np.float32),
        countertop=np.array([0.92, 0.93, 0.91], dtype=np.float32),
        metal=np.array([0.65, 0.71, 0.75], dtype=np.float32),
        glass=np.array([0.67, 0.81, 0.87], dtype=np.float32),
        rug=np.array([0.39, 0.58, 0.66], dtype=np.float32),
        decor=np.array([0.96, 0.77, 0.46], dtype=np.float32),
        pendant_tint=np.array([0.95, 0.99, 1.00], dtype=np.float32),
    ),
    "terracotta_studio": SceneTheme(
        name="terracotta_studio",
        floor_base=np.array([0.45, 0.28, 0.20], dtype=np.float32),
        floor_tiles=(
            np.array([0.63, 0.40, 0.28], dtype=np.float32),
            np.array([0.55, 0.34, 0.24], dtype=np.float32),
        ),
        grout=np.array([0.34, 0.21, 0.16], dtype=np.float32),
        wall_main=np.array([0.95, 0.88, 0.80], dtype=np.float32),
        wall_accent=np.array([0.70, 0.39, 0.29], dtype=np.float32),
        trim=np.array([0.49, 0.30, 0.23], dtype=np.float32),
        cabinet=np.array([0.74, 0.49, 0.34], dtype=np.float32),
        countertop=np.array([0.84, 0.73, 0.62], dtype=np.float32),
        metal=np.array([0.49, 0.41, 0.36], dtype=np.float32),
        glass=np.array([0.64, 0.74, 0.80], dtype=np.float32),
        rug=np.array([0.40, 0.51, 0.34], dtype=np.float32),
        decor=np.array([0.92, 0.65, 0.34], dtype=np.float32),
        pendant_tint=np.array([1.00, 0.87, 0.72], dtype=np.float32),
    ),
}


def theme_names() -> tuple[str, ...]:
    return tuple(SCENE_THEMES.keys())


def get_theme(theme_name: str) -> SceneTheme:
    if theme_name not in SCENE_THEMES:
        raise ValueError(f"Unknown theme: {theme_name}. Choices: {sorted(SCENE_THEMES)}")
    return SCENE_THEMES[theme_name]


def _add_box(
    world: World,
    *,
    name: str,
    prim_path: str,
    position: np.ndarray,
    scale: np.ndarray,
    color: np.ndarray,
) -> None:
    world.scene.add(
        FixedCuboid(
            name=name,
            prim_path=prim_path,
            position=np.asarray(position, dtype=np.float32),
            scale=np.asarray(scale, dtype=np.float32),
            size=1.0,
            color=np.asarray(color, dtype=np.float32),
        )
    )


def _set_xform(prim_path: str, position: np.ndarray, rotation_xyz_deg: tuple[float, float, float]) -> None:
    stage = get_current_stage()
    xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(prim_path))
    xform.SetTranslate(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
    xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)


@lru_cache(maxsize=8)
def _cached_usd_assets(asset_root: str) -> tuple[Path, ...]:
    root = Path(asset_root).expanduser().resolve()
    return tuple(
        path
        for path in root.rglob("*")
        if path.suffix.lower() in {".usd", ".usda", ".usdc"}
    )


def add_styled_room_shell(
    world: World,
    *,
    room_root: str,
    center_xy: np.ndarray,
    size_xy: np.ndarray,
    theme: SceneTheme,
    wall_height: float = 2.6,
    floor_thickness: float = 0.06,
) -> None:
    center_xy = np.asarray(center_xy, dtype=np.float32)
    size_xy = np.asarray(size_xy, dtype=np.float32)
    center_x = float(center_xy[0])
    center_y = float(center_xy[1])
    size_x = float(size_xy[0])
    size_y = float(size_xy[1])
    floor_top_z = 0.0
    wall_base_z = -floor_thickness / 2.0

    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_floor_base",
        prim_path=f"{room_root}/FloorBase",
        position=np.array([center_x, center_y, wall_base_z], dtype=np.float32),
        scale=np.array([size_x, size_y, floor_thickness], dtype=np.float32),
        color=theme.floor_base,
    )
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_ceiling",
        prim_path=f"{room_root}/Ceiling",
        position=np.array([center_x, center_y, wall_height + 0.02], dtype=np.float32),
        scale=np.array([size_x, size_y, 0.04], dtype=np.float32),
        color=np.clip(theme.wall_main + 0.03, 0.0, 1.0),
    )

    wall_thickness = 0.04
    half_x = size_x / 2.0
    half_y = size_y / 2.0
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_back_wall",
        prim_path=f"{room_root}/BackWall",
        position=np.array([center_x, center_y + half_y, wall_height / 2.0 + wall_base_z], dtype=np.float32),
        scale=np.array([size_x, wall_thickness, wall_height], dtype=np.float32),
        color=theme.wall_accent,
    )
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_front_wall",
        prim_path=f"{room_root}/FrontWall",
        position=np.array([center_x, center_y - half_y, wall_height / 2.0 + wall_base_z], dtype=np.float32),
        scale=np.array([size_x, wall_thickness, wall_height], dtype=np.float32),
        color=theme.wall_main,
    )
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_left_wall",
        prim_path=f"{room_root}/LeftWall",
        position=np.array([center_x - half_x, center_y, wall_height / 2.0 + wall_base_z], dtype=np.float32),
        scale=np.array([wall_thickness, size_y, wall_height], dtype=np.float32),
        color=theme.wall_main,
    )
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_right_wall",
        prim_path=f"{room_root}/RightWall",
        position=np.array([center_x + half_x, center_y, wall_height / 2.0 + wall_base_z], dtype=np.float32),
        scale=np.array([wall_thickness, size_y, wall_height], dtype=np.float32),
        color=theme.wall_main,
    )

    baseboard_h = 0.10
    baseboard_t = 0.02
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_baseboard_back",
        prim_path=f"{room_root}/BaseboardBack",
        position=np.array([center_x, center_y + half_y - 0.02, baseboard_h / 2.0], dtype=np.float32),
        scale=np.array([size_x - 0.02, baseboard_t, baseboard_h], dtype=np.float32),
        color=theme.trim,
    )
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_baseboard_left",
        prim_path=f"{room_root}/BaseboardLeft",
        position=np.array([center_x - half_x + 0.02, center_y, baseboard_h / 2.0], dtype=np.float32),
        scale=np.array([baseboard_t, size_y - 0.02, baseboard_h], dtype=np.float32),
        color=theme.trim,
    )
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_baseboard_right",
        prim_path=f"{room_root}/BaseboardRight",
        position=np.array([center_x + half_x - 0.02, center_y, baseboard_h / 2.0], dtype=np.float32),
        scale=np.array([baseboard_t, size_y - 0.02, baseboard_h], dtype=np.float32),
        color=theme.trim,
    )

    tile_size = 0.42
    cols = max(2, int(round(size_x / tile_size)))
    rows = max(2, int(round(size_y / tile_size)))
    tile_w = size_x / cols
    tile_h = size_y / rows
    for col in range(cols):
        for row in range(rows):
            color = theme.floor_tiles[(col + row) % len(theme.floor_tiles)]
            _add_box(
                world,
                name=f"{room_root.split('/')[-1]}_tile_{col:02d}_{row:02d}",
                prim_path=f"{room_root}/Tiles/Tile_{col:02d}_{row:02d}",
                position=np.array(
                    [
                        center_x - size_x / 2.0 + tile_w * (col + 0.5),
                        center_y - size_y / 2.0 + tile_h * (row + 0.5),
                        floor_top_z + 0.0015,
                    ],
                    dtype=np.float32,
                ),
                scale=np.array([tile_w - 0.012, tile_h - 0.012, 0.003], dtype=np.float32),
                color=color,
            )

    window_center = np.array([center_x - size_x * 0.23, center_y + half_y - 0.03, 1.35], dtype=np.float32)
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_window_glow",
        prim_path=f"{room_root}/Window/Glow",
        position=window_center,
        scale=np.array([0.90, 0.01, 0.82], dtype=np.float32),
        color=theme.glass,
    )
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_window_frame_top",
        prim_path=f"{room_root}/Window/FrameTop",
        position=window_center + np.array([0.0, 0.0, 0.42], dtype=np.float32),
        scale=np.array([0.98, 0.03, 0.05], dtype=np.float32),
        color=theme.trim,
    )
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_window_frame_bottom",
        prim_path=f"{room_root}/Window/FrameBottom",
        position=window_center + np.array([0.0, 0.0, -0.42], dtype=np.float32),
        scale=np.array([0.98, 0.03, 0.05], dtype=np.float32),
        color=theme.trim,
    )
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_window_frame_left",
        prim_path=f"{room_root}/Window/FrameLeft",
        position=window_center + np.array([-0.46, 0.0, 0.0], dtype=np.float32),
        scale=np.array([0.05, 0.03, 0.85], dtype=np.float32),
        color=theme.trim,
    )
    _add_box(
        world,
        name=f"{room_root.split('/')[-1]}_window_frame_right",
        prim_path=f"{room_root}/Window/FrameRight",
        position=window_center + np.array([0.46, 0.0, 0.0], dtype=np.float32),
        scale=np.array([0.05, 0.03, 0.85], dtype=np.float32),
        color=theme.trim,
    )


def add_styled_workcell_decor(
    world: World,
    *,
    decor_root: str,
    station_center: np.ndarray,
    theme: SceneTheme,
    table_size: np.ndarray,
) -> None:
    station_center = np.asarray(station_center, dtype=np.float32)
    table_size = np.asarray(table_size, dtype=np.float32)
    floor_z = 0.0
    table_surface_z = float(station_center[2] + table_size[2] / 2.0)
    counter_depth_center_y = float(station_center[1] + table_size[1] / 2.0 + 0.10)
    counter_body_h = 0.58
    counter_top_h = 0.04
    counter_body_top_z = counter_body_h
    counter_top_center_z = counter_body_top_z + counter_top_h / 2.0
    counter_top_surface_z = counter_body_top_z + counter_top_h
    console_body_h = 0.58
    console_top_h = 0.03
    console_center_y = float(station_center[1] - table_size[1] / 2.0 - 0.19)
    console_top_center_z = console_body_h + console_top_h / 2.0
    console_top_surface_z = console_body_h + console_top_h

    rug_center = np.array([station_center[0] + 0.04, station_center[1] - 0.02, floor_z + 0.001], dtype=np.float32)
    _add_box(
        world,
        name=f"{decor_root.split('/')[-1]}_rug",
        prim_path=f"{decor_root}/Rug",
        position=rug_center,
        scale=np.array([table_size[0] * 1.15, table_size[1] * 0.92, 0.002], dtype=np.float32),
        color=theme.rug,
    )

    _add_box(
        world,
        name=f"{decor_root.split('/')[-1]}_backsplash",
        prim_path=f"{decor_root}/Backsplash",
        position=np.array([station_center[0] + 0.22, station_center[1] + table_size[1] / 2.0 + 0.22, counter_top_surface_z + 0.36], dtype=np.float32),
        scale=np.array([1.00, 0.02, 0.72], dtype=np.float32),
        color=theme.wall_main,
    )
    _add_box(
        world,
        name=f"{decor_root.split('/')[-1]}_counter_lower",
        prim_path=f"{decor_root}/CounterLower",
        position=np.array([station_center[0] + 0.22, counter_depth_center_y, counter_body_h / 2.0], dtype=np.float32),
        scale=np.array([1.02, 0.30, counter_body_h], dtype=np.float32),
        color=theme.cabinet,
    )
    _add_box(
        world,
        name=f"{decor_root.split('/')[-1]}_counter_top",
        prim_path=f"{decor_root}/CounterTop",
        position=np.array([station_center[0] + 0.22, counter_depth_center_y, counter_top_center_z], dtype=np.float32),
        scale=np.array([1.08, 0.34, counter_top_h], dtype=np.float32),
        color=theme.countertop,
    )
    _add_box(
        world,
        name=f"{decor_root.split('/')[-1]}_shelf",
        prim_path=f"{decor_root}/OpenShelf",
        position=np.array([station_center[0] + 0.68, station_center[1] + table_size[1] / 2.0 + 0.05, 1.16], dtype=np.float32),
        scale=np.array([0.32, 0.14, 0.03], dtype=np.float32),
        color=theme.trim,
    )
    _add_box(
        world,
        name=f"{decor_root.split('/')[-1]}_side_console",
        prim_path=f"{decor_root}/SideConsole",
        position=np.array([station_center[0] - 0.34, console_center_y, console_body_h / 2.0], dtype=np.float32),
        scale=np.array([0.36, 0.20, console_body_h], dtype=np.float32),
        color=theme.cabinet,
    )
    _add_box(
        world,
        name=f"{decor_root.split('/')[-1]}_console_top",
        prim_path=f"{decor_root}/SideConsoleTop",
        position=np.array([station_center[0] - 0.34, console_center_y, console_top_center_z], dtype=np.float32),
        scale=np.array([0.40, 0.24, console_top_h], dtype=np.float32),
        color=theme.countertop,
    )

    for idx, x_offset in enumerate((0.55, 0.79)):
        decor_h = 0.10 if idx == 0 else 0.12
        _add_box(
            world,
            name=f"{decor_root.split('/')[-1]}_decor_{idx:02d}",
            prim_path=f"{decor_root}/CounterDecor/Decor_{idx:02d}",
            position=np.array([station_center[0] + x_offset, counter_depth_center_y, counter_top_surface_z + decor_h / 2.0], dtype=np.float32),
            scale=np.array([0.05, 0.05, decor_h], dtype=np.float32),
            color=theme.decor if idx == 0 else theme.metal,
        )

    pot_h = 0.10
    _add_box(
        world,
        name=f"{decor_root.split('/')[-1]}_plant_pot",
        prim_path=f"{decor_root}/Plant/Pot",
        position=np.array([station_center[0] - 0.34, console_center_y, console_top_surface_z + pot_h / 2.0], dtype=np.float32),
        scale=np.array([0.08, 0.08, pot_h], dtype=np.float32),
        color=theme.decor,
    )
    _add_box(
        world,
        name=f"{decor_root.split('/')[-1]}_plant_leaves",
        prim_path=f"{decor_root}/Plant/Leaves",
        position=np.array([station_center[0] - 0.34, console_center_y, console_top_surface_z + pot_h + 0.12], dtype=np.float32),
        scale=np.array([0.14, 0.14, 0.22], dtype=np.float32),
        color=np.clip(theme.glass * np.array([0.45, 1.05, 0.55], dtype=np.float32), 0.0, 1.0),
    )

    stage = get_current_stage()
    for idx, x_offset in enumerate((-0.18, 0.18)):
        light_path = f"{decor_root}/PendantLight_{idx:02d}"
        light = UsdLux.SphereLight.Define(stage, light_path)
        light.CreateRadiusAttr(0.08)
        light.CreateIntensityAttr(2400.0)
        light.CreateColorAttr(
            Gf.Vec3f(
                float(theme.pendant_tint[0]),
                float(theme.pendant_tint[1]),
                float(theme.pendant_tint[2]),
            )
        )
        _set_xform(
            light_path,
            station_center + np.array([x_offset, table_size[1] / 2.0 + 0.04, 1.92], dtype=np.float32),
            (0.0, 0.0, 0.0),
        )


def add_table_finish(
    world: World,
    *,
    table_root: str,
    table_center: np.ndarray,
    table_size: np.ndarray,
    theme: SceneTheme,
) -> None:
    table_center = np.asarray(table_center, dtype=np.float32)
    table_size = np.asarray(table_size, dtype=np.float32)
    top_position = table_center + np.array([0.0, 0.0, table_size[2] / 2.0 - 0.02], dtype=np.float32)
    _add_box(
        world,
        name=f"{table_root.split('/')[-1]}_surface_top",
        prim_path=f"{table_root}/SurfaceTop",
        position=top_position,
        scale=np.array([table_size[0] * 0.98, table_size[1] * 0.98, 0.03], dtype=np.float32),
        color=theme.countertop,
    )
    for idx, x_sign in enumerate((-1.0, 1.0)):
        for jdx, y_sign in enumerate((-1.0, 1.0)):
            leg_center = table_center + np.array(
                [
                    x_sign * (table_size[0] / 2.0 - 0.08),
                    y_sign * (table_size[1] / 2.0 - 0.08),
                    -0.02,
                ],
                dtype=np.float32,
            )
            _add_box(
                world,
                name=f"{table_root.split('/')[-1]}_leg_{idx}{jdx}",
                prim_path=f"{table_root}/DecorLegs/Leg_{idx}{jdx}",
                position=leg_center,
                scale=np.array([0.05, 0.05, table_size[2] - 0.04], dtype=np.float32),
                color=theme.trim,
            )


def add_optional_robocasa_assets(
    *,
    asset_root: str | None,
    world_root: str,
    placements: list[tuple[str, tuple[str, ...], np.ndarray, tuple[float, float, float]]],
) -> int:
    if not asset_root:
        return 0

    asset_root_path = Path(asset_root).expanduser().resolve()
    if not asset_root_path.exists():
        print(f"[robocasa] asset root not found: {asset_root_path}")
        return 0

    usd_files = list(_cached_usd_assets(str(asset_root_path)))
    if not usd_files:
        print(f"[robocasa] no USD assets found under: {asset_root_path}")
        return 0

    used_paths: set[Path] = set()
    stage = get_current_stage()
    added_count = 0
    for index, (label, keywords, position, rotation_xyz_deg) in enumerate(placements):
        selected_path: Path | None = None
        for candidate in usd_files:
            if candidate in used_paths:
                continue
            candidate_lower = candidate.as_posix().lower()
            if all(keyword.lower() not in candidate_lower for keyword in keywords):
                continue
            selected_path = candidate
            break
        if selected_path is None:
            continue

        prim_path = f"{world_root}/RoboCasaAssets/{label}_{index:02d}"
        add_reference_to_stage(usd_path=str(selected_path), prim_path=prim_path)
        xform = UsdGeom.XformCommonAPI(stage.GetPrimAtPath(prim_path))
        xform.SetTranslate(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
        xform.SetRotate(Gf.Vec3f(*rotation_xyz_deg), UsdGeom.XformCommonAPI.RotationOrderXYZ)
        used_paths.add(selected_path)
        added_count += 1

    if added_count:
        print(f"[robocasa] added {added_count} optional assets from {asset_root_path}")
    else:
        print(f"[robocasa] no matching optional USD assets found under {asset_root_path}")
    return added_count

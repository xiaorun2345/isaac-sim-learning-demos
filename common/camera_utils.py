"""Helpers for simple camera demos."""


def format_rgb_shape(rgb):
    if rgb is None:
        return "no frame received"
    shape = getattr(rgb, "shape", None)
    dtype = getattr(rgb, "dtype", None)
    return f"frame shape={shape}, dtype={dtype}"

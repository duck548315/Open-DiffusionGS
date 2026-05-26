"""
Turntable video / GIF export for GaussianModel.

Replaces the horizontal-strip approach of render_turntable() with a proper
animated visualization so the object reads as a single rotating 3D model.
"""

import os
import numpy as np
import torch
from PIL import Image

from diffusionGS.models.gsrenderer.gs_core import render_opencv_cam, get_turntable_cameras


@torch.no_grad()
def render_turntable_frames(
    gaussians,
    resolution: int = 384,
    num_views: int = 24,
    elevation: float = 15.0,
) -> list[np.ndarray]:
    """
    Render a full turntable rotation and return individual frames.

    Returns
    -------
    List of num_views numpy arrays, each [H, W, 3] uint8.
    """
    w, h, _v, fxfycxcy_np, c2w_np = get_turntable_cameras(
        h=resolution, w=resolution, num_views=num_views, elevation=elevation
    )
    device = gaussians._xyz.device
    fxfycxcy = torch.from_numpy(fxfycxcy_np).float().to(device)
    c2w = torch.from_numpy(c2w_np).float().to(device)

    frames = []
    for vi in range(num_views):
        rendered = render_opencv_cam(gaussians, h, w, c2w[vi], fxfycxcy[vi])["render"]
        frame = (rendered.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        frames.append(frame)

    torch.cuda.empty_cache()
    return frames


def save_turntable_gif(
    gaussians,
    path: str,
    resolution: int = 384,
    num_views: int = 24,
    fps: int = 12,
    elevation: float = 15.0,
):
    """
    Render a looping GIF of the object rotating and save to `path` (.gif).
    Use save_turntable_video() for Gradio display (mp4 is more compatible).
    """
    frames = render_turntable_frames(gaussians, resolution, num_views, elevation)
    duration_ms = int(1000 / fps)
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
    )
    return path


def save_view_grid(
    gaussians,
    path: str,
    resolution: int = 384,
    num_views: int = 4,
    elevation: float = 15.0,
    layout: str = "horizontal",
):
    """
    Render num_views evenly-spaced views and save as a PNG grid.

    layout: "horizontal" → 1×N strip  |  "grid" → 2×2 (only when num_views==4)
    """
    frames = render_turntable_frames(gaussians, resolution, num_views, elevation)
    n = len(frames)
    h, w = frames[0].shape[:2]

    if layout == "grid" and n == 4:
        top = np.concatenate(frames[:2], axis=1)
        bot = np.concatenate(frames[2:], axis=1)
        canvas = np.concatenate([top, bot], axis=0)
    else:
        canvas = np.concatenate(frames, axis=1)

    Image.fromarray(canvas).save(path)
    return path


def save_turntable_video(
    gaussians,
    path: str,
    resolution: int = 384,
    num_views: int = 24,
    fps: int = 12,
    elevation: float = 15.0,
):
    """
    Render a turntable MP4 video and save to `path` (.mp4).
    Preferred for Gradio gr.Video components.
    Falls back to GIF if imageio-ffmpeg is unavailable.
    """
    import imageio

    frames = render_turntable_frames(gaussians, resolution, num_views, elevation)
    ext = os.path.splitext(path)[1].lower()

    if ext == ".gif":
        return save_turntable_gif(gaussians, path, resolution, num_views, fps, elevation)

    try:
        writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=7)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
    except Exception:
        # ffmpeg not available — fall back to GIF with same stem
        gif_path = os.path.splitext(path)[0] + ".gif"
        save_turntable_gif(gaussians, gif_path, resolution, num_views, fps, elevation)
        return gif_path

    return path

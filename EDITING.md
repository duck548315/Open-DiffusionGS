# DiffusionGS — Image to 3D with VLM Style Editing

This extension adds text-guided appearance editing to DiffusionGS by inserting an InstructPix2Pix editing stage before 3D reconstruction.

**Key idea**: 2D image editing models are strong. Single-view 3D generation is strong. Combining them lets a single input photo produce multiple styled 3D objects from different text prompts.

---

## Pipeline

```
Input Image
    │
    ▼
[Stage 0]  InstructPix2Pix  (optional)
    │  instruction prompt → edited image
    │  Load → edit → unload  (frees VRAM before DiffusionGS)
    │
    ▼
[Stage 1]  DiffusionGS
    │  edited image → GaussianModel
    │
    ▼
Output
    ├── output.gif / output.mp4  (turntable)
    ├── output.ply               (Gaussian splat)
    └── output.obj               (mesh)
```

### Why edit in 2D first?

VRAM is tight during DiffusionGS denoising — inserting another model into the diffusion loop is not feasible on a single consumer GPU. The staged approach:

1. Runs InstructPix2Pix, then unloads it completely.
2. Runs DiffusionGS at fp16 on the edited image.

The 2D edit is fully absorbed by the 3D reconstruction, so the resulting GaussianModel reflects both the geometry and the appearance of the edited image.

---

## Files

| File | Description |
|------|-------------|
| `diffusionGS/editing/vlm_edit.py` | `edit_image()` — InstructPix2Pix 2D editing |
| `diffusionGS/editing/visualize.py` | `save_turntable_gif`, `save_turntable_video`, `render_turntable_frames` |
| `diffusionGS/editing/__init__.py` | Module exports |
| `edit_3d.py` | CLI entry point |
| `app.py` | Gradio web demo |

---

## Usage

### CLI

```bash
# Image → 3D only
python edit_3d.py --image photo.png

# VLM edit → 3D
python edit_3d.py --image photo.png --vlm_prompt "make it made of gold"

# Multiple styles from one image
python edit_3d.py --image photo.png --vlm_prompt "make it made of gold" --output_dir out/gold
python edit_3d.py --image photo.png --vlm_prompt "turn it into marble" --output_dir out/marble
python edit_3d.py --image photo.png --vlm_prompt "make it look like crystal" --output_dir out/crystal
```

### Gradio Demo

```bash
python app.py
```

### Python API

```python
import torch
from PIL import Image
from diffusionGS.pipline_obj import DiffusionGSPipeline
from diffusionGS.editing import edit_image

# Stage 0 — VLM 2D edit
src = Image.open("photo.png").convert("RGB")
edited = edit_image(src, prompt="make it made of gold", device="cuda:0")[0]

# Stage 1 — 3D reconstruction
pipeline = DiffusionGSPipeline.from_pretrained(
    "CaiYuanhao/DiffusionGS", device="cuda:0", torch_dtype=torch.float16
)
gs_output = pipeline(edited, seed=42, extract_mesh=True)

gs_output.gaussians.save_ply("out/output.ply")
gs_output.mesh.export("out/output.obj")
```

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | *(required)* | Input image path (jpg, png, avif, webp, …) |
| `--vlm_prompt` | `None` | InstructPix2Pix edit instruction |
| `--vlm_img_guidance` | `1.5` | Image guidance scale (higher = preserve more structure) |
| `--vlm_guidance` | `7.5` | Text guidance scale (higher = stronger edit) |
| `--seed` | `42` | RNG seed |
| `--foreground_ratio` | `0.825` | Foreground crop ratio |
| `--output_dir` | `debug/edit` | Output directory |
| `--device` | `cuda:0` | Torch device |
| `--no_mesh` | `False` | Skip mesh extraction (faster) |

---

## InstructPix2Pix Parameters

- **`vlm_img_guidance`** (1.0–3.0): how closely the edit preserves the original structure. Higher = more shape preserved.
- **`vlm_guidance`** (3–15): how strongly the text instruction is followed. Higher = stronger appearance change.

Typical starting point: `--vlm_img_guidance 1.5 --vlm_guidance 7.5`

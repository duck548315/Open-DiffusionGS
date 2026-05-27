# DiffusionGS — Dual-Path Geometry + Color Pipeline

## Core idea

User gives **one** edit instruction. Internally the system splits it and runs
two separate image-editing + 3D-reconstruction paths with **interleaved
geometry injection**:

- **Geometry path** — denoises from a geo-focused edit image; drives xyz / scale / rotation
- **Color path**    — denoises from a color-focused edit image; at every step
  its noisy novel views are replaced with renders of the current geo-path
  Gaussians, so it converges to the same geometry while producing accurate RGB

The final output is the fully-denoised `color_gs` — geometry was already
guided by geo throughout, so no post-hoc merge is needed.

---

## Pipeline

```
┌─────────────────────────────────────────────────────────┐
│  Image mode          │  Text mode                       │
│  user uploads photo  │  Step 1: user describes object   │
│                      │    → VLM → t2i prompt            │
│                      │    → FLUX generates base image   │
│                      │    → shown to user for review    │
│                      │  Step 2: user enters edit        │
└──────────┬───────────┴──────────────┬────────────────────┘
           │                          │
           └──────────┬───────────────┘
                      │  image + edit instruction
                      ▼
            ┌─────────────────┐
            │  Stage 1 — VLM  │  Gemini sees image + instruction
            │  (multimodal)   │  → geo_edit_prompt
            └────────┬────────┘  → color_edit_prompt
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
  Stage 2 — Flux Kontext   Stage 3 — Flux Kontext
  image + geo_edit          image + color_edit
  → geo_image               → color_image
          │                     │
          └──────────┬──────────┘
                     ▼
         Stage 4 — Dual DiffusionGS  (30 steps)
         ┌──────────────────────────────────────┐
         │  geo path   ──denoises──▶ geo_gs_t   │
         │                              │        │
         │             render 3 views   │        │
         │                              ▼        │
         │  color path ◀── inject as image_noisy │
         │             ──denoises──▶ color_gs_t  │
         └──────────────────────────────────────┘
                     │
                     ▼
             color_gs  (filtered, output)
             geometry guided by geo path
             colors from color path
```

---

## Code structure

```python
# app.py
_run_pipeline(pil_image, edit_instruction, ...)   # shared core
run_image_mode(...)    # np.array → PIL → _run_pipeline
run_text_step2(...)    # state PIL → _run_pipeline  (identical path)
generate_base(text)    # text-only Step 1: VLM t2i prompt → FLUX → show to user

# diffusionGS/pipline_obj.py
DGSPipeline.dual_call(geo_image, color_image, ...)
  # interleaved denoising loop — returns filtered color_gs
```

---

## Interleaved injection (dual_call)

Every denoising step (including step 0):
1. Both paths call `p_sample` to get predicted Gaussians
2. Take geo path's xyz/scaling/rotation/opacity + color path's features
3. Render 3 novel views via `render_opencv_cam` (cameras must be on GPU —
   `cam_template` is loaded from disk to CPU, move with `.to(device)`)
4. Feed those renders as `image_noisy` into color path's next step via `q_sample`

Final output: `color_gs` with `apply_all_filters(opacity_thres=0.02, crop_bbx=[-0.91…0.91])`

---

## Files

| File | Description |
|------|-------------|
| `diffusionGS/editing/generate.py` | `decompose_prompt()` — VLM splits one instruction into geo + color edit prompts; `text_to_image()` — FLUX / Gemini |
| `diffusionGS/editing/merge.py` | `merge_gaussians()` — kept for reference, no longer used in main path |
| `diffusionGS/editing/vlm_edit.py` | `edit_image()` — Flux Kontext / Gemini / IP2P 2D editing |
| `diffusionGS/editing/visualize.py` | turntable video / view grid helpers |
| `diffusionGS/editing/__init__.py` | module exports |
| `app.py` | Gradio UI — two tabs (image mode / text mode) |

---

## Backends

| Task | Replicate | Gemini | Local fallback |
|---|---|---|---|
| VLM decompose | — | `gemini-2.5-flash` | string augmentation |
| Text-to-image | `flux-schnell` | `gemini-2.5-flash-image` | — |
| Image editing | `flux-kontext-dev` | `gemini-2.5-flash-image` | InstructPix2Pix |

`.env` keys:
```
REPLICATE_API_TOKEN=r8_...
GEMINI_API_KEY=AI...
```

---

## VRAM notes

- DiT transformer uses `xformers.memory_efficient_attention` (chunked)
- `dual_call` runs under `@torch.no_grad()` — no gradient graph retained
- WSL2: VRAM overflow silently spills to system RAM → very slow (18 min/step
  observed). Run on native Linux with sufficient VRAM to avoid this.
- Server target: single GPU with ≥16 GB VRAM recommended

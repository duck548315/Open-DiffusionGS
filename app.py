"""
Gradio demo — Dual-path geometry+color splatter merge pipeline.

Flow (both modes share Phase 1 + Phase 2 after the input image is obtained):

  Phase 1: VLM decomposes edit instruction → Flux Kontext × 2
           → geo_img + color_img shown to user for review
  [user confirms or re-runs Phase 1 with adjusted instruction]
  Phase 2: DiffusionGS × 2 → merge splatter → 3DGS output

Text mode has an extra Step 0: text-to-image → base image shown to user
before they enter an edit instruction.
"""

import os
import tempfile

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image

import gradio as gr

load_dotenv()
from diffusionGS.pipline_obj import DiffusionGSPipeline
from diffusionGS.editing import (
    save_turntable_video,
    save_view_grid,
    edit_image,
    decompose_prompt,
    text_to_image,
)

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
_pipeline: DiffusionGSPipeline = None


def _get_pipeline() -> DiffusionGSPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DiffusionGSPipeline.from_pretrained(
            "CaiYuanhao/DiffusionGS",
            device=DEVICE,
            torch_dtype=torch.float16,
        )
    _pipeline.system = _pipeline.system.to(DEVICE)
    return _pipeline


def _offload_pipeline():
    global _pipeline
    if _pipeline is not None:
        _pipeline.system = _pipeline.system.cpu()
        torch.cuda.empty_cache()


def _dgs_dual(geo_img: Image.Image, color_img: Image.Image, seed: int, fg: float):
    return _get_pipeline().dual_call(
        geo_img, color_img,
        seed=int(seed),
        foreground_ratio=float(fg),
    )


def _flux(pil_image, prompt, backend, gemini_model, vlm_img_g, vlm_g, contrast, seed):
    return edit_image(
        pil_image,
        prompt=prompt,
        backend=backend,
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        replicate_token=os.environ.get("REPLICATE_API_TOKEN"),
        gemini_model=gemini_model or "gemini-2.5-flash-image",
        contrast_boost=float(contrast),
        image_guidance_scale=float(vlm_img_g),
        guidance_scale=float(vlm_g),
        seed=int(seed),
    )[0]


def _export(gaussians, layout: str, extract_mesh: bool = True):
    out_dir = tempfile.mkdtemp(prefix="diffusionGS_")
    mp4  = save_turntable_video(gaussians, os.path.join(out_dir, "output.mp4"))
    views = save_view_grid(gaussians, os.path.join(out_dir, "views.png"), layout=layout)
    ply  = os.path.join(out_dir, "output.ply")
    gaussians.save_ply(ply)
    obj = None
    if extract_mesh:
        try:
            mesh = gaussians.extract_mesh()
            obj = os.path.join(out_dir, "output.obj")
            mesh.export(obj)
        except Exception as e:
            print(f"[Export] mesh skipped: {e}")
    return mp4, views, ply, obj


# ------------------------------------------------------------------ #
# Phase 1 — VLM + Flux × 2  (shared)
# Yields: (geo_img, color_img, geo_state, color_state, status)
# ------------------------------------------------------------------ #

def _phase1(pil_image, edit_instruction, backend, gemini_model, decompose_model,
            vlm_img_g, vlm_g, contrast, seed):
    if not edit_instruction or not edit_instruction.strip():
        raise gr.Error("Please enter an edit instruction before generating edited images.")

    yield None, None, None, None, "⏳ Stage 1: VLM analyzing image + instruction..."
    geo_edit, color_edit = decompose_prompt(
        edit_instruction.strip(),
        image=pil_image,
        gemini_api_key=os.environ.get("GEMINI_API_KEY"),
        model=decompose_model or "gemini-2.5-flash",
    )
    print(f"[VLM] geo_edit:   {geo_edit}")
    print(f"[VLM] color_edit: {color_edit}")

    yield None, None, None, None, "⏳ Stage 2: Flux Kontext — geometry edit..."
    pil_geo = _flux(pil_image, geo_edit, backend, gemini_model,
                    vlm_img_g, vlm_g, contrast, seed)

    # Color edit is applied ON the geo image, not the original,
    # so both outputs share the same geometry base.
    yield pil_geo, None, None, None, "⏳ Stage 3: Flux Kontext — color edit on geo image..."
    pil_color = _flux(pil_geo, color_edit, backend, gemini_model,
                      vlm_img_g, vlm_g, contrast, seed)

    yield pil_geo, pil_color, pil_geo, pil_color, (
        "✅ Both edited images ready. Review them — if happy, click Generate 3D."
    )


# ------------------------------------------------------------------ #
# Phase 2 — DGS × 2 + merge + export  (shared)
# Yields: (video, views, ply, obj, status)
# ------------------------------------------------------------------ #

def _phase2(geo_img, color_img, seed, fg, layout, extract_mesh):
    if geo_img is None or color_img is None:
        raise gr.Error("Please run Phase 1 first to generate the edited images.")

    yield None, None, None, None, "⏳ Stage 4: DiffusionGS — interleaved dual denoising..."
    gaussians = _dgs_dual(geo_img, color_img, seed, fg)
    _offload_pipeline()

    yield None, None, None, None, "⏳ Exporting..."
    mp4, views, ply, obj = _export(gaussians, layout, extract_mesh=bool(extract_mesh))
    yield mp4, views, ply, obj, "✅ Done!"


# ------------------------------------------------------------------ #
# Image mode wrappers
# ------------------------------------------------------------------ #

def img_phase1(input_image, edit_instruction, backend, gemini_model, decompose_model,
               vlm_img_g, vlm_g, contrast, seed):
    if input_image is None:
        raise gr.Error("Please upload an input image.")
    pil = Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image
    yield from _phase1(pil, edit_instruction, backend, gemini_model, decompose_model,
                       vlm_img_g, vlm_g, contrast, seed)


def img_phase2(geo_state, color_state, seed, fg, layout, extract_mesh):
    yield from _phase2(geo_state, color_state, seed, fg, layout, extract_mesh)


# ------------------------------------------------------------------ #
# Text mode — Step 0: generate base image
# ------------------------------------------------------------------ #

def generate_base(text_prompt, backend, gemini_model, decompose_model):
    if not text_prompt or not text_prompt.strip():
        raise gr.Error("Please enter an object description.")

    gem_key  = os.environ.get("GEMINI_API_KEY")
    rep_token = os.environ.get("REPLICATE_API_TOKEN")

    T2I_SUFFIX = (
        ", entire object fully visible from top to bottom, not cropped, "
        "isolated single object centered in frame, plain white background, "
        "front-facing view, product photography style"
    )

    yield None, None, "⏳ VLM generating t2i prompt..."
    geo_t2i, _ = decompose_prompt(text_prompt.strip(), image=None, gemini_api_key=gem_key,
                                   model=decompose_model or "gemini-2.5-flash")
    geo_t2i_final = geo_t2i + T2I_SUFFIX
    print(f"[VLM] geo_t2i_prompt: {geo_t2i_final}")

    yield None, None, "⏳ Generating base image (text-to-image)..."
    pil_base = text_to_image(geo_t2i_final, backend=backend,
                             replicate_token=rep_token, gemini_api_key=gem_key)

    yield pil_base, pil_base, "✅ Base image ready — enter your edit instruction, then click Generate Edited Images."


# ------------------------------------------------------------------ #
# Text mode wrappers (Phase 1 + 2 use base image from state)
# ------------------------------------------------------------------ #

def txt_phase1(base_state, edit_instruction, backend, gemini_model, decompose_model,
               vlm_img_g, vlm_g, contrast, seed):
    if base_state is None:
        raise gr.Error("Please generate a base image first (Step 0).")
    yield from _phase1(base_state, edit_instruction, backend, gemini_model, decompose_model,
                       vlm_img_g, vlm_g, contrast, seed)


def txt_phase2(geo_state, color_state, seed, fg, layout, extract_mesh):
    yield from _phase2(geo_state, color_state, seed, fg, layout, extract_mesh)


# ------------------------------------------------------------------ #
# UI helpers
# ------------------------------------------------------------------ #

def _backend_controls():
    backend = gr.Radio(
        choices=["auto", "replicate", "gemini", "ip2p"],
        value="auto",
        label="Edit backend  (auto: Replicate → Gemini → IP2P)",
    )
    active = []
    if os.environ.get("REPLICATE_API_TOKEN"):
        active.append("Replicate ✓")
    if os.environ.get("GEMINI_API_KEY"):
        active.append("Gemini ✓")
    if not active:
        active.append("IP2P (local)")
    gr.Markdown(f"**Keys from .env:** {', '.join(active)}")
    gemini_model = gr.Dropdown(
        choices=["gemini-2.5-flash-image", "gemini-2.0-flash-preview-image-generation",
                 "gemini-1.5-flash", "gemini-3.1-flash-image-preview"],
        value="gemini-2.5-flash-image",
        label="Gemini edit model",
    )
    decompose_model = gr.Dropdown(
        choices=["gemini-2.5-flash", "gemini-2.5-flash-lite-preview-06-17",
                 "gemini-2.5-pro", "gemini-2.5-pro-preview-06-05"],
        value="gemini-2.5-flash",
        label="Gemini decompose model  (auto-fallbacks on quota error)",
    )
    with gr.Row():
        vlm_img_g = gr.Slider(1.0, 3.0, value=1.5, step=0.1, label="[IP2P] Structure preservation")
        vlm_g     = gr.Slider(3.0, 15.0, value=7.5, step=0.5, label="[IP2P] Edit strength")
    with gr.Accordion("Advanced", open=False):
        contrast      = gr.Slider(1.0, 2.0, value=1.0, step=0.05,  label="Contrast boost")
        seed          = gr.Slider(0, 9999, value=42, step=1,        label="Seed")
        fg            = gr.Slider(0.5, 1.0, value=0.825, step=0.025, label="Foreground ratio")
        layout        = gr.Radio(["horizontal", "grid"], value="horizontal", label="4-view layout")
        extract_mesh  = gr.Checkbox(value=False, label="Extract mesh (OBJ)  — slow, skip if not needed")
    return backend, gemini_model, decompose_model, vlm_img_g, vlm_g, contrast, seed, fg, layout, extract_mesh


# ------------------------------------------------------------------ #
# Gradio UI
# ------------------------------------------------------------------ #

CSS = "#status { font-size: 1.05em; font-weight: 500; }"

with gr.Blocks(title="DiffusionGS — Dual-Path", css=CSS) as demo:
    gr.Markdown(
        """
        # DiffusionGS — Geometry + Color Dual-Path Pipeline

        One edit instruction → VLM splits it → Flux Kontext × 2 → **review both images** →
        confirm → DiffusionGS × 2 → merge → 3DGS.
        """
    )

    with gr.Tabs():

        # ============================================================
        # TAB 1 — Image mode
        # ============================================================
        with gr.Tab("📷 Image → 3D"):
            img_geo_state   = gr.State(value=None)
            img_color_state = gr.State(value=None)

            with gr.Row():
                # ---- inputs -----------------------------------------
                with gr.Column(scale=1):
                    img_input = gr.Image(label="Input image", type="numpy", height=260)
                    img_edit  = gr.Textbox(
                        label="Edit instruction",
                        placeholder='"make the head red and body golden"  /  "turn it into marble"',
                    )
                    img_back, img_gem, img_dec, img_vi, img_vg, img_ct, img_seed, img_fg, img_lay, img_mesh = _backend_controls()
                    img_p1_btn = gr.Button("① Generate Edited Images", variant="secondary")
                    img_p2_btn = gr.Button("② Generate 3D", variant="primary", interactive=False)

                # ---- outputs ----------------------------------------
                with gr.Column(scale=1):
                    img_p1_status = gr.Textbox(label="Status", interactive=False)
                    with gr.Row():
                        img_geo_prev   = gr.Image(label="Geometry-path image", type="pil")
                        img_color_prev = gr.Image(label="Color-path image", type="pil")
                    img_p2_status = gr.Textbox(label="3D Status", interactive=False)
                    img_video = gr.Video(label="3D Turntable", autoplay=True)
                    img_views = gr.Image(label="4-View Grid", type="filepath")
                    with gr.Row():
                        img_ply = gr.File(label="PLY")
                        img_obj = gr.File(label="OBJ")

            # Phase 1
            img_p1_btn.click(
                fn=img_phase1,
                inputs=[img_input, img_edit, img_back, img_gem, img_dec,
                        img_vi, img_vg, img_ct, img_seed],
                outputs=[img_geo_prev, img_color_prev,
                         img_geo_state, img_color_state, img_p1_status],
            ).then(
                fn=lambda: gr.update(interactive=True),
                outputs=[img_p2_btn],
            )

            # Phase 2
            img_p2_btn.click(
                fn=img_phase2,
                inputs=[img_geo_state, img_color_state, img_seed, img_fg, img_lay, img_mesh],
                outputs=[img_video, img_views, img_ply, img_obj, img_p2_status],
            )

        # ============================================================
        # TAB 2 — Text mode
        # ============================================================
        with gr.Tab("📝 Text → 3D"):
            txt_base_state  = gr.State(value=None)
            txt_geo_state   = gr.State(value=None)
            txt_color_state = gr.State(value=None)

            with gr.Row():
                # ---- inputs -----------------------------------------
                with gr.Column(scale=1):
                    txt_desc   = gr.Textbox(
                        label="Object description",
                        placeholder='"a rubber duck"  /  "a leather boot"',
                        lines=2,
                    )
                    txt_s0_btn = gr.Button("Step 0: Generate Base Image", variant="secondary")
                    txt_s0_status = gr.Textbox(label="", interactive=False, lines=1)

                    gr.Markdown("---")
                    txt_edit = gr.Textbox(
                        label="Edit instruction  (enter after reviewing the base image)",
                        placeholder='"make the beak orange and body golden"  /  "turn into marble"',
                    )
                    txt_back, txt_gem, txt_dec, txt_vi, txt_vg, txt_ct, txt_seed, txt_fg, txt_lay, txt_mesh = _backend_controls()
                    txt_p1_btn = gr.Button("① Generate Edited Images", variant="secondary", interactive=False)
                    txt_p2_btn = gr.Button("② Generate 3D", variant="primary",    interactive=False)

                # ---- outputs ----------------------------------------
                with gr.Column(scale=1):
                    txt_base_prev  = gr.Image(label="Base image — review before editing", type="pil")
                    txt_p1_status  = gr.Textbox(label="Status", interactive=False)
                    with gr.Row():
                        txt_geo_prev   = gr.Image(label="Geometry-path image", type="pil")
                        txt_color_prev = gr.Image(label="Color-path image", type="pil")
                    txt_p2_status  = gr.Textbox(label="3D Status", interactive=False)
                    txt_video = gr.Video(label="3D Turntable", autoplay=True)
                    txt_views = gr.Image(label="4-View Grid", type="filepath")
                    with gr.Row():
                        txt_ply = gr.File(label="PLY")
                        txt_obj = gr.File(label="OBJ")

            # Step 0
            txt_s0_btn.click(
                fn=generate_base,
                inputs=[txt_desc, txt_back, txt_gem, txt_dec],
                outputs=[txt_base_prev, txt_base_state, txt_s0_status],
            ).then(
                fn=lambda: gr.update(interactive=True),
                outputs=[txt_p1_btn],
            )

            # Phase 1
            txt_p1_btn.click(
                fn=txt_phase1,
                inputs=[txt_base_state, txt_edit, txt_back, txt_gem, txt_dec,
                        txt_vi, txt_vg, txt_ct, txt_seed],
                outputs=[txt_geo_prev, txt_color_prev,
                         txt_geo_state, txt_color_state, txt_p1_status],
            ).then(
                fn=lambda: gr.update(interactive=True),
                outputs=[txt_p2_btn],
            )

            # Phase 2
            txt_p2_btn.click(
                fn=txt_phase2,
                inputs=[txt_geo_state, txt_color_state, txt_seed, txt_fg, txt_lay, txt_mesh],
                outputs=[txt_video, txt_views, txt_ply, txt_obj, txt_p2_status],
            )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

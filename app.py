"""
Gradio demo: Image → VLM 2D Edit → 3D Gaussian Splatting
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
from diffusionGS.editing import save_turntable_video, save_view_grid

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


def run(
    input_image,
    vlm_prompt: str,
    backend: str,
    gemini_model: str,
    vlm_img_guidance: float,
    vlm_guidance: float,
    contrast_boost: float,
    seed: int,
    foreground_ratio: float,
    view_layout: str,
):
    if input_image is None:
        raise gr.Error("Please upload an input image.")
    view_layout = str(view_layout)

    pil_input = Image.fromarray(input_image) if isinstance(input_image, np.ndarray) else input_image

    # ---- Stage 0 (optional): VLM 2D editing -------------------------
    if vlm_prompt and vlm_prompt.strip():
        from diffusionGS.editing import edit_image
        yield (None, None, None, None, None, f"⏳ Stage 0: VLM editing → '{vlm_prompt.strip()}' ...")
        pil_input = edit_image(
            pil_input,
            prompt=vlm_prompt.strip(),
            backend=backend,
            contrast_boost=float(contrast_boost),
            gemini_model=gemini_model or "gemini-2.5-flash-image",
            device=DEVICE,
            image_guidance_scale=float(vlm_img_guidance),
            guidance_scale=float(vlm_guidance),
            seed=int(seed),
        )[0]
        yield (None, None, None, None, None, "✅ VLM edit done. Starting 3D generation ...")

    # ---- Stage 1: Generate 3D ---------------------------------------
    yield (None, None, None, None, None, "⏳ Stage 1: Generating 3D from image ...")

    pipeline = _get_pipeline()
    gs_output = pipeline(
        pil_input,
        seed=int(seed),
        foreground_ratio=float(foreground_ratio),
        extract_mesh=True,
    )
    gaussians = gs_output.gaussians
    _offload_pipeline()

    # ---- Export -----------------------------------------------------
    yield (None, None, None, None, None, "⏳ Rendering & exporting ...")

    out_dir = tempfile.mkdtemp(prefix="diffusionGS_")
    out_mp4 = save_turntable_video(gaussians, os.path.join(out_dir, "output.mp4"))

    views_path = save_view_grid(
        gaussians,
        os.path.join(out_dir, "views_4.png"),
        layout=view_layout,
    )

    ply_path = os.path.join(out_dir, "output.ply")
    gaussians.save_ply(ply_path)

    obj_path = None
    if gs_output.mesh is not None:
        obj_path = os.path.join(out_dir, "output.obj")
        gs_output.mesh.export(obj_path)

    # Also save VLM-edited image if it was generated
    vlm_img_path = None
    if vlm_prompt and vlm_prompt.strip():
        vlm_img_path = os.path.join(out_dir, "vlm_edited.png")
        pil_input.save(vlm_img_path)

    yield (
        vlm_img_path,
        out_mp4,
        views_path,
        ply_path,
        obj_path,
        "✅ Done!",
    )


# ------------------------------------------------------------------ #
# Gradio UI
# ------------------------------------------------------------------ #

CSS = """
#status { font-size: 1.05em; font-weight: 500; }
"""

with gr.Blocks(title="DiffusionGS", css=CSS) as demo:
    gr.Markdown(
        """
        # DiffusionGS — Image to 3D with Style Editing

        Upload a photo to generate a 3D object.
        Optionally describe how you want to change its appearance before reconstruction.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Input Image", type="numpy", height=300)

            vlm_prompt = gr.Textbox(
                label="Edit instruction (optional)",
                placeholder='e.g. "make it made of gold" / "turn it into marble"',
            )
            backend = gr.Radio(
                choices=["auto", "replicate", "gemini", "ip2p"],
                value="auto",
                label="Backend  (auto: Replicate → Gemini → IP2P)",
            )
            active = []
            if os.environ.get("REPLICATE_API_TOKEN"):
                active.append("Replicate ✓")
            if os.environ.get("GEMINI_API_KEY"):
                active.append("Gemini ✓")
            if not active:
                active.append("IP2P (local)")
            gr.Markdown(f"**Keys loaded from .env:** {', '.join(active)}")

            gemini_model = gr.Dropdown(
                choices=["gemini-2.5-flash-image", "gemini-3.1-flash-image-preview"],
                value="gemini-2.5-flash-image",
                label="Gemini model",
            )
            with gr.Row():
                vlm_img_guidance = gr.Slider(1.0, 3.0, value=1.5, step=0.1,
                                             label="[IP2P] Structure preservation")
                vlm_guidance = gr.Slider(3.0, 15.0, value=7.5, step=0.5,
                                         label="[IP2P] Edit strength")

            with gr.Accordion("Advanced", open=False):
                contrast_boost = gr.Slider(1.0, 2.0, value=1.0, step=0.05,
                                           label="Contrast boost (1.0=off, helps when 3D looks flat)")
                seed = gr.Slider(0, 9999, value=42, step=1, label="Seed")
                foreground_ratio = gr.Slider(0.5, 1.0, value=0.825, step=0.025,
                                             label="Foreground ratio")
                view_layout = gr.Radio(
                    choices=["horizontal", "grid"],
                    value="horizontal",
                    label="4-view layout (horizontal = 1×4 strip, grid = 2×2)",
                )

            run_btn = gr.Button("🚀 Generate 3D", variant="primary")

        with gr.Column(scale=1):
            status = gr.Textbox(label="Status", interactive=False, elem_id="status")
            vlm_out = gr.Image(label="VLM Edited Image", type="filepath")
            output_video = gr.Video(label="3D Turntable", autoplay=True)
            views_out = gr.Image(label="4-View Grid (0° / 90° / 180° / 270°)", type="filepath")
            with gr.Row():
                ply_out = gr.File(label="Download PLY")
                obj_out = gr.File(label="Download OBJ")

    run_btn.click(
        fn=run,
        inputs=[
            input_image,
            vlm_prompt, backend, gemini_model,
            vlm_img_guidance, vlm_guidance,
            contrast_boost, seed, foreground_ratio, view_layout,
        ],
        outputs=[vlm_out, output_video, views_out, ply_out, obj_out, status],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)

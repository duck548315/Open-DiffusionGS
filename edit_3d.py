"""
edit_3d.py — Image → 3D Gaussian Splatting with optional VLM appearance editing

Pipeline:
  Stage 0 (optional): InstructPix2Pix 2D editing  (text prompt → edited image)
  Stage 1:            DiffusionGS 3D reconstruction (edited image → GaussianModel)

Usage examples
--------------
# image → 3D only
python edit_3d.py --image photo.png

# image → VLM edit → 3D
python edit_3d.py --image photo.png --vlm_prompt "make it made of gold"

# multiple styles from one image
python edit_3d.py --image photo.png --vlm_prompt "turn it into marble" --output_dir out/marble
python edit_3d.py --image photo.png --vlm_prompt "make it look like crystal" --output_dir out/crystal
"""

import argparse
import os

import torch
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

from diffusionGS.pipline_obj import DiffusionGSPipeline
from diffusionGS.editing import save_turntable_gif


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--vlm_prompt", default=None,
                   help='Edit instruction, e.g. "make it made of gold"')
    p.add_argument("--backend", default="auto",
                   choices=["auto", "replicate", "gemini", "ip2p"],
                   help="VLM backend (auto: replicate > gemini > ip2p)")
    p.add_argument("--replicate_model", default="black-forest-labs/flux-kontext-dev",
                   help="Replicate model (default: flux-kontext-dev)")
    p.add_argument("--gemini_api_key", default=None,
                   help="Gemini API key (overrides GEMINI_API_KEY in .env)")
    p.add_argument("--gemini_model", default="gemini-2.5-flash-image",
                   help="Gemini model (default: gemini-2.5-flash-image)")
    p.add_argument("--contrast_boost", type=float, default=1.0,
                   help="Contrast multiplier applied after VLM edit (1.0=off, 1.3=mild, 1.6=strong)")
    p.add_argument("--vlm_img_guidance", type=float, default=1.5,
                   help="[IP2P only] Image guidance scale")
    p.add_argument("--vlm_guidance", type=float, default=7.5,
                   help="[IP2P only] Text guidance scale")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--foreground_ratio", type=float, default=0.825)
    p.add_argument("--output_dir", default="debug/edit")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--no_mesh", action="store_true", help="Skip mesh extraction")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Stage 0 (optional): VLM 2D editing via InstructPix2Pix
    # ------------------------------------------------------------------ #
    input_image = args.image
    if args.vlm_prompt:
        from diffusionGS.editing import edit_image
        print(f"\n[0/2] VLM 2D editing: '{args.vlm_prompt}' ...")
        src = Image.open(args.image).convert("RGB")
        edited = edit_image(
            src,
            prompt=args.vlm_prompt,
            backend=args.backend,
            contrast_boost=args.contrast_boost,
            replicate_model=args.replicate_model,
            gemini_api_key=args.gemini_api_key,
            gemini_model=args.gemini_model,
            device=args.device,
            image_guidance_scale=args.vlm_img_guidance,
            guidance_scale=args.vlm_guidance,
            seed=args.seed,
        )[0]
        input_image = os.path.join(args.output_dir, "vlm_edited.png")
        edited.save(input_image)
        print(f"  VLM edit saved → {input_image}")
    else:
        print("\n[0/2] No VLM prompt — skipping 2D editing.")

    # ------------------------------------------------------------------ #
    # Stage 1: Generate 3D from (edited) image
    # ------------------------------------------------------------------ #
    print("\n[1/2] Generating 3D from image ...")
    pipeline = DiffusionGSPipeline.from_pretrained(
        "CaiYuanhao/DiffusionGS",
        device=args.device,
        torch_dtype=torch.float16,
    )
    gs_output = pipeline(
        input_image,
        seed=args.seed,
        foreground_ratio=args.foreground_ratio,
        extract_mesh=not args.no_mesh,
    )
    gaussians = gs_output.gaussians

    # ------------------------------------------------------------------ #
    # Stage 2: Export
    # ------------------------------------------------------------------ #
    print("\n[2/2] Exporting ...")

    save_turntable_gif(gaussians, os.path.join(args.output_dir, "output.gif"))
    gaussians.save_ply(os.path.join(args.output_dir, "output.ply"))
    print(f"  GIF saved → {args.output_dir}/output.gif")
    print(f"  PLY saved → {args.output_dir}/output.ply")

    if not args.no_mesh and gs_output.mesh is not None:
        gs_output.mesh.export(os.path.join(args.output_dir, "output.obj"))
        print(f"  Mesh saved → {args.output_dir}/output.obj")

    print(f"\nDone.  Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()

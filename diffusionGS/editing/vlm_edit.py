"""
2D image editing before 3D reconstruction.

Three backends (priority order when backend="auto"):
  1. Replicate  — FLUX Kontext, strongest editing, needs REPLICATE_API_TOKEN
  2. Gemini     — good for complex instructions, needs GEMINI_API_KEY
  3. IP2P       — free, local, simpler edits only (fallback)
"""

import io
import os
import time
import urllib.request

import torch
from PIL import Image


# --------------------------------------------------------------------------- #
# Replicate / FLUX Kontext
# --------------------------------------------------------------------------- #

def _edit_replicate(
    image: Image.Image,
    prompt: str,
    api_token: str,
    model: str = "black-forest-labs/flux-kontext-dev",
    num_variants: int = 1,
    max_retries: int = 6,
) -> list[Image.Image]:
    import replicate

    client = replicate.Client(api_token=api_token)

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")

    results = []
    for _ in range(num_variants):
        for attempt in range(max_retries):
            try:
                buf.seek(0)
                output = client.run(model, input={"prompt": prompt, "input_image": buf})
                if not isinstance(output, (list, tuple)):
                    output = [output]
                for item in output:
                    url = item.url if hasattr(item, "url") else str(item)
                    with urllib.request.urlopen(url) as r:
                        results.append(Image.open(io.BytesIO(r.read())).convert("RGB"))
                    break
                break  # success — exit retry loop
            except Exception as e:
                err = str(e)
                is_rate_limit = (
                    "429" in err
                    or "throttled" in err.lower()
                    or "rate limit" in err.lower()
                )
                is_server_err = "500" in err or "internal server error" in err.lower()
                if (is_rate_limit or is_server_err) and attempt < max_retries - 1:
                    wait = 3 if is_server_err else min(5 * (2 ** attempt), 60)
                    print(f"[Replicate] {'500' if is_server_err else '429'} "
                          f"(attempt {attempt+1}/{max_retries}), retrying in {wait}s…")
                    time.sleep(wait)
                else:
                    raise

    return results or [image]


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #

def _edit_gemini(
    image: Image.Image,
    prompt: str,
    api_key: str,
    model: str = "gemini-2.5-flash-image",
    num_variants: int = 1,
) -> list[Image.Image]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    image_bytes = buf.getvalue()

    results = []
    for _ in range(num_variants):
        response = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"]
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                results.append(
                    Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
                )
                break

    return results or [image]


# --------------------------------------------------------------------------- #
# InstructPix2Pix (local fallback)
# --------------------------------------------------------------------------- #

def _edit_ip2p(
    image: Image.Image,
    prompt: str,
    device: str = "cuda",
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_variants: int = 1,
    seed: int = 42,
) -> list[Image.Image]:
    from diffusers import StableDiffusionInstructPix2PixPipeline

    print("[VLM Edit] Loading InstructPix2Pix ...")
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooks/instruct-pix2pix",
        torch_dtype=torch.float16,
        safety_checker=None,
    ).to(device)
    pipe.set_progress_bar_config(desc="VLM Edit", leave=False)

    generator = torch.Generator(device=device).manual_seed(seed)
    results = pipe(
        prompt=prompt,
        image=image.convert("RGB"),
        image_guidance_scale=image_guidance_scale,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        num_images_per_prompt=num_variants,
        generator=generator,
    ).images

    del pipe
    torch.cuda.empty_cache()
    print("[VLM Edit] Done.")
    return results


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _boost_contrast(image: Image.Image, factor: float) -> Image.Image:
    """Enhance contrast of a PIL image to improve depth cues for DiffusionGS."""
    from PIL import ImageEnhance
    return ImageEnhance.Contrast(image).enhance(factor)


def edit_image(
    image: Image.Image,
    prompt: str,
    backend: str = "auto",
    contrast_boost: float = 1.0,
    # Replicate
    replicate_token: str = None,
    replicate_model: str = "black-forest-labs/flux-kontext-dev",
    # Gemini
    gemini_api_key: str = None,
    gemini_model: str = "gemini-2.5-flash-image",
    # IP2P
    device: str = "cuda",
    image_guidance_scale: float = 1.5,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 50,
    num_variants: int = 1,
    seed: int = 42,
) -> list[Image.Image]:
    """
    Edit a PIL image with a text instruction.

    backend: "auto" | "replicate" | "gemini" | "ip2p"
      auto — tries Replicate → Gemini → IP2P based on available keys.
    """
    rep_token = replicate_token or os.environ.get("REPLICATE_API_TOKEN")
    gem_key   = gemini_api_key  or os.environ.get("GEMINI_API_KEY")

    use = backend
    if use == "auto":
        use = "replicate" if rep_token else ("gemini" if gem_key else "ip2p")

    if use == "replicate":
        if not rep_token:
            raise ValueError("REPLICATE_API_TOKEN not set")
        print(f"[VLM Edit] Replicate / {replicate_model}")
        results = _edit_replicate(image, prompt, rep_token, replicate_model, num_variants)
    elif use == "gemini":
        if not gem_key:
            raise ValueError("GEMINI_API_KEY not set")
        print(f"[VLM Edit] Gemini / {gemini_model}")
        results = _edit_gemini(image, prompt, gem_key, gemini_model, num_variants)
    else:
        print("[VLM Edit] InstructPix2Pix (local)")
        results = _edit_ip2p(
            image, prompt, device,
            image_guidance_scale, guidance_scale,
            num_inference_steps, num_variants, seed,
        )

    if contrast_boost != 1.0:
        print(f"[VLM Edit] Contrast boost ×{contrast_boost}")
        results = [_boost_contrast(r, contrast_boost) for r in results]
    return results

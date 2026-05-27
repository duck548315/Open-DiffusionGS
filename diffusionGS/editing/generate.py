"""
Text-to-image and VLM prompt decomposition for the dual-path pipeline.
"""

import io
import os
import urllib.request

from PIL import Image


_DECOMPOSE_FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
    "gemini-2.5-pro",
    "gemini-2.5-pro-preview-06-05",
]


def decompose_prompt(
    edit_instruction: str,
    image=None,
    gemini_api_key: str = None,
    model: str = "gemini-2.5-flash",
) -> tuple[str, str]:
    """
    Decompose a user's edit instruction into (geo_prompt, color_prompt).

    If `image` is provided (PIL.Image), the VLM sees both the image and the
    instruction — used in image mode so the VLM understands what object it's
    looking at before splitting the edit.

    Without an image (text mode), it works on the text description alone.

    Returns
    -------
    geo_prompt   : for image mode → Flux Kontext edit emphasising geometry cues.
                   for text mode  → text-to-image prompt for a geometry-rich photo.
    color_prompt : Flux Kontext edit instruction focusing on material / color.
    """
    gem_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

    if not gem_key:
        if image is not None:
            geo = f"{edit_instruction}, emphasize sharp edges and geometry contrast"
        else:
            geo = (
                f"{edit_instruction}, single object on white background, "
                "high contrast, vivid geometry detail, front view, studio lighting"
            )
        return geo, edit_instruction

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=gem_key)

    # Constraint appended to every generated prompt to keep Flux on track
    OBJECT_CONSTRAINT = (
        "The entire object must be fully visible and not cropped. "
        "Single isolated object centered in frame, plain white background, "
        "front-facing view, product-photography style, no close-up or partial view."
    )

    if image is not None:
        # Image mode — VLM sees the actual object
        system = (
            "You are a 3D reconstruction assistant. "
            "The user wants to edit the object in the image. "
            "Split the edit instruction into two specialized prompts for Flux Kontext (a 2D image editor). "
            "Both prompts MUST preserve the original framing: the complete object stays fully visible, "
            "centered, on a plain background — never zoom in or crop.\n\n"
            "GEO: <edit that sharpens edges, boosts depth contrast and structural detail "
            "for reliable 3D shape extraction. Keep colors vivid but prioritise geometry. "
            f"{OBJECT_CONSTRAINT}>\n"
            "COLOR: <edit that changes only the material, color, and texture as the user asked. "
            f"Do not alter the shape or framing. {OBJECT_CONSTRAINT}>\n\n"
            "Reply with exactly two lines starting with GEO: and COLOR: and nothing else."
        )
        buf = io.BytesIO()
        image.convert("RGB").save(buf, format="PNG")
        contents = [
            types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"),
            f"{system}\n\nEdit instruction: {edit_instruction}",
        ]
    else:
        # Text mode — no image, generate t2i prompt + color edit prompt
        system = (
            "You are a 3D reconstruction assistant. "
            "Given an object description, output exactly two lines:\n"
            "GEO: <text-to-image prompt for a geometry-rich, high-contrast image of the complete object. "
            "Requirements: entire object fully visible and not cropped, centered in frame, "
            "plain white background, front-facing view, product-photography style, "
            "vivid colors, strong depth and edge contrast for 3D reconstruction.>\n"
            "COLOR: <short Flux Kontext edit instruction that applies only the desired "
            "material / color / texture to the base image, without changing the shape or framing. "
            f"{OBJECT_CONSTRAINT}>\n\n"
            "Reply with exactly two lines starting with GEO: and COLOR: and nothing else."
        )
        contents = [f"{system}\n\nObject description: {edit_instruction}"]

    # Build fallback list: requested model first, then the rest (de-duped, in order)
    candidates = [model] + [m for m in _DECOMPOSE_FALLBACK_MODELS if m != model]

    last_err = None
    for candidate in candidates:
        try:
            response = client.models.generate_content(
                model=candidate,
                contents=contents,
            )
            if candidate != model:
                print(f"[VLM decompose] fell back to {candidate}")
            lines = response.text.strip().splitlines()
            geo_line   = next((l for l in lines if l.startswith("GEO:")),   None)
            color_line = next((l for l in lines if l.startswith("COLOR:")), None)
            geo_prompt   = geo_line.removeprefix("GEO:").strip()     if geo_line   else edit_instruction
            color_prompt = color_line.removeprefix("COLOR:").strip() if color_line else edit_instruction
            return geo_prompt, color_prompt
        except Exception as e:
            err = str(e)
            if any(k in err.lower() for k in ("429", "quota", "resource_exhausted",
                                               "overloaded", "503", "unavailable")):
                print(f"[VLM decompose] {candidate} unavailable ({err[:80]}), trying next…")
                last_err = e
            else:
                raise  # non-quota error → don't mask it

    # All models exhausted → fall back to string augmentation
    print(f"[VLM decompose] all Gemini models exhausted, using string fallback. Last error: {last_err}")
    if image is not None:
        geo = f"{edit_instruction}, emphasize sharp edges and geometry contrast"
    else:
        geo = (
            f"{edit_instruction}, single object on white background, "
            "high contrast, vivid geometry detail, front view, studio lighting"
        )
    return geo, edit_instruction


def text_to_image(
    prompt: str,
    backend: str = "auto",
    replicate_token: str = None,
    gemini_api_key: str = None,
) -> Image.Image:
    """Generate a PIL image from a text prompt using Replicate or Gemini."""
    rep_token = replicate_token or os.environ.get("REPLICATE_API_TOKEN")
    gem_key = gemini_api_key or os.environ.get("GEMINI_API_KEY")

    use = backend
    if use == "auto":
        use = "replicate" if rep_token else ("gemini" if gem_key else None)

    if use == "replicate":
        if not rep_token:
            raise ValueError("REPLICATE_API_TOKEN not set")
        import replicate

        client = replicate.Client(api_token=rep_token)
        output = client.run(
            "black-forest-labs/flux-schnell",
            input={"prompt": prompt, "num_outputs": 1},
        )
        if not isinstance(output, (list, tuple)):
            output = [output]
        item = output[0]
        url = item.url if hasattr(item, "url") else str(item)
        with urllib.request.urlopen(url) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")

    elif use == "gemini":
        if not gem_key:
            raise ValueError("GEMINI_API_KEY not set")
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=gem_key)
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
            config=types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"]),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
        raise RuntimeError("Gemini returned no image for the given prompt")

    else:
        raise ValueError(
            f"No text-to-image backend available (backend={backend!r}). "
            "Set REPLICATE_API_TOKEN or GEMINI_API_KEY."
        )

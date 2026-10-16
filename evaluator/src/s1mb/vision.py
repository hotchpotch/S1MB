"""Local vision inputs and explicit, reproducible provider image preparation."""

import hashlib
import io
from pathlib import Path

from .data import Box, Case, ImageAsset, read_json


def load_vision_cases(directory: Path) -> list[Case]:
    rows = read_json(directory / "vision-cases.json")
    cases = []
    for row in rows:
        for image in row.get("images", {}).values():
            path = (directory / image["path"]).resolve()
            if not path.is_relative_to(directory.parent.resolve()):
                raise ValueError("Vision image path escapes dataset storage")
            image["path"] = str(path)
        cases.append(Case.model_validate(row))
    return cases


def prepare_image(asset: ImageAsset) -> tuple[bytes, str, dict]:
    """Fit the full image, never crop; preserve IQA bytes under original policy.

    Small/elongated images get centered white padding after uniform scaling.
    The transform records the content rectangle for inverse Box coordinates.
    """
    from PIL import Image, ImageOps

    original = Path(asset.path).read_bytes()
    if hashlib.sha256(original).hexdigest() != asset.sha256:
        raise ValueError("Image checksum mismatch")
    with Image.open(io.BytesIO(original)) as source:
        if getattr(source, "n_frames", 1) != 1:
            raise ValueError("Only still images are supported")
        upright = ImageOps.exif_transpose(source)
        width, height = upright.size
        if asset.policy == "original":
            if not (280 <= width <= 1568 and 280 <= height <= 1568):
                raise ValueError("Original image exceeds provider limits")
            if source.format not in {"JPEG", "PNG"} or source.getexif().get(274, 1) != 1:
                raise ValueError("Original policy requires upright PNG/JPEG")
            payload, mime = original, "image/jpeg" if source.format == "JPEG" else "image/png"
            content_width, content_height = width, height
            out_width, out_height = width, height
            x = y = 0
        else:
            maximum = 1568 if asset.policy == "detail-1568" else 896
            scale = min(1.0, maximum / max(width, height))
            # Upscale uniformly when both API minimum sides fit in the maximum.
            scale = min(max(scale, 280 / min(width, height)), maximum / max(width, height))
            content_width, content_height = (
                max(1, round(width * scale)),
                max(1, round(height * scale)),
            )
            image = upright.convert("RGB").resize(
                (content_width, content_height), Image.Resampling.LANCZOS
            )
            out_width, out_height = max(280, content_width), max(280, content_height)
            x, y = (out_width - content_width) // 2, (out_height - content_height) // 2
            canvas = Image.new("RGB", (out_width, out_height), "white")
            canvas.paste(image, (x, y))
            buffer = io.BytesIO()
            # Lossless transport avoids adding JPEG artifacts to anomaly/detail tasks.
            canvas.save(buffer, format="PNG")
            payload, mime = buffer.getvalue(), "image/png"
    if len(payload) > 16 * 1024 * 1024:
        raise ValueError("Prepared image exceeds 16 MiB")
    return (
        payload,
        mime,
        {
            "policy": asset.policy,
            "original_sha256": asset.sha256,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "source_size": [width, height],
            "payload_size": [out_width, out_height],
            "content_rectangle": [x, y, content_width, content_height],
            "payload_bytes": len(payload),
        },
    )


def original_box(box: Box, transform: dict) -> Box:
    width, height = transform["payload_size"]
    x, y, cw, ch = transform["content_rectangle"]
    edges = {
        "left": (box.left * width - x) / cw,
        "right": (box.right * width - x) / cw,
        "top": (box.top * height - y) / ch,
        "bottom": (box.bottom * height - y) / ch,
    }
    # A prediction entirely in padding becomes zero-area and scores zero IoU.
    return Box.model_validate({key: min(1.0, max(0.0, value)) for key, value in edges.items()})

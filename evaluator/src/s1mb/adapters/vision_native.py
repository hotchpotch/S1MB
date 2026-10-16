"""Lossless visual decision transport shared by local native runtime bridges."""

import io
import json

from .onejev import onejev_questions


def ordered_images(case):
    """Prepare every declared image without exposing paths or provenance."""
    from PIL import Image

    from s1mb.vision import prepare_image

    images, references, receipts = [], {}, []
    for index, (name, asset) in enumerate(case.images.items(), 1):
        payload, _, receipt = prepare_image(asset)
        with Image.open(io.BytesIO(payload)) as image:
            images.append(image.convert("RGB"))
        references[name] = f"<image:{index}>"
        receipts.append(receipt)
    return images, {"context": case.state, "visual_inputs": references}, receipts


def choice_transport(question):
    """Transport authored binary and numeric criteria through a native choice head."""
    task = question["type"]
    criteria = question["criteria"]
    if task == "choice":
        values = list(criteria.values())
    elif task == "noul":
        values = [criteria["false"], criteria["true"]]
    else:
        values = criteria
    return [v if isinstance(v, str) else json.dumps(v, ensure_ascii=False) for v in values]


def typed_questions(case):
    questions = onejev_questions(case)
    for question in questions.values():
        if not isinstance(question["instructions"], str):
            question["instructions"] = json.dumps(question["instructions"], ensure_ascii=False)
    return questions

"""TypeSafe HTTP integration using the documented v1 typed-question contract."""

import json
import os
import time
from threading import Lock
from uuid import uuid4

import httpx

from s1mb.data import InferenceCase, ModelInfo, Prediction

from .base import decode_answers, questions_for_api


class TypeSafeAdapter:
    supports_images = True
    provider = "typesafe"
    api_key_env = "TYPESAFE_API_KEY"
    base_url = "https://api.typesafe.ai"
    endpoint = "/v1/systemone"
    rounding_decimals = 2

    def __init__(self, model: str):
        key = os.environ.get(self.api_key_env)
        if not key:
            raise ValueError(f"Set {self.api_key_env} to use the {self.provider} adapter")
        self._lock = Lock()
        self.model = model
        self.resolved: str | None = None
        self.image_receipts: dict[str, list[dict]] = {}
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "retries": 0}
        self.client = httpx.Client(
            base_url=self.base_url,
            timeout=120,
            headers={"Authorization": f"Bearer {key}"},
        )

    def predict(self, case: InferenceCase) -> list[Prediction]:
        request = {
            "model": self.model,
            "state": case.state,
            "questions": questions_for_api(case.questions, sort_score=True, structured=True),
        }
        if any(q.task == "score" and len(q.options) > 10 for q in case.questions):
            raise ValueError("TypeSafe Score supports at most 10 levels")
        files = None
        transforms = []
        headers = {}
        if case.images:
            from s1mb.vision import prepare_image

            if not isinstance(case.state, dict):
                raise ValueError("Vision state must be an object")
            assets, image_state, parts = {}, {}, []
            for i, (name, asset) in enumerate(case.images.items()):
                payload, mime, transform = prepare_image(asset)
                reference = str(uuid4())
                part = f"image_{i + 1}"
                assets[reference] = {"type": "image", "part": part}
                image_state[name] = reference
                parts.append(
                    (part, (part + (".png" if mime == "image/png" else ".jpg"), payload, mime))
                )
                transforms.append(transform)
            request["assets"] = assets
            request["state"] = {**case.state, "images": image_state}
            files = [("request", (None, json.dumps(request), "application/json")), *parts]
            headers["TypeSafe-Danger-Zone-Images"] = "true"
            self.image_receipts[case.case_id] = transforms
        elif any(q.task == "box" for q in case.questions):
            raise ValueError("Box questions require an image")
        if any(q.task == "box" for q in case.questions) and len(case.images) != 1:
            raise ValueError("Box currently requires exactly one image")
        for attempt in range(3):
            with self._lock:
                self.usage["requests"] += 1
                self.usage["retries"] += int(attempt > 0)
            try:
                response = self.client.post(
                    self.endpoint,
                    headers=headers,
                    files=files,
                    json=request if files is None else None,
                )
            except httpx.TransportError:
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
                continue
            if response.status_code not in {429, 500, 502, 503, 504, 529} or attempt == 2:
                break
            time.sleep(2**attempt)
        if response.is_error:
            # Do not persist response bodies, which can contain inputs or credentials.
            raise RuntimeError(f"{self.provider} HTTP {response.status_code}")
        payload = response.json()
        with self._lock:
            if self.resolved is not None and self.resolved != payload["model"]:
                raise ValueError("Resolved model changed during the run")
            self.resolved = payload["model"]
            for k in ("input_tokens", "output_tokens"):
                self.usage[k] += payload["usage"][k]
        # Live Jev 1.13 responses use two decimal places; sums can be 0.99 or 1.01.
        predictions = decode_answers(
            case,
            payload["answers"],
            rounded=True,
            rounding_decimals=self.rounding_decimals,
            sort_score=True,
        )

        if transforms:
            from s1mb.vision import original_box

            for pred in predictions:
                if pred.box is not None:
                    pred.box = original_box(pred.box, transforms[0])
        return predictions

    def metadata(self) -> ModelInfo:
        return ModelInfo(
            id=self.model,
            adapter=self.provider,
            revision=self.resolved or self.model,
            settings={
                "requested_model": self.model,
                "degenerate_box_policy": "retain-zero-area-score-zero-iou",
                "vision_transport": "multipart-full-image-v1",
                "score_criteria_order": "ascending-value-v1",
                "renderer": "system-one-structured-v2",
                "input_length_policy": "full-input-no-local-truncation",
                "probability_normalization": (
                    f"sum-normalized-within-{self.rounding_decimals}-decimal-rounding"
                ),
            },
        )

    def close(self) -> None:
        self.client.close()

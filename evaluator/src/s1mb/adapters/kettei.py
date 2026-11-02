"""Sqwish kettei decision models over the System One typed-question contract.

Sqwish serves ``POST /v1/systemone`` with the TypeSafe v1 request and response
shape, so this adapter reuses the TypeSafe request mapping and decoder. Only the
host, the credential and bounded per-case concurrency differ. It never reads
``TYPESAFE_API_KEY`` and never contacts TypeSafe.
"""

from concurrent.futures import ThreadPoolExecutor

from s1mb.data import InferenceCase, ModelInfo, Prediction

from .typesafe import TypeSafeAdapter


class KetteiAdapter(TypeSafeAdapter):
    provider = "kettei"
    api_key_env = "SQWISH_API_KEY"
    base_url = "https://api.sqwish.ai"
    endpoint = "/v1/systemone"
    # The API returns unrounded probabilities; this only bounds float noise in the sum.
    rounding_decimals = 4

    def __init__(self, model: str, case_batch_size: int = 1):
        if not 1 <= case_batch_size <= 32:
            raise ValueError("Kettei case batch size must be between 1 and 32")
        super().__init__(model)
        self.case_batch_size = case_batch_size

    def predict_batch(self, cases: list[InferenceCase]) -> list[list[Prediction] | Exception]:
        def predict_one(case: InferenceCase) -> list[Prediction] | Exception:
            try:
                return self.predict(case)
            except Exception as exc:  # noqa: BLE001 - preserve individual request failures
                return exc

        with ThreadPoolExecutor(max_workers=self.case_batch_size) as pool:
            return list(pool.map(predict_one, cases))

    def metadata(self) -> ModelInfo:
        model = super().metadata()
        model.settings["case_batch_size"] = self.case_batch_size
        model.settings["endpoint"] = self.base_url + self.endpoint
        return model

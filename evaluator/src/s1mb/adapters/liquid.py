"""Liquid decision models using the shared typed-question HTTP contract."""

from concurrent.futures import ThreadPoolExecutor

from s1mb.data import InferenceCase, ModelInfo, Prediction

from .typesafe import TypeSafeAdapter


class LiquidAdapter(TypeSafeAdapter):
    provider = "liquid"
    api_key_env = "LIQUID_API_KEY"
    base_url = "https://api.liquid.ai"
    endpoint = "/decisions/v1/systemone"
    rounding_decimals = 4

    def __init__(self, model: str, case_batch_size: int = 1):
        if not 1 <= case_batch_size <= 32:
            raise ValueError("Liquid case batch size must be between 1 and 32")
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

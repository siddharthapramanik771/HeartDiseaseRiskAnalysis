from dataclasses import dataclass
from typing import Any
from typing import Mapping

import pandas as pd

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.model_bundle import ModelArtifact, ModelArtifactRepository
from src.preprocessing import MISSING_CATEGORY


@dataclass(frozen=True)
class HeartDiseasePrediction:
    risk_probability: float
    label: int
    prediction: str
    risk_band: str
    model_name: str

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "risk_probability": self.risk_probability,
            "label": self.label,
            "prediction": self.prediction,
            "risk_band": self.risk_band,
            "model_name": self.model_name,
        }


class FeaturePayloadBuilder:
    """Aligns arbitrary app input with the training-time feature schema."""

    def __init__(self, artifact: ModelArtifact) -> None:
        self.artifact = artifact

    def prepare(self, payload: Mapping[str, Any]) -> pd.DataFrame:
        if not self.artifact.feature_columns:
            return pd.DataFrame([payload])

        row: dict[str, Any] = {}
        for column in self.artifact.feature_columns:
            value = payload.get(column)
            row[column] = self._coerce_value(column, value)

        return pd.DataFrame([row], columns=self.artifact.feature_columns)

    def _coerce_value(self, column: str, value: Any) -> float | str:
        if column in self.artifact.numeric_defaults:
            return self._coerce_numeric(column, value)

        fallback = self.artifact.categorical_defaults.get(column, MISSING_CATEGORY)
        return str(value).strip() if value not in (None, "") else fallback

    def _coerce_numeric(self, column: str, value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return self.artifact.numeric_defaults[column]


class HeartDiseasePredictor:
    """Application service for loading a saved model artifact and scoring payloads."""

    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        artifact_repository: ModelArtifactRepository | None = None,
    ) -> None:
        self.config = config
        self.artifact_repository = artifact_repository or ModelArtifactRepository(
            config.model_path
        )
        self._artifact: ModelArtifact | None = None

    def predict(self, payload: Mapping[str, Any]) -> HeartDiseasePrediction:
        artifact = self.load_artifact()
        probability = self._predict_probability(payload, artifact)
        label = int(probability >= artifact.prediction_threshold)
        prediction = (
            artifact.positive_target_label if label else artifact.negative_target_label
        )
        return HeartDiseasePrediction(
            risk_probability=probability,
            label=label,
            prediction=prediction,
            risk_band=risk_band_from_probability(probability),
            model_name=artifact.model_name,
        )

    def predict_probability(self, payload: Mapping[str, Any]) -> float:
        artifact = self.load_artifact()
        return self._predict_probability(payload, artifact)

    def _predict_probability(
        self, payload: Mapping[str, Any], artifact: ModelArtifact
    ) -> float:
        features = FeaturePayloadBuilder(artifact).prepare(payload)
        probability = artifact.pipeline.predict_proba(features)[0, 1]
        return float(probability)

    def load_artifact(self) -> ModelArtifact:
        if self._artifact is None:
            self._artifact = self.artifact_repository.load()
        return self._artifact


def risk_band_from_probability(probability: float) -> str:
    if probability < 0.33:
        return "Low"
    if probability < 0.66:
        return "Moderate"
    return "High"


_predictor: HeartDiseasePredictor | None = None


def get_predictor() -> HeartDiseasePredictor:
    global _predictor
    if _predictor is None:
        _predictor = HeartDiseasePredictor()
    return _predictor


def predict_proba_single(payload: Mapping[str, Any]) -> float:
    return get_predictor().predict_probability(payload)

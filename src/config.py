from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class RuntimeConfig:
    """Application settings shared by training, prediction, and the UI."""

    project_root: Path
    data_path: Path
    model_path: Path
    metrics_path: Path
    target_column: str
    id_column: str
    positive_target_label: str
    negative_target_label: str
    prediction_threshold: float
    include_dataset_feature: bool

    @classmethod
    def from_project_root(cls, project_root: Path) -> "RuntimeConfig":
        return cls(
            project_root=project_root,
            data_path=project_root / "data" / "heart_disease_uci.csv",
            model_path=project_root / "models" / "model.joblib",
            metrics_path=project_root / "models" / "training_metrics.json",
            target_column="num",
            id_column="id",
            positive_target_label="Heart disease",
            negative_target_label="No heart disease",
            prediction_threshold=0.65,
            include_dataset_feature=False,
        )

    def load_dataset(self, path: Path | None = None) -> pd.DataFrame:
        dataset_path = path or self.data_path
        suffix = dataset_path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(dataset_path)

        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(dataset_path)

        raise ValueError(f"Unsupported dataset format: {dataset_path}")

    def with_data_path(self, data_path: Path) -> "RuntimeConfig":
        return replace(self, data_path=data_path)

    def with_threshold(self, prediction_threshold: float) -> "RuntimeConfig":
        return replace(self, prediction_threshold=prediction_threshold)

    def with_dataset_feature(self, include_dataset_feature: bool) -> "RuntimeConfig":
        return replace(self, include_dataset_feature=include_dataset_feature)

    def ensure_runtime_dirs(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)


RUNTIME_CONFIG = RuntimeConfig.from_project_root(
    Path(__file__).resolve().parent.parent
)

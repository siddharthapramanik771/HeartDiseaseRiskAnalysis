import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.model_bundle import ModelArtifact, ModelArtifactRepository
from src.preprocessing import (
    BINARY_TARGET,
    DataPreprocessor,
    FeatureDefaults,
    FeatureSchema,
)
from src.training_settings import TrainingSettings


@dataclass(frozen=True)
class HoldoutMetrics:
    test_accuracy: float
    test_precision: float
    test_recall: float
    test_f1: float
    test_roc_auc: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, float | list[list[int]]]:
        return {
            "test_accuracy": self.test_accuracy,
            "test_precision": self.test_precision,
            "test_recall": self.test_recall,
            "test_f1": self.test_f1,
            "test_roc_auc": self.test_roc_auc,
            "confusion_matrix": self.confusion_matrix,
        }


class HeartDiseaseModelTrainer:
    """Coordinates the offline heart disease model training workflow."""

    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        preprocessor: DataPreprocessor | None = None,
        settings: TrainingSettings | None = None,
        artifact_repository: ModelArtifactRepository | None = None,
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor or DataPreprocessor.from_config(config)
        self.settings = settings or TrainingSettings()
        self.artifact_repository = artifact_repository or ModelArtifactRepository(
            config.model_path
        )

    def train(self, selected_model: str = "best") -> dict[str, Any]:
        self.config.ensure_runtime_dirs()
        cleaned = self.load_training_frame()
        X, y = self.build_training_matrix(cleaned)
        X_train, X_test, y_train, y_test = self.split_training_data(X, y)
        pipelines, schema = self.create_candidate_pipelines(X_train)
        feature_defaults = self.preprocessor.derive_feature_defaults(X_train)

        results: dict[str, dict[str, Any]] = {}
        for model_name, pipeline in pipelines.items():
            pipeline.fit(X_train, y_train)
            probabilities = pipeline.predict_proba(X_test)[:, 1]
            predictions = (probabilities >= self.config.prediction_threshold).astype(int)
            metrics = self.evaluate(y_test, probabilities, predictions)
            results[model_name] = {
                "pipeline": pipeline,
                "metrics": metrics,
            }

        chosen_model_name = self.choose_model(results, selected_model)
        artifact = self.build_artifact(
            pipeline=results[chosen_model_name]["pipeline"],
            model_name=chosen_model_name,
            feature_columns=X_train.columns.tolist(),
            feature_defaults=feature_defaults,
        )
        self.artifact_repository.save(artifact)
        self.save_metrics_artifact(
            selected_model_name=chosen_model_name,
            results=results,
            schema=schema,
            train_rows=len(X_train),
            test_rows=len(X_test),
        )
        self.print_summary(chosen_model_name, results)
        return results

    def load_training_frame(self) -> pd.DataFrame:
        raw_df = self.config.load_dataset()
        cleaned = self.preprocessor.clean(raw_df)
        return cleaned.dropna(subset=[self.config.target_column]).copy()

    def build_training_matrix(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = self.preprocessor.get_feature_frame(df)
        y = self.preprocessor.encode_target(df)
        return X, y

    def split_training_data(
        self, X: pd.DataFrame, y: pd.Series
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        return train_test_split(
            X,
            y,
            test_size=self.settings.test_size,
            random_state=self.settings.random_state,
            stratify=y,
        )

    def create_candidate_pipelines(
        self, X_train: pd.DataFrame
    ) -> tuple[dict[str, Pipeline], FeatureSchema]:
        transformer, schema = self.preprocessor.build_transformer(X_train)
        random_forest = Pipeline(
            [
                ("preprocessor", transformer),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=self.settings.random_forest_estimators,
                        max_depth=self.settings.random_forest_max_depth,
                        min_samples_leaf=self.settings.random_forest_min_samples_leaf,
                        class_weight="balanced",
                        random_state=self.settings.random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        )

        knn_transformer, _ = self.preprocessor.build_transformer(X_train)
        knn = Pipeline(
            [
                ("preprocessor", knn_transformer),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=self.settings.knn_neighbors,
                        weights="distance",
                    ),
                ),
            ]
        )
        return {"Random Forest": random_forest, "KNN": knn}, schema

    def evaluate(
        self,
        y_test: pd.Series,
        probabilities: np.ndarray,
        predictions: np.ndarray,
    ) -> HoldoutMetrics:
        return HoldoutMetrics(
            test_accuracy=accuracy_score(y_test, predictions),
            test_precision=precision_score(y_test, predictions, zero_division=0),
            test_recall=recall_score(y_test, predictions, zero_division=0),
            test_f1=f1_score(y_test, predictions, zero_division=0),
            test_roc_auc=roc_auc_score(y_test, probabilities),
            confusion_matrix=confusion_matrix(y_test, predictions).tolist(),
        )

    def choose_model(self, results: dict[str, dict[str, Any]], selected_model: str) -> str:
        normalized = selected_model.strip().lower().replace("_", " ")
        if normalized in {"random forest", "forest", "rf"}:
            return "Random Forest"
        if normalized in {"knn", "k nearest neighbors", "k-nearest neighbors"}:
            return "KNN"

        metric_name = self.settings.selection_metric
        return max(
            results,
            key=lambda name: getattr(results[name]["metrics"], metric_name),
        )

    def build_artifact(
        self,
        pipeline: Pipeline,
        model_name: str,
        feature_columns: list[str],
        feature_defaults: FeatureDefaults,
    ) -> ModelArtifact:
        return ModelArtifact(
            pipeline=pipeline,
            model_name=model_name,
            feature_columns=feature_columns,
            numeric_defaults=feature_defaults.numeric_defaults,
            categorical_defaults=feature_defaults.categorical_defaults,
            target_column=self.config.target_column,
            positive_target_label=self.config.positive_target_label,
            negative_target_label=self.config.negative_target_label,
            prediction_threshold=self.config.prediction_threshold,
        )

    def save_metrics_artifact(
        self,
        selected_model_name: str,
        results: dict[str, dict[str, Any]],
        schema: FeatureSchema,
        train_rows: int,
        test_rows: int,
    ) -> None:
        payload = {
            "selected_model": selected_model_name,
            "model_metrics": {
                name: result["metrics"].to_dict() for name, result in results.items()
            },
            "training_data_path": self.relative_project_path(self.config.data_path),
            "training_data_name": self.config.data_path.name,
            "target_column": self.config.target_column,
            "id_column": self.config.id_column,
            "binary_target": BINARY_TARGET,
            "positive_target_label": self.config.positive_target_label,
            "negative_target_label": self.config.negative_target_label,
            "prediction_threshold": self.config.prediction_threshold,
            "include_dataset_feature": self.config.include_dataset_feature,
            "train_rows": train_rows,
            "test_rows": test_rows,
            "numeric_columns": schema.numeric_columns,
            "categorical_columns": schema.categorical_columns,
            "settings": {
                **self.settings.to_dict(),
            },
        }
        self.config.metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def relative_project_path(self, path: Path) -> str:
        resolved_path = path.resolve()
        try:
            return resolved_path.relative_to(self.config.project_root.resolve()).as_posix()
        except ValueError:
            return str(resolved_path)

    def print_summary(self, selected_model_name: str, results: dict[str, dict[str, Any]]) -> None:
        print("Saved model to", self.artifact_repository.model_path)
        print("Saved metrics to", self.config.metrics_path)
        print("Selected model:", selected_model_name)
        for model_name, result in results.items():
            metrics = result["metrics"]
            print(
                model_name,
                f"F1={metrics.test_f1:.3f}",
                f"ROC_AUC={metrics.test_roc_auc:.3f}",
                f"Accuracy={metrics.test_accuracy:.3f}",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train heart disease risk models.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to a CSV/XLS/XLSX training dataset.",
    )
    parser.add_argument(
        "--model",
        choices=["best", "random_forest", "knn"],
        default="best",
        help="Which fitted model to save. Defaults to the best F1 model.",
    )
    parser.add_argument(
        "--include-dataset-feature",
        action="store_true",
        help="Use the UCI source dataset column as a model feature.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=RUNTIME_CONFIG.prediction_threshold,
        help="Probability threshold used to convert risk probability into a class.",
    )
    return parser.parse_args()


def resolve_data_path(data_path: Path | None, config: RuntimeConfig) -> Path | None:
    if data_path is None:
        return None
    if data_path.is_absolute():
        return data_path
    return config.project_root / data_path


def main() -> None:
    args = parse_args()
    data_path = resolve_data_path(args.data, RUNTIME_CONFIG)
    config = RUNTIME_CONFIG
    if data_path:
        config = config.with_data_path(data_path)
    config = config.with_dataset_feature(args.include_dataset_feature)
    config = config.with_threshold(args.threshold)

    trainer = HeartDiseaseModelTrainer(config=config)
    trainer.train(selected_model=args.model)


if __name__ == "__main__":
    main()

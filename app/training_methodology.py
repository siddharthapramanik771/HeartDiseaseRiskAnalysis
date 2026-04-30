from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.training_settings import TrainingSettings


@dataclass(frozen=True)
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: list[list[int]]


class TrainingMethodologyRenderer:
    """Renders the training workflow from config, settings, and saved metrics."""

    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        settings: TrainingSettings | None = None,
    ) -> None:
        self.config = config
        self.settings = settings or TrainingSettings()

    def render(self) -> None:
        payload = self.safe_load_metrics_payload()
        target_column = self.payload_value(payload, "target_column", self.config.target_column)
        st.header("Training Methodology")
        st.write(
            "The project trains a supervised binary classifier from the UCI severity "
            f"target. The original `{target_column}` value is preserved for analysis, while the "
            "model target is derived dynamically during preprocessing."
        )

        if payload:
            st.caption(
                f"Loaded training report from `{self.relative_project_path(self.config.metrics_path)}`."
            )
        else:
            st.warning(
                "Saved training metrics are not available yet. Run `python -m src.train` "
                "to generate the dynamic report artifact."
            )

        self.render_workflow(payload)
        self.render_pipeline(payload)
        self.render_model_structure(payload)
        self.render_model_details(payload)
        self.render_validation(payload)
        self.render_metric_interpretation(payload)
        self.render_saved_metrics(payload)
        self.render_artifact_contract(payload)
        self.render_github_automation()

    def render_workflow(self, payload: dict[str, Any] | None) -> None:
        st.subheader("Workflow")
        model_names = ", ".join(self.model_names(payload))
        data_name = self.payload_value(payload, "training_data_name", self.config.data_path.name)
        target_column = self.payload_value(payload, "target_column", self.config.target_column)
        id_column = self.payload_value(payload, "id_column", self.config.id_column)
        binary_target = self.payload_value(payload, "binary_target", "derived binary target")

        steps = [
            f"Load the dataset from `{data_name}`.",
            "Normalize column names, strings, missing tokens, numeric fields, and booleans.",
            f"Remove the identifier column `{id_column}` before training.",
            f"Create `{binary_target}` from `{target_column}`.",
            "Split data into stratified training and holdout test sets.",
            "Fit preprocessing and each classifier inside a scikit-learn pipeline.",
            f"Compare candidate models from the training artifact: {model_names}.",
            f"Save the selected fitted pipeline to `{self.relative_project_path(self.config.model_path)}`.",
        ]

        for index, step in enumerate(steps, start=1):
            st.write(f"{index}. {step}")

    def render_pipeline(self, payload: dict[str, Any] | None) -> None:
        st.subheader("Model Pipeline")
        numeric_columns = self.payload_value(payload, "numeric_columns", [])
        categorical_columns = self.payload_value(payload, "categorical_columns", [])
        model_names = self.model_names(payload)

        rows = [
            {
                "Stage": "Cleaning",
                "Method": "DataPreprocessor.clean",
                "Purpose": "Normalize raw values and derive risk labels",
            },
            {
                "Stage": "Numeric features",
                "Method": "SimpleImputer + StandardScaler",
                "Purpose": f"Prepare {len(numeric_columns)} numeric feature(s)",
            },
            {
                "Stage": "Categorical features",
                "Method": "SimpleImputer + OneHotEncoder",
                "Purpose": f"Prepare {len(categorical_columns)} categorical feature(s)",
            },
            {
                "Stage": "Classifiers",
                "Method": ", ".join(model_names),
                "Purpose": "Compare configured supervised classifiers on the same split",
            },
            {
                "Stage": "Artifact",
                "Method": "joblib bundle",
                "Purpose": "Persist the selected fitted pipeline and feature defaults",
            },
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        include_dataset = bool(
            self.payload_value(payload, "include_dataset_feature", self.config.include_dataset_feature)
        )
        dataset_note = "included" if include_dataset else "excluded"
        st.info(
            f"The source `dataset` column is currently {dataset_note} as a model "
            "feature. Excluding it helps reduce collection-site bias."
        )

    def render_model_structure(self, payload: dict[str, Any] | None) -> None:
        st.subheader("Model Structure")
        numeric_columns = self.payload_value(payload, "numeric_columns", [])
        categorical_columns = self.payload_value(payload, "categorical_columns", [])
        selected_model = self.payload_value(payload, "selected_model", "Selected classifier")
        threshold = float(
            self.payload_value(payload, "prediction_threshold", self.config.prediction_threshold)
        )

        st.graphviz_chart(
            self.model_structure_dot(str(selected_model)),
            width="stretch",
        )

        rows = [
            {
                "Layer": "Artifact wrapper",
                "Component": "ModelArtifact",
                "Structure detail": "Stores the fitted pipeline, schema, feature defaults, labels, threshold, and version",
            },
            {
                "Layer": "Pipeline step 1",
                "Component": "ColumnTransformer",
                "Structure detail": "Splits raw features into numeric and categorical preprocessing branches",
            },
            {
                "Layer": "Numeric branch",
                "Component": "SimpleImputer(strategy='median') + StandardScaler",
                "Structure detail": self.feature_summary(numeric_columns),
            },
            {
                "Layer": "Categorical branch",
                "Component": "SimpleImputer(fill_value='Unknown') + OneHotEncoder(handle_unknown='ignore')",
                "Structure detail": self.feature_summary(categorical_columns),
            },
            {
                "Layer": "Pipeline step 2",
                "Component": str(selected_model),
                "Structure detail": "Final classifier selected by holdout F1 unless `--model` forces a candidate",
            },
            {
                "Layer": "Prediction output",
                "Component": f"Probability threshold >= {threshold:.2f}",
                "Structure detail": "Converts disease probability into the dashboard class label",
            },
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.write(
            "At inference time the dashboard sends one feature row through this saved "
            "pipeline. The app does not refit preprocessing or train a model during "
            "interactive prediction."
        )

    def render_model_details(self, payload: dict[str, Any] | None) -> None:
        st.subheader("Model Details")
        settings = self.resolve_settings(payload)
        rows = []

        for model_name in self.model_names(payload):
            if model_name == "KNN":
                rows.append(
                    {
                        "Model": model_name,
                        "Role": "Similarity-based classifier",
                        "Dynamic settings": f"neighbors={settings.get('knn_neighbors', 'n/a')}, weights=distance",
                    }
                )
            elif model_name == "Random Forest":
                rows.append(
                    {
                        "Model": model_name,
                        "Role": "Tree ensemble classifier",
                        "Dynamic settings": (
                            f"trees={settings.get('random_forest_estimators', 'n/a')}, "
                            f"max_depth={settings.get('random_forest_max_depth', 'n/a')}, "
                            f"min_samples_leaf={settings.get('random_forest_min_samples_leaf', 'n/a')}, "
                            "class_weight=balanced"
                        ),
                    }
                )
            else:
                rows.append(
                    {
                        "Model": model_name,
                        "Role": "Configured classifier",
                        "Dynamic settings": "Loaded from training configuration",
                    }
                )

        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.write(
            "KNN benefits from scaling because distance calculations are sensitive to "
            "feature ranges. Random Forest is less sensitive to scale and helps capture "
            "non-linear clinical feature interactions."
        )

    def render_validation(self, payload: dict[str, Any] | None) -> None:
        st.subheader("Validation Strategy")
        settings = self.resolve_settings(payload)
        threshold = float(
            self.payload_value(payload, "prediction_threshold", self.config.prediction_threshold)
        )
        train_rows = self.payload_value(payload, "train_rows", "n/a")
        test_rows = self.payload_value(payload, "test_rows", "n/a")

        columns = st.columns(5)
        columns[0].metric("Train rows", f"{train_rows}")
        columns[1].metric("Test rows", f"{test_rows}")
        columns[2].metric("Test split", f"{float(settings.get('test_size', 0)):.0%}")
        columns[3].metric("Random seed", str(settings.get("random_state", "n/a")))
        columns[4].metric("Risk threshold", f"{threshold:.2f}")

        st.write(
            "The split is stratified to preserve the disease/no-disease ratio. "
            f"The selection metric is `{settings.get('selection_metric', 'n/a')}`."
        )

    def render_metric_interpretation(self, payload: dict[str, Any] | None) -> None:
        st.subheader("Metric Interpretation")
        threshold = float(
            self.payload_value(payload, "prediction_threshold", self.config.prediction_threshold)
        )
        rows = [
            {
                "Metric": "Accuracy",
                "Question answered": "How often is the final class correct overall?",
                "Why it matters": "Useful headline metric, but can hide class-specific errors",
            },
            {
                "Metric": "Precision",
                "Question answered": "When risk is predicted, how often is it correct?",
                "Why it matters": "Helps control false positive risk alerts",
            },
            {
                "Metric": "Recall",
                "Question answered": "Out of true disease cases, how many are found?",
                "Why it matters": "Important when missing a risk case is costly",
            },
            {
                "Metric": "F1",
                "Question answered": "How balanced are precision and recall?",
                "Why it matters": "Used for default model selection",
            },
            {
                "Metric": "ROC AUC",
                "Question answered": "How well do probabilities rank disease above no disease?",
                "Why it matters": "Judges probability quality across thresholds",
            },
            {
                "Metric": "Confusion matrix",
                "Question answered": "Which error type is the model making?",
                "Why it matters": "Shows true/false positives and negatives",
            },
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.info(
            f"The current artifact threshold is {threshold:.2f}. Lowering it usually "
            "improves recall but creates more false positives. Raising it usually "
            "improves precision but can miss more disease-positive cases."
        )

    def render_saved_metrics(self, payload: dict[str, Any] | None) -> None:
        st.subheader("Current Test Metrics")
        if not payload:
            st.warning("Run `python -m src.train` to create the dynamic metrics artifact.")
            return

        metrics_by_model = self.parse_model_metrics(payload)
        selected_model = payload.get("selected_model", next(iter(metrics_by_model)))
        selected_metrics = metrics_by_model[selected_model]

        metric_columns = st.columns(5)
        metric_columns[0].metric("Accuracy", f"{selected_metrics.accuracy:.2%}")
        metric_columns[1].metric("ROC AUC", f"{selected_metrics.roc_auc:.3f}")
        metric_columns[2].metric("Precision", f"{selected_metrics.precision:.2%}")
        metric_columns[3].metric("Recall", f"{selected_metrics.recall:.2%}")
        metric_columns[4].metric("F1 score", f"{selected_metrics.f1:.3f}")

        left, right = st.columns(2)
        left.plotly_chart(
            self.model_comparison_chart(metrics_by_model),
            width="stretch",
        )
        right.plotly_chart(
            self.confusion_matrix_chart(selected_metrics, payload),
            width="stretch",
        )

        with st.expander("Saved training metadata", expanded=False):
            st.json(payload)

    def render_artifact_contract(self, payload: dict[str, Any] | None) -> None:
        st.subheader("Saved Model Artifact")
        rows = [
            {"Artifact Field": "Model path", "Value": self.relative_project_path(self.config.model_path)},
            {"Artifact Field": "Metrics path", "Value": self.relative_project_path(self.config.metrics_path)},
            {
                "Artifact Field": "Selected model",
                "Value": self.payload_value(payload, "selected_model", "not trained"),
            },
            {
                "Artifact Field": "Target column",
                "Value": self.payload_value(payload, "target_column", self.config.target_column),
            },
            {
                "Artifact Field": "Positive label",
                "Value": self.payload_value(payload, "positive_target_label", self.config.positive_target_label),
            },
            {
                "Artifact Field": "Negative label",
                "Value": self.payload_value(payload, "negative_target_label", self.config.negative_target_label),
            },
            {
                "Artifact Field": "Prediction threshold",
                "Value": self.payload_value(payload, "prediction_threshold", self.config.prediction_threshold),
            },
        ]
        artifact_df = pd.DataFrame(rows)
        artifact_df["Value"] = artifact_df["Value"].astype(str)
        st.dataframe(artifact_df, width="stretch", hide_index=True)
        st.write(
            "Prediction uses the saved model bundle, so the dashboard does not retrain "
            "while users interact with it."
        )

    def render_github_automation(self) -> None:
        st.subheader("GitHub Training Automation")
        workflow_path = Path(".github") / "workflows" / "ci.yml"
        artifact_paths = [
            self.relative_project_path(self.config.model_path),
            self.relative_project_path(self.config.metrics_path),
        ]
        st.write(
            "The repository includes a GitHub Actions workflow that retrains the model "
            "when training-relevant files change on `main`."
        )
        rows = [
            {"Step": 1, "Action": "Check out the repository"},
            {"Step": 2, "Action": "Install Python and training dependencies"},
            {"Step": 3, "Action": "Compile tracked Python files"},
            {"Step": 4, "Action": "Run `python -m src.train`"},
            {"Step": 5, "Action": f"Commit updated artifacts: {', '.join(artifact_paths)}"},
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(f"Workflow file: `{workflow_path.as_posix()}`")

    def safe_load_metrics_payload(self) -> dict[str, Any] | None:
        try:
            return self.load_metrics_payload()
        except (FileNotFoundError, json.JSONDecodeError, KeyError, OSError):
            return None

    def load_metrics_payload(self) -> dict[str, Any]:
        if not self.config.metrics_path.exists():
            raise FileNotFoundError(f"Metrics artifact not found at {self.config.metrics_path}")
        return json.loads(self.config.metrics_path.read_text(encoding="utf-8"))

    def resolve_settings(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        if payload and isinstance(payload.get("settings"), dict):
            return payload["settings"]
        return self.settings.to_dict()

    def model_names(self, payload: dict[str, Any] | None) -> list[str]:
        if payload and isinstance(payload.get("model_metrics"), dict):
            return list(payload["model_metrics"].keys())
        return list(self.settings.candidate_models)

    def relative_project_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.config.project_root.resolve()).as_posix()
        except ValueError:
            return str(path)

    @staticmethod
    def feature_summary(columns: list[str]) -> str:
        if not columns:
            return "Loaded from the metrics artifact after training"
        return f"{len(columns)} feature(s): {', '.join(columns)}"

    @classmethod
    def model_structure_dot(cls, selected_model: str) -> str:
        selected_label = cls.escape_dot_label(selected_model)
        return f"""
digraph model_structure {{
    graph [rankdir=LR, bgcolor="transparent", pad="0.2", nodesep="0.5", ranksep="0.7"];
    node [shape=box, style="rounded,filled", fontname="Arial", fontsize=11, color="#0f172a", fillcolor="#f8fafc", fontcolor="#0f172a"];
    edge [color="#475569", arrowsize=0.8, fontname="Arial", fontsize=10];

    raw [label="Raw UCI rows"];
    clean [label="DataPreprocessor.clean"];
    features [label="Feature matrix"];
    target [label="Binary target\\nnum > 0", fillcolor="#ecfeff"];
    split [label="Stratified\\ntrain/test split"];
    transformer [label="ColumnTransformer", fillcolor="#eef2ff"];
    numeric [label="Numeric branch\\nmedian imputer + scaler"];
    categorical [label="Categorical branch\\nUnknown imputer + one-hot"];
    transformed [label="Transformed\\nfeature space"];
    candidate [shape=diamond, label="Candidate\\nclassifier", fillcolor="#fef3c7"];
    rf [label="Random Forest"];
    knn [label="KNN"];
    metrics [label="Holdout metrics"];
    selected [label="Selected: {selected_label}", fillcolor="#dcfce7"];
    artifact [label="ModelArtifact\\njoblib bundle", fillcolor="#fee2e2"];
    app [label="Streamlit\\nprediction"];

    raw -> clean;
    clean -> features;
    clean -> target;
    features -> split;
    target -> split;
    split -> transformer;
    transformer -> numeric;
    transformer -> categorical;
    numeric -> transformed;
    categorical -> transformed;
    transformed -> candidate;
    candidate -> rf;
    candidate -> knn;
    rf -> metrics;
    knn -> metrics;
    metrics -> selected;
    selected -> artifact;
    artifact -> app;
}}
"""

    @staticmethod
    def escape_dot_label(label: str) -> str:
        return label.replace("\\", "\\\\").replace('"', '\\"')

    @staticmethod
    def payload_value(payload: dict[str, Any] | None, key: str, fallback: Any) -> Any:
        if payload is None:
            return fallback
        value = payload.get(key, fallback)
        return fallback if value is None else value

    @staticmethod
    def parse_model_metrics(payload: dict[str, Any]) -> dict[str, ModelMetrics]:
        parsed: dict[str, ModelMetrics] = {}
        for model_name, metrics in payload["model_metrics"].items():
            parsed[model_name] = ModelMetrics(
                accuracy=metrics["test_accuracy"],
                precision=metrics["test_precision"],
                recall=metrics["test_recall"],
                f1=metrics["test_f1"],
                roc_auc=metrics["test_roc_auc"],
                confusion=metrics["confusion_matrix"],
            )
        return parsed

    @staticmethod
    def model_comparison_chart(metrics_by_model: dict[str, ModelMetrics]) -> go.Figure:
        rows = []
        for model_name, metrics in metrics_by_model.items():
            rows.extend(
                [
                    {"Model": model_name, "Metric": "Accuracy", "Score": metrics.accuracy},
                    {"Model": model_name, "Metric": "Precision", "Score": metrics.precision},
                    {"Model": model_name, "Metric": "Recall", "Score": metrics.recall},
                    {"Model": model_name, "Metric": "F1", "Score": metrics.f1},
                    {"Model": model_name, "Metric": "ROC AUC", "Score": metrics.roc_auc},
                ]
            )
        metric_df = pd.DataFrame(rows)
        fig = px.bar(
            metric_df,
            x="Metric",
            y="Score",
            color="Model",
            barmode="group",
            text_auto=".2f",
            title="Model comparison",
            color_discrete_sequence=["#0f766e", "#be123c", "#2563eb", "#b45309"],
        )
        fig.update_layout(yaxis_range=[0, 1], yaxis_title="Score")
        return fig

    def confusion_matrix_chart(
        self, metrics: ModelMetrics, payload: dict[str, Any]
    ) -> go.Figure:
        labels = [
            payload.get("negative_target_label", self.config.negative_target_label),
            payload.get("positive_target_label", self.config.positive_target_label),
        ]
        fig = px.imshow(
            metrics.confusion,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale=["#ecfeff", "#155e75"],
            labels={"x": "Predicted", "y": "Actual", "color": "Records"},
            title="Selected model confusion matrix",
        )
        return fig

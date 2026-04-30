from dataclasses import dataclass
import json
from pathlib import Path
import time

import pandas as pd
import streamlit as st
from pandas.api.types import is_integer_dtype, is_numeric_dtype

from app.data_analysis import DataAnalysisRenderer
from app.styles import GITHUB_REPOSITORY_URL, apply_page_styles
from app.training_methodology import TrainingMethodologyRenderer
from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.model_bundle import ModelArtifactRepository
from src.predict import HeartDiseasePrediction, HeartDiseasePredictor
from src.preprocessing import BINARY_TARGET, DataPreprocessor, display_name


@dataclass(frozen=True)
class ReferenceDataset:
    frame: pd.DataFrame
    source_path: Path


class ReferenceDataService:
    def __init__(
        self,
        config: RuntimeConfig = RUNTIME_CONFIG,
        preprocessor: DataPreprocessor | None = None,
    ) -> None:
        self.config = config
        self.preprocessor = preprocessor or DataPreprocessor.from_config(config)

    def load(self) -> ReferenceDataset | None:
        data_path = self.resolve_reference_data_path()
        if not data_path.exists():
            return None
        raw_df = self.config.load_dataset(data_path)
        return ReferenceDataset(self.preprocessor.clean(raw_df), data_path)

    def resolve_reference_data_path(self) -> Path:
        metrics_data_path = self.read_metrics_data_path()
        if metrics_data_path and metrics_data_path.exists():
            return metrics_data_path
        return self.config.data_path

    def read_metrics_data_path(self) -> Path | None:
        if not self.config.metrics_path.exists():
            return None

        try:
            payload = json.loads(self.config.metrics_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

        training_data_path = payload.get("training_data_path")
        if not training_data_path:
            return None

        path = Path(training_data_path)
        if path.is_absolute():
            return path
        return self.config.project_root / path


class LocalPredictionService:
    def __init__(
        self,
        predictor: HeartDiseasePredictor | None = None,
        config: RuntimeConfig = RUNTIME_CONFIG,
    ) -> None:
        self.predictor = predictor or HeartDiseasePredictor(config)

    def predict(self, payload: dict) -> HeartDiseasePrediction:
        return self.predictor.predict(payload)


class DashboardRenderer:
    def __init__(
        self,
        reference_data_service: ReferenceDataService | None = None,
        prediction_service: LocalPredictionService | None = None,
        config: RuntimeConfig = RUNTIME_CONFIG,
    ) -> None:
        self.config = config
        self.preprocessor = DataPreprocessor.from_config(config)
        self.reference_data_service = reference_data_service or ReferenceDataService(
            config=config,
            preprocessor=self.preprocessor,
        )
        self.prediction_service = prediction_service or LocalPredictionService(
            config=config
        )
        self.data_analysis_renderer = DataAnalysisRenderer(config)
        self.training_methodology_renderer = TrainingMethodologyRenderer(config)

    def render(self) -> None:
        st.set_page_config(
            page_title="Heart Disease Risk Dashboard",
            page_icon="HD",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        apply_page_styles()
        self.render_hero()

        reference_dataset = self.reference_data_service.load()
        if reference_dataset is None:
            self.render_sidebar()
            st.error(
                f"Dataset not found at {self.config.data_path}. "
                "Add the UCI CSV to enable the dashboard."
            )
            st.stop()

        df = reference_dataset.frame
        feature_df = self.resolve_prediction_features(df)
        self.render_sidebar()
        self.render_status_strip(df, feature_df, reference_dataset.source_path)

        prediction_tab, analysis_tab, methodology_tab = st.tabs(
            ["Predict Risk", "Data Analysis", "Training Methodology"]
        )

        with prediction_tab:
            self.render_prediction_tab(feature_df)
        with analysis_tab:
            self.data_analysis_renderer.render(df, feature_df)
        with methodology_tab:
            self.training_methodology_renderer.render()

    def render_sidebar(self) -> None:
        with st.sidebar:
            st.markdown("### Heart Disease ML")
            st.markdown(
                "Interactive scoring, dataset analysis, and training notes for "
                "the UCI heart disease risk model."
            )
            st.link_button("View GitHub repository", GITHUB_REPOSITORY_URL)
            st.divider()
            st.markdown("### Model Artifact")

            if not self.config.model_path.exists():
                st.info("Train the model to enable prediction and artifact download.")
                st.code("python -m src.train", language="bash")
                return

            artifact = ModelArtifactRepository(self.config.model_path).load()
            st.caption(f"Loaded model: {artifact.model_name}")
            st.caption(f"Prediction threshold: {artifact.prediction_threshold:.2f}")
            st.download_button(
                "Download model",
                data=self.config.model_path.read_bytes(),
                file_name=self.config.model_path.name,
                mime="application/octet-stream",
                width="stretch",
            )

            if self.config.metrics_path.exists():
                st.download_button(
                    "Download metrics",
                    data=self.config.metrics_path.read_bytes(),
                    file_name=self.config.metrics_path.name,
                    mime="application/json",
                    width="stretch",
                )

    @staticmethod
    def render_hero() -> None:
        st.markdown(
            f"""
<section class="hero">
    <div class="hero__eyebrow">Machine learning dashboard</div>
    <h1>Heart Disease Risk Analysis</h1>
    <p>
        Explore clinical risk patterns, compare KNN and Random Forest classifiers,
        review the training workflow, and score heart disease risk from a saved
        local model artifact.
    </p>
    <div class="hero__actions">
        <a class="hero__link" href="{GITHUB_REPOSITORY_URL}" target="_blank" rel="noopener">
            GitHub repository
        </a>
        <span class="hero__note">UCI heart disease dataset with interpretable ML workflow</span>
    </div>
</section>
""",
            unsafe_allow_html=True,
        )

    def render_status_strip(
        self, df: pd.DataFrame, feature_df: pd.DataFrame, source_path: Path
    ) -> None:
        risk_rate = "n/a"
        if BINARY_TARGET in df.columns:
            risk_rate = f"{df[BINARY_TARGET].dropna().mean():.1%}"

        model_status = "Not trained"
        model_note = "Run python -m src.train"
        if self.config.model_path.exists():
            artifact = ModelArtifactRepository(self.config.model_path).load()
            model_status = artifact.model_name
            model_note = "Saved artifact ready"

        st.markdown(
            f"""
<div class="status-strip">
    <div class="status-tile">
        <span>Clean rows</span>
        <strong>{len(df):,}</strong>
        <small>Ready for exploration</small>
    </div>
    <div class="status-tile">
        <span>Input features</span>
        <strong>{feature_df.shape[1]:,}</strong>
        <small>Used by prediction form</small>
    </div>
    <div class="status-tile">
        <span>Disease rate</span>
        <strong>{risk_rate}</strong>
        <small>Binary target from num &gt; 0</small>
    </div>
    <div class="status-tile">
        <span>Model</span>
        <strong>{model_status}</strong>
        <small>{model_note}</small>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.caption(f"Reference data source: `{source_path.as_posix()}`")

    def resolve_prediction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_df = self.preprocessor.get_feature_frame(df)
        if not self.config.model_path.exists():
            return feature_df

        try:
            artifact = ModelArtifactRepository(self.config.model_path).load()
        except Exception:
            return feature_df

        available_columns = [
            column for column in artifact.feature_columns if column in feature_df.columns
        ]
        return feature_df[available_columns] if available_columns else feature_df

    def render_prediction_tab(self, feature_df: pd.DataFrame) -> None:
        st.subheader("Predict Heart Disease Risk")
        st.caption("Enter a patient profile and run the saved pipeline locally.")

        if not self.config.model_path.exists():
            st.warning(
                "Prediction requires a trained artifact. Run `python -m src.train` "
                "from the project root, then refresh the dashboard."
            )
            return

        with st.form("prediction_form"):
            payload = self.render_feature_inputs(feature_df)
            submitted = st.form_submit_button("Predict risk")

        if submitted:
            try:
                start = time.time()
                result = self.prediction_service.predict(payload)
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
            else:
                self.render_prediction_result(result, time.time() - start)

    def render_feature_inputs(self, feature_df: pd.DataFrame) -> dict:
        payload = {}
        columns = st.columns(3)

        for index, column_name in enumerate(feature_df.columns):
            series = feature_df[column_name]
            panel = columns[index % 3]

            with panel:
                if is_numeric_dtype(series):
                    payload[column_name] = self.render_numeric_input(column_name, series)
                else:
                    payload[column_name] = self.render_categorical_input(
                        column_name, series
                    )

        return payload

    @staticmethod
    def render_numeric_input(column_name: str, series: pd.Series) -> int | float:
        clean = series.dropna()
        if clean.empty:
            return float(st.number_input(display_name(column_name), value=0.0))

        median_value = clean.median()
        min_value = clean.min()
        max_value = clean.max()

        if is_integer_dtype(clean):
            return int(
                st.number_input(
                    display_name(column_name),
                    min_value=int(min_value),
                    max_value=int(max_value),
                    value=int(round(median_value)),
                    step=1,
                )
            )

        step = 0.1 if column_name == "oldpeak" else 1.0
        return float(
            st.number_input(
                display_name(column_name),
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(round(median_value, 2)),
                step=step,
                format="%.2f",
            )
        )

    @staticmethod
    def render_categorical_input(column_name: str, series: pd.Series) -> str:
        options = sorted(series.dropna().astype(str).unique().tolist())
        if not options:
            options = ["Unknown"]
        modes = series.dropna().astype(str).mode()
        default_value = str(modes.iloc[0]) if not modes.empty else options[0]
        default_index = options.index(default_value) if default_value in options else 0
        return st.selectbox(display_name(column_name), options, index=default_index)

    @staticmethod
    def render_prediction_result(
        result: HeartDiseasePrediction, elapsed_seconds: float
    ) -> None:
        st.success("Prediction complete")
        left, middle, right, extra = st.columns(4)
        left.metric("Prediction", result.prediction)
        middle.metric("Probability", f"{result.risk_probability:.2%}")
        right.metric("Risk band", result.risk_band)
        extra.metric("Latency", f"{elapsed_seconds:.2f}s")

        st.progress(float(result.risk_probability), text=f"{result.model_name} risk score")
        st.json(result.to_dict())
        st.info(
            "This output is an ML estimate for educational analysis. It is not a "
            "clinical diagnosis or treatment recommendation."
        )


def main() -> None:
    DashboardRenderer().render()

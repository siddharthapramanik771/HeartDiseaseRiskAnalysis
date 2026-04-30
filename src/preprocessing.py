from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RUNTIME_CONFIG, RuntimeConfig


BINARY_TARGET = "heart_disease"
RISK_LABEL = "risk_label"
SEVERITY_LABEL = "severity_label"
MISSING_CATEGORY = "Unknown"

NUMERIC_HINTS = ["id", "age", "trestbps", "chol", "thalch", "oldpeak", "ca", "num"]
BOOLEAN_HINTS = ["fbs", "exang"]
MISSING_TOKENS = ["?", "", "NA", "N/A", "na", "n/a", "None", "none", "nan", "NaN"]
DERIVED_LABEL_COLUMNS = {BINARY_TARGET, RISK_LABEL, SEVERITY_LABEL}

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "dataset": "Source dataset",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure",
    "chol": "Serum cholesterol",
    "fbs": "Fasting blood sugar > 120",
    "restecg": "Resting ECG",
    "thalch": "Max heart rate",
    "exang": "Exercise angina",
    "oldpeak": "ST depression",
    "slope": "ST slope",
    "ca": "Major vessels count",
    "thal": "Thalassemia",
    "num": "Disease severity",
    BINARY_TARGET: "Heart disease",
}

SEVERITY_LABELS = {
    0: "0 - No disease",
    1: "1 - Mild",
    2: "2 - Moderate",
    3: "3 - Severe",
    4: "4 - Very severe",
}

SEVERITY_ORDER = [
    "0 - No disease",
    "1 - Mild",
    "2 - Moderate",
    "3 - Severe",
    "4 - Very severe",
]


@dataclass(frozen=True)
class FeatureSchema:
    numeric_columns: list[str]
    categorical_columns: list[str]

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_columns + self.categorical_columns


@dataclass(frozen=True)
class FeatureDefaults:
    numeric_defaults: dict[str, float]
    categorical_defaults: dict[str, str]


@dataclass(frozen=True)
class DataPreprocessor:
    """Owns UCI heart disease cleaning and feature preparation rules."""

    target_column: str
    id_column: str
    include_dataset_feature: bool = False

    @classmethod
    def from_config(cls, config: RuntimeConfig = RUNTIME_CONFIG) -> "DataPreprocessor":
        return cls(
            target_column=config.target_column,
            id_column=config.id_column,
            include_dataset_feature=config.include_dataset_feature,
        )

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cleaned = df.copy()
        cleaned.columns = cleaned.columns.str.strip()
        cleaned = cleaned.replace(MISSING_TOKENS, np.nan)
        cleaned = self._normalize_object_columns(cleaned)
        cleaned = self._coerce_known_columns(cleaned)
        cleaned = self.add_target_labels(cleaned)
        return cleaned

    def add_target_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.target_column not in df.columns:
            return df

        labeled = df.copy()
        labeled[BINARY_TARGET] = np.where(labeled[self.target_column] > 0, 1, 0)
        labeled.loc[labeled[self.target_column].isna(), BINARY_TARGET] = np.nan
        labeled[RISK_LABEL] = (
            labeled[BINARY_TARGET]
            .map({0: "No heart disease", 1: "Heart disease"})
            .fillna(MISSING_CATEGORY)
        )
        labeled[SEVERITY_LABEL] = (
            labeled[self.target_column]
            .map(SEVERITY_LABELS)
            .fillna(MISSING_CATEGORY)
        )
        return labeled

    def get_feature_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        excluded_columns = {
            self.id_column,
            self.target_column,
            *DERIVED_LABEL_COLUMNS,
        }
        feature_df = df.drop(
            columns=[column for column in excluded_columns if column in df],
            errors="ignore",
        )
        if not self.include_dataset_feature and "dataset" in feature_df.columns:
            feature_df = feature_df.drop(columns=["dataset"])
        return feature_df

    def build_transformer(self, feature_df: pd.DataFrame) -> tuple[ColumnTransformer, FeatureSchema]:
        schema = self.infer_schema(feature_df)
        transformers = []

        if schema.numeric_columns:
            transformers.append(
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    schema.numeric_columns,
                )
            )

        if schema.categorical_columns:
            transformers.append(
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
                            ("onehot", make_one_hot_encoder()),
                        ]
                    ),
                    schema.categorical_columns,
                )
            )

        return ColumnTransformer(transformers, remainder="drop"), schema

    def infer_schema(self, feature_df: pd.DataFrame) -> FeatureSchema:
        numeric_columns = [
            column for column in feature_df.columns if is_numeric_dtype(feature_df[column])
        ]
        categorical_columns = [
            column for column in feature_df.columns if not is_numeric_dtype(feature_df[column])
        ]
        return FeatureSchema(numeric_columns, categorical_columns)

    def encode_target(self, df: pd.DataFrame) -> pd.Series:
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' was not found.")
        target = pd.to_numeric(df[self.target_column], errors="coerce")
        if target.isna().any():
            raise ValueError("Training target contains missing or non-numeric severity values.")
        return (target > 0).astype(int)

    def derive_feature_defaults(self, feature_df: pd.DataFrame) -> FeatureDefaults:
        numeric_defaults: dict[str, float] = {}
        categorical_defaults: dict[str, str] = {}

        for column in feature_df.columns:
            series = feature_df[column]
            if is_numeric_dtype(series):
                numeric_defaults[column] = float(series.median())
            else:
                modes = series.dropna().astype(str).mode()
                categorical_defaults[column] = (
                    str(modes.iloc[0]) if not modes.empty else MISSING_CATEGORY
                )

        return FeatureDefaults(numeric_defaults, categorical_defaults)

    def _normalize_object_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in df.select_dtypes(include=["object", "string"]).columns:
            df[column] = df[column].apply(
                lambda value: value.strip() if isinstance(value, str) else value
            )
        return df

    def _coerce_known_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        for column in NUMERIC_HINTS:
            if column in df.columns:
                df[column] = pd.to_numeric(df[column], errors="coerce")

        for column in BOOLEAN_HINTS:
            if column in df.columns:
                df[column] = df[column].apply(normalize_boolean)

        return df


def normalize_boolean(value: Any) -> str | float:
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return "True"
    if text in {"false", "0", "no", "n"}:
        return "False"
    return str(value).strip()


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def display_name(column: str) -> str:
    return FEATURE_LABELS.get(column, column.replace("_", " ").title())

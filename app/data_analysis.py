import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import RUNTIME_CONFIG, RuntimeConfig
from src.preprocessing import (
    BINARY_TARGET,
    RISK_LABEL,
    SEVERITY_LABEL,
    SEVERITY_ORDER,
    display_name,
)


RISK_COLORS = {
    "No heart disease": "#0f766e",
    "Heart disease": "#be123c",
    "Unknown": "#64748b",
}


class DataAnalysisRenderer:
    """Renders interactive analysis for the UCI heart disease dataset."""

    def __init__(self, config: RuntimeConfig = RUNTIME_CONFIG) -> None:
        self.config = config

    def render(self, df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
        st.header("Data Analysis")
        st.write(
            "Explore the UCI heart disease data through target balance, clinical "
            "segments, numeric distributions, relationships, and data quality checks."
        )

        self.render_overview(df, feature_df)
        story_tab, segment_tab, numeric_tab, relation_tab, quality_tab, table_tab = st.tabs(
            [
                "Risk Story",
                "Clinical Segments",
                "Vitals & Labs",
                "Relationships",
                "Data Quality",
                "Data Table",
            ]
        )

        with story_tab:
            self.render_risk_story(df)
        with segment_tab:
            self.render_segments(df)
        with numeric_tab:
            self.render_numeric_trends(df)
        with relation_tab:
            self.render_relationships(df)
        with quality_tab:
            self.render_quality(df)
        with table_tab:
            self.render_table(df)

    def render_overview(self, df: pd.DataFrame, feature_df: pd.DataFrame) -> None:
        columns = st.columns(4)
        columns[0].metric("Rows", f"{len(df):,}")
        columns[1].metric("Input features", f"{feature_df.shape[1]:,}")
        columns[2].metric("Disease prevalence", f"{self.risk_rate(df):.1%}")
        missing_rate = df.isna().sum().sum() / df.size if df.size else 0
        columns[3].metric("Missing cells", f"{missing_rate:.1%}")

    def render_risk_story(self, df: pd.DataFrame) -> None:
        left, right = st.columns(2)
        left.plotly_chart(self.target_distribution_chart(df), width="stretch")
        right.plotly_chart(self.severity_chart(df), width="stretch")

        left, right = st.columns(2)
        left.plotly_chart(self.age_band_chart(df), width="stretch")
        right.plotly_chart(self.source_dataset_chart(df), width="stretch")

    def render_segments(self, df: pd.DataFrame) -> None:
        categorical_cols = [
            column
            for column in df.select_dtypes(exclude=[np.number]).columns.tolist()
            if column not in {RISK_LABEL, SEVERITY_LABEL}
        ]
        if not categorical_cols:
            st.info("No categorical clinical fields are available.")
            return

        preferred = ["sex", "cp", "exang", "slope", "thal", "restecg", "fbs", "dataset"]
        options = [column for column in preferred if column in categorical_cols]
        options.extend([column for column in categorical_cols if column not in options])
        segment_col = st.selectbox("Segment by", options, format_func=display_name)

        left, right = st.columns(2)
        left.plotly_chart(self.category_count_chart(df, segment_col), width="stretch")
        right.plotly_chart(self.category_risk_chart(df, segment_col), width="stretch")

        st.subheader("Highest-risk clinical groups")
        group_table = self.build_group_risk_table(df)
        if group_table.empty:
            st.info("Not enough complete grouped records to build a group-risk table.")
        else:
            st.dataframe(
                group_table.style.format({"Risk rate": "{:.1%}", "Records": "{:,.0f}"}),
                width="stretch",
                height=360,
            )

    def render_numeric_trends(self, df: pd.DataFrame) -> None:
        numeric_cols = [
            column
            for column in df.select_dtypes(include=[np.number]).columns.tolist()
            if column not in {self.config.id_column, self.config.target_column, BINARY_TARGET}
        ]
        if not numeric_cols:
            st.info("No numeric clinical fields are available.")
            return

        preferred = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
        options = [column for column in preferred if column in numeric_cols]
        options.extend([column for column in numeric_cols if column not in options])
        numeric_col = st.selectbox("Analyze numeric feature", options, format_func=display_name)

        left, right = st.columns(2)
        left.plotly_chart(self.numeric_histogram(df, numeric_col), width="stretch")
        right.plotly_chart(self.numeric_boxplot(df, numeric_col), width="stretch")

        st.subheader("Summary by risk group")
        summary = (
            df.groupby(RISK_LABEL, dropna=False)[numeric_col]
            .describe()
            .reset_index()
            .rename(columns={RISK_LABEL: "Risk group"})
        )
        st.dataframe(summary.style.format(precision=2), width="stretch")

    def render_relationships(self, df: pd.DataFrame) -> None:
        numeric_cols = [
            column
            for column in df.select_dtypes(include=[np.number]).columns.tolist()
            if column != self.config.id_column
        ]
        if len(numeric_cols) < 2:
            st.info("At least two numeric fields are needed for relationship charts.")
            return

        left, right = st.columns(2)
        default_x = numeric_cols.index("age") if "age" in numeric_cols else 0
        default_y = numeric_cols.index("thalch") if "thalch" in numeric_cols else min(1, len(numeric_cols) - 1)
        x_col = left.selectbox("X axis", numeric_cols, index=default_x, format_func=display_name)
        y_col = right.selectbox("Y axis", numeric_cols, index=default_y, format_func=display_name)

        st.plotly_chart(self.scatter_chart(df, x_col, y_col), width="stretch")
        st.plotly_chart(self.correlation_heatmap(df, numeric_cols), width="stretch")

    def render_quality(self, df: pd.DataFrame) -> None:
        duplicate_rows = int(df.duplicated().sum())
        columns = st.columns(4)
        columns[0].metric("Rows", f"{len(df):,}")
        columns[1].metric("Columns", f"{df.shape[1]:,}")
        columns[2].metric("Duplicate rows", f"{duplicate_rows:,}")
        columns[3].metric("Missing cells", f"{int(df.isna().sum().sum()):,}")

        left, right = st.columns([1.15, 1])
        left.plotly_chart(self.missingness_chart(df), width="stretch")
        profile = pd.DataFrame(
            {
                "Column": df.columns,
                "Type": [str(df[column].dtype) for column in df.columns],
                "Unique values": [df[column].nunique(dropna=True) for column in df.columns],
                "Missing": [df[column].isna().sum() for column in df.columns],
            }
        )
        right.dataframe(profile, width="stretch", height=430)

        st.subheader("Numeric summary")
        st.dataframe(df.select_dtypes(include=[np.number]).describe().T, width="stretch")

    @staticmethod
    def render_table(df: pd.DataFrame) -> None:
        st.subheader("Preview")
        st.dataframe(arrow_safe_frame(df.head(100)), width="stretch", height=360)
        st.subheader("Full descriptive summary")
        st.dataframe(arrow_safe_frame(df.describe(include="all")), width="stretch")

    def target_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        counts = (
            df[RISK_LABEL]
            .value_counts()
            .rename_axis("Risk group")
            .reset_index(name="Records")
        )
        fig = px.pie(
            counts,
            names="Risk group",
            values="Records",
            hole=0.48,
            title="Binary target distribution",
            color="Risk group",
            color_discrete_map=RISK_COLORS,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        return fig

    def severity_chart(self, df: pd.DataFrame) -> go.Figure:
        counts = (
            df[SEVERITY_LABEL]
            .value_counts()
            .reindex(SEVERITY_ORDER, fill_value=0)
            .rename_axis("Severity")
            .reset_index(name="Records")
        )
        fig = px.bar(
            counts,
            x="Severity",
            y="Records",
            color="Severity",
            color_discrete_sequence=["#0f766e", "#b45309", "#db2777", "#dc2626", "#7f1d1d"],
            text="Records",
            title="Original UCI severity classes",
        )
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Records")
        fig.update_traces(textposition="outside", cliponaxis=False)
        return fig

    def age_band_chart(self, df: pd.DataFrame) -> go.Figure:
        if "age" not in df.columns:
            return empty_figure("Age column not available")

        working = df.dropna(subset=["age", BINARY_TARGET]).copy()
        working["Age band"] = pd.cut(
            working["age"],
            bins=[0, 35, 45, 55, 65, 75, 120],
            labels=["<=35", "36-45", "46-55", "56-65", "66-75", "76+"],
        )
        risk = (
            working.groupby("Age band", observed=False)[BINARY_TARGET]
            .mean()
            .mul(100)
            .reset_index(name="Disease rate")
        )
        fig = px.line(
            risk,
            x="Age band",
            y="Disease rate",
            markers=True,
            title="Heart disease rate by age band",
        )
        fig.update_traces(line_color="#be123c", marker_size=9)
        fig.update_layout(yaxis_ticksuffix="%", xaxis_title="", yaxis_title="Disease rate")
        return fig

    def source_dataset_chart(self, df: pd.DataFrame) -> go.Figure:
        if "dataset" not in df.columns:
            return empty_figure("Source dataset column not available")

        grouped = (
            df.groupby(["dataset", RISK_LABEL], dropna=False)
            .size()
            .reset_index(name="Records")
        )
        fig = px.bar(
            grouped,
            x="dataset",
            y="Records",
            color=RISK_LABEL,
            barmode="group",
            color_discrete_map=RISK_COLORS,
            title="Risk distribution by source dataset",
        )
        fig.update_layout(xaxis_title="", yaxis_title="Records")
        return fig

    def category_count_chart(self, df: pd.DataFrame, column: str) -> go.Figure:
        counts = (
            df[column]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .nlargest(12)
            .rename_axis(display_name(column))
            .reset_index(name="Records")
        )
        fig = px.bar(
            counts,
            x="Records",
            y=display_name(column),
            orientation="h",
            color="Records",
            color_continuous_scale=["#dbeafe", "#0f766e"],
            title=f"{display_name(column)} record count",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return fig

    def category_risk_chart(self, df: pd.DataFrame, column: str) -> go.Figure:
        working = df.dropna(subset=[column, BINARY_TARGET]).copy()
        grouped = (
            working.groupby(column, dropna=False)[BINARY_TARGET]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "Records", "mean": "Disease rate"})
            .sort_values("Disease rate", ascending=False)
            .head(12)
        )
        grouped["Disease rate"] *= 100
        fig = px.bar(
            grouped,
            x="Disease rate",
            y=column,
            orientation="h",
            color="Disease rate",
            color_continuous_scale=["#d1fae5", "#f59e0b", "#be123c"],
            hover_data=["Records"],
            title=f"Disease rate by {display_name(column)}",
        )
        fig.update_layout(yaxis_title=display_name(column), xaxis_ticksuffix="%")
        return fig

    def numeric_histogram(self, df: pd.DataFrame, column: str) -> go.Figure:
        fig = px.histogram(
            df,
            x=column,
            color=RISK_LABEL,
            marginal="box",
            nbins=28,
            barmode="overlay",
            opacity=0.72,
            color_discrete_map=RISK_COLORS,
            title=f"{display_name(column)} distribution",
        )
        fig.update_layout(xaxis_title=display_name(column), yaxis_title="Records")
        return fig

    def numeric_boxplot(self, df: pd.DataFrame, column: str) -> go.Figure:
        fig = px.box(
            df,
            x=RISK_LABEL,
            y=column,
            color=RISK_LABEL,
            points="outliers",
            color_discrete_map=RISK_COLORS,
            title=f"{display_name(column)} by risk group",
        )
        fig.update_layout(xaxis_title="", yaxis_title=display_name(column), showlegend=False)
        return fig

    def scatter_chart(self, df: pd.DataFrame, x_col: str, y_col: str) -> go.Figure:
        hover_columns = [
            column for column in ["sex", "cp", "oldpeak", "chol", self.config.target_column]
            if column in df.columns and column not in {x_col, y_col}
        ]
        plotting_df = df.copy()
        size_column = None
        if "chol" in plotting_df.columns and x_col != "chol" and y_col != "chol":
            size_column = "_chol_marker_size"
            median_chol = plotting_df["chol"].median()
            plotting_df[size_column] = plotting_df["chol"].fillna(median_chol).clip(lower=0)

        fig = px.scatter(
            plotting_df,
            x=x_col,
            y=y_col,
            color=RISK_LABEL,
            size=size_column,
            hover_data=hover_columns,
            color_discrete_map=RISK_COLORS,
            title=f"{display_name(x_col)} vs {display_name(y_col)}",
        )
        fig.update_layout(xaxis_title=display_name(x_col), yaxis_title=display_name(y_col))
        return fig

    @staticmethod
    def correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str]) -> go.Figure:
        corr = df[numeric_cols].corr(numeric_only=True)
        fig = px.imshow(
            corr,
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu_r",
            text_auto=".2f",
            title="Numeric correlation heatmap",
        )
        fig.update_layout(height=560)
        return fig

    @staticmethod
    def missingness_chart(df: pd.DataFrame) -> go.Figure:
        missing = (
            df.isna()
            .sum()
            .rename("Missing")
            .reset_index()
            .rename(columns={"index": "Column"})
        )
        missing["Missing percent"] = missing["Missing"] / len(df)
        plotted = missing.sort_values("Missing", ascending=False)
        fig = px.bar(
            plotted,
            x="Missing",
            y="Column",
            orientation="h",
            color="Missing percent",
            color_continuous_scale=["#d1fae5", "#f59e0b", "#be123c"],
            title="Missing values by column",
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return fig

    @staticmethod
    def build_group_risk_table(df: pd.DataFrame) -> pd.DataFrame:
        columns = [column for column in ["sex", "cp", "exang", "slope"] if column in df.columns]
        if not columns:
            return pd.DataFrame(columns=["Group", "Records", "Risk rate"])

        working = df.dropna(subset=columns + [BINARY_TARGET]).copy()
        grouped = (
            working.groupby(columns, observed=False)[BINARY_TARGET]
            .agg(["count", "mean"])
            .reset_index()
            .rename(columns={"count": "Records", "mean": "Risk rate"})
        )
        grouped = grouped[grouped["Records"] >= 12].copy()
        grouped["Group"] = grouped[columns].astype(str).agg(" | ".join, axis=1)
        return grouped[["Group", "Records", "Risk rate"]].sort_values(
            ["Risk rate", "Records"],
            ascending=[False, False],
        )

    @staticmethod
    def risk_rate(df: pd.DataFrame) -> float:
        if BINARY_TARGET not in df.columns:
            return 0.0
        return float(df[BINARY_TARGET].dropna().mean())


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def arrow_safe_frame(df: pd.DataFrame) -> pd.DataFrame:
    safe = df.copy()
    for column in safe.select_dtypes(include=["object", "string"]).columns:
        safe[column] = safe[column].astype("string")
    return safe

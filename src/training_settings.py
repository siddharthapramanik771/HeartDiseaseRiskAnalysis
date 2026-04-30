from dataclasses import asdict
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TrainingSettings:
    """Settings that control reproducibility and holdout evaluation."""

    candidate_models: tuple[str, ...] = ("Random Forest", "KNN")
    random_state: int = 42
    test_size: float = 0.25
    selection_metric: str = "test_f1"
    random_forest_estimators: int = 300
    random_forest_max_depth: int | None = 8
    random_forest_min_samples_leaf: int = 2
    knn_neighbors: int = 9

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["candidate_models"] = list(self.candidate_models)
        return payload

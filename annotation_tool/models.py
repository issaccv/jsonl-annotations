from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any


class FeedbackFilter(str, Enum):
    ALL = "all"
    PENDING = "pending"
    BAD = "bad"

    @property
    def label(self) -> str:
        return {
            self.ALL: "全部",
            self.PENDING: "未标注",
            self.BAD: "已标注不好",
        }[self]


@dataclass(frozen=True)
class CaseRecord:
    dataset_path: Path
    source_file: str
    line_number: int
    raw: dict[str, Any]

    @property
    def id(self) -> str:
        value = self.raw.get("id")
        return value if isinstance(value, str) and value else f"line-{self.line_number}"

    @property
    def question(self) -> Any:
        return self.raw.get("question")

    @property
    def function(self) -> Any:
        return self.raw.get("function")

    @property
    def ground_truth(self) -> Any:
        return self.raw.get("ground_truth")

    @property
    def source(self) -> str:
        value = self.raw.get("source")
        return value if isinstance(value, str) else ""

    @property
    def type_label(self) -> str:
        value = self.raw.get("type")
        return value if isinstance(value, str) else ""

    @property
    def question_type(self) -> str:
        value = self.raw.get("question_type")
        return value if isinstance(value, str) else ""

    @property
    def language(self) -> str:
        value = self.raw.get("language")
        return value if isinstance(value, str) else ""

    @property
    def base_feedback(self) -> str | None:
        value = self.raw.get("iteration_feedback")
        return value if isinstance(value, str) and value else None

    @property
    def lookup_key(self) -> tuple[str, str]:
        return (self.source_file, self.id)


@dataclass(frozen=True)
class FeedbackRecord:
    source_file: str
    line_number: int
    id: str
    iteration_feedback: str
    updated_at: str

    @property
    def lookup_key(self) -> tuple[str, str]:
        return (self.source_file, self.id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "line_number": self.line_number,
            "id": self.id,
            "iteration_feedback": self.iteration_feedback,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class DatasetSummary:
    path: Path
    total_cases: int
    annotated_cases: int
    bad_cases: int

    @property
    def completion_ratio(self) -> float:
        if self.total_cases == 0:
            return 0.0
        return self.annotated_cases / self.total_cases

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import CaseRecord, DatasetSummary, FeedbackRecord


class DataError(RuntimeError):
    """Raised when dataset or sidecar contents are invalid."""


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return current


def resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path.resolve()
    return (base_dir / path).resolve()


def relative_source_path(dataset_path: Path, project_root: Path) -> str:
    resolved = dataset_path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved)


def sidecar_path_for(dataset_path: Path, project_root: Path) -> Path:
    return project_root / "annotations" / f"{dataset_path.stem}.feedback.jsonl"


def export_path_for(dataset_path: Path) -> Path:
    return dataset_path.with_name(f"{dataset_path.stem}.annotated.jsonl")


def discover_datasets(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    paths = [
        path.resolve()
        for path in data_dir.rglob("*.jsonl")
        if not path.name.endswith(".annotated.jsonl")
    ]
    return sorted(paths)


def _read_json_lines(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DataError(f"{path}:{line_number} 不是有效的 JSONL: {exc}") from exc
            if not isinstance(value, dict):
                raise DataError(f"{path}:{line_number} 需要是 JSON object")
            rows.append(value)
    return rows


def load_cases(dataset_path: Path, project_root: Path) -> list[CaseRecord]:
    rows = _read_json_lines(dataset_path)
    source_file = relative_source_path(dataset_path, project_root)
    cases: list[CaseRecord] = []
    for line_number, row in enumerate(rows, start=1):
        cases.append(
            CaseRecord(
                dataset_path=dataset_path,
                source_file=source_file,
                line_number=line_number,
                raw=row,
            )
        )
    return cases


class FeedbackStore:
    def __init__(self, project_root: Path):
        self.project_root = project_root.resolve()

    def load_feedback_map(self, dataset_path: Path) -> dict[tuple[str, str], FeedbackRecord]:
        sidecar_path = sidecar_path_for(dataset_path, self.project_root)
        if not sidecar_path.exists():
            return {}

        feedback_map: dict[tuple[str, str], FeedbackRecord] = {}
        for line_number, payload in enumerate(_read_json_lines(sidecar_path), start=1):
            try:
                record = FeedbackRecord(
                    source_file=str(payload["source_file"]),
                    line_number=int(payload["line_number"]),
                    id=str(payload["id"]),
                    iteration_feedback=str(payload["iteration_feedback"]),
                    updated_at=str(payload["updated_at"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise DataError(
                    f"{sidecar_path}:{line_number} 不是有效的反馈记录: {exc}"
                ) from exc
            feedback_map[record.lookup_key] = record
        return feedback_map

    def resolve_feedback(
        self,
        case: CaseRecord,
        feedback_map: dict[tuple[str, str], FeedbackRecord],
    ) -> str | None:
        entry = feedback_map.get(case.lookup_key)
        return entry.iteration_feedback if entry else case.base_feedback

    def write_feedback(self, dataset_path: Path, case: CaseRecord, feedback: str) -> FeedbackRecord:
        record = FeedbackRecord(
            source_file=relative_source_path(dataset_path, self.project_root),
            line_number=case.line_number,
            id=case.id,
            iteration_feedback=feedback,
            updated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        sidecar_path = sidecar_path_for(dataset_path, self.project_root)
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with sidecar_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        return record

    def summarize_dataset(self, dataset_path: Path) -> DatasetSummary:
        cases = load_cases(dataset_path, self.project_root)
        feedback_map = self.load_feedback_map(dataset_path)
        annotated = 0
        bad = 0
        for case in cases:
            feedback = self.resolve_feedback(case, feedback_map)
            if feedback:
                annotated += 1
            if feedback and feedback.startswith("不好"):
                bad += 1
        return DatasetSummary(
            path=dataset_path,
            total_cases=len(cases),
            annotated_cases=annotated,
            bad_cases=bad,
        )

    def export_dataset(self, dataset_path: Path) -> Path:
        cases = load_cases(dataset_path, self.project_root)
        feedback_map = self.load_feedback_map(dataset_path)
        output_path = export_path_for(dataset_path)
        with output_path.open("w", encoding="utf-8") as handle:
            for case in cases:
                row = dict(case.raw)
                feedback = self.resolve_feedback(case, feedback_map)
                if feedback:
                    row["iteration_feedback"] = feedback
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return output_path


QUESTION_TYPE_HINTS = {
    "Parallel": "一次请求需要并行生成多次函数调用。",
    "Multiple": "同一请求包含多轮或多步函数调用。",
    "Simple": "单次请求对应单个函数调用。",
    "AST": "请求更强调结构化参数解析与抽取。",
}

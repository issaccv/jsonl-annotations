from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .models import CanonicalCase, CaseRecord, CaseSection, SectionKind
from .renderers import (
    dump_json,
    render_mc_answer_text,
    render_mc_options_text,
    render_function_text,
    render_ground_truth_text,
    render_question_text,
    render_text_value,
)


SECTION_KEYS = {
    "question",
    "options",
    "answer",
    "solution",
    "function",
    "ground_truth",
    "output_requirement",
}
PREFERRED_METADATA_KEYS = ("type", "question_type", "language", "source", "output_requirement")
EXCLUDED_METADATA_KEYS = {"id", "iteration_feedback"}


class CaseAdapter(Protocol):
    name: str

    def can_adapt(self, row: dict[str, Any]) -> int: ...

    def to_canonical(self, case: CaseRecord) -> CanonicalCase: ...


@dataclass(frozen=True)
class ParallelAdapter:
    name: str = "parallel"

    def can_adapt(self, row: dict[str, Any]) -> int:
        has_question = "question" in row
        has_function = "function" in row
        has_ground_truth = "ground_truth" in row
        if has_question and has_function and has_ground_truth:
            return 100
        if has_function and has_ground_truth:
            return 90
        if has_function or has_ground_truth:
            return 60
        return 0

    def to_canonical(self, case: CaseRecord) -> CanonicalCase:
        row = case.raw
        sections: list[CaseSection] = []
        sections.append(
            CaseSection(
                key="question",
                title="question",
                kind=SectionKind.TEXT,
                rendered=render_question_text(row.get("question")),
                raw=row.get("question"),
            )
        )
        sections.append(
            CaseSection(
                key="function",
                title="function",
                kind=SectionKind.CODE,
                rendered=render_function_text(row.get("function")),
                raw=row.get("function"),
            )
        )
        sections.append(
            CaseSection(
                key="ground_truth",
                title="ground_truth",
                kind=SectionKind.CODE,
                rendered=render_ground_truth_text(row.get("ground_truth")),
                raw=row.get("ground_truth"),
            )
        )
        extras = _extra_payload(row, handled_keys={"question", "function", "ground_truth"})
        if extras:
            sections.append(
                CaseSection(
                    key="extra",
                    title="extra",
                    kind=SectionKind.JSON,
                    rendered=dump_json(extras),
                    raw=extras,
                )
            )
        metadata = _build_metadata(row, self.name)
        return CanonicalCase(
            case_id=case.id,
            source_file=case.source_file,
            line_number=case.line_number,
            metadata=metadata,
            sections=sections,
            raw=row,
        )


@dataclass(frozen=True)
class QAAdapter:
    name: str = "qa"

    def can_adapt(self, row: dict[str, Any]) -> int:
        has_question = "question" in row
        has_answer = "answer" in row
        has_solution = "solution" in row
        if has_question and has_answer and has_solution:
            return 95
        if has_question and has_answer:
            return 85
        return 0

    def to_canonical(self, case: CaseRecord) -> CanonicalCase:
        row = case.raw
        sections = [
            CaseSection(
                key="question",
                title="question",
                kind=SectionKind.TEXT,
                rendered=render_text_value(row.get("question")),
                raw=row.get("question"),
            ),
            CaseSection(
                key="answer",
                title="answer",
                kind=SectionKind.TEXT,
                rendered=render_text_value(row.get("answer")),
                raw=row.get("answer"),
            ),
            CaseSection(
                key="solution",
                title="solution",
                kind=SectionKind.TEXT,
                rendered=render_text_value(row.get("solution")),
                raw=row.get("solution"),
            ),
        ]
        extras = _extra_payload(row, handled_keys={"question", "answer", "solution"})
        if extras:
            sections.append(
                CaseSection(
                    key="extra",
                    title="extra",
                    kind=SectionKind.JSON,
                    rendered=dump_json(extras),
                    raw=extras,
                )
            )
        metadata = _build_metadata(row, self.name)
        return CanonicalCase(
            case_id=case.id,
            source_file=case.source_file,
            line_number=case.line_number,
            metadata=metadata,
            sections=sections,
            raw=row,
        )


@dataclass(frozen=True)
class MCAdapter:
    name: str = "mc"

    def can_adapt(self, row: dict[str, Any]) -> int:
        question_type = row.get("question_type")
        has_question = "question" in row
        has_options = "options" in row
        has_answer = "answer" in row
        if question_type == "mc" and has_question and has_options and has_answer:
            return 110
        if has_question and has_options and has_answer:
            return 100
        return 0

    def to_canonical(self, case: CaseRecord) -> CanonicalCase:
        row = case.raw
        sections = [
            CaseSection(
                key="question",
                title="question",
                kind=SectionKind.TEXT,
                rendered=render_text_value(row.get("question")),
                raw=row.get("question"),
            ),
            CaseSection(
                key="options",
                title="options",
                kind=SectionKind.TEXT,
                rendered=render_mc_options_text(row.get("options")),
                raw=row.get("options"),
            ),
            CaseSection(
                key="answer",
                title="answer",
                kind=SectionKind.TEXT,
                rendered=render_mc_answer_text(row.get("answer")),
                raw=row.get("answer"),
            ),
            CaseSection(
                key="solution",
                title="solution",
                kind=SectionKind.TEXT,
                rendered=render_text_value(row.get("solution")),
                raw=row.get("solution"),
            ),
        ]
        extras = _extra_payload(row, handled_keys={"question", "options", "answer", "solution"})
        if extras:
            sections.append(
                CaseSection(
                    key="extra",
                    title="extra",
                    kind=SectionKind.JSON,
                    rendered=dump_json(extras),
                    raw=extras,
                )
            )
        metadata = _build_metadata(row, self.name)
        return CanonicalCase(
            case_id=case.id,
            source_file=case.source_file,
            line_number=case.line_number,
            metadata=metadata,
            sections=sections,
            raw=row,
        )


@dataclass(frozen=True)
class GenericAdapter:
    name: str = "generic"

    def can_adapt(self, row: dict[str, Any]) -> int:
        return 1 if isinstance(row, dict) else 0

    def to_canonical(self, case: CaseRecord) -> CanonicalCase:
        row = case.raw
        sections: list[CaseSection] = []
        for key in ("question", "answer", "solution", "function", "ground_truth"):
            if key not in row:
                continue
            value = row.get(key)
            sections.append(_build_generic_section(key, value))

        extras = _extra_payload(row, handled_keys={section.key for section in sections})
        if extras:
            sections.append(
                CaseSection(
                    key="extra",
                    title="extra",
                    kind=SectionKind.JSON,
                    rendered=dump_json(extras),
                    raw=extras,
                )
            )
        if not sections:
            sections.append(
                CaseSection(
                    key="raw",
                    title="raw",
                    kind=SectionKind.JSON,
                    rendered=dump_json(row),
                    raw=row,
                )
            )

        metadata = _build_metadata(row, self.name)
        return CanonicalCase(
            case_id=case.id,
            source_file=case.source_file,
            line_number=case.line_number,
            metadata=metadata,
            sections=sections,
            raw=row,
        )


ADAPTERS: tuple[CaseAdapter, ...] = (
    ParallelAdapter(),
    MCAdapter(),
    QAAdapter(),
    GenericAdapter(),
)


def adapt_case(case: CaseRecord) -> CanonicalCase:
    adapter = max(ADAPTERS, key=lambda item: item.can_adapt(case.raw))
    return adapter.to_canonical(case)


def _build_metadata(row: dict[str, Any], schema_name: str) -> dict[str, str]:
    metadata: dict[str, str] = {"schema": schema_name}
    for key in PREFERRED_METADATA_KEYS:
        text = _coerce_metadata_text(row.get(key))
        if text:
            metadata[key] = text

    for key, value in row.items():
        if key in EXCLUDED_METADATA_KEYS or key in SECTION_KEYS or key in PREFERRED_METADATA_KEYS:
            continue
        text = _coerce_metadata_text(value)
        if text:
            metadata[key] = text
    return metadata


def _coerce_metadata_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return ""


def _extra_payload(row: dict[str, Any], handled_keys: set[str]) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    for key, value in row.items():
        if key in handled_keys:
            continue
        if key in EXCLUDED_METADATA_KEYS or key in PREFERRED_METADATA_KEYS:
            continue
        extras[key] = value
    return extras


def _build_generic_section(key: str, value: Any) -> CaseSection:
    if key == "function":
        return CaseSection(
            key=key,
            title=key,
            kind=SectionKind.CODE,
            rendered=render_function_text(value),
            raw=value,
        )
    if key == "ground_truth":
        return CaseSection(
            key=key,
            title=key,
            kind=SectionKind.CODE,
            rendered=render_ground_truth_text(value),
            raw=value,
        )
    if key == "question" and isinstance(value, list):
        rendered = render_question_text(value)
    else:
        rendered = render_text_value(value)
    return CaseSection(
        key=key,
        title=key,
        kind=SectionKind.TEXT,
        rendered=rendered,
        raw=value,
    )

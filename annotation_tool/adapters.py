from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .models import CanonicalCase, CaseRecord, CaseSection, SectionKind
from .renderers import (
    dump_json,
    render_function_text,
    render_ground_truth_text,
    render_mc_answer_text,
    render_mc_options_text,
    render_question_text,
    render_text_value,
)


DEFAULT_METADATA_FIELDS = ("type", "question_type", "language", "source", "output_requirement")
EXCLUDED_KEYS = {"id", "iteration_feedback"}


class SchemaConfigError(RuntimeError):
    """Raised when schema mapping config is invalid."""


@dataclass(frozen=True)
class PanelSpec:
    title: str
    source_key: str
    formatter: str = "text_value"
    kind: SectionKind = SectionKind.TEXT
    formatter_args: dict[str, Any] = field(default_factory=dict)
    visible_when_missing: bool = True


@dataclass(frozen=True)
class SchemaSpec:
    name: str
    version: int
    file_globs: tuple[str, ...]
    metadata_fields: tuple[str, ...]
    panels: tuple[PanelSpec, ...]
    auto_extra: bool = True
    priority: int = 0


class ConfiguredAdapter:
    def __init__(self, schema: SchemaSpec):
        self.schema = schema
        self.name = schema.name

    def to_canonical(self, case: CaseRecord) -> CanonicalCase:
        row = case.raw
        sections: list[CaseSection] = []
        handled_keys: set[str] = set()

        for panel in self.schema.panels:
            has_key = panel.source_key in row
            if not has_key and not panel.visible_when_missing:
                continue

            value = row.get(panel.source_key)
            rendered = _render_with_formatter(panel.formatter, value, panel.formatter_args)
            sections.append(
                CaseSection(
                    key=panel.source_key,
                    title=panel.title,
                    kind=panel.kind,
                    rendered=rendered,
                    raw=value,
                )
            )
            handled_keys.add(panel.source_key)

        if self.schema.auto_extra:
            extras = _extra_payload(
                row,
                handled_keys=handled_keys | set(self.schema.metadata_fields),
            )
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

        metadata = _build_metadata(row, schema_name=self.name, metadata_fields=self.schema.metadata_fields)
        return CanonicalCase(
            case_id=case.id,
            source_file=case.source_file,
            line_number=case.line_number,
            metadata=metadata,
            sections=sections,
            raw=row,
        )


class GenericAdapter:
    name = "generic"

    def to_canonical(self, case: CaseRecord) -> CanonicalCase:
        row = case.raw
        sections: list[CaseSection] = []

        for key in ("question", "options", "answer", "solution", "function", "ground_truth"):
            if key not in row:
                continue
            sections.append(_build_generic_section(key, row.get(key)))

        extras = _extra_payload(
            row,
            handled_keys={section.key for section in sections} | set(DEFAULT_METADATA_FIELDS),
        )
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

        metadata = _build_metadata(
            row,
            schema_name=self.name,
            metadata_fields=DEFAULT_METADATA_FIELDS,
        )
        return CanonicalCase(
            case_id=case.id,
            source_file=case.source_file,
            line_number=case.line_number,
            metadata=metadata,
            sections=sections,
            raw=row,
        )


def adapt_case(case: CaseRecord) -> CanonicalCase:
    adapter = _resolve_adapter(case.dataset_path)
    return adapter.to_canonical(case)


def clear_schema_cache() -> None:
    _resolve_adapter.cache_clear()
    _resolve_schema.cache_clear()
    _load_schema_specs.cache_clear()


@lru_cache(maxsize=128)
def _resolve_adapter(dataset_path: Path) -> ConfiguredAdapter | GenericAdapter:
    schema = _resolve_schema(dataset_path)
    if schema is None:
        return GenericAdapter()
    return ConfiguredAdapter(schema)


@lru_cache(maxsize=128)
def _resolve_schema(dataset_path: Path) -> SchemaSpec | None:
    dataset_path = dataset_path.resolve()
    schemas = _load_schema_specs(dataset_path.parent)

    matches = [schema for schema in schemas if any(fnmatch(dataset_path.name, pattern) for pattern in schema.file_globs)]
    if not matches:
        return None

    matches.sort(key=lambda schema: (schema.priority, max(len(pattern) for pattern in schema.file_globs)), reverse=True)
    return matches[0]


@lru_cache(maxsize=32)
def _load_schema_specs(data_dir: Path) -> tuple[SchemaSpec, ...]:
    data_dir = data_dir.resolve()
    if not data_dir.exists():
        return ()

    schemas: list[SchemaSpec] = []
    for path in sorted(data_dir.glob("*.schema.yaml")):
        schemas.append(_parse_schema_file(path))
    return tuple(schemas)


def _parse_schema_file(path: Path) -> SchemaSpec:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise SchemaConfigError(f"{path} is not valid YAML: {exc}") from exc

    if not isinstance(payload, dict):
        raise SchemaConfigError(f"{path} must be a YAML object")

    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise SchemaConfigError(f"{path} missing required string field: name")

    version = payload.get("version", 1)
    if not isinstance(version, int):
        raise SchemaConfigError(f"{path} field version must be int")

    priority = payload.get("priority", 0)
    if not isinstance(priority, int):
        raise SchemaConfigError(f"{path} field priority must be int")

    match = payload.get("match")
    if not isinstance(match, dict):
        raise SchemaConfigError(f"{path} missing required object field: match")

    file_globs = _parse_file_globs(match.get("file_glob"), path)

    metadata_fields = payload.get("metadata_fields", list(DEFAULT_METADATA_FIELDS))
    if not isinstance(metadata_fields, list) or not all(isinstance(item, str) for item in metadata_fields):
        raise SchemaConfigError(f"{path} field metadata_fields must be list[str]")

    panels_payload = payload.get("panels")
    if not isinstance(panels_payload, list) or not panels_payload:
        raise SchemaConfigError(f"{path} field panels must be a non-empty list")

    panels = tuple(_parse_panel_spec(panel, path) for panel in panels_payload)

    fallback = payload.get("fallback", {})
    if fallback is None:
        fallback = {}
    if not isinstance(fallback, dict):
        raise SchemaConfigError(f"{path} field fallback must be object")

    auto_extra = fallback.get("auto_extra", True)
    if not isinstance(auto_extra, bool):
        raise SchemaConfigError(f"{path} fallback.auto_extra must be bool")

    return SchemaSpec(
        name=name.strip(),
        version=version,
        file_globs=file_globs,
        metadata_fields=tuple(metadata_fields),
        panels=panels,
        auto_extra=auto_extra,
        priority=priority,
    )


def _parse_file_globs(value: object, path: Path) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if isinstance(value, list) and value and all(isinstance(item, str) and item.strip() for item in value):
        return tuple(item.strip() for item in value)
    raise SchemaConfigError(f"{path} match.file_glob must be string or list[str]")


def _parse_panel_spec(payload: object, path: Path) -> PanelSpec:
    if not isinstance(payload, dict):
        raise SchemaConfigError(f"{path} each panel must be object")

    source_key = payload.get("source_key")
    if not isinstance(source_key, str) or not source_key.strip():
        raise SchemaConfigError(f"{path} panel missing required string field: source_key")

    title = payload.get("title", source_key)
    if not isinstance(title, str) or not title.strip():
        raise SchemaConfigError(f"{path} panel title must be string")

    formatter = payload.get("formatter", "text_value")
    if not isinstance(formatter, str) or not formatter.strip():
        raise SchemaConfigError(f"{path} panel formatter must be string")
    if formatter not in FORMATTERS:
        raise SchemaConfigError(f"{path} panel formatter is unsupported: {formatter}")

    kind = _parse_kind(payload.get("kind", "text"), path)

    formatter_args = payload.get("formatter_args", {})
    if not isinstance(formatter_args, dict):
        raise SchemaConfigError(f"{path} panel formatter_args must be object")

    visible_when_missing = payload.get("visible_when_missing", True)
    if not isinstance(visible_when_missing, bool):
        raise SchemaConfigError(f"{path} panel visible_when_missing must be bool")

    return PanelSpec(
        title=title.strip(),
        source_key=source_key.strip(),
        formatter=formatter.strip(),
        kind=kind,
        formatter_args=formatter_args,
        visible_when_missing=visible_when_missing,
    )


def _parse_kind(value: object, path: Path) -> SectionKind:
    if not isinstance(value, str):
        raise SchemaConfigError(f"{path} panel kind must be string")
    try:
        return SectionKind(value.strip())
    except ValueError as exc:
        raise SchemaConfigError(f"{path} panel kind is unsupported: {value}") from exc


def _build_metadata(
    row: dict[str, Any],
    schema_name: str,
    metadata_fields: tuple[str, ...],
) -> dict[str, str]:
    metadata: dict[str, str] = {"schema": schema_name}
    for key in metadata_fields:
        text = _coerce_metadata_text(row.get(key))
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
        if key in handled_keys or key in EXCLUDED_KEYS:
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
    elif key == "options":
        rendered = render_mc_options_text(value)
    elif key == "answer":
        rendered = render_mc_answer_text(value)
    else:
        rendered = render_text_value(value)

    return CaseSection(
        key=key,
        title=key,
        kind=SectionKind.TEXT,
        rendered=rendered,
        raw=value,
    )


def _render_with_formatter(formatter: str, value: object, formatter_args: dict[str, Any]) -> str:
    handler = FORMATTERS[formatter]
    return handler(value, formatter_args)


def _format_text_value(value: object, _: dict[str, Any]) -> str:
    return render_text_value(value)


def _format_conversation(value: object, _: dict[str, Any]) -> str:
    return render_question_text(value)


def _format_function_signature(value: object, _: dict[str, Any]) -> str:
    return render_function_text(value)


def _format_tool_calls(value: object, _: dict[str, Any]) -> str:
    return render_ground_truth_text(value)


def _format_options_list(value: object, _: dict[str, Any]) -> str:
    return render_mc_options_text(value)


def _format_answer_labels(value: object, _: dict[str, Any]) -> str:
    return render_mc_answer_text(value)


def _format_json(value: object, _: dict[str, Any]) -> str:
    return dump_json(value)


FORMATTERS = {
    "text_value": _format_text_value,
    "conversation": _format_conversation,
    "function_signature": _format_function_signature,
    "tool_calls": _format_tool_calls,
    "options_list": _format_options_list,
    "answer_labels": _format_answer_labels,
    "json": _format_json,
}

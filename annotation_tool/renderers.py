from __future__ import annotations

import json
from typing import Any


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def render_question_text(question: object) -> str:
    blocks = _coerce_question_blocks(question)
    if blocks is None:
        return dump_json(question)

    rendered_blocks: list[str] = []
    for block in blocks:
        lines: list[str] = []
        for message in block:
            if isinstance(message, dict):
                role = message.get("role")
                content = message.get("content")
                if isinstance(role, str) and isinstance(content, str):
                    lines.append(f"{role}: {content}")
                    continue
            lines.append(dump_json(message))
        rendered_blocks.append("\n".join(lines))

    return "\n\n".join(rendered_blocks) if rendered_blocks else dump_json(question)


def render_function_text(functions: object) -> str:
    if not isinstance(functions, list):
        return dump_json(functions)

    chunks: list[str] = []
    for function in functions:
        if isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str) and name:
                chunks.append(_render_single_function(function))
                continue
        chunks.append(dump_json(function))

    return "\n\n".join(chunks) if chunks else dump_json(functions)


def render_ground_truth_text(ground_truth: object) -> str:
    if not isinstance(ground_truth, list):
        return dump_json(ground_truth)

    chunks: list[str] = []
    for item in ground_truth:
        if isinstance(item, dict) and len(item) == 1:
            function_name, arguments = next(iter(item.items()))
            if isinstance(function_name, str):
                chunks.append(_render_single_ground_truth_call(function_name, arguments))
                continue
        chunks.append(dump_json(item))

    return "\n\n".join(chunks) if chunks else dump_json(ground_truth)


def _coerce_question_blocks(question: object) -> list[list[object]] | None:
    if not isinstance(question, list):
        return None

    if all(isinstance(item, dict) for item in question):
        return [list(question)]

    blocks: list[list[object]] = []
    for item in question:
        if isinstance(item, list):
            blocks.append(list(item))
        elif isinstance(item, dict):
            blocks.append([item])
        else:
            return None
    return blocks


def _render_single_function(function: dict[str, Any]) -> str:
    name = str(function["name"])
    description = function.get("description")
    parameters = function.get("parameters")
    properties = parameters.get("properties") if isinstance(parameters, dict) else None
    required = parameters.get("required") if isinstance(parameters, dict) else None

    if isinstance(properties, dict):
        ordered_names = _ordered_parameter_names(properties, required)
        lines = [f"# {name}("]
        for param_name in ordered_names:
            prop = properties.get(param_name)
            annotation = _render_parameter_annotation(prop)
            description_text = _render_parameter_description(prop)
            line = f"#     {param_name}: {annotation},"
            if description_text:
                line += f"  # {description_text}"
            lines.append(line)
        lines.append("# )")
    else:
        lines = [f"# {name}()"]

    if isinstance(description, str) and description.strip():
        for line in description.strip().splitlines():
            lines.append(f"# {line.strip()}")

    return "\n".join(lines)


def _ordered_parameter_names(properties: dict[str, Any], required: object) -> list[str]:
    ordered: list[str] = []
    if isinstance(required, list):
        for item in required:
            if isinstance(item, str) and item in properties and item not in ordered:
                ordered.append(item)
    for key in properties:
        if key not in ordered:
            ordered.append(key)
    return ordered


def _render_parameter_annotation(prop: Any) -> str:
    if not isinstance(prop, dict):
        return "Any"
    type_name = prop.get("type")
    if isinstance(type_name, str) and type_name:
        return type_name
    if isinstance(type_name, list):
        rendered = [item for item in type_name if isinstance(item, str) and item]
        if rendered:
            return " | ".join(rendered)
    return "Any"


def _render_parameter_description(prop: Any) -> str:
    if not isinstance(prop, dict):
        return ""
    description = prop.get("description")
    return description.strip() if isinstance(description, str) else ""


def _render_single_ground_truth_call(function_name: str, arguments: object) -> str:
    if not isinstance(arguments, dict):
        return dump_json({function_name: arguments})

    if not arguments:
        return f"{function_name}()"

    lines = [f"{function_name}("]
    for key, value in arguments.items():
        lines.append(f"    {key}={_render_python_literal(_unwrap_singleton_list(value))},")
    lines.append(")")
    return "\n".join(lines)


def _unwrap_singleton_list(value: object) -> object:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _render_python_literal(value: object) -> str:
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, list):
        inner = ", ".join(_render_python_literal(item) for item in value)
        return f"[{inner}]"
    if isinstance(value, dict):
        items = ", ".join(
            f"{json.dumps(str(key), ensure_ascii=False)}: {_render_python_literal(item)}"
            for key, item in value.items()
        )
        return f"{{{items}}}"
    return repr(value)

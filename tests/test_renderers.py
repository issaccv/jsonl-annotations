from __future__ import annotations

import asyncio
import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from annotation_tool.renderers import (
    render_mc_answer_text,
    render_mc_options_text,
    render_function_text,
    render_ground_truth_text,
    render_question_text,
    render_text_value,
)


class RendererTests(unittest.TestCase):
    def test_render_question_text_flattens_messages(self) -> None:
        question = [
            [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "first question"},
            ],
            [
                {"role": "assistant", "content": "first answer"},
                {"role": "user", "content": "second question"},
            ],
        ]

        rendered = render_question_text(question)

        self.assertIn("system: You are helpful.", rendered)
        self.assertIn("user: first question", rendered)
        self.assertIn("assistant: first answer", rendered)
        self.assertIn("\n\n", rendered)

    def test_render_question_text_falls_back_to_json(self) -> None:
        rendered = render_question_text({"unexpected": True})
        self.assertIn('"unexpected": true', rendered)

    def test_render_function_text_builds_commented_signature(self) -> None:
        functions = [
            {
                "name": "inspect_plugin_resource",
                "description": "Inspect the plugin resource.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "plugin_name": {
                            "type": "string",
                            "description": "The name of the Docker plugin.",
                        },
                        "resource_path": {
                            "type": "string",
                            "description": "The file system path.",
                        },
                    },
                    "required": ["resource_path", "plugin_name"],
                },
            }
        ]

        rendered = render_function_text(functions)

        self.assertIn("# inspect_plugin_resource(", rendered)
        self.assertLess(rendered.index("resource_path"), rendered.index("plugin_name"))
        self.assertIn("#     resource_path: string,", rendered)
        self.assertIn("The file system path.", rendered)
        self.assertTrue(rendered.rstrip().endswith("# Inspect the plugin resource."))

    def test_render_ground_truth_text_builds_python_calls(self) -> None:
        ground_truth = [
            {
                "inspect_plugin_resource": {
                    "plugin_name": ["flocker"],
                    "resource_path": ["/run/docker/plugins/flocker.sock"],
                    "replicas": [1, 2],
                }
            }
        ]

        rendered = render_ground_truth_text(ground_truth)

        self.assertIn("inspect_plugin_resource(", rendered)
        self.assertIn('plugin_name="flocker",', rendered)
        self.assertIn('resource_path="/run/docker/plugins/flocker.sock",', rendered)
        self.assertIn("replicas=[1, 2],", rendered)

    def test_render_text_value_handles_scalars_and_collections(self) -> None:
        self.assertEqual(render_text_value("plain"), "plain")
        self.assertEqual(render_text_value(["a", "b"]), "a\nb")
        self.assertIn('"nested": true', render_text_value({"nested": True}))
        self.assertEqual(render_text_value(None), "-")

    def test_render_mc_options_text_formats_keyed_options(self) -> None:
        options = [
            {"key": "A", "text": "first"},
            {"key": "B", "text": "second"},
        ]

        rendered = render_mc_options_text(options)

        self.assertEqual(rendered, "[A] first\n[B] second")

    def test_render_mc_answer_text_formats_answer_labels(self) -> None:
        self.assertEqual(render_mc_answer_text(["A", "C"]), "A, C")
        self.assertEqual(render_mc_answer_text([]), "-")


class AnnotationAppToggleTests(unittest.TestCase):
    def _write_schema(self, data_dir: Path, name: str, content: str) -> None:
        body = textwrap.dedent(content).strip() + "\n"
        (data_dir / f"{name}.schema.yaml").write_text(body, encoding="utf-8")

    def test_toggle_detail_view_switches_modes(self) -> None:
        try:
            from annotation_tool.app import AnnotationApp, DetailViewMode
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                data_dir = root / "data"
                data_dir.mkdir()
                dataset = data_dir / "sample.jsonl"
                row = {
                    "id": "case_0",
                    "question": [[{"role": "user", "content": "first"}]],
                    "function": [{"name": "tool_a"}],
                    "ground_truth": [{"tool_a": {"value": ["1"]}}],
                    "source": "https://example.com/a",
                    "type": "根因定位",
                    "question_type": "Parallel",
                    "language": "en",
                    "iteration_feedback": None,
                }
                dataset.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

                app = AnnotationApp(project_root=root, data_dir=data_dir, initial_dataset=dataset)
                async with app.run_test() as pilot:
                    await pilot.pause()
                    self.assertEqual(app.detail_view_mode, DetailViewMode.NATURAL)
                    await pilot.press("r")
                    await pilot.pause()
                    self.assertEqual(app.detail_view_mode, DetailViewMode.JSON)
                    await pilot.press("r")
                    await pilot.pause()
                    self.assertEqual(app.detail_view_mode, DetailViewMode.NATURAL)

        asyncio.run(scenario())

    def test_qa_dataset_builds_question_answer_solution_sections(self) -> None:
        try:
            from annotation_tool.app import AnnotationApp
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                data_dir = root / "data"
                data_dir.mkdir()
                self._write_schema(
                    data_dir,
                    "qa",
                    """
                    name: qa
                    version: 1
                    match:
                      file_glob: qa*.jsonl
                    metadata_fields: [type, question_type, language, source, output_requirement]
                    panels:
                      - title: question
                        source_key: question
                        formatter: text_value
                        kind: text
                      - title: answer
                        source_key: answer
                        formatter: text_value
                        kind: text
                      - title: solution
                        source_key: solution
                        formatter: text_value
                        kind: text
                    fallback:
                      auto_extra: true
                    """,
                )
                dataset = data_dir / "qa.jsonl"
                row = {
                    "id": "qa_0",
                    "question": "在配置之前需要什么？",
                    "answer": "先申请密钥。",
                    "output_requirement": "自然语言说明",
                    "solution": "提取前置步骤。",
                    "source": "https://example.com/doc",
                    "type": "用户手册说明",
                    "question_type": "qa",
                    "language": "zh",
                }
                dataset.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

                app = AnnotationApp(project_root=root, data_dir=data_dir, initial_dataset=dataset)
                async with app.run_test() as pilot:
                    await pilot.pause()
                    canonical = app.current_canonical_case
                    self.assertIsNotNone(canonical)
                    assert canonical is not None
                    self.assertEqual(
                        [section.key for section in canonical.sections[:3]],
                        ["question", "answer", "solution"],
                    )
                    self.assertEqual(canonical.metadata.get("output_requirement"), "自然语言说明")

        asyncio.run(scenario())

    def test_mc_dataset_builds_options_panel(self) -> None:
        try:
            from annotation_tool.app import AnnotationApp
        except ModuleNotFoundError as exc:
            self.skipTest(str(exc))

        async def scenario() -> None:
            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                data_dir = root / "data"
                data_dir.mkdir()
                self._write_schema(
                    data_dir,
                    "mc",
                    """
                    name: mc
                    version: 1
                    match:
                      file_glob: mc*.jsonl
                    metadata_fields: [type, question_type, language, source]
                    panels:
                      - title: question
                        source_key: question
                        formatter: text_value
                        kind: text
                      - title: options
                        source_key: options
                        formatter: options_list
                        kind: text
                      - title: answer
                        source_key: answer
                        formatter: answer_labels
                        kind: text
                      - title: solution
                        source_key: solution
                        formatter: text_value
                        kind: text
                    fallback:
                      auto_extra: true
                    """,
                )
                dataset = data_dir / "mc.jsonl"
                row = {
                    "id": "mc_0",
                    "question": "Which actions are valid?",
                    "options": [
                        {"key": "A", "text": "First option"},
                        {"key": "B", "text": "Second option"},
                    ],
                    "answer": ["A", "B"],
                    "solution": "Choose the documented options.",
                    "source": "https://example.com/mc",
                    "type": "用户手册说明",
                    "question_type": "mc",
                    "language": "en",
                }
                dataset.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

                app = AnnotationApp(project_root=root, data_dir=data_dir, initial_dataset=dataset)
                async with app.run_test() as pilot:
                    await pilot.pause()
                    canonical = app.current_canonical_case
                    self.assertIsNotNone(canonical)
                    assert canonical is not None
                    self.assertEqual(
                        [section.key for section in canonical.sections[:4]],
                        ["question", "options", "answer", "solution"],
                    )
                    self.assertEqual(canonical.sections[1].rendered, "[A] First option\n[B] Second option")
                    self.assertEqual(canonical.sections[2].rendered, "A, B")

        asyncio.run(scenario())


if __name__ == "__main__":
    unittest.main()

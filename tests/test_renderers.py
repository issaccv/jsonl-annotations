from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path

from annotation_tool.renderers import (
    render_function_text,
    render_ground_truth_text,
    render_question_text,
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


class AnnotationAppToggleTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()

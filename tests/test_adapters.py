from __future__ import annotations

import unittest
from pathlib import Path

from annotation_tool.adapters import adapt_case
from annotation_tool.models import CaseRecord, SectionKind


class AdapterTests(unittest.TestCase):
    def _case(self, row: dict[str, object], line_number: int = 1) -> CaseRecord:
        return CaseRecord(
            dataset_path=Path("data/sample.jsonl"),
            source_file="data/sample.jsonl",
            line_number=line_number,
            raw=row,
        )

    def test_parallel_adapter_builds_function_call_sections(self) -> None:
        row = {
            "id": "parallel_1",
            "question": [[{"role": "user", "content": "check plugin"}]],
            "function": [{"name": "inspect_plugin_resource"}],
            "ground_truth": [{"inspect_plugin_resource": {"plugin_name": ["demo"]}}],
            "source": "https://example.com/parallel",
            "question_type": "Parallel",
            "language": "en",
            "type": "可观测性",
        }

        canonical = adapt_case(self._case(row))

        self.assertEqual(canonical.metadata["schema"], "parallel")
        self.assertEqual([section.key for section in canonical.sections], ["question", "function", "ground_truth"])
        self.assertEqual(canonical.sections[1].kind, SectionKind.CODE)
        self.assertIn("user: check plugin", canonical.sections[0].rendered)

    def test_qa_adapter_uses_three_panel_layout(self) -> None:
        row = {
            "id": "qa_1",
            "question": "配置验证码之前需要做什么？",
            "answer": "先获取密钥。",
            "solution": "从配置流程抽取前置依赖。",
            "output_requirement": "自然语言说明",
            "notes": {"source_rank": 1},
            "source": "https://example.com/qa",
            "question_type": "qa",
            "language": "zh",
            "type": "用户手册说明",
        }

        canonical = adapt_case(self._case(row))

        self.assertEqual(canonical.metadata["schema"], "qa")
        self.assertEqual([section.key for section in canonical.sections[:3]], ["question", "answer", "solution"])
        self.assertEqual(canonical.metadata["output_requirement"], "自然语言说明")
        self.assertEqual(canonical.sections[3].key, "extra")
        self.assertEqual(canonical.sections[3].kind, SectionKind.JSON)

    def test_mc_adapter_renders_options_and_answer(self) -> None:
        row = {
            "id": "mc_1",
            "question": "Choose all valid actions.",
            "options": [
                {"key": "A", "text": "Enable persistence"},
                {"key": "B", "text": "Skip approval step"},
            ],
            "answer": ["A"],
            "solution": "A follows the deployment guide.",
            "source": "https://example.com/mc",
            "question_type": "mc",
            "language": "en",
            "type": "用户手册说明",
        }

        canonical = adapt_case(self._case(row))

        self.assertEqual(canonical.metadata["schema"], "mc")
        self.assertEqual(
            [section.key for section in canonical.sections[:4]],
            ["question", "options", "answer", "solution"],
        )
        self.assertEqual(canonical.sections[1].rendered, "[A] Enable persistence\n[B] Skip approval step")
        self.assertEqual(canonical.sections[2].rendered, "A")

    def test_generic_adapter_falls_back_to_raw_when_schema_is_unknown(self) -> None:
        row = {
            "id": "unknown_1",
            "prompt": "hello",
            "metadata": {"domain": "docs"},
        }

        canonical = adapt_case(self._case(row))

        self.assertEqual(canonical.metadata["schema"], "generic")
        self.assertEqual(len(canonical.sections), 1)
        self.assertEqual(canonical.sections[0].key, "extra")
        self.assertEqual(canonical.sections[0].kind, SectionKind.JSON)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from annotation_tool.storage import FeedbackStore, discover_datasets, export_path_for, load_cases


class FeedbackStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        self.dataset_path = self.data_dir / "sample.jsonl"
        rows = [
            {
                "id": "case_0",
                "question": [[{"role": "user", "content": "first"}]],
                "function": [{"name": "tool_a"}],
                "ground_truth": [{"tool_a": {"value": ["1"]}}],
                "source": "https://example.com/a",
                "type": "根因定位",
                "question_type": "Parallel",
                "language": "en",
                "iteration_feedback": None,
            },
            {
                "id": "case_1",
                "question": [[{"role": "user", "content": "second"}]],
                "function": [{"name": "tool_b"}],
                "ground_truth": [{"tool_b": {"value": ["2"]}}],
                "source": "https://example.com/b",
                "type": "故障修复",
                "question_type": "Simple",
                "language": "zh",
                "iteration_feedback": None,
            },
        ]
        with self.dataset_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_feedback_round_trip_uses_latest_entry(self) -> None:
        store = FeedbackStore(self.root)
        cases = load_cases(self.dataset_path, self.root)

        store.write_feedback(self.dataset_path, cases[0], "好")
        store.write_feedback(self.dataset_path, cases[0], "不好：函数选择错误")
        feedback_map = store.load_feedback_map(self.dataset_path)

        self.assertEqual(
            store.resolve_feedback(cases[0], feedback_map),
            "不好：函数选择错误",
        )
        self.assertIsNone(store.resolve_feedback(cases[1], feedback_map))

    def test_export_merges_sidecar_feedback(self) -> None:
        store = FeedbackStore(self.root)
        cases = load_cases(self.dataset_path, self.root)

        store.write_feedback(self.dataset_path, cases[0], "好")
        store.write_feedback(self.dataset_path, cases[1], "不好：参数抽取不完整")

        output_path = store.export_dataset(self.dataset_path)
        self.assertEqual(output_path, export_path_for(self.dataset_path))
        self.assertTrue(output_path.exists())

        rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(rows[0]["iteration_feedback"], "好")
        self.assertEqual(rows[1]["iteration_feedback"], "不好：参数抽取不完整")

    def test_discover_and_summary_ignore_exported_files(self) -> None:
        store = FeedbackStore(self.root)
        cases = load_cases(self.dataset_path, self.root)
        store.write_feedback(self.dataset_path, cases[0], "好")

        exported = self.data_dir / "sample.annotated.jsonl"
        exported.write_text("", encoding="utf-8")

        self.assertEqual(discover_datasets(self.data_dir), [self.dataset_path.resolve()])

        summary = store.summarize_dataset(self.dataset_path)
        self.assertEqual(summary.total_cases, 2)
        self.assertEqual(summary.annotated_cases, 1)
        self.assertEqual(summary.bad_cases, 0)

    def test_missing_id_falls_back_to_line_number_key(self) -> None:
        dataset_path = self.data_dir / "missing_id.jsonl"
        row = {
            "question": "plain question",
            "answer": "plain answer",
            "solution": "plain solution",
            "source": "https://example.com/missing-id",
            "type": "用户手册说明",
            "question_type": "qa",
            "language": "zh",
        }
        dataset_path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

        store = FeedbackStore(self.root)
        cases = load_cases(dataset_path, self.root)
        self.assertEqual(cases[0].id, "line-1")

        store.write_feedback(dataset_path, cases[0], "好")
        feedback_map = store.load_feedback_map(dataset_path)
        self.assertEqual(store.resolve_feedback(cases[0], feedback_map), "好")


if __name__ == "__main__":
    unittest.main()

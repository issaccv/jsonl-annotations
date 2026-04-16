from __future__ import annotations

import argparse
from pathlib import Path

from .storage import FeedbackStore, discover_datasets, find_project_root, resolve_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="JSONL 命令行数据标注工具")
    parser.add_argument("--data-dir", default="data", help="数据目录，默认是 ./data")
    parser.add_argument("--file", help="直接打开指定的 jsonl 文件")
    subparsers = parser.add_subparsers(dest="command")

    export_parser = subparsers.add_parser("export", help="把 sidecar 合并成新的 annotated jsonl")
    export_parser.add_argument("--file", required=True, help="需要导出的 jsonl 文件")

    return parser


def _resolve_dataset_argument(file_arg: str | None, data_dir_arg: str, project_root: Path) -> tuple[Path | None, Path]:
    cwd = Path.cwd()
    data_dir = resolve_path(data_dir_arg, cwd if Path(data_dir_arg).is_absolute() else project_root)
    dataset_path = None
    if file_arg:
        dataset_path = resolve_path(file_arg, cwd)
    return dataset_path, data_dir


def launch_tui(project_root: Path, data_dir: Path, dataset_path: Path | None) -> int:
    try:
        from .app import AnnotationApp
    except ModuleNotFoundError as exc:
        if exc.name == "textual":
            raise SystemExit("缺少 `textual` 依赖，先运行 `uv sync`。") from exc
        raise

    app = AnnotationApp(project_root=project_root, data_dir=data_dir, initial_dataset=dataset_path)
    app.run()
    return 0


def run_export(project_root: Path, dataset_path: Path) -> int:
    if not dataset_path.exists():
        raise SystemExit(f"找不到文件: {dataset_path}")
    store = FeedbackStore(project_root)
    output_path = store.export_dataset(dataset_path)
    print(output_path)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    project_root = find_project_root()
    dataset_path, data_dir = _resolve_dataset_argument(args.file, args.data_dir, project_root)

    if args.command == "export":
        return run_export(project_root, dataset_path)

    if dataset_path is None:
        datasets = discover_datasets(data_dir)
        if not datasets:
            raise SystemExit(f"{data_dir} 下没有可标注的 jsonl 文件。")
    else:
        if not dataset_path.exists():
            raise SystemExit(f"找不到文件: {dataset_path}")

    return launch_tui(project_root, data_dir, dataset_path)

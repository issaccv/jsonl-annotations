from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from rich.console import Group
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, ListItem, ListView, Static

from .models import CaseRecord, DatasetSummary, FeedbackFilter, FeedbackRecord
from .renderers import render_function_text, render_ground_truth_text, render_question_text
from .storage import DataError, FeedbackStore, QUESTION_TYPE_HINTS, discover_datasets, load_cases


class DetailViewMode(str, Enum):
    NATURAL = "natural"
    JSON = "json"

    @property
    def label(self) -> str:
        return {
            self.NATURAL: "自然",
            self.JSON: "原始 JSON",
        }[self]


class TextInputScreen(ModalScreen[str | None]):
    CSS = """
    TextInputScreen {
        align: center middle;
        background: $background 70%;
    }

    .dialog {
        width: 80;
        max-width: 90;
        height: auto;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }

    .dialog-title {
        margin-bottom: 1;
        text-style: bold;
    }

    .dialog-buttons {
        height: auto;
        margin-top: 1;
        align-horizontal: right;
    }

    .dialog-buttons Button {
        margin-left: 1;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "取消")]

    def __init__(self, title: str, placeholder: str = "", value: str = ""):
        super().__init__()
        self.title = title
        self.placeholder = placeholder
        self.value = value

    def compose(self) -> ComposeResult:
        with Container(classes="dialog"):
            yield Label(self.title, classes="dialog-title")
            yield Input(value=self.value, placeholder=self.placeholder, id="text_input")
            with Container(classes="dialog-buttons"):
                yield Button("取消", id="cancel")
                yield Button("保存", id="submit", variant="primary")

    def on_mount(self) -> None:
        self.query_one(Input).focus()

    @on(Button.Pressed, "#submit")
    def handle_submit(self) -> None:
        self.dismiss(self.query_one(Input).value)

    @on(Button.Pressed, "#cancel")
    def handle_cancel(self) -> None:
        self.dismiss(None)

    @on(Input.Submitted, "#text_input")
    def submit_from_input(self) -> None:
        self.dismiss(self.query_one(Input).value)

    def action_cancel(self) -> None:
        self.dismiss(None)


class MessageScreen(ModalScreen[None]):
    CSS = """
    MessageScreen {
        align: center middle;
        background: $background 70%;
    }

    .message-dialog {
        width: 100;
        max-width: 95;
        height: 70%;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }

    .message-title {
        margin-bottom: 1;
        text-style: bold;
    }

    .message-scroll {
        height: 1fr;
        border: round $panel;
        padding: 0 1;
    }

    .message-close {
        margin-top: 1;
        width: 12;
        dock: bottom;
    }
    """

    BINDINGS = [Binding("escape", "close", "关闭")]

    def __init__(self, title: str, body: str):
        super().__init__()
        self.title = title
        self.body = body

    def compose(self) -> ComposeResult:
        with Container(classes="message-dialog"):
            yield Label(self.title, classes="message-title")
            with VerticalScroll(classes="message-scroll"):
                yield Static(self.body)
            yield Button("关闭", id="close", classes="message-close", variant="primary")

    @on(Button.Pressed, "#close")
    def close_screen(self) -> None:
        self.dismiss(None)

    def action_close(self) -> None:
        self.dismiss(None)


class ChoiceScreen(ModalScreen[str | None]):
    CSS = """
    ChoiceScreen {
        align: center middle;
        background: $background 70%;
    }

    .choice-dialog {
        width: 90;
        max-width: 95;
        height: 70%;
        background: $surface;
        border: round $accent;
        padding: 1 2;
    }

    .choice-title {
        margin-bottom: 1;
        text-style: bold;
    }

    .choice-list {
        height: 1fr;
        border: round $panel;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "取消")]

    def __init__(self, title: str, options: list[tuple[str, str]]):
        super().__init__()
        self.title = title
        self.options = options

    def compose(self) -> ComposeResult:
        with Container(classes="choice-dialog"):
            yield Label(self.title, classes="choice-title")
            yield ListView(
                *[ListItem(Label(label), name=value) for value, label in self.options],
                classes="choice-list",
                id="choice_list",
            )

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    @on(ListView.Selected, "#choice_list")
    def handle_selection(self, event: ListView.Selected) -> None:
        item = event.item
        self.dismiss(item.name or None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class DatasetPickerScreen(ModalScreen[Path | None]):
    CSS = ChoiceScreen.CSS

    BINDINGS = [Binding("escape", "cancel", "退出")]

    def __init__(self, summaries: list[DatasetSummary]):
        super().__init__()
        self.summaries = summaries

    def compose(self) -> ComposeResult:
        with Container(classes="choice-dialog"):
            yield Label("选择要标注的数据集", classes="choice-title")
            yield ListView(
                *[
                    ListItem(
                        Label(
                            f"{summary.path.name}  已标 {summary.annotated_cases}/{summary.total_cases}  "
                            f"不好 {summary.bad_cases}  完成 {summary.completion_ratio:.0%}"
                        ),
                        name=str(summary.path),
                    )
                    for summary in self.summaries
                ],
                classes="choice-list",
                id="dataset_list",
            )

    def on_mount(self) -> None:
        self.query_one(ListView).focus()

    @on(ListView.Selected, "#dataset_list")
    def handle_selection(self, event: ListView.Selected) -> None:
        item = event.item
        self.dismiss(Path(item.name) if item.name else None)

    def action_cancel(self) -> None:
        self.dismiss(None)


class AnnotationApp(App[None]):
    CSS = """
    Screen {
        layout: vertical;
        background: $background;
    }

    #summary {
        height: 3;
        padding: 0 1;
        background: $surface;
        color: $text;
        border-bottom: solid $panel;
    }

    #body_scroll {
        height: 1fr;
        padding: 0 1 1 1;
    }

    #body {
        height: auto;
        padding: 1 0;
    }

    #message {
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
    }
    """

    BINDINGS = [
        Binding("g", "mark_good", "标记好"),
        Binding("b", "mark_bad", "标记不好"),
        Binding("e", "edit_reason", "编辑原因"),
        Binding("n", "next_case", "下一条"),
        Binding("p", "prev_case", "上一条"),
        Binding("j", "jump_to_case", "跳转"),
        Binding("f", "change_filter", "过滤"),
        Binding("r", "toggle_detail_view", "切换视图"),
        Binding("o", "show_source", "来源"),
        Binding("q", "quit", "退出"),
    ]

    TITLE = "JSONL 标注工具"

    def __init__(self, project_root: Path, data_dir: Path, initial_dataset: Path | None = None):
        super().__init__()
        self.project_root = project_root.resolve()
        self.data_dir = data_dir.resolve()
        self.initial_dataset = initial_dataset.resolve() if initial_dataset else None
        self.store = FeedbackStore(self.project_root)
        self.summaries: list[DatasetSummary] = []
        self.dataset_path: Path | None = None
        self.cases: list[CaseRecord] = []
        self.feedback_map: dict[tuple[str, str], FeedbackRecord] = {}
        self.filter_mode = FeedbackFilter.ALL
        self.detail_view_mode = DetailViewMode.NATURAL
        self.filtered_indices: list[int] = []
        self.current_index = 0
        self.status_message = "准备就绪"

    def compose(self) -> ComposeResult:
        yield Header(show_clock=False)
        yield Static(id="summary")
        with VerticalScroll(id="body_scroll"):
            yield Static(id="body")
        yield Static(id="message")
        yield Footer()

    def on_mount(self) -> None:
        try:
            self.summaries = self._load_dataset_summaries()
        except DataError as exc:
            raise SystemExit(str(exc)) from exc

        if self.initial_dataset:
            self._open_dataset(self.initial_dataset)
            return

        if not self.summaries:
            raise SystemExit(f"{self.data_dir} 下没有可标注的 jsonl 文件。")

        if len(self.summaries) == 1:
            self._open_dataset(self.summaries[0].path)
            return

        self.push_screen(DatasetPickerScreen(self.summaries), self._handle_dataset_selection)

    def action_mark_good(self) -> None:
        case = self.current_case
        if case is None:
            self._set_status("当前过滤结果里没有 case")
            return
        self._save_feedback(case, "好", auto_advance=True)

    def action_mark_bad(self) -> None:
        case = self.current_case
        if case is None:
            self._set_status("当前过滤结果里没有 case")
            return
        self.push_screen(
            TextInputScreen(
                title=f"为 {case.id} 填写不好的原因",
                placeholder="例如：参数抽取不完整、函数选择错误",
                value=self._existing_reason(case),
            ),
            self._handle_bad_reason,
        )

    def action_edit_reason(self) -> None:
        case = self.current_case
        if case is None:
            self._set_status("当前过滤结果里没有 case")
            return
        self.push_screen(
            TextInputScreen(
                title=f"编辑 {case.id} 的原因",
                placeholder="输入原因后会保存为“不好：原因”",
                value=self._existing_reason(case),
            ),
            self._handle_bad_reason,
        )

    def action_next_case(self) -> None:
        if not self.filtered_indices:
            self._set_status("当前过滤结果里没有 case")
            return
        position = self.filtered_indices.index(self.current_index)
        if position >= len(self.filtered_indices) - 1:
            self._set_status("已经到最后一条")
            return
        self.current_index = self.filtered_indices[position + 1]
        self._refresh_view()

    def action_prev_case(self) -> None:
        if not self.filtered_indices:
            self._set_status("当前过滤结果里没有 case")
            return
        position = self.filtered_indices.index(self.current_index)
        if position == 0:
            self._set_status("已经到第一条")
            return
        self.current_index = self.filtered_indices[position - 1]
        self._refresh_view()

    def action_jump_to_case(self) -> None:
        if not self.cases:
            self._set_status("当前没有可跳转的数据")
            return
        self.push_screen(
            TextInputScreen(
                title="跳转到 ID 或行号",
                placeholder="例如 parallel_12 或 13",
            ),
            self._handle_jump,
        )

    def action_change_filter(self) -> None:
        options = [(mode.value, mode.label) for mode in FeedbackFilter]
        self.push_screen(ChoiceScreen("选择过滤视图", options), self._handle_filter_change)

    def action_show_source(self) -> None:
        case = self.current_case
        if case is None:
            self._set_status("当前没有来源可查看")
            return
        hint = QUESTION_TYPE_HINTS.get(case.question_type, "")
        body = f"case: {case.id}\n\nsource:\n{case.source}"
        if hint:
            body += f"\n\nBFCL question_type 说明:\n{case.question_type} - {hint}"
        self.push_screen(MessageScreen("原始来源", body))

    def action_toggle_detail_view(self) -> None:
        if self.detail_view_mode is DetailViewMode.NATURAL:
            self.detail_view_mode = DetailViewMode.JSON
        else:
            self.detail_view_mode = DetailViewMode.NATURAL
        self._set_status(f"详情视图已切换为 {self.detail_view_mode.label}")
        self._refresh_view()

    def _handle_dataset_selection(self, selected: Path | None) -> None:
        if selected is None:
            self.exit()
            return
        self._open_dataset(selected)

    def _handle_bad_reason(self, value: str | None) -> None:
        if value is None:
            self._set_status("已取消")
            return
        reason = value.strip()
        if not reason:
            self._set_status("原因不能为空")
            return
        case = self.current_case
        if case is None:
            self._set_status("当前没有可标注的 case")
            return
        self._save_feedback(case, f"不好：{reason}", auto_advance=True)

    def _handle_jump(self, value: str | None) -> None:
        if value is None:
            self._set_status("已取消跳转")
            return
        target = value.strip()
        if not target:
            self._set_status("请输入 ID 或行号")
            return
        found_index = None
        for index, case in enumerate(self.cases):
            if case.id == target:
                found_index = index
                break
        if found_index is None and target.isdigit():
            line_number = int(target)
            if 1 <= line_number <= len(self.cases):
                found_index = line_number - 1
        if found_index is None:
            self._set_status(f"找不到: {target}")
            return
        if self.filter_mode != FeedbackFilter.ALL and found_index not in self.filtered_indices:
            self.filter_mode = FeedbackFilter.ALL
            self._rebuild_filtered_indices(preferred_index=found_index)
            self._set_status("已切回全部视图后跳转")
        self.current_index = found_index
        self._refresh_view()

    def _handle_filter_change(self, value: str | None) -> None:
        if value is None:
            self._set_status("已取消过滤切换")
            return
        self.filter_mode = FeedbackFilter(value)
        self._rebuild_filtered_indices(preferred_index=self.current_index)
        if self.filtered_indices:
            self._set_status(f"过滤已切换为 {self.filter_mode.label}")
        else:
            self._set_status(f"{self.filter_mode.label} 里还没有 case")
        self._refresh_view()

    def _load_dataset_summaries(self) -> list[DatasetSummary]:
        return [self.store.summarize_dataset(path) for path in discover_datasets(self.data_dir)]

    def _open_dataset(self, dataset_path: Path) -> None:
        try:
            self.dataset_path = dataset_path.resolve()
            self.cases = load_cases(self.dataset_path, self.project_root)
            self.feedback_map = self.store.load_feedback_map(self.dataset_path)
        except DataError as exc:
            raise SystemExit(str(exc)) from exc

        if not self.cases:
            raise SystemExit(f"{self.dataset_path} 里没有可标注的记录。")

        self.filter_mode = FeedbackFilter.ALL
        self.current_index = self._first_pending_index()
        self._rebuild_filtered_indices(preferred_index=self.current_index)
        self.sub_title = self.dataset_path.name
        self._set_status("数据集已加载")
        self._refresh_view()

    def _save_feedback(self, case: CaseRecord, feedback: str, auto_advance: bool) -> None:
        if self.dataset_path is None:
            self._set_status("当前没有打开数据集")
            return
        previous_index = self.current_index
        record = self.store.write_feedback(self.dataset_path, case, feedback)
        self.feedback_map[record.lookup_key] = record
        self.summaries = self._load_dataset_summaries()
        next_index = self._next_index_after_save(previous_index) if auto_advance else previous_index
        self._rebuild_filtered_indices(preferred_index=next_index)
        self._set_status(f"已保存 {case.id}: {feedback}")
        self._refresh_view()

    def _next_index_after_save(self, previous_index: int) -> int:
        if self.filter_mode is FeedbackFilter.PENDING:
            for index in range(previous_index + 1, len(self.cases)):
                if self._feedback_for_case(self.cases[index]) is None:
                    return index
            for index, case in enumerate(self.cases):
                if self._feedback_for_case(case) is None:
                    return index
            return previous_index

        matching_indices = [
            index for index, case in enumerate(self.cases) if self._matches_filter(case)
        ]
        for index in matching_indices:
            if index > previous_index:
                return index
        if previous_index in matching_indices:
            return previous_index
        if matching_indices:
            return matching_indices[-1]
        return previous_index

    def _rebuild_filtered_indices(self, preferred_index: int | None = None) -> None:
        self.filtered_indices = [
            index for index, case in enumerate(self.cases) if self._matches_filter(case)
        ]
        if not self.filtered_indices:
            self.current_index = 0
            return
        if preferred_index in self.filtered_indices:
            self.current_index = preferred_index
            return
        nearest = min(self.filtered_indices, key=lambda index: abs(index - (preferred_index or 0)))
        self.current_index = nearest

    def _matches_filter(self, case: CaseRecord) -> bool:
        feedback = self._feedback_for_case(case)
        if self.filter_mode is FeedbackFilter.ALL:
            return True
        if self.filter_mode is FeedbackFilter.PENDING:
            return feedback is None
        return bool(feedback and feedback.startswith("不好"))

    def _feedback_for_case(self, case: CaseRecord) -> str | None:
        return self.store.resolve_feedback(case, self.feedback_map)

    def _existing_reason(self, case: CaseRecord) -> str:
        feedback = self._feedback_for_case(case)
        if feedback and feedback.startswith("不好："):
            return feedback.split("：", maxsplit=1)[1]
        return ""

    def _first_pending_index(self) -> int:
        for index, case in enumerate(self.cases):
            if self._feedback_for_case(case) is None:
                return index
        return 0

    @property
    def current_case(self) -> CaseRecord | None:
        if (
            not self.cases
            or not self.filtered_indices
            or self.current_index not in self.filtered_indices
        ):
            return None
        return self.cases[self.current_index]

    def _render_summary(self) -> Text:
        dataset_name = self.dataset_path.name if self.dataset_path else "未选择"
        total = len(self.cases)
        done = sum(1 for case in self.cases if self._feedback_for_case(case))
        bad = sum(
            1
            for case in self.cases
            if (feedback := self._feedback_for_case(case)) and feedback.startswith("不好")
        )
        current_position = 0
        if self.filtered_indices and self.current_index in self.filtered_indices:
            current_position = self.filtered_indices.index(self.current_index) + 1
        filtered_total = len(self.filtered_indices)
        return Text.from_markup(
            "\n".join(
                [
                    f"[b]{dataset_name}[/b]  进度 {done}/{total}  不好 {bad}  过滤 {self.filter_mode.label}",
                    f"当前 {current_position}/{filtered_total}  绝对行号 {self.current_index + 1 if self.cases else 0}  视图 {self.detail_view_mode.label}",
                ]
            )
        )

    def _render_metadata_table(self, case: CaseRecord) -> Table:
        table = Table.grid(expand=True)
        table.add_column(style="bold cyan", ratio=1)
        table.add_column(ratio=4)
        table.add_row("id", case.id)
        table.add_row("type", case.type_label or "-")
        table.add_row("question_type", case.question_type or "-")
        hint = QUESTION_TYPE_HINTS.get(case.question_type, "")
        if hint:
            table.add_row("BFCL hint", hint)
        table.add_row("language", case.language or "-")
        table.add_row("source", case.source or "-")
        feedback = self._feedback_for_case(case) or "未标注"
        table.add_row("iteration_feedback", feedback)
        return table

    def _render_json_panel(self, title: str, payload: object) -> Panel:
        content = json.dumps(payload, ensure_ascii=False, indent=2)
        syntax = Syntax(content, "json", word_wrap=True, line_numbers=False)
        return Panel(syntax, title=title, border_style="blue")

    def _render_text_panel(self, title: str, content: str) -> Panel:
        return Panel(Text(content), title=title, border_style="blue")

    def _render_code_panel(self, title: str, content: str) -> Panel:
        syntax = Syntax(content, "python", word_wrap=True, line_numbers=False)
        return Panel(syntax, title=title, border_style="blue")

    def _render_body(self) -> object:
        case = self.current_case
        if case is None:
            return Panel(
                Text(f"{self.filter_mode.label} 里还没有可展示的 case", justify="center"),
                title="空结果",
                border_style="yellow",
            )

        return Group(
            Panel(self._render_metadata_table(case), title=f"Case {case.line_number}/{len(self.cases)}", border_style="green"),
            self._render_question_panel(case),
            self._render_function_panel(case),
            self._render_ground_truth_panel(case),
        )

    def _render_question_panel(self, case: CaseRecord) -> Panel:
        if self.detail_view_mode is DetailViewMode.JSON:
            return self._render_json_panel("question", case.question)
        return self._render_text_panel("question", render_question_text(case.question))

    def _render_function_panel(self, case: CaseRecord) -> Panel:
        if self.detail_view_mode is DetailViewMode.JSON:
            return self._render_json_panel("function", case.function)
        return self._render_code_panel("function", render_function_text(case.function))

    def _render_ground_truth_panel(self, case: CaseRecord) -> Panel:
        if self.detail_view_mode is DetailViewMode.JSON:
            return self._render_json_panel("ground_truth", case.ground_truth)
        return self._render_code_panel("ground_truth", render_ground_truth_text(case.ground_truth))

    def _refresh_view(self) -> None:
        self.query_one("#summary", Static).update(self._render_summary())
        self.query_one("#body", Static).update(self._render_body())
        self.query_one("#message", Static).update(self.status_message)

    def _set_status(self, message: str) -> None:
        self.status_message = message
        if self.is_mounted:
            self.query_one("#message", Static).update(message)

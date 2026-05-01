from __future__ import annotations

import asyncio
from pathlib import Path

from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Button, Input, Select, Static, TextArea

from models import IncidentMatch, Ticket
from streaming_assistant import StreamingSupportAssistant, SupportStateSummary


class ChatBubble(Static):
    def __init__(self, speaker: str, prefix: str, label: str, align: str, style_class: str) -> None:
        super().__init__("", classes=f"chat-bubble {style_class}")
        self.speaker = speaker
        self.prefix = prefix
        self.label = label
        self.align = align
        self.header_text = ""
        self.message_text = ""
        self.footer_text = ""

    def set_header(self, message: str) -> None:
        self.header_text = message
        self.update(self._render_text())

    def set_message(self, message: str) -> None:
        self.message_text = message
        self.update(self._render_text())

    def append_message(self, token: str) -> None:
        self.message_text += token
        self.update(self._render_text())

    def set_footer(self, message: str) -> None:
        self.footer_text = message
        self.update(self._render_text())

    def _render_text(self) -> Text:
        label_style = "bold bright_blue" if self.speaker == "user" else "bold bright_green"
        if "error" in self.classes:
            label_style = "bold red"
        body_style = "white" if "error" not in self.classes else "red"
        meta_style = "bright_black" if "error" not in self.classes else "red"

        text = Text(justify=self.align)
        text.append(f"{self.prefix} ", style=label_style)
        text.append(f"{self.label}\n", style=label_style)
        if self.header_text:
            text.append(self.header_text.rstrip() + "\n", style=meta_style)
        if self.message_text:
            text.append(self.message_text, style=body_style)
        if self.footer_text:
            if self.message_text:
                text.append("\n", style=body_style)
            text.append(self.footer_text.rstrip(), style=meta_style)
        return text


class ConversationHistory(VerticalScroll):
    def add_bubble(self, bubble: ChatBubble) -> None:
        self.mount(bubble)
        self.scroll_end(animate=False)


class AnalysisReady(Message):
    def __init__(self, summary: SupportStateSummary) -> None:
        self.summary = summary
        super().__init__()


class StreamingUpdate(Message):
    def __init__(self, token: str) -> None:
        self.token = token
        super().__init__()


class RecommendationComplete(Message):
    def __init__(self, summary: SupportStateSummary) -> None:
        self.summary = summary
        super().__init__()


class SimilarIncidentsReady(Message):
    def __init__(self, incidents: list[IncidentMatch]) -> None:
        self.incidents = incidents
        super().__init__()


class CsvIngested(Message):
    def __init__(self, output_path: str, count: int) -> None:
        self.output_path = output_path
        self.count = count
        super().__init__()


class AsyncFailed(Message):
    def __init__(self, error_text: str) -> None:
        self.error_text = error_text
        super().__init__()


class SupportAssistantApp(App[None]):
    CSS = """
    Screen {
        background: #0b1020;
        color: white;
    }

    #app-shell {
        height: 100%;
        layout: vertical;
        padding: 1;
        background: #0b1020;
    }

    #title-bar {
        height: 3;
        content-align: center middle;
        background: #111827;
        color: #dbeafe;
        text-style: bold;
        border: round #1f2937;
        margin-bottom: 1;
    }

    #history {
        height: 1fr;
        border: round #1f2937;
        background: #0f172a;
        padding: 1;
        min-height: 12;
    }

    #composer-panel {
        height: auto;
        border: round #1f2937;
        background: #111827;
        padding: 1;
        margin-top: 1;
    }

    .composer-row {
        height: auto;
        margin-bottom: 1;
    }

    #context-row {
        height: auto;
    }

    #company-select {
        width: 20;
    }

    #subject-input, #csv-path-input {
        width: 1fr;
        background: #0f172a;
        color: white;
        border: round #334155;
    }

    #issue-input {
        height: 5;
        background: #0f172a;
        color: white;
        border: round #334155;
    }

    #action-row-primary, #action-row-secondary {
        height: 3;
        margin-top: 0;
    }

    .action-button {
        margin-right: 1;
    }

    .compact-button {
        min-width: 12;
    }

    .chat-bubble {
        width: 100%;
        padding: 0 1;
        margin-bottom: 1;
    }

    .user {
        background: #172554;
        color: #dbeafe;
        content-align: right middle;
        border: round #1d4ed8;
    }

    .assistant {
        background: #052e16;
        color: #dcfce7;
        content-align: left middle;
        border: round #15803d;
    }

    .error {
        background: #450a0a;
        color: #fecaca;
        border: round #dc2626;
    }
    """

    TITLE = "Support Engineer Assistant"
    BINDINGS = [
        ("ctrl+r", "recommend", "AI Recommendation"),
        ("ctrl+k", "similar", "Retrieve Similar"),
        ("ctrl+q", "quit", "Exit"),
    ]

    assistant: reactive[StreamingSupportAssistant | None] = reactive(None)

    def __init__(self, repo_root: Path) -> None:
        super().__init__()
        self.repo_root = repo_root
        self._active_ai_bubble: ChatBubble | None = None
        self._request_in_flight = False

    def compose(self) -> ComposeResult:
        with Vertical(id="app-shell"):
            yield Static("Support Engineer Assistant", id="title-bar")
            yield ConversationHistory(id="history")
            with Vertical(id="composer-panel"):
                with Horizontal(id="context-row", classes="composer-row"):
                    yield Select.from_values(
                        ["HackerRank", "Claude", "Visa", "None"],
                        value="None",
                        allow_blank=False,
                        id="company-select",
                    )
                    yield Input(placeholder="Optional subject for the ticket", id="subject-input")
                yield TextArea(
                    placeholder="Describe the support issue here. Multi-line input is supported, similar to a chat composer.",
                    id="issue-input",
                )
                with Horizontal(id="action-row-primary", classes="composer-row"):
                    yield Button("AI Recommendation", id="ai-button", classes="action-button compact-button", variant="primary")
                    yield Button("Retrieve Similar", id="similar-button", classes="action-button compact-button")
                    yield Button("Clear", id="clear-button", classes="action-button compact-button")
                    yield Button("Exit", id="exit-button", classes="action-button compact-button", variant="error")
                with Horizontal(id="action-row-secondary", classes="composer-row"):
                    yield Input(placeholder="Optional .csv path for ingestion", id="csv-path-input")
                    yield Button("Ingest CSV", id="csv-button", classes="action-button compact-button")

    async def on_mount(self) -> None:
        self.theme = "textual-dark"
        history = self.query_one("#history", ConversationHistory)
        history.add_bubble(self._make_assistant_bubble("Support AI is starting up..."))
        try:
            self.assistant = await asyncio.to_thread(StreamingSupportAssistant, self.repo_root)
            history.add_bubble(self._make_assistant_bubble(self.assistant.describe_backend()))
            history.add_bubble(
                self._make_assistant_bubble(
                    "Choose a company, optionally add a subject, type the issue like a chat message, then ask for an AI recommendation or retrieve similar incidents."
                )
            )
            self.query_one("#issue-input", TextArea).focus()
        except Exception as exc:
            history.add_bubble(self._make_error_bubble(f"Startup error: {exc}"))

    @on(Button.Pressed, "#ai-button")
    async def handle_ai_button(self) -> None:
        ticket = self._build_ticket_from_form()
        if not ticket or self._request_in_flight:
            return
        self._start_request(self._format_user_request(ticket))
        bubble = self._make_assistant_bubble("")
        bubble.set_header("Thinking...")
        self.query_one("#history", ConversationHistory).add_bubble(bubble)
        self._active_ai_bubble = bubble

        async def emit_analysis(summary: SupportStateSummary) -> None:
            self.post_message(AnalysisReady(summary))

        async def emit_token(token: str) -> None:
            self.post_message(StreamingUpdate(token))

        async def run_stream() -> None:
            try:
                if not self.assistant:
                    raise RuntimeError("Support AI is not ready yet.")
                analysis = await self.assistant.analyze_ticket(ticket)
                summary = self.assistant.summarize_state(analysis)
                await emit_analysis(summary)
                await self.assistant.stream_from_analysis(analysis, emit_token)
                self.post_message(RecommendationComplete(summary))
            except Exception as exc:
                self.post_message(AsyncFailed(str(exc)))

        asyncio.create_task(run_stream())

    @on(Button.Pressed, "#similar-button")
    async def handle_similar_button(self) -> None:
        ticket = self._build_ticket_from_form()
        if not ticket or self._request_in_flight:
            return
        self._start_request(self._format_user_request(ticket))

        async def run_similar() -> None:
            try:
                if not self.assistant:
                    raise RuntimeError("Support AI is not ready yet.")
                incidents = await self.assistant.retrieve_similar_incidents(ticket, limit=5)
                self.post_message(SimilarIncidentsReady(incidents))
            except Exception as exc:
                self.post_message(AsyncFailed(str(exc)))

        asyncio.create_task(run_similar())

    @on(Button.Pressed, "#csv-button")
    async def handle_csv_button(self) -> None:
        if self._request_in_flight:
            return
        csv_input = self.query_one("#csv-path-input", Input).value.strip()
        csv_path = Path(csv_input) if csv_input else self.repo_root / "support_tickets" / "support_tickets.csv"
        if not csv_path.is_absolute():
            csv_path = (self.repo_root / csv_path).resolve()
        self._start_request(f"CSV ingestion requested\npath={csv_path}")

        async def run_ingest() -> None:
            try:
                if not self.assistant:
                    raise RuntimeError("Support AI is not ready yet.")
                output_path, count = await self.assistant.ingest_csv(csv_path)
                self.post_message(CsvIngested(str(output_path), count))
            except Exception as exc:
                self.post_message(AsyncFailed(str(exc)))

        asyncio.create_task(run_ingest())

    @on(Button.Pressed, "#clear-button")
    def handle_clear_button(self) -> None:
        self.query_one("#subject-input", Input).value = ""
        self.query_one("#csv-path-input", Input).value = ""
        self.query_one("#issue-input", TextArea).clear()
        self.query_one("#company-select", Select).value = "None"
        self.query_one("#issue-input", TextArea).focus()

    @on(Button.Pressed, "#exit-button")
    def handle_exit_button(self) -> None:
        self.exit()

    def action_recommend(self) -> None:
        self.call_later(lambda: asyncio.create_task(self.handle_ai_button()))

    def action_similar(self) -> None:
        self.call_later(lambda: asyncio.create_task(self.handle_similar_button()))

    def on_analysis_ready(self, message: AnalysisReady) -> None:
        if self._active_ai_bubble is None:
            return
        self._active_ai_bubble.set_header(self._format_state_summary(message.summary))
        self._active_ai_bubble.set_message("Thinking...")
        self.query_one("#history", ConversationHistory).scroll_end(animate=False)

    def on_streaming_update(self, message: StreamingUpdate) -> None:
        if self._active_ai_bubble is None:
            return
        if self._active_ai_bubble.message_text == "Thinking...":
            self._active_ai_bubble.set_message("")
        self._active_ai_bubble.append_message(message.token)
        self.query_one("#history", ConversationHistory).scroll_end(animate=False)

    def on_recommendation_complete(self, message: RecommendationComplete) -> None:
        if self._active_ai_bubble is not None:
            self._active_ai_bubble.set_footer(self._format_references(message.summary.references))
        self._finish_request()

    def on_similar_incidents_ready(self, message: SimilarIncidentsReady) -> None:
        content = self._format_similar_incidents(message.incidents)
        self.query_one("#history", ConversationHistory).add_bubble(self._make_assistant_bubble(content))
        self._finish_request()

    def on_csv_ingested(self, message: CsvIngested) -> None:
        content = (
            f"CSV ingestion complete.\n"
            f"rows_processed: {message.count}\n"
            f"output_path: {message.output_path}"
        )
        self.query_one("#history", ConversationHistory).add_bubble(self._make_assistant_bubble(content))
        self._finish_request()

    def on_async_failed(self, message: AsyncFailed) -> None:
        history = self.query_one("#history", ConversationHistory)
        if self._active_ai_bubble is not None:
            self._active_ai_bubble.remove()
            self._active_ai_bubble = None
        history.add_bubble(self._make_error_bubble(message.error_text))
        self._finish_request()

    def _start_request(self, user_text: str) -> None:
        self._request_in_flight = True
        self._set_controls_disabled(True)
        self.query_one("#history", ConversationHistory).add_bubble(self._make_user_bubble(user_text))

    def _finish_request(self) -> None:
        self._active_ai_bubble = None
        self._request_in_flight = False
        self._set_controls_disabled(False)
        self.query_one("#issue-input", TextArea).focus()
        self.query_one("#history", ConversationHistory).scroll_end(animate=False)

    def _set_controls_disabled(self, disabled: bool) -> None:
        for widget_id in ["#ai-button", "#similar-button", "#csv-button", "#clear-button", "#subject-input", "#csv-path-input", "#company-select", "#issue-input"]:
            self.query_one(widget_id).disabled = disabled

    def _build_ticket_from_form(self) -> Ticket | None:
        company = self.query_one("#company-select", Select).value
        subject = self.query_one("#subject-input", Input).value.strip()
        issue = self.query_one("#issue-input", TextArea).text.strip()
        if not issue:
            self.query_one("#history", ConversationHistory).add_bubble(
                self._make_error_bubble("Please enter an issue description before submitting.")
            )
            return None
        return Ticket(issue=issue, subject=subject, company=str(company or "None"))

    @staticmethod
    def _format_user_request(ticket: Ticket) -> str:
        return (
            f"company: {ticket.company}\n"
            f"subject: {ticket.subject or '(none)'}\n"
            f"issue:\n{ticket.issue}"
        )

    @staticmethod
    def _format_state_summary(summary: SupportStateSummary) -> str:
        return (
            f"status: {summary.status}\n"
            f"product_area: {summary.product_area}\n"
            f"request_type: {summary.request_type}\n"
            f"risk_level: {summary.risk_level}\n"
            f"confidence: {summary.confidence:.2f}\n"
            f"response:"
        )

    @staticmethod
    def _format_references(references: list[str]) -> str:
        if not references:
            return "references:\n- none"
        return "references:\n" + "\n".join(f"- {reference}" for reference in references)

    @staticmethod
    def _format_similar_incidents(incidents: list[IncidentMatch]) -> str:
        if not incidents:
            return "No similar incidents were found in the sample incident set."
        lines = ["Similar incidents (semantic retrieval):"]
        for idx, incident in enumerate(incidents, start=1):
            lines.extend(
                [
                    f"{idx}. company={incident.company} | similarity={incident.score:.3f}",
                    f"   subject={incident.subject or '(none)'}",
                    f"   status={incident.status} | product_area={incident.product_area} | request_type={incident.request_type}",
                    f"   issue={incident.issue[:180]}",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _make_user_bubble(content: str) -> ChatBubble:
        bubble = ChatBubble("user", "▶", "You:", "right", "user")
        bubble.set_message(content)
        return bubble

    @staticmethod
    def _make_assistant_bubble(content: str) -> ChatBubble:
        bubble = ChatBubble("assistant", "◆", "Support AI:", "left", "assistant")
        bubble.set_message(content)
        return bubble

    @staticmethod
    def _make_error_bubble(content: str) -> ChatBubble:
        bubble = ChatBubble("error", "◆", "Support AI:", "left", "error")
        bubble.set_message(content)
        return bubble

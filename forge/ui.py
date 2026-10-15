"""Everything Forge draws in the terminal, built on the rich library.

Keeping the display code here means agent.py, tool_executor.py and cli.py
only say *what* to show, not *how* it looks.
"""
import json
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

VERSION = "v0.1.0"
ACCENT = "dark_orange"

LOGO = r"""
 ███████╗ ██████╗ ██████╗  ██████╗ ███████╗
 ██╔════╝██╔═══██╗██╔══██╗██╔════╝ ██╔════╝
 █████╗  ██║   ██║██████╔╝██║  ███╗█████╗
 ██╔══╝  ██║   ██║██╔══██╗██║   ██║██╔══╝
 ██║     ╚██████╔╝██║  ██║╚██████╔╝███████╗
 ╚═╝      ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚══════╝"""

# How much of a tool's output to preview on screen (the LLM still gets more)
PREVIEW_LINES = 10
PREVIEW_CHARS = 700


def banner(workspace: str, provider: str, model: str, mode: str) -> None:
    """Logo plus an info box with the session settings."""
    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("Workspace", str(workspace))
    info.add_row("Provider", provider)
    info.add_row("Model", model)
    info.add_row("Mode", mode_label(mode))
    info.add_row("Version", Text(VERSION, style="dim"))

    logo = Text(LOGO, style=f"bold {ACCENT}")
    tagline = Text("  your command line coding assistant", style="italic dim")
    console.print(Panel(Group(logo, tagline, Text(""), info),
                        border_style=ACCENT, box=box.ROUNDED, padding=(0, 2)))


def mode_label(mode: str) -> Text:
    if mode == "auto":
        return Text("auto (tools run without asking)", style="yellow")
    return Text("confirm (asks before each tool)", style="green")


def server_status(status: dict[str, str], tool_counts: dict[str, int]) -> None:
    """One line per MCP server: connected, skipped or failed."""
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold", pad_edge=False)
    table.add_column("", width=2)
    table.add_column("MCP server")
    table.add_column("Tools", justify="right")
    table.add_column("Status")
    for name, state in status.items():
        if state == "connected":
            icon, style = "✓", "green"
        elif state.startswith("skipped"):
            icon, style = "–", "yellow"
        else:
            icon, style = "✗", "red"
        table.add_row(Text(icon, style=style), name, str(tool_counts.get(name, 0)),
                      Text(state, style=style))
    console.print(table)
    console.print(Text("Type a task, or /help for commands.\n", style="dim"))


def help_text() -> None:
    table = Table.grid(padding=(0, 3))
    table.add_column(style=f"bold {ACCENT}")
    table.add_column()
    table.add_row("/auto", "run tools without asking")
    table.add_row("/confirm", "ask before every tool call")
    table.add_row("/tools", "list every tool the agent can use")
    table.add_row("/help", "show this list")
    table.add_row("exit", "quit Forge")
    console.print(Panel(table, title="Commands", border_style="dim", box=box.ROUNDED))


def tool_list(tools: list[dict[str, Any]], server_for) -> None:
    table = Table(box=box.SIMPLE, header_style="bold")
    table.add_column("Tool", style=f"bold {ACCENT}")
    table.add_column("Server")
    for tool in tools:
        name = tool["function"]["name"]
        table.add_row(name, server_for(name) or "forge (shell)")
    console.print(table)


def tool_call(name: str, arguments: dict[str, Any], server: str) -> None:
    """Show which tool the agent picked and what it passed in."""
    args = Table.grid(padding=(0, 2))
    args.add_column(style="dim")
    args.add_column()
    for key, value in arguments.items():
        shown = value if isinstance(value, str) else json.dumps(value)
        if len(shown) > 200:
            shown = shown[:200] + "…"
        args.add_row(key, shown)
    if not arguments:
        args.add_row("", Text("(no arguments)", style="dim"))
    console.print(Panel(args, title=f"[bold]⚙ {name}[/]", title_align="left",
                        subtitle=f"[dim]{server}[/]", subtitle_align="right",
                        border_style="cyan", box=box.ROUNDED))


def confirm() -> bool:
    answer = console.input("  [bold]Run this tool?[/] [dim](y/N)[/] ")
    return answer.strip().lower() in {"y", "yes"}


def tool_result(name: str, text: str) -> None:
    """Short preview of what the tool returned."""
    # Drop blank lines so the preview stays compact
    lines = [line for line in text.strip().splitlines() if line.strip()]
    preview = "\n".join(lines[:PREVIEW_LINES])
    if len(preview) > PREVIEW_CHARS:
        preview = preview[:PREVIEW_CHARS]
    if len(lines) > PREVIEW_LINES or len(text) > len(preview):
        preview += "\n…"
    failed = text.startswith(("Error", "Blocked", "Tool execution error", "Invalid"))
    style = "red" if failed else "grey50"
    label = "✗" if failed else "↳"
    console.print(Panel(Text(preview or "(no output)", style=style),
                        title=f"[{style}]{label} {name}[/]", title_align="left",
                        border_style=style, box=box.ROUNDED))


def answer(markdown_text: str) -> Markdown:
    return Markdown(markdown_text)


def info(message: str) -> None:
    console.print(Text(message, style="dim"))


def warn(message: str) -> None:
    console.print(Text(f"⚠ {message}", style="yellow"))


def error(message: str) -> None:
    console.print(Panel(Text(message, style="red"), title="[red]Forge error[/]",
                        title_align="left", border_style="red", box=box.ROUNDED))

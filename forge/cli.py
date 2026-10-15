"""Forge's interactive terminal: banner, server status, and the prompt loop."""
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from . import ui
from .agent import Agent
from .providers import create_client
from .tool_executor import ToolExecutor


def _workspace() -> Path:
    return Path(os.getenv("FORGE_WORKSPACE", Path.cwd())).resolve()


def _handle_command(command: str, executor: ToolExecutor, agent: Agent, mcp) -> bool:
    """Run a /command. Returns False if it isn't one we know."""
    if command == "/auto":
        executor.mode = "auto"
        ui.console.print("Mode:", ui.mode_label("auto"))
    elif command == "/confirm":
        executor.mode = "confirm"
        ui.console.print("Mode:", ui.mode_label("confirm"))
    elif command == "/tools":
        ui.tool_list(agent._get_tools(), mcp.server_for)
    elif command == "/help":
        ui.help_text()
    else:
        return False
    return True


async def main() -> None:
    load_dotenv()
    # Imported here so the core modules can be tested without starting MCP servers
    from .mcp_client import MCPClient

    workspace = _workspace()
    mode = os.getenv("FORGE_EXECUTION_MODE", "confirm").strip().lower()
    provider = os.getenv("FORGE_PROVIDER", "ollama")
    client, model = create_client(provider)

    ui.banner(workspace, provider, model, mode)

    mcp = MCPClient()
    with ui.console.status("[bold]Starting MCP servers...[/]", spinner="dots"):
        await mcp.connect_all()
    ui.server_status(mcp.status, mcp.tool_counts())

    try:
        executor = ToolExecutor(mcp, workspace=workspace, mode=mode)
        agent = Agent(client, model, mcp, executor)
        while True:
            try:
                prompt = ui.console.input(f"\n[bold {ui.ACCENT}]forge ›[/] ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if prompt.lower() in {"exit", "quit"}:
                break
            if not prompt:
                continue
            if prompt.startswith("/"):
                if not _handle_command(prompt.lower(), executor, agent, mcp):
                    ui.warn(f"Unknown command {prompt}. Type /help for the list.")
                continue
            try:
                # The agent streams its answer and tool calls to the screen itself
                await agent.run(prompt)
            except KeyboardInterrupt:
                ui.warn("Stopped.")
            except Exception as exc:
                ui.error(str(exc))
    finally:
        ui.info("\nGoodbye!")
        await mcp.close()


if __name__ == "__main__":
    asyncio.run(main())

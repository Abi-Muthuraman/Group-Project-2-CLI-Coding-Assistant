"""minimal command-line interface for Forge. assumes that config.py and mcp_client.py live in forge directory."""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from .agent import Agent
from .providers import create_client
from .tool_executor import ToolExecutor

def _workspace() -> Path:
    return Path(os.getenv("FORGE_WORKSPACE", Path.cwd())).resolve()

async def main() -> None:
    load_dotenv()
    #import here so that the core modules can tested without starting the MCP server
    from .mcp_client import MCPClient

    workspace = _workspace()
    mode = os.getenv("FORGE_EXECUTION_MODE", "confirm").strip().lower()
    client, model = create_client(os.getenv("FORGE_PROVIDER", 'ollama'))
    async with MCPClient() as mcp:
        print("Forge CLI. Type 'exit' or 'quit' to leave.")
        print(f"Provider: {os.getenv('FORGE_PROVIDER', 'ollama')} | Model: {model} | Mode: {mode}")
        print(f"Workspace: {workspace}")
        executor = ToolExecutor(mcp, workspace=workspace, mode=mode)
        agent = Agent(client, model, mcp, executor)
        while True:
            try:
                prompt = input("\nforge> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if prompt.lower() in {"exit", "quit"}:
                break
            if not prompt:
                continue
            try:
                answer = await agent.run(prompt)
                print(f"\n{answer}")
            except Exception as exc:
                print(f"Forge error: {exc}")


if __name__ == "__main__":
    asyncio.run(main())

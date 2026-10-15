"""simple tool-calling loop using OpenAI compatible chat API."""
import json
from typing import Any

from rich.live import Live

from . import ui
from .tool_executor import SHELL_TOOL_NAME, SHELL_TOOL_SCHEMA, ToolExecutor

SYSTEM_PROMPT = """You are FORGE, a command-line AI coding assistant. Help with code in the current workspace. 
Use the available tools when needed. Explain what you plan on doing. Do not claim a tool succeeeded unless you have confirmation of the success. 
Prefer small, reversible changes. The shell tool is available for running one command at a time. Do not chain commands or use redirection. 
It doesn't support shell operators.
When you use search_langchain_docs, base any code on the documentation it returns and name the page you used."""

class Agent:
    def __init__(self, client: Any, model: str, mcp_client: Any, executor: ToolExecutor, max_iterations: int = 8):
        self.client = client
        self.model = model
        self.mcp_client = mcp_client
        self.executor = executor
        self.max_iterations = max_iterations

    def _get_tools(self) -> list[dict[str, Any]]:
        # MCPClient.get_tools() in the team code returns OpenAI function-tool schemas.
        mcp_tools = self.mcp_client.get_tools()
        names = {tool.get("function", {}).get("name") for tool in mcp_tools}
        tools = list(mcp_tools)
        if "run_shell_command" not in names:
            tools.append(SHELL_TOOL_SCHEMA)
        return tools

    async def run(self, user_prompt: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        tools = self._get_tools()
        for _ in range(self.max_iterations):
            content, tool_calls = self._stream_turn(messages, tools)
            assistant_message: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {"name": call["name"], "arguments": call["arguments"]},
                    }
                    for call in tool_calls
                ]
            messages.append(assistant_message)

            if not tool_calls:
                if not content:
                    ui.warn("The model returned an empty response.")
                return content

            for call in tool_calls:
                try:
                    arguments = json.loads(call["arguments"] or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                    tool_name = call["name"]
                    # Every tool goes through the executor so confirm mode applies to all of them
                    if tool_name == SHELL_TOOL_NAME or self.mcp_client.server_for(tool_name) is not None:
                        result = await self.executor.execute(tool_name, arguments)
                    else:
                        result = (
                            f"Error: tool '{tool_name}' is not available. "
                            "Use only the tools available in the list"
                        )
                        ui.warn(f"The model asked for a tool that doesn't exist: {tool_name}")
                except (json.JSONDecodeError, ValueError) as exc:
                    result = f"Invalid tool arguments: {exc}"
                    ui.error(f"{call['name']}: {result}")
                # Keep tool output short so we stay under Groq's free-tier token limit
                result = str(result)
                if len(result) > 4000:
                    result = result[:4000] + "\n...(output truncated)"
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})

        message = f"Stopped after reaching the {self.max_iterations}-step limit. Please try a smaller request."
        ui.warn(message)
        return message

    def _stream_turn(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> tuple[str, list[dict[str, str]]]:
        """Ask the LLM for its next step, streaming the reply to the screen.

        Shows a spinner until the first words arrive, then renders the text as
        markdown while it streams. Tool calls arrive in pieces, so they are
        stitched back together by their index. Returns (text, tool_calls).
        """
        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=tools if tools else None,
            tool_choice="auto" if tools else None,
            stream=True,
        )
        content = ""
        calls: dict[int, dict[str, str]] = {}
        live = None
        status = ui.console.status("[bold]Thinking...[/]", spinner="dots")
        status.start()
        try:
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    if live is None:
                        status.stop()
                        live = Live(ui.answer(""), console=ui.console,
                                    refresh_per_second=12, vertical_overflow="visible")
                        live.start()
                    content += delta.content
                    live.update(ui.answer(content))
                for piece in delta.tool_calls or []:
                    slot = calls.setdefault(piece.index, {"id": "", "name": "", "arguments": ""})
                    if piece.id:
                        slot["id"] = piece.id
                    if piece.function and piece.function.name:
                        slot["name"] += piece.function.name
                    if piece.function and piece.function.arguments:
                        slot["arguments"] += piece.function.arguments
        finally:
            status.stop()
            if live is not None:
                live.stop()
        return content, [calls[i] for i in sorted(calls)]
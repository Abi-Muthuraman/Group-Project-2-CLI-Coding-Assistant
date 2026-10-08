"""simple tool-calling loop using OpenAI compatible chat API."""
import json
from typing import Any

from .tool_executor import SHELL_TOOL_SCHEMA, ToolExecutor

SYSTEM_PROMPT = """You are FORGE, a command-line AI coding assistant. Help with code in the current workspace. 
Use the available tools when needed. Explain what you plan on doing. Do not claim a tool succeeeded unless you have confirmation of the success. 
Prefer small, reversible changes. The shell tool is available for running one command at a time. Do not chain commands or use redirection. 
It doesn't support shell operators."""

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
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools if tools else None,
                tool_choice="auto" if tools else None,
            )
            message = response.choices[0].message
            assistant_message: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.function.name, "arguments": call.function.arguments},
                    }
                    for call in message.tool_calls
                ]
            messages.append(assistant_message)

            if not message.tool_calls:
                return message.content or "(The model returned an empty response.)"

            for call in message.tool_calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("Tool arguments must be a JSON object.")
                    tool_name = call.function.name
                    if tool_name == "run_shell_comand":
                        result = await self.executor.execute(tool_name, arguments)
                    elif self.mcp_client.server_for(tool_name) is not None:
                        result = await self.mcp_client.call_tool(tool_name, arguments)
                    else:
                        result = (
                            f"Error: tool '{tool_name}' is not available."
                            "Use only the tools available in the list"
                        )
                except (json.JSONDecodeError, ValueError) as exc:
                    result = f"Invalid tool arguments: {exc}"
                messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})

        return f"Stopped after reaching the {self.max_iterations}-iteration limit. Please try a smaller request."
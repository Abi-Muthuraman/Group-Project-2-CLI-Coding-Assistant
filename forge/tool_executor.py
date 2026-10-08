"""execute mcp tools and forge's local shell tool with confirmation safeguards"""
import asyncio
import shlex
from pathlib import Path
from typing import Any

SHELL_TOOL_NAME = "run_shell_command"
SHELL_TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": SHELL_TOOL_NAME,
        "description": "Run one shell command in the Forge workspace. No chaining or redirection is allowed",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "One command and its arguments"}},
            "required": ["command"],
            "additionalProperties": False,
        } 
    }
}

#conservative blocklist of shell commands that are not allowed to be run by the agent. not a complete sandbox
_BLOCKED_EXECUTABLES = {"sudo", "shutdown", "reboot", "mkfs", "dd", "diskutil"}
_BLOCKED_TOKENS = {";", "&&", "||", "|", ">", ">>", "<", "`"}

class ToolExecutor:
    def __init__(self, mcp_client: Any, workspace: str | Path, mode: str = "confirm", timeout: int = 30):
        if mode not in {"confirm", "auto"}:
            raise ValueError("mode must be 'confirm' or 'auto'")
        self.mcp_client = mcp_client
        self.workspace = Path(workspace).resolve()
        self.mode = mode
        self.timeout = timeout

    def _approved(self, name: str, arguments: dict[str, Any]) -> bool:
        if self.mode == "auto":
            return True
        print(f"\nTool requested: {name}\nArguments: {arguments}")
        return input("Run this tool? [y/N]: ").strip().lower() in {"y", "yes"}    

    async def execute(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        arguments = arguments or {}
        if not self._approved(name, arguments):
            return "Tool call cancelled by user."
        if name == SHELL_TOOL_NAME:
            return await self._run_shell(arguments)
        try:
            return await self.mcp_client.call_tool(name, arguments)
        except Exception as exc:
            return f"Tool execution error: {exc}"

    async def _run_shell(self, arguments: dict[str, Any]) -> str:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return "Error: 'command' must be a non-empty string."
        try:
            parts = shlex.split(command)
        except ValueError as exc:
            return f"Error parsing command: {exc}"
        if not parts:
            return "Error: empty command."
        if any(token in command for token in _BLOCKED_TOKENS):
            return "Blocked: use one command at a time; shell operators and redirection are not allowed."
        if Path(parts[0]).name in _BLOCKED_EXECUTABLES:
            return f"Blocked potentially destructive command: {parts[0]}"
        # Avoid running common recursive deletion commands.
        if Path(parts[0]).name == "rm" and any(arg in {"-r", "-rf", "-fr", "--recursive"} for arg in parts[1:]):
            return "Blocked recursive deletion."
        try:
            process = await asyncio.create_subprocess_exec(
                *parts,
                cwd=str(self.workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=self.timeout)
        except asyncio.TimeoutError:
            process.kill()
            await process.communicate()
            return f"Command timed out after {self.timeout} seconds."
        except FileNotFoundError:
            return f"Command not found: {parts[0]}"
        output = stdout.decode(errors="replace").strip()
        error = stderr.decode(errors="replace").strip()
        combined = output
        if error:
            combined = f"{combined}\nSTDERR:\n{error}" if combined else f"STDERR:\n{error}"
        if not combined:
            combined = "(no output)"
        return f"Exit code: {process.returncode}\n{combined}"

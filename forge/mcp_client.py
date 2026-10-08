from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from .config import ServerConfig, build_servers

@dataclass
class _Route:
    server:str
    session: ClientSession
    original_name: str


class MCPClient:
    def __init__(self, servers: list[ServerConfig] | None = None):
        self._configs = servers if servers is not None else build_servers()
        self._stack = AsyncExitStack()
        self._routes: dict[str, _Route] = {}
        self._schemas: list[dict[str, Any]] = []
        self.status: dict[str, str] = {}

    async def __aenter__(self) -> "MCPClient":
        await self.connect_all()
        return self

    async def __aexit__(self, *exc) -> None:
        await self.close()

    async def connect_all(self) -> None:
        for cfg in self._configs:
            if not cfg.enabled:
                self.status[cfg.name] = "skipped (disabled or missing config)"
                continue
            try:
                await self._connect(cfg)
                self.status[cfg.name] = "connected"
            except Exception as e:  # one broken server must not kill the whole agent
                self.status[cfg.name] = f"failed: {e}"
                
    async def _connect(self, cfg: ServerConfig) -> None:
        params = StdioServerParameters(command=cfg.command, args=cfg.args, env=cfg.env or None)
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
 
        listed = await session.list_tools()
        for tool in listed.tools:
             # Prefix on name collisions so two servers can't shadow each other.
             name = tool.name if tool.name not in self._routes else f"{cfg.name}__{tool.name}"
             self._routes[name] = _Route(cfg.name, session, tool.name)
             self._schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.description or "",
                        "parameters": tool.input_schema
                        if hasattr(tool, "inputSchema")
                        else tool.input_schema,
                    },
                }
            )

    def get_tools(self) -> list[dict[str, Any]]:
        return list(self._schemas)

    def server_for(self, tool_name: str) -> str | None:
        route = self._routes.get(tool_name)
        return route.server if route else None


    async def call_tool(self, name: str, args: dict[str, Any] | None = None) -> str:
        route = self._routes.get(name)
        if route is None:
            return f"Error: unknown tool '{name}'. Available: {', '.join(self._routes)}"
        try:
            result = await route.session.call_tool(route.original_name, args or {})
        except Exception as e:
            return f"Error calling {name}: {e}"
        text = "\n".join(
            block.text for block in result.content if getattr(block, "type", "") == "text"
        ) or "(no output)"
        return f"Error: {text}" if getattr(result, "isError", getattr(result, "is_error", False)) else text

    async def close(self) -> None:
            await self._stack.aclose()

    
        


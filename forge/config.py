import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKSPACE = Path(os.getenv("FORGE_WORKSPACE", Path.cwd())).resolve()


@dataclass
class ServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True



def build_servers() -> list[ServerConfig]:
    servers = [
        # Filesystem: only this directory is readable/writable by the agent.
        ServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", str(WORKSPACE)],
        ),
        # Tavily web search.
        ServerConfig(
            name="tavily",
            command="npx",
            args=["-y", "tavily-mcp@latest"],
            env={"TAVILY_API_KEY": os.getenv("TAVILY_API_KEY", "")},
            enabled=bool(os.getenv("TAVILY_API_KEY")),
        ),
        # Varshita RAG server
        ServerConfig(
            name="rag",
            command=sys.executable,
            args=["-m", "rag_server.server"],
            enabled=(PROJECT_ROOT / "rag_server" / "server.py").exists(),
        ),
    ]
    return servers
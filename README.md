# Forge: CLI AI Coding Assistant

Forge is a command line AI coding assistant. You give it a task in plain English, and it figures out which tools to use, reads and edits files, runs commands, searches the web or the LangChain docs, and keeps going until the task is done.

It connects to three MCP servers:

- **Filesystem server** for reading, writing, and editing files in your workspace
- **Tavily** for web search
- **Custom RAG server** that answers questions from the LangChain docs using Corrective RAG (CRAG)

It works with local models through **Ollama** and cloud models through **Groq**.

See [planning/Forge_Project_Plan.pdf](planning/Forge_Project_Plan.pdf) for the original architecture, team roles, and timeline.

## How it works

1. You type a task at the `forge>` prompt.
2. The agent sends your task and the list of available tools to the LLM.
3. The LLM either answers or picks tools to call.
4. In **confirm** mode, Forge shows each tool call and asks before running it. In **auto** mode, it runs right away.
5. The MCP client sends each call to the right server and passes the result back to the LLM.
6. This repeats until the task is done, up to 8 steps.

## Requirements

- Python 3.10 or newer
- Node.js 18 or newer (the filesystem and Tavily servers run through `npx`)
- A free Groq API key from [console.groq.com](https://console.groq.com)
- A free Tavily API key from [app.tavily.com](https://app.tavily.com)
- Optional: [Ollama](https://ollama.com) if you want to run a local model

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/Abi-Muthuraman/Group-Project-2-CLI-Coding-Assistant.git
cd Group-Project-2-CLI-Coding-Assistant
```

**2. Create a virtual environment and install packages**

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**3. Add your API keys**

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

**4. Build the RAG database (one time only)**

```bash
python rag_server/download_docs.py
python rag_server/ingest.py
```

This downloads the LangChain docs, splits them into about 1,300 chunks, embeds them, and saves them to `rag_server/chroma_db/`. Every later session reuses this database, so you never need to run these again unless you want to rebuild with `python rag_server/ingest.py --rebuild`.

**5. Run Forge**

```bash
python -m forge.cli
```

## Configuration

All settings go in `.env` at the repo root.

| Variable | What it does | Default |
|---|---|---|
| `FORGE_PROVIDER` | `groq` or `ollama` | `ollama` |
| `GROQ_API_KEY` | Groq API key | none |
| `GROQ_MODEL` | Groq model for the agent | `llama-3.3-70b-versatile` |
| `OLLAMA_MODEL` | Local model for the agent | `qwen2.5-coder:7b` |
| `OLLAMA_BASE_URL` | Ollama server address | `http://localhost:11434/v1` |
| `TAVILY_API_KEY` | Tavily API key. Tavily is skipped if this is empty | none |
| `CRAG_PROVIDER` | Model provider for CRAG grading: `groq` or `ollama` | `groq` |
| `CRAG_MODEL` | Model that grades retrieved chunks | `openai/gpt-oss-20b` |
| `FORGE_EXECUTION_MODE` | `confirm` asks before each tool call, `auto` runs without asking | `confirm` |
| `FORGE_WORKSPACE` | Folder the agent can read and edit | current folder |

Groq retires models often. If you see a "model not found" error, pick a current model from [console.groq.com/docs/models](https://console.groq.com/docs/models) and update `GROQ_MODEL` or `CRAG_MODEL`.

## Example prompts

```
forge> list the files in this folder
forge> read forge_demo.txt and tell me the secret word
forge> search the web for the latest langchain version
forge> use the langchain docs tool to explain how to add memory to an agent
forge> create a python file called two_sum.py that solves two sum, then run it
```

Type `exit` or `quit` to leave.

## Project structure

```
├── forge/                  # the coding assistant
│   ├── cli.py              # REPL and startup
│   ├── agent.py            # agentic loop
│   ├── providers.py        # Ollama and Groq setup
│   ├── mcp_client.py       # connects to MCP servers and loads their tools
│   ├── tool_executor.py    # confirm/auto modes and the shell tool
│   └── config.py           # which MCP servers to launch
├── rag_server/             # custom RAG MCP server (see rag_server/README.md)
│   ├── download_docs.py    # downloads the LangChain docs
│   ├── ingest.py           # one-time build of the vector DB
│   ├── crag.py             # Corrective RAG logic
│   ├── server.py           # the MCP server
│   └── try_search.py       # test retrieval without the agent
├── planning/               # project plan and architecture diagram
├── requirements.txt
└── .env.example
```

## Troubleshooting

**`node: command not found`**
Install Node.js from [nodejs.org](https://nodejs.org), then open a new terminal.

**`Request too large` or `rate_limit_exceeded` from Groq**
The free tier allows about 8,000 tokens per minute. Wait a minute and try again, or use a smaller request.

**`NameError: name 'nn' is not defined` or numpy errors on Intel Macs**
Run `pip install "numpy<2"`. Intel Macs can only install an older PyTorch, which doesn't work with numpy 2.

**RAG server shows "vector DB is empty"**
Run `python rag_server/ingest.py`.

## Team

- Varshita Yarabadi: custom RAG MCP server with CRAG
- Abi Muthuraman: agentic loop and provider abstraction
- Aadi: MCP client, filesystem and Tavily servers, CLI

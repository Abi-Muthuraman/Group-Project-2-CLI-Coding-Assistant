"""create an OpenAI client for  Ollama or Groq.
uses OpenAI python sdk as a common interface.
"""
import os
from openai import OpenAI
def create_client(provider: str | None = None) -> tuple[OpenAI, str]:
    """Return (client, model_name), using environment variable for configuration.
    Providers that are supported:
    - ollama: local Ollama server with OpenAi-compatible API.
    - groq: hosted Groq API (requires GROQ_API_KEY)
    """
    selected = (provider or os.getenv("FORGE_PROVIDER", "ollama")).strip().lower()
    if selected == "ollama":
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        model = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")
        #doesn't need an API key but the sdk expects a value
        client = OpenAI(base_url=base_url, api_key=os.getenv("OLLAMA_API_KEY", "ollama"))
        return client, model

    if selected == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required for Groq provider.")
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key), model

    raise ValueError(f"Unsupported provider: {selected}. Supported providers are 'ollama' and 'groq'. Please select one of these providers.")
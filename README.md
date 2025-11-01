
# LangGraph Sidekick Agent (Single File)

A single-file **LangGraph** agent (**worker + evaluator**) with **Gradio UI**, **MemorySaver** checkpointing, and built-in tools.

- Minimal tools: `get_time`, `write_text_file`
- Optional extended tools when `USE_PLAYWRIGHT=1`: Playwright browser toolkit, Serper search, Wikipedia, Python REPL, Pushover

## Quickstart
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U langgraph langchain-openai langchain-community langchain-core gradio python-dotenv pydantic typing_extensions nest_asyncio
# Optional:
pip install -U playwright wikipedia
playwright install

# Env
cp .env.example .env
# put your OPENAI_API_KEY in .env

python sidekick_agent.py
```

### Optional (enable extended tools)
```bash
# Windows
set USE_PLAYWRIGHT=1
# macOS/Linux
export USE_PLAYWRIGHT=1
```

## Topics
```
langgraph
multi-agent
structured-outputs
gradio
ai-agent
tool-calling
memory
openai
playwright
pydantic
state-machine
```

## License
MIT

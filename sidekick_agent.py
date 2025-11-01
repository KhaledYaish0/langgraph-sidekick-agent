
"""
Sidekick Agent (single-file)
============================
- LangGraph multi-agent (Worker + Evaluator + Router)
- Gradio UI
- MemorySaver checkpointing
- Minimal tools (always): get_time, write_text_file
- Optional extended tools (if USE_PLAYWRIGHT=1): Playwright browser toolkit, Serper search, Wikipedia, Python REPL, Pushover

Quickstart
----------
pip install -U langgraph langchain-openai langchain-community langchain-core gradio python-dotenv pydantic typing_extensions nest_asyncio
# Optional tools:
pip install -U playwright wikipedia
playwright install

# Environment
# Windows: set OPENAI_API_KEY=sk-...
# macOS/Linux: export OPENAI_API_KEY=sk-...
# Optional: set USE_PLAYWRIGHT=1 to enable browser & other extended tools

python sidekick_agent.py
"""

from __future__ import annotations

import os
import sys
import uuid
import asyncio
from datetime import datetime
from typing import Annotated, List, Any, Optional, Dict

import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv
load_dotenv(override=True)

from typing_extensions import TypedDict
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

import gradio as gr

# -------------------------
# Global config
# -------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
USE_PLAYWRIGHT = os.getenv("USE_PLAYWRIGHT", "0").lower() in ("1", "true", "yes", "y")

# -------------------------
# Minimal tools (always available)
# -------------------------
@tool
def get_time(_: str = "") -> str:
    """Return the current local date & time as 'YYYY-MM-DD HH:MM:SS'."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def write_text_file(params: str) -> str:
    """Write a text file. Params JSON:
    {"path": "output.txt", "content": "hello"}
    Returns the absolute path on success.
    """
    import json, os
    try:
        data = json.loads(params) if isinstance(params, str) else params
        path = data.get("path", "sidekick_output.txt")
        content = data.get("content", "")
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return os.path.abspath(path)
    except Exception as e:
        return f"[write_text_file error] {e}"

# -------------------------
# Optional extended tools
# -------------------------
async def load_extended_tools():
    """Load optional tools when USE_PLAYWRIGHT=1.
    Returns (tools_list, cleanup_callable)
    """
    if not USE_PLAYWRIGHT:
        return [], (lambda: None)

    tools = []
    cleanup_hooks = []

    # Pushover push tool
    try:
        import requests
        P_TOKEN = os.getenv("PUSHOVER_TOKEN")
        P_USER = os.getenv("PUSHOVER_USER")
        P_URL = "https://api.pushover.net/1/messages.json"

        def push(text: str):
            if not (P_TOKEN and P_USER):
                return "[push] Missing PUSHOVER_TOKEN or PUSHOVER_USER"
            try:
                requests.post(P_URL, data={"token": P_TOKEN, "user": P_USER, "message": text})
                return "success"
            except Exception as e:
                return f"[push error] {e}"

        from langchain.agents import Tool as LC_Tool
        tools.append(LC_Tool(name="send_push_notification", func=push, description="Send a Pushover notification"))
    except Exception as e:
        print(f"[Sidekick] Pushover tool not available: {e}")

    # Playwright
    try:
        from playwright.async_api import async_playwright
        from langchain_community.agent_toolkits import PlayWrightBrowserToolkit

        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=False)
        toolkit = PlayWrightBrowserToolkit.from_browser(async_browser=browser)
        tools.extend(toolkit.get_tools())

        async def cleanup_browser_async():
            try:
                await browser.close()
            except Exception:
                pass
            try:
                await pw.stop()
            except Exception:
                pass

        def cleanup_browser():
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(cleanup_browser_async())
                else:
                    loop.run_until_complete(cleanup_browser_async())
            except Exception:
                pass

        cleanup_hooks.append(cleanup_browser)
    except Exception as e:
        print(f"[Sidekick] Playwright tools not available: {e}")

    # Serper
    try:
        from langchain_community.utilities import GoogleSerperAPIWrapper
        from langchain.agents import Tool as LC_Tool
        serper = GoogleSerperAPIWrapper()
        tools.append(LC_Tool(name="search", func=serper.run, description="Web search via Google Serper"))
    except Exception as e:
        print(f"[Sidekick] Serper tool not available: {e}")

    # Wikipedia
    try:
        from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
        from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        tools.append(wiki)
    except Exception as e:
        print(f"[Sidekick] Wikipedia tool not available: {e}")

    # Python REPL
    try:
        from langchain_experimental.tools import PythonREPLTool
        tools.append(PythonREPLTool())
    except Exception as e:
        print(f"[Sidekick] Python REPL tool not available: {e}")

    def cleanup_all():
        for c in cleanup_hooks:
            try: c()
            except Exception: pass

    return tools, cleanup_all

# -------------------------
# State & Evaluator schema
# -------------------------
class State(TypedDict):
    messages: Annotated[List[Any], add_messages]
    success_criteria: str
    feedback_on_work: Optional[str]
    success_criteria_met: bool
    user_input_needed: bool

class EvaluatorOutput(BaseModel):
    feedback: str = Field(description="Feedback on the assistant's response")
    success_criteria_met: bool = Field(description="Whether the success criteria have been met")
    user_input_needed: bool = Field(description="True if more input is needed from the user")

# -------------------------
# Sidekick core
# -------------------------
class Sidekick:
    def __init__(self):
        self.worker_llm_with_tools = None
        self.evaluator_llm_with_output = None
        self.tools = []
        self.graph = None
        self.memory = MemorySaver()
        self.cleanup_hook = (lambda: None)
        self.thread_id = str(uuid.uuid4())

    async def setup(self):
        base_tools = [get_time, write_text_file]
        ext_tools, cleaner = await load_extended_tools()
        self.cleanup_hook = cleaner
        self.tools = base_tools + ext_tools

        worker_llm = ChatOpenAI(model=MODEL)
        self.worker_llm_with_tools = worker_llm.bind_tools(self.tools)

        evaluator_llm = ChatOpenAI(model=MODEL)
        self.evaluator_llm_with_output = evaluator_llm.with_structured_output(EvaluatorOutput)

        await self._build_graph()

    def _worker(self, state: State) -> Dict[str, Any]:
        system_message = (
            "You are a helpful assistant that can use tools to complete tasks.\n"
            "Keep working until you have a question for the user or the success criteria are met.\n"
            "If you need clarification, ask a clear question. If finished, reply ONLY with the final answer.\n\n"
            f"Success criteria:\n{state['success_criteria']}\n"
        )
        if state.get("feedback_on_work"):
            system_message += (
                "\nPreviously your reply was rejected because the criteria were not met.\n"
                f"Feedback:\n{state['feedback_on_work']}\n"
                "Continue and satisfy the criteria or ask a clear question.\n"
            )

        messages = state["messages"]
        found_system = False
        for m in messages:
            if isinstance(m, SystemMessage):
                m.content = system_message
                found_system = True
        if not found_system:
            messages = [SystemMessage(content=system_message)] + messages

        response = self.worker_llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _worker_router(self, state: State) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "evaluator"

    def _format_conv(self, messages: List[Any]) -> str:
        out = "Conversation history:\n\n"
        for m in messages:
            if isinstance(m, HumanMessage):
                out += f"User: {m.content}\n"
            elif isinstance(m, AIMessage):
                out += f"Assistant: {m.content or '[Tools use]'}\n"
        return out

    def _evaluator(self, state: State) -> State:
        last_text = getattr(state["messages"][-1], "content", "")
        system_message = (
            "You are an evaluator that checks whether the assistant met the success criteria.\n"
            "Return feedback, whether criteria are met, and whether more user input is needed."
        )
        user_message = (
            f"{self._format_conv(state['messages'])}\n\n"
            f"Success criteria:\n{state['success_criteria']}\n\n"
            f"Assistant's last response:\n{last_text}\n\n"
            f"Return feedback, success_criteria_met, user_input_needed."
        )

        result = self.evaluator_llm_with_output.invoke(
            [SystemMessage(content=system_message), HumanMessage(content=user_message)]
        )
        return {
            "messages": [{"role": "assistant", "content": f"Evaluator Feedback: {result.feedback}"}],
            "feedback_on_work": result.feedback,
            "success_criteria_met": result.success_criteria_met,
            "user_input_needed": result.user_input_needed,
        }  # type: ignore

    def _route_after_eval(self, state: State) -> str:
        if state["success_criteria_met"] or state["user_input_needed"]:
            return "END"
        return "worker"

    async def _build_graph(self):
        gb = StateGraph(State)
        gb.add_node("worker", self._worker)
        gb.add_node("tools", ToolNode(tools=self.tools))
        gb.add_node("evaluator", self._evaluator)

        gb.add_conditional_edges("worker", self._worker_router, {"tools": "tools", "evaluator": "evaluator"})
        gb.add_edge("tools", "worker")
        gb.add_conditional_edges("evaluator", self._route_after_eval, {"worker": "worker", "END": END})
        gb.add_edge(START, "worker")

        self.graph = gb.compile(checkpointer=self.memory)

    async def run_superstep(self, message: str, criteria: str, history: list | None) -> list:
        config = {"configurable": {"thread_id": self.thread_id}}
        state: State = {
            "messages": [HumanMessage(content=message)],
            "success_criteria": criteria or "Clear, correct, concise answer.",
            "feedback_on_work": None,
            "success_criteria_met": False,
            "user_input_needed": False,
        }
        result = await self.graph.ainvoke(state, config=config)
        user = {"role": "user", "content": message}
        reply = {"role": "assistant", "content": result["messages"][-2].content}
        feedback = {"role": "assistant", "content": result["messages"][-1].content}
        return (history or []) + [user, reply, feedback]

    def cleanup(self):
        try: self.cleanup_hook()
        except Exception: pass

# -------------------------
# Gradio UI
# -------------------------
async def setup():
    sk = Sidekick()
    await sk.setup()
    return sk

async def process_message(sk, message, criteria, history):
    return await sk.run_superstep(message, criteria, history), sk

async def reset():
    sk = Sidekick()
    await sk.setup()
    return "", "", None, sk

def free_resources(sk):
    try:
        if sk: sk.cleanup()
    except Exception as e:
        print(f"Cleanup error: {e}")

def launch_ui():
    with gr.Blocks(title="Sidekick", theme=gr.themes.Default(primary_hue="emerald")) as ui:
        gr.Markdown("## Sidekick — Personal Co-Worker (Single File)")
        sk_state = gr.State(delete_callback=free_resources)

        with gr.Row():
            chatbot = gr.Chatbot(label="Sidekick", height=320, type="messages")
        with gr.Group():
            with gr.Row():
                message = gr.Textbox(show_label=False, placeholder="Your request to the Sidekick")
            with gr.Row():
                criteria = gr.Textbox(show_label=False, placeholder="What are your success criteria?")

        with gr.Row():
            reset_btn = gr.Button("Reset", variant="stop")
            go_btn = gr.Button("Go!", variant="primary")

        ui.load(setup, [], [sk_state])
        message.submit(process_message, [sk_state, message, criteria, chatbot], [chatbot, sk_state])
        criteria.submit(process_message, [sk_state, message, criteria, chatbot], [chatbot, sk_state])
        go_btn.click(process_message, [sk_state, message, criteria, chatbot], [chatbot, sk_state])
        reset_btn.click(reset, [], [message, criteria, chatbot, sk_state])

    ui.launch(inbrowser=True)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY is not set. Put it in a .env file or environment variable.")
        sys.exit(1)
    # Windows event loop policy to avoid subprocess issues
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        except Exception:
            pass
    launch_ui()

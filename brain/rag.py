from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph.state import CompiledStateGraph

from brain.app_config import AppConfig
from brain.tools.builtin import (
    ToolContext,
    build_tools,
    compute_mail_sender_breakdown,
    wants_mail_sender_breakdown_query,
)
from brain.vectorstore import as_retriever, get_vectorstore

SYSTEM_PROMPT = (
    "You are a local assistant answering from tools and retrieved excerpts—avoid vague meta-talk "
    "when excerpts contain usable facts.\n\n"
    "Main tool: search_knowledge(query, source_scope). "
    'Scopes: any (default), screenshots, mail, notes, pdfs; use "mail" or "email" for inbox-style '
    "questions.\n"
    "For aggregate counts **by sender/From**, call mail_sender_breakdown (don't invent totals).\n\n"
    "Email excerpts look like normal messages (Subject/From/Date, then body). Tokens such as "
    "`mail:Envelope Index-v9:12345` are citation ids only.\n\n"
    "After tools finish: answer directly in prose—never paste raw JSON tool-call payloads.\n\n"
    "Other tools: summarize_meeting, draft_reply, todos, write_note, open_calendar."
)


def build_agent(cfg: AppConfig, ctx: ToolContext) -> CompiledStateGraph[Any, Any, Any, Any]:
    store = get_vectorstore(cfg.chroma_persist_directory, cfg.embedding_model_name)
    retriever = as_retriever(store, cfg.retrieve_k)
    tools = build_tools(cfg, ctx, retriever, store)
    llm = ChatOllama(
        base_url=cfg.ollama_base_url,
        model=cfg.ollama_chat_model,
        temperature=0.2,
    )
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


def run_chat_turn(
    user_text: str,
    *,
    cfg: AppConfig,
    get_agent: Callable[[], CompiledStateGraph[Any, Any, Any, Any]],
) -> str:
    t = user_text.strip().lower()
    if re.fullmatch(r"(summarize|summarise)\s+(my\s+)?(emails?|mail|inbox)", t):
        return (
            "I can summarize mail, but I need a scope.\n\n"
            "- Say a **time window**: e.g. `summarize my email from the last 7 days`\n"
            "- Or a **topic**: e.g. `summarize emails about taxes` "
            "(and I’ll search `source_scope=mail`)\n"
            "- Or ask for **senders**: `categorize all email senders`\n\n"
            "Without a scope, any summary would just be a small (and potentially misleading) "
            "sample."
        )
    # Small models still ignore injected breakdown and retrieve unrelated chunks; answer directly.
    if wants_mail_sender_breakdown_query(user_text):
        store = get_vectorstore(cfg.chroma_persist_directory, cfg.embedding_model_name)
        return compute_mail_sender_breakdown(store, include_tool_instruction=False)

    agent = get_agent()
    state: Mapping[str, Any] = agent.invoke({"messages": [HumanMessage(content=user_text)]})
    messages = state.get("messages") or []
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            tc = getattr(m, "tool_calls", None) or []
            if not tc and m.content:
                return str(m.content)
    if messages:
        last = messages[-1]
        return str(getattr(last, "content", last) or "")
    return ""

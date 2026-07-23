from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .configuration import ResearchSettings


def create_research_llm(settings: ResearchSettings):
    if not settings.llm_api_key:
        return None
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        temperature=0.2,
    )


async def invoke_text(llm, *, system: str, user: str) -> str:
    if llm is None:
        return ""
    response = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    content = getattr(response, "content", response)
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                pieces.append(str(item["text"]))
            elif isinstance(item, str):
                pieces.append(item)
        return "\n".join(pieces).strip()
    return str(content or "").strip()


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    else:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            candidate = candidate[start : end + 1]
    value = json.loads(candidate)
    if not isinstance(value, dict):
        raise ValueError("LLM output is not a JSON object")
    return value

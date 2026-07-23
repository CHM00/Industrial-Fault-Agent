from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Literal

from langgraph.graph import END, START, StateGraph

from ..budget import SearchBudget
from ..configuration import ResearchSettings
from ..contracts import (
    ExternalSource,
    ResearchOptions,
    ResearchTask,
    ResearchTaskResult,
    ResearchWarning,
    deduplicate_sources,
    models_from_state,
    models_to_state,
    warning,
)
from ..llm import invoke_text, parse_json_object
from ..providers import SearchProvider, SearchProviderError
from .prompts import QUERY_SYSTEM, REFLECTION_SYSTEM, SUMMARY_SYSTEM
from .state import BasicResearchState


@dataclass
class BasicResearchContext:
    options: ResearchOptions
    settings: ResearchSettings
    provider: SearchProvider
    budget: SearchBudget
    llm: object | None


def _fallback_query(topic: str, iteration: int = 0) -> str:
    suffixes = [
        "故障诊断 维修方案 安全步骤",
        "常见原因 检测方法 参数标准",
        "报警代码 厂商手册 排查流程",
        "电气机械原因 现场案例",
        "预防措施 复发验证",
    ]
    return f"{topic} {suffixes[min(iteration, len(suffixes) - 1)]}"[:400]


def _format_sources(sources: list[ExternalSource], max_chars: int) -> str:
    chunks = []
    used = 0
    for source in sources:
        block = (
            f"来源标题: {source.title}\n"
            f"来源URL: {source.url}\n"
            f"来源内容: {(source.content or source.snippet)}\n"
        )
        remaining = max_chars - used
        if remaining <= 0:
            break
        chunks.append(block[:remaining])
        used += min(len(block), remaining)
    return "\n---\n".join(chunks)


def _fallback_summary(sources: list[ExternalSource]) -> str:
    if not sources:
        return ""
    lines = []
    for source in sources:
        excerpt = (source.snippet or source.content).strip().replace("\n", " ")[:500]
        lines.append(f"- {source.title}：{excerpt}")
    return "\n".join(lines)


def create_basic_research_graph(context: BasicResearchContext):
    async def generate_query(state: BasicResearchState) -> dict:
        topic = state["research_topic"]
        query = _fallback_query(topic)
        if context.llm is not None:
            try:
                text = await invoke_text(
                    context.llm,
                    system=QUERY_SYSTEM,
                    user=f"<fault_topic>\n{topic}\n</fault_topic>",
                )
                value = str(parse_json_object(text).get("query", "")).strip()
                if value:
                    query = value[:400]
            except Exception:
                pass
        return {"current_query": query}

    async def web_research(state: BasicResearchState) -> dict:
        query = state["current_query"].strip()
        warnings = list(state.get("warnings", []))
        search_count = state.get("search_count", 0)
        acquired_any = False
        latest_sources: list[ExternalSource] = []

        for attempt in range(context.options.search_max_retries + 1):
            if not await context.budget.acquire(state["task_id"]):
                warnings.append(
                    warning(
                        "RESEARCH_BUDGET_EXHAUSTED",
                        "搜索预算已用尽，研究提前结束",
                        task_id=state["task_id"],
                    ).model_dump(mode="json")
                )
                return {
                    "latest_sources": [],
                    "warnings": warnings,
                    "search_count": search_count,
                    "stop_reason": "budget_exhausted",
                }

            acquired_any = True
            search_count += 1
            try:
                latest_sources = await context.provider.search(
                    query,
                    max_results=context.settings.max_results_per_search,
                    timeout_seconds=context.options.search_timeout_seconds,
                    max_source_chars=context.options.max_source_chars,
                )
                if not latest_sources:
                    warnings.append(
                        warning(
                            "SEARCH_EMPTY",
                            f"查询未返回有效来源: {query}",
                            task_id=state["task_id"],
                        ).model_dump(mode="json")
                    )
                break
            except SearchProviderError as exc:
                can_retry = exc.retryable and attempt < context.options.search_max_retries
                if can_retry:
                    await asyncio.sleep(min(2**attempt, 4))
                    continue
                warnings.append(
                    warning(
                        exc.code,
                        str(exc),
                        task_id=state["task_id"],
                        retryable=exc.retryable,
                    ).model_dump(mode="json")
                )
                break

        existing_sources = models_from_state(ExternalSource, state.get("sources", []))
        all_sources = deduplicate_sources(existing_sources + latest_sources)
        return {
            "sources": models_to_state(all_sources),
            "latest_sources": models_to_state(latest_sources),
            "query_count": state.get("query_count", 0) + (1 if acquired_any else 0),
            "search_count": search_count,
            "warnings": warnings,
        }

    async def summarize_sources(state: BasicResearchState) -> dict:
        sources = models_from_state(ExternalSource, state.get("sources", []))
        if not sources:
            return {"summary": state.get("summary", "")}
        fallback = _fallback_summary(sources)
        if context.llm is None:
            return {"summary": fallback}
        source_text = _format_sources(
            models_from_state(ExternalSource, state.get("latest_sources", [])) or sources,
            context.options.max_external_context_chars,
        )
        try:
            summary = await invoke_text(
                context.llm,
                system=SUMMARY_SYSTEM,
                user=(
                    f"故障研究主题：{state['research_topic']}\n\n"
                    f"已有摘要：\n{state.get('summary', '')}\n\n"
                    f"<external_sources>\n{source_text}\n</external_sources>"
                ),
            )
            return {"summary": summary or fallback}
        except Exception as exc:
            warnings = list(state.get("warnings", []))
            warnings.append(
                warning(
                    "SUMMARY_FALLBACK",
                    f"研究摘要生成失败，已使用来源摘要: {exc}",
                    task_id=state["task_id"],
                ).model_dump(mode="json")
            )
            return {"summary": fallback, "warnings": warnings}

    async def reflect_on_summary(state: BasicResearchState) -> dict:
        iteration = state.get("query_count", 0)
        query = _fallback_query(state["research_topic"], iteration)
        if context.llm is not None:
            try:
                text = await invoke_text(
                    context.llm,
                    system=REFLECTION_SYSTEM,
                    user=(
                        f"研究主题：{state['research_topic']}\n"
                        f"<untrusted_summary>\n{state.get('summary', '')}\n</untrusted_summary>"
                    ),
                )
                value = str(parse_json_object(text).get("follow_up_query", "")).strip()
                if value:
                    query = value[:400]
            except Exception:
                pass
        return {"current_query": query}

    def route_research(state: BasicResearchState) -> Literal["reflect", "end"]:
        if state.get("stop_reason"):
            return "end"
        if state.get("query_count", 0) >= state["research_depth"]:
            return "end"
        task_id = state["task_id"]
        if context.budget.used_by(task_id) >= context.budget.task_limits.get(task_id, 0):
            return "end"
        return "reflect"

    builder = StateGraph(BasicResearchState)
    builder.add_node("generate_query", generate_query)
    builder.add_node("web_research", web_research)
    builder.add_node("summarize_sources", summarize_sources)
    builder.add_node("reflect_on_summary", reflect_on_summary)
    builder.add_edge(START, "generate_query")
    builder.add_edge("generate_query", "web_research")
    builder.add_edge("web_research", "summarize_sources")
    builder.add_conditional_edges(
        "summarize_sources",
        route_research,
        {"reflect": "reflect_on_summary", "end": END},
    )
    builder.add_edge("reflect_on_summary", "web_research")
    return builder.compile(checkpointer=False)


async def run_basic_research_task(
    task: ResearchTask,
    *,
    depth: int,
    context: BasicResearchContext,
) -> ResearchTaskResult:
    graph = create_basic_research_graph(context)
    output = await graph.ainvoke(
        {
            "task_id": task.id,
            "research_topic": task.description,
            "research_depth": depth,
            "current_query": "",
            "summary": "",
            "sources": [],
            "latest_sources": [],
            "query_count": 0,
            "search_count": 0,
            "warnings": [],
            "stop_reason": "",
        }
    )
    sources = deduplicate_sources(
        models_from_state(ExternalSource, output.get("sources", []))
    )
    warnings = models_from_state(ResearchWarning, output.get("warnings", []))
    if not sources:
        status = "failed"
    elif warnings:
        status = "partial"
    else:
        status = "success"
    return ResearchTaskResult(
        task_id=task.id,
        description=task.description,
        summary=output.get("summary", ""),
        sources=sources,
        query_count=output.get("query_count", 0),
        search_count=output.get("search_count", 0),
        status=status,
        warnings=warnings,
    )

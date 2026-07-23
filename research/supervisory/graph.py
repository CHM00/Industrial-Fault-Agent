from __future__ import annotations

import re
from dataclasses import dataclass

from langgraph.graph import END, START, StateGraph

from ..basic.graph import BasicResearchContext, run_basic_research_task
from ..budget import allocate_task_budgets
from ..contracts import (
    ExternalSource,
    ResearchTask,
    ResearchTaskPlan,
    ResearchTaskResult,
    ResearchWarning,
    deduplicate_sources,
    models_from_state,
    models_to_state,
    normalize_url,
    warning,
)
from ..llm import invoke_text, parse_json_object
from .prompts import ANALYSIS_SYSTEM, PLAN_SYSTEM, SYNTHESIS_SYSTEM
from .state import SupervisoryResearchState


@dataclass
class SupervisoryContext:
    basic: BasicResearchContext


def _fallback_task(fault: str) -> ResearchTask:
    return ResearchTask(id="task_1", description=fault[:500], priority=5)


def _normalize_tasks(tasks: list[ResearchTask], max_tasks: int) -> list[ResearchTask]:
    unique: list[ResearchTask] = []
    seen: set[str] = set()
    for task in sorted(tasks, key=lambda item: (-item.priority, item.id)):
        description = " ".join(task.description.split()).strip()
        key = description.casefold()
        if not description or key in seen:
            continue
        seen.add(key)
        unique.append(
            ResearchTask(
                id=f"task_{len(unique) + 1}",
                description=description[:500],
                priority=task.priority,
            )
        )
        if len(unique) >= max_tasks:
            break
    return unique


def _format_task_results(results: list[ResearchTaskResult], max_chars: int) -> str:
    chunks = []
    used = 0
    for result in results:
        source_lines = "\n".join(
            f"- {source.source_id}: {source.title} | {source.url}" for source in result.sources
        )
        block = (
            f"任务 {result.task_id}: {result.description}\n"
            f"状态: {result.status}\n"
            f"摘要:\n{result.summary}\n"
            f"来源:\n{source_lines}\n"
        )
        remaining = max_chars - used
        if remaining <= 0:
            break
        chunks.append(block[:remaining])
        used += min(len(block), remaining)
    return "\n---\n".join(chunks)


def _fallback_synthesis(results: list[ResearchTaskResult]) -> str:
    parts = [result.summary.strip() for result in results if result.summary.strip()]
    return "\n\n".join(parts)


def _remove_unverified_urls(
    text: str,
    sources: list[ExternalSource],
) -> tuple[str, list[ResearchWarning]]:
    allowed = {source.normalized_url for source in sources}
    warnings: list[ResearchWarning] = []
    pattern = re.compile(r"https?://[^\s)\]}>]+")

    def replace(match: re.Match) -> str:
        raw = match.group(0).rstrip(".,;，。；")
        punctuation = match.group(0)[len(raw) :]
        try:
            valid = normalize_url(raw) in allowed
        except ValueError:
            valid = False
        if valid:
            return match.group(0)
        warnings.append(
            warning(
                "UNVERIFIED_CITATION_REMOVED",
                f"已移除无法映射到搜索来源的引用: {raw}",
            )
        )
        return "[已移除无法验证的引用]" + punctuation

    return pattern.sub(replace, text), warnings


def create_supervisory_graph(context: SupervisoryContext):
    options = context.basic.options

    async def decompose_request(state: SupervisoryResearchState) -> dict:
        fault = state["fault_input"]
        warnings = list(state.get("warnings", []))
        tasks = [_fallback_task(fault)]
        if context.basic.llm is not None:
            try:
                text = await invoke_text(
                    context.basic.llm,
                    system=PLAN_SYSTEM,
                    user=f"<fault_description>\n{fault}\n</fault_description>",
                )
                plan = ResearchTaskPlan.model_validate(parse_json_object(text))
                tasks = _normalize_tasks(plan.tasks, options.max_research_tasks)
                if not tasks:
                    raise ValueError("任务列表为空")
            except Exception as exc:
                tasks = [_fallback_task(fault)]
                warnings.append(
                    warning(
                        "TASK_PLAN_FALLBACK",
                        f"任务分解失败，已降级为单任务研究: {exc}",
                    ).model_dump(mode="json")
                )

        allocations = allocate_task_budgets(
            tasks,
            research_depth=options.research_depth,
            max_research_tasks=options.max_research_tasks,
            max_total_searches=options.max_total_searches,
        )
        tasks = [task for task, _ in allocations]
        task_depths = {task.id: depth for task, depth in allocations}
        context.basic.budget.task_limits = dict(task_depths)
        return {
            "tasks": models_to_state(tasks),
            "task_depths": task_depths,
            "warnings": warnings,
        }

    async def execute_research(state: SupervisoryResearchState) -> dict:
        results: list[ResearchTaskResult] = []
        warnings = list(state.get("warnings", []))
        tasks = models_from_state(ResearchTask, state.get("tasks", []))
        for task in tasks:
            depth = state["task_depths"].get(task.id, 1)
            try:
                result = await run_basic_research_task(task, depth=depth, context=context.basic)
            except Exception as exc:
                result = ResearchTaskResult(
                    task_id=task.id,
                    description=task.description,
                    status="failed",
                    warnings=[
                        warning(
                            "RESEARCH_TASK_FAILED",
                            f"研究任务执行失败: {exc}",
                            task_id=task.id,
                        )
                    ],
                )
            results.append(result)
            warnings.extend(models_to_state(result.warnings))
        return {"task_results": models_to_state(results), "warnings": warnings}

    async def analyze_results(state: SupervisoryResearchState) -> dict:
        results = models_from_state(ResearchTaskResult, state.get("task_results", []))
        fallback = _fallback_synthesis(results)
        if context.basic.llm is None or not fallback:
            return {"analysis": fallback}
        try:
            text = await invoke_text(
                context.basic.llm,
                system=ANALYSIS_SYSTEM,
                user=(
                    f"原始故障：{state['fault_input']}\n\n"
                    f"<untrusted_research_results>\n"
                    f"{_format_task_results(results, options.max_external_context_chars)}\n"
                    f"</untrusted_research_results>"
                ),
            )
            return {"analysis": text or fallback}
        except Exception as exc:
            warnings = list(state.get("warnings", []))
            warnings.append(
                warning(
                    "ANALYSIS_FALLBACK",
                    f"分析失败，已使用任务摘要: {exc}",
                ).model_dump(mode="json")
            )
            return {"analysis": fallback, "warnings": warnings}

    async def synthesize_final_report(state: SupervisoryResearchState) -> dict:
        results = models_from_state(ResearchTaskResult, state.get("task_results", []))
        sources = deduplicate_sources([source for result in results for source in result.sources])
        fallback = state.get("analysis") or _fallback_synthesis(results)
        warnings = list(state.get("warnings", []))
        synthesis = fallback
        if context.basic.llm is not None and fallback:
            try:
                source_index = "\n".join(
                    f"- {source.source_id}: {source.title} | {source.url}" for source in sources
                )
                synthesis = await invoke_text(
                    context.basic.llm,
                    system=SYNTHESIS_SYSTEM,
                    user=(
                        f"原始故障：{state['fault_input']}\n\n"
                        f"证据分析：\n{state.get('analysis', '')}\n\n"
                        f"允许引用的来源：\n{source_index}"
                    ),
                ) or fallback
            except Exception as exc:
                warnings.append(
                    warning(
                        "SYNTHESIS_FALLBACK",
                        f"综合失败，已使用分析摘要: {exc}",
                    ).model_dump(mode="json")
                )
        synthesis, citation_warnings = _remove_unverified_urls(synthesis, sources)
        warnings.extend(models_to_state(citation_warnings))
        return {"final_synthesis": synthesis, "warnings": warnings}

    builder = StateGraph(SupervisoryResearchState)
    builder.add_node("decompose_request", decompose_request)
    builder.add_node("execute_research", execute_research)
    builder.add_node("analyze_results", analyze_results)
    builder.add_node("synthesize_final_report", synthesize_final_report)
    builder.add_edge(START, "decompose_request")
    builder.add_edge("decompose_request", "execute_research")
    builder.add_edge("execute_research", "analyze_results")
    builder.add_edge("analyze_results", "synthesize_final_report")
    builder.add_edge("synthesize_final_report", END)
    return builder.compile(checkpointer=False)


async def run_supervisory_research(
    fault: str,
    *,
    context: SupervisoryContext,
) -> SupervisoryResearchState:
    graph = create_supervisory_graph(context)
    return await graph.ainvoke(
        {
            "fault_input": fault,
            "tasks": [],
            "task_depths": {},
            "task_results": [],
            "analysis": "",
            "final_synthesis": "",
            "warnings": [],
        }
    )

from __future__ import annotations

import asyncio
import time

from .basic.graph import BasicResearchContext, run_basic_research_task
from .budget import SearchBudget
from .configuration import ResearchSettings
from .contracts import (
    EffectiveResearchMode,
    ExternalResearchResult,
    ExternalSource,
    ResearchOptions,
    ResearchTask,
    ResearchTaskResult,
    ResearchWarning,
    deduplicate_sources,
    models_from_state,
    warning,
)
from .llm import create_research_llm
from .providers import SearchProvider, SearchProviderError, create_search_provider
from .supervisory.graph import SupervisoryContext, run_supervisory_research


def select_research_mode(fault: str) -> EffectiveResearchMode:
    complex_markers = (
        "同时",
        "并且",
        "跨系统",
        "多个",
        "间歇",
        "振动",
        "电压波动",
        "通信",
    )
    marker_count = sum(marker in fault for marker in complex_markers)
    return "supervisory" if len(fault) >= 80 or marker_count >= 2 else "basic"


def _globalize_task_sources(
    task_results: list[ResearchTaskResult],
) -> tuple[list[ResearchTaskResult], list[ExternalSource]]:
    global_sources = deduplicate_sources(
        [source for result in task_results for source in result.sources]
    )
    id_by_url = {source.normalized_url: source.source_id for source in global_sources}
    updated_results = []
    for result in task_results:
        sources = []
        for source in result.sources:
            source_id = id_by_url.get(source.normalized_url, source.source_id)
            sources.append(source.model_copy(update={"source_id": source_id}))
        updated_results.append(result.model_copy(update={"sources": sources}))
    return updated_results, global_sources


def _result_status(sources: list[ExternalSource], warnings: list, task_results: list) -> str:
    if not sources:
        return "failed"
    if warnings or any(result.status != "success" for result in task_results):
        return "partial"
    return "success"


async def run_external_research(
    fault: str,
    options: ResearchOptions | dict | None = None,
    *,
    settings: ResearchSettings | None = None,
    provider: SearchProvider | None = None,
    llm=None,
) -> ExternalResearchResult:
    started = time.perf_counter()
    options = options if isinstance(options, ResearchOptions) else ResearchOptions.model_validate(options or {})
    requested_mode = options.research_mode

    if requested_mode == "off":
        return ExternalResearchResult(
            requested_mode="off",
            effective_mode="off",
            status="skipped",
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

    settings = settings or ResearchSettings.from_env()
    effective_timeout_seconds = (
        settings.total_timeout_seconds
        if options.research_timeout_seconds is None
        else options.research_timeout_seconds
    )
    effective_mode: EffectiveResearchMode = (
        select_research_mode(fault) if requested_mode == "auto" else requested_mode
    )
    budget = SearchBudget(total_limit=options.max_total_searches, task_limits={})

    try:
        provider = provider or create_search_provider(options.search_api, settings)
        llm = llm if llm is not None else create_research_llm(settings)

        async def execute():
            if effective_mode == "basic":
                task = ResearchTask(id="task_1", description=fault[:500], priority=5)
                depth = min(options.research_depth, options.max_total_searches)
                budget.task_limits = {task.id: depth}
                context = BasicResearchContext(
                    options=options,
                    settings=settings,
                    provider=provider,
                    budget=budget,
                    llm=llm,
                )
                task_result = await run_basic_research_task(task, depth=depth, context=context)
                return task_result.summary, [task_result], list(task_result.warnings)

            basic_context = BasicResearchContext(
                options=options,
                settings=settings,
                provider=provider,
                budget=budget,
                llm=llm,
            )
            state = await run_supervisory_research(
                fault,
                context=SupervisoryContext(basic=basic_context),
            )
            return (
                state.get("final_synthesis", ""),
                models_from_state(ResearchTaskResult, state.get("task_results", [])),
                models_from_state(ResearchWarning, state.get("warnings", [])),
            )

        if effective_timeout_seconds == 0:
            summary, task_results, warnings = await execute()
        else:
            summary, task_results, warnings = await asyncio.wait_for(
                execute(),
                timeout=effective_timeout_seconds,
            )
        task_results, sources = _globalize_task_sources(task_results)
        return ExternalResearchResult(
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            status=_result_status(sources, warnings, task_results),
            summary=summary,
            sources=sources,
            task_results=task_results,
            query_count=sum(result.query_count for result in task_results),
            search_count=budget.total_used,
            warnings=warnings,
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    except asyncio.TimeoutError:
        return ExternalResearchResult(
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            status="failed",
            search_count=budget.total_used,
            warnings=[
                warning(
                    "RESEARCH_TIMEOUT",
                    f"外部研究超过总时限（{effective_timeout_seconds}秒），已取消",
                )
            ],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    except SearchProviderError as exc:
        return ExternalResearchResult(
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            status="failed",
            search_count=budget.total_used,
            warnings=[warning(exc.code, str(exc), retryable=exc.retryable)],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )
    except Exception as exc:
        return ExternalResearchResult(
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            status="failed",
            search_count=budget.total_used,
            warnings=[warning("RESEARCH_FAILED", f"外部研究失败: {exc}")],
            elapsed_ms=int((time.perf_counter() - started) * 1000),
        )

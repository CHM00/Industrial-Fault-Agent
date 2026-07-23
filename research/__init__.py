"""External research subsystem for the industrial fault diagnosis agent."""

from .configuration import ResearchSettings
from .contracts import (
    ExternalResearchResult,
    ExternalSource,
    ResearchOptions,
    ResearchTask,
    ResearchTaskResult,
    ResearchWarning,
)
from .gateway import run_external_research

__all__ = [
    "ExternalResearchResult",
    "ExternalSource",
    "ResearchOptions",
    "ResearchSettings",
    "ResearchTask",
    "ResearchTaskResult",
    "ResearchWarning",
    "run_external_research",
]

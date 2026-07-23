"""Mermaid extraction, normalization, and official-parser validation helpers."""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import uuid
import atexit
from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
VALIDATOR_SCRIPT = BASE_DIR / "validate_mermaid.mjs"
MAX_MERMAID_CHARS = int(os.environ.get("MERMAID_MAX_CHARS", "20000"))
VALIDATION_TIMEOUT_SECONDS = float(
    os.environ.get("MERMAID_VALIDATION_TIMEOUT_SECONDS", "8")
)
VALIDATOR_STARTUP_TIMEOUT_SECONDS = float(
    os.environ.get("MERMAID_VALIDATOR_STARTUP_TIMEOUT_SECONDS", "25")
)


class MermaidProcessingError(ValueError):
    """Raised when Mermaid content cannot safely enter the workflow state."""


@dataclass(frozen=True)
class MermaidValidationResult:
    valid: bool
    code: str
    error: str = ""
    diagram_type: str = ""

    @property
    def repairable(self) -> bool:
        return self.code == "SYNTAX_ERROR"

    def model_dump(self) -> dict:
        return {
            "valid": self.valid,
            "code": self.code,
            "error": self.error,
            "diagram_type": self.diagram_type,
        }


def extract_mermaid(content: str) -> str:
    """Extract the first Mermaid fenced block, then normalize it."""
    text = str(content or "")
    match = re.search(
        r"```[ \t]*(?:mermaid)?[ \t]*\r?\n(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    source = match.group(1) if match else text
    return normalize_mermaid(source)


def normalize_mermaid(source: str) -> str:
    """Apply deterministic cleanup without changing nodes or graph topology."""
    diagram = str(source or "").lstrip("\ufeff")
    diagram = diagram.replace("\r\n", "\n").replace("\r", "\n")
    diagram = re.sub(r"<br\s*>", "<br/>", diagram, flags=re.IGNORECASE)
    diagram = "\n".join(line.rstrip() for line in diagram.splitlines()).strip()

    if not diagram:
        raise MermaidProcessingError("模型未返回 Mermaid 流程图")
    if len(diagram) > MAX_MERMAID_CHARS:
        raise MermaidProcessingError(
            f"Mermaid 流程图超过长度限制（{len(diagram)} > {MAX_MERMAID_CHARS}）"
        )
    if not re.match(
        r"^(?:flowchart|graph)\s+(?:TD|TB|BT|LR|RL)\b",
        diagram,
        flags=re.IGNORECASE,
    ):
        raise MermaidProcessingError(
            "Mermaid 流程图必须以 flowchart TD 等合法方向声明开始"
        )

    forbidden = (
        (r"%%\{.*?\binit\b.*?\}%%", "初始化指令"),
        (r"^\s*click\s+", "click 交互指令"),
        (r"javascript\s*:", "JavaScript URL"),
        (r"<\s*script\b", "script 标签"),
    )
    for pattern, label in forbidden:
        if re.search(pattern, diagram, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL):
            raise MermaidProcessingError(f"Mermaid 流程图包含不允许的{label}")

    return diagram


class _ValidatorProcess:
    def __init__(self):
        self._lock = threading.Lock()
        self._responses: queue.Queue = queue.Queue()
        self._process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None

    def _start(self) -> bool:
        if self._process and self._process.poll() is None:
            return False
        self.close()
        self._responses = queue.Queue()
        self._process = subprocess.Popen(
            ["node", str(VALIDATOR_SCRIPT)],
            text=True,
            encoding="utf-8",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=str(BASE_DIR),
            shell=False,
            bufsize=1,
        )

        def read_stdout():
            assert self._process and self._process.stdout
            for line in self._process.stdout:
                self._responses.put(line)
            self._responses.put(None)

        self._reader = threading.Thread(
            target=read_stdout, name="mermaid-validator-reader", daemon=True
        )
        self._reader.start()
        return True

    def validate(self, source: str) -> dict:
        with self._lock:
            started = self._start()
            assert self._process and self._process.stdin
            request_id = uuid.uuid4().hex
            request = json.dumps({"id": request_id, "source": source}, ensure_ascii=False)
            try:
                self._process.stdin.write(request + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                self.close()
                raise RuntimeError(f"Mermaid 校验器进程已退出: {exc}") from exc
            timeout = VALIDATOR_STARTUP_TIMEOUT_SECONDS if started else VALIDATION_TIMEOUT_SECONDS
            try:
                line = self._responses.get(timeout=timeout)
            except queue.Empty as exc:
                self.close()
                raise TimeoutError(f"Mermaid 语法校验超过 {timeout:g} 秒") from exc
            if line is None:
                self.close()
                raise RuntimeError("Mermaid 校验器未返回结果")
            payload = json.loads(line)
            if payload.get("id") != request_id:
                self.close()
                raise RuntimeError("Mermaid 校验器响应标识不匹配")
            return payload

    def close(self) -> None:
        process = self._process
        self._process = None
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()


_validator = _ValidatorProcess()
atexit.register(_validator.close)


def validate_mermaid(source: str) -> MermaidValidationResult:
    """Validate Mermaid syntax through a warm official-parser process."""
    try:
        diagram = normalize_mermaid(source)
    except MermaidProcessingError as exc:
        return MermaidValidationResult(False, "NORMALIZATION_ERROR", str(exc))

    try:
        payload = _validator.validate(diagram)
    except TimeoutError as exc:
        return MermaidValidationResult(False, "VALIDATOR_TIMEOUT", str(exc))
    except (OSError, RuntimeError, json.JSONDecodeError, TypeError) as exc:
        return MermaidValidationResult(
            False, "VALIDATOR_UNAVAILABLE", f"Mermaid 校验器不可用：{str(exc)[:1000]}"
        )

    if payload.get("valid"):
        return MermaidValidationResult(
            True, "VALID", diagram_type=str(payload.get("diagramType") or "flowchart")
        )
    return MermaidValidationResult(
        False, "SYNTAX_ERROR", str(payload.get("error") or "Mermaid 语法错误")[:2000]
    )


def validator_health() -> dict:
    result = validate_mermaid('flowchart TD\n    health["health"]')
    return {"ok": result.valid, "code": result.code, "error": result.error}

import os
import json
import uuid
import asyncio
import threading
import datetime
import traceback
import time
import urllib.parse
import io
import sqlite3
import re
import fnmatch
import math
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import (
    FileResponse,
    StreamingResponse,
    JSONResponse,
    PlainTextResponse,
    RedirectResponse,
)
from pydantic import BaseModel, Field, field_validator

from langgraph.types import Command

import fault_agent
from research import ResearchOptions
from mcp_tools import (
    save_diagram_and_report,
    save_diagnosis_memory,
    search_diagnosis_memory,
    DIAGRAM_DIR,
    MEMORY_DIR,
)
from checkpointing import checkpointer_status
from governance import DiagnosisConcurrency, SlidingWindowRateLimiter
from mermaid_pipeline import validator_health
from observability import metrics
from runtime_store import JobRecord, JobStore
from security import ApiKeyAuth, current_principal, get_principal
from business_store import BusinessStore
from document_ingestion import DocumentIngestionError, MAX_UPLOAD_BYTES, extract_text
from artifact_export import export_artifact
from evidence_mapping import map_evidence
from mermaid_pipeline import validate_mermaid
from managed_knowledge_vector import ManagedKnowledgeVector

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MERMAID_DIST_DIR = BASE_DIR / "node_modules" / "mermaid" / "dist"
OML2D_DIST_DIR = BASE_DIR / "node_modules" / "oh-my-live2d" / "dist"
LIVE2D_SHIZUKU_DIR = BASE_DIR / "node_modules" / "live2d-widget-model-shizuku" / "assets"
_knowledge_files_config = Path(
    os.environ.get("KNOWLEDGE_FILES_DIR", "output/knowledge")
)
KNOWLEDGE_FILES_DIR = (
    _knowledge_files_config if _knowledge_files_config.is_absolute()
    else BASE_DIR / _knowledge_files_config
)

app = FastAPI(title="工业故障诊断 Agent", version="2.0")

# ---- build the LangGraph app once; all sessions share one MemorySaver,
#      differentiated by thread_id -------------------------------------------
_agent_app = fault_agent.build_app()
_checkpointer = getattr(_agent_app, "checkpointer", None)
_job_store = JobStore()
_business_store = BusinessStore(_job_store.path)
_managed_vectors = ManagedKnowledgeVector()
_job_store.expire_stale()
_job_store.reconcile_interrupted()
_auth = ApiKeyAuth()
_rate_limiter = SlidingWindowRateLimiter()
_diagnosis_slots = DiagnosisConcurrency()

# session registry: session_id -> {config, lock, first, pending_feedback}
_sessions: dict = {}
_sessions_lock = threading.Lock()


PUBLIC_PATHS = {"/", "/health/live", "/health/ready", "/api/auth/login", "/api/auth/logout"}


def _required_role(method: str, path: str) -> str:
    if path == "/metrics":
        return "admin"
    if method == "POST" and path == "/api/start":
        return "operator"
    if method in {"POST", "PATCH"} and path.startswith("/api/knowledge"):
        return "expert"
    if method == "POST" and (path.endswith("/approve") or path.endswith("/reject") or path.endswith("/publish") or path.endswith("/confirm")):
        return "expert"
    if path.startswith("/api/dashboard"):
        return "expert"
    if method in {"POST", "PATCH"} and path.startswith("/api/assets"):
        return "operator"
    if method == "POST" and path.startswith("/api/procedures"):
        return "operator"
    if method == "POST" and (path.startswith("/api/jobs/draft") or path.endswith("/start") or path.endswith("/archive") or path == "/api/jobs/archive-batch"):
        return "operator"
    if method == "POST" and (path == "/api/resume" or path.endswith("/cancel") or path.endswith("/retry")):
        return "operator"
    return "viewer"


@app.middleware("http")
async def security_governance_middleware(request: Request, call_next):
    started = time.perf_counter()
    path = request.url.path
    public = path in PUBLIC_PATHS or path.startswith("/static/") or path.startswith("/vendor/")
    principal = _auth.authenticate(
        request.headers.get("Authorization"),
        request.headers.get("X-API-Key"),
        request.cookies.get(_auth.session_cookie),
    )
    if not public:
        if not _auth.configured:
            return JSONResponse(
                status_code=503,
                content={"detail": "认证已启用但未配置用户或系统凭据"},
            )
        if principal is None:
            metrics.inc("auth_failures_total")
            return JSONResponse(status_code=401, content={"detail": "登录已失效或未登录"})
        required = _required_role(request.method, path)
        if not principal.can(required):
            metrics.inc("authorization_denied_total")
            return JSONResponse(status_code=403, content={"detail": f"需要 {required} 角色"})
        allowed, retry_after = _rate_limiter.allow(
            f"{principal.tenant_id}:{principal.subject}"
        )
        if not allowed:
            metrics.inc("rate_limited_total")
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={"detail": "请求过于频繁，请稍后重试"},
            )
    principal = principal or get_principal()
    token = current_principal.set(principal)
    try:
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Content-Security-Policy",
            # unsafe-eval is required by PixiJS shader compilation inside the Live2D widget.
            "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'",
        )
        metrics.inc(f"http_status_{response.status_code}_total")
        return response
    finally:
        current_principal.reset(token)
        metrics.inc("http_requests_total")
        metrics.inc("http_request_seconds_total", time.perf_counter() - started)


ROLE_CAPABILITIES = {
    "viewer": ["查看获授权的任务、历史记录和诊断产物"],
    "operator": ["发起、恢复、取消和重试诊断", "维护设备资产和流程草稿"],
    "expert": ["管理受控 SOP", "执行高风险审批", "审批流程并查看质量看板"],
    "admin": ["管理本租户全部数据", "访问系统运行指标"],
}


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=100)
    password: str = Field(min_length=1, max_length=500)


def _principal_payload(principal) -> dict:
    capabilities: list[str] = []
    for role in ("viewer", "operator", "expert", "admin"):
        if principal.can(role):
            capabilities.extend(ROLE_CAPABILITIES[role])
    return {
        "subject": principal.subject,
        "display_name": principal.display_name or principal.subject,
        "role": principal.role,
        "tenant_id": principal.tenant_id,
        "capabilities": capabilities,
        "allowed_roles": [
            role for role in ("viewer", "operator", "expert", "admin")
            if principal.can(role)
        ],
        "authenticated": principal.authenticated,
    }


@app.post("/api/auth/login")
def api_auth_login(payload: LoginRequest, request: Request):
    if not _auth.enabled:
        return {"user": _principal_payload(get_principal()), "authentication_disabled": True}
    if not _auth.login_enabled:
        raise HTTPException(status_code=503, detail="尚未配置网页登录账号")
    client_host = request.client.host if request.client else "unknown"
    allowed, retry_after = _rate_limiter.allow(f"login:{client_host}")
    if not allowed:
        raise HTTPException(status_code=429, detail=f"登录尝试过于频繁，请在 {retry_after} 秒后重试")
    principal = _auth.login(payload.username.strip(), payload.password)
    if principal is None:
        metrics.inc("auth_failures_total")
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    response = JSONResponse({"user": _principal_payload(principal)})
    response.set_cookie(
        _auth.session_cookie,
        _auth.issue_session(principal),
        max_age=_auth.session_ttl_seconds,
        httponly=True,
        secure=_auth.session_secure,
        samesite="lax",
        path="/",
    )
    return response


@app.post("/api/auth/logout")
def api_auth_logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(_auth.session_cookie, path="/")
    return response


@app.get("/api/auth/me")
def api_auth_me():
    return {"user": _principal_payload(get_principal())}

# node name -> human-readable label (Chinese, terminal-style)
NODE_LABELS = {
    "dispatch": "[dispatch] 初始化诊断会话",
    "assess_safety": "[safety] 执行工业安全预检",
    "safety_gate": "[safety] 检查高风险专家审批",
    "start_research": "[dispatch] 安全门通过，开始检索",
    "retrieve_internal": "[retrieve] 检索内部 SOP 知识库",
    "external_research": "[research] 执行统一外部研究",
    "filter": "[fuse] 内外知识分层融合",
    "generate": "[generate] 生成 Mermaid 故障排查流程图",
    "map_evidence": "[evidence] 建立流程节点与证据映射",
    "audit": "[audit] 审计流程图完整性与准确性",
    "evaluate_questions": "[evaluate] 汇总审计问题，等待专家反馈",
    "ask_expert": "[ask] 逐条收集专家反馈",
    "refine": "[refine] 根据专家反馈修订流程图",
}


def _sse(event: str, data: dict) -> str:
    """Format a single Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _extract_message(update: dict) -> str:
    """Pull the last human-readable message out of a node state update."""
    msgs = update.get("messages") or []
    if msgs:
        last = msgs[-1]
        content = getattr(last, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()
    # fall back to any non-empty string field
    for key in ("internal_knowledge", "filtered_context", "mermaid_diagram", "audit_result"):
        val = update.get(key)
        if isinstance(val, str) and val.strip():
            preview = val.strip().splitlines()[0][:120]
            return f"{key} <- {preview}"
    return ""


async def _stream_updates(payload, config):
    """Bridge LangGraph's sync stream into asyncio without blocking FastAPI.

    LangGraph's ``interrupt()`` relies on runnable context stored in a
    ContextVar.  On Python 3.10 that context is not preserved by ``astream()``,
    so the graph must be driven through its synchronous ``stream()`` API.  A
    dedicated worker thread keeps the synchronous iterator off the event loop,
    while an asyncio queue preserves incremental SSE delivery.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    stopped = threading.Event()
    graph = _agent_app

    def publish(kind: str, value=None) -> bool:
        if stopped.is_set():
            return False
        try:
            loop.call_soon_threadsafe(queue.put_nowait, (kind, value))
            return True
        except RuntimeError:
            # The request/event loop may already be closed after disconnect.
            return False

    def run_stream() -> None:
        try:
            for event in graph.stream(payload, config, stream_mode="updates"):
                if not publish("event", event):
                    break
        except Exception as exc:
            publish("error", exc)
        finally:
            publish("done")

    worker = threading.Thread(
        target=run_stream,
        name="langgraph-sync-stream",
        daemon=True,
    )
    worker.start()

    try:
        while True:
            kind, value = await queue.get()
            if kind == "event":
                yield value
            elif kind == "error":
                raise value
            else:
                return
    finally:
        stopped.set()


async def _get_state(config):
    # Keep all graph runtime access on the Python-3.10-safe synchronous API.
    return await asyncio.to_thread(_agent_app.get_state, config)


def _interrupt_question(state_snapshot) -> str:
    """Extract the question string from a paused interrupt, if any."""
    try:
        for task in state_snapshot.tasks:
            for intr in getattr(task, "interrupts", []) or []:
                val = getattr(intr, "value", None)
                if val:
                    return str(val)
    except Exception:
        pass
    return ""


def _interrupt_value(state_snapshot):
    try:
        for task in state_snapshot.tasks:
            for intr in getattr(task, "interrupts", []) or []:
                value = getattr(intr, "value", None)
                if value is not None:
                    return value
    except Exception:
        pass
    return None


def _structured_context(req: "StartReq", asset: dict | None) -> str:
    lines = []
    if asset:
        lines.extend([
            f"设备编号：{asset.get('asset_code', '')}",
            f"设备名称：{asset.get('name', '')}",
            f"类型/厂商/型号：{asset.get('asset_type', '')} / {asset.get('vendor', '')} / {asset.get('model', '')}",
            f"序列号/固件：{asset.get('serial_no', '')} / {asset.get('firmware', '')}",
            f"位置/关键度：{asset.get('location', '')} / {asset.get('criticality', '')}",
        ])
    if req.alarm_code:
        lines.append(f"报警码：{req.alarm_code}")
    if req.operating_context:
        lines.append(f"运行工况：{req.operating_context}")
    if req.measurements:
        lines.append("测点：\n" + _format_measurements(req.measurements, asset or {}))
    if req.maintenance_history:
        lines.append(f"近期维修/变更：{req.maintenance_history}")
    if req.attachments:
        lines.append("附件引用：" + "、".join(req.attachments))
    return "\n".join(lines)


def _format_measurements(measurements: dict, asset: dict) -> str:
    template = (asset.get("metadata") or {}).get("measurement_template", {})
    lines = []
    for name, raw in measurements.items():
        spec = template.get(name, {}) if isinstance(template, dict) else {}
        if isinstance(raw, dict):
            value = raw.get("value")
            unit = str(raw.get("unit") or spec.get("unit") or "")
        else:
            value = raw
            unit = str(spec.get("unit") or "")
        expected_unit = str(spec.get("unit") or "")
        if expected_unit and unit and unit != expected_unit:
            raise HTTPException(
                status_code=400,
                detail=f"测点 {name} 单位应为 {expected_unit}，实际为 {unit}",
            )
        note = ""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            low, high = spec.get("normal_min"), spec.get("normal_max")
            if low is not None and value < float(low):
                note = f"（低于正常下限 {low}{expected_unit}）"
            elif high is not None and value > float(high):
                note = f"（高于正常上限 {high}{expected_unit}）"
            elif low is not None or high is not None:
                note = "（正常范围内）"
        lines.append(f"- {name}：{value}{(' ' + unit) if unit else ''}{note}")
    return "\n".join(lines)


def _search_managed_knowledge(
    tenant_id: str, role: str, query: str, limit: int,
    asset_context: dict | None = None,
) -> tuple[list[dict], str, str]:
    """Use Milvus semantics first, retaining SQLite as a safe fallback."""
    if limit <= 0:
        return [], "disabled", ""
    accessible = {
        item["id"]: item
        for item in _business_store.list_documents(tenant_id, role)
        if item["status"] == "active"
        and _sop_applies_to_asset(item.get("applicability") or {}, asset_context or {})
    }
    allowed_ids = set(accessible)
    if not allowed_ids:
        return [], "none", "没有适用于当前设备且可访问的受控 SOP"
    if _managed_vectors.enabled:
        try:
            hits = _managed_vectors.search(
                tenant_id, role, query, min(20, max(limit, limit * 3)),
                document_ids=allowed_ids,
            )
            hits = [
                item for item in hits
                if item.get("document_id") in accessible
                and item.get("version_id")
                == accessible[item["document_id"]].get("current_version_id")
            ]
            if hits:
                return hits[:limit], "milvus", ""
        except Exception as exc:
            _managed_vectors.remember_error(exc)
            warning = f"Milvus 受控 SOP 检索失败，已降级到 SQLite：{exc}"
        else:
            warning = "Milvus 尚无匹配的受控 SOP 向量，已降级到 SQLite"
    else:
        warning = "受控 SOP Milvus 索引已禁用，使用 SQLite"
    hits = _business_store.search_knowledge(
        tenant_id, role, query, limit=limit, allowed_document_ids=allowed_ids
    )
    for item in hits:
        item["retrieval_backend"] = "sqlite"
    return hits, "sqlite", warning


def _sop_applies_to_asset(applicability: dict, asset: dict) -> bool:
    """Deterministic applicability gate before semantic ranking."""
    today = datetime.date.today().isoformat()
    if applicability.get("valid_from") and today < str(applicability["valid_from"]):
        return False
    if applicability.get("valid_to") and today > str(applicability["valid_to"]):
        return False
    for field in ("asset_type", "vendor", "model", "firmware"):
        expected = applicability.get(field)
        if not expected:
            continue
        actual = str(asset.get(field) or "").strip().casefold()
        if not actual:
            return False
        patterns = expected if isinstance(expected, list) else str(expected).split(",")
        if not any(
            fnmatch.fnmatch(actual, str(pattern).strip().casefold())
            for pattern in patterns if str(pattern).strip()
        ):
            return False
    return True


def _safe_path_segment(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._\-\u4e00-\u9fff]+", "_", str(value or ""))
    return cleaned.strip(" ._")[:160] or fallback


def _persist_knowledge_source(
    tenant_id: str, result: dict, filename: str, raw: bytes,
) -> str:
    safe_tenant = _safe_path_segment(tenant_id, "default")
    safe_name = _safe_path_segment(Path(filename).name, "document.bin")
    target_dir = (
        KNOWLEDGE_FILES_DIR / safe_tenant / result["document_id"]
        / f"v{result['version']}"
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / safe_name
    target.write_bytes(raw)
    try:
        return str(target.relative_to(BASE_DIR))
    except ValueError:
        return str(target)


def _sync_managed_document(tenant_id: str, document_id: str) -> dict:
    payload = _business_store.document_for_indexing(tenant_id, document_id)
    if not payload:
        raise KeyError("知识文档不存在")
    version_id = payload["version_id"]
    if not _managed_vectors.enabled:
        _business_store.update_version_artifacts(
            version_id, index_status="disabled", index_backend="sqlite",
            index_error="MANAGED_KNOWLEDGE_VECTOR_ENABLED=false",
        )
        return {"status": "disabled", "backend": "sqlite", "chunks": 0}
    _business_store.update_version_artifacts(
        version_id, index_status="indexing", index_backend="milvus", index_error="",
    )
    try:
        vector_result = _managed_vectors.sync_document(payload)
        status = "ready" if payload["document_status"] == "active" else "removed"
        _business_store.update_version_artifacts(
            version_id, index_status=status, index_backend="milvus", index_error="",
        )
        return {"status": status, **vector_result}
    except Exception as exc:
        _managed_vectors.remember_error(exc)
        _business_store.update_version_artifacts(
            version_id, index_status="failed", index_backend="milvus",
            index_error=str(exc),
        )
        return {
            "status": "failed", "backend": "sqlite",
            "chunks": 0, "error": str(exc),
            "warning": "Milvus 同步失败，文档已保存，诊断将降级使用 SQLite 检索",
        }


def _prepare_product_context(req: "StartReq", principal) -> dict:
    asset = None
    if req.asset_id:
        asset = _business_store.get_asset(principal.tenant_id, req.asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="设备资产不存在")
        if asset.get("status") != "active":
            raise HTTPException(status_code=409, detail="设备已停用或归档，不能发起新的诊断")
    structured = _structured_context(req, asset)
    query = "\n".join(filter(None, [
        req.fault_input, req.alarm_code, req.operating_context,
        json.dumps(req.measurements, ensure_ascii=False) if req.measurements else "",
        structured,
    ]))
    managed, managed_backend, managed_warning = _search_managed_knowledge(
        principal.tenant_id, principal.role, query, req.max_managed_knowledge_hits,
        asset_context=asset,
    )
    cases = (
        _business_store.search_cases(
            principal.tenant_id, query, asset_id=req.asset_id, limit=req.max_historical_cases
        )
        if req.max_historical_cases else []
    )
    return {
        "asset_context": asset or {},
        "structured_context": structured,
        "managed_knowledge_hits": managed,
        "managed_knowledge_backend": managed_backend,
        "managed_knowledge_warning": managed_warning,
        "historical_case_hits": cases,
    }


def _initial_state(req: "StartReq", request_id: str, product_context: dict | None = None) -> dict:
    options = ResearchOptions.model_validate(req.model_dump())
    return fault_agent.build_initial_state(
        req.fault_input.strip(),
        auto_mode=req.auto_mode,
        research_options=options,
        request_id=request_id,
        product_context=product_context,
    )


def _langfuse_handler():
    try:
        return fault_agent.get_langfuse_handler()
    except Exception:
        return None


def _new_session(req: "StartReq", request_id: str) -> tuple[str, dict]:
    _job_store.expire_stale()
    principal = get_principal()
    sid = uuid.uuid4().hex[:12]
    thread_id = f"web_{sid}"
    config = {"configurable": {"thread_id": thread_id}}
    handler = _langfuse_handler()
    if handler:
        config["callbacks"] = [handler]
    sess = {
        "config": config,
        "lock": asyncio.Lock(),
        "first": True,
        "pending_feedback": None,
        "auto_mode": req.auto_mode,
        "fault_input": req.fault_input.strip(),
        "request_id": request_id,
        "owner_id": principal.subject,
        "tenant_id": principal.tenant_id,
        "cancelled": False,
        "product_context": {},
    }
    with _sessions_lock:
        _sessions[sid] = sess
    _job_store.create(JobRecord(
        session_id=sid,
        request_id=request_id,
        thread_id=thread_id,
        owner_id=principal.subject,
        tenant_id=principal.tenant_id,
        status="queued",
        fault_input=req.fault_input.strip(),
        auto_mode=req.auto_mode,
        request_payload=req.model_dump(mode="json"),
    ))
    return sid, sess


def _restore_session(session_id: str) -> dict | None:
    _job_store.expire_stale()
    with _sessions_lock:
        cached = _sessions.get(session_id)
    if cached:
        return cached
    record = _job_store.get(session_id)
    if not record or record.status not in {"waiting_feedback", "waiting_safety", "running", "queued"}:
        return None
    config = {"configurable": {"thread_id": record.thread_id}}
    handler = _langfuse_handler()
    if handler:
        config["callbacks"] = [handler]
    sess = {
        "config": config,
        "lock": asyncio.Lock(),
        "first": False,
        "pending_feedback": None,
        "auto_mode": record.auto_mode,
        "fault_input": record.fault_input,
        "request_id": record.request_id,
        "owner_id": record.owner_id,
        "tenant_id": record.tenant_id,
        "cancelled": record.status == "cancel_requested",
    }
    with _sessions_lock:
        return _sessions.setdefault(session_id, sess)


def _assert_job_access(sess: dict, *, allow_expert: bool = False) -> None:
    principal = get_principal()
    if principal.tenant_id != sess["tenant_id"]:
        raise HTTPException(status_code=404, detail="session 不存在")
    elevated = principal.can("admin") or (allow_expert and principal.can("expert"))
    if principal.subject != sess["owner_id"] and not elevated:
        raise HTTPException(status_code=403, detail="无权访问该诊断任务")


def _save_outputs(fault_input: str, values: dict) -> dict:
    paths = {"diagram": None, "report": None, "memory": None}
    try:
        d, r = save_diagram_and_report(fault_input, values)
        paths["diagram"] = d
        paths["report"] = r
    except Exception as e:
        print(f"[web] save report failed: {e}")
    try:
        m = save_diagnosis_memory(fault_input, values)
        paths["memory"] = m
    except Exception as e:
        print(f"[web] save memory failed: {e}")
    return paths


def _elapsed_ms_for_job(session_id: str) -> int:
    record = _job_store.get(session_id)
    if not record:
        return 0
    try:
        started = datetime.datetime.fromisoformat(record.created_at)
        now = datetime.datetime.now(datetime.timezone.utc)
        return max(0, int((now - started).total_seconds() * 1000))
    except (TypeError, ValueError):
        return 0


def _complete_business_outputs(session_id: str, sess: dict, values: dict) -> dict:
    record = _job_store.get(session_id)
    payload = (record.request_payload if record else {}) or {}
    asset_id = str(payload.get("asset_id") or "")
    tenant_id = sess["tenant_id"]
    case = _business_store.save_case(session_id, tenant_id, asset_id, values)
    versions = _business_store.list_procedures(tenant_id, session_id)
    if versions:
        procedure = versions[0]
    else:
        procedure = _business_store.create_procedure_version(
            session_id, tenant_id, sess["owner_id"], values.get("mermaid_diagram", ""),
            "Agent 生成并完成审计的初始版本",
        )
    _business_store.record_metrics(
        session_id, tenant_id, asset_id, values, _elapsed_ms_for_job(session_id)
    )
    return {"case": case, "procedure": procedure}


def _summarize_done(values: dict, paths: dict) -> dict:
    return {
        "schema_version": "1.0",
        "mermaid_diagram": values.get("mermaid_diagram", ""),
        "mermaid_validation": values.get("mermaid_validation", {}),
        "audit_result": values.get("audit_result", ""),
        "audit_questions": values.get("audit_questions", []),
        "has_gaps": values.get("has_gaps", False),
        "revision_count": values.get("revision_count", 0),
        "internal_knowledge": values.get("internal_knowledge", ""),
        "external_knowledge": values.get("external_knowledge", ""),
        "filtered_context": values.get("filtered_context", ""),
        "external_result": values.get("external_result", {}),
        "external_sources": values.get("external_sources", []),
        "research_task_results": values.get("research_task_results", []),
        "research_warning": values.get("research_warning", ""),
        "internal_warning": values.get("internal_warning", ""),
        "safety_assessment": values.get("safety_assessment", {}),
        "safety_approval": values.get("safety_approval", {}),
        "asset_context": values.get("asset_context", {}),
        "structured_context": values.get("structured_context", ""),
        "managed_knowledge_hits": values.get("managed_knowledge_hits", []),
        "managed_knowledge_backend": values.get("managed_knowledge_backend", ""),
        "managed_knowledge_warning": values.get("managed_knowledge_warning", ""),
        "historical_case_hits": values.get("historical_case_hits", []),
        "evidence_mappings": values.get("evidence_mappings", []),
        "usage_events": values.get("usage_events", []),
        "paths": paths,
    }


# ============================ API models ====================================

def _validate_asset_metadata(value: dict) -> dict:
    template = value.get("measurement_template", {})
    if template and not isinstance(template, dict):
        raise ValueError("measurement_template 必须是 JSON 对象")
    for name, spec in template.items():
        if not isinstance(spec, dict):
            raise ValueError(f"测点模板 {name} 必须是对象")
        unknown = set(spec) - {"unit", "normal_min", "normal_max"}
        if unknown:
            raise ValueError(f"测点模板 {name} 包含未知字段：{', '.join(sorted(unknown))}")
        low, high = spec.get("normal_min"), spec.get("normal_max")
        if low is not None and not isinstance(low, (int, float)):
            raise ValueError(f"测点模板 {name} normal_min 必须是数字")
        if high is not None and not isinstance(high, (int, float)):
            raise ValueError(f"测点模板 {name} normal_max 必须是数字")
        if low is not None and high is not None and low > high:
            raise ValueError(f"测点模板 {name} 的正常下限不能大于上限")
    return value

class StartReq(ResearchOptions):
    research_mode: Literal["off", "basic", "supervisory"] = "basic"
    fault_input: str = Field(min_length=1, max_length=2000)
    auto_mode: bool = False
    asset_id: str = Field(default="", max_length=64)
    alarm_code: str = Field(default="", max_length=200)
    operating_context: str = Field(default="", max_length=3000)
    measurements: dict[str, Any] = Field(default_factory=dict)
    maintenance_history: str = Field(default="", max_length=4000)
    attachments: list[str] = Field(default_factory=list, max_length=20)
    max_managed_knowledge_hits: int = Field(default=5, ge=0, le=20)
    max_historical_cases: int = Field(default=3, ge=0, le=10)

    @field_validator("measurements")
    @classmethod
    def validate_measurements(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(value) > 50:
            raise ValueError("测点数量不能超过 50")
        for name, raw in value.items():
            if not str(name).strip() or len(str(name)) > 100:
                raise ValueError("测点名称不能为空且不能超过 100 字符")
            measured = raw.get("value") if isinstance(raw, dict) else raw
            if isinstance(measured, float) and not math.isfinite(measured):
                raise ValueError(f"测点 {name} 不能是 NaN 或无穷大")
            if isinstance(raw, dict) and "value" not in raw:
                raise ValueError(f"测点 {name} 对象必须包含 value")
            if not isinstance(measured, (str, int, float, bool, type(None))):
                raise ValueError(f"测点 {name} 的值类型不受支持")
        return value


class ResumeReq(BaseModel):
    session_id: str
    feedback: str = Field(default="", max_length=4000)
    approved: bool | None = None


class AssetCreateReq(BaseModel):
    asset_code: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    asset_type: str = Field(default="", max_length=100)
    vendor: str = Field(default="", max_length=100)
    model: str = Field(default="", max_length=100)
    serial_no: str = Field(default="", max_length=100)
    firmware: str = Field(default="", max_length=100)
    criticality: Literal["low", "medium", "high", "critical"] = "medium"
    location: str = Field(default="", max_length=200)
    metadata: dict = Field(default_factory=dict)
    parent_id: str = Field(default="", max_length=64)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict) -> dict:
        return _validate_asset_metadata(value)


class AssetUpdateReq(BaseModel):
    asset_code: str | None = Field(default=None, max_length=100)
    name: str | None = Field(default=None, max_length=200)
    asset_type: str | None = Field(default=None, max_length=100)
    vendor: str | None = Field(default=None, max_length=100)
    model: str | None = Field(default=None, max_length=100)
    serial_no: str | None = Field(default=None, max_length=100)
    firmware: str | None = Field(default=None, max_length=100)
    criticality: Literal["low", "medium", "high", "critical"] | None = None
    location: str | None = Field(default=None, max_length=200)
    metadata: dict | None = None
    parent_id: str | None = Field(default=None, max_length=64)

    @field_validator("metadata")
    @classmethod
    def validate_metadata(cls, value: dict | None) -> dict | None:
        return _validate_asset_metadata(value) if value is not None else None


class AssetImportReq(BaseModel):
    source: str = Field(default="external", max_length=100)
    assets: list[AssetCreateReq] = Field(min_length=1, max_length=1000)
    status: Literal["active", "inactive", "archived"] | None = None


class KnowledgeStatusReq(BaseModel):
    status: Literal["active", "inactive", "archived"]


class CaseConfirmReq(BaseModel):
    confirmed_root_cause: str = Field(min_length=1, max_length=4000)
    resolution: str = Field(min_length=1, max_length=6000)
    outcome: Literal["resolved", "partially_resolved", "unresolved"] = "resolved"
    credibility: float = Field(default=0.9, ge=0, le=1)
    expert_rating: int = Field(default=5, ge=1, le=5)


class ProcedureVersionReq(BaseModel):
    mermaid: str = Field(min_length=10, max_length=30000)
    change_summary: str = Field(default="", max_length=2000)


class ProcedureDecisionReq(BaseModel):
    note: str = Field(default="", max_length=2000)


class ProcedureAdoptReq(BaseModel):
    left_version_id: str = Field(min_length=1, max_length=64)
    right_version_id: str = Field(min_length=1, max_length=64)
    accepted_hunks: list[int] = Field(default_factory=list, max_length=200)
    change_summary: str = Field(default="逐条采纳流程差异", max_length=2000)


class JobBatchArchiveReq(BaseModel):
    session_ids: list[str] = Field(min_length=1, max_length=200)


# ============================ SSE: start ====================================

@app.post("/api/start")
async def api_start(req: StartReq):
    fault = (req.fault_input or "").strip()
    if not fault:
        raise HTTPException(status_code=400, detail="fault_input 不能为空")
    request_id = uuid.uuid4().hex
    product_context = _prepare_product_context(req, get_principal())
    sid, sess = _new_session(req, request_id)
    sess["product_context"] = product_context
    _business_store.record_measurements(
        get_principal().tenant_id, req.asset_id, sid, req.measurements,
        product_context.get("asset_context") or {},
    )
    config = sess["config"]

    async def gen():
        def emit(event: str, data: dict) -> str:
            return _sse(event, {"request_id": request_id, **data})

        acquired = False
        try:
            yield emit("queued", {"session_id": sid, "max_concurrency": _diagnosis_slots.maximum})
            acquired = await asyncio.to_thread(_diagnosis_slots.acquire, 30)
            if not acquired:
                _job_store.update(sid, status="failed", error_code="QUEUE_TIMEOUT")
                yield emit("error", {"code": "QUEUE_TIMEOUT", "message": "诊断队列等待超时，请稍后重试"})
                return
            metrics.set("active_diagnoses", _diagnosis_slots.active)
            _job_store.update(sid, actor_id=sess["owner_id"], status="running")
            async with sess["lock"]:
              try:
                payload = _initial_state(req, request_id, sess.get("product_context"))
                yield emit(
                    "research_started",
                    {
                        "requested_mode": req.research_mode,
                        "effective_mode": req.research_mode,
                        "max_total_searches": req.max_total_searches,
                        "research_timeout_seconds": req.research_timeout_seconds,
                    },
                )
                while True:
                    async for event in _stream_updates(payload, config):
                        if sess.get("cancelled"):
                            _job_store.update(sid, actor_id=sess["owner_id"], status="cancelled")
                            yield emit("cancelled", {"session_id": sid})
                            return
                        for node, update in event.items():
                            if node == "__interrupt__":
                                continue
                            if not isinstance(update, dict):
                                continue
                            label = NODE_LABELS.get(node, f"[{node}]")
                            msg = _extract_message(update)
                            if not msg:
                                msg = label
                            yield emit("progress", {"node": node, "label": label, "message": msg})
                            if node == "external_research":
                                result = update.get("external_result", {})
                                yield emit(
                                    "research_completed",
                                    {
                                        "effective_mode": result.get("effective_mode"),
                                        "status": result.get("status"),
                                        "query_count": result.get("query_count", 0),
                                        "search_count": result.get("search_count", 0),
                                        "source_count": len(result.get("sources", [])),
                                        "elapsed_ms": result.get("elapsed_ms", 0),
                                    },
                                )
                                for item in result.get("warnings", []):
                                    yield emit("research_warning", item)

                    snap = await _get_state(config)
                    values = snap.values or {}

                    if snap.next:  # paused at an interrupt (expert question)
                        interrupt_value = _interrupt_value(snap)
                        if isinstance(interrupt_value, dict) and interrupt_value.get("kind") == "safety_approval":
                            _job_store.update(
                                sid,
                                actor_id=sess["owner_id"],
                                status="waiting_safety",
                                pending_kind="safety_approval",
                            )
                            yield emit("safety_approval_required", {
                                "session_id": sid,
                                **interrupt_value,
                            })
                            return
                        question = str(interrupt_value or "请提供专家反馈")
                        idx = values.get("current_question_idx", 0)
                        total = len(values.get("audit_questions", []))
                        qpayload = {
                            "session_id": sid,
                            "question": question,
                            "question_index": idx,
                            "total": total,
                            "diagram": values.get("mermaid_diagram", ""),
                            "revision_count": values.get("revision_count", 0),
                            "audit_questions": values.get("audit_questions", []),
                        }
                        if req.auto_mode:
                            auto_fb = f"针对「{question}」，请自动补充相关的判断步骤和异常路径。"
                            yield emit("progress", {
                                "node": "auto",
                                "label": "[auto] 自动模式反馈",
                                "message": auto_fb,
                            })
                            payload = Command(resume=auto_fb)
                            continue
                        else:
                            _job_store.update(
                                sid,
                                actor_id=sess["owner_id"],
                                status="waiting_feedback",
                                pending_kind="expert_question",
                            )
                            yield emit("expert_question", qpayload)
                            return
                    else:
                        if values.get("safety_denied"):
                            _job_store.update(
                                sid, actor_id=sess["owner_id"], status="denied", pending_kind=""
                            )
                            yield emit("denied", {
                                "session_id": sid,
                                "message": "高风险诊断未获专家批准，任务已安全停止",
                                "safety_assessment": values.get("safety_assessment", {}),
                            })
                            return
                        paths = await asyncio.to_thread(_save_outputs, fault, values)
                        business = await asyncio.to_thread(
                            _complete_business_outputs, sid, sess, values
                        )
                        _job_store.update(
                            sid,
                            actor_id=sess["owner_id"],
                            status="pending_approval",
                            pending_kind="",
                            result_paths=paths,
                        )
                        yield emit("done", {
                            "session_id": sid,
                            "fault_input": fault,
                            "business": business,
                            **_summarize_done(values, paths),
                        })
                        with _sessions_lock:
                            _sessions.pop(sid, None)
                        return
              except asyncio.CancelledError:
                _job_store.update(sid, actor_id=sess["owner_id"], status="cancelled")
                with _sessions_lock:
                    _sessions.pop(sid, None)
                raise
              except TimeoutError as exc:
                print(f"[web] request {request_id} timed out: {exc}")
                _job_store.update(
                    sid,
                    actor_id=sess["owner_id"],
                    status="failed",
                    error_code="EXECUTION_TIMEOUT",
                )
                with _sessions_lock:
                    _sessions.pop(sid, None)
                yield emit("error", {
                    "code": "EXECUTION_TIMEOUT",
                    "message": "诊断节点执行超时，任务已安全停止，可稍后重试",
                })
                return
              except Exception as exc:
                print(f"[web] request {request_id} failed: {exc}")
                traceback.print_exc()
                _job_store.update(sid, actor_id=sess["owner_id"], status="failed", error_code="DIAGNOSIS_FAILED")
                with _sessions_lock:
                    _sessions.pop(sid, None)
                yield emit("error", {"code": "DIAGNOSIS_FAILED", "message": "诊断失败，请检查服务日志"})
                return
        finally:
            if acquired:
                _diagnosis_slots.release()
                metrics.set("active_diagnoses", _diagnosis_slots.active)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# ============================ SSE: resume ===================================

@app.post("/api/resume")
async def api_resume(req: ResumeReq):
    sess = _restore_session(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session 不存在或已过期，请重新开始诊断")
    _assert_job_access(sess, allow_expert=True)
    principal = get_principal()
    record = _job_store.get(req.session_id)
    if not record or record.status not in {"waiting_feedback", "waiting_safety"}:
        raise HTTPException(status_code=409, detail="任务当前状态不可恢复")
    config = sess["config"]
    fault = sess["fault_input"]
    request_id = sess["request_id"]
    feedback = (req.feedback or "").strip() or f"跳过该问题（用户未提供反馈）"
    if record.status == "waiting_safety":
        if not principal.can("expert"):
            raise HTTPException(status_code=403, detail="高风险安全门需要 expert 角色批准或拒绝")
        if req.approved is None:
            raise HTTPException(status_code=400, detail="安全审批必须明确提交 approved=true/false")
        resume_value = {
            "approved": req.approved,
            "feedback": feedback,
            "actor": principal.subject,
        }
    else:
        resume_value = feedback

    async def gen():
        def emit(event: str, data: dict) -> str:
            return _sse(event, {"request_id": request_id, **data})

        acquired = await asyncio.to_thread(_diagnosis_slots.acquire, 30)
        if not acquired:
            yield emit("error", {"code": "QUEUE_TIMEOUT", "message": "诊断队列等待超时，请稍后重试"})
            return
        metrics.set("active_diagnoses", _diagnosis_slots.active)
        try:
          async with sess["lock"]:
            try:
                latest = _job_store.get(req.session_id)
                if not latest or latest.status not in {"waiting_feedback", "waiting_safety"}:
                    yield emit("error", {"code": "DUPLICATE_RESUME", "message": "任务已由其他请求恢复"})
                    return
                if latest.status == "waiting_safety":
                    _job_store.add_event(
                        req.session_id,
                        principal.subject,
                        "safety_decision",
                        {"approved": req.approved, "feedback": feedback[:2000]},
                    )
                else:
                    _job_store.add_event(
                        req.session_id,
                        principal.subject,
                        "expert_feedback",
                        {"feedback": feedback[:2000]},
                    )
                _job_store.update(
                    req.session_id,
                    actor_id=principal.subject,
                    status="running",
                    pending_kind="",
                )
                payload = Command(resume=resume_value)
                while True:
                    async for event in _stream_updates(payload, config):
                        if sess.get("cancelled"):
                            _job_store.update(req.session_id, status="cancelled")
                            yield emit("cancelled", {"session_id": req.session_id})
                            return
                        for node, update in event.items():
                            if node == "__interrupt__":
                                continue
                            if not isinstance(update, dict):
                                continue
                            label = NODE_LABELS.get(node, f"[{node}]")
                            msg = _extract_message(update)
                            if not msg:
                                msg = label
                            yield emit("progress", {"node": node, "label": label, "message": msg})

                    snap = await _get_state(config)
                    values = snap.values or {}

                    if snap.next:
                        interrupt_value = _interrupt_value(snap)
                        if isinstance(interrupt_value, dict) and interrupt_value.get("kind") == "safety_approval":
                            _job_store.update(
                                req.session_id,
                                actor_id=principal.subject,
                                status="waiting_safety",
                                pending_kind="safety_approval",
                            )
                            yield emit("safety_approval_required", {
                                "session_id": req.session_id,
                                **interrupt_value,
                            })
                            return
                        question = str(interrupt_value or "请提供专家反馈")
                        idx = values.get("current_question_idx", 0)
                        total = len(values.get("audit_questions", []))
                        _job_store.update(
                            req.session_id,
                            actor_id=principal.subject,
                            status="waiting_feedback",
                            pending_kind="expert_question",
                        )
                        yield emit("expert_question", {
                            "session_id": req.session_id,
                            "question": question,
                            "question_index": idx,
                            "total": total,
                            "diagram": values.get("mermaid_diagram", ""),
                            "revision_count": values.get("revision_count", 0),
                            "audit_questions": values.get("audit_questions", []),
                        })
                        return
                    else:
                        if values.get("safety_denied"):
                            _job_store.update(
                                req.session_id,
                                actor_id=principal.subject,
                                status="denied",
                                pending_kind="",
                            )
                            yield emit("denied", {
                                "session_id": req.session_id,
                                "message": "高风险诊断未获专家批准，任务已安全停止",
                                "safety_assessment": values.get("safety_assessment", {}),
                            })
                            return
                        paths = await asyncio.to_thread(_save_outputs, fault, values)
                        business = await asyncio.to_thread(
                            _complete_business_outputs, req.session_id, sess, values
                        )
                        _job_store.update(
                            req.session_id,
                            actor_id=principal.subject,
                            status="pending_approval",
                            pending_kind="",
                            result_paths=paths,
                        )
                        yield emit("done", {
                            "session_id": req.session_id,
                            "fault_input": fault,
                            "business": business,
                            **_summarize_done(values, paths),
                        })
                        # session complete -> clean up
                        with _sessions_lock:
                            _sessions.pop(req.session_id, None)
                        return
            except asyncio.CancelledError:
                _job_store.update(req.session_id, status="cancelled")
                raise
            except TimeoutError as exc:
                print(f"[web] resume request {request_id} timed out: {exc}")
                _job_store.update(
                    req.session_id,
                    actor_id=principal.subject,
                    status="failed",
                    error_code="EXECUTION_TIMEOUT",
                )
                yield emit("error", {
                    "code": "EXECUTION_TIMEOUT",
                    "message": "诊断节点执行超时，任务已安全停止，可稍后重试",
                })
                return
            except Exception as exc:
                print(f"[web] resume request {request_id} failed: {exc}")
                traceback.print_exc()
                _job_store.update(
                    req.session_id,
                    actor_id=principal.subject,
                    status="failed",
                    error_code="RESUME_FAILED",
                )
                yield emit("error", {"code": "RESUME_FAILED", "message": "恢复诊断失败，请检查服务日志"})
                return
        finally:
            _diagnosis_slots.release()
            metrics.set("active_diagnoses", _diagnosis_slots.active)

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# ============================ P1 business APIs ===============================


def _job_for_principal(session_id: str, *, expert_access: bool = False) -> JobRecord:
    record = _job_store.get(session_id)
    if not record:
        raise HTTPException(status_code=404, detail="任务不存在")
    principal = get_principal()
    if record.tenant_id != principal.tenant_id:
        raise HTTPException(status_code=404, detail="任务不存在")
    elevated = principal.can("admin") or (expert_access and principal.can("expert"))
    if record.owner_id != principal.subject and not elevated:
        raise HTTPException(status_code=403, detail="无权访问该任务")
    return record


def _state_for_record(record: JobRecord) -> dict:
    snapshot = _agent_app.get_state(
        {"configurable": {"thread_id": record.thread_id}}
    )
    values = snapshot.values or {}
    if not values:
        raise HTTPException(status_code=409, detail="任务状态不可用，请确认 checkpoint 数据完整")
    return values


@app.get("/api/assets")
def api_assets(include_inactive: bool = False):
    principal = get_principal()
    return {"assets": _business_store.list_assets(principal.tenant_id, include_inactive)}


def _validate_asset_parent(tenant_id: str, asset_id: str, parent_id: str) -> None:
    seen = {asset_id} if asset_id else set()
    current = parent_id
    while current:
        if current in seen:
            raise HTTPException(status_code=400, detail="设备层级不能形成循环")
        seen.add(current)
        parent = _business_store.get_asset(tenant_id, current)
        if not parent:
            raise HTTPException(status_code=400, detail="上级设备不存在")
        current = str(parent.get("parent_id") or "")


@app.post("/api/assets")
def api_create_asset(req: AssetCreateReq):
    principal = get_principal()
    if req.parent_id:
        _validate_asset_parent(principal.tenant_id, "", req.parent_id)
    try:
        return {"asset": _business_store.create_asset(principal.tenant_id, req.model_dump())}
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="同一租户下设备编号已存在")


@app.patch("/api/assets/{asset_id}")
def api_update_asset(asset_id: str, req: AssetUpdateReq):
    principal = get_principal()
    if req.parent_id:
        _validate_asset_parent(principal.tenant_id, asset_id, req.parent_id)
    try:
        asset = _business_store.update_asset(
            principal.tenant_id, asset_id, req.model_dump(exclude_none=True)
        )
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="设备编号冲突")
    if not asset:
        raise HTTPException(status_code=404, detail="设备不存在")
    return {"asset": asset}


@app.post("/api/assets/import")
def api_import_assets(req: AssetImportReq):
    """Generic CMMS/EAM/ERP bridge: idempotent upsert by tenant + asset_code."""
    principal = get_principal()
    created, updated, errors = [], [], []
    for item in req.assets:
        payload = item.model_dump()
        payload["metadata"] = {**payload.get("metadata", {}), "import_source": req.source}
        try:
            current = _business_store.get_asset_by_code(
                principal.tenant_id, item.asset_code
            )
            if item.parent_id:
                _validate_asset_parent(
                    principal.tenant_id, current["id"] if current else "", item.parent_id
                )
            if current:
                result = _business_store.update_asset(
                    principal.tenant_id, current["id"], payload
                )
                updated.append(result)
            else:
                created.append(_business_store.create_asset(principal.tenant_id, payload))
        except Exception as exc:
            errors.append({"asset_code": item.asset_code, "error": str(exc)})
    return {"created": created, "updated": updated, "errors": errors}


@app.get("/api/assets/{asset_id}/detail")
def api_asset_detail(asset_id: str):
    principal = get_principal()
    detail = _business_store.asset_detail(principal.tenant_id, asset_id)
    if not detail:
        raise HTTPException(status_code=404, detail="设备不存在")
    return detail


@app.get("/api/knowledge/documents")
def api_knowledge_documents():
    principal = get_principal()
    return {
        "documents": _business_store.list_documents(principal.tenant_id, principal.role),
        "vector_index": _managed_vectors.health(),
        "builtin_index": fault_agent.builtin_milvus_status(),
    }


@app.post("/api/knowledge/documents")
async def api_import_knowledge_document(
    request: Request,
    title: str,
    filename: str,
    min_role: Literal["viewer", "operator", "expert", "admin"] = "viewer",
    document_id: str = "",
    change_summary: str = "",
    applicability_json: str = "",
):
    principal = get_principal()
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail=f"文件超过上传限制：{MAX_UPLOAD_BYTES} 字节")
        chunks.append(chunk)
    raw = b"".join(chunks)
    try:
        applicability = json.loads(applicability_json) if applicability_json else {}
        if not isinstance(applicability, dict):
            raise ValueError("SOP 适用范围必须是 JSON 对象")
        allowed_applicability = {
            "asset_type", "vendor", "model", "firmware", "valid_from", "valid_to"
        }
        unknown = set(applicability) - allowed_applicability
        if unknown:
            raise ValueError(f"未知的 SOP 适用范围字段：{', '.join(sorted(unknown))}")
        for field in ("valid_from", "valid_to"):
            if applicability.get(field):
                datetime.date.fromisoformat(str(applicability[field]))
        if applicability.get("valid_from") and applicability.get("valid_to"):
            if str(applicability["valid_from"]) > str(applicability["valid_to"]):
                raise ValueError("SOP 生效日期不能晚于失效日期")
        content, file_type = extract_text(filename, raw)
        result = _business_store.import_document(
            principal.tenant_id, principal.subject, title.strip() or filename,
            filename, file_type, content, min_role=min_role,
            document_id=document_id or None, change_summary=change_summary,
            applicability=applicability,
        )
        source_path = _persist_knowledge_source(
            principal.tenant_id, result, filename, raw
        )
        _business_store.update_version_artifacts(
            result["version_id"], source_path=source_path
        )
        vector_index = await asyncio.to_thread(
            _sync_managed_document, principal.tenant_id, result["document_id"]
        )
        result["source_path"] = source_path
        result["vector_index"] = vector_index
        return result
    except DocumentIngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/api/knowledge/documents/{document_id}/versions")
def api_knowledge_versions(document_id: str):
    principal = get_principal()
    versions = _business_store.document_versions(principal.tenant_id, document_id, principal.role)
    if not versions:
        raise HTTPException(status_code=404, detail="知识文档不存在或无权访问")
    return {"versions": versions}


@app.get("/api/knowledge/documents/{document_id}/versions/{version_id}/source")
def api_knowledge_source(document_id: str, version_id: str):
    principal = get_principal()
    versions = _business_store.document_versions(
        principal.tenant_id, document_id, principal.role
    )
    version = next((item for item in versions if item["id"] == version_id), None)
    if not version:
        raise HTTPException(status_code=404, detail="知识文档版本不存在或无权访问")
    source_path = str(version.get("source_path") or "")
    if not source_path:
        raise HTTPException(status_code=404, detail="该历史版本未保存原文件")
    target = Path(source_path)
    if not target.is_absolute():
        target = BASE_DIR / target
    target = target.resolve()
    root = KNOWLEDGE_FILES_DIR.resolve()
    if target != root and root not in target.parents:
        raise HTTPException(status_code=400, detail="原文件路径无效")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="原文件不存在")
    return FileResponse(target, filename=target.name, media_type="application/octet-stream")


@app.patch("/api/knowledge/documents/{document_id}/status")
async def api_knowledge_status(document_id: str, req: KnowledgeStatusReq):
    principal = get_principal()
    if not _business_store.set_document_status(principal.tenant_id, document_id, req.status):
        raise HTTPException(status_code=404, detail="知识文档不存在")
    vector_index = await asyncio.to_thread(
        _sync_managed_document, principal.tenant_id, document_id
    )
    return {"ok": True, "status": req.status, "vector_index": vector_index}


@app.post("/api/knowledge/documents/{document_id}/reindex")
async def api_knowledge_reindex(document_id: str):
    principal = get_principal()
    if not _business_store.document_for_indexing(principal.tenant_id, document_id):
        raise HTTPException(status_code=404, detail="知识文档不存在")
    vector_index = await asyncio.to_thread(
        _sync_managed_document, principal.tenant_id, document_id
    )
    return {"ok": vector_index.get("status") == "ready", "vector_index": vector_index}


@app.get("/api/knowledge/search")
def api_knowledge_search(q: str, limit: int = 5, asset_id: str = ""):
    principal = get_principal()
    asset = None
    if asset_id:
        asset = _business_store.get_asset(principal.tenant_id, asset_id)
        if not asset:
            raise HTTPException(status_code=404, detail="设备资产不存在")
    results, backend, warning = _search_managed_knowledge(
        principal.tenant_id, principal.role, q, max(1, min(limit, 20)), asset
    )
    return {
        "results": results, "backend": backend, "warning": warning,
    }


@app.post("/api/cases/{session_id}/confirm")
def api_confirm_case(session_id: str, req: CaseConfirmReq):
    record = _job_for_principal(session_id, expert_access=True)
    case = _business_store.confirm_case(record.tenant_id, session_id, req.model_dump())
    if not case:
        raise HTTPException(status_code=404, detail="诊断案例不存在")
    _job_store.add_event(session_id, get_principal().subject, "case_confirmed", req.model_dump())
    return {"case": case}


@app.get("/api/procedures/{session_id}/versions")
def api_procedure_versions(session_id: str):
    record = _job_for_principal(session_id, expert_access=True)
    return {"versions": _business_store.list_procedures(record.tenant_id, session_id)}


@app.post("/api/procedures/{session_id}/versions")
def api_create_procedure_version(session_id: str, req: ProcedureVersionReq):
    record = _job_for_principal(session_id, expert_access=True)
    validation = validate_mermaid(req.mermaid)
    if not validation.valid:
        raise HTTPException(
            status_code=400,
            detail={"code": validation.code, "message": validation.error},
        )
    version = _business_store.create_procedure_version(
        session_id, record.tenant_id, get_principal().subject,
        req.mermaid, req.change_summary,
    )
    _job_store.update(session_id, actor_id=get_principal().subject, status="pending_approval")
    return {"version": version, "validation": validation.model_dump()}


@app.get("/api/procedures/{session_id}/diff")
def api_procedure_diff(session_id: str, left: str, right: str):
    record = _job_for_principal(session_id, expert_access=True)
    try:
        return {
            "diff": _business_store.procedure_diff(record.tenant_id, left, right),
            "hunks": _business_store.procedure_diff_hunks(record.tenant_id, left, right),
        }
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.post("/api/procedures/{session_id}/adopt")
def api_adopt_procedure_hunks(session_id: str, req: ProcedureAdoptReq):
    record = _job_for_principal(session_id, expert_access=True)
    left = _business_store.procedure_version(record.tenant_id, req.left_version_id)
    right = _business_store.procedure_version(record.tenant_id, req.right_version_id)
    if not left or not right or left["session_id"] != session_id or right["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="流程版本不存在")
    try:
        mermaid = _business_store.adopt_procedure_hunks(
            record.tenant_id, req.left_version_id, req.right_version_id, req.accepted_hunks
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    validation = validate_mermaid(mermaid)
    if not validation.valid:
        raise HTTPException(status_code=400, detail={"code": validation.code, "message": validation.error})
    version = _business_store.create_procedure_version(
        session_id, record.tenant_id, get_principal().subject, mermaid, req.change_summary
    )
    _job_store.update(session_id, actor_id=get_principal().subject, status="pending_approval")
    _job_store.add_event(session_id, get_principal().subject, "procedure_hunks_adopted", {
        "left_version_id": req.left_version_id, "right_version_id": req.right_version_id,
        "accepted_hunks": req.accepted_hunks, "new_version_id": version["id"],
    })
    return {"version": version, "validation": validation.model_dump()}


@app.post("/api/procedures/{session_id}/versions/{version_id}/approve")
def api_approve_procedure(session_id: str, version_id: str, req: ProcedureDecisionReq):
    record = _job_for_principal(session_id, expert_access=True)
    current = _business_store.procedure_version(record.tenant_id, version_id)
    if not current or current["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="流程版本不存在")
    if current["status"] != "draft":
        raise HTTPException(status_code=409, detail="只有草稿版本可以批准")
    version = _business_store.decide_procedure(
        record.tenant_id, version_id, get_principal().subject, "approved", req.note
    )
    _job_store.add_event(session_id, get_principal().subject, "procedure_approved", {"version_id": version_id, "note": req.note})
    return {"version": version}


@app.post("/api/procedures/{session_id}/versions/{version_id}/reject")
def api_reject_procedure(session_id: str, version_id: str, req: ProcedureDecisionReq):
    record = _job_for_principal(session_id, expert_access=True)
    current = _business_store.procedure_version(record.tenant_id, version_id)
    if not current or current["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="流程版本不存在")
    if current["status"] not in {"draft", "approved"}:
        raise HTTPException(status_code=409, detail="当前版本不可拒绝")
    version = _business_store.decide_procedure(
        record.tenant_id, version_id, get_principal().subject, "rejected", req.note
    )
    _job_store.add_event(session_id, get_principal().subject, "procedure_rejected", {"version_id": version_id, "note": req.note})
    return {"version": version}


@app.post("/api/procedures/{session_id}/versions/{version_id}/publish")
def api_publish_procedure(session_id: str, version_id: str, req: ProcedureDecisionReq):
    record = _job_for_principal(session_id, expert_access=True)
    current = _business_store.procedure_version(record.tenant_id, version_id)
    if not current or current["session_id"] != session_id:
        raise HTTPException(status_code=404, detail="流程版本不存在")
    if current["status"] != "approved":
        raise HTTPException(status_code=409, detail="流程版本必须先批准才能发布")
    version = _business_store.decide_procedure(
        record.tenant_id, version_id, get_principal().subject, "published", req.note
    )
    _job_store.update(session_id, actor_id=get_principal().subject, status="completed")
    _job_store.add_event(session_id, get_principal().subject, "procedure_published", {"version_id": version_id, "note": req.note})
    return {"version": version}


@app.get("/api/jobs/{session_id}/export")
def api_export_job(session_id: str, format: Literal["docx", "pdf", "checklist"] = "docx"):
    record = _job_for_principal(session_id, expert_access=True)
    state = dict(_state_for_record(record))
    procedures = _business_store.list_procedures(record.tenant_id, session_id)
    selected = next((item for item in procedures if item["status"] == "published"), None)
    selected = selected or (procedures[0] if procedures else None)
    if selected:
        state["mermaid_diagram"] = selected["mermaid"]
        state["procedure_context"] = (
            f"版本：v{selected['version']}\n状态：{selected['status']}\n"
            f"作者：{selected['author_id']}\n审批/发布人：{selected['approver_id']}\n"
            f"变更说明：{selected['change_summary']}\n决定说明：{selected['decision_note']}"
        )
        catalog = list(state.get("managed_knowledge_hits", []) or [])
        catalog.extend(state.get("historical_case_hits", []) or [])
        for source in state.get("external_sources", []) or []:
            catalog.append({
                **source, "evidence_id": source.get("source_id", ""),
                "source_type": "external_research", "trust_level": "reference",
            })
        state["evidence_mappings"] = map_evidence(selected["mermaid"], catalog)
    try:
        data, filename, media_type = export_artifact(format, state)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail=f"导出依赖未安装：{exc}")
    encoded = urllib.parse.quote(filename)
    return StreamingResponse(
        io.BytesIO(data),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{encoded}"},
    )


@app.get("/api/dashboard/quality")
def api_quality_dashboard():
    return _business_store.dashboard(get_principal().tenant_id)


@app.post("/api/jobs/draft")
def api_create_draft(req: StartReq):
    principal = get_principal()
    sid, request_id = uuid.uuid4().hex[:12], uuid.uuid4().hex
    record = _job_store.create(JobRecord(
        session_id=sid, request_id=request_id, thread_id=f"draft_{sid}",
        owner_id=principal.subject, tenant_id=principal.tenant_id, status="draft",
        fault_input=req.fault_input.strip(), auto_mode=req.auto_mode,
        request_payload=req.model_dump(mode="json"),
    ))
    return {"job": record.public_dict()}


@app.post("/api/jobs/{session_id}/start")
async def api_start_draft(session_id: str):
    record = _job_for_principal(session_id)
    if record.status != "draft":
        raise HTTPException(status_code=409, detail="只有草稿任务可以启动")
    request = StartReq.model_validate(record.request_payload)
    response = await api_start(request)
    _job_store.update(session_id, actor_id=get_principal().subject, status="archived")
    _job_store.add_event(session_id, get_principal().subject, "draft_started", {})
    return response


@app.post("/api/jobs/{session_id}/archive")
def api_archive_job(session_id: str):
    record = _job_for_principal(session_id)
    if record.status in {"running", "queued", "waiting_feedback", "waiting_safety"}:
        raise HTTPException(status_code=409, detail="运行中的任务不能归档")
    updated = _job_store.update(
        session_id, actor_id=get_principal().subject, status="archived", pending_kind=""
    )
    return {"job": updated.public_dict() if updated else None}


@app.post("/api/jobs/archive-batch")
def api_archive_jobs_batch(req: JobBatchArchiveReq):
    principal = get_principal()
    archived, skipped = [], []
    active_states = {"running", "queued", "waiting_feedback", "waiting_safety", "cancel_requested"}
    for session_id in dict.fromkeys(req.session_ids):
        record = _job_store.get(session_id)
        if not record or record.tenant_id != principal.tenant_id:
            skipped.append({"session_id": session_id, "reason": "不存在"})
            continue
        if record.owner_id != principal.subject and not principal.can("admin"):
            skipped.append({"session_id": session_id, "reason": "无权限"})
            continue
        if record.status in active_states:
            skipped.append({"session_id": session_id, "reason": "任务仍在等待或运行"})
            continue
        _job_store.update(
            session_id, actor_id=principal.subject, status="archived", pending_kind=""
        )
        archived.append(session_id)
    return {"archived": archived, "skipped": skipped}


# ============================ Jobs / health / metrics ========================


@app.get("/api/jobs")
def api_jobs(
    limit: int = 100, offset: int = 0, status: str = "", q: str = "",
    asset_id: str = "",
):
    _job_store.expire_stale()
    principal = get_principal()
    owner = None if principal.can("expert") else principal.subject
    jobs = _job_store.list_jobs(principal.tenant_id, owner_id=owner, limit=5000)
    if status:
        requested = {item.strip() for item in status.split(",") if item.strip()}
        jobs = [job for job in jobs if job.status in requested]
    if q.strip():
        needle = q.strip().lower()
        jobs = [
            job for job in jobs
            if needle in job.fault_input.lower() or needle in job.session_id.lower()
            or needle in str((job.request_payload or {}).get("alarm_code", "")).lower()
        ]
    if asset_id:
        jobs = [
            job for job in jobs
            if str((job.request_payload or {}).get("asset_id") or "") == asset_id
        ]
    total = len(jobs)
    offset = max(0, offset)
    limit = max(1, min(limit, 200))
    assets = {
        item["id"]: item for item in _business_store.list_assets(
            principal.tenant_id, include_inactive=True
        )
    }
    result = []
    for job in jobs[offset:offset + limit]:
        item = job.public_dict()
        asset = assets.get(item.get("asset_id"), {})
        item["asset_name"] = asset.get("name", "")
        item["asset_code"] = asset.get("asset_code", "")
        result.append(item)
    return {"jobs": result, "total": total, "offset": offset, "limit": limit}


@app.get("/api/jobs/{session_id}")
def api_job_detail(session_id: str):
    record = _job_store.get(session_id)
    if not record:
        raise HTTPException(status_code=404, detail="任务不存在")
    principal = get_principal()
    if record.tenant_id != principal.tenant_id:
        raise HTTPException(status_code=404, detail="任务不存在")
    if record.owner_id != principal.subject and not principal.can("expert"):
        raise HTTPException(status_code=403, detail="无权访问该任务")
    pending = {}
    if record.status in {"waiting_feedback", "waiting_safety"}:
        try:
            snapshot = _agent_app.get_state(
                {"configurable": {"thread_id": record.thread_id}}
            )
            values = snapshot.values or {}
            interrupt_value = _interrupt_value(snapshot)
            if record.status == "waiting_safety" and isinstance(interrupt_value, dict):
                pending = interrupt_value
            else:
                pending = {
                    "kind": "expert_question",
                    "question": str(interrupt_value or "请提供专家反馈"),
                    "question_index": values.get("current_question_idx", 0),
                    "total": len(values.get("audit_questions", [])),
                    "diagram": values.get("mermaid_diagram", ""),
                }
        except Exception as exc:
            pending = {"kind": record.pending_kind, "error": str(exc)}
    return {
        "job": record.public_dict(),
        "asset": _business_store.get_asset(
            record.tenant_id, str((record.request_payload or {}).get("asset_id") or "")
        ) if (record.request_payload or {}).get("asset_id") else None,
        "events": _job_store.events(session_id),
        "procedures": _business_store.list_procedures(record.tenant_id, session_id),
        "case": _business_store.get_case_by_session(record.tenant_id, session_id),
        "pending": pending,
    }


@app.post("/api/jobs/{session_id}/cancel")
def api_cancel_job(session_id: str):
    record = _job_store.get(session_id)
    if not record:
        raise HTTPException(status_code=404, detail="任务不存在")
    principal = get_principal()
    if record.tenant_id != principal.tenant_id:
        raise HTTPException(status_code=404, detail="任务不存在")
    if record.owner_id != principal.subject and not principal.can("admin"):
        raise HTTPException(status_code=403, detail="无权取消该任务")
    if record.status in {"completed", "pending_approval", "archived", "failed", "cancelled", "denied", "expired"}:
        raise HTTPException(status_code=409, detail=f"任务已处于终态: {record.status}")
    with _sessions_lock:
        sess = _sessions.get(session_id)
        if sess:
            sess["cancelled"] = True
    new_status = "cancel_requested" if record.status == "running" else "cancelled"
    updated = _job_store.update(
        session_id, actor_id=principal.subject, status=new_status, pending_kind=""
    )
    return {"job": updated.public_dict() if updated else None}


@app.post("/api/jobs/{session_id}/retry")
async def api_retry_job(session_id: str):
    record = _job_store.get(session_id)
    if not record:
        raise HTTPException(status_code=404, detail="任务不存在")
    principal = get_principal()
    if record.tenant_id != principal.tenant_id:
        raise HTTPException(status_code=404, detail="任务不存在")
    if record.owner_id != principal.subject and not principal.can("admin"):
        raise HTTPException(status_code=403, detail="无权重试该任务")
    if record.status not in {"failed", "cancelled", "denied", "expired", "interrupted"}:
        raise HTTPException(status_code=409, detail=f"当前状态不可重试: {record.status}")
    payload = dict(record.request_payload or {})
    payload["fault_input"] = record.fault_input
    request = StartReq.model_validate(payload)
    _job_store.add_event(session_id, principal.subject, "retry_created", {})
    return await api_start(request)


@app.get("/health/live")
def health_live():
    return {"ok": True, "service": "langgraph-agent"}


@app.get("/health/ready")
def health_ready():
    components = {
        "job_store": _job_store.health(),
        "business_store": _business_store.health(),
        "managed_knowledge_vector": _managed_vectors.health(),
        "checkpoint": checkpointer_status(_checkpointer),
        "mermaid_validator": validator_health(),
        "authentication": {"ok": _auth.configured, "enabled": _auth.enabled},
    }
    ready = all(item.get("ok", False) for item in components.values())
    payload = {"ok": ready, "components": components}
    return JSONResponse(payload, status_code=200 if ready else 503)


@app.get("/metrics")
def api_metrics():
    metrics.set("active_diagnoses", _diagnosis_slots.active)
    return PlainTextResponse(metrics.prometheus(), media_type="text/plain; version=0.0.4")


# ============================ History & files ===============================


def _allowed_output_files() -> set[str] | None:
    principal = get_principal()
    if not _auth.enabled:
        return None
    allowed: set[str] = set()
    owner = None if principal.can("expert") else principal.subject
    for job in _job_store.list_jobs(principal.tenant_id, owner_id=owner, limit=500):
        for value in (job.result_paths or {}).values():
            if value:
                allowed.add(os.path.basename(str(value)))
    return allowed


def _assert_output_access(filename: str) -> None:
    allowed = _allowed_output_files()
    if allowed is not None and filename not in allowed:
        raise HTTPException(status_code=404, detail="文件不存在")

@app.get("/api/history")
def api_history():
    allowed_files = _allowed_output_files()
    items = []
    # memory json files
    mem = []
    if os.path.isdir(MEMORY_DIR):
        for fname in sorted(os.listdir(MEMORY_DIR), reverse=True):
            if not fname.endswith(".json"):
                continue
            if allowed_files is not None and fname not in allowed_files:
                continue
            fpath = os.path.join(MEMORY_DIR, fname)
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                mem.append({
                    "id": entry.get("id", fname),
                    "file": fname,
                    "timestamp": entry.get("timestamp", ""),
                    "fault": entry.get("fault_description", ""),
                    "audit_passed": entry.get("audit_passed"),
                    "revision_count": entry.get("revision_count", 0),
                    "diagram_preview": (entry.get("diagram_summary") or "")[:160],
                })
            except Exception:
                continue
    # diagram files (.mmd) with mtime
    diags = []
    if os.path.isdir(DIAGRAM_DIR):
        for fname in sorted(os.listdir(DIAGRAM_DIR), reverse=True):
            if not fname.endswith(".mmd"):
                continue
            if allowed_files is not None and fname not in allowed_files:
                continue
            fpath = os.path.join(DIAGRAM_DIR, fname)
            try:
                mtime = datetime.datetime.fromtimestamp(os.path.getmtime(fpath))
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                diags.append({
                    "file": fname,
                    "mtime": mtime.strftime("%Y-%m-%d %H:%M:%S"),
                    "chars": len(content),
                    "preview": content[:200],
                })
            except Exception:
                continue
    return JSONResponse({"memory": mem, "diagrams": diags})


@app.get("/api/memory/{name:path}")
def api_memory(name: str):
    safe = os.path.basename(name)
    _assert_output_access(safe)
    fpath = os.path.join(MEMORY_DIR, safe)
    if not os.path.isfile(fpath) or not safe.endswith(".json"):
        raise HTTPException(status_code=404, detail="璁板繂涓嶅瓨鍦?")
    with open(fpath, "r", encoding="utf-8") as f:
        return JSONResponse({"file": safe, "content": json.load(f)})


@app.get("/api/diagram/{name:path}")
def api_diagram(name: str):
    safe = os.path.basename(name)
    _assert_output_access(safe)
    fpath = os.path.join(DIAGRAM_DIR, safe)
    if not os.path.isfile(fpath) or not safe.endswith(".mmd"):
        raise HTTPException(status_code=404, detail="流程图不存在")
    with open(fpath, "r", encoding="utf-8") as f:
        return JSONResponse({"file": safe, "content": f.read()})


@app.get("/api/report/{name:path}")
def api_report(name: str):
    safe = os.path.basename(name)
    _assert_output_access(safe)
    fpath = os.path.join(DIAGRAM_DIR, safe)
    if not os.path.isfile(fpath) or not safe.endswith(".md"):
        raise HTTPException(status_code=404, detail="报告不存在")
    with open(fpath, "r", encoding="utf-8") as f:
        return JSONResponse({"file": safe, "content": f.read()})


# ============================ Static & root =================================


@app.get("/static/index.html", include_in_schema=False)
def legacy_frontend_redirect():
    """Retired prototype frontend; keep bookmarks alive on the current app."""
    return RedirectResponse(url="/", status_code=301)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
if MERMAID_DIST_DIR.is_dir():
    app.mount(
        "/vendor/mermaid",
        StaticFiles(directory=str(MERMAID_DIST_DIR)),
        name="vendor-mermaid",
    )
if OML2D_DIST_DIR.is_dir():
    app.mount(
        "/vendor/oml2d",
        StaticFiles(directory=str(OML2D_DIST_DIR)),
        name="vendor-oml2d",
    )
if LIVE2D_SHIZUKU_DIR.is_dir():
    app.mount(
        "/vendor/live2d-models/shizuku",
        StaticFiles(directory=str(LIVE2D_SHIZUKU_DIR)),
        name="vendor-live2d-shizuku",
    )


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "app.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server:app", host="0.0.0.0", port=8000, reload=False)

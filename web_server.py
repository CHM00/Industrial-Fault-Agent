import os
import json
import re
import uuid
import threading
import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from pydantic import BaseModel

from langgraph.types import Command

import fault_agent
from mcp_tools import (
    save_diagram_and_report,
    save_diagnosis_memory,
    search_diagnosis_memory,
    DIAGRAM_DIR,
    MEMORY_DIR,
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="工业故障诊断 Agent", version="1.0")

# ---- build the LangGraph app once; all sessions share one MemorySaver,
#      differentiated by thread_id -------------------------------------------
_agent_app = fault_agent.build_app()

# session registry: session_id -> {config, lock, first, pending_feedback}
_sessions: dict = {}
_sessions_lock = threading.Lock()

# node name -> human-readable label (Chinese, terminal-style)
NODE_LABELS = {
    "dispatch": "[dispatch] 初始化诊断会话",
    "retrieve_internal": "[retrieve] 检索内部 SOP 知识库",
    "search_agent": "[search] 决定外部搜索关键词",
    "tools_node": "[tools] 执行 Tavily 互联网搜索",
    "filter": "[fuse] 内外知识分层融合",
    "generate": "[generate] 生成 Mermaid 故障排查流程图",
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


def _get_state(config):
    return _agent_app.get_state(config)


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


def _initial_state(fault_input: str, auto_mode: bool) -> dict:
    return {
        "messages": [],
        "fault_input": fault_input,
        "internal_knowledge": "",
        "external_knowledge": "",
        "filtered_context": "",
        "mermaid_diagram": "",
        "audit_result": "",
        "audit_questions": [],
        "has_gaps": False,
        "revision_count": 0,
        "auto_mode": auto_mode,
        "current_question_idx": 0,
        "expert_feedbacks": [],
        "thinking_framework": "",
    }


def _langfuse_handler():
    try:
        return fault_agent.get_langfuse_handler()
    except Exception:
        return None


def _new_session(auto_mode: bool) -> dict:
    sid = uuid.uuid4().hex[:12]
    thread_id = f"web_{sid}"
    config = {"configurable": {"thread_id": thread_id}}
    handler = _langfuse_handler()
    if handler:
        config["callbacks"] = [handler]
    sess = {
        "config": config,
        "lock": threading.Lock(),
        "first": True,
        "pending_feedback": None,
        "auto_mode": auto_mode,
        "fault_input": "",
    }
    with _sessions_lock:
        _sessions[sid] = sess
    return sid, sess


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


def _summarize_done(values: dict, paths: dict) -> dict:
    return {
        "mermaid_diagram": values.get("mermaid_diagram", ""),
        "audit_result": values.get("audit_result", ""),
        "audit_questions": values.get("audit_questions", []),
        "has_gaps": values.get("has_gaps", False),
        "revision_count": values.get("revision_count", 0),
        "internal_knowledge": values.get("internal_knowledge", ""),
        "external_knowledge": values.get("external_knowledge", ""),
        "filtered_context": values.get("filtered_context", ""),
        "paths": paths,
    }


# ============================ API models ====================================

class StartReq(BaseModel):
    fault_input: str
    auto_mode: bool = False


class ResumeReq(BaseModel):
    session_id: str
    feedback: str


# ============================ SSE: start ====================================

@app.post("/api/start")
def api_start(req: StartReq):
    fault = (req.fault_input or "").strip()
    if not fault:
        raise HTTPException(status_code=400, detail="fault_input 不能为空")
    sid, sess = _new_session(req.auto_mode)
    sess["fault_input"] = fault
    config = sess["config"]

    def gen():
        with sess["lock"]:
            try:
                payload = _initial_state(fault, req.auto_mode)
                while True:
                    # run one graph turn (until pause or finish)
                    for event in _agent_app.stream(payload, config, stream_mode="updates"):
                        for node, update in event.items():
                            if node == "__interrupt__":
                                continue
                            if not isinstance(update, dict):
                                continue
                            label = NODE_LABELS.get(node, f"[{node}]")
                            msg = _extract_message(update)
                            if not msg:
                                msg = label
                            yield _sse("progress", {"node": node, "label": label, "message": msg})

                    snap = _get_state(config)
                    values = snap.values or {}

                    if snap.next:  # paused at an interrupt (expert question)
                        question = _interrupt_question(snap) or "请提供专家反馈"
                        idx = values.get("current_question_idx", 0)
                        total = len(values.get("audit_questions", []))
                        qpayload = {
                            "session_id": sid,
                            "question": question,
                            "question_index": idx,
                            "total": total,
                            "diagram": values.get("mermaid_diagram", ""),
                            "audit_questions": values.get("audit_questions", []),
                        }
                        if req.auto_mode:
                            auto_fb = f"针对「{question}」，请自动补充相关的判断步骤和异常路径。"
                            yield _sse("progress", {
                                "node": "auto",
                                "label": "[auto] 自动模式反馈",
                                "message": auto_fb,
                            })
                            payload = Command(resume=auto_fb)
                            continue
                        else:
                            yield _sse("question", qpayload)
                            return
                    else:
                        # finished
                        paths = _save_outputs(fault, values)
                        yield _sse("done", {
                            "session_id": sid,
                            "fault_input": fault,
                            **_summarize_done(values, paths),
                        })
                        return
            except Exception as e:
                yield _sse("error", {"message": f"诊断失败: {e}"})
                return

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# ============================ SSE: resume ===================================

@app.post("/api/resume")
def api_resume(req: ResumeReq):
    with _sessions_lock:
        sess = _sessions.get(req.session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="session 不存在或已过期，请重新开始诊断")
    config = sess["config"]
    fault = sess["fault_input"]
    feedback = (req.feedback or "").strip() or f"跳过该问题（用户未提供反馈）"

    def gen():
        with sess["lock"]:
            try:
                payload = Command(resume=feedback)
                while True:
                    for event in _agent_app.stream(payload, config, stream_mode="updates"):
                        for node, update in event.items():
                            if node == "__interrupt__":
                                continue
                            if not isinstance(update, dict):
                                continue
                            label = NODE_LABELS.get(node, f"[{node}]")
                            msg = _extract_message(update)
                            if not msg:
                                msg = label
                            yield _sse("progress", {"node": node, "label": label, "message": msg})

                    snap = _get_state(config)
                    values = snap.values or {}

                    if snap.next:
                        question = _interrupt_question(snap) or "请提供专家反馈"
                        idx = values.get("current_question_idx", 0)
                        total = len(values.get("audit_questions", []))
                        yield _sse("question", {
                            "session_id": req.session_id,
                            "question": question,
                            "question_index": idx,
                            "total": total,
                            "diagram": values.get("mermaid_diagram", ""),
                            "audit_questions": values.get("audit_questions", []),
                        })
                        return
                    else:
                        paths = _save_outputs(fault, values)
                        yield _sse("done", {
                            "session_id": req.session_id,
                            "fault_input": fault,
                            **_summarize_done(values, paths),
                        })
                        # session complete -> clean up
                        with _sessions_lock:
                            _sessions.pop(req.session_id, None)
                        return
            except Exception as e:
                yield _sse("error", {"message": f"恢复诊断失败: {e}"})
                return

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


# ============================ History & files ===============================

@app.get("/api/history")
def api_history():
    items = []
    # memory json files
    mem = []
    if os.path.isdir(MEMORY_DIR):
        for fname in sorted(os.listdir(MEMORY_DIR), reverse=True):
            if not fname.endswith(".json"):
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
    fpath = os.path.join(MEMORY_DIR, safe)
    if not os.path.isfile(fpath) or not safe.endswith(".json"):
        raise HTTPException(status_code=404, detail="璁板繂涓嶅瓨鍦?")
    with open(fpath, "r", encoding="utf-8") as f:
        return JSONResponse({"file": safe, "content": json.load(f)})


@app.get("/api/diagram/{name:path}")
def api_diagram(name: str):
    safe = os.path.basename(name)
    fpath = os.path.join(DIAGRAM_DIR, safe)
    if not os.path.isfile(fpath) or not safe.endswith(".mmd"):
        raise HTTPException(status_code=404, detail="流程图不存在")
    with open(fpath, "r", encoding="utf-8") as f:
        return JSONResponse({"file": safe, "content": f.read()})


@app.get("/api/report/{name:path}")
def api_report(name: str):
    safe = os.path.basename(name)
    fpath = os.path.join(DIAGRAM_DIR, safe)
    if not os.path.isfile(fpath) or not safe.endswith(".md"):
        raise HTTPException(status_code=404, detail="报告不存在")
    with open(fpath, "r", encoding="utf-8") as f:
        return JSONResponse({"file": safe, "content": f.read()})


# ============================ Static & root =================================

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "app.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_server:app", host="0.0.0.0", port=8000, reload=False)

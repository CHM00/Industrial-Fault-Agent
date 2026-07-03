import os
import json
import re
import datetime
from pathlib import Path

OUTPUT_DIR = os.path.abspath("output")
MEMORY_DIR = os.path.join(OUTPUT_DIR, "memory")
DIAGRAM_DIR = os.path.join(OUTPUT_DIR, "diagrams")


def _ensure_dirs():
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(DIAGRAM_DIR, exist_ok=True)


def sequential_think(problem: str, max_steps: int = 6) -> str:
    """结构化推理工具：逐步分析问题，返回推理过程文本。

    Semantic equivalent of the Sequential Thinking MCP server.
    Forces the LLM to break down analysis into explicit steps,
    making the audit reasoning traceable.

    Args:
        problem: The problem or question to analyze step by step.
        max_steps: Maximum number of reasoning steps.

    Returns:
        A numbered reasoning chain string.
    """
    steps = [
        "1. 检查流程图是否有入口节点（故障报告作为起点）",
        "2. 检查每个判断节点是否有'是'和'否'两个分支",
        "3. 检查异常路径是否完整（每个判断的'否'分支是否指向处理步骤）",
        "4. 检查流程图是否有终止节点（故障排除或升级处理）",
        "5. 检查是否有遗漏的关键诊断步骤（如安全检查、参数测量等）",
        "6. 检查流程图是否符合工业现场实际操作顺序",
    ]
    steps = steps[:max_steps]
    reasoning = "逐步推理框架：\n"
    for s in steps:
        reasoning += f"  {s}\n"
    reasoning += f"\n请严格按照以上{len(steps)}个维度逐一审查流程图。\n"
    return reasoning


def save_diagram_and_report(fault_input: str, state: dict) -> tuple:
    """Filesystem MCP equivalent: save Mermaid diagram and diagnostic report to files.

    Args:
        fault_input: Fault description string.
        state: AgentState dict containing all diagnostic results.

    Returns:
        Tuple of (diagram_path, report_path)
    """
    _ensure_dirs()
    safe_name = re.sub(r'[^\w]', '_', fault_input)[:50]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = f"{safe_name}_{timestamp}"

    diagram_path = os.path.join(DIAGRAM_DIR, f"{filename_prefix}_diagram.mmd")
    with open(diagram_path, "w", encoding="utf-8") as f:
        f.write(state.get("mermaid_diagram", ""))

    report_path = os.path.join(DIAGRAM_DIR, f"{filename_prefix}_report.md")
    fault = state.get("fault_input", "")
    internal = state.get("internal_knowledge", "")
    external = state.get("external_knowledge", "")
    filtered = state.get("filtered_context", "")
    audit_result = state.get("audit_result", "")
    has_gaps = state.get("has_gaps", False)
    revision_count = state.get("revision_count", 0)
    questions = state.get("audit_questions", [])
    feedbacks = state.get("expert_feedbacks", [])

    report = f"# 故障诊断报告：{fault}\n\n"
    report += f"- **诊断时间**: {timestamp}\n"
    report += f"- **审计通过**: {'是' if not has_gaps else '否（经修订）'}\n"
    report += f"- **修订次数**: {revision_count}\n\n"
    report += f"## 一、故障描述\n\n{fault}\n\n"
    report += f"## 二、内部SOP知识\n\n{internal[:1000] if internal else '无'}\n\n"
    report += f"## 三、外部搜索信息\n\n{external[:1000] if external else '无'}\n\n"
    report += f"## 四、知识融合上下文\n\n{filtered[:2000] if filtered else '无'}\n\n"
    report += f"## 五、审计结果\n\n{audit_result}\n\n"
    if questions:
        report += f"## 六、审计问题与专家反馈\n\n"
        for i, (q, fb) in enumerate(zip(questions, feedbacks)):
            report += f"### 问题 {i+1}\n- **问题**: {q}\n- **反馈**: {fb}\n\n"
    report += f"## 七、故障排查流程图\n\n```mermaid\n{state.get('mermaid_diagram', '')}\n```\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[Filesystem] 诊断报告已保存到: {report_path}")
    print(f"[Filesystem] 流程图已保存到: {diagram_path}")
    return diagram_path, report_path


def save_diagnosis_memory(fault_input: str, state: dict) -> str:
    """Memory MCP equivalent: save diagnosis result to local JSON knowledge base.

    Stores structured diagnosis history so that future diagnoses can
    reference past cases for similar faults.

    Args:
        fault_input: Fault description string.
        state: AgentState dict containing all diagnostic results.

    Returns:
        Path to the saved memory file.
    """
    _ensure_dirs()
    safe_name = re.sub(r'[^\w]', '_', fault_input)[:50]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    memory_entry = {
        "id": f"{safe_name}_{timestamp}",
        "timestamp": timestamp,
        "fault_description": state.get("fault_input", ""),
        "internal_knowledge_summary": state.get("internal_knowledge", "")[:500],
        "external_knowledge_summary": state.get("external_knowledge", "")[:500] if state.get("external_knowledge") else "",
        "filtered_context_summary": state.get("filtered_context", "")[:500],
        "audit_passed": not state.get("has_gaps", True),
        "revision_count": state.get("revision_count", 0),
        "diagram_summary": state.get("mermaid_diagram", "")[:300],
    }

    memory_path = os.path.join(MEMORY_DIR, f"{safe_name}_{timestamp}.json")
    with open(memory_path, "w", encoding="utf-8") as f:
        json.dump(memory_entry, f, ensure_ascii=False, indent=2)

    print(f"[Memory] 诊断记忆已保存到: {memory_path}")
    return memory_path


def search_diagnosis_memory(query: str, top_k: int = 3) -> list:
    """Memory MCP equivalent: search past diagnosis history by keyword.

    Searches through saved diagnosis JSON files for relevant past cases.

    Args:
        query: Search query string.
        top_k: Maximum number of results to return.

    Returns:
        List of matching diagnosis records.
    """
    _ensure_dirs()
    results = []
    if not os.path.exists(MEMORY_DIR):
        return results

    query_lower = query.lower()
    for fname in os.listdir(MEMORY_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(MEMORY_DIR, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                entry = json.load(f)
            score = 0
            for field in ["fault_description", "filtered_context_summary", "internal_knowledge_summary"]:
                if field in entry and entry[field]:
                    text = entry[field].lower()
                    for word in query_lower.split():
                        if word in text:
                            score += 1
            if score > 0:
                results.append((score, entry))
        except (json.JSONDecodeError, KeyError):
            continue

    results.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, entry in results[:top_k]]
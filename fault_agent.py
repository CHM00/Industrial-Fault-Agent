import os
import json
import re
import time
import requests
import warnings
from typing import Annotated, TypedDict, List
from dotenv import load_dotenv
from pymilvus import Collection, connections, utility
from mcp_tools import sequential_think, save_diagram_and_report, save_diagnosis_memory

warnings.filterwarnings("ignore", category=DeprecationWarning, module="pymilvus")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain_openai")

MAX_INTERNAL_CHARS = 6000
MAX_EXTERNAL_CHARS = 6000
LLM_MAX_RETRIES = 5
LLM_RETRY_DELAY = 3

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langfuse.langchain import CallbackHandler

load_dotenv()

ARK_API_KEY = os.environ.get("ARK_API_KEY", "")
ARK_BASE_URL = os.environ.get("ARK_BASE_URL", "https://api.siliconflow.cn/v1")
LLM_MODEL = "deepseek-ai/DeepSeek-V4-Flash"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-4B"
EMBEDDING_DIM = 2560
TAVILY_API_KEY = os.environ.get("trivily_key", os.environ.get("TAVILY_API_KEY", ""))
MILVUS_HOST = os.environ.get("MILVUS_HOST", "")
MILVUS_PORT = os.environ.get("MILVUS_PORT", "19530")
MILVUS_USER = os.environ.get("MILVUS_USER", "root")
MILVUS_PASSWORD = os.environ.get("MILVUS_PASSWORD", "")
MILVUS_URI = os.environ.get("URL", "")
MILVUS_TOKEN = os.environ.get("Token", "")

COLLECTION_NAME = "industrial_fault_knowledge"
FAULT_CONN_ALIAS = "fault_agent_conn"


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    fault_input: str
    internal_knowledge: str
    external_knowledge: str
    filtered_context: str
    mermaid_diagram: str
    audit_result: str
    audit_questions: List[str]
    has_gaps: bool
    revision_count: int
    auto_mode: bool
    current_question_idx: int
    expert_feedbacks: List[str]
    thinking_framework: str


# ============== Embedding & Milvus Search ==============

def get_embedding(text: str) -> list:
    url = f"{ARK_BASE_URL}/embeddings"
    payload = {"model": EMBEDDING_MODEL, "input": text}
    headers = {"Authorization": f"Bearer {ARK_API_KEY}", "Content-Type": "application/json"}
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


_milvus_connected = False


def _ensure_milvus():
    global _milvus_connected
    if _milvus_connected:
        return
    if MILVUS_HOST:
        connect_kwargs = {"alias": FAULT_CONN_ALIAS, "host": MILVUS_HOST, "port": MILVUS_PORT}
        if MILVUS_USER:
            connect_kwargs["user"] = MILVUS_USER
        if MILVUS_PASSWORD:
            connect_kwargs["password"] = MILVUS_PASSWORD
        connections.connect(**connect_kwargs)
    elif MILVUS_URI:
        connect_kwargs = {"alias": FAULT_CONN_ALIAS, "uri": MILVUS_URI}
        if MILVUS_TOKEN:
            connect_kwargs["token"] = MILVUS_TOKEN
        connections.connect(**connect_kwargs)
    else:
        from milvus_lite import MilvusLite
        local_path = os.path.abspath("output/milvus_local.db")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        connections.connect(alias=FAULT_CONN_ALIAS, uri=local_path)
    _milvus_connected = True


def search_milvus(query: str, top_k: int = 3) -> str:
    _ensure_milvus()
    if not utility.has_collection(COLLECTION_NAME, using=FAULT_CONN_ALIAS):
        return ""
    collection = Collection(name=COLLECTION_NAME, using=FAULT_CONN_ALIAS)
    collection.load()
    query_vec = get_embedding(query)
    results = collection.search(
        data=[query_vec],
        anns_field="vector",
        param={"metric_type": "IP", "params": {"nlist": 128}},
        limit=top_k,
        output_fields=["title", "content", "category"],
    )
    if not results or not results[0]:
        return ""
    chunks = []
    for hit in results[0]:
        title = hit.entity.get("title", "")
        category = hit.entity.get("category", "")
        content = hit.entity.get("content", "")
        score = f"{hit.distance:.4f}"
        chunks.append(f"[{category}] {title} (相关度:{score})\n{content}")
    return "\n\n---\n\n".join(chunks)


# ============== @tool: Tavily Search ==============

@tool
def tavily_search(query: str) -> str:
    """搜索互联网获取工业故障维修的最新资讯和解决方案。当内部知识库信息不足时使用此工具补充外部信息。"""
    from tavily import TavilyClient
    client = TavilyClient(api_key=TAVILY_API_KEY)
    results = client.search(query, max_results=3)
    snippets = []
    for r in results.get("results", []):
        title = r.get("title", "")
        content = r.get("content", "")
        url = r.get("url", "")
        snippets.append(f"标题: {title}\n来源: {url}\n内容: {content}")
    return "\n\n---\n\n".join(snippets) if snippets else "未找到相关外部资讯。"


tools_list = [tavily_search]

# ============== LLM ==============

llm = ChatOpenAI(
    model=LLM_MODEL,
    api_key=ARK_API_KEY,
    base_url=ARK_BASE_URL,
    temperature=0.3,
)
llm_with_tools = llm.bind_tools(tools_list)


def invoke_llm_with_retry(llm_instance, messages, max_retries=LLM_MAX_RETRIES, delay=LLM_RETRY_DELAY):
    for attempt in range(1, max_retries + 1):
        try:
            return llm_instance.invoke(messages)
        except Exception as e:
            if attempt == max_retries:
                raise
            wait = delay * (2 ** (attempt - 1))
            print(f"[重试] LLM调用失败({attempt}/{max_retries}): {e}, {wait}秒后重试...")
            time.sleep(wait)


# ============== Node Functions ==============

def dispatch_node(state: AgentState) -> dict:
    system_msg = SystemMessage(content=(
        "你是一个专业的工业故障诊断AI助手。你的任务是：\n"
        "1. 通过内部知识库（Milvus）检索标准操作规程（SOP）\n"
        "2. 通过外部搜索（Tavily）获取最新故障解决方案\n"
        "3. 过滤融合内外知识，生成故障排查流程图\n"
        "4. 审计流程图的完整性和准确性\n"
        "5. 根据人类专家反馈修订流程图\n\n"
        "始终使用中文回答，输出的Mermaid流程图使用flowchart TD格式，节点文本使用中文。"
    ))
    return {
        "messages": [system_msg],
        "internal_knowledge": "",
        "external_knowledge": "",
        "filtered_context": "",
        "mermaid_diagram": "",
        "audit_result": "",
        "audit_questions": [],
        "has_gaps": False,
        "revision_count": 0,
        "current_question_idx": 0,
        "expert_feedbacks": [],
        "thinking_framework": "",
    }


def retrieve_internal_node(state: AgentState) -> dict:
    fault = state["fault_input"]
    results = search_milvus(fault, top_k=3)
    if not results:
        results = "内部知识库未检索到相关SOP信息。"
    ai_msg = AIMessage(content=f"[内部检索] 根据故障描述'{fault}'检索到以下SOP信息，相关度按降序排列。")
    return {
        "internal_knowledge": results,
        "messages": [ai_msg],
    }


def _extract_core_fault(fault: str) -> str:
    if "：" in fault or ":" in fault:
        core = re.split(r"[：:]", fault, 1)[0].strip()
    else:
        core = fault
    core = re.sub(r"[→\-，,。.、\s]+", " ", core).strip()
    return core if core else fault


def search_agent_node(state: AgentState) -> dict:
    fault = state["fault_input"]
    core_fault = _extract_core_fault(fault)
    search_query = f"{core_fault} 故障诊断 维修方案"

    tool_call_id = f"call_tavily_{abs(hash(search_query)) % 1000000:06d}"
    ai_msg = AIMessage(
        content=f"[搜索决策] 需要外部信息补充，搜索关键词：{search_query}",
        tool_calls=[{
            "name": "tavily_search",
            "args": {"query": search_query},
            "id": tool_call_id,
            "type": "tool_call",
        }],
    )
    return {"messages": [ai_msg]}


def extract_external_from_tool_messages(state: AgentState) -> str:
    messages = state.get("messages", [])
    seen = set()
    tool_contents = []
    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.name == "tavily_search":
            content_hash = hash(msg.content[:500])
            if content_hash not in seen:
                seen.add(content_hash)
                tool_contents.append(msg.content)
    if tool_contents:
        return "\n\n---\n\n".join(tool_contents)
    return ""

FILTER_PROMPT = (
    "你是一位资深工业故障诊断工程师。以下是关于故障「{fault}」的内外部知识：\n\n"
    "【内部SOP知识】（权威来源：经过验证的标准操作规程，优先级最高）\n"
    "{internal}\n\n"
    "【外部搜索信息】（参考来源：互联网最新资讯，需甄别准确性，优先级低于内部SOP）\n"
    "{external}\n\n"
    "请执行以下分层融合任务：\n"
    "1. 以内部SOP知识为主要框架，优先采用内部标准操作规程中的诊断步骤和处理方案\n"
    "2. 如果内部SOP覆盖不全，用外部搜索信息补充缺失部分，但需标注为「参考信息」\n"
    "3. 如果外部信息与内部SOP冲突，以内部SOP为准，并在末尾注明存在冲突\n"
    "4. 剔除外部信息中的广告、重复和无关内容\n"
    "5. 如果外部信息为空，仅基于内部SOP输出\n"
    "6. 如果内部SOP为空，基于外部信息输出，但在开头添加提示：「以下信息来源于互联网搜索，未经现场验证」\n"
    "7. 输出一份精炼的故障诊断参考上下文，包含：故障现象、可能原因、诊断步骤、处理方案\n"
    "8. 诊断步骤和处理方案中，来自内部SOP的内容不加标注，来自外部搜索的内容标注「(参考)」"
)


def filter_knowledge_node(state: AgentState) -> dict:
    internal = state.get("internal_knowledge", "")
    external = extract_external_from_tool_messages(state)
    fault = state["fault_input"]

    print(f"[知识融合] 内部知识: {len(internal)}字符, 外部知识: {len(external)}字符")

    prompt = FILTER_PROMPT.format(fault=fault, internal=internal[:MAX_INTERNAL_CHARS], external=external[:MAX_EXTERNAL_CHARS])
    response = invoke_llm_with_retry(llm, [HumanMessage(content=prompt)])
    return {
        "external_knowledge": external,
        "filtered_context": response.content,
        "messages": [AIMessage(content="[知识融合] 已完成内外知识分层融合，生成诊断上下文。")],
    }


def generate_diagram_node(state: AgentState) -> dict:
    context = state.get("filtered_context", "")
    fault = state["fault_input"]

    prompt = (
        f"请根据以下故障诊断上下文，为故障「{fault}」生成一份工业故障排查 Mermaid 流程图。\n\n"
        f"故障诊断上下文：\n{context[:8000]}\n\n"
        f"要求：\n"
        f"1. 使用 flowchart TD 格式\n"
        f"2. 包含判断节点（菱形）和操作步骤（矩形）\n"
        f"3. 包含正常路径和异常/故障路径\n"
        f"4. 节点文本使用中文\n"
        f"5. 必须以 ```mermaid``` 代码块格式输出\n"
        f"6. 流程图应完整且可执行，从故障报告开始，以故障排除或升级处理结束\n"
        f"7. 每个判断节点必须有是/否两个分支"
    )
    response = invoke_llm_with_retry(llm, [HumanMessage(content=prompt)])

    content = response.content
    mermaid_match = re.search(r"```mermaid\s*\n(.*?)```", content, re.DOTALL)
    if mermaid_match:
        diagram = mermaid_match.group(1).strip()
    else:
        diagram = content.strip()

    return {
        "mermaid_diagram": diagram,
        "messages": [AIMessage(content=f"[流程图生成] 已生成故障排查流程图（{len(diagram)}字符）。")],
    }


AUDIT_PROMPT = (
    "你是一位工业故障诊断流程图审计专家。请审计以下故障排查流程图。\n\n"
    "请按照以下推理框架逐步分析：\n"
    "{thinking_framework}\n\n"
    "检查要点：\n"
    "1. 是否有逻辑断裂或遗漏步骤\n"
    "2. 异常路径是否完整（每个判断节点是否有 是/否 两个分支）\n"
    "3. 是否需要补充特定设备的操作细节\n"
    "4. 流程图是否能指导现场工程师完成故障排查\n\n"
    "请严格按照以下JSON格式回复（不要输出其他内容）：\n"
    "```json\n"
    "{{\n"
    '  "has_gaps": true或false,\n'
    '  "audit_result": "审计结果描述",\n'
    '  "questions": ["问题1", "问题2", ...]\n'
    "}}\n"
    "```\n\n"
    "重要约束：questions列表最多只能包含3个问题，请只列出最关键的3个问题。\n"
    "如果流程图完整无误，has_gaps设为false，questions为空列表。"
)


def audit_diagram_node(state: AgentState) -> dict:
    diagram = state.get("mermaid_diagram", "")
    context = state.get("filtered_context", "")
    fault = state["fault_input"]

    thinking_framework = state.get("thinking_framework", "")
    if not thinking_framework:
        thinking_framework = sequential_think(fault)

    prompt = (
        f"故障描述：{fault}\n\n"
        f"故障诊断上下文：\n{context[:8000]}\n\n"
        f"待审计的Mermaid流程图：\n```mermaid\n{diagram}\n```\n\n"
        f"{AUDIT_PROMPT.format(thinking_framework=thinking_framework)}"
    )
    response = invoke_llm_with_retry(llm, [HumanMessage(content=prompt)])

    content = response.content
    json_match = re.search(r"```json\s*\n(.*?)```", content, re.DOTALL)
    if json_match:
        json_str = json_match.group(1).strip()
    else:
        json_str = content.strip()

    try:
        audit_data = json.loads(json_str)
        has_gaps = bool(audit_data.get("has_gaps", False))
        audit_result = audit_data.get("audit_result", "审计完成")
        questions = audit_data.get("questions", [])[:3]
    except (json.JSONDecodeError, KeyError):
        has_gaps = "PASS" not in content and "完整" not in content and "pass" not in content.lower()
        audit_result = content[:200] if len(content) > 200 else content
        questions = ["请审阅流程图是否完整"] if has_gaps else []

    return {
        "has_gaps": has_gaps,
        "audit_result": audit_result,
        "audit_questions": questions,
        "thinking_framework": thinking_framework,
        "messages": [AIMessage(content=f"[审计] has_gaps={has_gaps}, 发现{len(questions)}个问题")],
    }


def evaluate_questions_node(state: AgentState) -> dict:
    questions = state.get("audit_questions", [])
    num_q = len(questions)
    print(f"\n{'='*60}")
    print(f"[审计发现] 存在 {num_q} 个需要专家确认的问题：")
    for i, q in enumerate(questions):
        print(f"  {i+1}. {q}")
    print(f"{'='*60}")
    return {
        "current_question_idx": 0,
        "expert_feedbacks": [],
        "messages": [AIMessage(content=f"[问题迭代] 开始逐个收集专家反馈，共{num_q}个问题。")],
    }


def ask_expert_per_question_node(state: AgentState) -> dict:
    questions = state.get("audit_questions", [])
    idx = state.get("current_question_idx", 0)
    feedbacks = list(state.get("expert_feedbacks", []))
    diagram = state.get("mermaid_diagram", "")

    question = questions[idx] if idx < len(questions) else "请提供一般反馈"

    print(f"\n--- 审计问题 [{idx+1}/{len(questions)}] ---")
    print(f"问题: {question}")
    print(f"当前流程图:")
    print(f"```mermaid\n{diagram}\n```")

    from langgraph.types import interrupt
    feedback = interrupt(f"请针对审计问题「{question}」输入专家反馈")

    if not feedback or not str(feedback).strip():
        feedback = f"跳过该问题（用户未提供反馈）"

    feedbacks.append(f"Q{idx+1}: {question}\n专家反馈: {feedback}")

    return {
        "current_question_idx": idx + 1,
        "expert_feedbacks": feedbacks,
        "messages": [HumanMessage(content=f"针对问题「{question}」的反馈: {feedback}")],
    }


def more_questions_router(state: AgentState) -> str:
    idx = state.get("current_question_idx", 0)
    questions = state.get("audit_questions", [])
    if idx < len(questions):
        return "next_question"
    return "all_answered"


def refine_diagram_node(state: AgentState) -> dict:
    diagram = state.get("mermaid_diagram", "")
    context = state.get("filtered_context", "")
    fault = state["fault_input"]
    feedbacks = state.get("expert_feedbacks", [])

    feedback_text = "\n\n".join(feedbacks)

    prompt = (
        f"故障描述：{fault}\n\n"
        f"故障诊断上下文：\n{context[:8000]}\n\n"
        f"当前流程图：\n```mermaid\n{diagram}\n```\n\n"
        f"专家逐条反馈：\n{feedback_text}\n\n"
        f"请根据以上专家反馈修改流程图。你必须在当前流程图的基础上进行最小化修改，严格遵守以下要求：\n"
        f"1. 只修改与专家反馈直接相关的节点和连线，未涉及的节点和连线必须完全保持原样，不得重新组织、重命名、增删无关部分\n"
        f"2. 逐一解决每个专家反馈指出的问题\n"
        f"3. 仍然使用 flowchart TD 格式，节点文本使用中文\n"
        f"4. 必须以 ```mermaid``` 代码块格式输出完整流程图\n"
        f"5. 每个判断节点必须有是/否两个分支\n"
        f"6. 不要改变整体结构和风格，仅做局部修补"
    )
    response = invoke_llm_with_retry(llm, [HumanMessage(content=prompt)])

    content = response.content
    mermaid_match = re.search(r"```mermaid\s*\n(.*?)```", content, re.DOTALL)
    if mermaid_match:
        new_diagram = mermaid_match.group(1).strip()
    else:
        new_diagram = content.strip()

    revision_count = state.get("revision_count", 0) + 1
    return {
        "mermaid_diagram": new_diagram,
        "revision_count": revision_count,
        "messages": [AIMessage(content=f"[修订] 已根据{len(feedbacks)}条专家反馈修订流程图（第{revision_count}次修订）。")],
    }


# ============== Condition Routers ==============

def check_gaps(state: AgentState) -> str:
    revision_count = state.get("revision_count", 0)
    has_gaps = state.get("has_gaps", False)
    if revision_count >= 3:
        return "finished"
    if has_gaps:
        return "needs_feedback"
    return "finished"


def more_questions_router(state: AgentState) -> str:
    idx = state.get("current_question_idx", 0)
    questions = state.get("audit_questions", [])
    if idx < len(questions):
        return "next_question"
    return "all_answered"


# ============== Build Graph ==============

def build_app():
    workflow = StateGraph(AgentState)

    workflow.add_node("dispatch", dispatch_node)
    workflow.add_node("retrieve_internal", retrieve_internal_node)
    workflow.add_node("search_agent", search_agent_node)
    workflow.add_node("tools_node", ToolNode(tools_list))
    workflow.add_node("filter", filter_knowledge_node)
    workflow.add_node("generate", generate_diagram_node)
    workflow.add_node("audit", audit_diagram_node)
    workflow.add_node("evaluate_questions", evaluate_questions_node)
    workflow.add_node("ask_expert", ask_expert_per_question_node)
    workflow.add_node("refine", refine_diagram_node)

    workflow.set_entry_point("dispatch")

    workflow.add_edge("dispatch", "search_agent")
    workflow.add_edge("search_agent", "retrieve_internal")
    workflow.add_edge("search_agent", "tools_node")
    workflow.add_edge("retrieve_internal", "filter")
    workflow.add_edge("tools_node", "filter")
    workflow.add_edge("filter", "generate")
    workflow.add_edge("generate", "audit")

    workflow.add_conditional_edges("audit", check_gaps, {
        "needs_feedback": "evaluate_questions",
        "finished": END,
    })

    workflow.add_edge("evaluate_questions", "ask_expert")
    workflow.add_conditional_edges("ask_expert", more_questions_router, {
        "next_question": "ask_expert",
        "all_answered": "refine",
    })

    workflow.add_edge("refine", "audit")

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app


# ============== Run ==============

def get_langfuse_handler():
    langfuse_enabled = os.environ.get("LANGFUSE_ENABLED", "").lower() == "true"
    if langfuse_enabled:
        try:
            handler = CallbackHandler()
            print("[Langfuse] 可观测性已启用，trace 将上报至 Langfuse 平台")
            return handler
        except Exception as e:
            print(f"[Langfuse] 初始化失败，降级为无追踪模式: {e}")
            return None
    return None


def _save_and_remember(fault_input: str, state: dict):
    """保存诊断报告到文件（Filesystem MCP 等效）并存入记忆（Memory MCP 等效）。"""
    try:
        save_diagram_and_report(fault_input, state)
    except Exception as e:
        print(f"[Filesystem] 保存诊断报告失败: {e}")
    try:
        save_diagnosis_memory(fault_input, state)
    except Exception as e:
        print(f"[Memory] 存入诊断记忆失败: {e}")


def run_fault_agent(fault_input: str, auto_mode: bool = False):
    app = build_app()

    initial_state = {
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

    langfuse_handler = get_langfuse_handler()
    config = {
        "configurable": {"thread_id": "fault_agent_session"},
    }
    if langfuse_handler:
        config["callbacks"] = [langfuse_handler]

    print(f"\n{'='*60}")
    print(f"工业故障诊断 Agent 启动")
    print(f"故障描述: {fault_input}")
    print(f"模式: {'自动' if auto_mode else '人机协同（逐问中断）'}")
    print(f"{'='*60}\n")

    result = app.invoke(initial_state, config=config)

    while True:
        has_gaps = result.get("has_gaps", False)
        revision_count = result.get("revision_count", 0)
        audit_questions = result.get("audit_questions", [])

        if not has_gaps or not audit_questions:
            print(f"\n{'='*60}")
            if revision_count == 0:
                print("审计通过，无需修订！")
            else:
                print(f"审计通过！（经过 {revision_count} 次修订）")
            print(f"\n===== 最终故障排查流程图 =====")
            print(f"```mermaid\n{result.get('mermaid_diagram', 'N/A')}\n```")
            print(f"===== 诊断完成 =====")
            print(f"{'='*60}\n")
            _save_and_remember(fault_input, result)
            return result

        if revision_count >= 3:
            print(f"\n{'='*60}")
            print(f"已达最大修订次数({revision_count}次)，输出当前版本。")
            print(f"\n===== 最终故障排查流程图 =====")
            print(f"```mermaid\n{result.get('mermaid_diagram', 'N/A')}\n```")
            print(f"===== 诊断完成 =====")
            print(f"{'='*60}\n")
            _save_and_remember(fault_input, result)
            return result

        print(f"\n{'='*60}")
        print(f"[修订轮次 {revision_count + 1}] 审计发现 {len(audit_questions)} 个问题")
        print(f"审计结果: {result.get('audit_result', 'N/A')}")
        print(f"{'='*60}")

        for q_idx, question in enumerate(audit_questions):
            if auto_mode:
                feedback = f"针对「{question}」，请自动补充相关的判断步骤和异常路径。"
                print(f"\n[自动模式] 问题 [{q_idx+1}/{len(audit_questions)}]: {question}")
                print(f"[自动模式] 反馈: {feedback}")
            else:
                print(f"\n--- 审计问题 [{q_idx+1}/{len(audit_questions)}] ---")
                print(f"问题: {question}")
                print(f"\n当前流程图（最近版本）:")
                print(f"```mermaid\n{result.get('mermaid_diagram', '')}\n```")
                feedback = input(f"\n请针对该问题输入专家反馈（直接回车可跳过）:\n> ")
                if not feedback.strip():
                    feedback = f"跳过：{question}"

            result = app.invoke(
                Command(resume=feedback),
                config=config,
            )

        print(f"\n所有问题已回答完毕，流程图正在修订并重新审计...")
        print(f"{'='*60}")

    return result


if __name__ == "__main__":
    fault = input("请输入故障描述: ")
    auto = input("是否自动模式？(y/n): ").strip().lower() == "y"
    run_fault_agent(fault, auto_mode=auto)
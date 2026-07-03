import csv
import json
import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# ==================== 环境配置 ====================
load_dotenv("env.env")
API_KEY = os.getenv("API_KEY")
API_BASE = os.getenv("API_BASE")

# ==================== 加载评论数据 ====================
def load_reviews(filepath, num=20):
    reviews = []
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= num:
                break
            comment = row.get('Comments', '').strip()
            if comment:
                reviews.append(comment)
    return reviews

# ==================== 全局变量 ====================
current_review_global = ""
current_tags_global = []
all_results = []

# ==================== 状态定义 ====================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ==================== 工具定义 ====================
@tool
def update_tags(tags: str) -> str:
    """更新当前评论的主题标签列表。

    Args:
        tags: 逗号分隔的标签字符串，例如 "干净舒适,设施齐全,位置便利"
    """
    global current_tags_global
    current_tags_global = [t.strip() for t in tags.split(",")]
    return f"标签已更新为: {current_tags_global}"

@tool
def save_review(filename: str) -> str:
    """将当前评论和标签保存到文本文件，并标记当前评论处理完成。

    Args:
        filename: 保存的文本文件名。
    """
    global current_tags_global, current_review_global, all_results

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    result = {"review": current_review_global, "tags": list(current_tags_global)}
    all_results.append(result)

    with open(filename, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n\n")

    return f"已保存到 {filename}！标签: {current_tags_global}"

tools_list = [update_tags, save_review]

# ==================== LLM 配置 ====================
model = ChatOpenAI(
    model="Qwen/Qwen2.5-72B-Instruct",
    api_key=API_KEY,
    base_url=API_BASE,
    temperature=0.3
).bind_tools(tools_list)

# ==================== 节点函数 ====================

def review_loader(state: AgentState) -> AgentState:
    """加载评论，构建系统提示词和初始人类消息"""
    global current_tags_global
    current_tags_global = []

    system_prompt = SystemMessage(content=f"""你是一个Airbnb评论主题提炼助手。你的任务是：
1. 阅读评论内容，提炼出2-5个主题标签
2. 立即使用 update_tags 工具设置标签
3. 展示当前标签并等待人类反馈
4. 如果人类要求修改标签，使用 update_tags 工具更新标签
5. 当人类输入"保存"、"确认"、"好的"、"ok"或类似确认词时，使用 save_review 工具保存结果，文件名固定为 review_results

标签要求：
- 用中文描述，简洁（2-6个字）
- 反映评论中提到的具体方面（如"干净舒适"、"设施齐全"、"位置便利"、"房东热情"等）
- 每次只调用一个工具

当前评论内容：{current_review_global}""")

    human_message = HumanMessage(
        content=f"请分析以下Airbnb评论，提炼出主题标签：\n\n{current_review_global}"
    )

    print(f"\n{'─'*60}")
    print(f"评论内容: {current_review_global[:100]}...")
    print(f"{'─'*60}")

    return {"messages": [system_prompt, human_message]}


def topic_analyzer(state: AgentState) -> AgentState:
    """LLM分析节点：分析评论提炼标签，或根据人类反馈修改标签"""
    messages = list(state["messages"])

    # 非首次调用时，询问人类反馈
    if len(messages) > 2:
        last_msg = messages[-1]
        if isinstance(last_msg, ToolMessage) and last_msg.name == "update_tags":
            print(f"\n  当前标签: {current_tags_global}")
            user_input = input(
                "\n  请输入指令（修改标签如'添加XX'、'删除XX'；直接回车保存）: "
            )
            if not user_input.strip():
                user_input = "保存"
            print(f"  用户: {user_input}")
            messages.append(HumanMessage(content=user_input))
        elif isinstance(last_msg, AIMessage):
            user_input = input(
                "\n  请输入指令（修改标签如'添加XX'、'删除XX'；直接回车保存）: "
            )
            if not user_input.strip():
                user_input = "保存"
            print(f"  用户: {user_input}")
            messages.append(HumanMessage(content=user_input))

    response = model.invoke(messages)

    print(f"\n  AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"  调用工具: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": [response]}


def should_use_tools(state: AgentState) -> str:
    """条件路由：判断LLM是否调用了工具"""
    messages = state["messages"]
    if not messages:
        return "topic_analyzer"
    last_msg = messages[-1]
    if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "topic_analyzer"


def should_continue(state: AgentState) -> str:
    """条件路由：继续修改标签还是保存并进入下一条评论"""
    messages = state["messages"]
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            if msg.name == "save_review":
                return "result_formatter"
            elif msg.name == "update_tags":
                return "topic_analyzer"
            break
    return "topic_analyzer"


def result_formatter(state: AgentState) -> AgentState:
    """格式化输出当前评论的处理结果"""
    print(f"\n  ✓ 本条评论处理完成！")
    print(f"    评论: {current_review_global[:60]}...")
    print(f"    标签: {current_tags_global}")
    return state


# ==================== 图构建 ====================
graph = StateGraph(AgentState)

graph.add_node("review_loader", review_loader)
graph.add_node("topic_analyzer", topic_analyzer)
graph.add_node("tools", ToolNode(tools_list))
graph.add_node("result_formatter", result_formatter)

graph.set_entry_point("review_loader")

graph.add_edge("review_loader", "topic_analyzer")

graph.add_conditional_edges(
    "topic_analyzer",
    should_use_tools,
    {"tools": "tools", "topic_analyzer": "topic_analyzer"},
)

graph.add_conditional_edges(
    "tools",
    should_continue,
    {"topic_analyzer": "topic_analyzer", "result_formatter": "result_formatter"},
)

graph.add_edge("result_formatter", END)

app = graph.compile()


# ==================== 运行入口 ====================
def process_single_review(review_text):
    """处理单条评论"""
    global current_review_global, current_tags_global
    current_review_global = review_text
    current_tags_global = []

    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        pass

    return {"review": current_review_global, "tags": list(current_tags_global)}


def run_review_agent(num_reviews=20):
    """运行Agent处理指定数量的评论"""
    global all_results

    reviews = load_reviews("reviews.csv", num=num_reviews)
    all_results = []

    # 清空已有的结果文件
    if os.path.exists("review_results.txt"):
        os.remove("review_results.txt")

    print(f"\n{'='*60}")
    print(f"  Airbnb评论主题提炼 Agent")
    print(f"  将处理 {len(reviews)} 条评论")
    print(f"{'='*60}")

    for i, review in enumerate(reviews):
        print(f"\n{'#'*60}")
        print(f"  正在处理第 {i+1}/{len(reviews)} 条评论")
        print(f"{'#'*60}")
        process_single_review(review)

    # 打印最终结果
    print(f"\n\n{'='*60}")
    print(f"  所有评论处理完成！最终结果：")
    print(f"{'='*60}")
    for r in all_results:
        print(f'\n  评论: {r["review"][:80]}...')
        print(f'  标签: {r["tags"]}')

    # 打印txt文件内容
    print(f"\n\n{'='*60}")
    print(f"  review_results.txt 文件内容：")
    print(f"{'='*60}")
    try:
        with open("review_results.txt", "r", encoding="utf-8") as f:
            print(f.read())
    except FileNotFoundError:
        print("  文件未找到")

    return all_results


if __name__ == "__main__":
    run_review_agent()
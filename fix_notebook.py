"""修改 Notebook 使其与当前 fault_agent.py 代码对齐"""
import json

with open('期末大作业.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 7: 更新导入（移除不再需要的 llm_with_tools，添加 Command）
cell7_src = """from fault_agent import (
    build_app, run_fault_agent,
    AgentState, tools_list,
    dispatch_node, retrieve_internal_node, search_agent_node,
    filter_knowledge_node, generate_diagram_node,
    audit_diagram_node, evaluate_questions_node,
    ask_expert_per_question_node, refine_diagram_node,
    tavily_search, llm
)
from langgraph.types import Command

app = build_app()
print('Agent 构建完成！')
print(f'图节点: {list(app.get_graph().nodes.keys())}')
print(f'图边数: {len(list(app.get_graph().edges))}')"""

nb['cells'][7]['source'] = cell7_src.split('\n')
nb['cells'][7]['source'] = [line + '\n' for line in cell7_src.split('\n')]
# fix last line no newline
if nb['cells'][7]['source'] and nb['cells'][7]['source'][-1].endswith('\n'):
    nb['cells'][7]['source'][-1] = nb['cells'][7]['source'][-1].rstrip('\n')

# Cell 9: 更新图可视化（添加边信息打印）
cell9_src = """from IPython.display import Image, display

try:
    img = app.get_graph().draw_mermaid_png()
    display(Image(img))
except Exception as e:
    print(f'无法渲染图片: {e}')

print('\\n=== 图结构详情 ===')
print('节点:', list(app.get_graph().nodes.keys()))
print('\\n边:')
for edge in app.get_graph().edges:
    cond = ' (条件)' if edge.conditional else ''
    data = f' [{edge.data}]' if edge.data else ''
    print(f'  {edge.source} -> {edge.target}{data}{cond}')"""

nb['cells'][9]['source'] = [line + '\n' for line in cell9_src.split('\n')]
if nb['cells'][9]['source'][-1].endswith('\n'):
    nb['cells'][9]['source'][-1] = nb['cells'][9]['source'][-1].rstrip('\n')

# Cell 16: 修复标题编号（从"6."改为"7."）
cell16_src = nb['cells'][16]['source']
cell16_text = ''.join(cell16_src)
cell16_text = cell16_text.replace('## 6. MCP等效功能验证', '## 7. MCP等效功能验证')
nb['cells'][16]['source'] = [cell16_text] if '\n' not in cell16_text else [line + '\n' for line in cell16_text.split('\n')]
if nb['cells'][16]['source'][-1].endswith('\n\n'):
    nb['cells'][16]['source'][-1] = nb['cells'][16]['source'][-1].rstrip('\n')

# Cell 18: 更新架构总结
cell18_text = """## 8. 架构总结

### 节点一览（10个功能节点 + 1个ToolNode）
| 节点 | 功能 | MessageState类型 |
|------|------|-----------------|
| dispatch | 系统角色注入 | SystemMessage |
| search_agent | 程序化创建tool_calls，触发外部搜索 | AIMessage (tool_calls) |
| retrieve_internal | Milvus RAG检索内部SOP | AIMessage |
| tools_node | 执行tavily_search @tool返回搜索结果 | ToolMessage |
| filter | 内外知识分层融合（内SOP为主，外搜索为辅） | AIMessage |
| generate | 生成Mermaid流程图 | AIMessage |
| audit | 自主审计流程图（缓存推理框架thinking_framework） | AIMessage |
| evaluate_questions | 初始化逐问迭代 | AIMessage |
| ask_expert | 逐个问题收集专家反馈（interrupt暂停/Command恢复） | HumanMessage |
| refine | 结合反馈最小化修改流程图 | AIMessage |

### 两条循环（Loop）
- **逐问循环**：ask_expert → more_questions_router → ask_expert（逐个审计问题收集反馈，使用interrupt()暂停）
- **修订循环**：refine → audit → check_gaps → evaluate_questions → ask_expert → refine（修订后重新审计，revision_count限制3次）

### 并发（Concurrency）
- **Fan-out/Fan-in**：search_agent → retrieve_internal ‖ tools_node → filter
- 内部SOP和外部搜索并发执行，在filter节点合并

### 知识分层融合
- 内SOP为主（权威）、外搜索为辅（标注参考），冲突以内SOP为准
- 方案B：search_agent程序化创建tool_calls，确保每次都执行外部搜索并产生ToolMessage

### 特性满足
- **>=3种MessageState**: SystemMessage, HumanMessage, AIMessage, ToolMessage (4种) ✅
- **>=4个功能节点**: 10个功能节点 + 1个ToolNode = 11个 ✅
- **Loop**: 逐问循环 + 修订循环（两条循环） ✅
- **Concurrency**: Fan-out/Fan-in ✅
- **知识分层融合**：内SOP为主，外搜索为辅 ✅
- **逻辑闭环**: 故障输入 → 并发检索 → 分层融合 → 生成 → 审计 → 逐问反馈 → 修订 → 重新审计 → 输出 ✅
- **云端大模型**: SiliconFlow DeepSeek-V4-Flash ✅

### 人机协同机制
- **interrupt()**：ask_expert节点内部调用interrupt()暂停等待专家反馈
- **Command(resume=feedback)**：run_fault_agent通过Command恢复图执行
- **auto_mode**：自动模式下由run_fault_agent自动生成反馈，无需人工输入

### MCP等效功能集成
- **Sequential Thinking**: 审计节点注入结构化推理框架，首次生成后缓存到state（替代 @modelcontextprotocol/server-sequential-thinking）
- **Filesystem**: 自动保存Mermaid流程图(.mmd)和诊断报告(.md)到output/diagrams/（替代 @modelcontextprotocol/server-filesystem）
- **Memory**: 诊断结果存入output/memory/的JSON文件，支持关键词搜索历史诊断（替代 @modelcontextprotocol/server-memory）

### 生产级优化
- LLM调用指数退避重试（5次）
- Prompt长度截断保护（internal 6000字符、external 6000字符、context 8000字符）
- 审计问题数限制（最多3个）
- 修订Prompt最小化修改约束
- thinking_framework缓存避免审计焦点漂移
- external_knowledge去重避免累加
"""

nb['cells'][18]['source'] = [line + '\n' for line in cell18_text.split('\n')]
if nb['cells'][18]['source'][-1].endswith('\n\n'):
    nb['cells'][18]['source'][-1] = nb['cells'][18]['source'][-1].rstrip('\n')

# 清除所有代码 Cell 的执行结果
for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []

with open('期末大作业.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print('Notebook 更新完成')
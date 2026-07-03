import os
import warnings
warnings.filterwarnings("ignore")
os.environ['PYMILVUS_DEPRECATION_WARNINGS'] = '0'

from dotenv import load_dotenv
load_dotenv()

print('环境变量加载完成:')
ark_key = os.environ.get('ARK_API_KEY', '')
print(f'  ARK_API_KEY: ***{ark_key[-6:]}' if ark_key else '  ARK_API_KEY: NOT SET')
base_url = os.environ.get('ARK_BASE_URL', 'NOT SET')
print(f'  ARK_BASE_URL: {base_url}')
milvus_host = os.environ.get('MILVUS_HOST', 'NOT SET')
print(f'  MILVUS_HOST: {milvus_host}')
tavily_key = os.environ.get('trivily_key', '')
print(f'  TAVILY_API_KEY: ***{tavily_key[-6:]}' if tavily_key else '  TAVILY_API_KEY: NOT SET')
langfuse = os.environ.get('LANGFUSE_ENABLED', 'false')
print(f'  LANGFUSE_ENABLED: {langfuse}')

print('\n--- Cell 4: SOP文档 ---')
from sop_documents import SOP_DOCUMENTS
print(f'共 {len(SOP_DOCUMENTS)} 条SOP文档:')
for doc in SOP_DOCUMENTS:
    print(f'  [{doc["category"]}] {doc["title"]}')

print('\n--- Cell 5: 初始化Milvus ---')
from init_knowledge_base import connect_milvus, create_collection, insert_documents, verify_search
connect_milvus()
collection = create_collection()
insert_documents(collection)
verify_search(collection)
print('知识库初始化完成！')

print('\n--- Cell 7: 构建Agent ---')
from fault_agent import (
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
nodes = list(app.get_graph().nodes.keys())
edges = list(app.get_graph().edges)
print(f'图节点: {nodes}')
print(f'图边数: {len(edges)}')

print('\n--- Cell 9: 图结构 ---')
print('边:')
for edge in edges:
    cond = ' (条件)' if edge.conditional else ''
    data = f' [{edge.data}]' if edge.data else ''
    print(f'  {edge.source} -> {edge.target}{data}{cond}')

print('\n=== 准备类Cell执行完成 ===')
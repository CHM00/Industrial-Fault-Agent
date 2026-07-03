import os
import warnings
import json
warnings.filterwarnings("ignore")
os.environ['PYMILVUS_DEPRECATION_WARNINGS'] = '0'

from dotenv import load_dotenv
load_dotenv()

from fault_agent import build_app, run_fault_agent, AgentState
from langgraph.types import Command

app = build_app()
print('Agent 构建完成！')
print(f'图节点: {list(app.get_graph().nodes.keys())}')
print()

print('='*60)
print('测试用例1: 西门子 S7-1200 PLC 通讯故障 (auto_mode=True)')
print('='*60)

result1 = run_fault_agent('西门子 S7-1200 PLC 通讯故障', auto_mode=True)

print('\n' + '='*60)
print('测试用例1 - 消息类型验证:')
msg_types = {}
for m in result1.get('messages', []):
    t = type(m).__name__
    msg_types[t] = msg_types.get(t, 0) + 1
print(f'  {msg_types}')
print(f'  内部知识: {"有" if result1.get("internal_knowledge") else "无"}')
print(f'  过滤上下文: {"有" if result1.get("filtered_context") else "无"}')
print(f'  流程图长度: {len(result1.get("mermaid_diagram", ""))}字符')
print(f'  审计结果: has_gaps={result1.get("has_gaps")}, questions={len(result1.get("audit_questions", []))}')
print(f'  修订次数: {result1.get("revision_count", 0)}')
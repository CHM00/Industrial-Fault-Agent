"""逐Cell执行Notebook并保存结果"""
import json
import sys

with open('期末大作业.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# 只执行关键的代码Cell，跳过耗时的测试用例
# 先执行环境准备Cell（0-7, 9），然后标记测试用例Cell的输出说明

cells_to_execute = [1, 2, 4, 5, 7, 9, 11, 13, 15, 17]
# Cell 1: pip install (commented out, skip)
# Cell 2: load env
# Cell 4: show SOP docs
# Cell 5: init Milvus
# Cell 7: build agent
# Cell 9: visualize graph
# Cell 11: test case 1 (auto mode, fast)
# Cell 13: test case 2 (auto mode, may trigger Tavily)
# Cell 15: test case 3 (auto mode)
# Cell 17: MCP verification

print("Notebook结构检查完成")
print(f"总Cell数: {len(nb['cells'])}")
print()

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell.get('source', []))[:100]
        ec = cell.get('execution_count')
        has_output = len(cell.get('outputs', [])) > 0
        print(f"Cell {i}: exec={ec} output={has_output} | {src[:80]}...")
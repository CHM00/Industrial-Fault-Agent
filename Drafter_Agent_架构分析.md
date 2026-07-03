# Drafter Agent 架构分析报告

## 1. 概述

Drafter Agent 是一个基于 **LangGraph** 构建的轻量级文档起草助手，使用 **LangChain** 生态与通义千问（qwen-plus）模型，实现用户交互式的文档创建、更新与保存功能。

**技术栈**：LangGraph + LangChain + ChatOpenAI（qwen-plus）

**复杂程度**：⭐☆☆☆☆（极简）

---

## 2. 架构图

```
┌─────────────────────────────────────────────┐
│              StateGraph(AgentState)          │
│                                              │
│  ┌──────────┐         ┌──────────────┐     │
│  │ the_agent │────────▶│    tools     │     │
│  │  (LLM节点) │         │ (ToolNode)   │     │
│  └──────────┘         └──────┬───────┘     │
│       ▲                      │              │
│       │                      │              │
│       │   ┌──────────────┐   │              │
│       │   │should_continue│◀──┘              │
│       │   │  (条件路由)    │                  │
│       │   └──┬───────┬───┘                  │
│       │      │       │                      │
│       │ "continue"  "end"                  │
│       └──────┘       └──▶ END               │
│                                              │
│  Entry Point: the_agent                      │
└─────────────────────────────────────────────┘
```

**执行流程**：

1. 进入 `the_agent` 节点 → LLM 接收用户输入并决定是否调用工具
2. 强制进入 `tools` 节点 → 执行工具调用（update 或 save）
3. `should_continue` 条件路由：
   - 检测到 `save` 工具返回含 "saved" + "document" → 结束（END）
   - 否则 → 回到 `the_agent` 继续交互

---

## 3. 核心组件详解

### 3.1 状态定义（AgentState）

```python
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
```

- 仅维护一个 `messages` 字段，使用 `add_messages` reducer 实现消息累积追加
- 使用全局变量 `document_content` 存储文档内容（非状态内管理）

**设计问题**：文档内容通过全局变量而非 Graph State 管理，导致不支持多实例并行、无法持久化文档状态。

### 3.2 工具定义

| 工具 | 功能 | 参数 |
|------|------|------|
| `update` | 用新内容覆盖文档 | `content: str` |
| `save` | 将文档写入 .txt 文件并终止流程 | `filename: str` |

- `update`：直接替换全局变量 `document_content`，无增量更新能力
- `save`：自动补 `.txt` 后缀，写入本地文件，作为流程终止信号

### 3.3 Agent 节点（the_agent）

- 构建含 System Prompt + 历史消息 + 用户输入的消息列表
- System Prompt 中注入当前 `document_content`（让 LLM 感知文档现状）
- 首轮无消息时使用默认问候语，后续通过 `input()` 获取用户输入
- 输出 LLM 响应及工具调用信息到控制台

### 3.4 条件路由（should_continue）

- 遍历消息列表（倒序），查找 `ToolMessage` 中同时包含 "saved" 和 "document" 的内容
- 匹配 → 返回 `"end"` 终止流程
- 否则 → 返回 `"continue"` 继续循环

### 3.5 图结构（StateGraph）

```
the_agent → tools → [should_continue] → the_agent（继续）或 END（结束）
```

- `the_agent` 到 `tools` 为**无条件边**，即每次 LLM 响应后都会执行工具节点
- **设计缺陷**：即使 LLM 未调用任何工具，也会进入 ToolNode，可能导致空执行或异常

---

## 4. 复杂度评估

| 维度 | 评级 | 说明 |
|------|------|------|
| 节点数 | ⭐☆☆☆☆ | 仅 2 个节点（agent + tools） |
| 工具数 | ⭐☆☆☆☆ | 仅 2 个工具（update、save） |
| 条件路由 | ⭐☆☆☆☆ | 1 个简单条件判断 |
| 状态复杂度 | ⭐☆☆☆☆ | 单字段状态 + 全局变量 |
| 错误处理 | ⭐☆☆☆☆ | 仅 save 有 try-except |
| 可扩展性 | ⭐☆☆☆☆ | 全局变量管理文档，难以扩展 |
| **总体** | **⭐☆☆☆☆** | **极简原型级 Agent** |

---

## 5. 设计问题与改进建议

| 问题 | 说明 | 建议 |
|------|------|------|
| 全局变量管理文档 | `document_content` 为全局变量，无法多实例运行、不支持 Checkpoint 持久化 | 将文档内容纳入 `AgentState` 管理 |
| 无条件边到工具 | `the_agent → tools` 为固定边，LLM 未调用工具时也会进入 ToolNode | 改为条件边：有 tool_calls → tools，无 → 继续或结束 |
| 文档更新为全量替换 | `update` 工具直接覆盖内容，无增量编辑能力 | 增加 `append`/`edit` 工具支持局部修改 |
| 无用户输入校验 | `input()` 可返回空字符串，LLM 无法处理 | 增加输入验证与重试机制 |
| 终止判断脆弱 | 依赖 ToolMessage 内容中是否含 "saved"+"document" 字符串 | 改为基于工具名称（`save`）判断，更鲁棒 |
| 无流式输出 | 整体 `invoke` 后才返回，用户等待时间长 | 使用 `astream` 实现流式响应 |

---

## 6. 文件结构

```
Drafter Agent.ipynb
├── Cell 1: 环境初始化（导入、API配置）
├── Cell 2: 测试代码（全局变量实验，可删除）
├── Cell 3: 状态定义（AgentState + 全局变量）
├── Cell 4: 工具定义（update、save）
├── Cell 5: 核心逻辑（LLM、agent节点、条件路由、打印函数）
├── Cell 6: 图构建与编译（StateGraph → app）
├── Cell 7: 运行入口（run_document_agent）
└── Cell 8: 执行调用
```

总代码量约 **120 行**（含空白和注释），属于教学/原型级别的最小 LangGraph Agent 实现。
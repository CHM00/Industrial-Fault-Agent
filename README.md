# 工业故障诊断 LangGraph Agent

这是一个面向工业现场故障排查的 AI Agent 项目。项目以 LangGraph 编排诊断流程，结合 Milvus/Zilliz 向量知识库、Tavily 外部检索和大模型推理，自动生成可审计、可修订的 Mermaid 故障排查流程图，并将诊断报告与历史案例保存到本地。

## 核心能力

- **SOP 知识库检索**：从 Milvus/Zilliz 或本地 milvus-lite 中检索工业故障 SOP 知识。
- **可控外部研究**：支持关闭、基础研究和主管研究模式，可选 Tavily、DuckDuckGo、SearXNG 或 Perplexity，并对任务数、搜索深度和总调用量设限。
- **LangGraph 多节点编排**：按“检索 -> 融合 -> 生成 -> 审计 -> 专家反馈 -> 修订”的流程执行。
- **人机协同审计**：当流程图存在缺口时，Agent 会提出问题并等待专家反馈；也支持自动反馈模式。
- **报告与记忆沉淀**：输出 Mermaid 流程图、Markdown 诊断报告和 JSON 历史诊断记忆。
- **Web/SSE 接口**：`web_server.py` 提供流式诊断 API，便于接入前端。
- **稳定试点治理**：支持持久化任务/checkpoint、API Key 与 RBAC、租户隔离、限流、并发控制、任务取消与重试。
- **工业安全门**：确定性规则在任何检索和模型调用之前评估风险，高风险任务必须由 expert/admin 明确审批。
- **P1 业务闭环**：设备资产、结构化诊断、SOP 版本管理、历史案例 RAG、步骤证据、流程审批、现场文档导出、任务中心和质量成本看板。

## 模型框架

![工业故障诊断模型框架图](fig/model.png)

## 诊断流程

```text
故障描述
  -> 设备/报警/工况/测点结构化上下文
  -> 内部 SOP 检索（Milvus + 受控知识库）与历史案例召回
  -> 外部研究网关（off/basic/supervisory）
  -> 知识过滤与融合
  -> 生成 Mermaid 排查流程图并映射步骤证据
  -> 审计完整性与准确性
  -> 如有缺口，收集专家反馈并修订
  -> 专家编辑、版本 Diff、审批发布并沉淀案例/指标
```

## 项目结构

```text
.
├── fault_agent.py             # 核心 LangGraph Agent：诊断、审计、反馈修订
├── mermaid_pipeline.py        # Mermaid 提取、清洗与官方解析器校验
├── validate_mermaid.mjs       # Node.js Mermaid 语法校验入口
├── init_knowledge_base.py     # 初始化工业故障 SOP 向量知识库
├── sop_documents.py           # 内置 SOP 文档数据
├── mcp_tools.py               # 保存报告、流程图、诊断记忆等工具函数
├── business_store.py          # P1 资产、知识版本、案例、流程版本和指标存储
├── document_ingestion.py      # SOP 文档安全解析、分块与增量索引
├── evidence_mapping.py        # Mermaid 节点到 SOP/案例/外部证据映射
├── artifact_export.py         # Word、PDF 和现场检查单导出
├── research/                  # 统一研究契约、预算、供应商、基础图和主管图
├── web_server.py              # FastAPI + SSE Web/API 服务
├── Milvus.py                  # Milvus 连接与集合管理封装
├── md_to_word.py              # Markdown 报告转 Word 的辅助脚本
├── run_*.py / fix_*.py        # Notebook/测试辅助脚本
└── output/
    ├── diagrams/              # 生成的 Mermaid 与 Markdown 报告
    └── memory/                # 诊断历史 JSON
```

## 环境变量

在项目根目录创建 `.env`，至少配置：

```env
LLM_API_KEY=你的大模型或兼容 OpenAI 接口的 API Key
LLM_BASE_URL=https://api.siliconflow.cn/v1
LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
TAVILY_API_KEY=你的 Tavily API Key
# CLI/未指定请求的研究总时限；Web 界面可按会话覆盖，0 表示不限时
RESEARCH_TOTAL_TIMEOUT_SECONDS=120

# Docker/生产试点必填
APP_ENV=production
AUTH_ENABLED=true
AUTH_SESSION_SECRET=请替换为至少32字符的随机值
AUTH_USERS_JSON={"admin":{"display_name":"系统管理员","password":"请替换强密码","subject":"admin-1","role":"admin","tenant_id":"factory-a"}}
CHECKPOINT_BACKEND=sqlite

# 可选：用于质量成本看板的模型单价
LLM_INPUT_COST_PER_1M=0
LLM_OUTPUT_COST_PER_1M=0
```

Web 用户使用配置账号登录，页面会显示当前用户、角色和租户，服务端使用 HttpOnly 会话 Cookie 承载登录状态。执行 `python security.py` 可生成生产环境推荐的 `password_hash`。`PILOT_API_KEY` 和 `API_KEYS_JSON` 仅保留给 CMMS/EAM 等机器系统调用，不再出现在 Web 页面中。

旧的 `ARK_API_KEY`、`ARK_BASE_URL` 和 `trivily_key` 暂时兼容，但新部署应使用 `LLM_*` 与正确拼写的 `TAVILY_API_KEY`。

Web 界面的“总时限”支持不限时、2 分钟、5 分钟和 10 分钟。界面默认“不限时”；API 未传 `research_timeout_seconds` 时继续使用 `RESEARCH_TOTAL_TIMEOUT_SECONDS`，传 `0` 表示禁用研究总时限。

Milvus 连接二选一即可：

```env
# 方式一：自建 Milvus
MILVUS_HOST=127.0.0.1
MILVUS_PORT=19530
MILVUS_USER=root
MILVUS_PASSWORD=

# 方式二：Zilliz Cloud
URL=你的 Zilliz/Milvus URI
Token=你的 Token
```

如果不配置 Milvus 服务，代码会尝试使用 `output/milvus_local.db` 作为本地 milvus-lite 存储。

## 安装依赖

使用 Python 3.10 或更高版本安装 Python 依赖：

```bash
pip install -r requirements.txt
```

流程图在进入 LangGraph 状态前会调用官方 `mermaid.parse()` 校验，因此还需安装 Node.js 18 或更高版本以及固定版本的 Mermaid 依赖：

```bash
npm ci
```

生成或专家修订后的流程图如果存在语法错误，系统会把解析器错误交给模型做一次最小化修复并再次校验。第二次仍失败时终止本次诊断，不会保存未经校验的流程图。可通过 `MERMAID_MAX_CHARS` 和 `MERMAID_VALIDATION_TIMEOUT_SECONDS` 调整最大字符数与单次校验超时。

## 快速开始

### 1. 初始化知识库

```bash
python init_knowledge_base.py
```

该命令会读取 `sop_documents.py`，生成 embedding，并写入 `industrial_fault_knowledge` 集合。

### 2. 命令行诊断

```bash
python fault_agent.py
```

按提示输入故障描述，并选择：

- `y`：自动模式，Agent 自动补充审计反馈。
- `n`：人机协同模式，审计发现缺口时等待专家逐条反馈。

随后选择外部研究模式和检索深度。`off` 不访问互联网，`basic` 适合一般故障，`supervisory` 会拆分多个研究任务后综合。

### 3. Web/API 服务

```bash
python web_server.py
```

默认监听：

```text
http://localhost:8000
```

主要接口：

- `POST /api/start`：开始诊断，返回 SSE 流。
- `POST /api/resume`：提交专家反馈并继续诊断。
- `GET /api/history`：查看历史诊断记录。
- `GET /api/diagram/{name}`：读取 Mermaid 流程图。
- `GET /api/report/{name}`：读取 Markdown 诊断报告。
- `GET /api/memory/{name}`：读取诊断记忆 JSON。
- `GET/POST/PATCH /api/assets`：设备资产管理。
- `GET/POST/PATCH /api/knowledge/...`：SOP 上传、版本、权限、生效状态与检索。
- `GET/POST /api/procedures/...`：流程版本、Diff、批准、拒绝和发布。
- `POST /api/cases/{session_id}/confirm`：回填根因与处理结果，沉淀可召回案例。
- `GET /api/jobs/{session_id}/export?format=docx|pdf|checklist`：导出现场文档。
- `GET /api/dashboard/quality`：质量、耗时、Token 与估算成本汇总。

请求示例：

```json
{
  "fault_input": "PLC 与上位机通信中断，HMI 显示通信超时",
  "auto_mode": false,
  "research_mode": "basic",
  "research_depth": 2,
  "max_research_tasks": 3,
  "max_total_searches": 6,
  "search_api": "tavily",
  "asset_id": "可选的设备资产ID",
  "alarm_code": "F4",
  "operating_context": "冷启动、50%负载",
  "measurements": {"母线电压": 540, "电流": 83},
  "maintenance_history": "昨日更换电机电缆"
}
```

`auto_mode` 控制审计问题是否自动反馈，与尚未开放的外部研究 `auto` 模式不是同一个参数。Web API 当前只接受 `off/basic/supervisory`。

## 输出结果

诊断完成后会生成：

- `output/diagrams/*_diagram.mmd`：Mermaid 流程图。
- `output/diagrams/*_report.md`：完整诊断报告。
- `output/memory/*.json`：结构化历史诊断记忆。
- `output/runtime/pilot.sqlite3`：任务、资产、知识版本、案例、流程审批和指标。
- Web 端按任务导出 `.docx`、`.pdf` 与带 BOM 的现场检查单 `.csv`。

## 注意事项

- `web_server.py` 默认挂载 `static/app.html`。
- `main.py` 引用了 `Local_Model`、`Knowledge_Grpah`、`ContextRouter` 等当前仓库未提供的模块，更像是实验性/扩展入口；推荐优先使用 `fault_agent.py` 和 `web_server.py`。
- 源码中部分中文注释或字符串存在编码乱码，不影响整体架构理解，但建议后续统一为 UTF-8。
- `.env`、`env.env` 等密钥文件已加入 `.gitignore`，不要提交真实 API Key。
- Docker/生产模式不会在 SQLite checkpoint 初始化失败时静默降级到内存；启动前必须安装 `requirements.txt` 中的官方 checkpoint 依赖。
- SQLite 方案面向单实例试点。多副本部署前应迁移到集中式 checkpoint 与任务数据库。

## P0 稳定试点

认证角色、持久化文件、任务状态、安全审批、健康检查和运维说明参见 [P0 稳定试点部署指南](docs/P0_稳定试点部署指南.md)。

## P1 专家效率与业务闭环

资产/SOP 管理、结构化诊断、案例沉淀、流程审批、导出、任务中心和看板的操作与验收方式参见 [P1 功能使用与验收指南](docs/P1_功能使用与验收指南.md)。

### 受控 SOP 与 Milvus 联动

页面“受控 SOP 文档库”采用三层存储：

- 原文件保存在 `KNOWLEDGE_FILES_DIR`（默认 `output/knowledge`）；
- SQLite `knowledge_documents` / `knowledge_versions` / `knowledge_chunks` 保存文档、版本、权限和分块，是权威数据源；
- 当前有效版本的分块会生成 Embedding，并同步到 Milvus 集合 `managed_sop_knowledge`（可用 `MANAGED_KNOWLEDGE_COLLECTION` 修改）。

上传新版本时会替换该文档的旧向量；停用或归档文档时会从 Milvus 删除向量；重新启用或点击“重新同步”会重建向量。诊断优先使用 Milvus 语义检索，Milvus 或 Embedding 服务异常时自动降级为 SQLite 文本相似度检索，并在页面显示同步状态和错误。

原来的 `industrial_fault_knowledge` 集合仍保存 `sop_documents.py` 初始化的内置 SOP。两个集合在诊断的内部知识融合节点统一使用，但受控集合额外支持租户、角色、版本和分块定位，因此不与旧集合共用 schema。

### 设备治理与任务中心

- 设备资产支持上级设备、序列号、固件、关键度和扩展 metadata；`metadata.measurement_template` 可定义测点单位及正常上下限。
- 诊断测点支持简单值或 `{ "value": 85, "unit": "℃" }`，系统会校验单位、标注是否超出正常范围，并按设备沉淀趋势统计。
- `critical` 设备会把确定性安全等级至少提升到 `high`，必须经过 expert 安全复核。
- SOP 可设置 `asset_type`、`vendor`、`model`、`firmware`、`valid_from` 和 `valid_to`；系统先进行适用范围硬过滤，再进入 Milvus 语义排序。
- `POST /api/assets/import` 提供 CMMS/EAM/ERP 通用幂等导入桥，按租户和设备编号新增或更新。
- 设备详情展示诊断任务、已确认案例、质量指标和历史测点趋势。
- 任务中心支持中文状态、设备/状态/关键字筛选、分页、批量归档，并可直接恢复 `waiting_feedback` 或处理 `waiting_safety`。

## 适用场景

该项目适合用于工业设备故障诊断、维修 SOP 辅助生成、专家经验沉淀，以及将历史排障案例逐步转化为可检索知识库的原型系统。

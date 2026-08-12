# MiroFish: Zep Cloud → Graphiti 迁移设计

> **状态：代码迁移已完成（P1–P7），待用户安装 graphiti-core 后跑测试与端到端验证。**

## Context（为什么做这个改造）

MiroFish 当前用 **Zep Cloud**（`zep-cloud==3.25.0`）作为时序知识图谱后端，承担三件事：

1. **图谱构建**：上传文档 → 生成本体 → 分块摄入（Batch API）→ 形成 namespaced graph
2. **仿真记忆**：仿真运行时把 agent 行为流式写入已有 graph（`graph.add`）
3. **检索问答**：报告生成阶段用 `graph.search` + 节点/边分页读取做证据召回

Zep Community Edition 已停更（代码移入 `legacy/`），官方明确把 **Graphiti**（`getzep/graphiti`，Apache-2.0）作为开源时序知识图谱框架——它就是 Zep Cloud 背后的开源引擎本身。能力上 1:1 对齐：bi-temporal、自定义本体、episode+provenance、语义+图遍历检索。

**这不是改个 `base_url` 就行**——Zep Cloud SDK 用自家专有图数据库 REST API，Graphiti 用 Neo4j + Python async SDK，API 契约完全不同。本设计目标是**最小化对调用方（API 路由、report agent、simulation runner）的破坏**，通过 adapter 层吸收差异。

---

## 一、关键决策（4 个，需你确认）

> **已核实 graphiti-core 实际 API（v0.29.3，2026-08 查证）**，下文已据实修正。关键修正：
> - 版本：`graphiti-core>=0.6.0,<0.7` ❌ → 实际 `graphiti-core>=0.29`，已据此修正 P1 依赖范围。
> - **Neo4j 驱动版本**：README 说"需 Neo4j **服务器** 5.26+"，但 graphiti-core 的 Python **驱动包依赖是 neo4j 6.x**（不是 5.x）。故 pyproject 用 `neo4j>=5.26`，实际装 6.2.0；docker-compose 用 `neo4j:5.26` 服务器（驱动向后兼容）。
> - **camel-oasis 打包 bug**：`camel-oasis==0.2.5` 把 neo4j 硬钉成 `==5.23.0`（camel-ai 实际只需 `>=5.18,<6`，且 MiroFish 不用 camel-ai 的 neo4j 图存储）。与 graphiti-core 的 neo4j 6.x 冲突。解法：pyproject 加 `[tool.uv] override-dependencies = ["neo4j>=6.0.0"]` 强制放宽，camel-ai 的 neo4j import（Query/ClientError/CypherSyntaxError）在 6.x 仍存在，运行兼容。
> - 客户端构造：`Graphiti(uri, user, password, llm_client, embedder)`（或 `graph_driver=Neo4jDriver(...)`），**非** `driver=`。
> - 初始化：`build_indices_and_constraints()`（Neo4jDriver init 会自动后台触发，无需手动调）。
> - **本体注册是 per-call 而非 client-global**：无 `add_custom_types` 方法，自定义类型通过 `add_episode(entity_types=, edge_types=, edge_type_map=)` 每次调用传入——决策 3 的约束自动消失，多本体天然隔离。
> - 检索：`search(query, group_ids, num_results, search_filter)`→`list[EntityEdge]`（仅边）；`search_(query, config, group_ids)`→`SearchResults`（含 nodes/edges/episodes）。用 `search_` 获取节点+边。
> - 节点/边读取走 **namespace**：`client.nodes.entity.get_by_group_ids(group_ids, limit, uuid_cursor)`、`client.edges.entity.get_by_group_ids(...)`、`client.edges.entity.get_by_node_uuid(node_uuid)`（**双向**，无 Zep SDK 的单向 bug）。
> - 删除：无 `delete_group`/`delete_all`；需对 4 节点 namespace + 5 边 namespace 各调 `delete_by_group_id(group_id)`，封装进 `graphiti_client.delete_group()` helper。
> - 字段：Node.uuid（非 `uuid_`）、Edge.source_node_uuid/target_node_uuid（非 `*_id`）、EntityEdge 无 `fact_type`（关系名就是 `name`）。

> **本地验证状态（2026-08-11）**：`uv sync` 成功；130 测试全过；Neo4j 5.26 容器启动、Bolt 连通、neo4j 6.2.0 驱动连接 5.26 服务器查询成功；**端到端验证脚本 PASSED**（2 episodes 摄入成功，LLM 抽取出 5 节点 3 边落库，cleanup 删除成功）。

> **网关兼容性修复（红杉 ai-gateway）**：本地验证中发现并修复了 4 个非标 OpenAI 网关兼容问题，均集中在 `graphiti_client.py`：
> 1. **neo4j 版本冲突**：camel-oasis==0.2.5 错误硬钉 neo4j==5.23.0，与 graphiti-core 需要的 neo4j 6.x 冲突。用 `[tool.uv] override-dependencies` 放宽。
> 2. **cross_encoder/small_model 初始化失败**：Graphiti 默认用 `OpenAIRerankerClient`（读 OPENAI_API_KEY）和 `small_model=gpt-4.1-nano`（网关无此模型）。改为复用 MiroFish 的 LLM 配置构造 cross_encoder、small_model 复用主模型。
> 3. **base_url 漏 /v1**：红杉网关 base_url 不带 /v1 时，responses 端点拼成 `/responses`（404）。加自动补 /v1 逻辑。
> 4. **Responses API structured output 缺陷**：红杉网关的 `responses.parse`（及 `responses.create` + `text.format`）返回的 `output_text` 把 JSON Schema（`input_schema` 字段）与数据混在一起，甚至有时只返回 schema 不返回数据。openai SDK 的 `responses.parse` 内部用 `model_validate_json` 验证，遇噪声直接抛 ValidationError 且异常里丢失 response 对象无法恢复。**解法**：patch `OpenAIClient._create_structured_completion`，改用 `responses.create` + `text.format=json_schema`（不走 SDK 的 post_parser 验证，保留 response 对象），自己 `json.loads(output_text)` 后剔除 `input_schema`/`schema`/`json_schema` 噪声键，并做数据完整性校验（缺必填字段时抛异常触发 graphiti 重试）；`MAX_RETRIES` 从 2 提到 6。注意：Responses API 路径在网关状态差时成功率较低，重试 6 次仍有失败可能，根本修复需网关侧修正 structured output 响应格式。
> 5. **delete_by_group_id 事务冲突**：graphiti 的 `delete_by_group_id` 用 `driver.session()` 执行 `CALL {} IN TRANSACTIONS`，neo4j 6.x 驱动要求该语法在 implicit transaction 运行。`graphiti_client.delete_group` helper 改用 `driver.execute_query`（implicit）执行简单 `DETACH DELETE`。
>
> 上述修复均为环境兼容性适配，MiroFish 业务逻辑未变。换用 OpenAI 原生端点时这些 patch 无副作用（patch 幂等、条件触发）。

### 决策 1：in-process Graphiti（推荐） vs standalone REST server

Graphiti 是 **async-first**；MiroFish 是 **sync Flask + daemon 线程**。两种桥接方式：

| 维度 | (a) in-process（共享 event loop 线程） | (b) Graphiti FastAPI server + httpx sync |
|---|---|---|
| 网络跳数 | 0（进程内） | 1（HTTP） |
| 额外服务 | 无 | 多一个 server 进程 + 一套配置 |
| 错误模型 | 原生 Graphiti/Neo4j 异常，映射干净 | HTTP 状态码，lossy |
| `add_episode` 长调用 | `await` 天然支持，loop 线程不阻塞其他 sync 调用 | 需把 httpx timeout 拉到 ~300s，破坏现有 60s 读策略 |
| 现有 retry 策略复用 | `call_zep_read_with_retry` 形状几乎不变，改名 `call_graphiti_read_with_retry` | 直接复用 httpx retry，但语义错位 |
| 风险 | loop 线程成为串行瓶颈；绝不能在 loop 线程内回调 sync 代码（死锁） | 部署面变大；长 episode 的 HTTP timeout 难调 |

**推荐 (a) in-process**。理由：MiroFish 是单租户、单仿真/graph 的并发模型，graph 操作不是热路径，loop 串行化可接受；且避免第二套服务运维成本。loop 放在新模块 `backend/app/utils/graphiti_runtime.py`。

### 决策 2：保留服务类名（推荐） vs 重命名

`ZepToolsService` / `ZepEntityReader` / `ZepGraphMemoryUpdater` / `ZepGraphMemoryManager` 这些类名被 `report_agent.py`、`simulation_runner.py`、`simulation_manager.py`、`api/{graph,simulation,report}.py` 等 6+ 处引用。

**推荐保留类名**（内部实现换成 Graphiti），把重命名留作独立重构。这样迁移 diff 聚焦在"换后端"而非"改名"，降低 review 难度。

### 决策 3：本体注册语义变化

Zep 的 `set_ontology` 是 **per-graph**（`graph_ids=[graph_id]`）；Graphiti 的 `add_custom_types` 是 **client-global**（自定义类型注册到整个 Graphiti 实例，不按 group 隔离）。

**应对**：MiroFish 单 Graphiti 实例只服务一个本体（每次 build 新本体前可 `clear_custom_types` 或直接重建 client）。若未来要多本体共存，需每本体一个 Graphiti 实例。当前单租户场景下接受此约束。

### 决策 4：Batch API 丢失 → UUID5 幂等 + gather

Zep Batch API（`batch.create/add/process`）提供原子批摄入 + `operation_id` 幂等 surrogate + per-item `episode_uuid` 跟踪。**Graphiti 无对应物**。

**替代方案**：`asyncio.gather` + `Semaphore(8)` 并发调用 `add_episode`，每个 episode 用 **`uuid5(operation_id, chunk_index)`** 作为客户端生成 UUID——重跑同 `operation_id`+chunk_index 产出相同 UUID，`add_episode(uuid=...)` 幂等，**这比 Zep 的 `graph.add`（无幂等键）更安全**。代价：丢失原子性（中批失败留半成品图），恢复策略 = `force=True` 重建（UUID5 保证不重复）。

---

## 二、改造范围（核实后的精确文件清单）

### 客户端层（utils/）

| 文件 | 处置 | 说明 |
|---|---|---|
| `utils/zep.py` (163 行) | **重写为 `utils/graphiti_client.py`** | 客户端工厂 + retry 策略；`get_zep_client`→`get_graphiti_client`；`is_retryable_zep_error`→`is_retryable_graphiti_error`（`asyncio.TimeoutError`/`neo4j.ServiceUnavailable`/`OSError` 可重试，`AuthError` 不可重试，删除 `Retry-After` 逻辑） |
| `utils/zep_paging.py` (161 行) | **重写为 `utils/graphiti_paging.py`** | Zep 的 `zep-next-cursor` 响应头分页 → Graphiti 的 `get_nodes(group_ids=[...])`/`get_edges(group_ids=[...])` 返回全量列表（无 cursor）。保留 `fetch_all_nodes/fetch_all_edges` 签名 + `max_items` 截断，`page_size`/`max_retries`/`retry_delay` 变 no-op kwargs 保后向兼容 |
| `utils/zep_lifecycle.py` (52 行) | **保留不动** | 纯 `threading.RLock` + 读者租约，无 SDK 调用，对任何后端通用。`graph_id` 参数名现在装的是 Graphiti `group_id`，语义不变 |
| `utils/graphiti_runtime.py` | **新建** | 共享 asyncio loop 守护线程 + `run_async(coro, timeout)`；带死锁 guard（禁止 loop 线程内回调 sync） |
| `app/exceptions.py` | **新建** | `GraphNotFound`（替代 `zep_cloud.NotFoundError`）、`GraphInUseError`（从 api/graph.py 提取复用） |

### 服务层（services/）

| 文件 | 处置 | 关键改动 |
|---|---|---|
| `services/graph_builder.py` (880 行) | **重写** | `create_graph`：Graphiti 无显式建图，`group_id` 首次 `add_episode` 隐式创建 → 方法退化为生成 ID + journal。`set_ontology`：删 `zep_cloud.external_clients.ontology.*` import，动态 Pydantic 类改继承 Graphiti `EntityBase`/`EdgeBase`，属性 `EntityText`→`str`，调 `add_custom_types`。`add_text_batches`：全重写为 `gather+Semaphore`，UUID5 幂等。`_wait_for_batch` 退化为校验数量。删 `_find_batch_by_operation_id`/`_list_batch_items`/`get_batch_summary`。`delete_graph`→`delete_group`。`get_graph_data`/`_get_graph_info`：换 paging + 字段映射 |
| `services/zep_graph_memory_updater.py` (792 行) | **重写 `_send_batch_activities`/`_wait_for_pending_episodes`** | `client.graph.add`→`client.add_episode`（`group_id` 替代 `graph_id`，`reference_time` 替代 `created_at`，UUID5 幂等）。`add_episode` 已 `await` 完整抽取 → `_wait_for_pending_episodes` 退化为防御性 episode 计数校验。线程/queue/buffer/`_failed_batches`/manager registry 全保留 |
| `services/zep_tools.py` (1735 行) | **重写原语层** | `search_graph`：`graph.search`→`graphiti.search`，`scope="edges"`→`EDGE_HYBRID_SEARCH_CROSS_ENCODER`，`scope="nodes"`→`NODE_HYBRID_SEARCH_RRF`。`get_node_detail`→`get_node`。`get_node_edges`→优先 `get_edges_for_node`。dataclass（`SearchResult`/`NodeInfo`/`EdgeInfo` 等）全保留作稳定契约。复合方法（`insight_forge`/`panorama_search`/`quick_search`）不动 |
| `services/zep_entity_reader.py` (447 行) | **重写原语层** | `get_all_nodes/get_all_edges` 换 paging + 字段映射；`get_node_edges`/`get_entity_with_context` 换 `get_edges_for_node`/`get_node`；`NotFoundError`→`GraphNotFound` |
| `services/oasis_profile_generator.py` (1249 行) | **重写 `_search_zep_for_entity`** | 两个并发 `graph.search(reranker="rrf")`→`graphiti.search(NODE/EDGE_HYBRID_SEARCH_RRF)`，保留 `ThreadPoolExecutor(max_workers=2)`（`run_coroutine_threadsafe` 本身线程安全） |
| `services/ontology_generator.py` (733 行) | **改 codegen 模板** | `generate_python_code` 输出的 import 串 `from zep_cloud.external_clients.ontology import ...`→`from graphiti_core.models.entity.entity import ...`；`EntityText`→`str` |

### 调用方（API 路由等，最小改动）

| 文件 | 改动点 |
|---|---|
| `api/graph.py` | L12 `NotFoundError`→`GraphNotFound`；L487/549/663/885 `Config.ZEP_API_KEY` 检查→`Config.GRAPHITI_LLM_API_KEY` 或删（`Config.validate()` 已覆盖）；L543-556 batch-resume 路径简化（UUID5 幂等让 resume=重跑）；L663/891 `GraphBuilderService(api_key=...)`→`GraphBuilderService()` |
| `api/report.py` | 无实质改动（引用 `ZepGraphMemoryManager`/`ZepToolsService` 类名保留） |
| `api/simulation.py` | 无实质改动（引用 `ZepEntityReader`/`ZepGraphMemoryManager` 类名保留） |
| `services/report_agent.py` | 无改动（引用 `ZepToolsService` 类名保留） |
| `services/simulation_runner.py` | 无改动（引用 `ZepGraphMemoryManager` 类名保留，仅 `ZEP_HTTP_REQUEST_TIMEOUT_SECONDS`/`ZEP_INGESTION_WAIT_TIMEOUT_SECONDS` 常量名改 `GRAPHITI_*`） |
| `services/simulation_manager.py` | 无改动（引用 `ZepEntityReader`/`OasisProfileGenerator` 类名保留） |
| `models/project.py` | 保留 `zep_batch_id`/`zep_batch_operation_id` 字段名（改名是 churn，加注释说明现装 Graphiti 的 operation_id surrogate） |
| `services/__init__.py` | 若类名未改则不动 |

### 配置与依赖

| 文件 | 改动 |
|---|---|
| `backend/requirements.txt` + `pyproject.toml` | 删 `zep-cloud==3.25.0`；加 `graphiti-core>=0.6.0,<0.7`、`neo4j>=5.26,<6`；保留 `httpx>=0.27.0` |
| `app/config.py` | 删 `ZEP_API_KEY`/`ZEP_API_URL` 拦截；加 `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD`、`GRAPHITI_LLM_API_KEY`（fallback `LLM_API_KEY`）/`GRAPHITI_LLM_BASE_URL`/`GRAPHITI_LLM_MODEL`/`GRAPHITI_EMBEDDER_MODEL`/`GRAPHITI_REQUEST_TIMEOUT_SECONDS=60`/`GRAPHITI_INGESTION_WAIT_TIMEOUT_SECONDS=600`。`Config.validate()` 改查 Neo4j + LLM key |
| `.env.example` | `ZEP_API_KEY` 块→Neo4j + Graphiti LLM 块（默认复用 `LLM_API_KEY`） |
| `docker-compose.yml` | 加 `neo4j:5.26` 服务（`NEO4J_AUTH=neo4j/mirofish`、`NEO4J_PLUGINS=["apoc"]`、7687/7474 端口、`neo4j_data` volume）；`mirofish` 服务 `depends_on: [neo4j]` |

### 测试

| 文件 | 处置 |
|---|---|
| `tests/test_zep_cloud_contracts.py` | → `test_graphiti_contracts.py`：注入 `neo4j.ServiceUnavailable`/`asyncio.TimeoutError` 测 `is_retryable_graphiti_error`；`AuthError` 不重试 |
| `tests/test_zep_retry_and_client.py` | → `test_graphiti_retry_and_client.py`：测 `get_graphiti_client` 单例 + `clear_graphiti_client_cache` |
| `tests/test_zep_edge_paging.py` | → `test_graphiti_paging.py`：mock `get_nodes`/`get_edges` 返回列表，测全量返回 + `max_items` 截断 |
| `tests/test_zep_graph_lifecycle.py` | 不变（纯 threading） |
| `tests/test_zep_graph_memory_updater.py` | mock `add_episode`（async），测 UUID5 确定性 + fail-closed |
| `tests/test_zep_entity_reader_edges.py` | mock `get_edges_for_node`，测双向边 |
| `tests/test_zep_simulation_barrier.py` / `test_zep_report_barrier.py` | 不变（测 barrier 逻辑，不测后端） |
| `tests/test_ontology_attributes.py` | import 换 Graphiti base；`safe_attr_name` 保留名前缀行为不变 |
| `tests/test_simulation_prepare_failure.py` | mock 目标换 `ZepEntityReader` 内部 |
| `tests/test_zep_cloud_validation_script.py` | → 指向新脚本 |
| `scripts/validate_zep_cloud_integration.py` | → `scripts/validate_graphiti_integration.py`：重写为对 live Neo4j 跑完整 adapter 链路 |

---

## 三、字段映射（Graphiti 对象 → MiroFish 既有 dict 形状）

调用方依赖的稳定 dict 契约必须不变：

**Node →** `{uuid, name, labels, summary, attributes}`
- `node.uuid`（Zep 是 `uuid_`，Graphiti 是 `uuid`——用 `getattr(n,'uuid_',None) or getattr(n,'uuid',None)` 兼容）
- `node.labels`、`node.summary`、`node.attributes`（dict）

**Edge →** `{uuid, name, fact, source_node_uuid, target_node_uuid, created_at, valid_at, invalid_at, expired_at}`
- `edge.uuid`、`edge.name`、`edge.fact`
- `edge.source_node_id`→`source_node_uuid`、`edge.target_node_id`→`target_node_uuid`
- temporal: `created_at`/`valid_at`/`invalid_at`/`expired_at` 直接透传

**Search 结果 →** `SearchResult.facts`（edge.fact + `[node.name]: node.summary`）、`.edges`、`.nodes`、`.total_count`。注意：**score 在任何模块都未被读取**，无需映射。

---

## 四、分阶段实施（7 个逻辑提交）

| 阶段 | 内容 | 文件 |
|---|---|---|
| **P1** | 依赖 + 配置 + docker | `pyproject.toml`、`requirements.txt`、`config.py`、`.env.example`、`docker-compose.yml` |
| **P2** | client/adapter 层（迁移核心） | 新建 `graphiti_runtime.py`、`graphiti_client.py`、`graphiti_paging.py`、`exceptions.py` |
| **P3** | graph_builder 重写 | `services/graph_builder.py`、`api/graph.py`、`services/ontology_generator.py` |
| **P4** | 检索服务重写 | `services/zep_tools.py`、`services/zep_entity_reader.py`、`services/oasis_profile_generator.py` |
| **P5** | memory updater 重写 | `services/zep_graph_memory_updater.py` |
| **P6** | 测试 | `tests/test_*.py`、`scripts/validate_graphiti_integration.py` |
| **P7** | 清理 | 删 `utils/zep.py`、`utils/zep_paging.py`；`grep zep_cloud` 归零；`uv.lock` 重生 |

---

## 五、4 个最高风险项

1. **本体 custom-types 注册（P3）**——Graphiti 的 `add_custom_types` 签名与 `EntityBase`/`EdgeBase` 基类随版本变化。Zep 的 `EntityModel`/`EntityText`/`EdgeModel` 稳定钉在 3.25.0。**缓解**：钉 `graphiti-core>=0.6.0,<0.7`，先写一个 2 实体 1 边的契约测试跑通再动真实服务。

2. **Batch 摄入重写（P3）**——原子性丢失：Zep batch 全有或全无，Graphiti episode 独立处理，中批失败留半成品图。**缓解**：UUID5 让全重跑安全；`api/graph.py` 已支持 `force=True` 重建；进度上报精度略降（advisory）。

3. **async-to-sync 桥死锁（P2）**——若 Graphiti 协程内部回调进试图用同一 loop 的 sync 代码则死锁。Neo4j driver + httpx LLM/embedder 都是 async-native 不回调 sync，理论安全。**缓解**：`run_async` 加 guard `if asyncio.get_event_loop() is _loop: raise RuntimeError("cannot call run_async from loop thread")`。

4. **`_wait_for_pending_episodes` 退化为 no-op（P5）**——Zep 的 `.processed` 轮询是 report barrier 的安全网。Graphiti `add_episode` 已 inline await 抽取，no-op 正确，但若未来版本改成 fire-and-forget 则 barrier 静默失效。**缓解**：no-op 内加防御性 `get_episodes(group_id, limit)` 计数校验，确认 episode 数 == `len(_pending_episode_uuids)`。

---

## 六、端到端验证

1. `docker compose up neo4j` → `bolt://localhost:7687` 可达，`neo4j/mirofish` 认证通过
2. `cd backend && uv sync` → 装 graphiti-core + neo4j，去 zep-cloud
3. `uv run pytest tests/ -v` → 全绿
4. `uv run python scripts/validate_graphiti_integration.py` → 对 live Neo4j 跑完整 adapter 链路
5. `uv run flask run` 启动
6. `POST /api/graph/ontology/generate`（样本文档 + 仿真需求）→ 返回本体
7. `POST /api/graph/build` → 轮询 `/api/graph/task/<id>` 至 `completed` → `node_count/edge_count > 0` → `GET /api/graph/data/<graph_id>` 返回节点/边
8. `POST /api/simulation/prepare` → `POST /api/simulation/start`（`enable_graph_memory_update=true`）→ 跑 5-10 轮 → `POST /api/simulation/stop` → `ZepGraphMemoryManager.get_updater()` 返回 None（drain 成功）
9. `POST /api/report/generate` → `insight_forge`/`panorama_search`/`quick_search` 返回非空 → `GET /api/report/<id>` 报告有内容
10. `POST /api/graph/project/<id>/reset` → Neo4j 该 `group_id` 子图被删（`delete_group`）

---

## 七、关键文件清单（实施时优先看）

- `backend/app/utils/graphiti_client.py`（新——替代 `zep.py` 的 adapter 核心）
- `backend/app/utils/graphiti_runtime.py`（新——async→sync loop 桥）
- `backend/app/services/graph_builder.py`（重写——本体 + 批摄入）
- `backend/app/services/zep_graph_memory_updater.py`（重写——`add_episode` + no-op drain）
- `backend/app/config.py`（Neo4j/Graphiti 配置替代 Zep）

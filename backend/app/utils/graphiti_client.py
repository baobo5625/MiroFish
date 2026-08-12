"""Shared Graphiti client, request limits, retry policy, and group helpers.

替代原先的 ``app/utils/zep.py``。对外暴露与 ``zep.py`` 同形的函数
（``get_*_client``、``normalize_*_search_*``、``is_retryable_*_error``、
``call_*_read_with_retry``、``clear_*_client_cache``），使得上层服务
只需改 import 与函数名即可，无需改控制流。

Graphiti 是 async-first；本模块通过 :mod:`graphiti_runtime` 把所有调用
桥接到共享 loop 线程。注意 Graphiti 没有 ``ZepApiError`` / ``Retry-After``
头这类 HTTP 语义，重试仅依据底层异常类型。
"""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Any, Callable, TypeVar

import httpx

from ..config import Config
from .graphiti_runtime import run_async
from .logger import get_logger

logger = get_logger("mirofish.graphiti")

T = TypeVar("T")

# ---------------------------------------------------------------------------
# 超时与限额（与原 zep.py 对齐，保持上层不变）
# ---------------------------------------------------------------------------
# 单次图查询的等待上限，替代 ZEP_HTTP_REQUEST_TIMEOUT_SECONDS。
GRAPHITI_REQUEST_TIMEOUT_SECONDS = float(
    getattr(Config, "GRAPHITI_REQUEST_TIMEOUT_SECONDS", 60.0)
)
# 整批 episode 摄入的等待上限，替代 ZEP_INGESTION_WAIT_TIMEOUT_SECONDS。
GRAPHITI_INGESTION_WAIT_TIMEOUT_SECONDS = int(
    getattr(Config, "GRAPHITI_INGESTION_WAIT_TIMEOUT_SECONDS", 600)
)
# Graphiti 对 query 长度无硬限制，保留 MiroFish 既有护栏。
MAX_GRAPHITI_SEARCH_QUERY_CHARS = 400
MAX_GRAPHITI_SEARCH_RESULTS = 50

# Neo4j driver 在某些瞬时错误下抛 ClientError，其 code 形如 "Neo.ClientError..."
# 这里只把可重试的瞬时码列出（ServiceUnavailable / DatabaseUnavailable /
# TransientError），其余 ClientError（如 AuthError / 语法错）不重试。
_RETRYABLE_NEO4J_CODE_PREFIXES = (
    "Neo.TransientError",
    "Neo.ClientError.Database.DatabaseUnavailable",
)


# ---------------------------------------------------------------------------
# 搜索输入归一化
# ---------------------------------------------------------------------------
def normalize_graphiti_search_query(query: Any) -> str:
    """Return a non-empty query within MiroFish's guard limit."""

    if not isinstance(query, str):
        raise ValueError("Graphiti search query must be a string")
    normalized = query.strip()
    if not normalized:
        raise ValueError("Graphiti search query must not be empty")
    return normalized[:MAX_GRAPHITI_SEARCH_QUERY_CHARS]


def normalize_graphiti_search_limit(limit: Any) -> int:
    """Clamp a search result limit to MiroFish's guard range."""

    try:
        normalized = int(limit)
    except (TypeError, ValueError) as exc:
        raise ValueError("Graphiti search limit must be an integer") from exc
    if normalized < 1:
        raise ValueError("Graphiti search limit must be at least 1")
    return min(normalized, MAX_GRAPHITI_SEARCH_RESULTS)


# ---------------------------------------------------------------------------
# 客户端工厂
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _cached_graphiti_client(
    uri: str, user: str, password: str, llm_api_key: str, llm_base_url: str,
    llm_model: str, embedder_model: str,
) -> Any:
    """构造并缓存 Graphiti 实例。

    Graphiti 的 Neo4jDriver 在 ``__init__`` 中会自动后台触发
    ``build_indices_and_constraints``，故无需手动调用 schema 构建方法。
    """
    # 延迟导入，避免在未安装 graphiti-core 的环境（如纯测试）下 import 失败。
    from graphiti_core import Graphiti
    from graphiti_core.driver.neo4j_driver import Neo4jDriver
    from graphiti_core.embedder import OpenAIEmbedder, OpenAIEmbedderConfig
    from graphiti_core.cross_encoder.openai_reranker_client import (
        OpenAIRerankerClient,
    )
    from graphiti_core.llm_client import LLMConfig, OpenAIClient

    driver = Neo4jDriver(uri=uri, user=user, password=password)
    # OpenAI SDK 的 responses/embeddings 端点要求 base_url 含 /v1，否则会拼成
    # /responses（漏掉 /v1）导致自建网关 404。若用户传的 base_url 不以 /v1
    # 结尾（如红杉网关 https://host），自动补 /v1。
    normalized_base_url = llm_base_url.rstrip("/")
    if not normalized_base_url.endswith("/v1"):
        normalized_base_url = normalized_base_url + "/v1"
    # small_model 默认是 gpt-4.1-nano，自建网关上没有该模型会 404。
    # 复用主模型作为 small_model（graphiti 用 small_model 做轻量抽取步骤）。
    llm_config = LLMConfig(
        api_key=llm_api_key,
        model=llm_model,
        small_model=llm_model,
        base_url=normalized_base_url,
    )
    # 用 OpenAIClient（走 OpenAI Responses API / responses.parse）。
    # 自建网关对 responses.parse 的 structured output 支持可能不完整：
    # output_text 会把 JSON Schema（input_schema 字段）与数据混在一起返回，
    # 甚至有时只返回 schema 不返回数据。靠 _install_responses_cleanup_patch
    # 清洗噪声 + 增强 MAX_RETRIES 重试处理。
    llm_client = OpenAIClient(config=llm_config)
    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=llm_api_key,
            base_url=normalized_base_url,
            embedding_model=embedder_model,
        )
    )
    # Graphiti 默认会构造 OpenAIRerankerClient（cross-encoder reranker），
    # 它读 OPENAI_API_KEY 环境变量。复用 MiroFish 的 LLM 配置构造它，
    # 避免 reranker 在无 OPENAI_API_KEY 环境时初始化失败。
    cross_encoder = OpenAIRerankerClient(config=llm_config, client=llm_client)
    graphiti = Graphiti(
        uri=uri,
        user=user,
        password=password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
        graph_driver=driver,
    )
    _install_responses_cleanup_patch()
    return graphiti


def _install_responses_cleanup_patch() -> None:
    """网关 responses.parse 兼容：优先原生 parse，失败回退 create+清洗。

    策略：优先用 openai SDK 原生 ``responses.parse``（``text_format``，SDK
    内部做 pydantic 验证）——网关支持良好时（如红杉 dev 网关 5/5 成功）这是
    最干净的路径。

    当 ``responses.parse`` 抛 ValidationError（网关返回的 output_text 混入
    ``input_schema`` 噪声或只返回 schema 不返回数据）时，回退到
    ``responses.create`` + ``text.format=json_schema``（不走 SDK post_parser，
    保留 response 对象），自己 ``json.loads(output_text)`` 并剔除噪声键，
    再做数据完整性校验，缺字段时抛异常触发 graphiti 重试。

    MAX_RETRIES 从默认 2 提到 6，增加撞上完整响应的概率。
    """
    try:
        from graphiti_core.llm_client.openai_client import OpenAIClient
    except Exception:  # pragma: no cover
        return

    if getattr(OpenAIClient, "_mirofish_cleanup_patched", False):
        return

    original_create = OpenAIClient._create_structured_completion

    async def _patched_create_structured_completion(
        self, model, messages, temperature, max_tokens, response_model,
        reasoning=None, verbosity=None,
    ):
        # 优先走原生 responses.parse（网关支持好时最干净）
        try:
            return await original_create(
                self, model, messages, temperature, max_tokens, response_model,
                reasoning, verbosity,
            )
        except Exception as parse_error:
            # 非 ValidationError（如网络错误）直接抛
            if "ValidationError" not in type(parse_error).__name__:
                raise

            # 回退：responses.create + text.format=json_schema + 手动清洗
            import json as _json
            from types import SimpleNamespace

            is_reasoning_model = (
                model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")
            )
            try:
                schema = response_model.model_json_schema()
            except Exception:
                schema = {"type": "object"}

            request_kwargs = {
                "model": model,
                "input": messages,
                "max_output_tokens": max_tokens,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": getattr(response_model, "__name__", "Response"),
                            "schema": schema,
                            "strict": False,
                        },
                    }
                },
            }
            if temperature is not None and not is_reasoning_model:
                request_kwargs["temperature"] = temperature

            raw = await self.client.responses.create(**request_kwargs)
            text = getattr(raw, "output_text", None) or ""

            # 清洗网关噪声：递归剔除 schema 字段
            def _deep_clean(obj):
                if isinstance(obj, dict):
                    # 剔除 schema 噪声键
                    for noise_key in ("input_schema", "schema", "json_schema"):
                        obj.pop(noise_key, None)
                    # 剔除值为 schema 描述的噪声字段（含 additionalProperties/type:object/properties）
                    noise_value_keys = []
                    for k, v in obj.items():
                        if isinstance(v, dict) and _is_schema_dict(v):
                            noise_value_keys.append(k)
                        else:
                            _deep_clean(v)
                    for k in noise_value_keys:
                        obj.pop(k, None)
                elif isinstance(obj, list):
                    for item in obj:
                        _deep_clean(item)
                return obj

            def _is_schema_dict(d):
                """判断 dict 是否是 JSON Schema 描述（而非数据）。"""
                schema_markers = {"additionalProperties", "properties", "$defs", "$ref", "allOf", "anyOf", "oneOf"}
                return any(m in d for m in schema_markers) and d.get("type") in ("object", "array", "string", "integer", "boolean", "number")

            cleaned = None
            if text:
                try:
                    data = _json.loads(text)
                    cleaned = _deep_clean(data)
                except _json.JSONDecodeError:
                    pass

            cleaned_text = _json.dumps(cleaned) if cleaned is not None else ""

            # 数据完整性校验：缺字段时抛异常触发 graphiti 重试
            if cleaned is not None:
                try:
                    response_model.model_validate(cleaned)
                except Exception as ve:
                    raise Exception(
                        f"网关返回数据不完整（清洗后缺字段），触发重试: {ve}"
                    ) from parse_error

            usage = getattr(raw, "usage", None)
            return SimpleNamespace(
                output_text=cleaned_text,
                usage=usage,
                refusal=getattr(raw, "refusal", None),
            )

    OpenAIClient._create_structured_completion = _patched_create_structured_completion
    OpenAIClient._mirofish_cleanup_patched = True

    # 增强 LLM 抽取重试：网关偶发返回坏格式（清洗后仍可能缺数据字段），
    # graphiti 默认 MAX_RETRIES=2 不够。提到 6（单次成功率 ~80% 时，
    # 6 次重试后失败率约 0.0064%）。
    try:
        from graphiti_core.llm_client.openai_base_client import BaseOpenAIClient
        BaseOpenAIClient.MAX_RETRIES = 6
    except Exception:  # pragma: no cover
        pass


def get_graphiti_client(api_key: str | None = None, timeout: float | None = None) -> Any:
    """返回进程共享的 Graphiti 客户端。

    ``api_key`` 参数仅为签名兼容保留（原 ``get_zep_client`` 接受它），
    Graphiti 的认证经由 Neo4j 凭据 + LLM key，不按调用键控。
    """

    uri = Config.NEO4J_URI
    user = Config.NEO4J_USER
    password = Config.NEO4J_PASSWORD
    llm_api_key = (api_key or Config.GRAPHITI_LLM_API_KEY or "").strip()
    if not llm_api_key:
        raise ValueError("GRAPHITI_LLM_API_KEY 未配置（可复用 LLM_API_KEY）")
    if not uri:
        raise ValueError("NEO4J_URI 未配置")

    return _cached_graphiti_client(
        uri, user, password,
        llm_api_key, Config.GRAPHITI_LLM_BASE_URL,
        Config.GRAPHITI_LLM_MODEL, Config.GRAPHITI_EMBEDDER_MODEL,
    )


def clear_graphiti_client_cache() -> None:
    """清缓存（测试与受控重配用）。"""

    _cached_graphiti_client.cache_clear()


# ---------------------------------------------------------------------------
# 重试策略
# ---------------------------------------------------------------------------
def is_retryable_graphiti_error(error: BaseException) -> bool:
    """Return whether a failed *read* is safe and useful to retry.

    与原 ``is_retryable_zep_error`` 对齐：传输/超时/瞬时 Neo4j 错误重试，
    认证、未找到、4xx 语义错误一律不重试。
    """

    # 网络传输层：httpx（LLM/embedder 走 httpx async）+ 系统级。
    # Python 3.11+ 中 asyncio.TimeoutError 即 TimeoutError，故此处统一覆盖。
    if isinstance(error, (httpx.TimeoutException, httpx.TransportError)):
        return True
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True

    # Neo4j driver 异常
    try:
        from neo4j.exceptions import (
            ServiceUnavailable,
            DatabaseUnavailable,
            AuthError,
            ClientError,
        )
    except Exception:  # pragma: no cover - neo4j 必装，仅防御
        ClientError = AuthError = ServiceUnavailable = DatabaseUnavailable = ()  # type: ignore[assignment]

    if isinstance(error, (ServiceUnavailable, DatabaseUnavailable)):
        return True
    if isinstance(error, AuthError):
        return False
    if isinstance(error, ClientError):
        code = getattr(error, "code", None) or ""
        return isinstance(code, str) and code.startswith(_RETRYABLE_NEO4J_CODE_PREFIXES)
    return False


def call_graphiti_read_with_retry(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_attempts: int = 3,
    initial_delay: float = 2.0,
    max_delay: float = 60.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Retry a safe Graphiti read only for transport / transient errors.

    与原 ``call_zep_read_with_retry`` 同形：纯指数退避，无 Retry-After
    （Graphiti 非 HTTP，无该头）。``operation`` 应是同步可调用对象，内部
    自行用 :func:`run_async` 桥接协程。
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if attempt == max_attempts or not is_retryable_graphiti_error(error):
                raise

            delay = min(initial_delay * (2 ** (attempt - 1)), max_delay)
            logger.warning(
                "Graphiti %s attempt %s/%s failed (%s); retrying in %.1fs",
                operation_name, attempt, max_attempts,
                type(error).__name__, delay,
            )
            sleep(delay)

    raise AssertionError("unreachable")


# ---------------------------------------------------------------------------
# group 删除 helper（Graphiti 无 delete_group，需逐 namespace 删）
# ---------------------------------------------------------------------------
def delete_group(client: Any, group_id: str, *, timeout: float | None = None) -> None:
    """删除一个 group_id 名下的全部图数据。

    Graphiti 自带的 ``delete_by_group_id`` 用 ``driver.session()`` 执行
    ``CALL {} IN TRANSACTIONS``，但 neo4j 6.x 驱动要求该语法在 implicit
    transaction 里运行，explicit session 会报 ``TransactionStartFailed``。
    故这里绕过 graphiti 的方法，直接用 ``driver.execute_query``（implicit
    transaction）执行简单 ``DETACH DELETE``。

    删除顺序：先删边（RELATES_TO），再删节点（Entity/Episodic/Community/Saga）。
    """
    from .graphiti_runtime import run_async as _run

    deadline = timeout if timeout is not None else max(
        GRAPHITI_REQUEST_TIMEOUT_SECONDS * 2, 120.0
    )

    driver = getattr(client, "driver", None)
    if driver is None:
        raise RuntimeError("Graphiti 客户端缺少 driver")

    async def _do_delete():
        # 先删边（RELATES_TO 关系），再删节点（DETACH DELETE 也能删边，
        # 但先删边可避免大图删除时的事务冲突）。
        await driver.execute_query(
            "MATCH ()-[r:RELATES_TO {group_id: $group_id}]->() DELETE r",
            group_id=group_id,
        )
        await driver.execute_query(
            "MATCH (n) WHERE n.group_id = $group_id DETACH DELETE n",
            group_id=group_id,
        )

    _run(_do_delete(), timeout=deadline)


async def _call_delete_by_group_id(deleter: Callable, group_id: str) -> None:
    """``delete_by_group_id`` 既可能是 async 方法也可能是普通方法，统一适配。"""

    import inspect

    result = deleter(group_id)
    if inspect.isawaitable(result):
        await result

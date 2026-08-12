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
    from graphiti_core.llm_client import LLMConfig
    from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
    from openai import AsyncOpenAI

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
    # 用 AzureOpenAILLMClient 而非 OpenAIClient：
    # OpenAIClient 一律用 responses.parse（OpenAI Responses API），但红杉网关
    # 对 responses.parse 的 structured output 支持有缺陷（output_text 混入
    # input_schema 噪声）。AzureOpenAILLMClient 对非 reasoning 模型
    # （deepseek-v4-flash 不是 reasoning 模型）走 beta.chat.completions.parse
    # （/v1/chat/completions 的 response_format），红杉网关完全支持。
    async_openai = AsyncOpenAI(api_key=llm_api_key, base_url=normalized_base_url)
    llm_client = AzureOpenAILLMClient(azure_client=async_openai, config=llm_config)
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
    _install_azure_tuple_patch()
    return graphiti


def _install_azure_tuple_patch() -> None:
    """修正 AzureOpenAILLMClient._handle_structured_response：清洗网关噪声 + 补三元组。

    解决两个问题：
    1. **input_schema 噪声**：红杉网关的 structured output 响应会把 JSON Schema
       （``input_schema`` 字段）与数据混在同一对象里返回，甚至有时只返回 schema
       不返回数据。原实现 ``json.loads(output_text)`` 后直接返回，graphiti 上层
       ``ExtractedEdges(**dict)`` 因缺 ``edges`` 字段而 ValidationError。这里在
       解析后剔除 ``input_schema``/``schema``/``json_schema`` 等噪声键。
    2. **三元组返回**：graphiti ``_generate_response`` 期望
       ``_handle_structured_response`` 返回 ``(dict, input_tokens, output_tokens)``，
       但 AzureOpenAILLMClient 只返回 ``dict``，导致 ``expected 3, got 1``。
    """
    try:
        from graphiti_core.llm_client.azure_openai_client import AzureOpenAILLMClient
    except Exception:  # pragma: no cover
        return

    if getattr(AzureOpenAILLMClient, "_mirofish_tuple_patched", False):
        return

    original = AzureOpenAILLMClient._handle_structured_response

    def _clean_data(data):
        """剔除网关混入的 schema 噪声字段，只留数据。"""
        if isinstance(data, dict):
            for noise_key in ("input_schema", "schema", "json_schema"):
                data.pop(noise_key, None)
        return data

    def _usage_tokens(response):
        usage = getattr(response, "usage", None)
        if not usage:
            return 0, 0
        i = getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0
        o = getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0) or 0
        return i, o

    def _patched_handle(self, response):
        import json as _json

        # 路径 B：chat.completions.parse 返回 ParsedChatCompletion
        if hasattr(response, "choices") and response.choices:
            msg = response.choices[0].message
            # 优先用已解析的 pydantic 对象
            parsed = getattr(msg, "parsed", None)
            if parsed is not None:
                data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
                i, o = _usage_tokens(response)
                return _clean_data(data), i, o
            # 回退：从 content 文本解析并清洗（网关噪声最可能出现在这里）
            content = getattr(msg, "content", None)
            if content:
                try:
                    data = _json.loads(content)
                    i, o = _usage_tokens(response)
                    return _clean_data(data), i, o
                except _json.JSONDecodeError:
                    pass
            if getattr(msg, "refusal", None):
                from graphiti_core.llm_client.errors import RefusalError
                raise RefusalError(msg.refusal)
            raise Exception(f"Invalid response from LLM: {response.model_dump()}")

        # 路径 A：responses.parse 返回 ParsedResponse（用 output_text）
        if hasattr(response, "output_text"):
            text = response.output_text
            if text:
                try:
                    data = _json.loads(text)
                    i, o = _usage_tokens(response)
                    return _clean_data(data), i, o
                except _json.JSONDecodeError:
                    # 回退到原实现（纯文本响应等）
                    return original(self, response)
            if getattr(response, "refusal", None):
                from graphiti_core.llm_client.errors import RefusalError
                raise RefusalError(response.refusal)

        # 其他情况回退原实现
        return original(self, response)

    AzureOpenAILLMClient._handle_structured_response = _patched_handle
    AzureOpenAILLMClient._mirofish_tuple_patched = True

    # 增强 LLM 抽取重试：网关偶发返回坏格式（清洗后仍可能缺数据字段），
    # graphiti 默认 MAX_RETRIES=2 偶尔不够。提到 4，配合清洗让成功率从 ~80% 升至 ~99%。
    try:
        AzureOpenAILLMClient.MAX_RETRIES = 4
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

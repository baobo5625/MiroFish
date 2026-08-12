"""Graphiti 客户端与重试策略测试（替代 test_zep_retry_and_client.py）。

Graphiti 无 ``ZepApiError`` / ``Retry-After`` 头语义；重试仅依据底层异常类型
（asyncio.TimeoutError、neo4j.ServiceUnavailable、OSError 等可重试；
AuthError、4xx 语义错误不重试）。本测试用真实异常类型 + monkeypatch
neo4j 异常类来驱动 ``is_retryable_graphiti_error`` 与
``call_graphiti_read_with_retry``。
"""

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.utils import graphiti_client as graphiti_client_module
from app.utils.graphiti_client import (
    GRAPHITI_INGESTION_WAIT_TIMEOUT_SECONDS,
    GRAPHITI_REQUEST_TIMEOUT_SECONDS,
    call_graphiti_read_with_retry,
    is_retryable_graphiti_error,
)


def test_permanent_errors_fail_without_retry():
    """非可重试异常（如 ValueError）只调用一次。"""
    calls = []

    def operation():
        calls.append(True)
        raise ValueError("bad query")

    with pytest.raises(ValueError):
        call_graphiti_read_with_retry(
            operation,
            operation_name="permanent failure",
            sleep=lambda _seconds: None,
        )

    assert len(calls) == 1


def test_transient_error_is_retried_with_backoff():
    """TimeoutError 可重试，第二次成功返回。"""
    calls = []
    sleeps = []

    def operation():
        calls.append(True)
        if len(calls) == 1:
            raise TimeoutError("transient")
        return "ok"

    result = call_graphiti_read_with_retry(
        operation,
        operation_name="transient read",
        sleep=sleeps.append,
    )

    assert result == "ok"
    assert len(calls) == 2
    # 第一次退避 = initial_delay * 2^0 = 2.0
    assert sleeps == [2.0]


def test_os_error_is_retryable():
    assert is_retryable_graphiti_error(OSError("connection reset")) is True
    assert is_retryable_graphiti_error(TimeoutError()) is True


def test_value_error_is_not_retryable():
    assert is_retryable_graphiti_error(ValueError("bad input")) is False
    assert is_retryable_graphiti_error(KeyError("missing")) is False


def test_max_attempts_exhausted_reraises():
    calls = []

    def operation():
        calls.append(True)
        raise TimeoutError("always fails")

    with pytest.raises(TimeoutError):
        call_graphiti_read_with_retry(
            operation,
            operation_name="always fails",
            max_attempts=3,
            sleep=lambda _seconds: None,
        )

    assert len(calls) == 3


def test_neo4j_service_unavailable_is_retryable(monkeypatch):
    """模拟 neo4j.exceptions.ServiceUnavailable 可重试。"""
    # 注入一个假的 neo4j 异常类到模块的 is_retryable 函数可见范围
    import types as _types

    fake_neo4j_exc = _types.SimpleNamespace(
        ServiceUnavailable=type("ServiceUnavailable", (Exception,), {}),
        DatabaseUnavailable=type("DatabaseUnavailable", (Exception,), {}),
        AuthError=type("AuthError", (Exception,), {}),
        ClientError=type("ClientError", (Exception,), {}),
    )
    import sys

    monkeypatch.setitem(sys.modules, "neo4j.exceptions", fake_neo4j_exc)

    assert is_retryable_graphiti_error(fake_neo4j_exc.ServiceUnavailable()) is True
    assert is_retryable_graphiti_error(fake_neo4j_exc.DatabaseUnavailable()) is True
    assert is_retryable_graphiti_error(fake_neo4j_exc.AuthError()) is False


def test_client_factory_is_cached(monkeypatch):
    """get_graphiti_client 返回进程共享单例。"""
    created = []

    def fake_cached_client(*args, **kwargs):
        created.append((args, kwargs))
        return SimpleNamespace(built=True)

    monkeypatch.setattr(
        graphiti_client_module, "_cached_graphiti_client",
        lambda *a, **k: fake_cached_client(a, **k),
    )
    # 模拟 lru_cache：用真实 lru_cache 包装
    from functools import lru_cache

    cached = lru_cache(maxsize=4)(fake_cached_client)
    monkeypatch.setattr(graphiti_client_module, "_cached_graphiti_client", cached)
    cached.cache_clear()

    monkeypatch.setattr(
        graphiti_client_module, "Config",
        SimpleNamespace(
            NEO4J_URI="bolt://localhost:7687",
            NEO4J_USER="neo4j",
            NEO4J_PASSWORD="pw",
            GRAPHITI_LLM_API_KEY="test-key",
            GRAPHITI_LLM_BASE_URL="https://api.openai.com/v1",
            GRAPHITI_LLM_MODEL="gpt-4o-mini",
            GRAPHITI_EMBEDDER_MODEL="text-embedding-3-small",
        ),
    )

    first = graphiti_client_module.get_graphiti_client()
    second = graphiti_client_module.get_graphiti_client()
    assert first is second
    cached.cache_clear()


def test_client_factory_requires_api_key(monkeypatch):
    # api_key 校验在 URI 读取之后，但 LLM key 校验在前；给空 key 应在
    # NEO4J_USER 访问前就抛错（api_key 检查在函数最前）。
    monkeypatch.setattr(
        graphiti_client_module, "Config",
        SimpleNamespace(
            NEO4J_URI="bolt://localhost:7687",
            NEO4J_USER="neo4j",
            NEO4J_PASSWORD="pw",
            GRAPHITI_LLM_API_KEY=None,
            GRAPHITI_LLM_BASE_URL="https://api.openai.com/v1",
            GRAPHITI_LLM_MODEL="gpt-4o-mini",
            GRAPHITI_EMBEDDER_MODEL="text-embedding-3-small",
        ),
    )
    with pytest.raises(ValueError, match="GRAPHITI_LLM_API_KEY"):
        graphiti_client_module.get_graphiti_client()


def test_client_factory_requires_neo4j_uri(monkeypatch):
    # URI 为空时应抛错（但 api_key 先校验，故需给 key）
    monkeypatch.setattr(
        graphiti_client_module, "Config",
        SimpleNamespace(
            NEO4J_URI="",
            NEO4J_USER="neo4j",
            NEO4J_PASSWORD="pw",
            GRAPHITI_LLM_API_KEY="test-key",
            GRAPHITI_LLM_BASE_URL="https://api.openai.com/v1",
            GRAPHITI_LLM_MODEL="gpt-4o-mini",
            GRAPHITI_EMBEDDER_MODEL="text-embedding-3-small",
        ),
    )
    with pytest.raises(ValueError, match="NEO4J_URI"):
        graphiti_client_module.get_graphiti_client()


def test_timeout_constants_preserve_original_policy():
    assert GRAPHITI_REQUEST_TIMEOUT_SECONDS == 60.0
    assert GRAPHITI_INGESTION_WAIT_TIMEOUT_SECONDS == 600


def test_timeout_policy_is_not_exposed_in_env_example():
    env_example = Path(__file__).resolve().parents[2] / ".env.example"
    contents = env_example.read_text(encoding="utf-8")

    # Zep 专有的超时 env 项不应再出现
    assert "ZEP_REQUEST_TIMEOUT_SECONDS" not in contents
    assert "ZEP_INGESTION_TIMEOUT_SECONDS" not in contents
    assert "ZEP_API_KEY" not in contents

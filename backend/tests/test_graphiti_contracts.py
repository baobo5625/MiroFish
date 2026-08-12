"""Graphiti 适配层契约测试（替代 test_zep_cloud_contracts.py）。

覆盖：
* 搜索 query 截断（仍保留 MiroFish 的 400 字符护栏）
* 实体上下文双向边检索（Graphiti get_by_node_uuid 修复了 Zep 单向缺陷）
* 错误传播：auth/transport 失败不被吞成空结果
* add_episode 摄入：UUID5 幂等、episode_body/source_description/group_id 正确
* _wait_for_batch 退化为数量校验

原 Zep Batch API / SDK 序列化 / _wait_for_episodes 轮询测试已删除
（Graphiti 无 Batch API，add_episode 已 inline 处理）。
"""

import asyncio
from types import SimpleNamespace

import pytest

from app.services import graph_builder as graph_builder_module
from app.services.graph_builder import BatchSubmission, GraphBuilderService
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.zep_entity_reader import EntityNode, ZepEntityReader
from app.services.zep_tools import ZepToolsService


def _sync_run_async(coro, timeout):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ---------------------------------------------------------------------------
# 搜索 query 截断
# ---------------------------------------------------------------------------
def test_report_search_caps_the_query(monkeypatch):
    import app.services.zep_tools as tools_module

    calls = []

    async def fake_search_(query, config=None, group_ids=None, **kwargs):
        calls.append(query)
        return SimpleNamespace(edges=[], nodes=[])

    service = object.__new__(ZepToolsService)
    service.client = SimpleNamespace(search_=fake_search_)
    service.MAX_RETRIES = 3
    service.RETRY_DELAY = 0
    monkeypatch.setattr(tools_module, "run_async", _sync_run_async)

    original_query = "q" * 401
    result = service.search_graph("graph-id", original_query)

    assert calls[0] == original_query[:400]
    assert result.query == original_query


def test_profile_context_search_caps_both_queries(monkeypatch):
    import app.services.oasis_profile_generator as gen_module

    calls = []

    async def fake_search_(query, config=None, group_ids=None, **kwargs):
        calls.append(query)
        return SimpleNamespace(edges=[], nodes=[])

    generator = object.__new__(OasisProfileGenerator)
    generator.zep_client = SimpleNamespace(search_=fake_search_)
    generator.graph_id = "graph-id"
    monkeypatch.setattr(gen_module, "run_async", _sync_run_async)

    entity = EntityNode(
        uuid="node-id",
        name="n" * 500,
        labels=["Entity", "Person"],
        summary="",
        attributes={},
    )
    generator._search_zep_for_entity(entity)

    assert len(calls) == 2
    assert all(0 < len(call) <= 400 for call in calls)


# ---------------------------------------------------------------------------
# 实体上下文双向边
# ---------------------------------------------------------------------------
def test_entity_context_includes_incoming_edges_from_the_full_graph(monkeypatch):
    incoming = {
        "uuid": "edge-in",
        "name": "WORKS_AT",
        "fact": "Alice works at Acme",
        "source_node_uuid": "alice",
        "target_node_uuid": "acme",
        "attributes": {},
    }
    outgoing = {
        "uuid": "edge-out",
        "name": "BUILDS",
        "fact": "Acme builds Product",
        "source_node_uuid": "acme",
        "target_node_uuid": "product",
        "attributes": {},
    }
    unrelated = {
        "uuid": "edge-unrelated",
        "name": "LOCATED_IN",
        "fact": "OtherCo is located in Paris",
        "source_node_uuid": "other-company",
        "target_node_uuid": "paris",
        "attributes": {},
    }

    # mock get_by_uuid 返回 acme 节点本身
    acme_node = SimpleNamespace(
        uuid="acme", name="Acme", labels=["Company"], summary="", attributes={},
    )

    async def fake_get_by_uuid(uuid):
        return acme_node

    reader = object.__new__(ZepEntityReader)
    reader.client = SimpleNamespace(
        nodes=SimpleNamespace(
            entity=SimpleNamespace(get_by_uuid=fake_get_by_uuid)
        )
    )
    reader._call_with_retry = lambda func, **kw: func()
    reader.get_all_edges = lambda _graph_id: [incoming, outgoing, unrelated]
    reader.get_all_nodes = lambda _graph_id: [
        {"uuid": "alice", "name": "Alice", "labels": ["Person"], "summary": ""},
        {"uuid": "acme", "name": "Acme", "labels": ["Company"], "summary": ""},
        {"uuid": "product", "name": "Product", "labels": ["Product"], "summary": ""},
    ]
    # mock get_node_edges 走 graph_id 路径（全图过滤，双向）
    reader.get_node_edges = lambda entity_uuid, **kw: [
        e for e in reader.get_all_edges(kw.get("graph_id"))
        if e["source_node_uuid"] == entity_uuid or e["target_node_uuid"] == entity_uuid
    ]

    entity = reader.get_entity_with_context("graph-id", "acme")

    assert entity is not None
    assert len(entity.related_edges) == 2
    assert {edge["edge_name"] for edge in entity.related_edges} == {
        "WORKS_AT",
        "BUILDS",
    }
    assert {edge["direction"] for edge in entity.related_edges} == {
        "incoming",
        "outgoing",
    }
    assert {node["name"] for node in entity.related_nodes} == {"Alice", "Product"}


# ---------------------------------------------------------------------------
# 错误传播
# ---------------------------------------------------------------------------
def test_entity_reader_does_not_turn_auth_failure_into_missing_entity(monkeypatch):
    import app.services.zep_entity_reader as reader_module

    def unauthorized(**_kwargs):
        raise PermissionError("unauthorized")

    reader = object.__new__(ZepEntityReader)
    reader._call_with_retry = lambda func, **kw: func()
    monkeypatch.setattr(reader_module, "run_async", _sync_run_async)

    async def fake_get_by_uuid(uuid):
        raise PermissionError("unauthorized")

    reader.client = SimpleNamespace(
        nodes=SimpleNamespace(entity=SimpleNamespace(get_by_uuid=fake_get_by_uuid))
    )

    with pytest.raises(PermissionError):
        reader.get_entity_with_context("graph-id", "node-id")


def test_report_tools_do_not_turn_read_failures_into_empty_data(monkeypatch):
    """get_node_detail 的非 GraphNotFound 错误应上抛，不返回 None。"""
    import app.services.zep_tools as tools_module

    def unauthorized(**_kwargs):
        raise PermissionError("unauthorized")

    service = object.__new__(ZepToolsService)
    service.MAX_RETRIES = 3
    service.RETRY_DELAY = 0
    service._call_with_retry = lambda func, **kw: func()
    monkeypatch.setattr(tools_module, "run_async", _sync_run_async)

    async def fake_get_by_uuid(uuid):
        raise PermissionError("unauthorized")

    service.client = SimpleNamespace(
        nodes=SimpleNamespace(entity=SimpleNamespace(get_by_uuid=fake_get_by_uuid))
    )

    with pytest.raises(PermissionError):
        service.get_node_detail("node-id")


# ---------------------------------------------------------------------------
# add_episode 摄入
# ---------------------------------------------------------------------------
def test_document_ingestion_calls_add_episode_with_correct_params(monkeypatch):
    import app.services.graph_builder as gb_module

    calls = []

    async def fake_add_episode(**kwargs):
        calls.append(kwargs)
        # 返回 graphiti 自动生成的 UUID
        return SimpleNamespace(episode=SimpleNamespace(uuid=f"auto-{len(calls)}"))

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(add_episode=fake_add_episode)
    builder.EPISODE_CONCURRENCY = 8
    monkeypatch.setattr(gb_module, "run_async", _sync_run_async)
    persisted = []

    submission = builder.add_text_batches(
        "graph-id",
        ["chunk one", "chunk two"],
        batch_created_callback=lambda batch_id, operation_id: persisted.append(
            (batch_id, operation_id)
        ),
    )

    assert submission.item_count == 2
    assert len(submission.operation_id) == 64
    # 两个 episode 都被摄入
    assert len(calls) == 2
    assert persisted == [(None, submission.operation_id)]
    # 每个 call 都带 group_id、source_description、EpisodeType.text
    for call in calls:
        assert call["group_id"] == "graph-id"
        assert call["source_description"] == "MiroFish source document chunk"
        assert call["source"].value == "text" or str(call["source"]) == "EpisodeType.TEXT" or call["source"].value == "text"
    # 不传 uuid（Graphiti 的 uuid 是 get_or_create 语义，应让 graphiti 自动生成）
    for call in calls:
        assert "uuid" not in call or call["uuid"] is None


def test_add_episode_does_not_pass_uuid_to_graphiti(monkeypatch):
    """add_episode 不传 uuid（Graphiti 的 uuid 参数是 get_or_create 语义，
    传不存在的 UUID 会抛 NodeNotFoundError）。UUID 由 graphiti 自动生成，
    幂等性靠重建前 delete_graph 清空保证。"""
    import app.services.graph_builder as gb_module

    captured = []

    async def fake_add_episode(**kwargs):
        captured.append(kwargs)
        # 返回 graphiti 自动生成的 UUID
        return SimpleNamespace(episode=SimpleNamespace(uuid=f"auto-{len(captured)}"))

    builder = object.__new__(GraphBuilderService)
    builder.client = SimpleNamespace(add_episode=fake_add_episode)
    builder.EPISODE_CONCURRENCY = 8
    monkeypatch.setattr(gb_module, "run_async", _sync_run_async)

    submission = builder.add_text_batches("graph-id", ["chunk one", "chunk two"])

    assert submission.item_count == 2
    assert len(captured) == 2
    # 关键：不应传 uuid 参数（graphiti 会误读为 get existing）
    for call in captured:
        assert "uuid" not in call or call["uuid"] is None


def test_wait_for_batch_validates_episode_count():
    builder = object.__new__(GraphBuilderService)
    submission = BatchSubmission("op-1", "op-1", ["ep-1", "ep-2"], 2)

    assert builder._wait_for_batch(submission) == ["ep-1", "ep-2"]


def test_wait_for_batch_rejects_count_mismatch():
    builder = object.__new__(GraphBuilderService)
    submission = BatchSubmission("op-1", "op-1", ["ep-1"], 2)  # 1 != 2

    with pytest.raises(RuntimeError, match="episodes"):
        builder._wait_for_batch(submission)


def test_create_graph_journals_id_and_does_not_call_sdk():
    """Graphiti 无显式建图，create_graph 只生成 ID + journal。"""
    persisted = []

    builder = object.__new__(GraphBuilderService)
    graph_id = builder.create_graph(
        "Graph",
        graph_id_callback=lambda value: persisted.append(value),
    )

    assert graph_id.startswith("mirofish_")
    assert persisted == [graph_id]


def test_create_graph_respects_caller_supplied_id():
    builder = object.__new__(GraphBuilderService)
    graph_id = builder.create_graph("Graph", graph_id="known-id")
    assert graph_id == "known-id"

"""Graphiti 节点/边读取测试（替代 test_zep_edge_paging.py 的 cursor 分页测试）。

Graphiti 的 namespace API（``client.nodes.entity.get_by_group_ids`` 等）返回
全量列表，无 ``zep-next-cursor`` 响应头分页。本测试验证
``fetch_all_nodes`` / ``fetch_all_edges`` 正确桥接 async 调用、尊重
``max_items`` 截断、保留 no-op kwargs 的后向兼容。
"""

from types import SimpleNamespace

import pytest

from app.utils import graphiti_paging


def _client(nodes=None, edges=None):
    """构造 mock Graphiti 客户端，nodes/edges 为返回列表。"""

    async def _get_nodes(group_ids=None, limit=None, uuid_cursor=None):
        return list(nodes or [])

    async def _get_edges(group_ids=None, limit=None, uuid_cursor=None):
        return list(edges or [])

    entity_nodes = SimpleNamespace(get_by_group_ids=_get_nodes)
    entity_edges = SimpleNamespace(get_by_group_ids=_get_edges)
    return SimpleNamespace(
        nodes=SimpleNamespace(entity=entity_nodes),
        edges=SimpleNamespace(entity=entity_edges),
    )


def test_fetch_all_nodes_returns_full_list(monkeypatch):
    nodes = [SimpleNamespace(uuid=f"n{i}") for i in range(3)]
    client = _client(nodes=nodes)

    result = graphiti_paging.fetch_all_nodes(client, "graph")

    assert [n.uuid for n in result] == ["n0", "n1", "n2"]


def test_fetch_all_edges_returns_full_list():
    edges = [SimpleNamespace(uuid=f"e{i}") for i in range(2)]
    client = _client(edges=edges)

    result = graphiti_paging.fetch_all_edges(client, "graph")

    assert [e.uuid for e in result] == ["e0", "e1"]


def test_max_items_truncates_nodes():
    nodes = [SimpleNamespace(uuid=f"n{i}") for i in range(5)]
    client = _client(nodes=nodes)

    result = graphiti_paging.fetch_all_nodes(client, "graph", max_items=3)

    assert [n.uuid for n in result] == ["n0", "n1", "n2"]


def test_max_items_truncates_edges():
    edges = [SimpleNamespace(uuid=f"e{i}") for i in range(5)]
    client = _client(edges=edges)

    result = graphiti_paging.fetch_all_edges(client, "graph", max_items=2)

    assert [e.uuid for e in result] == ["e0", "e1"]


def test_empty_graph_returns_empty_list():
    client = _client(nodes=[], edges=[])

    assert graphiti_paging.fetch_all_nodes(client, "graph") == []
    assert graphiti_paging.fetch_all_edges(client, "graph") == []


def test_noop_kwargs_preserve_backward_compat():
    """page_size/max_retries/retry_delay 作为 no-op kwargs 仍可传入不报错。"""
    client = _client(nodes=[SimpleNamespace(uuid="n1")])

    result = graphiti_paging.fetch_all_nodes(
        client, "graph",
        page_size=25, max_retries=7, retry_delay=0.25,
    )

    assert [n.uuid for n in result] == ["n1"]


def test_empty_graph_id_raises():
    client = _client()
    with pytest.raises(ValueError, match="graph_id"):
        graphiti_paging.fetch_all_nodes(client, "")


def test_invalid_max_items_raises():
    client = _client()
    with pytest.raises(ValueError, match="max_items"):
        graphiti_paging.fetch_all_nodes(client, "graph", max_items=0)

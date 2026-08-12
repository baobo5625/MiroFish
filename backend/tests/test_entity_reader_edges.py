"""ZepEntityReader.get_node_edges 测试（替代 test_zep_entity_reader_edges.py）。

Graphiti 下无 graph_id 的 ``get_node_edges`` 走
``client.edges.entity.get_by_node_uuid``（async，双向返回边，修复了
Zep SDK 只返回 outgoing 的缺陷）。本测试 mock ``_call_with_retry``，
验证它调用了 namespace 方法并正确映射返回字段。
"""

from types import SimpleNamespace

from app.services.zep_entity_reader import ZepEntityReader


def test_get_node_edges_without_graph_id_uses_namespace_method():
    """无 graph_id 时直接调 edges.entity.get_by_node_uuid（双向）。"""
    captured = {}

    def fake_call(func, *, operation_name, **kwargs):
        # 直接执行 func（func 内部应是 run_async，这里 mock 掉 run_async）
        captured["op"] = operation_name
        return func()

    # mock run_async：直接 await 协程（测试在同步上下文里）
    import asyncio

    def fake_run_async(coro, timeout):
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    async def fake_get_by_node_uuid(node_uuid):
        captured["node_uuid"] = node_uuid
        return [SimpleNamespace(
            uuid="edge-1",
            name="KNOWS",
            fact="Alice knows Bob",
            source_node_uuid="node-1",
            target_node_uuid="node-2",
            attributes={"since": "2024"},
        )]

    client = SimpleNamespace(
        edges=SimpleNamespace(
            entity=SimpleNamespace(get_by_node_uuid=fake_get_by_node_uuid)
        )
    )

    reader = object.__new__(ZepEntityReader)
    reader.client = client
    reader._call_with_retry = fake_call

    # monkeypatch run_async in the entity_reader module
    import app.services.zep_entity_reader as reader_module

    original_run = reader_module.run_async
    reader_module.run_async = fake_run_async
    try:
        result = reader.get_node_edges("node-1")
    finally:
        reader_module.run_async = original_run

    assert result == [{
        "uuid": "edge-1",
        "name": "KNOWS",
        "fact": "Alice knows Bob",
        "source_node_uuid": "node-1",
        "target_node_uuid": "node-2",
        "attributes": {"since": "2024"},
    }]
    assert captured["node_uuid"] == "node-1"


def test_get_node_edges_with_graph_id_filters_bidirectionally():
    """有 graph_id 时走全图过滤，双向匹配 source/target。"""
    all_edges = [
        SimpleNamespace(
            uuid="e1", name="KNOWS", fact="A knows B",
            source_node_uuid="node-1", target_node_uuid="node-2",
            attributes={},
        ),
        SimpleNamespace(
            uuid="e2", name="LIKES", fact="C likes A",
            source_node_uuid="node-3", target_node_uuid="node-1",
            attributes={},
        ),
        SimpleNamespace(
            uuid="e3", name="UNRELATED", fact="X Y",
            source_node_uuid="node-4", target_node_uuid="node-5",
            attributes={},
        ),
    ]

    reader = object.__new__(ZepEntityReader)
    reader.get_all_edges = lambda graph_id: [
        {
            "uuid": e.uuid, "name": e.name, "fact": e.fact,
            "source_node_uuid": e.source_node_uuid,
            "target_node_uuid": e.target_node_uuid,
            "attributes": e.attributes,
        }
        for e in all_edges
    ]

    result = reader.get_node_edges("node-1", graph_id="graph")

    # node-1 作为 source (e1) 和 target (e2) 都应返回，e3 排除
    assert [e["uuid"] for e in result] == ["e1", "e2"]

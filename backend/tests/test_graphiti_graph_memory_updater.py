"""ZepGraphMemoryUpdater 测试（替代 test_zep_graph_memory_updater.py）。

Graphiti 下 ``_send_batch_activities`` 调 ``run_async(self.client.add_episode(...))``
而非 Zep 的 ``client.graph.add``。本测试 mock ``run_async`` 同步驱动 mock
协程，保留对 buffer/queue/threading/失败语义的全部断言。

关键变化：
* ``add_episode`` 返回 ``AddEpisodeResults``，episode UUID 在 ``.episode.uuid``
* UUID5 幂等：同 simulation+platform+内容产出相同 UUID
* ``_wait_for_pending_episodes`` 不再轮询 ``.processed``，改为防御性 episode 计数校验
"""

import asyncio
from types import SimpleNamespace
import threading
from queue import Queue

import pytest

from app.services import zep_graph_memory_updater as updater_module
from app.services.zep_graph_memory_updater import (
    AgentActivity,
    ZepGraphMemoryManager,
    ZepGraphMemoryUpdater,
)


def _activity(index=1, content="hello"):
    return AgentActivity(
        platform="twitter",
        agent_id=index,
        agent_name=f"Agent {index}",
        action_type="CREATE_POST",
        action_args={"content": content},
        round_num=index,
        timestamp="2026-07-22T12:00:00+08:00",
    )


def _sync_run_async(coro, timeout):
    """同步驱动协程，用于测试。"""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _client(add_episode):
    """mock Graphiti 客户端，add_episode 为 async 返回 AddEpisodeResults 形状。"""

    async def _add_episode(**kwargs):
        return add_episode(**kwargs)

    async def _get_episodes(group_ids=None, limit=None, uuid_cursor=None):
        return [SimpleNamespace(uuid="episode-1")]

    return SimpleNamespace(
        add_episode=_add_episode,
        nodes=SimpleNamespace(
            episode=SimpleNamespace(get_by_group_ids=_get_episodes),
        ),
    )


def _updater(monkeypatch, add_episode, simulation_id="sim-1"):
    client = _client(add_episode)
    monkeypatch.setattr(updater_module, "get_graphiti_client", lambda _key: client)
    monkeypatch.setattr(updater_module, "run_async", _sync_run_async)
    updater = ZepGraphMemoryUpdater(
        "graph-1",
        api_key="test-key",
        simulation_id=simulation_id,
    )
    updater.SEND_INTERVAL = 0
    return updater


def _episode_result(uuid="episode-1"):
    """构造 AddEpisodeResults 形状。"""
    return SimpleNamespace(episode=SimpleNamespace(uuid=uuid))


def test_stop_drains_an_immediately_queued_tail_activity(monkeypatch):
    writes = []

    def add_episode(**kwargs):
        writes.append(kwargs)
        return _episode_result("episode-1")

    updater = _updater(monkeypatch, add_episode)
    updater.start()
    updater.add_activity(_activity())
    updater.stop()

    assert len(writes) == 1
    assert updater.get_stats()["items_sent"] == 1
    assert updater.get_stats()["queue_size"] == 0


def test_network_write_happens_outside_the_buffer_lock(monkeypatch):
    lock_was_available = []
    updater = None

    def add_episode(**_kwargs):
        acquired = updater._buffer_lock.acquire(blocking=False)
        lock_was_available.append(acquired)
        if acquired:
            updater._buffer_lock.release()
        return _episode_result("episode-1")

    updater = _updater(monkeypatch, add_episode)
    updater.start()
    for index in range(updater.BATCH_SIZE):
        updater.add_activity(_activity(index))
    updater.stop()

    assert lock_was_available == [True]


def test_activity_episode_has_provenance_and_safe_size(monkeypatch):
    writes = []
    updater = _updater(
        monkeypatch,
        lambda **kwargs: writes.append(kwargs) or _episode_result("episode-1"),
        simulation_id="sim-provenance",
    )

    updater._send_batch_activities(
        [_activity(content="x" * 20_000)],
        "twitter",
    )

    assert len(writes) == 1
    write = writes[0]
    # episode_body 是内容（原 data）
    assert len(write["episode_body"]) <= updater.MAX_EPISODE_CHARS
    # reference_time 是 datetime（原 created_at）
    assert write["reference_time"].year == 2026
    assert "MiroFish simulation activity batch" in write["source_description"]
    # provenance 嵌入 source_description（Graphiti add_episode 无 metadata dict）
    assert "sim-provenance" in write["source_description"]
    assert "twitter" in write["source_description"]
    assert write["group_id"] == "graph-1"


def test_add_episode_does_not_pass_uuid(monkeypatch):
    """memory updater 的 add_episode 不传 uuid（Graphiti 的 uuid 参数是
    get_or_create 语义，传不存在的 UUID 会抛 NodeNotFoundError）。
    episode UUID 由 graphiti 自动生成，返回的 AddEpisodeResults.episode.uuid
    被记录到 _pending_episode_uuids。"""

    def add_episode(**kwargs):
        # 不应传 uuid
        assert "uuid" not in kwargs or kwargs["uuid"] is None
        return _episode_result("auto-generated-uuid")

    updater = _updater(monkeypatch, add_episode, simulation_id="sim-1")

    updater._send_batch_activities([_activity(1, "hello")], "twitter")

    # episode UUID 从返回值提取并记录
    assert updater._pending_episode_uuids == ["auto-generated-uuid"]
    assert updater.get_stats()["items_sent"] == 1


def test_failed_write_is_reported_by_stop(monkeypatch):
    def add_episode(**_kwargs):
        raise RuntimeError("write failed")

    updater = _updater(monkeypatch, add_episode)
    updater.start()
    updater.add_activity(_activity())

    with pytest.raises(RuntimeError, match="ingestion is incomplete"):
        updater.stop()

    assert updater.get_stats()["failed_count"] == 1


def test_failed_simulation_action_is_not_ingested(monkeypatch):
    updater = _updater(
        monkeypatch,
        lambda **_kwargs: _episode_result("unused"),
    )

    updater.add_activity_from_dict(
        {
            "agent_id": 1,
            "agent_name": "Agent",
            "action_type": "CREATE_POST",
            "action_args": {"content": "not actually posted"},
            "success": False,
        },
        "twitter",
    )

    assert updater.get_stats()["queue_size"] == 0
    assert updater.get_stats()["skipped_count"] == 1


def test_stop_cannot_finish_between_acceptance_check_and_enqueue(monkeypatch):
    writes = []
    updater = _updater(
        monkeypatch,
        lambda **kwargs: writes.append(kwargs) or _episode_result("episode-1"),
    )

    put_entered = threading.Event()
    allow_put = threading.Event()

    class BlockingQueue(Queue):
        def put(self, item, block=True, timeout=None):
            put_entered.set()
            assert allow_put.wait(timeout=2)
            return super().put(item, block=block, timeout=timeout)

    updater._activity_queue = BlockingQueue()
    updater.start()
    producer = threading.Thread(target=updater.add_activity, args=(_activity(),))
    producer.start()
    assert put_entered.wait(timeout=1)

    stopper = threading.Thread(target=updater.stop)
    stopper.start()
    stopper.join(timeout=0.1)
    assert stopper.is_alive()

    allow_put.set()
    producer.join(timeout=2)
    stopper.join(timeout=2)

    assert not producer.is_alive()
    assert not stopper.is_alive()
    assert len(writes) == 1


def test_wait_for_pending_episodes_clears_list(monkeypatch):
    """add_episode 已 inline 处理，_wait_for_pending_episodes 做防御性校验并清空。"""
    updater = _updater(
        monkeypatch,
        lambda **_kwargs: _episode_result("episode-1"),
    )
    updater._pending_episode_uuids = ["episode-1"]

    # 防御性校验会调 get_by_group_ids；mock 已配置返回 episode-1
    updater._wait_for_pending_episodes()

    assert updater._pending_episode_uuids == []


def test_explicit_graph_destruction_can_discard_a_stopped_failed_updater():
    updater = SimpleNamespace(
        graph_id="graph-1",
        _running=False,
        _worker_thread=SimpleNamespace(is_alive=lambda: False),
    )
    ZepGraphMemoryManager._updaters["sim-failed"] = updater
    try:
        assert ZepGraphMemoryManager.discard_inactive_updater("sim-failed") is True
        assert "sim-failed" not in ZepGraphMemoryManager._updaters
    finally:
        ZepGraphMemoryManager._updaters.pop("sim-failed", None)


def test_flush_deadline_keeps_unattempted_platform_for_a_safe_retry(monkeypatch):
    now = [0.0]
    writes = []

    def add_episode(**kwargs):
        writes.append(kwargs)
        now[0] = 2.0
        return _episode_result(f"episode-{len(writes)}")

    updater = _updater(monkeypatch, add_episode)
    updater._platform_buffers["twitter"] = [_activity(1)]
    reddit_activity = _activity(2)
    reddit_activity.platform = "reddit"
    updater._platform_buffers["reddit"] = [reddit_activity]
    monkeypatch.setattr(updater_module.time, "time", lambda: now[0])

    with pytest.raises(TimeoutError, match="deadline"):
        updater._flush_remaining(deadline=1.0)

    assert updater._platform_buffers["twitter"] == []
    assert updater._platform_buffers["reddit"] == [reddit_activity]

    now[0] = 0.0
    updater._flush_remaining(deadline=1.0)
    assert updater._platform_buffers["reddit"] == []
    assert len(writes) == 2

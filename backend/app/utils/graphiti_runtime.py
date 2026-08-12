"""Graphiti async→sync 桥接运行时。

Graphiti 是 async-first（基于 ``neo4j.AsyncGraphDatabase`` + ``httpx.AsyncClient``），
而 MiroFish 是同步 Flask + 守护线程。本模块提供一个进程级共享 asyncio 事件循环，
跑在独立的守护线程上；同步调用方通过 :func:`run_async` 把协程提交到该循环并阻塞等待结果。

设计要点
--------
* 单一循环、单一线程：MiroFish 是单租户单仿真并发模型，图操作非热路径，串行化可接受。
* ``run_coroutine_threadsafe`` 天然线程安全，可从任意工作线程调用。
* 死锁 guard：禁止在 loop 线程内调用 ``run_async``（Neo4j driver / httpx LLM
  embedder 都是 async-native，不会回调同步代码，但 guard 兜底）。
* ``run_async`` 的 ``timeout`` 与 ``Config.GRAPHITI_REQUEST_TIMEOUT_SECONDS`` 对齐，
  替代原先 Zep 的 ``ZEP_HTTP_REQUEST_TIMEOUT_SECONDS``。
"""

from __future__ import annotations

import asyncio
import threading
from typing import Awaitable, TypeVar

from .logger import get_logger

logger = get_logger("mirofish.graphiti_runtime")

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_loop_thread: threading.Thread | None = None
_loop_lock = threading.Lock()


def _ensure_loop() -> asyncio.AbstractEventLoop:
    """惰性启动共享事件循环（线程安全单例）。"""

    global _loop, _loop_thread
    if _loop is not None:
        return _loop

    with _loop_lock:
        if _loop is None:
            loop = asyncio.new_event_loop()

            def _run_forever() -> None:
                asyncio.set_event_loop(loop)
                loop.run_forever()

            thread = threading.Thread(
                target=_run_forever, daemon=True, name="graphiti-loop"
            )
            thread.start()
            _loop = loop
            _loop_thread = thread
            logger.debug("graphiti 事件循环已启动于守护线程 %s", thread.name)

    return _loop


def run_async(coro: Awaitable[T], timeout: float) -> T:
    """把协程提交到共享 loop 线程，同步阻塞等待结果。

    Args:
        coro: 待执行的协程（如 ``client.add_episode(...)``）。
        timeout: 最长等待秒数；超时抛 :class:`TimeoutError`。

    Raises:
        RuntimeError: 在 loop 线程内调用（会死锁）。
        TimeoutError: 超过 timeout。
        Exception: 协程内抛出的任何异常原样上抛。
    """

    loop = _ensure_loop()

    # 死锁 guard：若当前线程就是 loop 线程，提交 future 后无人 run，必然死锁。
    if threading.current_thread() is _loop_thread:
        raise RuntimeError(
            "run_async 不能在 graphiti-loop 线程内调用（会死锁）；"
            "请用 await 直接驱动协程"
        )

    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result(timeout=timeout)


def shutdown_loop(timeout: float = 5.0) -> None:
    """关闭共享 loop（仅用于测试或进程退出时的清理）。"""

    global _loop, _loop_thread
    if _loop is None:
        return

    loop = _loop
    _loop = None
    _loop_thread = None

    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    if _loop_thread is not None and _loop_thread.is_alive():
        _loop_thread.join(timeout=timeout)

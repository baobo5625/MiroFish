"""Graphiti 节点/边读取（替代 zep_paging.py 的 cursor 分页）。

Zep Cloud 用 ``zep-next-cursor`` 响应头做分页；Graphiti 的 namespace API
（``client.nodes.entity.get_by_group_ids`` / ``client.edges.entity.get_by_group_ids``）
直接返回全量列表，``limit`` + ``uuid_cursor`` 是可选的游标分页。

本模块保留原 ``fetch_all_nodes`` / ``fetch_all_edges`` 签名（含 ``page_size``、
``max_retries``、``retry_delay`` 等 no-op kwargs），使上层 ``get_all_nodes``、
``get_all_edges``、``get_node_edges`` 等无需改签名。

内存评估：MiroFish 图来自 500 字符切块、单批 ≤350 块，实际规模在数千节点/边量级，
全量加载可接受。若未来图超 ~5 万节点，应改用 ``uuid_cursor`` 增量分页。
"""

from __future__ import annotations

from typing import Any

from .graphiti_client import (
    GRAPHITI_REQUEST_TIMEOUT_SECONDS,
    call_graphiti_read_with_retry,
)
from .graphiti_runtime import run_async
from .logger import get_logger

logger = get_logger("mirofish.graphiti_paging")

# 原 zep_paging 的默认值，保留以兼容调用方传参。
_DEFAULT_PAGE_SIZE = 100
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_DELAY = 2.0


def _validate_args(graph_id: str, max_items: int | None) -> None:
    if not graph_id:
        raise ValueError("graph_id must not be empty")
    if max_items is not None and max_items < 1:
        raise ValueError("max_items must be at least 1 when provided")


def fetch_all_nodes(
    client: Any,
    graph_id: str,
    page_size: int = _DEFAULT_PAGE_SIZE,
    max_items: int | None = None,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
) -> list[Any]:
    """读取一个 group_id 下的全部节点（EntityNode 列表）。

    ``page_size`` 因 Graphiti 无页式分页而被忽略，保留仅为签名兼容。
    """

    _validate_args(graph_id, max_items)
    # 若给了 max_items，把它作为 limit 传给 Graphiti，避免拉全量再截断。
    limit = max_items
    nodes = call_graphiti_read_with_retry(
        lambda: run_async(
            client.nodes.entity.get_by_group_ids(
                group_ids=[graph_id], limit=limit
            ),
            timeout=GRAPHITI_REQUEST_TIMEOUT_SECONDS,
        ),
        operation_name=f"fetch nodes (group={graph_id})",
        max_attempts=max_retries,
        initial_delay=retry_delay,
    )
    result = list(nodes or [])
    if max_items is not None and len(result) > max_items:
        logger.warning(
            "Graphiti nodes for group %s exceeded max_items=%s; truncating",
            graph_id, max_items,
        )
        result = result[:max_items]
    return result


def fetch_all_edges(
    client: Any,
    graph_id: str,
    page_size: int = _DEFAULT_PAGE_SIZE,
    max_retries: int = _DEFAULT_MAX_RETRIES,
    retry_delay: float = _DEFAULT_RETRY_DELAY,
    max_items: int | None = None,
) -> list[Any]:
    """读取一个 group_id 下的全部边（EntityEdge 列表）。

    注意签名顺序与 ``fetch_all_nodes`` 不同（``max_items`` 在末尾），
    与原 ``zep_paging.fetch_all_edges`` 保持一致。
    """

    _validate_args(graph_id, max_items)
    limit = max_items
    edges = call_graphiti_read_with_retry(
        lambda: run_async(
            client.edges.entity.get_by_group_ids(
                group_ids=[graph_id], limit=limit
            ),
            timeout=GRAPHITI_REQUEST_TIMEOUT_SECONDS,
        ),
        operation_name=f"fetch edges (group={graph_id})",
        max_attempts=max_retries,
        initial_delay=retry_delay,
    )
    result = list(edges or [])
    if max_items is not None and len(result) > max_items:
        logger.warning(
            "Graphiti edges for group %s exceeded max_items=%s; truncating",
            graph_id, max_items,
        )
        result = result[:max_items]
    return result

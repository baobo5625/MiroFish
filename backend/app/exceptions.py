"""MiroFish 领域异常。

Graphiti 适配层用这些异常替代原先的 ``zep_cloud.NotFoundError``，
保持上层服务（graph_builder / zep_tools / zep_entity_reader / api 路由）
的 except 契约不变。
"""

from __future__ import annotations


class GraphNotFound(Exception):
    """请求的图（group_id）/节点/episode 不存在。

    替代 ``zep_cloud.NotFoundError``：在读取单节点、单 episode、或按
    group_id 拉取得到空结果时抛出。上层把它当作"未找到"而非"出错"处理。
    """


class GraphInUseError(RuntimeError):
    """图正在被报告读取或仿真使用，拒绝删除/重置。

    原先定义在 ``app/api/graph.py``，提到这里供多个模块复用。
    """

    def __init__(self, graph_id: str, message: str | None = None) -> None:
        self.graph_id = graph_id
        super().__init__(message or f"图 {graph_id} 正在被使用，无法删除或重置")

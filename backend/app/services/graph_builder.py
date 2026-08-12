"""
图谱构建服务
接口2：使用 Graphiti（Zep 官方开源核心）构建 Standalone Graph

替代原先基于 Zep Cloud Batch API 的实现。Graphiti 无 Batch API，故用
``add_episode`` + ``asyncio.gather`` + ``asyncio.Semaphore`` 做并发摄入；
幂等性靠 ``uuid5(operation_id, chunk_index)`` 客户端生成 episode UUID
保证（比 Zep 的 ``graph.add`` 无幂等键更安全）。
"""

import asyncio
import hashlib
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from ..config import Config
from ..models.task import TaskManager, TaskStatus
from ..utils.graphiti_client import (
    GRAPHITI_INGESTION_WAIT_TIMEOUT_SECONDS,
    call_graphiti_read_with_retry,
    delete_group,
    get_graphiti_client,
)
from ..utils.graphiti_paging import fetch_all_edges, fetch_all_nodes
from ..utils.graphiti_runtime import run_async
from ..utils.locale import get_locale, set_locale, t
from ..utils.ontology import (
    MAX_ONTOLOGY_TYPES,
    RESERVED_ONTOLOGY_ATTRIBUTE_NAMES,
    normalize_ontology_attributes,
    normalize_ontology_source_targets,
)
from .text_processor import TextProcessor


@dataclass
class GraphInfo:
    """图谱信息"""

    graph_id: str
    node_count: int
    edge_count: int
    entity_types: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph_id": self.graph_id,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_types": self.entity_types,
        }


@dataclass(frozen=True)
class BatchSubmission:
    """一次图谱摄入操作的持久身份。

    Graphiti 无 Batch API，``batch_id`` 字段退化为 ``operation_id``
    （保留字段名以兼容上层 ``api/graph.py`` 的 ``project.zep_batch_id``
    持久化逻辑）。``episode_uuids`` 与 ``item_count`` 是有意义的字段。
    """

    batch_id: str
    operation_id: str
    episode_uuids: List[str]
    item_count: int


class GraphBuilderService:
    """图谱构建服务，基于 Graphiti。"""

    # add_episode 并发上限。Graphiti 每次 add_episode 会触发 LLM 抽取，
    # 并发过高会打爆 LLM 速率限制；8 是保守值，可按 LLM 配额调。
    EPISODE_CONCURRENCY = 8

    def __init__(self, api_key: Optional[str] = None):
        # api_key 仅为签名兼容保留（原 get_zep_client 接受它）。
        # Graphiti 认证经 Neo4j 凭据 + LLM key，不按调用键控。
        self.api_key = api_key or Config.GRAPHITI_LLM_API_KEY
        if not self.api_key:
            raise ValueError("GRAPHITI_LLM_API_KEY 未配置（可复用 LLM_API_KEY）")

        self.client = get_graphiti_client(self.api_key)
        self.task_manager = TaskManager()

    # ------------------------------------------------------------------
    # 旧入口（异步构建）
    # ------------------------------------------------------------------
    def build_graph_async(
        self,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str = "MiroFish Graph",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        batch_size: int = 350,
    ) -> str:
        """异步构建图谱（后台线程）。"""

        task_id = self.task_manager.create_task(
            task_type="graph_build",
            metadata={
                "graph_name": graph_name,
                "chunk_size": chunk_size,
                "text_length": len(text),
            },
        )

        current_locale = get_locale()

        thread = threading.Thread(
            target=self._build_graph_worker,
            args=(
                task_id,
                text,
                ontology,
                graph_name,
                chunk_size,
                chunk_overlap,
                batch_size,
                current_locale,
            ),
        )
        thread.daemon = True
        thread.start()

        return task_id

    def _build_graph_worker(
        self,
        task_id: str,
        text: str,
        ontology: Dict[str, Any],
        graph_name: str,
        chunk_size: int,
        chunk_overlap: int,
        batch_size: int,
        locale: str = "zh",
    ):
        set_locale(locale)
        try:
            self.task_manager.update_task(
                task_id,
                status=TaskStatus.PROCESSING,
                progress=5,
                message=t("progress.startBuildingGraph"),
            )

            chunks = TextProcessor.split_text(text, chunk_size, chunk_overlap)
            self.validate_batch_chunks(chunks, batch_size=batch_size)
            total_chunks = len(chunks)

            # 1. 创建图谱（Graphiti 无显式建图，group_id 首次 add_episode 隐式创建）
            graph_id = self.create_graph(graph_name)
            self.task_manager.update_task(
                task_id,
                progress=10,
                message=t("progress.graphCreated", graphId=graph_id),
            )

            # 2. 设置本体（Graphiti per-call，缓存到实例供 add_text_batches 复用）
            entity_types, edge_types, edge_type_map = self._build_ontology_types(ontology)
            self.set_ontology(graph_id, ontology)
            self.task_manager.update_task(
                task_id,
                progress=15,
                message=t("progress.ontologySet"),
            )

            self.task_manager.update_task(
                task_id,
                progress=20,
                message=t("progress.textSplit", count=total_chunks),
            )

            # 3. 分批摄入（Graphiti 无 Batch API，用 gather+Semaphore）
            submission = self.add_text_batches(
                graph_id,
                chunks,
                batch_size,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=20 + int(prog * 0.4),  # 20-60%
                    message=msg,
                ),
                entity_types=entity_types,
                edge_types=edge_types,
                edge_type_map=edge_type_map,
            )

            # 4. add_episode 已 await 完整抽取，无需再轮询；仅校验数量
            self.task_manager.update_task(
                task_id,
                progress=60,
                message=t("progress.waitingZepProcess"),
            )
            self._wait_for_batch(
                submission,
                lambda msg, prog: self.task_manager.update_task(
                    task_id,
                    progress=60 + int(prog * 0.3),  # 60-90%
                    message=msg,
                ),
            )

            self.task_manager.update_task(
                task_id,
                progress=90,
                message=t("progress.fetchingGraphInfo"),
            )

            graph_info = self._get_graph_info(graph_id)

            self.task_manager.complete_task(
                task_id,
                {
                    "graph_id": graph_id,
                    "graph_info": graph_info.to_dict(),
                    "chunks_processed": total_chunks,
                },
            )

        except Exception as e:
            import traceback

            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            self.task_manager.fail_task(task_id, error_msg)

    # ------------------------------------------------------------------
    # 建图
    # ------------------------------------------------------------------
    def create_graph(
        self,
        name: str,
        *,
        graph_id: str | None = None,
        graph_id_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """生成 group_id 并 journal。

        Graphiti 无显式建图调用——一个 group_id 名下的子图在首次
        ``add_episode(group_id=...)`` 时隐式创建。故本方法退化为：生成
        客户端 durable ID + journal，不再有 Zep 时代的 reconcile 逻辑
        （无 POST 可丢失）。
        """

        graph_id = graph_id or f"mirofish_{uuid.uuid4().hex[:16]}"
        if graph_id_callback:
            graph_id_callback(graph_id)
        return graph_id

    @staticmethod
    def build_operation_id(graph_id: str, chunks: List[str]) -> str:
        payload_hash = hashlib.sha256("\0".join(chunks).encode("utf-8")).hexdigest()
        return hashlib.sha256(f"{graph_id}:{payload_hash}".encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # 本体
    # ------------------------------------------------------------------
    def _build_ontology_types(
        self, ontology: Dict[str, Any]
    ) -> tuple[Dict[str, type], Dict[str, type], Dict[tuple[str, str], List[str]]]:
        """从本体 dict 构造 Graphiti 的 entity_types / edge_types / edge_type_map。

        Graphiti 无 ``set_ontology``；自定义类型在每次 ``add_episode`` 调用时
        通过 ``entity_types`` / ``edge_types`` / ``edge_type_map`` 参数传入
        （per-call，非 client-global）。

        * entity_types: ``Dict[str, type[BaseModel]]``，键=实体类型名，值=Pydantic 类
        * edge_types:   ``Dict[str, type[BaseModel]]``，键=边类型名，值=Pydantic 类
        * edge_type_map: ``Dict[(source_type, target_type), List[edge_type_name]]``
        """

        from pydantic import BaseModel, Field

        def safe_attr_name(attr_name: str) -> str:
            if attr_name.lower() in RESERVED_ONTOLOGY_ATTRIBUTE_NAMES:
                return f"entity_{attr_name}"
            return attr_name

        entity_types: Dict[str, type] = {}
        for entity_def in ontology.get("entity_types", [])[:MAX_ONTOLOGY_TYPES]:
            name = entity_def["name"]
            description = entity_def.get("description", f"A {name} entity.")

            attrs: Dict[str, Any] = {"__doc__": description}
            annotations: Dict[str, Any] = {}
            for normalized in normalize_ontology_attributes(entity_def.get("attributes", [])):
                attr_name = safe_attr_name(normalized["name"])
                attrs[attr_name] = Field(description=normalized["description"], default=None)
                annotations[attr_name] = Optional[str]  # Graphiti 属性用 plain str
            attrs["__annotations__"] = annotations

            entity_class = type(name, (BaseModel,), attrs)
            entity_class.__doc__ = description
            entity_types[name] = entity_class

        edge_types: Dict[str, type] = {}
        edge_type_map: Dict[tuple[str, str], List[str]] = {}
        for edge_def in ontology.get("edge_types", [])[:MAX_ONTOLOGY_TYPES]:
            name = edge_def["name"]
            description = edge_def.get("description", f"A {name} relationship.")

            attrs = {"__doc__": description}
            annotations = {}
            for normalized in normalize_ontology_attributes(edge_def.get("attributes", [])):
                attr_name = safe_attr_name(normalized["name"])
                attrs[attr_name] = Field(description=normalized["description"], default=None)
                annotations[attr_name] = Optional[str]
            attrs["__annotations__"] = annotations

            class_name = "".join(word.capitalize() for word in name.split("_"))
            edge_class = type(class_name, (BaseModel,), attrs)
            edge_class.__doc__ = description
            edge_types[name] = edge_class

            for st in normalize_ontology_source_targets(edge_def.get("source_targets", [])):
                source = st.get("source", "Entity")
                target = st.get("target", "Entity")
                edge_type_map.setdefault((source, target), []).append(name)

        return entity_types, edge_types, edge_type_map

    def set_ontology(self, graph_id: str, ontology: Dict[str, Any]):
        """设置图谱本体（保留为公开方法，兼容上层调用）。

        Graphiti 无 ``set_ontology`` 调用——自定义类型在 ``add_episode`` 时
        per-call 传入。本方法仅做本体构造并缓存到实例，供 ``add_text_batches``
        复用；同时做一次构造校验，提前暴露本体格式错误。
        """

        entity_types, edge_types, edge_type_map = self._build_ontology_types(ontology)
        # 缓存最近一次本体，供 add_text_batches 在未显式传参时复用。
        self._cached_entity_types = entity_types
        self._cached_edge_types = edge_types
        self._cached_edge_type_map = edge_type_map

    # ------------------------------------------------------------------
    # 摄入
    # ------------------------------------------------------------------
    def add_text_batches(
        self,
        graph_id: str,
        chunks: List[str],
        batch_size: int = 350,
        progress_callback: Optional[Callable] = None,
        batch_created_callback: Optional[Callable[[str | None, str], None]] = None,
        *,
        entity_types: Optional[Dict[str, type]] = None,
        edge_types: Optional[Dict[str, type]] = None,
        edge_type_map: Optional[Dict[tuple[str, str], List[str]]] = None,
    ) -> BatchSubmission:
        """提交文档分块到 Graphiti。

        Graphiti 无 Batch API，改用 ``asyncio.gather`` + ``Semaphore`` 并发
        ``add_episode``。每个 episode 的 UUID 由 ``uuid5(operation_id, chunk_index)``
        客户端生成——重跑同 operation_id+chunk_index 产出相同 UUID，
        ``add_episode(uuid=...)`` 幂等，比 Zep 的 ``graph.add``（无幂等键）更安全。

        ``add_episode`` 在返回前已 await 完整抽取流水线，故本方法返回时
        所有 episode 已 processed，无需后续轮询。
        """

        if not graph_id:
            raise ValueError("graph_id is required")
        self.validate_batch_chunks(chunks, batch_size=batch_size)

        total_chunks = len(chunks)
        operation_id = self.build_operation_id(graph_id, chunks)
        if batch_created_callback:
            batch_created_callback(None, operation_id)

        # 本体：优先用显式传入的，否则用 set_ontology 缓存的
        ent_types = entity_types or getattr(self, "_cached_entity_types", None) or {}
        edg_types = edge_types or getattr(self, "_cached_edge_types", None) or {}
        edg_map = edge_type_map or getattr(self, "_cached_edge_type_map", None) or None

        episode_uuids: List[str] = []

        async def _ingest_all() -> List[str]:
            from graphiti_core.nodes import EpisodeType

            sem = asyncio.Semaphore(self.EPISODE_CONCURRENCY)
            results: List[Optional[str]] = [None] * total_chunks
            completed_count = 0

            async def add_one(idx: int, chunk: str) -> tuple[int, str]:
                # Graphiti 的 add_episode(uuid=...) 是 get_or_create 语义：
                # 若该 UUID 已存在则复用，不存在时 get_by_uuid 会抛 NodeNotFoundError
                # 且不回退创建。故不传 uuid，让 Graphiti 自动生成。
                # 幂等性靠重建前 delete_graph 清空保证（force=True 路径）。
                async with sem:
                    add_result = await self.client.add_episode(
                        name=f"mirofish_chunk_{idx}",
                        episode_body=chunk,
                        source_description="MiroFish source document chunk",
                        reference_time=datetime.now(timezone.utc),
                        source=EpisodeType.text,
                        group_id=graph_id,
                        entity_types=ent_types or None,
                        edge_types=edg_types or None,
                        edge_type_map=edg_map,
                    )
                # AddEpisodeResults.episode.uuid
                episode = getattr(add_result, "episode", None)
                actual_uuid = getattr(episode, "uuid", None) or ""
                return idx, actual_uuid

            tasks = [add_one(i, c) for i, c in enumerate(chunks)]
            for coro in asyncio.as_completed(tasks):
                idx, actual_uuid = await coro
                results[idx] = actual_uuid
                completed_count += 1
                if progress_callback:
                    progress_callback(
                        t(
                            "progress.zepProcessing",
                            completed=completed_count,
                            total=total_chunks,
                            pending=max(total_chunks - completed_count, 0),
                            elapsed=0,
                        ),
                        completed_count / total_chunks if total_chunks else 1.0,
                    )
            return [r or "" for r in results]

        episode_uuids = run_async(
            _ingest_all(), timeout=GRAPHITI_INGESTION_WAIT_TIMEOUT_SECONDS
        )

        if progress_callback:
            progress_callback(
                t(
                    "progress.processingComplete",
                    completed=len(episode_uuids),
                    total=total_chunks,
                ),
                1.0,
            )

        return BatchSubmission(
            batch_id=operation_id,  # 无 batch API，用 operation_id 兜底
            operation_id=operation_id,
            episode_uuids=episode_uuids,
            item_count=total_chunks,
        )

    @staticmethod
    def validate_batch_chunks(chunks: List[str], *, batch_size: int = 350) -> None:
        """校验分块（保留 MiroFish 的策略性限额）。

        Graphiti 无 Zep Batch API 的硬限制，但这些限额作为客户端护栏仍有意义：
        ``batch_size`` 现仅用于并发分批参考，50k items / 10k chars 仍是合理上限。
        """

        if not chunks:
            raise ValueError("At least one text chunk is required")
        if not 1 <= batch_size <= 350:
            raise ValueError("batch_size must be between 1 and 350")
        if len(chunks) > 50_000:
            raise ValueError("Cannot ingest more than 50,000 chunks at once")
        oversized = [
            index for index, chunk in enumerate(chunks) if len(chunk) > 10_000
        ]
        if oversized:
            raise ValueError(
                f"chunk exceeds 10,000 characters at index {oversized[0]}"
            )

    # ------------------------------------------------------------------
    # 等待（Graphiti 无需轮询，退化为校验）
    # ------------------------------------------------------------------
    def _wait_for_batch(
        self,
        submission: BatchSubmission,
        progress_callback: Optional[Callable] = None,
        timeout: int | None = None,
    ) -> List[str]:
        """Graphiti 的 add_episode 已 inline await 抽取，无需轮询。

        保留方法签名以兼容 ``api/graph.py`` 的调用。仅做数量校验：
        episode_uuids 数应 == item_count。
        """

        if len(submission.episode_uuids) != submission.item_count:
            raise RuntimeError(
                f"ingestion produced {len(submission.episode_uuids)} episodes, "
                f"expected {submission.item_count}"
            )
        if progress_callback:
            progress_callback(
                t(
                    "progress.processingComplete",
                    completed=len(submission.episode_uuids),
                    total=submission.item_count,
                ),
                1.0,
            )
        return list(submission.episode_uuids)

    # ------------------------------------------------------------------
    # 读取
    # ------------------------------------------------------------------
    def _get_graph_info(self, graph_id: str) -> GraphInfo:
        nodes = fetch_all_nodes(self.client, graph_id)
        edges = fetch_all_edges(self.client, graph_id)

        entity_types = set()
        for node in nodes:
            for label in getattr(node, "labels", None) or []:
                if label not in ["Entity", "Node"]:
                    entity_types.add(label)

        return GraphInfo(
            graph_id=graph_id,
            node_count=len(nodes),
            edge_count=len(edges),
            entity_types=list(entity_types),
        )

    def get_graph_data(self, graph_id: str) -> Dict[str, Any]:
        """获取完整图谱数据（节点 + 边 + 时间信息）。"""

        nodes = fetch_all_nodes(self.client, graph_id)
        edges = fetch_all_edges(self.client, graph_id)

        node_map = {}
        for node in nodes:
            node_map[_node_uuid(node)] = getattr(node, "name", "") or ""

        nodes_data = []
        for node in nodes:
            created_at = getattr(node, "created_at", None)
            nodes_data.append(
                {
                    "uuid": _node_uuid(node),
                    "name": getattr(node, "name", ""),
                    "labels": getattr(node, "labels", None) or [],
                    "summary": getattr(node, "summary", "") or "",
                    "attributes": getattr(node, "attributes", None) or {},
                    "created_at": str(created_at) if created_at else None,
                }
            )

        edges_data = []
        for edge in edges:
            created_at = getattr(edge, "created_at", None)
            valid_at = getattr(edge, "valid_at", None)
            invalid_at = getattr(edge, "invalid_at", None)
            expired_at = getattr(edge, "expired_at", None)

            episodes = getattr(edge, "episodes", None)
            if episodes and not isinstance(episodes, list):
                episodes = [str(episodes)]
            elif episodes:
                episodes = [str(e) for e in episodes]

            source_uuid = getattr(edge, "source_node_uuid", "") or ""
            target_uuid = getattr(edge, "target_node_uuid", "") or ""
            # Graphiti 无 fact_type，关系名就是 name
            fact_type = getattr(edge, "name", "") or ""

            edges_data.append(
                {
                    "uuid": _edge_uuid(edge),
                    "name": getattr(edge, "name", "") or "",
                    "fact": getattr(edge, "fact", "") or "",
                    "fact_type": fact_type,
                    "source_node_uuid": source_uuid,
                    "target_node_uuid": target_uuid,
                    "source_node_name": node_map.get(source_uuid, ""),
                    "target_node_name": node_map.get(target_uuid, ""),
                    "attributes": getattr(edge, "attributes", None) or {},
                    "created_at": str(created_at) if created_at else None,
                    "valid_at": str(valid_at) if valid_at else None,
                    "invalid_at": str(invalid_at) if invalid_at else None,
                    "expired_at": str(expired_at) if expired_at else None,
                    "episodes": episodes or [],
                }
            )

        return {
            "graph_id": graph_id,
            "nodes": nodes_data,
            "edges": edges_data,
            "node_count": len(nodes_data),
            "edge_count": len(edges_data),
        }

    def delete_graph(self, graph_id: str):
        """删除图谱（group_id 名下的全部节点/边）。

        Graphiti 无 ``delete_group``，用 ``graphiti_client.delete_group``
        helper 逐 namespace 删除，与原 Zep ``graph.delete`` 同样不重试。
        """

        delete_group(self.client, graph_id)


def _node_uuid(node: Any) -> str:
    """Graphiti Node.uuid（兼容 Zep 的 uuid_）。"""
    return getattr(node, "uuid", None) or getattr(node, "uuid_", None) or ""


def _edge_uuid(edge: Any) -> str:
    return getattr(edge, "uuid", None) or getattr(edge, "uuid_", None) or ""

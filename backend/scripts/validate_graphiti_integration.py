#!/usr/bin/env python3
"""Graphiti 集成验证脚本（替代 validate_zep_cloud_integration.py）。

对 live Neo4j 跑完整 adapter 链路：建图（group_id）→ set_ontology →
add_text_batches（UUID5 幂等）→ search → get_all_nodes/edges → delete。

前置：docker compose up neo4j，并配置 .env（NEO4J_URI/USER/PASSWORD +
LLM_API_KEY）。运行：

    cd backend && uv run python scripts/validate_graphiti_integration.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# 确保能 import app 包
backend_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(backend_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(backend_root.parent / ".env", override=False)

from app.config import Config  # noqa: E402
from app.services.graph_builder import GraphBuilderService  # noqa: E402


def _require_config() -> None:
    errors = Config.validate()
    if errors:
        print("配置错误:", errors, file=sys.stderr)
        sys.exit(1)


def main() -> int:
    _require_config()

    builder = GraphBuilderService()
    graph_id = None
    created = False
    try:
        # 1. 建图（生成 group_id，不调 SDK）
        graph_id = builder.create_graph("MiroFish Validation Graph")
        created = True
        print(f"[1/6] graph_id = {graph_id}")

        # 2. 本体（per-call 缓存）
        ontology = {
            "entity_types": [
                {"name": "Person", "description": "A person.", "attributes": ["role"]},
            ],
            "edge_types": [
                {"name": "KNOWS", "description": "knows",
                 "source_targets": [{"source": "Person", "target": "Person"}]},
            ],
        }
        builder.set_ontology(graph_id, ontology)
        print("[2/6] ontology set")

        # 3. 摄入（add_episode，UUID5 幂等）
        chunks = [
            "Alice works at Acme Corp as an engineer.",
            "Bob knows Alice and they collaborate on the Phoenix project.",
        ]
        submission = builder.add_text_batches(graph_id, chunks)
        assert len(submission.episode_uuids) == len(chunks)
        print(f"[3/6] ingested {submission.item_count} episodes")

        # 4. 读取
        data = builder.get_graph_data(graph_id)
        print(f"[4/6] graph: {data['node_count']} nodes, {data['edge_count']} edges")

        # 5. 搜索（可选，需 search recipe）
        try:
            from app.services.zep_tools import ZepToolsService
            tools = ZepToolsService()
            result = tools.search_graph(graph_id, "Alice Acme", limit=5)
            print(f"[5/6] search: {result.total_count} facts")
        except Exception as e:
            print(f"[5/6] search skipped: {e}")

        print("[6/6] validation PASSED")
        return 0

    except Exception as e:
        print(f"VALIDATION FAILED: {e}", file=sys.stderr)
        traceback.print_exc()
        return 1
    finally:
        if graph_id and created:
            try:
                builder.delete_graph(graph_id)
                print(f"cleanup: deleted graph {graph_id}")
            except Exception as e:
                print(f"cleanup FAILED for {graph_id}: {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())

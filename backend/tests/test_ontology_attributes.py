"""本体归一化与 Graphiti 类型构造测试（替代原 Zep set_ontology 契约测试）。

Graphiti 无 ``set_ontology`` 调用——自定义类型在 ``add_episode`` 时 per-call
传入。``GraphBuilderService.set_ontology`` 现在只做本体构造并缓存到实例，
``_build_ontology_types`` 是真正的被测单元，产出 Graphiti 所需的
``entity_types`` / ``edge_types`` / ``edge_type_map`` 三元组。
"""

from app.services.ontology_generator import OntologyGenerator
from app.services.graph_builder import GraphBuilderService
from app.utils.ontology import (
    MAX_ONTOLOGY_ATTRIBUTES,
    MAX_ONTOLOGY_SOURCE_TARGETS,
    normalize_ontology_attribute,
    normalize_ontology_attributes,
)


def test_normalize_string_attribute():
    assert normalize_ontology_attribute("role") == {
        "name": "role",
        "type": "text",
        "description": "role",
    }


def test_preserve_valid_dictionary_attribute():
    original = {"name": "role", "type": "text", "description": "Public role"}
    assert normalize_ontology_attribute(original) == original
    assert normalize_ontology_attribute(original) is not original


def test_reject_unusable_attribute_shapes():
    for value in (None, 7, [], {}, {"name": None}, {"name": ""}, "   "):
        assert normalize_ontology_attribute(value) is None


def test_attribute_list_is_non_empty_and_capped():
    assert normalize_ontology_attributes(None) == [{
        "name": "details",
        "type": "text",
        "description": "Additional details about this ontology type.",
    }]

    attributes = [None] + [f"field_{index}" for index in range(12)]
    normalized = normalize_ontology_attributes(attributes)

    assert len(normalized) == MAX_ONTOLOGY_ATTRIBUTES
    assert [attribute["name"] for attribute in normalized] == [
        f"field_{index}" for index in range(MAX_ONTOLOGY_ATTRIBUTES)
    ]


def test_generator_normalizes_entity_and_edge_attributes():
    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": [{"name": "speaker", "attributes": ["role", None]}],
        "edge_types": [{"name": "quotes", "attributes": ["source_url", {}]}],
    })

    assert result["entity_types"][0]["attributes"] == [{
        "name": "role",
        "type": "text",
        "description": "role",
    }]
    assert result["edge_types"][0]["attributes"] == [{
        "name": "source_url",
        "type": "text",
        "description": "source_url",
    }]


def test_generator_adds_a_property_to_empty_custom_types():
    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": [{"name": "speaker", "attributes": []}],
        "edge_types": [{"name": "quotes", "attributes": []}],
    })

    assert result["entity_types"][0]["attributes"][0]["name"] == "details"
    assert result["edge_types"][0]["attributes"][0]["name"] == "details"


def _build(ontology):
    """用 GraphBuilderService 实例构造本体三元组（不触碰 SDK）。"""
    builder = object.__new__(GraphBuilderService)
    builder.set_ontology("graph-id", ontology)
    return builder._build_ontology_types(ontology)


def test_build_types_renames_reserved_attribute_names():
    entity_types, _, _ = _build({
        "entity_types": [{
            "name": "Speaker",
            "attributes": ["role", None, {"name": "summary"}],
        }],
        "edge_types": [],
    })

    speaker = entity_types["Speaker"]
    # "summary" 是保留名，应被重命名为 "entity_summary"
    assert set(speaker.model_fields) == {"role", "entity_summary"}


def test_build_types_caps_attributes_at_max():
    entity_types, _, _ = _build({
        "entity_types": [{
            "name": "Speaker",
            "attributes": ["graph_id"] + [f"field_{i}" for i in range(10)],
        }],
        "edge_types": [],
    })

    speaker = entity_types["Speaker"]
    # graph_id 被重命名为 entity_graph_id；属性上限 MAX_ONTOLOGY_ATTRIBUTES
    assert "entity_graph_id" in speaker.model_fields
    assert len(speaker.model_fields) == MAX_ONTOLOGY_ATTRIBUTES


def test_build_types_constructs_edge_type_and_source_target_map():
    _, edge_types, edge_type_map = _build({
        "entity_types": [],
        "edge_types": [{
            "name": "MENTIONS",
            "attributes": [],
            "source_targets": [{"source": "Speaker", "target": "Speaker"}],
        }],
    })

    assert "MENTIONS" in edge_types
    # edge_type_map 按 (source, target) 聚合
    assert edge_type_map[("Speaker", "Speaker")] == ["MENTIONS"]


def test_build_types_handles_edge_only_ontology():
    entity_types, edge_types, edge_type_map = _build({
        "entity_types": [],
        "edge_types": [{
            "name": "RELATED_TO",
            "attributes": ["reason"],
            "source_targets": [{"source": "Entity", "target": "Entity"}],
        }],
    })

    assert entity_types == {}
    assert "RELATED_TO" in edge_types
    assert edge_type_map[("Entity", "Entity")] == ["RELATED_TO"]


def test_build_types_deduplicates_and_caps_edge_source_targets():
    source_targets = [
        {"source": f"Source{index}", "target": f"Target{index}"}
        for index in range(MAX_ONTOLOGY_SOURCE_TARGETS + 2)
    ]
    source_targets.insert(1, dict(source_targets[0]))  # 重复对

    _, _, edge_type_map = _build({
        "entity_types": [],
        "edge_types": [{
            "name": "RELATED_TO",
            "attributes": ["reason"],
            "source_targets": source_targets,
        }],
    })

    # edge_type_map 按 (source,target) 聚合，重复对自然去重
    pairs = list(edge_type_map.keys())
    assert len(pairs) == MAX_ONTOLOGY_SOURCE_TARGETS
    assert pairs[0] == ("Source0", "Target0")
    # 同一 (source,target) 的边类型列表应只有一个 RELATED_TO
    for pair in pairs:
        assert edge_type_map[pair] == ["RELATED_TO"]


def test_generator_ignores_invalid_entries_and_normalizes_edge_names():
    source_targets = [
        {"source": "speaker", "target": "news outlet"},
        {"source": "speaker", "target": "news outlet"},
        None,
    ] + [
        {"source": "speaker", "target": "news outlet" if index == 0 else "Person"}
        for index in range(12)
    ]

    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": ["speaker", None, 7, {"name": "news outlet"}],
        "edge_types": [
            "unusable edge",
            None,
            {"name": "worksFor", "source_targets": source_targets},
            {"name": "works-for", "source_targets": []},
        ],
    })

    assert [entity["name"] for entity in result["entity_types"][:2]] == [
        "Speaker",
        "NewsOutlet",
    ]
    assert [edge["name"] for edge in result["edge_types"]] == ["WORKS_FOR"]
    assert result["edge_types"][0]["source_targets"] == [
        {"source": "Speaker", "target": "NewsOutlet"},
        {"source": "Speaker", "target": "Person"},
    ]


def test_generator_caps_after_discarding_invalid_edge_endpoints():
    invalid_first = [
        {"source": f"Removed{index}", "target": "AlsoRemoved"}
        for index in range(MAX_ONTOLOGY_SOURCE_TARGETS)
    ]
    result = OntologyGenerator(llm_client=object())._validate_and_process({
        "entity_types": [{"name": "person"}, {"name": "organization"}],
        "edge_types": [{
            "name": "works_for",
            "source_targets": invalid_first + [
                {"source": "person", "target": "organization"}
            ],
        }],
    })

    assert result["edge_types"][0]["source_targets"] == [
        {"source": "Person", "target": "Organization"}
    ]

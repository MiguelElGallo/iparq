import json
import re
from pathlib import Path
from types import UnionType
from typing import Union, get_args, get_origin

from jsonschema.validators import validator_for

from iparq.source import ColumnInfo

REPOSITORY_ROOT = Path(__file__).parents[1]
PLUGIN_ROOT = REPOSITORY_ROOT / "plugins" / "iparq"
OPEN_MANIFEST_PATH = PLUGIN_ROOT / "plugin.json"
CODEX_MANIFEST_PATH = PLUGIN_ROOT / ".codex-plugin" / "plugin.json"
AGENT_PLUGIN_SCHEMA_PATH = (
    REPOSITORY_ROOT / "tests" / "schemas" / "agent-plugin-1.0.0.schema.json"
)
CODEX_MARKETPLACE_PATH = REPOSITORY_ROOT / ".agents" / "plugins" / "marketplace.json"
COPILOT_MARKETPLACE_PATH = REPOSITORY_ROOT / ".github" / "plugin" / "marketplace.json"
CANONICAL_SKILL = REPOSITORY_ROOT / ".agents" / "skills" / "iparq-parquet-inspector"
PLUGIN_SKILL = PLUGIN_ROOT / "skills" / "iparq-parquet-inspector"


def project_version() -> str:
    project = (REPOSITORY_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"$', project, re.MULTILINE)
    assert match is not None
    return match.group(1)


def test_open_agent_plugin_manifest_is_portable_and_versioned() -> None:
    manifest = json.loads(OPEN_MANIFEST_PATH.read_text())
    schema = json.loads(AGENT_PLUGIN_SCHEMA_PATH.read_text())
    validator_class = validator_for(schema)
    validator_class.check_schema(schema)
    validator_class(schema).validate(manifest)

    assert manifest["name"] == "iparq"
    assert manifest["version"] == project_version()


def test_plugin_does_not_bundle_an_mcp_server() -> None:
    forbidden_paths = {
        PLUGIN_ROOT / "mcp.json",
        PLUGIN_ROOT / ".mcp.json",
        PLUGIN_ROOT / ".github" / "mcp.json",
    }

    assert not any(path.exists() for path in forbidden_paths)
    for manifest_path in (OPEN_MANIFEST_PATH, CODEX_MANIFEST_PATH):
        manifest = json.loads(manifest_path.read_text())
        assert "mcpServers" not in manifest


def test_codex_plugin_manifest_matches_release() -> None:
    manifest = json.loads(CODEX_MANIFEST_PATH.read_text())

    assert manifest["name"] == PLUGIN_ROOT.name
    assert manifest["version"] == project_version()
    assert manifest["skills"] == "./skills/"
    assert "mcpServers" not in manifest
    assert manifest["interface"]["displayName"] == "iParq"


def test_plugin_skill_matches_canonical_skill() -> None:
    for relative_path in (Path("SKILL.md"), Path("agents/openai.yaml")):
        assert (PLUGIN_SKILL / relative_path).read_bytes() == (
            CANONICAL_SKILL / relative_path
        ).read_bytes()


def test_codex_marketplace_points_to_plugin() -> None:
    marketplace = json.loads(CODEX_MARKETPLACE_PATH.read_text())
    entry = next(entry for entry in marketplace["plugins"] if entry["name"] == "iparq")

    assert marketplace["name"] == "iparq"
    assert entry["source"] == {
        "source": "local",
        "path": "./plugins/iparq",
    }
    assert entry["policy"] == {
        "installation": "AVAILABLE",
        "authentication": "ON_INSTALL",
    }
    assert entry["category"] == "Developer Tools"


def test_copilot_marketplace_points_to_plugin() -> None:
    marketplace = json.loads(COPILOT_MARKETPLACE_PATH.read_text())
    entry = next(entry for entry in marketplace["plugins"] if entry["name"] == "iparq")

    assert marketplace["name"] == "iparq"
    assert marketplace["owner"] == {
        "name": "MiguelElGallo",
        "email": "miguel.zurcher@gmail.com",
    }
    assert marketplace["metadata"]["version"] == project_version()
    assert entry["version"] == project_version()
    assert entry["source"] == "./plugins/iparq"


def tri_state_boolean_fields() -> set[str]:
    """Return ColumnInfo fields typed ``bool | None``.

    These are the fields where ``null`` means "the reader could not determine
    this" rather than "false", so the skill must tell agents not to collapse
    the two.
    """
    fields = set()
    for name, field in ColumnInfo.model_fields.items():
        annotation = field.annotation
        if get_origin(annotation) in (Union, UnionType) and set(
            get_args(annotation)
        ) == {bool, type(None)}:
            fields.add(name)
    return fields


def test_skill_documents_tri_state_boolean_fields() -> None:
    skill = (CANONICAL_SKILL / "SKILL.md").read_text()
    tri_state = tri_state_boolean_fields()

    assert tri_state, "expected ColumnInfo to declare tri-state boolean fields"
    undocumented = sorted(name for name in tri_state if f"`{name}`" not in skill)
    assert not undocumented, (
        "SKILL.md must document these tri-state fields so agents do not read "
        f"null as false: {undocumented}"
    )


def test_skill_documents_semantically_subtle_fields() -> None:
    skill = (CANONICAL_SKILL / "SKILL.md").read_text()
    subtle_fields = ("statistics_num_values", "geo_statistics", "index_page_offset")

    for name in subtle_fields:
        assert name in ColumnInfo.model_fields, (
            f"{name} is no longer a ColumnInfo field"
        )
        assert f"`{name}`" in skill, f"SKILL.md must explain how to interpret {name}"


def test_skill_explains_that_statistics_num_values_excludes_nulls() -> None:
    skill = (CANONICAL_SKILL / "SKILL.md").read_text()
    explaining_lines = [
        line
        for line in skill.splitlines()
        if "`statistics_num_values`" in line and "non-null" in line
    ]

    assert explaining_lines, (
        "the line documenting statistics_num_values must state that it counts "
        "non-null values, so agents do not report it as a row count"
    )

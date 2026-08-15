import json
import re
from pathlib import Path

from jsonschema.validators import validator_for

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

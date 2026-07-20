import json
import re
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).parents[1]
CATALOG_PATH = REPOSITORY_ROOT / ".well-known" / "ai-catalog.json"
URN_PATTERN = re.compile(r"^urn:air:[a-zA-Z0-9.-]+(:[a-zA-Z0-9._-]+)+$")


def test_ai_catalog_entries_are_discoverable() -> None:
    catalog = json.loads(CATALOG_PATH.read_text())

    assert catalog["specVersion"] == "1.0"
    assert catalog["host"]["displayName"]
    assert catalog["entries"]

    for entry in catalog["entries"]:
        assert URN_PATTERN.fullmatch(entry["identifier"])
        assert entry["displayName"]
        assert entry["type"]
        assert ("url" in entry) != ("data" in entry)
        assert 2 <= len(entry["representativeQueries"]) <= 5


def test_cataloged_skill_exists_and_has_matching_identity() -> None:
    catalog = json.loads(CATALOG_PATH.read_text())
    entry = catalog["entries"][0]
    skill_path = REPOSITORY_ROOT / entry["metadata"]["sourcePath"] / "SKILL.md"
    skill = skill_path.read_text()

    assert skill_path.is_file()
    assert entry["type"] == 'text/markdown; profile="urn:air:agent-skills"'
    assert skill.startswith("---\nname: iparq-parquet-inspector\n")
    assert "Use when" in skill.split("---", 2)[1]
    assert entry["version"] == "0.6.0"

import json
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from urllib.parse import urlsplit

REPOSITORY_ROOT = Path(__file__).parents[1]
CATALOG_PATH = REPOSITORY_ROOT / ".well-known" / "ai-catalog.json"
SITE_PATH = REPOSITORY_ROOT / "catalog-site"
URN_PATTERN = re.compile(r"^urn:air:[a-zA-Z0-9.-]+(:[a-zA-Z0-9._-]+)+$")


def test_ai_catalog_entries_are_discoverable() -> None:
    catalog = json.loads(CATALOG_PATH.read_text())

    assert catalog["specVersion"] == "1.0"
    assert catalog["host"]["displayName"]
    assert catalog["host"]["identifier"] == "iparq.dev"
    assert catalog["host"]["documentationUrl"] == "https://iparq.dev/docs/"
    assert catalog["entries"]

    for entry in catalog["entries"]:
        assert URN_PATTERN.fullmatch(entry["identifier"])
        assert entry["identifier"].startswith("urn:air:iparq.dev:")
        assert entry["displayName"]
        assert entry["type"]
        assert ("url" in entry) != ("data" in entry)
        assert 2 <= len(entry["representativeQueries"]) <= 5


def test_cataloged_skill_exists_and_has_matching_identity() -> None:
    catalog = json.loads(CATALOG_PATH.read_text())
    entry = catalog["entries"][0]
    skill_path = REPOSITORY_ROOT / entry["metadata"]["sourcePath"] / "SKILL.md"
    skill = skill_path.read_text()
    project = (REPOSITORY_ROOT / "pyproject.toml").read_text()
    package_init = (REPOSITORY_ROOT / "src" / "iparq" / "__init__.py").read_text()
    version_match = re.search(r'^version = "([^"]+)"$', project, re.MULTILINE)
    package_version_match = re.search(
        r'^__version__ = "([^"]+)"$', package_init, re.MULTILINE
    )

    assert skill_path.is_file()
    assert version_match is not None
    assert package_version_match is not None
    assert entry["type"] == 'text/markdown; profile="urn:air:agent-skills"'
    assert skill.startswith("---\nname: iparq-parquet-inspector\n")
    assert "Use when" in skill.split("---", 2)[1]
    assert package_version_match.group(1) == version_match.group(1)
    assert entry["version"] == version_match.group(1)
    assert (
        f'"softwareVersion": "{version_match.group(1)}"'
        in (SITE_PATH / "index.html").read_text()
    )
    assert entry["trustManifest"]["identity"] == "https://iparq.dev/"


def test_wheel_bundles_the_cataloged_skill(tmp_path: Path) -> None:
    catalog = json.loads(CATALOG_PATH.read_text())
    skill_directory = REPOSITORY_ROOT / catalog["entries"][0]["metadata"]["sourcePath"]
    packaged_directory = "iparq/.agents/skills/iparq-parquet-inspector"

    subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    (wheel_path,) = tmp_path.glob("iparq-*.whl")

    with zipfile.ZipFile(wheel_path) as wheel:
        assert (
            wheel.read(f"{packaged_directory}/SKILL.md")
            == (skill_directory / "SKILL.md").read_bytes()
        )
        assert (
            wheel.read(f"{packaged_directory}/agents/openai.yaml")
            == (skill_directory / "agents" / "openai.yaml").read_bytes()
        )
        record_path = next(
            path for path in wheel.namelist() if path.endswith(".dist-info/RECORD")
        )
        assert f"{packaged_directory}/SKILL.md" in wheel.read(record_path).decode()


def test_catalog_site_has_agent_and_search_discovery_files() -> None:
    homepage = (SITE_PATH / "index.html").read_text()
    llms_txt = (SITE_PATH / "llms.txt").read_text()
    agent_overview = (SITE_PATH / "index.md").read_text()
    pricing = (SITE_PATH / "pricing.md").read_text()
    robots = (SITE_PATH / "robots.txt").read_text()

    assert '<script type="application/ld+json">' in homepage
    assert 'rel="ai-catalog"' in homepage
    assert "without uploading files" in homepage
    assert "https://iparq.dev/docs/" in llms_txt
    assert "uvx --refresh iparq inspect" in agent_overview
    assert "free to install and use" in pricing
    assert "Sitemap: https://iparq.dev/sitemap.xml" in robots


def test_sitemap_uses_canonical_https_urls() -> None:
    sitemap = ET.parse(SITE_PATH / "sitemap.xml")
    namespace = {"sitemap": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    locations = [
        element.text
        for element in sitemap.findall("sitemap:url/sitemap:loc", namespace)
    ]

    parsed_locations = [urlsplit(location) for location in locations if location]
    assert len(parsed_locations) == len(locations)
    assert all(location.scheme == "https" for location in parsed_locations)
    assert all(location.hostname == "iparq.dev" for location in parsed_locations)
    paths = {location.path for location in parsed_locations}
    assert "/" in paths
    assert "/.well-known/ai-catalog.json" in paths

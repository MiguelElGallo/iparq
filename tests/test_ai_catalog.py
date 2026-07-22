import json
import re
import xml.etree.ElementTree as ET
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

    assert skill_path.is_file()
    assert entry["type"] == 'text/markdown; profile="urn:air:agent-skills"'
    assert skill.startswith("---\nname: iparq-parquet-inspector\n")
    assert "Use when" in skill.split("---", 2)[1]
    assert entry["version"] == "0.6.0"
    assert entry["trustManifest"]["identity"] == "https://iparq.dev/"


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

    assert "https://iparq.dev/" in locations
    assert "https://iparq.dev/.well-known/ai-catalog.json" in locations
    parsed_locations = [urlsplit(location) for location in locations if location]
    assert len(parsed_locations) == len(locations)
    assert all(location.scheme == "https" for location in parsed_locations)
    assert all(location.hostname == "iparq.dev" for location in parsed_locations)

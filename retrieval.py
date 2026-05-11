"""
retrieval.py
------------
Loads the catalog once at startup and provides:
  - format_catalog_for_prompt()  → string injected into the system prompt
  - get_valid_urls()             → set used to validate LLM recommendations
  - find_by_name()               → lookup by name for comparison queries
"""

import json
from pathlib import Path
from models import CatalogEntry



_CATALOG_PATH = Path(__file__).parent / "data" / "catalog.json"

_catalog: list[CatalogEntry] = []
_valid_urls: set[str] = set()
_name_index: dict[str, CatalogEntry] = {}


def load_catalog() -> list[CatalogEntry]:
    """
    Load catalog.json into memory. Called ONCE at app startup.
    Raises FileNotFoundError if the catalog has not been scraped yet.
    """
    global _catalog, _valid_urls, _name_index

    if not _CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found at {_CATALOG_PATH}. "
            "Run: python scripts/scrape_catalog.py"
        )

    with open(_CATALOG_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    _catalog = [CatalogEntry(**item) for item in raw]
    _valid_urls = {entry.url for entry in _catalog}
    _name_index = {entry.name.lower(): entry for entry in _catalog}

    return _catalog


def get_catalog() -> list[CatalogEntry]:
    return _catalog


def get_valid_urls() -> set[str]:
    return _valid_urls


def find_by_name(name: str) -> CatalogEntry | None:
    return _name_index.get(name.lower())



def format_catalog_for_prompt(catalog: list[CatalogEntry]) -> str:
    """
    Convert the catalog into a compact, readable string for injection
    into the system prompt. Each assessment is one block.
    """
    lines = []
    for entry in catalog:
        block = [
            f"NAME: {entry.name}",
            f"URL: {entry.url}",
            f"TYPE: {entry.test_type} ({entry.test_type_label})",
        ]
        if entry.description:
            desc = entry.description[:300].strip()
            block.append(f"DESCRIPTION: {desc}")
        if entry.duration:
            block.append(f"DURATION: {entry.duration}")
        if entry.remote_testing:
            block.append("REMOTE: Yes")
        if entry.adaptive:
            block.append("ADAPTIVE/IRT: Yes")
        block.append("---")
        lines.extend(block)

    return "\n".join(lines)



def filter_recommendations_to_catalog(
    recs: list[dict],
    valid_urls: set[str],
) -> list[dict]:
    """
    Remove any recommendation whose URL is not in the scraped catalog.
    Also try to match by name if URL doesn't match exactly.
    """
    from .models import CatalogEntry
    from .retrieval import _name_index
    
    filtered = []
    for r in recs:
        url = r.get("url", "")
        
        if url in valid_urls:
            filtered.append(r)
            continue
            
        name = r.get("name", "")
        if name.lower() in _name_index:
            correct_entry = _name_index[name.lower()]
            r["url"] = correct_entry.url
            r["test_type"] = correct_entry.test_type
            filtered.append(r)
            print(f"Fixed URL for {name}: {correct_entry.url}")
            
    return filtered

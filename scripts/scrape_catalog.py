"""
SHL Catalog Scraper — with retry + resume
------------------------------------------
Scrapes all Individual Test Solutions from:
  https://www.shl.com/products/product-catalog/?start=0&type=1  (32 pages)

Features:
  - Retries on timeout (up to 5 attempts per page)
  - Saves a checkpoint after every page so you can resume if interrupted
  - Enriches each assessment with description from its detail page

Run:
    python scripts/scrape_catalog.py
Output:
    data/catalog.json
"""

import json
import time
import re
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_URL        = "https://www.shl.com"
CATALOG_BASE    = "https://www.shl.com/products/product-catalog/"
PAGE_SIZE       = 12
OUTPUT_PATH     = Path(__file__).parent.parent / "data" / "catalog.json"
CHECKPOINT_PATH = Path(__file__).parent.parent / "data" / "catalog_checkpoint.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "M": "Motivation",
    "P": "Personality & Behaviour",
    "S": "Simulations",
}


def fetch_with_retry(url: str, max_attempts: int = 5, timeout: int = 30) -> str:
    """
    Fetch a URL, retrying on timeout or connection errors.
    Waits 5s, 10s, 20s, 40s between attempts (exponential backoff).
    """
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except (requests.exceptions.Timeout,
                requests.exceptions.ConnectionError) as e:
            wait = 5 * (2 ** (attempt - 1))   # 5, 10, 20, 40 seconds
            if attempt == max_attempts:
                raise RuntimeError(f"Failed after {max_attempts} attempts: {url}") from e
            print(f"    ⚠  Timeout (attempt {attempt}/{max_attempts}), retrying in {wait}s...")
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(f"HTTP error for {url}: {e}") from e


def fetch_page(start: int) -> str:
    url = f"{CATALOG_BASE}?start={start}&type=1"
    return fetch_with_retry(url)


def parse_page(html: str) -> list[dict]:
    """
    The catalog page has two tables:
      tables[0] = Pre-packaged Job Solutions  (skip)
      tables[1] = Individual Test Solutions   (we want)
    Columns: Name | Remote Testing | Adaptive/IRT | Test Type
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return []

    target = tables[1] if len(tables) >= 2 else tables[0]
    results = []

    for row in target.find_all("tr"):
        cols = row.find_all("td")
        if not cols:
            continue

        a_tag = cols[0].find("a")
        if not a_tag:
            continue

        name = a_tag.get_text(strip=True)
        href = a_tag.get("href", "")
        url  = href if href.startswith("http") else BASE_URL + href
        if not name or not url:
            continue

        remote   = len(cols) > 1 and bool(cols[1].find("img"))
        adaptive = len(cols) > 2 and bool(cols[2].find("img"))

        test_types = []
        if len(cols) > 3:
            for token in cols[3].get_text(" ", strip=True).split():
                if token in TEST_TYPE_MAP:
                    test_types.append(token)

        primary = test_types[0] if test_types else ""

        results.append({
            "name":           name,
            "url":            url,
            "test_type":      primary,
            "test_type_label": TEST_TYPE_MAP.get(primary, ""),
            "all_test_types": test_types,
            "description":    "",
            "remote_testing": remote,
            "adaptive":       adaptive,
            "duration":       "",
        })

    return results


def get_last_start(html: str) -> int:
    """Find the highest start= value in pagination links for type=1."""
    soup = BeautifulSoup(html, "html.parser")
    max_start = 0
    for a in soup.find_all("a", href=re.compile(r"start=\d+&type=1")):
        m = re.search(r"start=(\d+)", a["href"])
        if m:
            max_start = max(max_start, int(m.group(1)))
    return max_start


def enrich(assessment: dict) -> dict:
    """Fetch the detail page and extract description + duration."""
    try:
        html = fetch_with_retry(assessment["url"], max_attempts=3, timeout=20)
        soup = BeautifulSoup(html, "html.parser")

        description = ""
        for selector in [
            "div.product-hero__description",
            "div.product-description",
            "div[class*='description']",
            "div[class*='overview']",
            "main p",
        ]:
            el = soup.select_one(selector)
            if el:
                text = el.get_text(" ", strip=True)
                if len(text) > 60:
                    description = text[:500]
                    break

        if not description:
            for p in soup.find_all("p"):
                text = p.get_text(strip=True)
                if len(text) > 80:
                    description = text[:500]
                    break

        assessment["description"] = description

        m = re.search(r"(\d+)\s*min", soup.get_text(" "), re.IGNORECASE)
        if m:
            assessment["duration"] = f"{m.group(1)} minutes"

        time.sleep(0.4)

    except Exception as e:
        print(f"    ⚠  Could not enrich '{assessment['name']}': {e}")

    return assessment


def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"scraped_pages": [], "items": []}


def save_checkpoint(scraped_pages: list[int], items: list[dict]):
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump({"scraped_pages": scraped_pages, "items": items}, f, indent=2)


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("── SHL Catalog Scraper ──────────────────────────────")
    checkpoint = load_checkpoint()
    scraped_pages: list[int] = checkpoint["scraped_pages"]
    all_items: list[dict]    = checkpoint["items"]

    if scraped_pages:
        print(f"  Resuming from checkpoint ({len(scraped_pages)} pages already done, "
              f"{len(all_items)} items collected)")

    # Get total pages from page 0
    print("  Fetching page 1 to detect pagination...")
    first_html  = fetch_page(start=0)
    last_start  = get_last_start(first_html)
    total_pages = (last_start // PAGE_SIZE) + 1 if last_start else 1
    print(f"  Detected {total_pages} pages\n")

    print("Step 1/2  Scraping catalog pages...")
    for page_num in range(total_pages):
        start = page_num * PAGE_SIZE

        if start in scraped_pages:
            print(f"  Page {page_num+1:3d}/{total_pages}  start={start:4d}  → skipped (checkpoint)")
            continue

        print(f"  Page {page_num+1:3d}/{total_pages}  start={start:4d}", end="  →  ", flush=True)
        try:
            html  = first_html if page_num == 0 else fetch_page(start)
            items = parse_page(html)
            print(f"{len(items)} items")
            all_items.extend(items)
            scraped_pages.append(start)
            save_checkpoint(scraped_pages, all_items)
            if page_num > 0:
                time.sleep(0.8)   # polite delay between pages
        except RuntimeError as e:
            print(f"\n  ERROR: {e}")
            print("  Progress saved to checkpoint. Re-run the script to resume.")
            return

    # Deduplicate by URL
    seen: set[str] = set()
    unique = [a for a in all_items if not (a["url"] in seen or seen.add(a["url"]))]
    print(f"\n  Total scraped: {len(all_items)}  |  Unique: {len(unique)}\n")
    existing: dict[str, dict] = {}
    if OUTPUT_PATH.exists():
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for item in json.load(f):
                existing[item["url"]] = item

    print("Step 2/2  Enriching with detail pages...")
    enriched = []
    for i, a in enumerate(unique, 1):
        if a["url"] in existing and existing[a["url"]].get("description"):
            # Already enriched in a previous run — reuse it
            enriched.append(existing[a["url"]])
            print(f"  [{i:3d}/{len(unique)}] {a['name'][:60]}  (cached)")
        else:
            print(f"  [{i:3d}/{len(unique)}] {a['name'][:60]}")
            enriched.append(enrich(a))
            # Save incrementally every 10 items
            if i % 10 == 0:
                with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
                    json.dump(enriched, f, indent=2, ensure_ascii=False)

    # Final save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    print(f"\n✓  Saved {len(enriched)} assessments → {OUTPUT_PATH}")

    # Type breakdown
    counts: dict[str, int] = {}
    for a in enriched:
        for t in (a.get("all_test_types") or ([a["test_type"]] if a["test_type"] else [])):
            counts[t] = counts.get(t, 0) + 1
    print("\nTest type breakdown:")
    for code, n in sorted(counts.items()):
        print(f"  {code}  {TEST_TYPE_MAP.get(code, code):<35} {n}")


if __name__ == "__main__":
    main()
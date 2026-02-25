"""
Scrape curling tournament result-book URLs from curlit.com/results.

Fetches the public results page, parses every tournament row, and writes a
CSV file (``result_urls.csv``) with columns:

    tournament_name, year, location, result_book_url, gender, result_summary_url

``gender`` uses the codes **m** (men), **w** (women), **mx** (mixed),
**mxd** (mixed doubles).

Usage:
    python scrape_results.py [--output-dir <dir>]
"""

import argparse
import csv
import os
import re
import urllib.request

from bs4 import BeautifulSoup

RESULTS_URL = "https://curlit.com/results"
BASE_URL = "https://curlit.com/"


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def fetch_page(url: str) -> str:
    """Return the HTML content of *url* as a string."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_tournament_text(text: str):
    """Extract (tournament_name, year, location) from a tournament description.

    Typical formats:
        "World Men's Curling Championship 2024 in Schaffhausen, Switzerland"
        "European Curling Championships 2022 A-Division in Östersund, Sweden"
        "World Junior-B Curling Championships 2022-2023 in Lohja, Finland"
        "Milano Cortina 2026 Olympic Winter Games, Curling in Cortina, Italy"

    Returns (name, year, location) where *year* is the **first** 4-digit year
    found and *location* is everything after the last `` in `` token (if any).
    """
    text = text.strip()
    # Extract year (first 4-digit number)
    year_match = re.search(r"\b(\d{4})\b", text)
    year = year_match.group(1) if year_match else ""

    # Split on " in " to separate location.  Use the *last* occurrence so that
    # tournament names that contain "in" (e.g. "Curling in Cortina") work.
    parts = text.rsplit(" in ", maxsplit=1)
    if len(parts) == 2:
        name = parts[0].strip()
        location = parts[1].strip()
    else:
        name = text
        location = ""

    return name, year, location


def _gender_from_title(title: str) -> str:
    """Map a link *title* like 'Results Book Men' to a gender code."""
    t = title.lower()
    if "mixed doubles" in t:
        return "mxd"
    if "women" in t:
        return "w"
    if "men" in t:
        return "m"
    return ""


def _gender_from_image(img_src: str) -> str:
    """Infer gender code from the image file name."""
    src = img_src.lower()
    if "pdf_md" in src:
        return "mxd"
    if "pdf_mx" in src:
        return "mx"
    if "pdf_m" in src:
        return "m"
    if "pdf_w" in src:
        return "w"
    return ""


def _gender_from_tournament_name(name: str) -> str:
    """Fallback: infer gender from the tournament name itself."""
    n = name.lower()
    if "mixed doubles" in n:
        return "mxd"
    if "mixed" in n:
        return "mx"
    if "women's" in n or "women " in n:
        return "w"
    if "men's" in n or "men " in n:
        return "m"
    return ""


def _resolve_gender(link_title: str, img_src: str, tournament_name: str) -> str:
    """Return the best gender code we can determine for a single PDF link."""
    g = _gender_from_title(link_title)
    if g:
        return g
    g = _gender_from_image(img_src)
    if g:
        return g
    return _gender_from_tournament_name(tournament_name)


def _make_absolute(href: str) -> str:
    """Turn a relative PDF href into an absolute URL."""
    if href.startswith(("http://", "https://")):
        return href
    return BASE_URL + href


def _extract_links(cell):
    """Return a list of (absolute_url, gender_code) tuples from a <td> cell."""
    links = []
    if cell is None:
        return links
    for a_tag in cell.find_all("a", href=True):
        href = a_tag["href"]
        if "ResultsBook" not in href and "Resultsbook" not in href and "ResultsBook" not in href.replace("_", ""):
            # Also catch "Resultsbook" and other variants
            if "resultsbook" not in href.lower() and "resultssummary" not in href.lower():
                continue
        title = a_tag.get("title", "")
        img = a_tag.find("img")
        img_src = img.get("src", "") if img else ""
        links.append((_make_absolute(href), title, img_src))
    return links


# ------------------------------------------------------------------
# Main scraping logic
# ------------------------------------------------------------------

def scrape_results(html: str):
    """Parse the results page HTML and return a list of row dicts.

    Each dict has keys: tournament_name, year, location, result_book_url,
    gender, result_summary_url.
    """
    # Fix malformed tags (e.g. <tr">) that appear in the live page
    html = re.sub(r'<tr">', "<tr>", html)

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="SearchTable")
    if table is None:
        return []

    rows = []
    for tr in table.find_all("tr"):
        tds = tr.find_all("td")
        if len(tds) < 5:
            continue  # header / spacer row

        tournament_text = tds[0].get_text(strip=True)
        if not tournament_text:
            continue

        name, year, location = parse_tournament_text(tournament_text)

        # Column index 3 = Results Book, 4 = Results Summary
        rb_cell = tds[3]
        rs_cell = tds[4]

        # Gather results-book links
        rb_links = []
        for a_tag in rb_cell.find_all("a", href=True):
            href = a_tag["href"]
            title = a_tag.get("title", "")
            img = a_tag.find("img")
            img_src = img.get("src", "") if img else ""
            # Only include actual PDF result-book links
            if "resultsbook" in href.lower() or "resultbook" in href.lower():
                gender = _resolve_gender(title, img_src, tournament_text)
                rb_links.append((_make_absolute(href), gender, title))

        # Gather results-summary links
        rs_links = []
        for a_tag in rs_cell.find_all("a", href=True):
            href = a_tag["href"]
            title = a_tag.get("title", "")
            img = a_tag.find("img")
            img_src = img.get("src", "") if img else ""
            if "resultssummary" in href.lower() or "resultsummary" in href.lower():
                gender_rs = _resolve_gender(title, img_src, tournament_text)
                rs_links.append((_make_absolute(href), gender_rs))

        if not rb_links:
            continue  # no result-book PDFs for this tournament

        # Build a lookup from gender→summary URL so we can pair them up
        summary_by_gender = {}
        for url, g in rs_links:
            summary_by_gender[g] = url

        for rb_url, gender, _title in rb_links:
            summary_url = summary_by_gender.get(gender, "")
            rows.append({
                "tournament_name": name,
                "year": year,
                "location": location,
                "result_book_url": rb_url,
                "gender": gender,
                "result_summary_url": summary_url,
            })

    return rows


def write_csv(rows, output_path: str):
    """Write rows to a CSV file."""
    fieldnames = [
        "tournament_name",
        "year",
        "location",
        "result_book_url",
        "gender",
        "result_summary_url",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scrape curling result-book URLs from curlit.com/results."
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory where result_urls.csv will be written (default: output).",
    )
    args = parser.parse_args()

    print(f"Fetching {RESULTS_URL} …")
    html = fetch_page(RESULTS_URL)

    print("Parsing tournament data …")
    rows = scrape_results(html)
    print(f"Found {len(rows)} result-book entries.")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "result_urls.csv")
    write_csv(rows, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()

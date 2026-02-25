"""Tests for scrape_results module."""

import csv
import os
import urllib.error
import urllib.request

import pytest

from scrape_results import (
    parse_tournament_text,
    scrape_results,
    write_csv,
    RESULTS_URL,
    _resolve_gender,
    _make_absolute,
)


# ---------------------------------------------------------------------------
# Unit tests (no network required)
# ---------------------------------------------------------------------------


class TestParseTournamentText:
    def test_standard_format(self):
        text = "World Men's Curling Championship 2024 in Schaffhausen, Switzerland"
        name, year, location = parse_tournament_text(text)
        assert name == "World Men's Curling Championship 2024"
        assert year == "2024"
        assert location == "Schaffhausen, Switzerland"

    def test_with_division(self):
        text = "European Curling Championships 2025 A-Division in Lohja, Finland"
        name, year, location = parse_tournament_text(text)
        assert name == "European Curling Championships 2025 A-Division"
        assert year == "2025"
        assert location == "Lohja, Finland"

    def test_hyphenated_year(self):
        text = "World Junior-B Curling Championships 2022-2023 in Lohja, Finland"
        name, year, location = parse_tournament_text(text)
        assert year == "2022"
        assert location == "Lohja, Finland"

    def test_special_format_with_in_in_name(self):
        text = "Milano Cortina 2026 Olympic Winter Games, Curling in Cortina, Italy"
        name, year, location = parse_tournament_text(text)
        assert year == "2026"
        assert location == "Cortina, Italy"

    def test_no_location(self):
        text = "Pacific, European, Mixed Doubles B, Mixed, Junior A+B, Seniors 2020-21"
        name, year, location = parse_tournament_text(text)
        assert year == "2020"
        assert location == ""

    def test_empty_string(self):
        name, year, location = parse_tournament_text("")
        assert name == ""
        assert year == ""
        assert location == ""


class TestResolveGender:
    def test_title_men(self):
        assert _resolve_gender("Results Book Men", "", "") == "m"

    def test_title_women(self):
        assert _resolve_gender("Results Book Women", "", "") == "w"

    def test_title_mixed_doubles(self):
        assert _resolve_gender("Results Book Mixed Doubles", "", "") == "mxd"

    def test_image_men(self):
        assert _resolve_gender("Results Book", "images/PDF_M.png", "") == "m"

    def test_image_women(self):
        assert _resolve_gender("Results Book", "images/PDF_W.png", "") == "w"

    def test_image_mixed_doubles(self):
        assert _resolve_gender("Results Book", "images/PDF_MD.png", "") == "mxd"

    def test_fallback_tournament_name_mens(self):
        assert _resolve_gender("Results Book", "images/PDF.png", "World Men's Curling Championship 2024") == "m"

    def test_fallback_tournament_name_womens(self):
        assert _resolve_gender("Results Book", "images/PDF.png", "World Women's Curling Championship 2024") == "w"

    def test_fallback_tournament_name_mixed_doubles(self):
        assert _resolve_gender("Results Book", "images/PDF.png", "World Mixed Doubles Curling Championship 2024") == "mxd"

    def test_fallback_tournament_name_mixed(self):
        assert _resolve_gender("Results Book", "images/PDF.png", "World Mixed Curling Championship 2023") == "mx"

    def test_generic_no_info(self):
        assert _resolve_gender("Results Book", "images/PDF.png", "Olympic Winter Games 2022") == ""


class TestMakeAbsolute:
    def test_relative(self):
        assert _make_absolute("PDF/WMCC2023_ResultsBook.pdf") == "https://curlit.com/PDF/WMCC2023_ResultsBook.pdf"

    def test_already_absolute(self):
        assert _make_absolute("https://curlit.com/PDF/WMCC2023_ResultsBook.pdf") == "https://curlit.com/PDF/WMCC2023_ResultsBook.pdf"


SAMPLE_HTML = """
<html><body>
<table id="SearchTable">
    <tr class="TRnoHover">
        <th>Season 2024-25</th>
        <th>World Curling</th>
        <th>Live Scores</th>
        <th>Results Book</th>
        <th>Results Summary</th>
    </tr>
    <tr class="TRHover">
        <td>World Men's Curling Championship 2024 in Schaffhausen, Switzerland</td>
        <td align="center">&nbsp;</td>
        <td align="center">&nbsp;</td>
        <td align="center"><a href="PDF/WMCC2024_ResultsBook.pdf" title="Results Book"><img src="images/PDF.png" /></a></td>
        <td align="center"><a href="PDF/WMCC2024_ResultsSummary.pdf" title="Results Summary"><img src="images/PDF.png" /></a></td>
    </tr>
    <tr class="TRHover">
        <td>European Curling Championships 2025 A-Division in Lohja, Finland</td>
        <td align="center">&nbsp;</td>
        <td align="center">&nbsp;</td>
        <td align="center">
            <a href="PDF/ECC2025_ResultsBook_Men_A-Division.pdf" title="Results Book Men"><img src="images/PDF_M.png" /></a>
            &nbsp;
            <a href="PDF/ECC2025_ResultsBook_Women_A-Division.pdf" title="Results Book Women"><img src="images/PDF_W.png" /></a>
        </td>
        <td align="center">
            <a href="PDF/ECC2025_ResultsSummary_Men_A-Division.pdf" title="Results Summary Men"><img src="images/PDF_M.png" /></a>
            &nbsp;
            <a href="PDF/ECC2025_ResultsSummary_Women_A-Division.pdf" title="Results Summary Women"><img src="images/PDF_W.png" /></a>
        </td>
    </tr>
    <tr class="TRHover">
        <td>World Mixed Doubles Qualification Event 2025 in Dumfries, Scotland</td>
        <td align="center">&nbsp;</td>
        <td align="center">&nbsp;</td>
        <td align="center"><a href="PDF/WMDQE2025_ResultsBook.pdf" title="Results Book"><img src="images/PDF.png" /></a></td>
        <td align="center"><a href="PDF/WMDQE2025_ResultsSummary.pdf" title="Results Summary"><img src="images/PDF.png" /></a></td>
    </tr>
    <tr class="TRHover">
        <td>Olympic Qualification Event 2025 in Kelowna, BC, Canada</td>
        <td align="center">&nbsp;</td>
        <td align="center">&nbsp;</td>
        <td align="center">
            <a href="PDF/OQE2025_ResultsBook_Men.pdf" title="Results Book Men"><img src="images/PDF_M.png" /></a>
            &nbsp;
            <a href="PDF/OQE2025_ResultsBook_Women.pdf" title="Results Book Women"><img src="images/PDF_W.png" /></a>
            &nbsp;
            <a href="PDF/OQE2025_ResultsBook_MixedDoubles.pdf" title="Results Book Mixed Doubles"><img src="images/PDF_MD.png" /></a>
        </td>
        <td align="center">
            <a href="PDF/OQE2025_ResultsSummary_Men.pdf" title="Results Summary Men"><img src="images/PDF_M.png" /></a>
            &nbsp;
            <a href="PDF/OQE2025_ResultsSummary_Women.pdf" title="Results Summary Women"><img src="images/PDF_W.png" /></a>
            &nbsp;
            <a href="PDF/OQE2025_ResultsSummary_MixedDoubles.pdf" title="Results Summary Mixed Doubles"><img src="images/PDF_MD.png" /></a>
        </td>
    </tr>
    <tr class="TRHover">
        <td>World Junior Mixed Doubles Curling Championship 2026 in Edmonton, AB, Canada</td>
        <td align="center">&nbsp;</td>
        <td align="center">&nbsp;</td>
        <td align="center">&nbsp;</td>
        <td align="center">&nbsp;</td>
    </tr>
</table>
</body></html>
"""


class TestScrapeResults:
    def test_sample_html(self):
        rows = scrape_results(SAMPLE_HTML)
        # 1 (WMCC) + 2 (ECC men+women) + 1 (WMDQE) + 3 (OQE m+w+mxd) = 7
        assert len(rows) == 7

    def test_wmcc_row(self):
        rows = scrape_results(SAMPLE_HTML)
        wmcc = rows[0]
        assert wmcc["tournament_name"] == "World Men's Curling Championship 2024"
        assert wmcc["year"] == "2024"
        assert wmcc["location"] == "Schaffhausen, Switzerland"
        assert wmcc["result_book_url"] == "https://curlit.com/PDF/WMCC2024_ResultsBook.pdf"
        assert wmcc["gender"] == "m"
        assert wmcc["result_summary_url"] == "https://curlit.com/PDF/WMCC2024_ResultsSummary.pdf"

    def test_ecc_two_genders(self):
        rows = scrape_results(SAMPLE_HTML)
        ecc_rows = [r for r in rows if "European" in r["tournament_name"]]
        assert len(ecc_rows) == 2
        genders = {r["gender"] for r in ecc_rows}
        assert genders == {"m", "w"}

    def test_mixed_doubles_qualification(self):
        rows = scrape_results(SAMPLE_HTML)
        wmdqe = [r for r in rows if "Mixed Doubles Qualification" in r["tournament_name"]]
        assert len(wmdqe) == 1
        assert wmdqe[0]["gender"] == "mxd"

    def test_oqe_three_genders(self):
        rows = scrape_results(SAMPLE_HTML)
        oqe = [r for r in rows if "Olympic Qualification" in r["tournament_name"]]
        assert len(oqe) == 3
        genders = {r["gender"] for r in oqe}
        assert genders == {"m", "w", "mxd"}

    def test_no_result_books_skipped(self):
        """Tournaments without result books should not appear."""
        rows = scrape_results(SAMPLE_HTML)
        names = [r["tournament_name"] for r in rows]
        assert not any("Junior Mixed Doubles" in n for n in names)

    def test_summary_url_paired(self):
        rows = scrape_results(SAMPLE_HTML)
        ecc_men = [r for r in rows if "European" in r["tournament_name"] and r["gender"] == "m"][0]
        assert "ResultsSummary_Men" in ecc_men["result_summary_url"]

    def test_empty_html(self):
        assert scrape_results("<html></html>") == []

    def test_no_table(self):
        assert scrape_results("<html><body><p>No table</p></body></html>") == []


class TestWriteCsv:
    def test_write_and_read(self, tmp_path):
        rows = [
            {
                "tournament_name": "Test Event 2024",
                "year": "2024",
                "location": "Test City, Country",
                "result_book_url": "https://curlit.com/PDF/TEST_ResultsBook.pdf",
                "gender": "m",
                "result_summary_url": "https://curlit.com/PDF/TEST_ResultsSummary.pdf",
            }
        ]
        output_path = str(tmp_path / "result_urls.csv")
        write_csv(rows, output_path)

        assert os.path.isfile(output_path)
        with open(output_path, encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
        assert len(reader) == 1
        assert reader[0]["tournament_name"] == "Test Event 2024"
        assert reader[0]["gender"] == "m"


# ---------------------------------------------------------------------------
# Integration test (requires network)
# ---------------------------------------------------------------------------


def _url_accessible(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        urllib.request.urlopen(req, timeout=10)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


SITE_AVAILABLE = _url_accessible(RESULTS_URL)


@pytest.mark.skipif(not SITE_AVAILABLE, reason="curlit.com not accessible")
class TestLiveScrape:
    def test_scrape_returns_results(self):
        from scrape_results import fetch_page
        html = fetch_page(RESULTS_URL)
        rows = scrape_results(html)
        # The page has many result books; we expect at least 100
        assert len(rows) > 100

    def test_all_rows_have_required_fields(self):
        from scrape_results import fetch_page
        html = fetch_page(RESULTS_URL)
        rows = scrape_results(html)
        for row in rows:
            assert row["tournament_name"]
            assert row["year"]
            assert row["result_book_url"].startswith("https://")
            assert row["gender"] in ("m", "w", "mx", "mxd", "")

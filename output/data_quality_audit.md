# Data Quality Audit — `output/`

**Date:** 2026-03-21

---

## What's Clean ✓

| Check | Result |
|---|---|
| Score arithmetic (before + this = after) | ✅ 0 errors |
| Both teams scoring same end | ✅ 0 |
| End number sequencing | ✅ 0 gaps |
| Shot number sequencing | ✅ 0 gaps |
| Hammer logic (non-scoring team gets hammer next end) | ✅ 0 errors |
| Duplicate keys (any table) | ✅ 0 |
| Accuracy values in [0, 100] | ✅ 0 errors |

---

## Issues Found

### Issue 1 — 171 events in `events.csv` have zero data in all other tables

Every event from 2013–2016 (32 events, IDs 210–260) and 139 further events from 2017–2025 have no rows in `matches`, `ends`, or `shots`. These include B-Division, C-Division, Senior, Wheelchair, Junior-B, Mixed (4-person), Universiade, Youth Olympics, etc.

This is a scraping coverage gap (unsupported PDF format or no PDF linked). Notably, the 2016 events — where a prior run incorrectly treated each end as a separate game (e.g. 220 games of 1 end instead of 22 games of 10 ends) — fall entirely in this blank zone, so no malformed rows exist in the current output; those events simply have no data at all.

---

### Issue 2 — 867 matches have metadata but zero ends or shots

These matches appear in `matches.csv` but have **no rows** in `ends.csv` or `shot_locations.csv`.

| Year | Event | Missing matches |
|------|-------|:-:|
| 2018 | Olympic Winter Games 2018 | 33 |
| 2018 | World Mixed Doubles 2018 | 27 |
| 2018 | Curling World Cup Legs 1–3 + Grand Final | 78 |
| 2019 | World Mixed Doubles 2019 | 16 |
| 2021 | Olympic Qualification 2021 | 20 |
| 2021 | World Mixed Doubles 2021 | 99 |
| 2022 | World Mixed Doubles 2022 | 98 |
| 2022 | Olympic Winter Games 2022 | 49 |
| 2023 | World Mixed Doubles 2023 | 98 |
| 2024 | World Mixed Doubles 2024 | 98 |
| 2025 | Olympic Qualification 2025 | 53 |
| 2025 | World Mixed Doubles 2025 | 99 |
| 2025 | Asian Winter Games 2025 | 18 |
| 2026 | Paralympic Winter Games 2026 | 32 |
| 2026 | Olympic Winter Games 2026 | 49 |

**Likely cause:** World Mixed Doubles is a structurally different format (5 ends, 6 stones per team, power play) which the scraper likely fails silently for. The Olympic/Paralympic events may involve alternative PDF layouts or partial PDFs.

---

### Issue 3 — 427 matches where `final_score` exceeds the last end's cumulative score

**Status: ✅ Root cause identified and fix applied (2026-03-21)**

The final score in `matches.csv` is higher than `ends.score_after` on the last recorded end. 253 of these have both a real final score and ends data; the remaining 174 are World Cup matches with NULL final scores (Issue 4, separate cause).

For the 253: the discrepancy is always 1–4 points, always on exactly one team — the classic fingerprint of a conceded end. When a team concedes during a partial end, that end's PDF page was previously skipped entirely by the `len(shot_images) != 16` guard in `extract_event()`, so its score row was never written to `ends.csv`. The `matches.final_score` (from `parse_score_box` on that same page) correctly captured the final result including the conceded points, but `ends.score_after` stopped at the last *complete* end.

**Fix:** The `len(shot_images) != 16` branch now records the end score row (without shot-level data or `hammer_team_code`) before skipping. See the Investigation Plan section for full details.

---

### Issue 4 — 252 matches have NULL final scores in `matches.csv`

**Status: ✅ Partially fixed (2026-03-21)**

| Sub-issue | Description | Matches | Resolution |
|---|---|:-:|---|
| 4b | WJCC 2026 (event IDs 2 & 3) + WWC 2019 | 5 | ✅ Patched in `matches.csv` |
| 4a | Curling World Cup 2018/19 — matches with ends data | 174 | ✅ Patched in `matches.csv` |
| 4a | Curling World Cup 2018/19 — matches with no ends/shots | 78 | ⚠️ Unresolvable from current data |

**174 matches patched:** Final scores for matches that had `ends.csv` data but NULL `matches.final_score` were derived by taking the `team1_score_after` / `team2_score_after` from the last recorded end (highest `end_number`) for each `(event_id, match_id)` pair, then written back to `matches.csv`.

**Scraper fix applied:** `extract_shot_data.py` now falls back to `last_end["team1_score_after"]` / `team2_score_after` when `parse_score_box()` returns empty scores. The CWC PDFs use a format that omits the `"Total Score"` label expected by `parse_score_box()`, so the final score was never written for any of the 247 CWC matches. Future re-scraping will populate final scores for all CWC matches that have parseable ends data.

**78 matches remain NULL** (all CWC Women's section matches): these entries have `team1_code`, `team2_code`, and `round` in `matches.csv` but zero rows in `ends.csv` or `shot_locations.csv`. `parse_end_line()` succeeded for the first page (the match header was captured), but no end pages were recorded — indicating the Women's section of the CWC PDFs uses a page layout that neither `parse_end_line()` nor `find_shot_pages()` can parse. Without network access to download the PDFs these cannot be diagnosed further.

| Affected event | NULL before fix | Patched from ends | Still NULL |
|---|:-:|:-:|:-:|
| CWC 2018/19 Grand Final (event 147) | 75 | 50 | 25 |
| CWC 2018/19 Leg 3 (event 160) | 74 | 50 | 24 |
| CWC 2018/19 Leg 2 (event 165) | 23 | 19 | 4 |
| CWC 2018/19 Leg 1 (event 174) | 75 | 50 | 25 |
| WJCC 2026 + WWC 2019 | 5 | 5 | 0 |
| **Total** | **252** | **174** | **78** |

---

### Issue 5 — 6 shots with corrupted `shot_type` values (parse artifacts)

**Status: ✅ Fixed (2026-03-21)**

Single stray characters from adjacent PDF table cells were appended to valid shot type names during parsing.

| event_id | match_id | end | shot | `shot_type` (raw) | Correct |
|---|---|---|---|---|---|
| 69 | 19 | 8 | 4 | `"Raise f"` | `Raise` |
| 78 | 27 | 9 | 13 | `"Raise r"` | `Raise` |
| 128 | 70 | 1 | 2 | `"Take-out 4"` | `Take-out` |
| 153 | 5 | 5 | 2 | `"Draw 3"` | `Draw` |
| 181 | 57 | 8 | 12 | `"Draw p"` | `Draw` |
| 181 | 77 | 1 | 7 | `"Guard 4"` | `Guard` |

**Fix applied:** In `_extract_shot_metadata_from_words()`, the token-accumulation loop now skips any continuation token that is a single lowercase letter or single digit — the exact pattern of all 6 corrupted values. All legitimate multi-word shot types ("Hit and Roll", "Wick / Soft Peeling", "Draw Time-out", etc.) use continuation tokens longer than one character. The 6 rows in `shot_locations.csv` and `shot_locations.parquet` were patched directly.

---

### Issue 6 — 28% of ends rows are missing `time_left` (informational)

8,598 end rows (28.1%) have `NULL` for both `team1_time_left` and `team2_time_left`. This is consistent across specific events that either don't record thinking time or whose PDFs don't include a clock column. Not a parsing error, but a coverage gap to be aware of for any thinking-time analysis.

---

### Issue 7 — 7 shots with NULL `turn` (minor)

Only 7 rows out of 488,800 shots are missing the `turn` value. Likely edge-case PDF rows that couldn't be parsed.

---

---

## Investigation & Resolution Plan

Each issue is classified as one of:
- **Fixable** — a code change can resolve or improve it
- **Data gap** — inherent to source data / PDF format; cannot be scraped
- **Needs investigation** — root cause not yet confirmed; steps below required first

---

### Issue 1 — 171 events with zero data

**Classification: Needs investigation → likely Data gap (partial)**

The `scrape_results.py` scraper only picks up PDFs that contain "resultsbook" or "resultbook" in the URL. Events with no data may fall into two categories:

**Step 1 — Check `result_urls.csv` coverage.**
Do the 171 event IDs appear in `result_urls.csv` at all? If not, those events were never scraped. Print the `event_id` values from `events.csv` where no matches exist and see which have corresponding rows in `result_urls.csv`.

**Step 2 — Check whether PDFs actually exist.**
For events that *do* have a URL in `result_urls.csv`, manually open a sample of 3–5 PDFs and verify whether they contain "Game - Shot by Shot" pages. If the PDF exists but uses a different layout (e.g. older pre-2017 format), the scraper's `find_shot_pages()` call will return an empty list and the event is silently skipped.

**Step 3 — Identify the oldest supported PDF format.**
Compare a 2016 PDF (if one exists) against a 2017 PDF visually. If the 2016 format differs, document the layout delta and decide whether to extend the parser.

**Expected outcome:** 2013–2016 events are likely an older PDF format not yet supported. 2017+ events with no data probably have no result-book PDF at all. The 2016 "ends-as-games" bug from a prior run is no longer present in output.

---

### Issue 2 — 867 matches with metadata but no ends or shots

**Classification: Needs investigation → likely partially Fixable**

There are two distinct sub-groups here:

#### 2a — World Mixed Doubles events (~600 matches)
Mixed Doubles uses a different match format: 5 ends (plus potential extra ends), 6 stones per team per end (not 8), and a power play rule. The scraper's logic likely breaks in one or more of these places:
- `group_pages_into_matches()` — page grouping may fail because end counts differ
- Stone-detection (`_detect_stones_in_crop()`) — colour thresholding tuned to 8 stones per team may label images incorrectly
- End-line regex (`parse_end_line()`) — may not match a differently formatted score table

**Step 1** — Open one Mixed Doubles PDF and check whether it contains "Game - Shot by Shot" pages. If not, this is a data gap and the format is simply unsupported.

**Step 2** — If "Game - Shot by Shot" pages exist, add verbose logging to `extract_event()` for a single Mixed Doubles PDF and run it locally to find the first point of failure.

**Step 3** — If fixable, extend the parser to handle 6-stone ends. If not, document as "Mixed Doubles format not supported" and consider it a data gap.

#### 2b — Olympic/Paralympic events (~130 matches) and Asian Winter Games/other
These events have some matches with ends data and some without (a *split* — some pages parsed, others not). This suggests the PDF contains multiple layouts (e.g. draw-sheet pages interspersed with shot pages that have a slightly different header format causing `group_pages_into_matches()` to assign pages to the wrong match or skip them).

**Step 1** — For event_id 4 (2026 Olympics), compare the count of matches with ends data vs without. Pull the list of `match_id` values with no ends, then check what `round` values those matches have in `matches.csv`. If all missing matches share a specific round type (e.g. "Tiebreaker", "Page playoff"), the header format for those pages may differ.

**Step 2** — Open the PDF at those specific round pages and check for any layout differences in the score table or header.

**Step 3** — If a pattern is found, extend `parse_match_header()` or `parse_end_line()` to handle the variant layout.

---

### Issue 3 — 427 matches where `final_score` > last end's `score_after`

**Classification: ✅ Investigated and Fixed**

#### Investigation findings (2026-03-21)

Data analysis conclusively identified the root cause:

1. **Exactly one team always receives the extra points** — in 100% of 253 affected matches (those with both a real final score and ends data), the difference is non-zero for exactly one team (XOR holds universally). This is the unmistakable signature of a conceded end.
2. **The last recorded end always has exactly 16 shots** — the `len(shot_images) != 16` guard in `extract_event()` was silently discarding a *subsequent* partial end entirely, not truncating the last recorded one.
3. **Affected matches have their max recorded end at or above the event median** — 201/253 cases. These are not short games; they played out normally and then had a partial final end.
4. **Conceded points go to the winning team in 200/253 cases** (79%) — the remaining 53 are where the winning team (with a large lead) conceded remaining shots to the losing team, still winning overall. Both are standard concession scenarios.
5. **The scale is correct for concessions**: 1–4 points, with 1 point most common (170 matches).

The remaining 174 of the original 427 are the World Cup matches with NULL final scores — a separate root cause covered in Issue 4.

#### Root cause

When a team concedes during an end, the partial end's PDF page:
- Contains a valid `"Game - Shot by Shot"` header → passes `find_shot_pages()`, included in `match_pages`
- Contains a valid "End N …" score line → `parse_end_line()` correctly reads the conceded-end score
- Contains a "Total Score" box showing the authoritative final result → `parse_score_box()` reads this correctly into `matches.final_score`
- But has **fewer than 16 shot diagram images** → `len(shot_images) != 16` fired, and the entire page was previously `continue`-d, discarding the end score row entirely

The result: `matches.final_score` = correct final result; `ends.score_after` on the last recorded end = score *before* the partial/conceded end.

#### Fix applied

In `extract_event()`, when `len(shot_images) != 16`, the code now still appends the end score summary row to `ends_rows` before `continue`-ing past shot processing. The `hammer_team_code` is left blank for these partial-end rows since shot 16 was never thrown. Shot-level data for the partial end remains unrecorded (correct — positions cannot be assigned without 16 diagrams).

**File changed:** `extract_shot_data.py` (~line 596).

After re-scraping, the `final_score` vs `ends.score_after` mismatch should be eliminated for these 253 matches. The ~174 World Cup matches remain unaffected (Issue 4 has a separate cause).

---

### Issue 4 — 252 matches with NULL final scores

**Classification: ✅ Investigated and Partially Fixed**

#### Investigation findings (2026-03-21)

Root cause confirmed: `parse_score_box()` looks for the string `"Total Score"` in the last end's page text. CWC PDFs omit this label, so the function returns an empty dict and `final_score` is written as `""` (converted to NULL).

**Fix 1 — Scraper fallback (forward-looking):** When `parse_score_box()` returns empty scores but `parse_end_line()` returns a valid `last_end`, the code now falls back to `last_end["team1_score_after"]` / `["team2_score_after"]`. This is mathematically equivalent: the last end's cumulative score is the final score. **File changed:** `extract_shot_data.py` (~line 576).

**Fix 2 — CSV patch (retroactive):** For the 174 CWC matches and 5 WJCC/WWC matches that had ends data but NULL final scores, the final scores were derived from `ends.csv` and written directly to `matches.csv` via `issue4_analysis.ipynb`.

#### 78 still-NULL matches — root cause

These are exclusively the Women's section matches from the four CWC PDFs:

| Event | Women's matches still NULL |
|---|:-:|
| CWC Grand Final 2018/19 | 25 |
| CWC Leg 3 2018/19 | 24 |
| CWC Leg 2 2018/19 | 4 |
| CWC Leg 1 2018/19 | 25 |

Each CWC PDF covers both Men's and Women's competition. The Men's section pages (50 per event) parsed correctly. The Women's section has `team1_code` and `team2_code` captured in `matches.csv` (confirming first-page parsing succeeded), but zero rows in `ends.csv` or `shot_locations.csv` — meaning `parse_end_line()` or `find_shot_pages()` saw no parseable shot pages. The Women's pages likely use a layout variant (e.g. different header or "Game - Shot by Shot" label) that the scraper silently skips. Diagnosis requires downloading the CWC PDFs.

**Next step (when network available):** Download `CWC2018-19_Leg1_ResultsBook.pdf`, open the Women's section pages in pdfplumber, and compare the extracted text against `_END_LINE_RE` and the `"Game - Shot by Shot"` sentinel.

---

### Issue 5 — 6 corrupted `shot_type` values

**Status: ✅ Fixed (2026-03-21)**

The corruption pattern was single lowercase letters (`f`, `r`, `p`) or single digits (`3`, `4`) appended to a valid base type by the PDF word-accumulation loop in `_extract_shot_metadata_from_words()`. All legitimate multi-word continuation tokens are longer than one character.

**Fix:** Added a guard in the token-accumulation `else` branch:
```python
if not (len(text) == 1 and (text.islower() or text.isdigit())):
    shot["shot_type"] += " " + text
```
The 6 affected rows in `shot_locations.csv` and `shot_locations.parquet` were patched directly.

---

### Issue 6 — 28% of ends rows missing `time_left`

**Classification: Data gap**

`parse_score_box()` only finds `time_left` if the PDF page contains the text "Time left". Many events (particularly earlier years and non-WCF-flagship events) do not record thinking time in the PDF at all.

**No code fix possible.** Document this as a structural coverage gap. Time-based analysis should filter to events with non-NULL `time_left` values. Adding an `event_has_time_data` flag to `events.csv` (derived from the null rate per event) could make this easier to filter.

**Investigation worth doing:** Group the NULL rate by year and event type to see if there's a clear cutoff year after which time data becomes reliable. This would be useful metadata for consumers of the dataset.

---

### Issue 7 — 7 shots with NULL `turn`

**Classification: Needs investigation → likely Data gap**

In `_extract_shot_metadata_from_words()`, `turn` is set by matching specific Unicode characters (↺/↻) or the literal "-" (Not considered). NULL means none of these were found at the expected position.

**Step 1** — Retrieve the 7 rows and find their originating PDFs. Check the raw PDF image for those shots to confirm whether a turn indicator is present.

**Step 2** — If the turn symbol is present but in an unexpected position, adjust the y-range used when scanning for turn symbols. If the PDF simply omits the turn indicator, this is a data gap.

Given only 7 rows out of 488,800, this is very low priority.

---

### Summary Table

| Check | Result |
|---|---|
| Score arithmetic (before + this = after) | ✅ 0 errors |
| Both teams scoring same end | ✅ 0 |
| End number sequencing | ✅ 0 gaps |
| Shot number sequencing | ✅ 0 gaps |
| Hammer logic | ✅ 0 errors |
| Duplicate keys (any table) | ✅ 0 |
| Accuracy in [0, 100] | ✅ 0 errors |
| Events with no data at all | ⚠️ 171 events |
| Matches with no detail data (ends/shots) | ⚠️ 867 matches |
| Final score vs ends cumulative mismatch | ⚠️ 427 matches |
| NULL final scores | ⚠️ 252 matches |
| Corrupted `shot_type` (parse artifacts) | ⚠️ 6 rows |
| NULL `time_left` | ℹ️ 28% of ends |
| NULL `turn` | ℹ️ 7 shots |

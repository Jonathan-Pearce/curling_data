# Curling Data

Extract shot-by-shot curling data from tournament PDF result books.

## Overview

This project scrapes and processes curling tournament PDF result books to build a structured, analysis-ready dataset covering every shot thrown in every end of every match. Data is sourced from [curlit.com](https://curlit.com/results), which hosts result books for World Curling Federation events dating back to 2013.

Key outputs:
- **Match metadata** (date, round, teams, final scores)
- **End metadata** (end number, scores, hammer team, thinking time remaining)
- **Shot metadata** (player, shot type, turn direction, accuracy percentage)
- **Stone positions** detected via OpenCV colour segmentation, normalised relative to the house centre
- **Stone tracking** across shots within an end via stable stone IDs and previous-position columns

PDFs are fetched directly from URLs at runtime — no large PDF files need to be stored in the repository. Multiple events are processed together, producing unified data tables with an `event_id` to distinguish between tournaments.

## Requirements

- Python 3.10+
- Dependencies listed in `requirements.txt`

## Installation

```bash
pip install -r requirements.txt
```

## Project Structure

```
curling_data/
├── output/                         # Generated data tables
│   ├── result_urls.csv             # Tournament PDF URL index (from scrape_results.py)
│   ├── events.csv
│   ├── matches.csv
│   ├── teams.csv
│   ├── players.csv
│   ├── ends.csv
│   └── shot_locations.parquet      # Primary shot data (binary; CSV is gitignored)
├── tests/
│   ├── eval_manifest.json          # PDF evaluation manifest for CI
│   ├── test_extract_shot_data.py
│   ├── test_generate_board_image.py
│   ├── test_scrape_results.py
│   ├── test_evaluate_detection.py
│   ├── test_track_stones.py
│   ├── test_build_features.py
│   └── test_verify_stone_tracking.py
├── docs/                           # Reference documentation and development notes
│   ├── data_quality_audit.md       # Data quality findings and re-scrape verification
│   ├── scraping_improvements.md    # Proposals for pipeline improvements
│   ├── stone_location_accuracy_improvements.md
│   └── tracking_verification_guide.md
├── src/                            # Python packages (add src/ to PYTHONPATH for CLI use)
│   ├── scraping/                   # PDF scraping and stone detection
│   │   ├── extract_shot_data.py    # Main extraction script
│   │   ├── scrape_results.py       # Scrapes curlit.com for tournament PDF URLs
│   │   └── evaluate_detection.py   # Stone detection accuracy evaluation
│   ├── tracking/                   # Stone tracking across shots
│   │   ├── track_stones.py         # Stone tracking across shots within an end
│   │   └── verify_stone_tracking.py  # Post-scrape stone tracking validation
│   ├── ml/
│   │   └── build_features.py       # Enriches shot data with game-context columns
│   └── shared/
│       └── generate_board_image.py # Board image generation for data QA
├── ci_scripts/
│   └── evaluate_all.py             # Runs evaluate_detection on all manifested PDFs
├── example raw data/               # Sample PDFs used by integration tests and CI
├── investigations/                 # Exploratory analysis notebooks
├── .github/workflows/
│   ├── ci.yml                      # Unit tests + detection accuracy evaluation
│   └── extract_shot_data.yml       # Manual/scheduled data extraction workflow
├── conftest.py                     # pytest root marker (pythonpath set in pyproject.toml)
├── pyproject.toml                  # Pytest config: pythonpath = ["src"]
├── requirements.txt
└── README.md
```

## Scraping Tournament URLs

Before running the main extraction, scrape curlit.com to build the index of all available result-book PDFs:

```bash
python src/scrape_results.py
```

This writes `output/result_urls.csv` with columns: `tournament_name`, `year`, `location`, `result_book_url`, `gender`, `result_summary_url`. Gender codes: `m`, `w`, `mx`, `mxd`.

## Usage

Process all PDFs in `output/result_urls.csv` (the default bulk mode — run `scrape_results.py` first):

```bash
python src/extract_shot_data.py
```

Process specific PDFs by URL:

```bash
python src/extract_shot_data.py https://curlit.com/PDF/ECC2025_ResultsBook_Men_A-Division.pdf --output-dir output
```

Process multiple PDFs (local paths and/or URLs):

```bash
python src/extract_shot_data.py https://curlit.com/PDF/ECC2025_ResultsBook_Men_A-Division.pdf https://curlit.com/PDF/WMCC2023_ResultsBook.pdf --output-dir output
```

## Verifying Stone Tracking

After a scrape run, validate that the sequential stone-tracking data is internally consistent:

```bash
python src/verify_stone_tracking.py output/shot_locations.parquet
```

Filter to a specific event or match:

```bash
python src/verify_stone_tracking.py output/shot_locations.parquet --event 1 --match 2
```

Generate displacement-arrow board images for manual comparison against the original PDFs:

```bash
python src/verify_stone_tracking.py output/shot_locations.parquet --visualise --output-dir verify_out
```

The script runs 7 automated checks covering schema presence, first-shot invariants, stone ID uniqueness and continuity, displacement plausibility, and new-stone counts per shot.

## Detection Accuracy Evaluation

`evaluate_detection.py` measures how accurately the stone detection pipeline extracts positions from the original PDF crops, without requiring manual annotation.

For each shot crop in a PDF it runs both a permissive ground-truth blob extraction (loose area filter, no deduplication) and the production pipeline, then matches the two sets and reports precision, recall, F1, and median centroid error.  Active stones and ghost rings are evaluated separately.

Quick smoke-test on the first 3 ends of a local PDF:

```bash
python src/scraping/evaluate_detection.py "example raw data/ECC2025_ResultsBook_Men_A-Division.pdf" --max-ends 3
```

Or directly from a URL:

```bash
python src/scraping/evaluate_detection.py https://curlit.com/PDF/ECC2025_ResultsBook_Men_A-Division.pdf --max-ends 3
```

Full evaluation with per-shot CSV output:

```bash
python src/scraping/evaluate_detection.py https://curlit.com/PDF/ECC2025_ResultsBook_Men_A-Division.pdf --csv-out eval_out/results.csv
```

Save annotated overlay images (green circles = TP, orange cross = FP, red cross = FN):

```bash
python src/scraping/evaluate_detection.py "example raw data/ECC2025_ResultsBook_Men_A-Division.pdf" --max-ends 2 --overlay-dir eval_out/overlays/
```

Example summary output:

```
========================================================
  Detection Evaluation — ECC2025_ResultsBook_Men_A-Division.pdf
========================================================
  Ends evaluated        : 167
  Shots evaluated       : 2672

  ── Active stones ──────────────────────────────────
  Ground-truth blobs    :  12 840
  Detected              :  12 815
  True Positives        :  12 790
  False Positives (FP)  :      25  (hallucinated)
  False Negatives (FN)  :      50  (missed)
  Precision             :   99.80 %
  Recall                :   99.61 %
  F1                    :   99.70 %
  Position error median :  0.0030 norm. units
  Position error p95    :  0.0120 norm. units
  Shots with FP         :      18
  Shots with FN         :      42

  ── Ghost rings ────────────────────────────────────
  Ground-truth ghosts   :     847
  Ghost Precision       :   99.20 %
  Ghost Recall          :   96.70 %
========================================================
```

The function can also be used programmatically:

```python
from scraping.evaluate_detection import evaluate_pdf, print_summary

summary, shot_records = evaluate_pdf("path/to/event.pdf", max_ends=5)
print_summary(summary)
```

## Board Image Generation

Recreate curling board images from the scraped data for data quality verification. This renders a single shot's stone positions onto a curling house diagram, allowing visual comparison against the original PDF boards.

Generate a board image for a specific shot:

```bash
python src/shared/generate_board_image.py --event 1 --match 1 --end 7 --shot 12 -o board.png
```

The function can also be used programmatically:

```python
from shared.generate_board_image import generate_board_image, generate_board_image_from_parquet

# From the parquet file
img = generate_board_image_from_parquet("output/shot_locations.parquet", event_id=1, match_id=1, end_number=7, shot_number=12)
img.save("board.png")

# From a shot data dict (e.g. a row from shot_locations.parquet)
img = generate_board_image(shot_data)
img.show()
```

## CI / Quality Tracking

`.github/workflows/ci.yml` runs two jobs on every push or PR that touches `src/`, `tests/`, or `pyproject.toml`:

### Job 1 — Unit Tests

Runs the full pytest suite (no network required). All tests must pass for the PR to merge.

```bash
# Locally equivalent:
pytest
```

### Job 2 — Detection Accuracy Evaluation (report-only)

Evaluates stone detection accuracy against any PDFs committed to `example raw data/`, then uploads a `metrics.json` artifact. **This job never fails the build** — it exists purely to track quality trends over time.

The manifest of PDFs to evaluate lives in [`tests/eval_manifest.json`](tests/eval_manifest.json). PDFs are not stored in the repo (binary size); add them manually:

| File | URL |
|------|-----|
| ECC2025_ResultsBook_Men_A-Division.pdf | https://curlit.com/PDF/ECC2025_ResultsBook_Men_A-Division.pdf |
| WMCC2023_ResultsBook.pdf | https://curlit.com/PDF/WMCC2023_ResultsBook.pdf |
| WWCC2024_ResultsBook.pdf | https://curlit.com/PDF/WWCC2024_ResultsBook.pdf |
| WMCC2024_ResultsBook.pdf | https://curlit.com/PDF/WMCC2024_ResultsBook.pdf |
| WJCC2025_ResultsBook_Men.pdf | https://curlit.com/PDF/WJCC2025_ResultsBook_Men.pdf |
| ECC2024_ResultsBook_Women_A-Division.pdf | https://curlit.com/PDF/ECC2024_ResultsBook_Women_A-Division.pdf |
| WMCC2022_ResultsBook.pdf | https://curlit.com/PDF/WMCC2022_ResultsBook.pdf |
| PCCC2024_ResultsBook_Men_A-Division.pdf | https://curlit.com/PDF/PCCC2024_ResultsBook_Men_A-Division.pdf |
| WMCC2021_ResultsBook.pdf | https://curlit.com/PDF/WMCC2021_ResultsBook.pdf |
| ECC2022_ResultsBook_Men_A-Division.pdf | https://curlit.com/PDF/ECC2022_ResultsBook_Men_A-Division.pdf |

Run the evaluation locally across all present PDFs:

```bash
python ci_scripts/evaluate_all.py
```

Or limit to a quick 2-end smoke-test on each:

```bash
python ci_scripts/evaluate_all.py --max-ends 2
```

## Output Tables

The extraction script writes six tables to the output directory. Five are plain CSV files; the sixth (`shot_locations`) is written as both a CSV (gitignored) and a Parquet file for efficient loading.

### `events.csv`
| Column | Description |
|--------|-------------|
| event_id | Unique event identifier |
| event_name | Name derived from the PDF filename |
| year | Year of the event |
| location | Location of the event |
| gender | Gender category (`m`, `w`, `mx`, `mxd`) |
| pdf_file | Source PDF filename |
| has_time_data | `True` if thinking-time data is present in the PDF for this event |

### `matches.csv`
| Column | Description |
|--------|-------------|
| event_id | Event identifier |
| match_id | Unique match identifier (per event) |
| date | Match date |
| round | Round name (e.g., "Gold Medal Game", "Round Robin Session 1 - Sheet A") |
| start_time | Scheduled start time |
| team1_code | Three-letter code for team 1 |
| team2_code | Three-letter code for team 2 |
| team1_final_score | Final score for team 1 |
| team2_final_score | Final score for team 2 |

### `teams.csv`
| Column | Description |
|--------|-------------|
| event_id | Event identifier |
| team_code | Three-letter team code |
| team_name | Full team/country name |
| player1_name … playerN_name | Player names on the roster |

### `players.csv`
| Column | Description |
|--------|-------------|
| player_id | Unique player identifier |
| event_id | Event identifier |
| team_code | Team the player belongs to |
| player_name | Player name as it appears in the PDF |

### `ends.csv`
| Column | Description |
|--------|-------------|
| event_id | Event identifier |
| match_id | Match identifier |
| end_number | End number (1-based) |
| team1_code, team2_code | Team codes |
| team1_score_before, team2_score_before | Cumulative score before this end |
| team1_score_this_end, team2_score_this_end | Points scored in this end ("X" if end not completed) |
| team1_score_after, team2_score_after | Cumulative score after this end |
| hammer_team_code | Team with last-stone advantage (hammer) |
| team1_time_left, team2_time_left | Thinking time remaining |

### `shot_locations` (parquet)

> **Note:** `shot_locations.parquet` is the primary output. A parallel `shot_locations.csv` is also written for convenience but is gitignored due to its size.
| Column | Description |
|--------|-------------|
| event_id | Event identifier |
| match_id | Match identifier |
| end_number | End number |
| shot_number | Shot number within the end (1–16) |
| team_code | Team delivering the shot |
| player_id | Player identifier |
| player_name | Player name |
| shot_type | Shot type (e.g., Draw, Take-out, Guard, Hit and Roll) |
| turn | Turn direction (Clockwise, Counter-clockwise, Not considered) |
| accuracy | Shot accuracy percentage |
| house_orientation | Which half of the shot image contains the house (`top` or `bottom`) |
| team1_stones_in_play | Number of team 1 stones in play after this shot |
| team2_stones_in_play | Number of team 2 stones in play after this shot |
| team1_stone1_x … team1_stone8_x | Normalised x-coordinate for each team 1 stone |
| team1_stone1_y … team1_stone8_y | Normalised y-coordinate for each team 1 stone |
| team1_stone1_dist … team1_stone8_dist | Distance from house centre (1.0 = 12-foot ring) |
| team1_stone1_angle … team1_stone8_angle | Angle in degrees from house centre |
| team1_stone1_id … team1_stone8_id | Stable integer ID tracking each stone across all shots in an end |
| team1_stone1_prev_x … team1_stone8_prev_x | x-coordinate of this stone on the previous shot (NULL if newly placed) |
| team1_stone1_prev_y … team1_stone8_prev_y | y-coordinate of this stone on the previous shot (NULL if newly placed) |
| team2_stone1_x … team2_stone8_x | Same position columns for team 2 stones |
| team2_stone1_id … team2_stone8_id | Same tracking ID columns for team 2 stones |
| team2_stone1_prev_x … team2_stone8_prev_x | Same previous-position columns for team 2 stones |
| team2_stone1_prev_y … team2_stone8_prev_y | Same previous-position columns for team 2 stones |

Stone positions are normalised so the 12-foot ring has radius 1.0. Stones are ordered ascending by distance from the button (house centre). Empty values indicate the stone is not in play.

Stone tracking: each stone receives a stable `_id` integer at the start of an end that persists across every subsequent shot. `_prev_x` / `_prev_y` record where that stone was on the prior shot, enabling displacement analysis and trajectory reconstruction.

## Coordinate System

Stone positions use a Cartesian coordinate system centred on the button (house centre):
- **x**: horizontal offset (positive = right)
- **y**: vertical offset (positive = toward the hack/away from centre)
- **dist**: Euclidean distance from centre (1.0 = edge of 12-foot ring)
- **angle**: angle in degrees (0° = right, 90° = toward hack, −90° = toward far end)

Both Cartesian (x, y) and polar (dist, angle) representations are included to support different analysis needs.

## Future Considerations

- **Database backend**: Migrate from flat CSV/Parquet files to a relational database (e.g., SQLite or PostgreSQL) for better querying and referential integrity.
- **Incremental processing**: Skip PDFs that have already been processed, supporting append-only workflows for newly published events.
- **Configuration file**: Use a YAML/JSON config to supply per-event metadata (name, gender, year) alongside each PDF path, avoiding reliance on filename parsing.
- **Ghost stone recording**: Record the outline-ring contours (displaced stone ghosts visible in PDF diagrams) as separate columns to provide displacement-origin features for downstream analysis.
- **Mixed-gender events**: Some event PDFs contain both men's and women's draws in a single file. Add logic to split and label them independently.
- **ML pipeline** (see [Issue #11](https://github.com/Jonathan-Pearce/curling_data/issues/11)): Implement the proposed Siamese GNN model for multi-task shot prediction (end score, accuracy, shot type) using the scraped dataset.

---

## Project Status

### Completed

| Issue | Description |
|-------|-------------|
| [#1](https://github.com/Jonathan-Pearce/curling_data/issues/1) | Core PDF scraping: match/end/shot extraction from result books |
| [#3](https://github.com/Jonathan-Pearce/curling_data/issues/3) | Multi-event support with `event_id` and `events.csv` |
| [#5](https://github.com/Jonathan-Pearce/curling_data/issues/5) | URL-based PDF ingestion; no raw PDFs stored in the repo |
| [#7](https://github.com/Jonathan-Pearce/curling_data/issues/7) | `scrape_results.py` — automated tournament URL index from curlit.com |
| [#9](https://github.com/Jonathan-Pearce/curling_data/issues/9) | Bulk extraction of all events from `result_urls.csv` |
| [#13](https://github.com/Jonathan-Pearce/curling_data/issues/13) | `generate_board_image.py` — recreate board diagrams for visual QA |
| Data quality | `has_time_data` flag in `events.csv`; ghost-stone filtering; `turn = "Not considered"` for Through shots |
| Stone tracking | Sequential stone-ID assignment and `prev_x`/`prev_y` propagation across shots within an end |
| Accuracy improvements | Dynamic house-radius detection, adaptive ring-colour thresholds, per-page stone-colour calibration |

### Open

| Issue | Description |
|-------|-------------|
| [#11](https://github.com/Jonathan-Pearce/curling_data/issues/11) | **ML pipeline** — Siamese GNN for multi-task shot prediction (not yet started) |
| [#15](https://github.com/Jonathan-Pearce/curling_data/issues/15) | **General improvements** — ghost stone columns, code modularisation, final scraping review |

---

## Recommended Next Steps

The data pipeline is mature and the dataset covers ~260 World Curling events (2013–2026). The natural next step is building the **ML pipeline** described in [Issue #11](https://github.com/Jonathan-Pearce/curling_data/issues/11):

1. **Build a GNN-based shot-prediction model** (Siamese architecture, multi-task head):
   - Input: board state before and after a shot (`S_{t-1}`, `S_t`) represented as graphs, plus metadata (shot number, end number, score, hammer)
   - Outputs: shot accuracy (regression), shot type (classification), end score (regression)
   - The stable `stone_id` and `prev_x`/`prev_y` columns already in the parquet provide the node-identity continuity a GNN needs

2. **Ghost stone columns** — implement `docs/scraping_improvements.md` Improvement 1 to record displaced-stone origin positions, unlocking richer displacement-vector features.

3. **Incremental scraping** — add skip-if-seen logic so new events can be appended without re-processing the entire URL list.

4. **Mixed-gender PDF splitting** — handle the small subset of PDFs that bundle men's and women's draws, ensuring all events are labelled with the correct `gender` code.

See `docs/` for detailed implementation notes on each of these areas.
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
│   ├── shot_locations_raw.parquet  # Raw detected stone positions (no tracking)
│   └── shot_locations.parquet      # Enriched shot data with tracking columns (CSV gitignored)
├── output_test/                    # Small test dataset used by the test suite
├── tests/
│   ├── test_extract_shot_data.py
│   ├── test_build_features.py
│   ├── test_generate_board_image.py
│   ├── test_scrape_results.py
│   ├── test_track_stones.py
│   └── test_verify_stone_tracking.py
├── docs/                           # Reference documentation and development notes
│   ├── action_plan.md              # Project status review and next steps
│   ├── data_quality_audit.md       # Data quality findings and re-scrape verification
│   ├── gnn_data_readiness.md       # GNN schema readiness assessment
│   ├── pipeline_overview.md        # End-to-end pipeline description
│   ├── scraping_improvements.md    # Scraping improvement proposals and status
│   ├── stone_location_accuracy_improvements.md
│   ├── tracking_evaluation.md      # Tracking evaluation methodologies
│   ├── tracking_metrics.md         # Reference for all tracking quality metrics
│   └── tracking_verification_guide.md
├── investigations/                 # Exploratory analysis scripts and notebooks
├── example raw data/               # Sample PDFs used by integration tests
├── extract_shot_data.py            # Phase 1 — extraction script (raw positions)
├── track_stones.py                 # Phase 2a — stone ID assignment and prev_x/prev_y
├── build_features.py               # Phase 2b — join game-context columns from ends.csv
├── verify_stone_tracking.py        # Phase 2c — post-tracking validation
├── generate_board_image.py         # Board image generation for data QA
├── scrape_results.py               # Scrapes curlit.com for tournament PDF URLs
├── requirements.txt
└── README.md
```

## End-to-End Run Order

### Phase 1 — Scrape tournament URLs

```bash
python scrape_results.py
```

This writes `output/result_urls.csv` with columns: `tournament_name`, `year`, `location`, `result_book_url`, `gender`, `result_summary_url`. Gender codes: `m`, `w`, `mx`, `mxd`.

### Phase 1 — Extract shot data

Process all PDFs in `output/result_urls.csv` (run `scrape_results.py` first):

```bash
python extract_shot_data.py
```

Process specific PDFs by URL:

```bash
python extract_shot_data.py https://curlit.com/PDF/ECC2025_ResultsBook_Men_A-Division.pdf --output-dir output
```

Outputs `output/shot_locations_raw.parquet` (raw stone positions, no tracking).

### Phase 2a — Stone tracking

Assign stable stone IDs and populate `prev_x` / `prev_y` displacement columns:

```bash
python track_stones.py [--method greedy|hungarian]
```

Outputs `output/shot_locations.parquet` (tracking columns added).

### Phase 2b — Feature enrichment

Join game-context columns from `ends.csv` onto the shot table:

```bash
python build_features.py
```

Adds `shooting_team_is_team1`, `hammer_team_code`, `shooting_team_has_hammer`,
`team1_score_before`, `team2_score_before`, `score_diff_before` to
`output/shot_locations.parquet` (overwrites in place).

### Phase 2c — Verify stone tracking

After running `track_stones.py`, validate the tracking data is internally consistent:

```bash
python verify_stone_tracking.py output/shot_locations.parquet
```

Filter to a specific event or match:

```bash
python verify_stone_tracking.py output/shot_locations.parquet --event 1 --match 2
```

Generate displacement-arrow board images for manual comparison against the original PDFs:

```bash
python verify_stone_tracking.py output/shot_locations.parquet --visualise --output-dir verify_out
```

The script runs 7 automated checks covering schema presence, first-shot invariants, stone ID uniqueness and continuity, displacement plausibility, and new-stone counts per shot.

## Board Image Generation

Recreate curling board images from the scraped data for data quality verification. This renders a single shot's stone positions onto a curling house diagram, allowing visual comparison against the original PDF boards.

Generate a board image for a specific shot:

```bash
python generate_board_image.py --event 1 --match 1 --end 7 --shot 12 -o board.png
```

The function can also be used programmatically:

```python
from generate_board_image import generate_board_image, generate_board_image_from_parquet

# From the parquet file
img = generate_board_image_from_parquet("output/shot_locations.parquet", event_id=1, match_id=1, end_number=7, shot_number=12)
img.save("board.png")

# From a shot data dict (e.g. a row from shot_locations.parquet)
img = generate_board_image(shot_data)
img.show()
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

> **Note:** `shot_locations.parquet` is the primary output after running `track_stones.py` and `build_features.py`. The CSV is gitignored due to its size.
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
| team1_stone1_is_shot_stone … team1_stone8_is_shot_stone | 1.0 if this was the delivered stone this shot; 0.0 otherwise; NaN if ambiguous |
| team1_ghosts_in_play | Number of team 1 ghost ring contours detected (displaced-stone origins) |
| team1_ghost1_x … team1_ghost8_x | Normalised x-coordinate of each ghost ring (NULL if none detected) |
| team1_ghost1_y … team1_ghost8_y | Normalised y-coordinate of each ghost ring |
| team2_stone1_x … team2_stone8_x | Same position columns for team 2 stones |
| team2_stone1_id … team2_stone8_id | Same tracking ID columns for team 2 stones |
| team2_stone1_prev_x … team2_stone8_prev_x | Same previous-position columns for team 2 stones |
| team2_stone1_prev_y … team2_stone8_prev_y | Same previous-position columns for team 2 stones |
| team2_stone1_is_shot_stone … team2_stone8_is_shot_stone | Same delivered-stone flag for team 2 stones |
| team2_ghosts_in_play | Number of team 2 ghost ring contours detected |
| team2_ghost1_x … team2_ghost8_x | Same ghost columns for team 2 |
| shooting_team_is_team1 | True when this shot's team_code equals team1_code for the end |
| hammer_team_code | Team with last-stone advantage this end |
| shooting_team_has_hammer | True when the shooting team holds the hammer |
| team1_score_before | Cumulative score for team 1 entering this end |
| team2_score_before | Cumulative score for team 2 entering this end |
| score_diff_before | team1_score_before − team2_score_before |

Stone positions are normalised so the 12-foot ring has radius 1.0. Stones are ordered ascending by distance from the button (house centre). Empty values indicate the stone is not in play.

Stone tracking: each stone receives a stable `_id` integer at the start of an end that persists across every subsequent shot. `_prev_x` / `_prev_y` record where that stone was on the prior shot, enabling displacement analysis and trajectory reconstruction. `_is_shot_stone` identifies the delivered stone on each shot.

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
- **Mixed-gender events**: Some event PDFs contain both men's and women's draws in a single file. Add logic to split and label them independently.
- **ML pipeline** (see [Issue #11](https://github.com/Jonathan-Pearce/curling_data/issues/11)): Implement the proposed Siamese GNN model for multi-task shot prediction (end score, accuracy, shot type) using the scraped dataset.

---

## Project Status

### Completed

| Issue / Area | Description |
|-------|-------------|
| [#1](https://github.com/Jonathan-Pearce/curling_data/issues/1) | Core PDF scraping: match/end/shot extraction from result books |
| [#3](https://github.com/Jonathan-Pearce/curling_data/issues/3) | Multi-event support with `event_id` and `events.csv` |
| [#5](https://github.com/Jonathan-Pearce/curling_data/issues/5) | URL-based PDF ingestion; no raw PDFs stored in the repo |
| [#7](https://github.com/Jonathan-Pearce/curling_data/issues/7) | `scrape_results.py` — automated tournament URL index from curlit.com |
| [#9](https://github.com/Jonathan-Pearce/curling_data/issues/9) | Bulk extraction of all events from `result_urls.csv` |
| [#13](https://github.com/Jonathan-Pearce/curling_data/issues/13) | `generate_board_image.py` — recreate board diagrams for visual QA |
| Data quality | `has_time_data` flag in `events.csv`; ghost-stone filtering; `turn = "Not considered"` for Through shots |
| Stone tracking | `track_stones.py` — sequential stone-ID assignment, `prev_x`/`prev_y` propagation, `is_shot_stone` flag |
| Feature enrichment | `build_features.py` — `shooting_team_is_team1`, `hammer_team_code`, `shooting_team_has_hammer`, score columns |
| Accuracy improvements | Dynamic house-radius detection, adaptive ring-colour thresholds, per-page stone-colour calibration |
| Ghost stone columns | Displaced-stone origin positions captured in `team{N}_ghost{M}_x/y/dist/angle` columns |
| CI/CD | Automated test workflow (`tests.yml`) runs pytest on every push and PR |

### Open

| Issue | Description |
|-------|-------------|
| [#11](https://github.com/Jonathan-Pearce/curling_data/issues/11) | **ML pipeline** — Siamese GNN for multi-task shot prediction |
| Tracking validation | Run full metric suite on production data; implement physics-based delivery check |
| Incremental scraping | Skip already-processed PDFs to reduce re-scrape time |

---

## Recommended Next Steps

The data pipeline is mature and the dataset covers ~260 World Curling events (2013–2026). The natural next step is building the **ML pipeline** described in [Issue #11](https://github.com/Jonathan-Pearce/curling_data/issues/11):

1. **Run tracking evaluation** on the production parquet to confirm `spurious_id_rate` and `id_continuity_rate` are within acceptable bounds before training.

2. **Write `build_graph_dataset.py`** — convert the enriched parquet into PyTorch Geometric `Data` objects (nodes = stones, edges = spatial/temporal links, graph features = game context).

3. **Implement the GNN model** — two-branch Siamese GCN encoder with multi-task output heads (shot accuracy, shot type, end score). See `docs/pipeline_overview.md` for the full architecture.

4. **Incremental scraping** — add skip-if-seen logic so new events can be appended without re-processing the entire URL list.

See `docs/action_plan.md` for the full prioritised action plan.
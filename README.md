# Curling Data

Extract shot-by-shot curling data from tournament PDF result books.

## Overview

This tool parses curling tournament PDF files to extract structured data about every shot in every end of every match, including:

- **Match metadata** (date, round, teams, final scores)
- **End metadata** (end number, scores, hammer team, time remaining)
- **Shot metadata** (player, shot type, turn direction, accuracy percentage)
- **Stone positions** detected via OpenCV colour segmentation, normalised relative to the house centre

Multiple events can be processed together, producing unified data tables with an `event_id` to distinguish between tournaments.

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
├── data/
│   └── pdfs/                  # Tournament PDF source files
│       ├── ECC2025_ResultsBook_Men_A-Division.pdf
│       └── WMCC2023_ResultsBook.pdf
├── output/                    # Generated CSV data tables
│   ├── events.csv
│   ├── matches.csv
│   ├── teams.csv
│   ├── players.csv
│   ├── ends.csv
│   └── shot_locations.csv
├── tests/
│   └── test_extract_shot_data.py
├── extract_shot_data.py       # Main extraction script
├── requirements.txt
└── README.md
```

## Usage

Process a single PDF:

```bash
python extract_shot_data.py data/pdfs/ECC2025_ResultsBook_Men_A-Division.pdf --output-dir output
```

Process multiple PDFs at once:

```bash
python extract_shot_data.py data/pdfs/ECC2025_ResultsBook_Men_A-Division.pdf data/pdfs/WMCC2023_ResultsBook.pdf --output-dir output
```

## Output Tables

The script produces six CSV files in the output directory:

### `events.csv`
| Column | Description |
|--------|-------------|
| event_id | Unique event identifier |
| event_name | Name derived from the PDF filename |
| pdf_file | Source PDF filename |

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

### `shot_locations.csv`
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
| team1_stones_in_play | Number of team 1 stones in play after this shot |
| team2_stones_in_play | Number of team 2 stones in play after this shot |
| team1_stone1_x … team1_stone8_x | Normalised x-coordinate for each team 1 stone |
| team1_stone1_y … team1_stone8_y | Normalised y-coordinate for each team 1 stone |
| team1_stone1_dist … team1_stone8_dist | Distance from house centre (1.0 = 12-foot ring) |
| team1_stone1_angle … team1_stone8_angle | Angle in degrees from house centre |
| team2_stone1_x … team2_stone8_x | Same columns for team 2 stones |

Stone positions are normalised so the 12-foot ring has radius 1.0. Stones are ordered ascending by distance from the button (house centre). Empty values indicate the stone is not in play.

## Coordinate System

Stone positions use a Cartesian coordinate system centred on the button (house centre):
- **x**: horizontal offset (positive = right)
- **y**: vertical offset (positive = toward the hack/away from centre)
- **dist**: Euclidean distance from centre (1.0 = edge of 12-foot ring)
- **angle**: angle in degrees (0° = right, 90° = toward hack, −90° = toward far end)

Both Cartesian (x, y) and polar (dist, angle) representations are included to support different analysis needs.

## Future Considerations

- **Event metadata enrichment**: Add fields like location, competition level, gender category, and date range to `events.csv`.
- **Database backend**: Migrate from flat CSV files to a relational database (e.g., SQLite or PostgreSQL) for better querying and referential integrity.
- **Incremental processing**: Skip PDFs that have already been processed, supporting append-only workflows.
- **Configuration file**: Use a YAML/JSON config to define event metadata (name, gender, year) alongside each PDF path.
- **Automated PDF ingestion**: Watch a directory for new PDFs and process them automatically.
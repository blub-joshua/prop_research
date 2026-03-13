# NBA Player Prop Research Tool

A local, offline-first research tool for analyzing NBA player prop bets using
statistical projection models and expected value (EV) calculations.

**This tool is for personal research use only.** It does not place bets
automatically. All wagering decisions are made manually at licensed Illinois
sportsbooks.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Setup](#setup)
3. [Historical Data Setup](#historical-data-setup)
4. [Daily Workflow](#daily-workflow)
5. [Configuration](#configuration)
6. [Module Reference](#module-reference)
7. [Data Sources](#data-sources)
8. [CLI Reference](#cli-reference)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Daily Workflow                               │
│                                                                     │
│  1. ingest_games.py ──► DuckDB (games, teams)                      │
│  2. ingest_boxscores.py ──► DuckDB (player_game_stats)             │
│  3. ingest_injuries.py ──► DuckDB (injuries)                       │
│         │                                                           │
│         ▼                                                           │
│  4. features.py ──► DuckDB (player_features)                       │
│         │                                                           │
│         ▼                                                           │
│  5. models.py (train/load) ──► models/*.pkl                        │
│         │                                                           │
│         ▼                                                           │
│  6. [Manual] Screenshot prop board → OCR → LLM → JSON/CSV          │
│         │                                                           │
│         ▼                                                           │
│  7. cli.py ──► EV Rankings + Combo Suggestions                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **DuckDB** is used as the local analytical database. It is fast, file-based,
  and requires no server process.
- **scikit-learn** (with optional LightGBM/XGBoost) handles projection models
  for Points, Rebounds, Assists, and 3-Pointers Made.
- **No cloud dependencies** — all data stays local. Ingestion scripts fetch
  from public APIs/websites and write directly to DuckDB.
- **OCR + LLM bridge** — you take screenshots of sportsbook prop boards,
  OCR extracts the text, you paste it into an LLM (e.g. ChatGPT or Claude),
  and the LLM returns structured JSON/CSV which you save as
  `data/today_props_raw.json`.

---

## Setup

### Prerequisites

- Python 3.11+
- Windows 10/11 (Intel CPU, NVIDIA GPU optional for LightGBM GPU mode)
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed and
  on your `PATH` (for the OCR helper)

### Install Tesseract (Windows)

Download the installer from:
https://github.com/UB-Mannheim/tesseract/wiki

After installation, add `C:\Program Files\Tesseract-OCR` to your system PATH,
or set `TESSERACT_CMD` in your `.env` file.

### Python Environment

**Option A — venv (recommended)**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Option B — conda**

```powershell
conda create -n nba_props python=3.11
conda activate nba_props
pip install -r requirements.txt
```

### Initialize the Database

```powershell
python src/cli.py db-init
```

This runs `schema.sql` against `data/nba_props.duckdb` and creates all tables.

### Copy and Edit Config

```powershell
copy .env.example .env
```

Edit `.env` with your preferences (seasons, bankroll, EV threshold, etc.).
Also review `config.yaml` for model and feature knobs.

---

## Historical Data Setup

Run these commands **once** to back-fill multiple NBA seasons.  Each step
builds on the previous one, so run them in order.

### 0. Prerequisites

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Install Tesseract for OCR (optional — only needed for prop screenshots)
# https://github.com/UB-Mannheim/tesseract/wiki

copy .env.example .env    # then edit with your seasons / bankroll
```

### 1. Initialise the database

```powershell
python src/cli.py db-init
# or equivalently:
python -m src.db
```

Creates `data/nba_props.duckdb` with all table schemas.

### 2. Ingest teams and game schedules

```powershell
python src/ingest_games.py
```

Default seasons from `.env` (`SEASONS=2020-21,2021-22,2022-23,2023-24,2024-25`).
To ingest specific seasons:

```powershell
python src/ingest_games.py --seasons 2022-23 2023-24 2024-25
```

**What it does:** Pulls `LeagueGameLog` from the NBA Stats API for Regular Season
and Playoffs.  Populates the `teams` and `games` tables.  Typical runtime:
~1–2 minutes for 5 seasons.

### 3. Ingest player box scores

```powershell
# Initial full back-fill — do in chunks to avoid long-running sessions
python src/ingest_boxscores.py --limit 500
python src/ingest_boxscores.py --limit 500
# ... repeat until the script reports 0 games remaining

# Or ingest all at once (will take 45–90 minutes for 5 seasons
# due to NBA API rate limits of ~0.7s per game)
python src/ingest_boxscores.py
```

**What it does:** For each game not yet in `player_game_stats`, fetches
`BoxScoreTraditionalV2` (PTS, REB, AST, FG3M, MIN, ±/−) and
`BoxScoreAdvancedV2` (USG%, pace, ratings).  Stores one row per player
per game.  Use `--limit N` to process N games per session; re-runs are
idempotent (already-ingested games are skipped).

> **Tip for large back-fills:** Run overnight or in 500-game chunks across
> multiple sessions.  Progress is automatically saved to the DB after each
> game, so you can safely interrupt and resume.

### 4. Ingest injuries (optional for back-testing)

```powershell
# Option A — today's injury report from Rotowire (live use)
python src/ingest_injuries.py --source rotowire

# Option B — ESPN fallback
python src/ingest_injuries.py --source espn

# Option C — load a manual CSV of historical injuries
#   (see data/injuries_manual_template.csv for the schema)
python src/ingest_injuries.py --source csv
```

For initial back-testing without injury data, you can skip this step —
the model will treat all players as "Active" (injury_severity = 0).

### 5. Build player features

```powershell
python src/features.py
# or for specific seasons:
python src/features.py --seasons 2022-23 2023-24 2024-25
```

**What it does:** Reads `player_game_stats` and `games`, computes rolling
averages (L5, L10), season-to-date averages, H/A splits, opponent defensive
allowed-stats (L10), rest/schedule context, and pace.  Writes results to
`player_features`.  Saves a snapshot CSV to `data/player_features_snapshot.csv`.

Typical runtime: 2–5 minutes for 5 seasons on an i7.

### 6. Train projection models

```powershell
python src/models.py
# or with LightGBM:
python src/models.py --backend lightgbm
# or force retraining:
python src/models.py --force-retrain
```

**What it does:** Trains one model per stat (PTS, REB, AST, 3PM) using the
`player_features` table.  The most recent season is held out as the
validation set.  Metrics (RMSE, MAE) are logged and saved to
`models/eval/training_metrics.csv`.  Residual plots are saved to
`models/eval/{target}_eval.png`.  Feature importances go to
`models/eval/{target}_feature_importance.csv`.

Models are saved as:
```
models/points_model.pkl
models/rebounds_model.pkl
models/assists_model.pkl
models/threepm_model.pkl
```

### 7. Run the synthetic backtest (sanity check)

```powershell
python src/backtest.py --mode synthetic
```

Simulates prop lines by offsetting each projection by ±k × std.  Runs the
full EV pipeline and reports win rates and P&L at various EV thresholds.
Results saved to `data/backtest_results.csv`.

To import real historical lines (e.g., from a CSV you've built up):

```powershell
python src/backtest.py --import-csv data/your_historical_lines.csv
python src/backtest.py --mode real --ev-threshold 0.03
```

### Full pipeline — one-liner order

```powershell
python src/cli.py db-init
python src/ingest_games.py
python src/ingest_boxscores.py --limit 500   # repeat until done
python src/ingest_injuries.py --source auto
python src/features.py
python src/models.py
python src/backtest.py --mode synthetic
```

### Expected disk usage

| Data | Approx size |
|---|---|
| 5 seasons of games (~6 000 games) | < 1 MB |
| 5 seasons of box scores (~180 000 player-game rows) | ~40 MB |
| player_features table | ~60 MB |
| Trained model .pkl files (4×) | ~20–80 MB each (LightGBM) |
| Total DuckDB file | ~150–250 MB |

---

## Daily Workflow

### Step 1 — Update Historical Data

```powershell
# Pull schedule + results for configured seasons
python src/ingest_games.py

# Pull player box scores
python src/ingest_boxscores.py

# Pull / update injury report
python src/ingest_injuries.py
```

These scripts are idempotent — re-running them will upsert rather than
duplicate rows.

### Step 2 — Rebuild Features and Retrain Models

```powershell
# Compute rolling/aggregate features into player_features table
python src/features.py

# Train (or load cached) projection models
python src/models.py
```

Models are saved under `models/` as `.pkl` files. If a model file already
exists and `FORCE_RETRAIN=false` in your `.env`, it will be loaded instead of
retrained.

### Step 3 — Capture Today's Props

1. Take a screenshot of the prop board on your sportsbook.
2. Run the OCR helper to extract raw text:

```powershell
python src/ocr_helper.py --image screenshots/props_today.png
```

This prints the extracted text to stdout and saves it to
`data/ocr_raw_output.txt`.

3. Paste the text into your LLM of choice with the following prompt:

```
You are a data extractor. Parse the following sportsbook prop board text into
a JSON array. Each element should have these fields:
  player_name, team, opponent, market, line, over_odds, under_odds, book, game_date

Return ONLY the JSON array, no explanation.

---
[PASTE OCR TEXT HERE]
---
```

4. Save the JSON response to `data/today_props_raw.json`.

   Alternatively, if you receive CSV output, save it as
   `data/today_props_raw.csv`.

### Step 4 — Run Daily EV Analysis

```powershell
# Using the new daily command (recommended)
python src/cli.py daily --props-file data/today_props_raw.json

# Optional flags:
python src/cli.py daily --props-file data/today_props_raw.json --top-n-singles 15 --top-n-combos 3
python src/cli.py daily --props-file data/today_props_raw.json --date 2026-03-13

# Or using the original analyze command:
python src/cli.py analyze
```

The ``daily`` command:

1. Loads and validates your props file (JSON or CSV).
2. Resolves player names to IDs using the ``players`` table.
3. Previews the props in a formatted table and asks for confirmation.
4. Builds feature vectors and runs projection models for each player.
5. Computes EV for every prop line (Over and Under).
6. Ranks single props by EV and suggests the top 2 bets with Kelly sizing.
7. Enumerates all 2-leg combos and ranks them by combo EV.
8. Saves a full CSV report to ``reports/daily_YYYYMMDD_props.csv``.

Output will include:

- A ranked table of all props by EV%.
- The top 2 single props above your configured EV threshold.
- The best 2-leg parlay combos (EV computed assuming independence).

### How to Generate ``data/today_props_raw.json``

1. **Screenshot** the prop board on your sportsbook app.
2. **OCR** the screenshot:
   ```powershell
   python src/ocr_helper.py --image screenshots/props_today.png
   ```
3. **Paste** the OCR text into an LLM (ChatGPT, Claude, etc.) with this prompt:
   ```
   Parse the following sportsbook prop board text into a JSON array.
   Each element must have these fields:
     player_name, team, opponent, market, line, over_odds, under_odds, book, game_date

   Use these market values: points, rebounds, assists, threepm,
   points_rebounds, points_assists, rebounds_assists, points_rebounds_assists

   Return ONLY the JSON array.

   ---
   [PASTE OCR TEXT]
   ---
   ```
4. **Save** the JSON response to ``data/today_props_raw.json``.
5. **Run** the daily command:
   ```powershell
   python src/cli.py daily --props-file data/today_props_raw.json
   ```

---

## Configuration

### `.env` file

```
# Tesseract binary path (Windows — adjust if needed)
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe

# Database path (relative to project root)
DB_PATH=data/nba_props.duckdb

# Model directory
MODELS_DIR=models

# Force retrain even if model files exist
FORCE_RETRAIN=false

# Bankroll for Kelly sizing display (dollars)
BANKROLL=1000

# Minimum EV% to display in output
EV_THRESHOLD=0.03

# Seasons to ingest (comma-separated, format: YYYY-YY)
SEASONS=2022-23,2023-24,2024-25
```

### `config.yaml`

See the file for detailed knobs including:
- Rolling window sizes for features (L5, L10, L20, season)
- Model hyperparameter grids
- Combo max legs
- Output formatting options

---

## Module Reference

| Module | Purpose |
|---|---|
| `src/db.py` | DuckDB connection helper and SQL runner |
| `src/schema.sql` | DDL for all tables |
| `src/ingest_games.py` | Fetch NBA schedule and game results |
| `src/ingest_boxscores.py` | Fetch player box score data |
| `src/ingest_injuries.py` | Fetch injury reports |
| `src/features.py` | Compute rolling/contextual player features |
| `src/models.py` | Train/load/predict projection models |
| `src/projections.py` | Build daily projection vectors for inference |
| `src/backtest.py` | Backtest projections against historical lines |
| `src/props_io.py` | Parse today's props from JSON/CSV |
| `src/ev_calc.py` | EV, Kelly, and combo ranking logic |
| `src/ocr_helper.py` | Tesseract OCR on prop board screenshots |
| `src/cli.py` | Main CLI entry point |

---

## Data Sources

Ingestion scripts use public APIs with no authentication required:

| Source | Data |
|---|---|
| [nba_api](https://github.com/swar/nba_api) | Schedules, box scores, team/player IDs |
| [Rotowire / CBS Sports](https://www.rotowire.com/basketball/nba-lineups.php) | Injury / lineup info (web fetch fallback) |

All data is stored locally in `data/nba_props.duckdb`. No data is ever sent
to external services.

---

## CLI Reference

```
python src/cli.py <command> [options]

Commands:
  db-init               Initialize database schema
  ingest                Run all ingestion scripts
  features              Rebuild player features
  train                 Train/retrain projection models
  daily                 Load props + compute projections + EV + suggest bets
  analyze               Run EV analysis on today's props
  backtest              Backtest projections vs historical lines
  show-props            Pretty-print today's loaded props
  show-projections      Show model projections for today's players

Global Options:
  --verbose             Print debug output

Daily Options:
  --props-file PATH     Path to today's props JSON/CSV (required)
  --top-n-singles INT   Number of top single props to show (default: 10)
  --top-n-combos INT    Number of top 2-leg combos to show (default: 2)
  --date DATE           Game date YYYY-MM-DD (default: today)

Analyze Options:
  --props-file PATH     Path to today's props JSON/CSV (default: data/today_props_raw.json)
  --ev-threshold FLOAT  Override EV threshold from .env
  --date DATE           Game date for analysis (default: today, format: YYYY-MM-DD)
  --force-retrain       Force model retraining even if .pkl exists
```

---

## Project Structure

```
nba_props/
├── .env.example
├── .gitignore
├── README.md
├── config.yaml
├── requirements.txt
├── pyproject.toml
│
├── data/
│   ├── .gitkeep
│   ├── nba_props.duckdb          # Created on db-init
│   ├── today_props_raw.json      # You create this daily
│   └── ocr_raw_output.txt        # Created by ocr_helper.py
│
├── models/
│   ├── .gitkeep
│   ├── points_model.pkl
│   ├── rebounds_model.pkl
│   ├── assists_model.pkl
│   └── threepm_model.pkl
│
├── logs/
│   └── .gitkeep
│
├── reports/
│   └── daily_YYYYMMDD_props.csv   # Generated by daily command
│
└── src/
    ├── db.py
    ├── schema.sql
    ├── ingest_games.py
    ├── ingest_boxscores.py
    ├── ingest_injuries.py
    ├── features.py
    ├── models.py
    ├── projections.py          # Loads models + builds daily projections
    ├── backtest.py
    ├── props_io.py
    ├── ev_calc.py
    ├── ocr_helper.py
    └── cli.py
```

---

## Legal Note

This tool is for personal research and entertainment purposes only. It does not
constitute gambling advice. All bets are placed manually and voluntarily at
licensed, legal sportsbooks in Illinois. Use responsibly.

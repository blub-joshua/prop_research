# NBA Player Prop Research Tool

Personal research tool for analysing NBA player prop bets. Uses LightGBM models
with quantile regression and isotonic calibration to project player stats and
estimate expected value (EV) on Underdog prop lines.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Initialise the database
python src/cli.py db-init

# 3. Ingest NBA data (games, box scores, injuries)
python src/cli.py ingest

# 4. Build player features (~5 min on first run)
python src/features.py

# 5. Train all models (mean + quantile + minutes + calibration, ~3–5 min)
python src/models.py --force-retrain

# 6. Run your first daily analysis
python src/cli.py daily --props-file data/today_props_raw.json
```

---

## Daily Workflow (Step-by-Step)

This is the routine you should follow each day you want to analyse props.

### Step 1: Update Data (morning)

Pull the latest box scores so the model has yesterday's results:

```bash
python src/cli.py ingest -m boxscores
python src/features.py
```

> **How often?** Run this every morning before analysing new props. It only
> fetches games that aren't already in the database, so it's fast after the
> first run.

### Step 2: Review Yesterday's Predictions

If you ran `daily` yesterday, check how those picks did:

```bash
python src/cli.py review
```

This grades your saved predictions against the actual box scores and appends
results to `data/tracking/performance_log.csv`.

To review a specific date:

```bash
python src/cli.py review --date 2025-03-12
```

To review all un-graded dates at once:

```bash
python src/cli.py review --all
```

### Step 3: Prepare Today's Props

1. **Screenshot** the prop lines you're interested in from Underdog.
2. **Paste** the screenshots into ChatGPT (or another LLM) with this prompt:

```
I need you to convert these sportsbook screenshots into a JSON array.
Each prop should be an object with exactly these fields:

- "player_name": full name (e.g. "LeBron James")
- "team": 3-letter abbreviation (e.g. "LAL")
- "opponent": 3-letter abbreviation of the opposing team (e.g. "GSW")
- "market": one of: "points", "rebounds", "assists", "threepm",
  "points_rebounds", "points_assists", "rebounds_assists",
  "points_rebounds_assists"
- "line": the prop line as a number (e.g. 25.5)
- "over_odds": American odds for the over (e.g. -115).
  If the book shows equal odds or "pick", use -110.
- "under_odds": American odds for the under (e.g. -105).
  If the book shows equal odds or "pick", use -110.
- "book": "Underdog"
- "game_date": today's date in YYYY-MM-DD format

Output ONLY the JSON array, no other text. Make sure all numbers are
valid JSON (no + prefix on positive numbers, no trailing commas).
```

3. **Save** the JSON output to `data/today_props_raw.json`.

### Step 4: Run the Daily Analysis

```bash
python src/cli.py daily --props-file data/today_props_raw.json
```

This will:
- Load and validate your props
- Show a preview table (confirm to continue)
- Build projections using all models (mean + quantile + minutes)
- Compute EV with calibrated probabilities
- Show the top picks ranked by expected value
- Show the best 2-leg combos
- **Save predictions** to `data/predictions/YYYY-MM-DD_predictions.json` for
  future review

### Step 5: Check Cumulative Performance

After several days of tracking:

```bash
python src/cli.py performance
```

This shows:
- Overall win rate and P&L across all tracked days
- Per-market breakdown (points, rebounds, assists, etc.)
- Calibration check (are the model's predicted win rates accurate?)
- Daily results history

---

## File Management Guide

### Files You Edit

| File | What it is | When to update |
|---|---|---|
| `data/today_props_raw.json` | Today's prop lines from Underdog | Every day before running `daily` |

> **Important:** You overwrite `today_props_raw.json` each day. The tool
> automatically saves dated copies of your predictions to
> `data/predictions/YYYY-MM-DD_predictions.json` — you never need to manually
> manage these.

### Files the Tool Creates (Don't Edit)

| File/Directory | Purpose |
|---|---|
| `data/nba_props.duckdb` | Main database (games, stats, features) |
| `data/predictions/*.json` | Saved daily predictions (auto-created by `daily`) |
| `data/tracking/performance_log.csv` | Cumulative graded results (auto-created by `review`) |
| `reports/daily_*_props.csv` | Full EV analysis reports (auto-created by `daily`) |
| `models/*.pkl` | Trained model files |
| `models/eval/` | Training evaluation plots and metrics |
| `logs/nba_props.log` | Application logs |

### Retraining Models

Retrain weekly (or after adding significant new data):

```bash
python src/models.py --force-retrain
```

This retrains everything: the minutes model, all 4 stat models, the 20 quantile
models (5 levels × 4 stats), and the 4 calibrators.

---

## How the Model Works

### Architecture

The system uses a multi-layer approach to estimate win probabilities:

1. **Minutes Model** — A separate LightGBM model predicts how many minutes a
   player will play. This feeds into the stat models as an additional feature,
   since minutes played is the single biggest driver of stat output.

2. **Mean Regression Models** — One LightGBM model per stat (points, rebounds,
   assists, 3PM) predicts the expected value. Uses 33 features including
   rolling averages, opponent defense, pace, rest, and projected minutes.

3. **Quantile Regression Models** — Five additional models per stat predict the
   10th, 25th, 50th, 75th, and 90th percentiles. This captures the full shape
   of the distribution (skew, fat tails) rather than assuming a bell curve.

4. **Probability Estimation** — Instead of using a Normal CDF (which
   systematically overestimates edge), the tool interpolates between quantile
   predictions to estimate P(over line). This gives much more accurate
   probabilities.

5. **Isotonic Calibration** — A calibration layer trained on the validation
   season maps raw model probabilities to historically accurate probabilities.
   When the model says "60% chance of hitting", calibration checks whether that
   actually hit 60% of the time and corrects any bias.

### Features Used

- Rolling averages (last 5, 10 games) for PTS, REB, AST, 3PM, MIN
- Rolling standard deviations (volatility signals)
- Season-to-date averages
- Home/away splits
- Opponent defensive stats (points/rebounds/assists/3PM allowed, L10)
- Days rest, back-to-back flag
- Injury severity
- Team pace
- **Projected minutes** (from the minutes model)

### EV Calculation

```
EV% = (model_probability × decimal_odds) - 1
```

A positive EV means the model thinks the bet is worth more than the odds imply.
The tool shows "Win %" (the model's estimated probability of hitting) alongside
EV so you can evaluate both the edge and the confidence.

---

## CLI Reference

| Command | Description |
|---|---|
| `db-init` | Create/update database tables |
| `ingest -m games` | Fetch NBA schedule data |
| `ingest -m boxscores` | Fetch player box scores |
| `ingest -m injuries` | Fetch injury reports |
| `features` | Rebuild player feature table |
| `train [--force-retrain]` | Train all models |
| `daily --props-file PATH` | Full daily analysis pipeline |
| `review [--date YYYY-MM-DD] [--all]` | Grade past predictions |
| `performance [--days N]` | Show cumulative stats |
| `analyze --props-file PATH` | Quick EV analysis (no tracking) |
| `backtest [--ev-threshold N]` | Run backtest on historical data |
| `show-props --props-file PATH` | Preview loaded props |
| `show-projections` | Show raw model projections |

---

## Module Reference

| Module | Purpose |
|---|---|
| `cli.py` | Command-line interface and orchestration |
| `db.py` | DuckDB connection, schema, upsert utilities |
| `features.py` | Feature engineering (rolling stats, opponent defense, etc.) |
| `models.py` | Model training: mean, quantile, minutes, calibration |
| `projections.py` | Live inference — builds projections for today's players |
| `ev_calc.py` | EV computation, Kelly sizing, combo ranking |
| `props_io.py` | Load/normalise/validate prop lines from JSON/CSV |
| `review.py` | Grade predictions, track performance over time |
| `backtest.py` | Historical backtesting against synthetic/real prop lines |
| `ingest_games.py` | NBA API game schedule ingestion |
| `ingest_boxscores.py` | NBA API box score ingestion |
| `ingest_injuries.py` | Injury report ingestion |

---

## Environment Variables

Set these in a `.env` file in the project root:

| Variable | Default | Description |
|---|---|---|
| `DB_PATH` | `data/nba_props.duckdb` | Database file location |
| `MODELS_DIR` | `models` | Directory for saved models |
| `LOG_FILE` | `logs/nba_props.log` | Log file path |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `EV_THRESHOLD` | `0.03` | Minimum EV% to display (3%) |
| `BANKROLL` | `1000` | Your bankroll in dollars (for Kelly sizing) |
| `MAX_KELLY_FRACTION` | `0.05` | Max fraction of bankroll per bet (5%) |
| `FORCE_RETRAIN` | `false` | Force model retraining on every run |

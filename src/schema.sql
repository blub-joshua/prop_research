-- =============================================================================
-- NBA Props Research Tool — Database Schema
-- Engine : DuckDB
-- File   : src/schema.sql
--
-- Run via:  python src/cli.py db-init
--           (which calls db.init_schema() → executes this file)
-- =============================================================================

-- ---------------------------------------------------------------------------
-- TEAMS
-- One row per NBA franchise.  team_id is the NBA API integer ID.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS teams (
    team_id         INTEGER PRIMARY KEY,    -- NBA API team ID (e.g. 1610612747)
    abbreviation    VARCHAR(5)  NOT NULL,   -- e.g. "LAL"
    full_name       VARCHAR(64) NOT NULL,   -- e.g. "Los Angeles Lakers"
    city            VARCHAR(64),
    conference      VARCHAR(4),             -- "East" | "West"
    division        VARCHAR(16),
    updated_at      TIMESTAMP DEFAULT current_timestamp
);

-- ---------------------------------------------------------------------------
-- PLAYERS
-- One row per NBA player ever seen in ingested data.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS players (
    player_id       INTEGER PRIMARY KEY,    -- NBA API person ID
    full_name       VARCHAR(128) NOT NULL,
    first_name      VARCHAR(64),
    last_name       VARCHAR(64),
    team_id         INTEGER REFERENCES teams(team_id),
    position        VARCHAR(8),             -- e.g. "PG", "SG-SF"
    height_inches   SMALLINT,
    weight_lbs      SMALLINT,
    birthdate       DATE,
    is_active       BOOLEAN DEFAULT TRUE,
    updated_at      TIMESTAMP DEFAULT current_timestamp
);

-- ---------------------------------------------------------------------------
-- GAMES
-- One row per NBA game (regular season + playoffs).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS games (
    game_id         VARCHAR(20) PRIMARY KEY,    -- NBA API game ID string
    game_date       DATE NOT NULL,
    season          VARCHAR(7)  NOT NULL,        -- e.g. "2024-25"
    season_type     VARCHAR(16) NOT NULL,        -- "Regular Season" | "Playoffs"
    home_team_id    INTEGER REFERENCES teams(team_id),
    away_team_id    INTEGER REFERENCES teams(team_id),
    home_score      SMALLINT,
    away_score      SMALLINT,
    status          VARCHAR(16),                 -- "Final" | "Live" | "Scheduled"
    arena           VARCHAR(64),
    tip_off_utc     TIMESTAMP,                   -- UTC tip-off time
    created_at      TIMESTAMP DEFAULT current_timestamp,
    updated_at      TIMESTAMP DEFAULT current_timestamp
);

CREATE INDEX IF NOT EXISTS idx_games_date   ON games (game_date);
CREATE INDEX IF NOT EXISTS idx_games_season ON games (season);

-- ---------------------------------------------------------------------------
-- PLAYER_GAME_STATS
-- Traditional box score stats for each player in each game.
-- One row per (player_id, game_id).
--
-- NOTE: id is assigned by a sequence — do NOT insert it explicitly.
--       Composite "combo" columns (pra, points_rebounds, etc.) are plain
--       SMALLINT columns populated by the ingest script; DuckDB does not
--       support GENERATED ALWAYS ... STORED in older releases.
-- ---------------------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS seq_pgs_id START 1;

CREATE TABLE IF NOT EXISTS player_game_stats (
    id              BIGINT PRIMARY KEY DEFAULT nextval('seq_pgs_id'),
    player_id       INTEGER NOT NULL REFERENCES players(player_id),
    game_id         VARCHAR(20) NOT NULL REFERENCES games(game_id),
    team_id         INTEGER REFERENCES teams(team_id),   -- nullable: fk checked at insert
    game_date       DATE NOT NULL,              -- denormalized for faster feature queries

    -- Participation flags
    did_not_play    BOOLEAN DEFAULT FALSE,      -- DNP-Coach's Decision or similar
    is_starter      BOOLEAN,

    -- Core counting stats
    minutes         NUMERIC(5,2),              -- e.g. 34.50
    points          SMALLINT,
    rebounds        SMALLINT,
    assists         SMALLINT,
    steals          SMALLINT,
    blocks          SMALLINT,
    turnovers       SMALLINT,
    fouls           SMALLINT,
    off_reb         SMALLINT,                  -- offensive rebounds
    def_reb         SMALLINT,                  -- defensive rebounds

    -- Shooting splits
    fgm             SMALLINT,                  -- field goals made
    fga             SMALLINT,                  -- field goals attempted
    fg_pct          NUMERIC(5,4),
    fg3m            SMALLINT,                  -- 3-pointers made
    fg3a            SMALLINT,
    fg3_pct         NUMERIC(5,4),
    ftm             SMALLINT,                  -- free throws made
    fta             SMALLINT,
    ft_pct          NUMERIC(5,4),

    -- Composite combo props (plain columns — populated by ingest script)
    points_rebounds SMALLINT,
    points_assists  SMALLINT,
    rebounds_assists SMALLINT,
    pra             SMALLINT,

    -- Advanced (optional — populated if BoxScoreAdvancedV3 succeeds)
    plus_minus      SMALLINT,
    usage_pct       NUMERIC(5,2),
    off_rating      NUMERIC(6,2),
    def_rating      NUMERIC(6,2),
    pace            FLOAT,

    created_at      TIMESTAMP DEFAULT current_timestamp,

    UNIQUE (player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_pgs_player_date ON player_game_stats (player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_pgs_game        ON player_game_stats (game_id);

-- ---------------------------------------------------------------------------
-- INJURIES
-- Tracks player injury status as reported on a given date.
-- Multiple rows per player_id are expected as status changes day to day.
-- ---------------------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS seq_injuries_id START 1;
CREATE TABLE IF NOT EXISTS injuries (
    id              BIGINT PRIMARY KEY DEFAULT nextval('seq_injuries_id'),
    player_id       INTEGER NOT NULL REFERENCES players(player_id),
    report_date     DATE NOT NULL,

    -- Injury status vocabulary (NBA/ESPN convention)
    status          VARCHAR(32),    -- "Out" | "Doubtful" | "Questionable" | "Probable" | "Active"
    injury_type     VARCHAR(128),   -- e.g. "Left Knee - Soreness"
    notes           TEXT,
    source          VARCHAR(32),    -- "nba_api" | "rotowire" | "manual"

    created_at      TIMESTAMP DEFAULT current_timestamp,

    UNIQUE (player_id, report_date, status)
);

CREATE INDEX IF NOT EXISTS idx_injuries_player ON injuries (player_id, report_date DESC);

-- ---------------------------------------------------------------------------
-- PLAYER_FEATURES
-- Pre-computed feature vectors used as model input.
-- Rebuilt by features.py.  One row per (player_id, game_id).
-- Column names encode the stat and rolling window, e.g. pts_avg_L10.
-- ---------------------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS seq_pf_id START 1;
CREATE TABLE IF NOT EXISTS player_features (
    id              BIGINT PRIMARY KEY DEFAULT nextval('seq_pf_id'),
    player_id       INTEGER NOT NULL REFERENCES players(player_id),
    game_id         VARCHAR(20) REFERENCES games(game_id),  -- the specific game
    game_date       DATE NOT NULL,          -- denormalised for fast range queries
    team_id         INTEGER,
    opponent_id     INTEGER,
    season          VARCHAR(7),
    is_home         INTEGER,               -- 1=home, 0=away

    pts_avg_L5      NUMERIC(6,2),
    pts_avg_L10     NUMERIC(6,2),
    reb_avg_L5      NUMERIC(6,2),
    reb_avg_L10     NUMERIC(6,2),
    ast_avg_L5      NUMERIC(6,2),
    ast_avg_L10     NUMERIC(6,2),
    fg3m_avg_L5     NUMERIC(6,2),
    fg3m_avg_L10    NUMERIC(6,2),
    min_avg_L5      NUMERIC(6,2),
    min_avg_L10     NUMERIC(6,2),

    pts_avg_season  NUMERIC(6,2),
    reb_avg_season  NUMERIC(6,2),
    ast_avg_season  NUMERIC(6,2),
    fg3m_avg_season NUMERIC(6,2),
    min_avg_season  NUMERIC(6,2),

    pts_std_L5      NUMERIC(6,2),
    pts_std_L10     NUMERIC(6,2),
    reb_std_L5      NUMERIC(6,2),
    reb_std_L10     NUMERIC(6,2),
    ast_std_L5      NUMERIC(6,2),
    ast_std_L10     NUMERIC(6,2),
    fg3m_std_L5     NUMERIC(6,2),
    fg3m_std_L10    NUMERIC(6,2),

    opp_pts_allowed_avg  NUMERIC(6,2),
    opp_reb_allowed_avg  NUMERIC(6,2),
    opp_ast_allowed_avg  NUMERIC(6,2),
    opp_fg3m_allowed_avg NUMERIC(6,2),

    pace_L5              NUMERIC(6,2),
    injury_severity      SMALLINT,

    pts_avg_home    NUMERIC(6,2),
    pts_avg_away    NUMERIC(6,2),
    reb_avg_home    NUMERIC(6,2),
    reb_avg_away    NUMERIC(6,2),
    ast_avg_home    NUMERIC(6,2),
    ast_avg_away    NUMERIC(6,2),

    days_rest               SMALLINT,
    games_played_season     SMALLINT,

    injury_status           VARCHAR(32),

    created_at      TIMESTAMP DEFAULT current_timestamp,
    updated_at      TIMESTAMP DEFAULT current_timestamp,

    UNIQUE (player_id, game_id)
);

CREATE INDEX IF NOT EXISTS idx_features_player_date ON player_features (player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_features_game        ON player_features (game_id);

-- ---------------------------------------------------------------------------
-- HISTORICAL_PROPS
-- Historical closing lines and results for backtesting.
-- Populated by backtest.py or future automated line ingestion.
-- ---------------------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS seq_hp_id START 1;
CREATE TABLE IF NOT EXISTS historical_props (
    id              BIGINT PRIMARY KEY DEFAULT nextval('seq_hp_id'),
    player_id       INTEGER REFERENCES players(player_id),
    player_name     VARCHAR(128),
    game_id         VARCHAR(20) REFERENCES games(game_id),
    game_date       DATE NOT NULL,
    book            VARCHAR(32),
    market          VARCHAR(32) NOT NULL,
    side            VARCHAR(8) NOT NULL,
    line            NUMERIC(6,2) NOT NULL,
    odds_american   SMALLINT NOT NULL,
    odds_decimal    NUMERIC(6,4),

    actual_value    NUMERIC(7,2),
    result          VARCHAR(8),

    projection      NUMERIC(7,2),
    ev_pct          NUMERIC(6,4),

    created_at      TIMESTAMP DEFAULT current_timestamp,

    UNIQUE (player_id, game_id, book, market, side)
);

CREATE INDEX IF NOT EXISTS idx_hprops_player   ON historical_props (player_id);
CREATE INDEX IF NOT EXISTS idx_hprops_date     ON historical_props (game_date);
CREATE INDEX IF NOT EXISTS idx_hprops_market   ON historical_props (market);

-- ---------------------------------------------------------------------------
-- PROJECTIONS_LOG
-- Stores every projection run so you can track model drift over time.
-- ---------------------------------------------------------------------------
CREATE SEQUENCE IF NOT EXISTS seq_pl_id START 1;
CREATE TABLE IF NOT EXISTS projections_log (
    id              BIGINT PRIMARY KEY DEFAULT nextval('seq_pl_id'),
    run_at          TIMESTAMP DEFAULT current_timestamp,
    player_id       INTEGER REFERENCES players(player_id),
    game_date       DATE NOT NULL,
    market          VARCHAR(32) NOT NULL,
    projection      NUMERIC(7,2),
    model_version   VARCHAR(32),
    feature_snapshot JSON
);

CREATE INDEX IF NOT EXISTS idx_projlog_player_date ON projections_log (player_id, game_date);

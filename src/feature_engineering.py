"""
Feature engineering pipeline (Section 5).

Point-in-time features only. Opponent and team stats computed as of
end of day before game (season-to-date or rolling) to prevent leakage.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Opponent and team stats are computed as of end of day before game (season-to-date
# or rolling) to prevent leakage.


def _load_config() -> dict:
    root = Path(__file__).resolve().parent.parent
    with open(root / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def _build_team_game_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Build (game_id, game_date, team_id, opponent_team_id, goals_for, goals_against) per team-game."""
    # Per game: get away/home team goals
    team_goals = df.groupby(["game_id", "team_id", "opponent_team_id", "game_date", "home_away"]).agg(
        goals_for=("goals", "sum")
    ).reset_index()
    # Merge to get goals_against (opponent's goals_for in same game)
    game_goals = (
        df.groupby(["game_id", "game_date"], group_keys=False)
        .apply(
            lambda g: pd.Series({
                "away_team": g[g["home_away"] == "away"]["team_id"].iloc[0],
                "home_team": g[g["home_away"] == "home"]["team_id"].iloc[0],
                "away_goals": g[g["home_away"] == "away"]["goals"].sum(),
                "home_goals": g[g["home_away"] == "home"]["goals"].sum(),
            }),
            include_groups=False,
        )
        .reset_index()
    )
    # Explode to team-level: each team gets a row with goals_for, goals_against
    rows = []
    for _, r in game_goals.iterrows():
        rows.append({
            "game_id": r["game_id"],
            "game_date": r["game_date"],
            "team_id": r["away_team"],
            "goals_for": r["away_goals"],
            "goals_against": r["home_goals"],
        })
        rows.append({
            "game_id": r["game_id"],
            "game_date": r["game_date"],
            "team_id": r["home_team"],
            "goals_for": r["home_goals"],
            "goals_against": r["away_goals"],
        })
    return pd.DataFrame(rows)


def _build_opponent_goals_allowed(team_games: pd.DataFrame) -> pd.DataFrame:
    """
    For each (team_id, game_date), compute goals_allowed_per_game
    as of end of day BEFORE game_date (point-in-time, no leakage).
    """
    team_games = team_games.copy()
    team_games["game_date"] = pd.to_datetime(team_games["game_date"])
    team_games = team_games.sort_values(["team_id", "game_date"])

    g = team_games.groupby("team_id")
    team_games["cum_goals_against"] = g["goals_against"].cumsum().shift(1)
    team_games["cum_games"] = g.cumcount()  # 0, 1, 2... = games before (0-indexed)
    team_games["opponent_goals_allowed_pg"] = (
        team_games["cum_goals_against"] / team_games["cum_games"].clip(lower=1)
    )
    return team_games[["game_id", "team_id", "game_date", "opponent_goals_allowed_pg"]]


def _add_player_rolling_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add rolling player stats. Uses only past games (shift(1)) for point-in-time."""
    df = df.sort_values(["player_id", "game_date"])
    w5 = cfg.get("rolling_window", 5)
    w3 = cfg.get("goals_window", 3)
    w5a = cfg.get("assists_window", 5)

    g = df.groupby("player_id")

    # Rolling sums/averages over PAST games only (shift to exclude current)
    df["_points_shift"] = g["points"].transform(lambda x: x.shift(1))
    df["_goals_shift"] = g["goals"].transform(lambda x: x.shift(1))
    df["_assists_shift"] = g["assists"].transform(lambda x: x.shift(1))
    df["_shots_shift"] = g["shots"].transform(lambda x: x.shift(1))
    df["_toi_shift"] = g["time_on_ice_minutes"].transform(lambda x: x.shift(1))

    df["rolling_points_5"] = g["_points_shift"].transform(lambda x: x.rolling(w5, min_periods=1).mean())
    df["rolling_goals_3"] = g["_goals_shift"].transform(lambda x: x.rolling(w3, min_periods=1).sum())
    df["rolling_assists_5"] = g["_assists_shift"].transform(lambda x: x.rolling(w5a, min_periods=1).sum())
    df["rolling_shots_5"] = g["_shots_shift"].transform(lambda x: x.rolling(w5, min_periods=1).mean())
    df["rolling_toi_5"] = g["_toi_shift"].transform(lambda x: x.rolling(w5, min_periods=1).mean())

    # Trend: slope of points over last 5 (simplified: diff of rolling mean)
    df["points_trend"] = g["_points_shift"].transform(
        lambda x: x.rolling(w5, min_periods=2).mean().diff().fillna(0)
    )
    # Volatility: std of points
    df["points_volatility"] = g["_points_shift"].transform(
        lambda x: x.rolling(cfg.get("volatility_window", 5), min_periods=2).std().fillna(0)
    )
    # Fatigue: days_rest and back-to-back
    df["game_date_dt"] = pd.to_datetime(df["game_date"])
    df["_prev_date"] = g["game_date_dt"].transform(lambda x: x.shift(1))
    df["days_rest"] = (df["game_date_dt"] - df["_prev_date"]).dt.days.fillna(7).clip(upper=14)
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(int)

    df.drop(columns=[c for c in df.columns if c.startswith("_")], inplace=True, errors="ignore")
    return df


def _add_game_context(df: pd.DataFrame) -> pd.DataFrame:
    """Home/away, season phase."""
    df["is_home"] = (df["home_away"] == "home").astype(int)
    # Season phase: early (1-27), mid (28-55), late (56+). Approx by game number per team.
    df["game_date_dt"] = pd.to_datetime(df["game_date"])
    df["team_season_game_num"] = df.groupby(["player_id", "season"])["game_date_dt"].rank(method="dense")
    df["season_phase"] = pd.cut(
        df["team_season_game_num"],
        bins=[0, 27, 55, 200],
        labels=["early", "mid", "late"]
    ).astype(str)
    return df


def build_features(
    processed_path: str | Path | None = None,
    output_path: str | Path | None = None,
    min_games: int | None = None,
) -> Path:
    """
    Build features for all player-games in processed data.
    Point-in-time: all features use only data before the game (no leakage).

    Returns path to output parquet file.
    """
    config = _load_config()
    root = Path(__file__).resolve().parent.parent
    cfg = config.get("features", {})
    min_g = min_games if min_games is not None else cfg.get("min_games_history", 5)

    if processed_path is None:
        proc_dir = root / config["paths"]["processed_data"]
        parquets = sorted(proc_dir.glob("*.parquet"), reverse=True)
        if not parquets:
            raise FileNotFoundError(f"No processed parquet files in {proc_dir}")
        processed_path = parquets[0]
    else:
        processed_path = Path(processed_path)

    if output_path is None:
        output_path = root / config["paths"]["features"]
    else:
        output_path = Path(output_path)

    logger.info("Loading processed data from %s", processed_path)
    df = pd.read_parquet(processed_path)

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["player_id", "game_date"])

    # 1. Team-game stats for opponent features
    team_games = _build_team_game_stats(df)
    opp_stats = _build_opponent_goals_allowed(team_games)

    # 2. Merge opponent goals allowed (we face opponent_team_id, so we want their goals_allowed)
    opp_stats = opp_stats.rename(columns={"team_id": "opponent_team_id"})
    df = df.merge(
        opp_stats[["game_id", "opponent_team_id", "opponent_goals_allowed_pg"]],
        on=["game_id", "opponent_team_id"],
        how="left",
    )
    df["opponent_goals_allowed_pg"] = df["opponent_goals_allowed_pg"].fillna(2.9)  # league avg placeholder

    # 3. Player rolling features (point-in-time via shift)
    df = _add_player_rolling_features(df, cfg)
    df = _add_game_context(df)

    # 4. Label: y = 1 if points >= 1
    df["label"] = (df["points"] >= 1).astype(int)

    # 5. Cold start: require min_games_history (games played before this one)
    df["games_played_before"] = df.groupby("player_id").cumcount()
    df = df[df["games_played_before"] >= min_g].drop(columns=["games_played_before"])

    # 6. Select feature columns
    feature_cols = [
        "rolling_points_5",
        "rolling_goals_3",
        "rolling_assists_5",
        "rolling_shots_5",
        "rolling_toi_5",
        "points_trend",
        "points_volatility",
        "is_back_to_back",
        "days_rest",
        "opponent_goals_allowed_pg",
        "is_home",
        "season_phase",
    ]
    meta_cols = ["game_id", "player_id", "team_id", "opponent_team_id", "game_date", "season", "label"]
    out_cols = meta_cols + [c for c in feature_cols if c in df.columns]
    df_out = df[out_cols].copy()

    # Encode season_phase
    df_out["season_phase_early"] = (df_out["season_phase"] == "early").astype(int)
    df_out["season_phase_mid"] = (df_out["season_phase"] == "mid").astype(int)
    df_out["season_phase_late"] = (df_out["season_phase"] == "late").astype(int)
    df_out = df_out.drop(columns=["season_phase"], errors="ignore")

    output_path.mkdir(parents=True, exist_ok=True)
    run_id = processed_path.stem.replace("player_games_", "")
    out_file = output_path / f"features_{run_id}.parquet"
    df_out.to_parquet(out_file, index=False)

    feature_names = [c for c in df_out.columns if c not in ["game_id", "player_id", "team_id", "opponent_team_id", "game_date", "season", "label"]]
    features_meta = output_path / f"features_{run_id}_meta.json"
    with open(features_meta, "w") as f:
        json.dump({"feature_names": feature_names, "min_games_history": min_g}, f, indent=2)

    logger.info("Wrote %d rows to %s (min_games=%d)", len(df_out), out_file, min_g)
    return out_file


if __name__ == "__main__":
    build_features()

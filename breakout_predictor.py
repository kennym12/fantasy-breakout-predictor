"""Fantasy football breakout predictor.

Scores every QB, RB, WR and TE season with a formula tailored to the position,
then checks whether the top-scored players went on to break out the next year.

Run with:  python breakout_predictor.py
It reads breakout_data.csv from the same folder, so it works offline.
Add --refresh to re-download the data from nflverse (needs internet).
Add --inspect to print the columns and first rows of the data.
"""
import argparse
from pathlib import Path

import pandas as pd

FIRST_SEASON = 2016
LAST_SEASON = 2025  # most recent completed season; its players are next year's candidates
EVAL_SEASONS = range(2018, 2024)
HOLDOUT_SEASONS = [2016, 2017, 2024]  # never used to build the formulas, so they are a fair test
POSITIONS = ["QB", "RB", "WR", "TE"]
PPR = "fantasy_points_ppr"
TOP_N = 10
MIN_GAMES = 4
SEASON_GAMES = 17
TOTAL_MULTIPLIER = 1.5  # route 1: 50% more total points, only for players under the cap
PPG_MULTIPLIER = 1.3  # route 2: 30% more points per game, open to everyone

# floor: next-season points needed to count as a breakout.
# cap: players above this are already stars, so a bigger total does not count
# for them. They can only break out by raising their points per game.
# QBs score far more than skill players and TEs far less, so each gets its own pair.
THRESHOLDS = {
    "QB": {"floor": 240, "cap": 280},
    "RB": {"floor": 150, "cap": 200},
    "WR": {"floor": 150, "cap": 200},
    "TE": {"floor": 110, "cap": 150},
}

# Players below this workload have rate stats too noisy to rank.
MIN_VOLUME = {
    "QB": ("attempts", 50),
    "RB": ("touches", 20),
    "WR": ("targets", 10),
    "TE": ("targets", 10),
}

# Each stat is turned into a 0-1 percentile within the position and season,
# so the weights are directly comparable. Each group below adds up to 1.
USAGE = {
    # Rushing is the cheapest source of QB points, efficiency earns the job,
    # and attempts per game shows he was the starter when he played.
    "QB": {"rush_yards_pg": 0.40, "epa_per_dropback": 0.35, "attempts_pg": 0.25},
    # Carries are the base of RB scoring, targets are worth more per touch in
    # PPR, and yards per touch says whether he earned a bigger role.
    "RB": {"carries_pg": 0.40, "targets_pg": 0.35, "yards_per_touch": 0.25},
    # Target share is the stickiest WR stat, air yards share shows downfield
    # usage that has not turned into points yet, and targets per game rewards
    # players who missed time.
    "WR": {"target_share": 0.45, "air_yards_share": 0.30, "targets_pg": 0.25},
    # Same idea as WR, but yards per game replaces targets per game because
    # many TEs are targeted only as a checkdown.
    "TE": {"target_share": 0.45, "air_yards_share": 0.25, "rec_yards_pg": 0.30},
}

# Next Gen Stats measure what the player added beyond what the play gave him.
NGS = {
    # Accuracy on throws harder than average, and willingness to attempt them.
    "QB": {"ngs_cpoe": 0.50, "ngs_aggressiveness": 0.50},
    # Rushing yards beyond what the blocking set up.
    "RB": {"ngs_ryoe_per_att": 1.00},
    # Yards after the catch beyond what the tracking model expected.
    "WR": {"ngs_yac_over_expected": 1.00},
    "TE": {"ngs_yac_over_expected": 1.00},
}


def build_weights(usage, youth=0.0, reachable=0.0, ngs=0.0):
    """Split the score between the usage formula, youth, reachability and Next Gen Stats."""
    weights = {}
    for position in POSITIONS:
        stats = {stat: w * usage for stat, w in USAGE[position].items()}
        stats.update({stat: w * ngs for stat, w in NGS[position].items()})
        stats.update({"youth": youth, "reachable": reachable})
        weights[position] = {stat: w for stat, w in stats.items() if w > 0}
    return weights


# youth: most breakouts come early in a career.
# reachable: the fewer points a player needs next year to count as a breakout,
# the likelier he gets there. Without it the usage stats push established
# players who would need a monster season.
VERSIONS = {
    "v1": build_weights(usage=1.0),
    "v2": build_weights(usage=0.7, youth=0.3),
    "v3": build_weights(usage=0.6, reachable=0.4),
    "v4": build_weights(usage=0.5, youth=0.2, reachable=0.3),
    "v5": build_weights(usage=0.3, youth=0.2, reachable=0.3, ngs=0.2),
}
FINAL_VERSION = "v5"

RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
STATS_URL = RELEASES + "/stats_player/stats_player_reg_{season}.csv"
ROSTER_URL = RELEASES + "/rosters/roster_{season}.csv"
NGS_URL = RELEASES + "/nextgen_stats/ngs_{kind}.csv.gz"
try:
    BASE_DIR = Path(__file__).parent
except NameError:  # pasted into a notebook or console, where there is no file
    BASE_DIR = Path(".")
# Everything the script needs, saved next to it so no internet is required.
SNAPSHOT = BASE_DIR / "breakout_data.csv"
ROSTER_COLUMNS = ["season", "gsis_id", "position", "rookie_year", "years_exp"]
# Older nflverse files use different names for the same columns.
ALIASES = {"team": "recent_team", "sacks": "sacks_suffered"}
# Next Gen Stats come from player tracking chips. Only players with enough
# volume to qualify for the season leaderboard have them.
NGS_STATS = {
    "passing": {
        "completion_percentage_above_expectation": "ngs_cpoe",
        "aggressiveness": "ngs_aggressiveness",
    },
    "receiving": {"avg_yac_above_expectation": "ngs_yac_over_expected"},
    "rushing": {"rush_yards_over_expected_per_att": "ngs_ryoe_per_att"},
}
COUNTING_STATS = [
    "attempts", "sacks_suffered", "passing_epa", "carries", "rushing_yards",
    "targets", "receptions", "receiving_yards", "receiving_air_yards",
]
# Columns kept in the snapshot. To use another nflverse stat, add it here and
# run once with --refresh.
SNAPSHOT_COLUMNS = (
    ["player_id", "player_display_name", "position", "season", "recent_team", "games", PPR]
    + COUNTING_STATS
    + ["target_share", "air_yards_share", "rookie_year", "years_exp"]
    + [name for columns in NGS_STATS.values() for name in columns.values()]
)


def _download_stats(seasons):
    try:
        import nflreadpy as nfl  # needs Python 3.10+
    except ImportError:
        # Backup: the same season-total files nflreadpy reads, straight from nflverse-data.
        return pd.concat(
            [pd.read_csv(STATS_URL.format(season=s), low_memory=False) for s in seasons],
            ignore_index=True,
        )
    return nfl.load_player_stats(seasons, summary_level="reg").to_pandas()


def _download_rosters(seasons):
    try:
        import nflreadpy as nfl
    except ImportError:
        rosters = pd.concat(
            [pd.read_csv(ROSTER_URL.format(season=s), low_memory=False) for s in seasons],
            ignore_index=True,
        )
    else:
        rosters = nfl.load_rosters(seasons).to_pandas()
    return rosters[ROSTER_COLUMNS]


def _download_ngs(kind, seasons):
    try:
        import nflreadpy as nfl
    except ImportError:
        ngs = pd.read_csv(NGS_URL.format(kind=kind), low_memory=False)
    else:
        ngs = nfl.load_nextgen_stats(seasons, stat_type=kind).to_pandas()
    keep = ["season", "season_type", "week", "player_gsis_id"] + list(NGS_STATS[kind])
    return ngs[ngs["season"].isin(seasons)][keep]


def load_ngs(seasons):
    """Return season-long Next Gen Stats, one row per player per season."""
    merged = None
    for kind, columns in NGS_STATS.items():
        ngs = _download_ngs(kind, seasons)
        # Week 0 is the season total.
        ngs = ngs[(ngs["week"] == 0) & (ngs["season_type"] == "REG")]
        ngs = ngs.rename(columns=dict(columns, player_gsis_id="player_id"))
        ngs = ngs.groupby(["player_id", "season"], as_index=False)[list(columns.values())].mean()
        merged = ngs if merged is None else merged.merge(ngs, on=["player_id", "season"], how="outer")
    return merged


def download_data(seasons):
    """Build the player-season table from nflverse. Needs internet."""
    stats = _download_stats(seasons)
    for old, new in ALIASES.items():
        if old in stats.columns and new not in stats.columns:
            stats = stats.rename(columns={old: new})
    if "season_type" in stats.columns:
        stats = stats[stats["season_type"] == "REG"]

    rosters = _download_rosters(seasons).rename(columns={"gsis_id": "player_id"})
    rosters = rosters.dropna(subset=["player_id"]).drop_duplicates(["player_id", "season"])
    roster_cols = ["player_id", "season", "rookie_year", "years_exp"]
    if "position" not in stats.columns:
        roster_cols.append("position")
    stats = stats.merge(rosters[roster_cols], on=["player_id", "season"], how="left")
    stats = stats.merge(load_ngs(seasons), on=["player_id", "season"], how="left")

    stats = stats[stats["position"].isin(POSITIONS)].copy()
    if stats.duplicated(["player_id", "season"]).any():
        raise ValueError("expected one row per player per season")
    stats[COUNTING_STATS] = stats[COUNTING_STATS].fillna(0)
    stats[PPR] = stats[PPR].fillna(0)
    return stats[[column for column in SNAPSHOT_COLUMNS if column in stats.columns]]


def load_data(seasons, refresh=False):
    """Return one row per player per regular season for QBs, RBs, WRs and TEs.

    Reads the snapshot saved next to this script. It is only rebuilt from
    nflverse when refresh is True or the snapshot is missing.
    """
    seasons = list(seasons)
    if refresh or not SNAPSHOT.exists():
        try:
            fresh = download_data(seasons)
        except OSError as error:
            raise SystemExit(
                "Could not download from nflverse: {}\n"
                "To run offline, keep {} in the same folder as this script.".format(error, SNAPSHOT.name)
            )
        fresh.to_csv(SNAPSHOT, index=False)
    stats = pd.read_csv(SNAPSHOT)
    if stats["season"].max() < seasons[-1]:
        raise SystemExit(
            "{} stops at {}. Run with --refresh while online to update it.".format(SNAPSHOT.name, stats["season"].max())
        )
    return stats


def _threshold(df, kind):
    """Return each row's floor or cap for its position."""
    return df["position"].map(lambda position: THRESHOLDS[position][kind])


def add_features(df):
    """Add the per-game and efficiency stats the score formulas use."""
    df = df.copy()
    games = df["games"].clip(lower=1)
    df["ppg"] = df[PPR] / games
    df["attempts_pg"] = df["attempts"] / games
    df["rush_yards_pg"] = df["rushing_yards"] / games
    df["carries_pg"] = df["carries"] / games
    df["targets_pg"] = df["targets"] / games
    df["rec_yards_pg"] = df["receiving_yards"] / games

    dropbacks = df["attempts"] + df["sacks_suffered"]
    df["epa_per_dropback"] = df["passing_epa"] / dropbacks.where(dropbacks > 0)
    df["touches"] = df["carries"] + df["receptions"]
    df["yards_per_touch"] = (df["rushing_yards"] + df["receiving_yards"]) / df["touches"].where(df["touches"] > 0)

    team = [df["season"], df["recent_team"]]
    if "target_share" not in df.columns:
        df["target_share"] = df["targets"] / df.groupby(team)["targets"].transform("sum")
    if "air_yards_share" not in df.columns:
        df["air_yards_share"] = df["receiving_air_yards"] / df.groupby(team)["receiving_air_yards"].transform("sum")

    experience = (df["season"] - df["rookie_year"]).fillna(df["years_exp"])
    df["experience"] = experience
    df["youth"] = -experience

    # Next-season points needed by whichever breakout route is easier.
    ppg_route = PPG_MULTIPLIER * df["ppg"] * SEASON_GAMES
    total_route = (TOTAL_MULTIPLIER * df[PPR]).where(df[PPR] <= _threshold(df, "cap"))
    easier_route = pd.concat([ppg_route, total_route], axis=1).min(axis=1)
    df["points_needed"] = easier_route.clip(lower=_threshold(df, "floor"))
    df["reachable"] = -df["points_needed"]
    return df


def define_breakout(df):
    """Attach each player's next-season points and flag the breakouts.

    A breakout clears his position's floor next season and either scores 50%
    more total points (players under the cap only) or 30% more points per game.
    """
    next_season = df[["player_id", "season", PPR, "ppg"]].copy()
    next_season["season"] -= 1
    next_season = next_season.rename(columns={PPR: "next_ppr", "ppg": "next_ppg"})

    merged = df.merge(next_season, on=["player_id", "season"], how="left")
    merged = merged.dropna(subset=["next_ppr"])
    total_jump = (merged[PPR] <= _threshold(merged, "cap")) & (merged["next_ppr"] >= TOTAL_MULTIPLIER * merged[PPR])
    ppg_jump = merged["next_ppg"] >= PPG_MULTIPLIER * merged["ppg"]
    merged["broke_out"] = (merged["next_ppr"] >= _threshold(merged, "floor")) & (total_jump | ppg_jump)
    return merged


def filter_eligible(df):
    """Keep players who played enough to judge.

    Players over the cap stay in, because they can still break out through the
    points-per-game route.
    """
    enough_volume = pd.Series(False, index=df.index)
    for position, (stat, minimum) in MIN_VOLUME.items():
        enough_volume |= (df["position"] == position) & (df[stat] >= minimum)
    return df[(df["games"] >= MIN_GAMES) & enough_volume]


def score_players(df, weights):
    """Add a 0-1 `score` column and sort by it, highest first."""
    scored = []
    for position, players in df.groupby("position"):
        players = players.copy()
        players["score"] = 0.0
        for stat, weight in weights[position].items():
            percentile = players.groupby("season")[stat].rank(pct=True).fillna(0.5)
            players["score"] += weight * percentile
        scored.append(players)
    return pd.concat(scored).sort_values("score", ascending=False)


def evaluate(df, season, position):
    """Return (top-N hit rate, baseline breakout rate) for one season and position."""
    pool = df[(df["season"] == season) & (df["position"] == position)]
    top = pool.sort_values("score", ascending=False).head(TOP_N)
    return top["broke_out"].mean(), pool["broke_out"].mean()


def backtest(scored):
    rows = []
    for position in POSITIONS:
        for season in EVAL_SEASONS:
            hit_rate, baseline = evaluate(scored, season, position)
            rows.append({"position": position, "season": season, "hit_rate": hit_rate, "baseline": baseline})
    return pd.DataFrame(rows)


def inspect(df):
    print("Columns:", ", ".join(df.columns))
    key = ["player_display_name", "position", "season", "games", PPR, "target_share", "air_yards_share"]
    key += [column for column in df.columns if column.startswith("ngs_")]
    print(df[key].head(10).to_string(index=False))
    print()


def print_breakout_counts(labeled):
    counts = labeled[labeled["broke_out"]].groupby(["season", "position"]).size().unstack(fill_value=0)
    print("Breakouts per season (the season shown is the year BEFORE the breakout)")
    print(counts.reindex(columns=POSITIONS, fill_value=0).to_string())
    print()


def print_backtest(results):
    """Print hit rate vs baseline for every version, one table per position."""
    print("Top-{} hit rate vs baseline, {}-{}".format(TOP_N, EVAL_SEASONS[0], EVAL_SEASONS[-1]))
    for position in POSITIONS:
        table = pd.DataFrame({"baseline": results["v1"].query("position == @position").set_index("season")["baseline"]})
        for name, result in results.items():
            table[name] = result.query("position == @position").set_index("season")["hit_rate"]
        table.loc["average"] = table.mean()
        print("\n" + position)
        print(table.to_string(float_format="{:.0%}".format))
    print("\nFinal model: {}\n".format(FINAL_VERSION))


def print_holdout(scored):
    """Check the final model on seasons that played no part in building it."""
    print("Holdout check on {} ({} only)".format(", ".join(str(s) for s in HOLDOUT_SEASONS), FINAL_VERSION))
    rows = []
    for position in POSITIONS:
        results = [evaluate(scored, season, position) for season in HOLDOUT_SEASONS]
        row = {"position": position}
        for season, (hit_rate, baseline) in zip(HOLDOUT_SEASONS, results):
            row[str(season)] = "{:.0%} vs {:.0%}".format(hit_rate, baseline)
        row["average"] = "{:.0%} vs {:.0%}".format(
            sum(r[0] for r in results) / len(results), sum(r[1] for r in results) / len(results)
        )
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("(each cell is top-{} hit rate vs baseline)\n".format(TOP_N))


def print_candidates(candidates):
    print("{} breakout candidates, scored on {} stats".format(LAST_SEASON + 1, LAST_SEASON))
    columns = {
        "player_display_name": "player", "recent_team": "team", "experience": "exp",
        "games": "games", PPR: "ppr", "ppg": "ppg", "points_needed": "needs", "score": "score",
    }
    for position in POSITIONS:
        top = candidates[candidates["position"] == position].head(TOP_N)
        print("\n" + position)
        print(top[list(columns)].rename(columns=columns).to_string(index=False, float_format="{:.2f}".format))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download the data from nflverse (needs internet)")
    parser.add_argument("--inspect", action="store_true", help="print the columns and first rows")
    args = parser.parse_known_args()[0]  # ignore the extra arguments notebooks pass in

    df = add_features(load_data(range(FIRST_SEASON, LAST_SEASON + 1), refresh=args.refresh))
    if args.inspect:
        inspect(df)

    labeled = define_breakout(df)
    print_breakout_counts(labeled)

    eligible = filter_eligible(labeled)
    results = {name: backtest(score_players(eligible, weights)) for name, weights in VERSIONS.items()}
    print_backtest(results)

    final_weights = VERSIONS[FINAL_VERSION]
    print_holdout(score_players(eligible, final_weights))
    this_year = filter_eligible(df[df["season"] == LAST_SEASON])
    print_candidates(score_players(this_year, final_weights))


if __name__ == "__main__":
    main()

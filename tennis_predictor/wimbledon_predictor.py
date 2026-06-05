#!/usr/bin/env python3
"""
Wimbledon 2026 Prediction System  (v3)
- ATP (Men's): XGBoost + isotonic calibration trained on all surfaces
- WTA (Women's): identical ML pipeline using JeffSackmann/tennis_wta data
- Tournament outcomes via Monte Carlo simulation (10,000 iterations each)
"""

import os
import warnings
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, log_loss, brier_score_loss,
    roc_auc_score, precision_score,
)

warnings.filterwarnings("ignore")

ATP_DIR = os.path.join(os.path.dirname(__file__), "..", "tennis_atp")
WTA_DIR = os.path.join(os.path.dirname(__file__), "..", "tennis_wta")
DATA_DIR = ATP_DIR   # kept for backward compat inside helpers
INITIAL_ELO = 1500
K_ALL   = 32
K_GRASS = 40       # higher K → surface ELO adapts faster to recent grass results
N_SIMS  = 10_000
TRAIN_END_YEAR = 2022   # train 2010-2022, calibrate 2023, validate 2024-2025
CALIB_YEAR     = 2023


# ---------------------------------------------------------------------------
# 1. DATA LOADING  (generic — used by both ATP and WTA)
# ---------------------------------------------------------------------------

def _load_matches(data_dir: str, prefix: str, levels: list,
                  start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    dfs = []
    for year in range(start_year, end_year + 1):
        path = os.path.join(data_dir, f"{prefix}_matches_{year}.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path, low_memory=False))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["tourney_level"].isin(levels)]
    invalid = df["score"].str.contains(r"W/O|RET|DEF|walkover", case=False, na=True)
    df = df[~invalid].copy()
    df["tourney_date"] = pd.to_datetime(df["tourney_date"].astype(str), format="%Y%m%d")
    df.sort_values("tourney_date", inplace=True)
    return df.reset_index(drop=True)


def load_atp_data(start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    # G=Grand Slam  M=Masters  A=ATP 500/250  F=Finals
    return _load_matches(ATP_DIR, "atp", ["G", "M", "A", "F"], start_year, end_year)


def load_wta_data(start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    # G=Grand Slam  PM=Premier Mandatory  P=Premier  I=International  F=Finals
    return _load_matches(WTA_DIR, "wta", ["G", "PM", "P", "I", "F"], start_year, end_year)


def _load_rankings(data_dir: str, filename: str) -> dict:
    df = pd.read_csv(os.path.join(data_dir, filename))
    latest = df["ranking_date"].max()
    return dict(zip(df.loc[df["ranking_date"] == latest, "player"],
                    df.loc[df["ranking_date"] == latest, "rank"]))


def load_current_rankings() -> dict:
    return _load_rankings(ATP_DIR, "atp_rankings_current.csv")


def load_wta_rankings() -> dict:
    return _load_rankings(WTA_DIR, "wta_rankings_current.csv")


def _load_player_names(data_dir: str, filename: str) -> dict:
    df = pd.read_csv(os.path.join(data_dir, filename), low_memory=False)
    df["full_name"] = df["name_first"].str.strip() + " " + df["name_last"].str.strip()
    return dict(zip(df["player_id"], df["full_name"]))


def load_player_names() -> dict:
    return _load_player_names(ATP_DIR, "atp_players.csv")


def load_wta_player_names() -> dict:
    return _load_player_names(WTA_DIR, "wta_players.csv")


# ---------------------------------------------------------------------------
# 2. ELO ENGINE  (overall + grass ELO + H2H — all tracked incrementally)
# ---------------------------------------------------------------------------

class EloEngine:
    """
    Tracks per-player overall ELO, grass ELO, and H2H win rates.
    Snapshots stored BEFORE each match update so they can be used as features
    without any look-ahead leakage.
    """

    def __init__(self, k_all: float = K_ALL, k_grass: float = K_GRASS):
        self.k_all   = k_all
        self.k_grass = k_grass
        self.elo_all:   dict = {}
        self.elo_grass: dict = {}
        self._h2h:      dict = {}   # (min_id, max_id) → [min_wins, max_wins]
        self._snapshots: list = []

    # -- helpers ---------------------------------------------------------------

    def _get(self, store, pid):
        return store.get(pid, INITIAL_ELO)

    @staticmethod
    def _expected(a: float, b: float) -> float:
        return 1.0 / (1.0 + 10.0 ** ((b - a) / 400.0))

    def _h2h_wr(self, a_id, b_id) -> float:
        """Return A's historical H2H win rate vs B (0.5 if < 2 meetings)."""
        k = (min(a_id, b_id), max(a_id, b_id))
        rec = self._h2h.get(k)
        if rec is None:
            return 0.5
        w_min, w_max = rec
        total = w_min + w_max
        if total < 2:
            return 0.5
        return (w_min if a_id == k[0] else w_max) / total

    def _update_h2h(self, winner_id, loser_id):
        k = (min(winner_id, loser_id), max(winner_id, loser_id))
        if k not in self._h2h:
            self._h2h[k] = [0, 0]
        if winner_id == k[0]:
            self._h2h[k][0] += 1
        else:
            self._h2h[k][1] += 1

    # -- main interface --------------------------------------------------------

    def process_match(self, winner_id, loser_id, surface: str, date):
        wa = self._get(self.elo_all,   winner_id)
        la = self._get(self.elo_all,   loser_id)
        wg = self._get(self.elo_grass, winner_id)
        lg = self._get(self.elo_grass, loser_id)

        exp_w = self._expected(wa, la)
        w_h2h = self._h2h_wr(winner_id, loser_id)   # snapshot BEFORE update

        self._snapshots.append({
            "winner_id":       winner_id,
            "loser_id":        loser_id,
            "surface":         surface,
            "date":            date,
            "w_elo_all_pre":   wa,
            "l_elo_all_pre":   la,
            "w_elo_grass_pre": wg,
            "l_elo_grass_pre": lg,
            "w_h2h_wr_pre":    w_h2h,
        })

        # Update overall ELO for every match
        self.elo_all[winner_id] = wa + self.k_all * (1 - exp_w)
        self.elo_all[loser_id]  = la + self.k_all * (0 - (1 - exp_w))

        # Update grass ELO only for grass matches
        if surface == "Grass":
            exp_g = self._expected(wg, lg)
            self.elo_grass[winner_id] = wg + self.k_grass * (1 - exp_g)
            self.elo_grass[loser_id]  = lg + self.k_grass * (0 - (1 - exp_g))

        self._update_h2h(winner_id, loser_id)   # update AFTER snapshot

    def process_dataframe(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            self.process_match(
                row["winner_id"], row["loser_id"], row["surface"], row["tourney_date"]
            )

    def get_snapshots_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._snapshots)

    def grass_elo(self, pid) -> float:
        return self._get(self.elo_grass, pid)

    def all_elo(self, pid) -> float:
        return self._get(self.elo_all, pid)


# ---------------------------------------------------------------------------
# 3. ROLLING STATS
# ---------------------------------------------------------------------------

def compute_form_window(df: pd.DataFrame, lookback_days: int,
                        surface: str = None, min_matches: int = 5) -> dict:
    """Win rate over the trailing window, optionally filtered to one surface."""
    cutoff = df["tourney_date"].max() - pd.Timedelta(days=lookback_days)
    sub = df[df["tourney_date"] >= cutoff]
    if surface:
        sub = sub[sub["surface"] == surface]
    wins   = sub.groupby("winner_id").size().rename("wins")
    losses = sub.groupby("loser_id").size().rename("losses")
    stats  = pd.concat([wins, losses], axis=1).fillna(0)
    stats["total"] = stats["wins"] + stats["losses"]
    stats = stats[stats["total"] >= min_matches]
    stats["wr"] = stats["wins"] / stats["total"]
    return stats["wr"].to_dict()


def compute_bo5_win_rate(df: pd.DataFrame) -> dict:
    """Win rate in Grand Slam matches (best-of-5). Wimbledon is always bo5."""
    gs = df[df["tourney_level"] == "G"]
    wins   = gs.groupby("winner_id").size().rename("wins")
    losses = gs.groupby("loser_id").size().rename("losses")
    stats  = pd.concat([wins, losses], axis=1).fillna(0)
    stats["total"] = stats["wins"] + stats["losses"]
    stats = stats[stats["total"] >= 5]
    stats["bo5_wr"] = stats["wins"] / stats["total"]
    return stats["bo5_wr"].to_dict()


def compute_grass_serve_stats(df: pd.DataFrame) -> dict:
    """Return {player_id: (ace_rate, first_serve_pct, bp_saved_pct)} on grass."""
    grass = df[(df["surface"] == "Grass") & df["w_svpt"].notna()].copy()
    w = grass[["winner_id","w_ace","w_svpt","w_1stIn","w_bpSaved","w_bpFaced"]].copy()
    l = grass[["loser_id", "l_ace","l_svpt","l_1stIn","l_bpSaved","l_bpFaced"]].copy()
    w.columns = l.columns = ["pid","ace","svpt","first_in","bp_saved","bp_faced"]
    agg = pd.concat([w, l], ignore_index=True).groupby("pid").sum()
    agg = agg[agg["svpt"] >= 100]
    agg["ace_rate"]       = agg["ace"]      / agg["svpt"]
    agg["first_serve_pct"]= agg["first_in"] / agg["svpt"]
    agg["bp_saved_pct"]   = (agg["bp_saved"] / agg["bp_faced"].replace(0, np.nan)).fillna(0.62)
    return {pid: (r["ace_rate"], r["first_serve_pct"], r["bp_saved_pct"])
            for pid, r in agg.iterrows()}


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING  (all surfaces — grass features masked by is_grass)
# ---------------------------------------------------------------------------

# 14 features:
FEATURE_NAMES = [
    "elo_all_diff",        # overall ELO difference
    "elo_grass_diff",      # grass ELO diff  × is_grass
    "grass_wr_3y_diff",    # 3-year grass win rate diff  × is_grass
    "form_365d_diff",      # 1-year all-surface form diff
    "form_90d_diff",       # 90-day all-surface form diff
    "grass_form_90d_diff", # 90-day grass form diff  × is_grass
    "bo5_wr_diff",         # Grand Slam win rate diff
    "rank_diff",           # rank diff (positive = A is better ranked)
    "age_diff",            # A age − B age
    "ace_rate_diff",       # grass ace rate diff  × is_grass
    "first_serve_diff",    # grass 1st-serve % diff  × is_grass
    "bp_saved_diff",       # grass BP-saved % diff  × is_grass
    "h2h_wr_centered",     # A's H2H win rate − 0.5
    "is_grass",            # surface indicator
]

_DS = (0.06, 0.62, 0.62)   # default serve stats


def _make_row(
    a_id, b_id,
    a_elo_all, b_elo_all,
    a_elo_g,   b_elo_g,
    grass_wr_3y: dict,
    form_365d: dict,
    form_90d: dict,
    grass_form_90d: dict,
    bo5_wr: dict,
    rankings: dict,
    a_age, b_age,
    serve_stats: dict,
    a_h2h_wr: float,   # A's h2h win rate vs B
    is_grass: float,
) -> list:
    a_srv = serve_stats.get(a_id, _DS)
    b_srv = serve_stats.get(b_id, _DS)
    return [
        a_elo_all  - b_elo_all,
        (a_elo_g   - b_elo_g)                                   * is_grass,
        (grass_wr_3y.get(a_id, 0.5) - grass_wr_3y.get(b_id, 0.5)) * is_grass,
        form_365d.get(a_id, 0.5)     - form_365d.get(b_id, 0.5),
        form_90d.get(a_id, 0.5)      - form_90d.get(b_id, 0.5),
        (grass_form_90d.get(a_id, 0.5) - grass_form_90d.get(b_id, 0.5)) * is_grass,
        bo5_wr.get(a_id, 0.5)        - bo5_wr.get(b_id, 0.5),
        float(rankings.get(b_id, 200) - rankings.get(a_id, 200)),
        float(a_age) - float(b_age),
        (a_srv[0] - b_srv[0]) * is_grass,
        (a_srv[1] - b_srv[1]) * is_grass,
        (a_srv[2] - b_srv[2]) * is_grass,
        a_h2h_wr - 0.5,
        is_grass,
    ]


def build_features(df, snapshots, grass_wr_3y, form_365d, form_90d,
                   grass_form_90d, bo5_wr, rankings, serve_stats):
    """
    Build symmetric training examples from ALL surface matches.
    Grass-specific features are zeroed for non-grass matches via is_grass mask.
    Sample weight: grass=3.0, hard/clay=1.0
    """
    snap_dict = {(s["winner_id"], s["loser_id"], s["date"]): s
                 for _, s in snapshots.iterrows()}

    X_rows, y_rows, w_rows = [], [], []
    for _, m in df.iterrows():
        w_id   = m["winner_id"]
        l_id   = m["loser_id"]
        date   = m["tourney_date"]
        is_g   = 1.0 if m["surface"] == "Grass" else 0.0
        weight = 3.0 if is_g else 1.0

        snap = snap_dict.get((w_id, l_id, date))
        if snap is None:
            continue

        w_h2h = float(snap["w_h2h_wr_pre"])  # winner's H2H wr before this match

        common = dict(
            grass_wr_3y=grass_wr_3y, form_365d=form_365d, form_90d=form_90d,
            grass_form_90d=grass_form_90d, bo5_wr=bo5_wr, rankings=rankings,
            serve_stats=serve_stats, is_grass=is_g,
        )
        w_age = m.get("winner_age", 26) or 26
        l_age = m.get("loser_age",  26) or 26

        # Winner perspective → label 1
        X_rows.append(_make_row(
            w_id, l_id,
            snap["w_elo_all_pre"], snap["l_elo_all_pre"],
            snap["w_elo_grass_pre"], snap["l_elo_grass_pre"],
            a_age=w_age, b_age=l_age,
            a_h2h_wr=w_h2h, **common,
        ))
        y_rows.append(1)
        w_rows.append(weight)

        # Loser perspective → label 0
        X_rows.append(_make_row(
            l_id, w_id,
            snap["l_elo_all_pre"], snap["w_elo_all_pre"],
            snap["l_elo_grass_pre"], snap["w_elo_grass_pre"],
            a_age=l_age, b_age=w_age,
            a_h2h_wr=(1 - w_h2h), **common,
        ))
        y_rows.append(0)
        w_rows.append(weight)

    return (np.array(X_rows, dtype=np.float32),
            np.array(y_rows, dtype=np.int32),
            np.array(w_rows, dtype=np.float32))


# ---------------------------------------------------------------------------
# 5. MODEL TRAINING & CALIBRATION
# ---------------------------------------------------------------------------

def _build_stats(df_context: pd.DataFrame):
    """Compute all rolling stat dicts from a context dataframe."""
    return {
        "grass_wr_3y":    compute_form_window(df_context, 3*365, surface="Grass", min_matches=5),
        "form_365d":      compute_form_window(df_context, 365,   min_matches=10),
        "form_90d":       compute_form_window(df_context, 90,    min_matches=5),
        "grass_form_90d": compute_form_window(df_context, 90,    surface="Grass", min_matches=3),
        "bo5_wr":         compute_bo5_win_rate(df_context),
        "serve_stats":    compute_grass_serve_stats(df_context),
    }


def _print_metrics(y_true, proba, label="", tag="ATP"):
    """Print comprehensive evaluation metrics."""
    y_pred = (proba >= 0.5).astype(int)
    acc    = accuracy_score(y_true, y_pred)
    ll     = log_loss(y_true, proba)
    brier  = brier_score_loss(y_true, proba)
    auc    = roc_auc_score(y_true, proba)

    print(f"\n[{tag}] {label}Evaluation metrics:")
    print(f"      Accuracy  : {acc:.4f}")
    print(f"      Log-loss  : {ll:.4f}")
    print(f"      Brier     : {brier:.4f}  (lower = better, perfect = 0)")
    print(f"      AUC-ROC   : {auc:.4f}")

    print(f"\n[{tag}] Precision at confidence thresholds:")
    print(f"      {'Threshold':<12} {'Precision':>10} {'Coverage':>10} {'n samples':>10}")
    for thresh in [0.55, 0.60, 0.65, 0.70, 0.75]:
        mask = proba >= thresh
        n = mask.sum()
        if n < 10:
            print(f"      {thresh:.0%}         {'—':>10} {'—':>10} {n:>10}")
            continue
        prec = y_true[mask].mean()
        cov  = mask.mean()
        print(f"      {thresh:.0%}         {prec:>10.4f} {cov:>10.2%} {n:>10,}")

    print(f"\n[{tag}] Calibration (predicted prob vs actual win rate):")
    print(f"      {'Bin':<14} {'Predicted':>10} {'Actual':>10} {'n':>8}")
    bins = [(0.45,0.55),(0.55,0.65),(0.65,0.75),(0.75,0.85),(0.85,1.01)]
    for lo, hi in bins:
        mask = (proba >= lo) & (proba < hi)
        n = mask.sum()
        if n < 5:
            continue
        pred_mean = proba[mask].mean()
        act_mean  = y_true[mask].mean()
        flag = " ✓" if abs(pred_mean - act_mean) < 0.05 else " ✗ (miscal.)"
        print(f"      {lo:.0%}–{hi:.0%}       {pred_mean:>10.3f} {act_mean:>10.3f} {n:>8,}{flag}")

    return ll  # return so callers can compare


def train_atp_model(df_train, df_calib, df_val):
    """
    1. Train XGBoost on df_train (2010-2022)
    2. Calibrate with isotonic regression on df_calib (2023)
    3. Evaluate uncalibrated vs calibrated on df_val (2024-2025)
    """
    rankings = load_current_rankings()

    # --- train ---------------------------------------------------------------
    print("\n[ATP] Building ELO + features for training set (2010–2022)...")
    engine = EloEngine()
    engine.process_dataframe(df_train)
    snaps_tr = engine.get_snapshots_df()
    stats_tr = _build_stats(df_train)
    X_tr, y_tr, w_tr = build_features(df_train, snaps_tr, rankings=rankings, **stats_tr)
    grass_tr = (snaps_tr["surface"] == "Grass").sum()
    print(f"[ATP] Training samples: {len(X_tr):,}  "
          f"(all surfaces × 2 symmetric; grass matches: {grass_tr:,})")

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_tr, y_tr, sample_weight=w_tr)

    # --- calibrate -----------------------------------------------------------
    print("[ATP] Calibrating on 2023 data (isotonic regression)...")
    engine.process_dataframe(df_calib)
    snaps_cal = engine.get_snapshots_df()
    ctx_cal   = pd.concat([df_train, df_calib])
    stats_cal = _build_stats(ctx_cal)
    X_cal, y_cal, _ = build_features(df_calib, snaps_cal, rankings=rankings, **stats_cal)
    # Only calibrate on grass examples — that's what Wimbledon will use
    grass_mask = X_cal[:, FEATURE_NAMES.index("is_grass")] == 1.0
    X_cal_g, y_cal_g = X_cal[grass_mask], y_cal[grass_mask]
    print(f"[ATP] Calibration grass samples: {len(X_cal_g):,}")

    calibrated = CalibratedClassifierCV(xgb, cv="prefit", method="isotonic")
    calibrated.fit(X_cal_g, y_cal_g)

    # --- evaluate ------------------------------------------------------------
    print("[ATP] Evaluating on validation set (2024–2025)...")
    engine.process_dataframe(df_val)
    snaps_val = engine.get_snapshots_df()
    ctx_val   = pd.concat([df_train, df_calib, df_val])
    stats_val = _build_stats(ctx_val)
    X_val, y_val, _ = build_features(df_val, snaps_val, rankings=rankings, **stats_val)
    # Evaluate only on grass matches (what matters for Wimbledon)
    g_mask = X_val[:, FEATURE_NAMES.index("is_grass")] == 1.0
    X_g, y_g = X_val[g_mask], y_val[g_mask]
    print(f"[ATP] Validation grass samples: {len(X_g):,}")

    raw_proba = xgb.predict_proba(X_g)[:, 1]
    cal_proba = calibrated.predict_proba(X_g)[:, 1]

    raw_ll = _print_metrics(y_g, raw_proba, label="Uncalibrated XGBoost — ", tag="ATP")
    cal_ll = _print_metrics(y_g, cal_proba, label="Calibrated model — ",     tag="ATP")

    # Fall back to uncalibrated if isotonic overfit the small calibration set
    if cal_ll > raw_ll:
        print("\n[ATP] Calibration increased loss — using uncalibrated model for predictions.")
        final_model = xgb
    else:
        final_model = calibrated

    print("\n[ATP] Feature importances (XGBoost):")
    for name, imp in sorted(zip(FEATURE_NAMES, xgb.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"       {name:<24} {imp:.4f}")

    return final_model, engine, stats_val, rankings


# ---------------------------------------------------------------------------
# 6. ATP WIMBLEDON 2026 SIMULATION
# ---------------------------------------------------------------------------

def get_draw_players(engine, rankings, player_names, n=128):
    rows = []
    for pid, rank in sorted(rankings.items(), key=lambda x: x[1])[:n]:
        rows.append({
            "player_id": pid,
            "name":      player_names.get(pid, f"Player {pid}"),
            "rank":      rank,
            "elo_all":   engine.all_elo(pid),
            "elo_grass": engine.grass_elo(pid),
        })
    return pd.DataFrame(rows)


def _match_prob(model, p1, p2, stats, rankings):
    """P(p1 beats p2) on grass using the calibrated model."""
    row = _make_row(
        p1["player_id"], p2["player_id"],
        p1["elo_all"], p2["elo_all"],
        p1["elo_grass"], p2["elo_grass"],
        grass_wr_3y    = stats["grass_wr_3y"],
        form_365d      = stats["form_365d"],
        form_90d       = stats["form_90d"],
        grass_form_90d = stats["grass_form_90d"],
        bo5_wr         = stats["bo5_wr"],
        rankings       = rankings,
        a_age=26, b_age=26,
        serve_stats    = stats["serve_stats"],
        a_h2h_wr       = 0.5,   # no H2H data at prediction time without draw info
        is_grass       = 1.0,
    )
    return float(model.predict_proba(np.array([row], dtype=np.float32))[0][1])


def simulate_atp_wimbledon(model, draw, stats, rankings, n_sims=N_SIMS):
    players  = draw.to_dict("records")
    n        = len(players)
    wins     = np.zeros(n, dtype=np.int32)
    rng      = np.random.default_rng(42)
    seeds    = players[:32]
    unseeded = players[32:]

    seed_pos   = [0, 127, 63, 64, 31, 96, 32, 95]
    slots_9_16 = [15, 16, 47, 48, 79, 80, 111, 112]
    slots_17_32 = [8, 23, 40, 55, 72, 87, 104, 119,
                   7, 24, 39, 56, 71, 88, 103, 120]

    for _ in range(n_sims):
        rng.shuffle(unseeded)
        bracket = [None] * 128

        for i, pos in enumerate(seed_pos[:8]):
            bracket[pos] = seeds[i]

        s916 = seeds[8:16].copy(); rng.shuffle(s916)
        for i, pos in enumerate(slots_9_16):
            bracket[pos] = s916[i]

        s1732 = seeds[16:32].copy(); rng.shuffle(s1732)
        for i, pos in enumerate(slots_17_32):
            bracket[pos] = s1732[i]

        uns = iter(unseeded)
        for j in range(128):
            if bracket[j] is None:
                bracket[j] = next(uns)

        curr = bracket[:]
        while len(curr) > 1:
            nxt = []
            for i in range(0, len(curr), 2):
                p = _match_prob(model, curr[i], curr[i+1], stats, rankings)
                nxt.append(curr[i] if rng.random() < p else curr[i+1])
            curr = nxt

        idx = next(i for i, p in enumerate(players) if p["player_id"] == curr[0]["player_id"])
        wins[idx] += 1

    result = draw.copy()
    result["win_pct"] = wins / n_sims * 100
    return result.sort_values("win_pct", ascending=False)


# ---------------------------------------------------------------------------
# 7. WTA PIPELINE  (mirrors ATP exactly — real match data from tennis_wta)
# ---------------------------------------------------------------------------

def train_wta_model(df_train, df_calib, df_val):
    """Same structure as train_atp_model but using WTA data."""
    rankings = load_wta_rankings()

    print("\n[WTA] Building ELO + features for training set (2010–2022)...")
    engine = EloEngine()
    engine.process_dataframe(df_train)
    snaps_tr = engine.get_snapshots_df()
    stats_tr = _build_stats(df_train)
    X_tr, y_tr, w_tr = build_features(df_train, snaps_tr, rankings=rankings, **stats_tr)
    grass_tr = (snaps_tr["surface"] == "Grass").sum()
    print(f"[WTA] Training samples: {len(X_tr):,}  "
          f"(all surfaces × 2 symmetric; grass matches: {grass_tr:,})")

    xgb = XGBClassifier(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_tr, y_tr, sample_weight=w_tr)

    print("[WTA] Calibrating on 2023 data (isotonic regression)...")
    engine.process_dataframe(df_calib)
    snaps_cal = engine.get_snapshots_df()
    stats_cal = _build_stats(pd.concat([df_train, df_calib]))
    X_cal, y_cal, _ = build_features(df_calib, snaps_cal, rankings=rankings, **stats_cal)
    grass_mask = X_cal[:, FEATURE_NAMES.index("is_grass")] == 1.0
    X_cal_g, y_cal_g = X_cal[grass_mask], y_cal[grass_mask]
    print(f"[WTA] Calibration grass samples: {len(X_cal_g):,}")

    calibrated = CalibratedClassifierCV(xgb, cv="prefit", method="isotonic")
    calibrated.fit(X_cal_g, y_cal_g)

    print("[WTA] Evaluating on validation set (2024–2025)...")
    engine.process_dataframe(df_val)
    snaps_val = engine.get_snapshots_df()
    stats_val = _build_stats(pd.concat([df_train, df_calib, df_val]))
    X_val, y_val, _ = build_features(df_val, snaps_val, rankings=rankings, **stats_val)
    g_mask = X_val[:, FEATURE_NAMES.index("is_grass")] == 1.0
    X_g, y_g = X_val[g_mask], y_val[g_mask]
    print(f"[WTA] Validation grass samples: {len(X_g):,}")

    raw_proba = xgb.predict_proba(X_g)[:, 1]
    cal_proba = calibrated.predict_proba(X_g)[:, 1]

    raw_ll = _print_metrics(y_g, raw_proba, label="Uncalibrated XGBoost — ", tag="WTA")
    cal_ll = _print_metrics(y_g, cal_proba, label="Calibrated model — ",     tag="WTA")

    if cal_ll > raw_ll:
        print("\n[WTA] Calibration increased loss — using uncalibrated model for predictions.")
        final_model = xgb
    else:
        final_model = calibrated

    print("\n[WTA] Feature importances (XGBoost):")
    for name, imp in sorted(zip(FEATURE_NAMES, xgb.feature_importances_),
                             key=lambda x: -x[1]):
        print(f"       {name:<24} {imp:.4f}")

    return final_model, engine, stats_val, rankings


def simulate_wta_wimbledon(model, draw, stats, rankings, n_sims=N_SIMS):
    """Monte Carlo simulation of WTA Wimbledon 2026 (128-player bracket)."""
    return simulate_atp_wimbledon(model, draw, stats, rankings, n_sims)


# ---------------------------------------------------------------------------
# 8. MAIN
# ---------------------------------------------------------------------------

def print_results(label, df, top_n=20):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  {'Rank':<5} {'Player':<30} {'Country':<8} {'Win %':>7}")
    print(f"  {'-'*52}")
    for i, (_, row) in enumerate(df.head(top_n).iterrows(), 1):
        country = row.get("country", row.get("ioc", "—"))
        print(f"  {i:<5} {row['name']:<30} {country:<8} {row['win_pct']:>6.2f}%")


def _run_tour(tour: str, load_data_fn, load_rankings_fn, load_names_fn,
              train_fn, simulate_fn):
    """Generic pipeline runner for ATP or WTA."""
    tag = f"[{tour}]"
    print(f"\n{tag} Loading match data (2010–2026)...")
    df_all   = load_data_fn(start_year=2010, end_year=2026)
    df_train = df_all[df_all["tourney_date"].dt.year <= TRAIN_END_YEAR]
    df_calib = df_all[df_all["tourney_date"].dt.year == CALIB_YEAR]
    df_val   = df_all[(df_all["tourney_date"].dt.year >= 2024) &
                      (df_all["tourney_date"].dt.year <= 2025)]
    df_2026  = df_all[df_all["tourney_date"].dt.year == 2026]
    print(f"{tag} Train: {len(df_train):,}  Calib: {len(df_calib):,}  "
          f"Val: {len(df_val):,}  2026: {len(df_2026):,}")

    model, engine, _, rankings = train_fn(df_train, df_calib, df_val)

    print(f"\n{tag} Updating ELO with 2026 results...")
    engine.process_dataframe(df_2026)
    stats_full = _build_stats(df_all)

    player_names = load_names_fn()
    draw = get_draw_players(engine, rankings, player_names)

    pid_to_ioc = {}
    for _, row in df_all[["winner_id", "winner_ioc"]].drop_duplicates().iterrows():
        pid_to_ioc[row["winner_id"]] = row["winner_ioc"]
    draw["country"] = draw["player_id"].map(pid_to_ioc).fillna("—")

    print(f"\n{tag} Running {N_SIMS:,} Wimbledon 2026 simulations...")
    results = simulate_fn(model, draw, stats_full, rankings)
    return results


def main():
    print("=" * 60)
    print("  WIMBLEDON 2026 PREDICTION SYSTEM  (v3)")
    print("  ELO + XGBoost + Isotonic Calibration")
    print("=" * 60)

    atp_results = _run_tour(
        "ATP",
        load_atp_data, load_current_rankings, load_player_names,
        train_atp_model, simulate_atp_wimbledon,
    )
    print_results("ATP MEN'S WIMBLEDON 2026 — WIN PROBABILITIES", atp_results)

    wta_results = _run_tour(
        "WTA",
        load_wta_data, load_wta_rankings, load_wta_player_names,
        train_wta_model, simulate_wta_wimbledon,
    )
    print_results("WTA WOMEN'S WIMBLEDON 2026 — WIN PROBABILITIES", wta_results)

    atp_fav = atp_results.iloc[0]
    wta_fav = wta_results.iloc[0]
    print(f"\n{'='*60}")
    print(f"  PREDICTED WINNERS")
    print(f"  Men's:   {atp_fav['name']} ({atp_fav['country']})  — {atp_fav['win_pct']:.1f}%")
    print(f"  Women's: {wta_fav['name']} ({wta_fav['country']})  — {wta_fav['win_pct']:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

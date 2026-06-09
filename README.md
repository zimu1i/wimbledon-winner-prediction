# Wimbledon 2026 Winner Prediction

Machine learning system to predict the winner of the 2026 Wimbledon Championships for both the Men's (ATP) and Women's (WTA) draws using historical match data, ELO ratings, XGBoost with isotonic calibration, and SHAP-based feature analysis.

## Key findings

- **Grass-specific features drive 47% of predictive signal.** SHAP analysis on the 2024–2025 grass holdout shows that 3-year grass win rate (mean |SHAP| = 0.343), break-point save rate (0.227), and ace rate (0.154) collectively explain nearly half the model's decisions — capturing players who structurally outperform their ATP ranking on this surface (e.g. Rybakina, Djokovic at his peak).

- **ATP ranking alone achieves 65.6% accuracy on grass; the full model reaches 71.5% — a +5.8pp lift (+8.7pp AUC).** ELO accounts for +3.0pp of that gain over raw ranking; the remaining +2.8pp comes from surface-specific serve stats, Grand Slam win rate, and isotonic probability calibration.

- **79% of high-confidence wrong predictions are genuine rank upsets.** Among the 48 matches where the model predicted ≥70% probability but was wrong, 38 (79%) involved a player ranked 80+ positions lower winning. The model's systematic blind spot is extreme underdog victories — cases where grass-court variance overwhelms every measurable signal.

*Full analysis with charts in [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb) · Design decisions in [`METHODOLOGY.md`](METHODOLOGY.md)*

---

## Quick start

```bash
pip install -r requirements.txt

# Run predictions
python3 tennis_predictor/wimbledon_predictor.py

# Generate evaluation plots (calibration curve, SHAP, upset analysis)
python3 metrics/evaluate_model.py

# Open analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

## Results (as of June 9, 2026 — post French Open)

### ⚠️ Alcaraz withdrawal — draw updated

Carlos Alcaraz (ESP, World No. 2) has announced he will not compete at Wimbledon 2026 due to health issues. Alcaraz was the pre-tournament favourite at **33.1%** win probability in the original model — the single largest share of any player.

His withdrawal significantly reshuffles the draw:

| | Original (pre-French Open) | Updated (Alcaraz out + French Open data) |
|---|---|---|
| **Winner** | Carlos Alcaraz (ESP) — 33.1% | Jannik Sinner (ITA) — 35.2% |
| **Runner-up** | Jannik Sinner (ITA) — 22.1% | Novak Djokovic (SRB) — 21.4% |

With Alcaraz's ~33% redistributed and French Open form baked in, **Sinner** now leads decisively at 35.2% — driven by updated ELO after his French Open results and the highest remaining Grand Slam win rate. Djokovic holds second at 21.4% from accumulated grass ELO across his six Wimbledon titles. Zverev climbs from 9.2% to 15.4%. The WTA draw sees Madison Keys rise to 4th (6.4%) on the back of strong 2026 hardcourt and clay form.

**Men's (ATP)** — Alcaraz withdrawn
| Rank | Player | Country | Win % |
|------|--------|---------|-------|
| 1 | Jannik Sinner | ITA | 35.2% |
| 2 | Novak Djokovic | SRB | 21.4% |
| 3 | Alexander Zverev | GER | 15.4% |
| 4 | Taylor Fritz | USA | 3.2% |
| 5 | Ben Shelton | USA | 2.9% |
| 6 | Daniil Medvedev | RUS | 2.9% |
| 7 | Jack Draper | GBR | 2.5% |
| 8 | Lorenzo Musetti | ITA | 1.7% |

**Women's (WTA)**
| Rank | Player | Country | Win % |
|------|--------|---------|-------|
| 1 | Aryna Sabalenka | BLR | 19.6% |
| 2 | Iga Swiatek | POL | 15.8% |
| 3 | Elena Rybakina | KAZ | 13.9% |
| 4 | Madison Keys | USA | 6.4% |
| 5 | Elina Svitolina | UKR | 4.9% |
| 6 | Jessica Pegula | USA | 4.4% |
| 7 | Amanda Anisimova | USA | 4.3% |
| 8 | Mirra Andreeva | RUS | 3.5% |

## Project structure

```
Wimbledon-Prediction/
├── tennis_predictor/           # ML pipeline package
│   ├── config.py               # constants (paths, ELO params, N_SIMS)
│   ├── data.py                 # CSV-based data loaders (pandas)
│   ├── duckdb_data.py          # SQL-based data loaders (DuckDB)
│   ├── elo.py                  # EloEngine — overall + grass ELO + H2H
│   ├── stats.py                # rolling win rates, serve stats, bo5 rate
│   ├── features.py             # feature engineering (14 features)
│   ├── model.py                # XGBoost training + isotonic calibration
│   ├── simulate.py             # Monte Carlo bracket simulation
│   ├── train.py                # pipeline orchestrator + entry point
│   └── wimbledon_predictor.py  # thin wrapper (calls train.main)
├── queries/                    # DuckDB SQL files
│   ├── 01_load_matches.sql     # load + clean all ATP match data
│   ├── 02_grass_stats.sql      # per-player grass serve stats
│   ├── 03_upset_rate_by_round.sql  # Wimbledon upset rates by round
│   └── 04_elo_vs_ranking.sql   # ranking accuracy baseline vs ELO
├── metrics/
│   └── evaluate_model.py       # saves calibration curve, SHAP, upset plots
├── notebooks/
│   └── analysis.ipynb          # deep-dive: SHAP, failure modes, ELO vs ranking
├── METHODOLOGY.md              # design decisions and trade-offs
├── tennis_atp/                 # JeffSackmann ATP match data (1968–2026)
└── tennis_wta/                 # JeffSackmann WTA match data (1968–2026)
```

## Approach

### 1. Data pipeline — DuckDB SQL

Match data is loaded via DuckDB, which reads all yearly CSV files in a single SQL scan across 60,000+ records with no ETL step. Ad-hoc queries can be run from Python using the `sql()` helper:

```python
from tennis_predictor.duckdb_data import sql

sql("""
    SELECT surface, COUNT(*) AS matches
    FROM read_csv_auto('tennis_atp/atp_matches_*.csv', union_by_name=true)
    WHERE tourney_level IN ('G','M','A','F')
      AND YEAR(STRPTIME(CAST(tourney_date AS VARCHAR), '%Y%m%d')) >= 2010
    GROUP BY surface ORDER BY matches DESC
""")
```

### 2. ELO ratings (ATP and WTA)

Computed incrementally from all main-tour matches (2010–2026):
- **Overall ELO** (K=32, all surfaces)
- **Grass-specific ELO** (K=40 — higher K means faster adaptation to recent grass results)
- **Head-to-head win rates** tracked per player pair

All snapshots are taken **before** each match update — zero look-ahead leakage.

### 3. XGBoost classifier — 14 features

Trained on **all surfaces** (64,114 ATP / 58,694 WTA examples) with grass samples upweighted **3×**. Grass-specific features are masked to zero on hard/clay via an `is_grass` indicator so they only contribute on grass.

| Feature | Type |
|---------|------|
| `elo_all_diff` | Overall ELO difference |
| `elo_grass_diff` | Grass ELO diff × is_grass |
| `grass_wr_3y_diff` | 3-year grass win rate diff × is_grass |
| `form_365d_diff` | 1-year all-surface form diff |
| `form_90d_diff` | 90-day form diff |
| `grass_form_90d_diff` | 90-day grass form diff × is_grass |
| `bo5_wr_diff` | Grand Slam (best-of-5) win rate diff |
| `rank_diff` | Ranking difference |
| `age_diff` | Age difference |
| `ace_rate_diff` | Grass ace rate diff × is_grass |
| `first_serve_diff` | Grass 1st-serve % diff × is_grass |
| `bp_saved_diff` | Break points saved % diff × is_grass |
| `h2h_wr_centered` | Head-to-head win rate − 0.5 |
| `is_grass` | Surface indicator |

### 4. Isotonic calibration

XGBoost tends to output overconfident probabilities. An isotonic regression layer is fitted on 2023 grass matches to correct this. The pipeline automatically falls back to the uncalibrated model if calibration worsens log-loss (relevant for WTA where calibration samples are smaller).

### 5. SHAP feature analysis

SHAP (SHapley Additive exPlanations) is used to explain individual predictions and identify which features drive the model's confidence. Run `metrics/evaluate_model.py` or open `notebooks/analysis.ipynb` to see the full analysis.

### 6. Monte Carlo simulation

10,000 iterations of the 128-player bracket with standard Wimbledon seeding placement (seeds 1–8 fixed, 9–16 and 17–32 randomised within their designated slots). Converges to < 0.3% standard error per player.

**Model performance (2024–2025 validation, grass matches only):**

| Metric | ATP | WTA |
|--------|-----|-----|
| Accuracy | 71.6% | 68.9% |
| Log-loss | 0.568 | 0.621 |
| AUC-ROC | 0.774 | 0.748 |
| Precision @ 70%+ confidence | 79.6% | 76.3% |

See [METHODOLOGY.md](METHODOLOGY.md) for full design decisions, hyperparameter search results, and known limitations.

## Data

- ATP: [`tennis_atp`](tennis_atp/) — Jeff Sackmann / [github.com/JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp)
- WTA: [`tennis_wta`](tennis_wta/) — Jeff Sackmann / [github.com/JeffSackmann/tennis_wta](https://github.com/JeffSackmann/tennis_wta)

Both cover 1968–2026. Only main-tour events are used (ATP: Grand Slams, Masters, 500/250, Finals; WTA: Grand Slams, Premier Mandatory/Premier/International, Finals). Retirements and walkovers are excluded.

## Dependencies

```
numpy>=2.0
pandas>=2.0
scikit-learn>=1.4
xgboost>=2.0
duckdb>=1.0
shap>=0.44
matplotlib>=3.8
seaborn>=0.13
jupyter>=1.0
```

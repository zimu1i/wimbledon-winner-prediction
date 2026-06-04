# Wimbledon 2026 Winner Prediction

Machine learning system to predict the winner of the 2026 Wimbledon Championships for both the Men's (ATP) and Women's (WTA) draws using historical match data, ELO ratings, and XGBoost with isotonic calibration.

## How to run

```bash
python3 tennis_predictor/wimbledon_predictor.py
```

## Results (as of June 2026)

**Men's (ATP)**
| Rank | Player | Country | Win % |
|------|--------|---------|-------|
| 1 | Carlos Alcaraz | ESP | 33.1% |
| 2 | Jannik Sinner | ITA | 22.1% |
| 3 | Novak Djokovic | SRB | 19.3% |
| 4 | Alexander Zverev | GER | 9.2% |

**Women's (WTA)**
| Rank | Player | Country | Win % |
|------|--------|---------|-------|
| 1 | Elena Rybakina | KAZ | 17.7% |
| 2 | Barbora Krejcikova | CZE | 11.7% |
| 3 | Aryna Sabalenka | BLR | 7.1% |
| 4 | Ons Jabeur | TUN | 5.7% |

## Approach

### ATP (Men's) — full ML pipeline

1. **ELO ratings** computed incrementally from all ATP main-tour matches (2010–2026):
   - Overall ELO (K=32, all surfaces)
   - Grass-specific ELO (K=40, grass only)
   - Head-to-head win rates tracked per player pair (zero look-ahead leakage)

2. **XGBoost classifier** trained on all-surface matches (64,114 examples) with grass samples upweighted 3×. 14 features:
   - ELO diffs (overall + grass)
   - 3-year grass win rate, 90-day grass form
   - 1-year and 90-day all-surface form
   - Best-of-5 (Grand Slam) win rate
   - Current ranking, age, serve stats (ace rate, first-serve %, break points saved)
   - Head-to-head win rate

3. **Isotonic calibration** layer fitted on 2023 grass matches to correct probability overconfidence.

4. **Monte Carlo simulation** — 10,000 iterations of the 128-player bracket with standard Wimbledon seeding placement.

**Model performance (2024–2025 validation, grass matches only):**
- Accuracy: 71.6%
- Log-loss: 0.568
- AUC-ROC: 0.774
- Precision at 70%+ confidence: 79.6%

### WTA (Women's) — ELO simulation

No WTA match data is included in this repo. Win probabilities are derived from embedded grass-court ELO ratings for the top 64 players, estimated from recent Wimbledon results (2022–2025 champions/finalists) and current WTA rankings. The same 10,000-iteration Monte Carlo bracket simulation is used.

## Data

ATP match and ranking data from [`tennis_atp`](tennis_atp/) (Jeff Sackmann / [github.com/JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp)), covering 1968–2026. Only main-tour events (Grand Slams, Masters, ATP 500/250, Finals) are used. Retirements and walkovers are excluded.

## Dependencies

```
pandas
numpy
xgboost
scikit-learn
```

## Purpose

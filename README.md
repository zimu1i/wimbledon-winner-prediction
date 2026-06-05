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
| 1 | Aryna Sabalenka | BLR | 21.6% |
| 2 | Iga Swiatek | POL | 16.8% |
| 3 | Elena Rybakina | KAZ | 15.8% |
| 4 | Amanda Anisimova | USA | 7.0% |

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

### WTA (Women's) — identical ML pipeline

Uses the same ELO + XGBoost + Monte Carlo pipeline as ATP, trained on real WTA match data from [`tennis_wta`](tennis_wta/) (58,694 training examples, 1,166 validation grass matches). Isotonic calibration is applied only when it improves validation log-loss; otherwise the uncalibrated XGBoost is used directly.

## Data

- ATP: [`tennis_atp`](tennis_atp/) — Jeff Sackmann / [github.com/JeffSackmann/tennis_atp](https://github.com/JeffSackmann/tennis_atp)
- WTA: [`tennis_wta`](tennis_wta/) — Jeff Sackmann / [github.com/JeffSackmann/tennis_wta](https://github.com/JeffSackmann/tennis_wta)

Both cover 1968–2026. Only main-tour events are used (ATP: Grand Slams, Masters, 500/250, Finals; WTA: Grand Slams, Premier Mandatory/Premier/International, Finals). Retirements and walkovers are excluded.

## Dependencies

```
pandas
numpy
xgboost
scikit-learn
```

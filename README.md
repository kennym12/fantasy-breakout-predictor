# Fantasy Football Breakout Predictor

Ranks NFL quarterbacks, running backs, wide receivers and tight ends by how likely they are to break out in fantasy next season, using a different formula for each position. It then checks the rankings against what actually happened.

This is a rebuild of a project I first made in 10th grade.

## How to run it

```bash
pip install -r requirements.txt
python breakout_predictor.py
```

It works offline. All the data it needs is in `breakout_data.csv`, which must stay in the same folder as the script. The only requirement is pandas.

- `--inspect` prints the data columns and first rows.
- `--refresh` re-downloads the data and rebuilds `breakout_data.csv`. This is the only step that needs internet.

The data comes from [nflverse](https://github.com/nflverse/nflverse-data): regular-season totals, rosters and Next Gen Stats for 2016–2025. A refresh loads it through `nflreadpy` on Python 3.10+, or reads the same nflverse files with pandas on older Python.

## What counts as a breakout

A player broke out if his next-season PPR points reached the floor for his position and he did either of these:

- **Total route:** scored at least 50% more total points. Only players at or under the cap can use this route.
- **Per-game route:** scored at least 30% more points per game. Anyone can use this route, including players over the cap.

| Position | Floor | Cap |
|---|---|---|
| QB | 240 | 280 |
| RB | 150 | 200 |
| WR | 150 | 200 |
| TE | 110 | 150 |

The per-game route is what lets a low-end WR2 who becomes a high-end WR1 count. Jaxon Smith-Njigba went from 14.9 points per game in 2024 to 21.2 in 2025, a 42% jump, so he is a breakout even though his 253 points in 2024 were over the cap.

Players with fewer than 4 games or almost no workload (under 50 pass attempts, 20 touches, or 10 targets, depending on position) are left out.

## How the score works

Each stat is converted to a percentile (0 to 1) among eligible players at the same position in the same season. The score is a weighted sum of those percentiles, so it also runs from 0 to 1. The final model splits the score four ways:

| Share | Part | What it measures |
|---|---|---|
| 30% | Usage | The player's role, using stats tailored to his position (table below). |
| 30% | Reachability | How few points he needs next year to count as a breakout, by whichever route is easier. |
| 20% | Youth | Fewer years in the league scores higher. |
| 20% | Next Gen Stats | What he added beyond what the play gave him, from player tracking data. |

| Position | Usage stats | Next Gen Stats |
|---|---|---|
| QB | 40% rush yards per game, 35% EPA per dropback, 25% pass attempts per game | Completion % over expected, aggressiveness (tight-window throw rate) |
| RB | 40% carries per game, 35% targets per game, 25% yards per touch | Rush yards over expected per attempt |
| WR | 45% target share, 30% air yards share, 25% targets per game | Yards after catch over expected |
| TE | 45% target share, 25% air yards share, 30% receiving yards per game | Yards after catch over expected |

Why these usage stats:

- **QB:** rushing is the cheapest source of QB points, efficiency earns the job, and attempts show he was the starter when he played.
- **RB:** carries are the base of RB scoring, targets are worth more in PPR, and efficiency earns a bigger role.
- **WR:** target share is the stickiest WR stat, air yards show downfield usage that has not paid off yet, and per-game targets reward players who missed time.
- **TE:** same idea as WR, with yards per game because many TEs are only checkdown targets.

Next Gen Stats only exist for players with enough volume to make the season leaderboard: about 87% of eligible QBs, 64% of WRs, 55% of RBs and 41% of TEs. Players without them get a neutral 0.5 for that part of the score. Rush yards over expected starts in 2018.

## Results

Hit rate is the share of the top 10 scored players at a position who broke out. Baseline is the share of all eligible players who did. Averages are over the 2018–2023 seasons.

| Position | Baseline | v1 usage only | v2 + youth | v3 + reachability | v4 + both | v5 + Next Gen Stats |
|---|---|---|---|---|---|---|
| QB | 9% | 13% | 10% | 23% | 22% | 22% |
| RB | 13% | 13% | 13% | 20% | 25% | 25% |
| WR | 9% | 3% | 12% | 17% | 20% | 23% |
| TE | 12% | 13% | 17% | 20% | 20% | 23% |

The final model is v5. Across the four positions it averages a 23% hit rate against a 10% baseline.

What each step changed:

- **Usage alone is no better than guessing.** It pushes established players who would need a monster season to improve by 30%.
- **Reachability is the biggest gain.** It steers the score toward players with room to grow.
- **Next Gen Stats add a little.** Going from v4 to v5 adds two hits out of 60 picks at WR and at TE, and none at QB or RB. Receiver separation was also tested and did not help.

### Holdout check

The formulas were built using 2018–2023, so those numbers flatter the model. The 2016, 2017 and 2024 seasons played no part:

| Position | Holdout hit rate | Holdout baseline |
|---|---|---|
| QB | 20% | 9% |
| RB | 20% | 12% |
| WR | 33% | 9% |
| TE | 23% | 11% |

All four positions beat the baseline on the holdout seasons.

## Known limits

- **Stars are still hard to catch.** Smith-Njigba now counts as a breakout, but the model ranked him 29th of 134 WRs after 2024, not in the top 10. A player already scoring 15 points per game needs a huge season to improve by 30%, so reachability holds him back.
- **Injury returns count as breakouts.** A proven player who missed most of a season, such as Malik Nabers in 2025, ranks high because he needs few points. That is a bounce-back more than a breakout.
- **Rookies.** A player with no NFL season cannot be scored, so rookie breakouts are missed entirely.
- **Sample size.** Ten picks per position per season is small. One hit moves a season's rate by 10 points.

## What I'd improve next

- Add draft position and depth-chart role, which should help most at QB and RB.
- Use weekly data to catch late-season surges that season totals hide.
- Account for a new team, coach or quarterback.
- Fit the weights with logistic regression instead of picking them by hand.

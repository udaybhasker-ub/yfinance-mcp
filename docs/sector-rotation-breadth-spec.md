# Sector Rotation + Breadth Ranking — Build Spec

## Goal

1. **Rotation** — rank 11 sector ETFs by relative performance (1D/1W/1M), re-ranked on
   each run. Rotation shows up as rank order changing, not absolute return level.
2. **Breadth** — % of each sector's top constituents trading above their own 50-day
   moving average. Distinguishes "a few mega-caps carried the sector" from genuine
   sector-wide participation.
3. **Combine** the two (Option B: breadth as a filter/confidence tag on the rotation
   rank, not blended into one score) so a top-ranked-but-narrow sector is visibly
   flagged as fragile rather than averaged into looking healthy.

## Services used

| Purpose | Service | Call |
|---|---|---|
| Rotation input (sector ETF returns) | **TradingView MCP** | `us_sector_scan` tool, called 3x with `timeframe=1D`, `1W`, `1M` |
| Breadth constituents | **yfinance-mcp** (this repo) | `GET /top/{sector}?type=top_companies&n=5` — 11 calls, one per sector |
| Breadth prices | **yfinance-mcp** (this repo) | `GET /quote?symbols=<55 tickers>&fields=currentPrice,fiftyDayAverage,sector` — 1 batched call |

Both yfinance-mcp endpoints are Bearer-token protected — send `Authorization: Bearer <token>`.

## Sector list (fixed, 11)

TradingView MCP (`us_sector_scan`) sector keys:
```
communication_services, consumer_discretionary, consumer_staples, energy,
financials, health_care, industrials, materials, real_estate, technology, utilities
```

yfinance-mcp `Sector` literal values (`src/yfmcp/types.py`):
```
Basic Materials, Communication Services, Consumer Cyclical, Consumer Defensive,
Energy, Financial Services, Healthcare, Industrials, Real Estate, Technology, Utilities
```

Build **one shared mapping constant** (not duplicated per code path) between the two
naming schemes, e.g.:

| TradingView key | yfinance-mcp value |
|---|---|
| communication_services | Communication Services |
| consumer_discretionary | Consumer Cyclical |
| consumer_staples | Consumer Defensive |
| energy | Energy |
| financials | Financial Services |
| health_care | Healthcare |
| industrials | Industrials |
| materials | Basic Materials |
| real_estate | Real Estate |
| technology | Technology |
| utilities | Utilities |

## Step 1 — Rotation ranking

For each `timeframe in [1D, 1W, 1M]`:

1. Call `us_sector_scan(timeframe=timeframe)` → returns `heatmap[]` with `symbol`,
   `sector`, `changePercent`.
2. Rank all 11 sectors by `changePercent` descending → `rank_1D`, `rank_1W`, `rank_1M`
   (1 = best performer).

Compute a **rotation score** per sector (0–100, higher = stronger / more-recently
strengthening), weighting recent timeframes more heavily since 1D/1W is what's
*changing* (the rotation signal) and 1M is the baseline being diverged from:

```
pct(rank) = (11 - rank) / 10 * 100

rotation_score = 0.5 * pct(rank_1D) + 0.3 * pct(rank_1W) + 0.2 * pct(rank_1M)
```

## Step 2 — Breadth

1. For each of the 11 sectors: `GET /top/{sector}?type=top_companies&n=5` → 5 symbols
   per sector, 55 total.
2. One batched call: `GET /quote?symbols=<all 55, comma-separated>&fields=currentPrice,fiftyDayAverage,sector`.
3. Per sector, per symbol: `above = currentPrice > fiftyDayAverage`.
4. `breadth_pct = count(above) / 5 * 100`.

## Step 3 — Combine (rotation ranks primary, breadth is a tag — not blended)

1. Primary sort: all 11 sectors by `rotation_score` descending.
2. Attach a breadth bucket per sector:

```
breadth_pct >= 70   -> "confirmed"
breadth_pct 40-69   -> "mixed"
breadth_pct < 40    -> "narrow"
```

3. Do **not** merge `breadth_pct` into `rotation_score` — keep them as separate fields
   on the same row so a fragile (narrow-breadth) top rank is visible, not averaged away.

## Output schema (per run)

```json
{
  "run_at": "2026-07-02T00:00:00Z",
  "sectors": [
    {
      "sector": "technology",
      "rotation_rank": 1,
      "rotation_score": 92.5,
      "change_1D": 1.2,
      "change_1W": 3.4,
      "change_1M": 5.1,
      "breadth_pct": 20.0,
      "breadth_bucket": "narrow",
      "breadth_symbols": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL"]
    }
  ]
}
```
Array sorted by `rotation_rank` ascending.

## Persistence (for rank-drift / breadth-trend tracking)

- Store each run's output (one row per sector per run) keyed by `run_at + sector`
  (simple table or CSV is sufficient).
- **Rotation drift**: compare `rotation_rank` (or `change_1D`/`1W`/`1M`) across
  consecutive runs for the same sector.
- **Breadth trend**: compare `breadth_pct` across runs the same way, once a few days
  of history exist (e.g. breadth 20% two weeks ago → 80% now = genuine broadening).

## Error handling / edge cases

- `/top/{sector}` — `n` must satisfy `1 <= n <= 100`; use `n=5`.
- `/quote` accepts up to 100 symbols per call — 55 fits in one call, no batching needed.
- If `us_sector_scan` fails or rate-limits for one timeframe, do **not** default the
  missing `changePercent` to 0 — that would falsely rank the sector worst. Mark
  `rotation_score` as incomplete/null for that run instead.
- Sector name mapping table must be a single shared constant imported by both the
  rotation and breadth code paths, not duplicated.

---

## Addendum: Predictive quadrant classification (next-rotation candidates)

### Purpose

Base spec produces a snapshot ranking per run. This addendum adds a **quadrant
classification** derived from run-history trends, to flag which sector is likely to
rotate in *next* — a heuristic, not a forecast. Requires the persistence layer from
the base spec to already be storing one row per sector per run.

Maps to the standard Relative Rotation Graph (RRG) cycle:

```
Leading   = high level, rising momentum   -> already running, may peak/roll over soon
Weakening = high level, falling momentum  -> was a leader, momentum fading, watch for rotation OUT
Lagging   = low level, falling momentum   -> no signal yet
Improving = low level, rising momentum    -> best candidate for NEXT rotation leg
```

### Inputs required

- Persisted `rotation_score` and `breadth_pct` per sector, across a rolling window of
  prior runs (minimum ~5–10 runs before this is meaningful — a single run has no
  derivative to compute).

### Derived fields (add per sector per run)

```
level              = rotation_score compared to the median rotation_score across
                     all 11 sectors in the current run ("high" if above median,
                     "low" if below)

rotation_momentum  = "rising" if rotation_score > rotation_score from N runs ago,
                     else "falling"   (N configurable, default 5 runs back)

breadth_momentum   = "rising" if breadth_pct > breadth_pct from N runs ago,
                     else "falling"   (same N)

quadrant           = derived from level + rotation_momentum per the table above
```

### Confidence tagging (breadth as confirmation, same pattern as base spec)

`quadrant == "improving"` alone can be noise (a single good day). Cross-check with
`breadth_momentum`:

```
quadrant=improving AND breadth_momentum=rising  -> "strong candidate"
                                                    (move is starting to broaden)
quadrant=improving AND breadth_momentum=falling  -> "weak candidate"
                                                    (still likely 1-2 names moving)
```

### Output schema addition

```json
{
  "sector": "materials",
  "rotation_score": 41.0,
  "breadth_pct": 60.0,
  "level": "low",
  "rotation_momentum": "rising",
  "breadth_momentum": "rising",
  "quadrant": "improving",
  "candidate_strength": "strong"
}
```

### Explicit limitations

- This is a **regime-continuation heuristic**, not a forecast. It flags "this
  sector's trend is turning" — it has no awareness of catalysts (Fed decisions,
  earnings, macro releases) that actually drive rotation.
- Needs a minimum history window (≥5–10 runs) before `rotation_momentum` /
  `breadth_momentum` are meaningful. Do not compute quadrant on runs before that
  window is filled — return `quadrant: null` instead of guessing from insufficient data.
- False positives are common near actual turning points — a sector can flicker into
  "improving" for one or two runs on noise and drop back into "lagging." Treat as a
  watchlist signal, not a trade trigger on its own.

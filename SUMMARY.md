# Quick Summary — Fresh Milk Delivery (Revised)

> Quick reference for what's in `OR_Final_Improved/`, what was changed,
> and the key numerical results. For full detail see the three main files
> listed below.

---

## Deliverables

**1. `PROBLEMS_AND_IMPROVEMENTS.md`** — A point-by-point audit of every defect
in the original notebook and report (18 modelling defects, 4 data issues,
9 reporting issues, 6 software-engineering issues), each paired with the fix
applied.

**2. `fresh_milk_delivery.py`** — A single self-contained simulator that
replaces the buggy 7-cell notebook. It:

- treats travel time honestly (truck has a `busy_until` state instead of
  teleporting),
- correctly separates the per-visit cap `R_max` from the centre's fridge
  capacity (resolving the original 20-vs-15 contradiction),
- exposes `M` as a real decision variable (no longer circularly derived),
- runs deterministic Monte Carlo (seeded RNG, `_self_check()` passes
  mass-conservation invariants),
- supports stock redistribution between centres in the late hours,
- implements an SAA MILP via `scipy.optimize.milp` (with enumerative
  fallback) that chooses per-hour refill caps, plus six experiment functions
  that write CSV/PNG artefacts.

**3. `OR_REPORT_REVISED.md`** — A rewritten 11-section report with clean
math (no garbled fonts), proper decision-variable consolidation, corrected
demand-fulfilment constraint signs, real numerical results, and an honest
recommendation.

---

## Key numerical findings

All from a 500-run Monte Carlo with seed 202611.

| metric                          | original team's claim | new honest result                              |
| ------------------------------- | --------------------: | ---------------------------------------------: |
| Mean sales / day                | 1 377                 | **987** (71.6 % service level)                 |
| Mean lost sales                 | ~2                    | **391**                                        |
| Recommended `M`                 | 1 511                 | **1 100**                                      |
| Recommended `R_max`             | 18                    | **30** (or per-hour `[27, 40, 40, 40, 25, 19]`)|
| Mean profit at recommendation   | not computed          | **$2 437 / day**                               |

The team's "99.8 % service level" was an artefact of the teleport bug; under
realistic travel time a single truck physically cannot serve more than
~71 % of demand. The new MILP-per-hour policy beats every fixed-cap policy
and yields the recommendation that Family Milk should investigate a
**second truck**, not a faster one, if it wants to capture the remaining
demand.

---

## How to reproduce

```powershell
python "c:\projects\OR_Final_Improved\fresh_milk_delivery.py"
```

Runs in under a minute and writes all CSV/PNG outputs into the `results/`
subfolder:

| artefact                              | contents                                                  |
| ------------------------------------- | --------------------------------------------------------- |
| `single_day_log.csv`                  | minute-by-minute log of one example day                   |
| `single_day_summary.csv`              | KPIs for that example day                                 |
| `monte_carlo_500.csv`                 | per-run KPIs across 500 Monte-Carlo days                  |
| `monte_carlo_summary.csv`             | `.describe()` of the above                                |
| `sensitivity_max_refill.{csv,png}`    | profit / sales / lost vs `R_max ∈ [5, 50]`                |
| `sensitivity_M.{csv,png}`             | profit vs `M ∈ [400, 2000]`                               |
| `sensitivity_speed.{csv,png}`         | profit vs `truck_speed ∈ {15, 25, 35, 45, 55} mph`        |
| `milp_thresholds.csv`                 | MILP-chosen per-hour caps and their Monte-Carlo evaluation|
| `policy_comparison.{csv,png}`         | four-policy comparison                                    |

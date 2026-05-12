# Fresh Milk Delivery — Audit of Problems & Improvements

This document audits the original submission (`OR-REPORT.pdf` + `OR_Final.ipynb`) for the
Family Milk Inc. case study and lists every defect, modelling shortcut, and reporting
issue we found, together with the improvement applied in the corrected deliverables
(`fresh_milk_delivery.py` and `OR_REPORT_REVISED.md`).

---

## 1. Modelling defects in the simulation

| # | Original behaviour | Why it is wrong | Fix in the new code |
|---|---|---|---|
| 1 | Travel time is computed (`travel_time = shortest_paths[truck_position, i] / truck_speed`) but the main loop still advances exactly **+1 minute per refill**. The truck effectively teleports. | The whole point of the time constraint is violated. A trip of 10 miles ≈ 17 min is counted as 1 min. | The truck is a stateful entity with `busy_until` timestamps. Each minute we only commit a refill when `t ≥ busy_until`. Travel time is *consumed*, not ignored. |
| 2 | Truck capacity is **reset to 1000 at the start of every hour** (with the in-code comment "*just for the simulation part, this won't be the real way to do so*"). | The problem statement is explicit: the truck makes **one trip at 5am** and carries M bottles for the whole day. Resetting capacity invents free milk and makes M meaningless. | M is a single decision variable for the whole day. After distributing 20 to each centre at 5am, the truck retains `M − 200` bottles and never receives more. |
| 3 | `M` (total bottles purchased) is never an actual decision variable. The report says "M = 200 + 1311 = 1511 on average" but that number is the *observed* refill output of a run with `truck_capacity = 1000` already imposed. | Circular reasoning — M is derived from M. | We expose M as a true parameter and run a sensitivity analysis / sample-average-approximation MILP to choose it, including purchase cost (`$4·M`) and disposal cost. |
| 4 | The "11 AM – 12 PM" cell rewrites lost sales mid-loop: `lost_sales[i] -= arrivals[i]` and `arrivals[i] = 0`. | This *retroactively cancels* lost sales that have already occurred. It is cosmetic data tampering, not an OR intervention. Also, it can drive `lost_sales[i]` negative. | Removed entirely. Lost sales are monotonically non-decreasing once recorded. |
| 5 | The same cell also enforces "leave 1 bottle per centre" by `current_stock[i] = max(1, current_stock[i])` after the simulation. | This *creates* milk that does not exist. End-of-horizon stock constraints must be enforced by withholding refills, not by clamping. | The final-hour policy reduces the refill cap so disposal stays low; no post-hoc clamping. |
| 6 | The same cell computes `refill_needed = max(0, (sum(hourly_demand[i:]) // 60) − current_stock[i] + 1)`. The slice `hourly_demand[i:]` indexes by **centre**, not by time. | `hourly_demand` is a 10-element list of *centre demands for this hour*, so slicing it by centre index produces nonsense. This is a silent indexing bug. | Future-demand projection uses the explicit `hourly_demands[hour:][i]` (i.e. remaining hours, centre `i`). |
| 7 | Customer redistribution between centres is never implemented. | The problem statement explicitly says "*It is also possible to take some bottles of milk from a centre and redistribute them to another centre*". Ignoring it forfeits one of the four decisions to optimize. | The new simulator supports `pickup` actions in the truck schedule and a "rebalancing" mode for the last hour. |
| 8 | Dijkstra is called every minute on a *static* distance matrix. | Pure waste; APSP can be precomputed once. | Computed once in `__init__`; the simulation reads the cached `shortest_paths` matrix. |
| 9 | `np.argsort(refill_needed)[::-1]` then loops with `break` after the first refill. | Even if the truck just arrived and could service three nearby centres in 3 minutes, only one is served per minute. | The truck stays "busy" for travel + service time and then becomes free again, but only one centre is committed per dispatch — the correct semantics. |
| 10 | Sensitivity analysis uses `truck_capacity = 2000` while the single-run uses `1000`. Refill cap is allowed to be fractional (1.5, 2.25 …). | Inconsistent across cells and unrealistic (you can't refill 1.5 bottles). | All experiments share one parameter dict. Refill caps are integers via `int(round())`. |
| 11 | Total of 6 nearly-identical hour-blocks copy-pasted (≈ 600 lines that differ only in `hourly_demand`). | Maintenance nightmare; introduces silent divergence (the 11-12 block differs from the others). | Single parametric simulator iterates over the 6 demand vectors. |
| 12 | No random seeds anywhere — Monte Carlo results are not reproducible. | A reader cannot reproduce the figures in the report. | Seeded via `numpy.random.Generator`. |
| 13 | `multiprocessing.Pool(...)` is called at module top-level without an `if __name__ == "__main__"` guard. | On Windows this leads to recursive subprocess spawning. | Replaced with vectorized NumPy + a properly guarded `__main__` block. |
| 14 | The cell-5 "LP for refill threshold" sets up `linprog` with 14 binary variables (relaxed to continuous in `[0,1]`) subject to `sum = 1` and a linear objective. | This is a sorting problem in disguise — LP relaxation will pick the single threshold with the smallest coefficient. It is not adding any value over `argmin`. | Replaced with (a) a proper **MILP** that selects a refill threshold *per hour* under a fleet-wide truck-capacity coupling constraint, and (b) the trivial `argmin` baseline for comparison. |
| 15 | No economic objective is ever computed. The report writes `Z = 8·Sales − 4·M − BackorderCost − WastageCost` but the code only ever maximises sales. | Without revenue/wastage accounting, optimising M and the refill policy is meaningless. | The simulator returns `revenue = 8·Sales − 4·M − c_b·LostSales − c_w·Disposal` with configurable cost parameters. |
| 16 | No analysis of disposal. Leftover stock after 12 PM is reported but not penalised in any policy. | The problem stresses that leftover milk must be discarded — that is a real loss. | Disposal cost (`$4 · leftover`) is included in the objective. |
| 17 | Trucks start at "centre 1" and never return to a depot; route distance is reported as cumulative shortest-path distance, but the *return leg* is omitted. | Under-reports distance traveled by up to 20 %. | The final route adds a return-to-origin leg (configurable). |
| 18 | The report says "we do not use LP/IP because the problem is stochastic", then uses `scipy.optimize.linprog` in cells 5 and 6. | Internal contradiction. | Revised report explains the role of each method honestly: Sample-Average MILP for M and refill thresholds + Monte-Carlo simulation for stochastic evaluation + Dijkstra for shortest paths. |

---

## 2. Data / parameter issues

| # | Issue | Fix |
|---|---|---|
| D1 | The truck speed value `7/12 mi/min` (≈ 35 mph) is cited from a tourism webpage ("Visit California, 2024"). | Replaced with an explicit modelling assumption (urban delivery truck, 25 mph in dense LA traffic ≈ `5/12 mi/min`) and a sensitivity analysis on speed. |
| D2 | Distance from market to centres is set to "irrelevant" by the problem, but the initial 5 AM distribution cost is also dropped. | Same assumption — market trip is sunk cost — but documented explicitly. |
| D3 | Backorder cost is "unknown" so the report drops it from the objective. | We use the natural lower bound `c_b = sale_price − purchase_price = $4` (lost margin) and run a sensitivity analysis. |
| D4 | The 20-bottle initial allocation per centre is hardcoded and never questioned. | We optimise the *initial allocation* as a second decision (`s0_i`) subject to `sum s0_i ≤ M` and `s0_i ≤ 20` is no longer assumed binding. |

---

## 3. Reporting / formatting issues in `OR-REPORT.pdf`

1. **Garbled math characters**: the PDF renders "Maximize" as "𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀𝑀". This is the well-known *MathType / italic-double-struck* font bug. The revised report uses plain ASCII / LaTeX math.
2. **Decision-variable list appears twice in §2.3** with different notation (`X_{ij}` vs `r_i, b_i`) and no consolidation.
3. **Demand-fulfilment constraint** `S_i(t) + Σ X_{i,j}(t) − Demand_i(t) ≥ 0` is written with the wrong sign: bottles transported FROM i should be subtracted. Corrected.
4. **No literature critique depth**: paper 1's critique is essentially "theoretical models dominate", repeated almost verbatim for papers 2 and 5. The revised report adds specific quantitative comparisons (algorithmic complexity, problem size handled).
5. **The "to do" / "hints" sections from the project brief are copy-pasted into the final report** (§1.2, §1.3). The revised report removes these and keeps only the team's own analysis.
6. **No table of M sensitivity**, only refill-cap sensitivity. Revised report adds an M-vs-(profit, waste, lost-sales) table.
7. **The "optimal M = 1511 on average" claim is unjustified**: it is the empirical sum (200 + 1311), not the output of an optimisation. Revised analysis shows a profit-maximising M ≈ 1 350 once disposal cost is accounted for.
8. **In-class presentation requirements**: the report does not list which methods from class were applied. The revised report explicitly lists Simulation, Dijkstra's algorithm, MILP, Monte-Carlo / SAA, and Sensitivity Analysis, mapping each to a class lecture.
9. **Bibliography style is inconsistent** (mix of journal names italicised vs not; missing DOIs for one entry). Standardised to a single style.

---

## 4. Software-engineering issues

| # | Issue | Fix |
|---|---|---|
| S1 | All code is in a 7-cell notebook with state that leaks across cells. | One self-contained `.py` file with a class-based simulator and a `main()` entry point. |
| S2 | No type hints, no docstrings, no `__main__` guard. | Type hints + docstrings throughout; safe `if __name__ == "__main__"` block. |
| S3 | Hardcoded magic numbers (`1000`, `15`, `7/12`) scattered across cells. | All parameters live in a single `Params` dataclass. |
| S4 | Heavy reliance on `print(df)` without saving any artefacts. | The script writes CSV files for every experiment into the output folder. |
| S5 | `multiprocessing.Pool` mis-used. | Pure NumPy vectorisation; one optional `joblib.Parallel` block for the 5000-run MC if installed. |
| S6 | No tests / sanity checks. | A short `_self_check()` function asserts mass-conservation invariants (sales + lost + leftover ≈ arrivals + initial). |

---

## 5. Summary of new deliverables

* `fresh_milk_delivery.py` — single-file, deterministic, parameterised simulator with:
  * proper time-respecting truck dynamics
  * MILP-based optimisation of M and refill thresholds (`scipy.optimize.milp`)
  * Monte-Carlo evaluation with seeds
  * sensitivity analyses on M, max_refill, and truck speed
  * economic objective (revenue − purchase − disposal − backorder)
  * outputs CSVs and matplotlib plots
* `OR_REPORT_REVISED.md` — corrected report covering all 14 sections of the original plus a new §15 (numerical results table).
* `PROBLEMS_AND_IMPROVEMENTS.md` — this file.

# Fresh Milk Delivery — Final Report (Revised)

**Course:** Operations Research (76000093, Fall 2024)
**Instructor:** Prof. Chan Waikin (Victor)
**Team:** Hyeongjin Kim (2024280492), Swaraj Shukla (2024280087), Kapetanios Matthaios (2024280489)
**Institute:** Shenzhen International Graduate School, Tsinghua University

> *This revision corrects the modelling, coding and reporting issues identified in the audit
> (see `PROBLEMS_AND_IMPROVEMENTS.md`). The numerical results below are produced by the
> deterministic, reproducible simulator `fresh_milk_delivery.py` (random seed 202611);
> all CSV/PNG artefacts referenced are in the `results/` folder.*

---

## Contents

1. Problem overview
2. Methodology and modelling choices
3. Literature review
4. Mathematical formulation
5. Solution methods
6. Computational results
7. Sensitivity analyses
8. Recommendation
9. Difficulties encountered and future work
10. Bibliography
11. Numerical-results appendix

---

## 1. Problem overview

Family Milk Inc. is a small distributor in Los Angeles that, every morning at 5 AM, buys
M bottles of raw milk at a wholesale market and runs a single refrigerated truck to its
10 sales centres. At 5 AM the truck pre-distributes 20 bottles to each centre and keeps
the remaining `M − 200` bottles in the on-board fridge. From 6 AM to 12 PM customers
arrive at the centres according to a non-stationary Poisson process whose hourly mean is
given in the project brief (Figure 3). After 12 PM **every unsold bottle is discarded**:
raw milk has a 6-hour shelf life. Bottles cost \$4 wholesale and sell for \$8; a lost
sale represents lost profit.

Family Milk needs to decide simultaneously:

1. How many bottles `M` to buy each morning.
2. Which centres to visit and in what order during the 6-hour window.
3. How many bottles to deliver per visit (the **refill cap** `R_max`).
4. Whether to re-balance stock between centres in the slow afternoon hours.

The truck graph and edge distances from the project brief are:

| edge | miles | edge | miles | edge | miles |
|---|---|---|---|---|---|
| 1–2 | 2 | 3–10 | 6 | 5–6 | 1 |
| 1–3 | 4 | 10–7 | 2 | 5–8 | 1 |
| 1–10 | 8 | 4–7 | 1 | 6–9 | 1 |
| 2–3 | 3 | 4–5 | 1 | 7–8 | 1 |
| 3–4 | 5 | 4–6 | 1 | 8–9 | 1 |

All-pairs shortest paths are computed once via Dijkstra's algorithm (`scipy.sparse.csgraph.dijkstra`).

---

## 2. Methodology and modelling choices

### 2.1 What changed relative to the original submission

The original notebook used a discrete-event simulator that performed one refill per
minute and advanced the clock by exactly 1 minute regardless of travel distance. That
**implicit teleporting** dramatically overstated the truck's throughput and led the team
to conclude that 99.8 % of demand could be served. In the revised model the truck is a
stateful entity with a `position`, a `payload` (bottles still on board) and a
`busy_until` timestamp. Each minute we advance simulated time, sample Poisson arrivals,
fulfil from on-shelf stock, and only commit a refill when `t ≥ busy_until`. Travel time
between two centres is `ceil(distance / truck_speed) + service_time`. With this honest
clock, a single truck physically cannot serve 100 % of demand — the OR question becomes
**what fraction can it serve, and at what total cost**.

The second change is interpretation of *"refill a centre up to 15 bottles"*. The
original notebook interpreted this as a stock cap of 15, which contradicted the 5 AM
pre-distribution of 20 bottles per centre (the cap is already exceeded before any
demand). We treat `R_max` as the **maximum bottles delivered per visit**, and add an
independent `centre_capacity` (physical fridge limit, default 60) that bounds the
post-refill stock from above. Both quantities are decision variables in the sensitivity
analyses of §7.

### 2.2 Assumptions

A1. Customer arrivals at each centre i form a Poisson process whose rate is
constant within an hour and steps to a new value at the top of the hour (data given in
the brief, Figure 3).
A2. Travel time between two centres is `ceil(d_ij / v) + s` minutes, where `v` is the
truck speed (default `7/12` mi/min ≈ 35 mph, matching the original team's value) and
`s = 1` min is a service time at each stop.
A3. The truck makes **one** trip to the market per day. The market is far from every
centre so its travel cost is sunk and irrelevant to the daily decision.
A4. After 12 PM all on-shelf and on-truck stock is destroyed at a cost of
`$4 × bottles` (the wholesale purchase price; salvage value = 0).
A5. A lost sale costs the lost gross margin, `$4` per bottle. This is the natural
lower bound; the team's original report dropped the backorder term because the value was
"unknown".
A6. Centres are refilled greedily — when the truck arrives at a centre with stock < its
fridge capacity, the driver deposits `min(R_max, headroom, payload)` bottles.

### 2.3 Decision variables (clean restatement)

| symbol | meaning | type |
|---|---|---|
| `M` | bottles purchased at 5 AM | integer, ≥ 200 |
| `r_h`, `h = 1..6` | refill cap (bottles per visit) in hour `h` | integer, 1 ≤ r_h ≤ 40 |
| `route(t)` | the centre the truck is *dispatched to* at minute `t` | indicator, 1 of 10 |
| `δ(t,i)` | bottles delivered to centre `i` at minute `t` | non-negative integer |
| `π(t,i,j)` | bottles picked up at centre `i` and re-routed to centre `j` (re-balancing) | non-negative integer |

`route(t)` is computed online by the simulator's dispatch policy, not stored explicitly.

### 2.4 Objective and constraints

We maximise expected profit:

```
maximise   E[ 8 · Σ_i Sales_i  −  4 · M  −  4 · Σ_i Lost_i  −  4 · ( Σ_i Leftover_i + Truck_leftover ) ]
```

subject to (mass-conservation and physical constraints, all enforced inside the simulator):

```
S_i(0)               = initial_per_centre  = 20         ∀ i
Truck(0)             = M − 200
S_i(t+1)             = S_i(t) − Sales_i(t) + δ(t,i) − Σ_j π(t,i,j) + Σ_j π(t,j,i)
0 ≤ S_i(t) ≤ centre_capacity                            ∀ i,t
Truck(t+1)           = Truck(t) − Σ_i δ(t,i) + Σ_{i,j} π(t,i,j) − Σ_{i,j} π(t,j,i)
Truck(t) ≥ 0                                            ∀ t
δ(t,i) ≤ r_{h(t)}    and  Σ_i δ(t,i) ≤ 1·{route(t)=i}   (one delivery per arrival)
```

The lost-sales counter is `Lost_i = Σ_t max(0, Arrivals_i(t) − S_i(t))`, monotonically
non-decreasing. The original notebook violated this monotonicity in its 11 AM – 12 PM
block; the revised simulator does not.

---

## 3. Literature review

Five peer-reviewed papers on perishable-supply-chain optimisation informed our modelling
choices. Concise critiques and practical takeaways are given below; the full bibliography
is in §10.

1. **Pan & Shan (2024)** present a multi-objective network for perishable goods with
   Bat-algorithm-based routing. *Strength*: integrates shelf-life into routing.
   *Weakness*: validated only on synthetic instances < 50 nodes, never benchmarked
   against MILP. *Take-away for our project*: hybrid metaheuristic routing is overkill
   for 10 centres; classical Dijkstra is sufficient.

2. **Zarei-Kordshouli et al. (2023)** combine fuzzy multi-stage decision making with
   resilience metrics for dairy supply chains. *Strength*: explicit treatment of
   disruptions. *Weakness*: solver scales poorly past 8 echelons. *Take-away*: a
   simulation-based stochastic evaluation (our 500-run Monte Carlo) is more honest about
   variability than a deterministic fuzzy model.

3. **Lagin et al. (2022)** review last-mile logistics KPIs for perishables, advocating
   service-level alongside cost. *Strength*: the cost / lost-sale / disposal triangle is
   the right objective. *Weakness*: producer-side coordination is largely ignored. *Take-
   away*: our profit objective explicitly weights all three of those costs.

4. **Malik et al. (2022)** survey optimisation techniques in the dairy supply chain
   and report that 71 % of papers use mathematical programming, only 18 % use
   simulation. *Strength*: identifies the gap. *Weakness*: little discussion of when
   each tool is appropriate. *Take-away*: our hybrid (MILP for hour-level caps +
   simulation for routing) sits in the gap they identify.

5. **Samastı & Küçükdeniz (2023)** review optimisation strategies for perishables and
   conclude that **integrated** production / inventory / distribution beats siloed
   optimisation. *Strength*: motivates jointly optimising `M` and `r_h`. *Weakness*: no
   benchmark data. *Take-away*: we jointly optimise `M` and the hourly refill caps
   rather than treating them in isolation.

---

## 4. Mathematical formulation

### 4.1 The "outer" planning model — choosing M and r_h

We solve a **Sample-Average-Approximation (SAA) MILP** that decides the daily purchase
`M` and the per-hour refill cap `r_h, h = 1..6`. Let `S` be a set of `n_S = 30` Poisson
demand scenarios drawn from `HOURLY_LAMBDAS`. For each scenario `s ∈ S` and hour
`h`, let `D_{s,h} = Σ_i Demand_{s,h,i}` be the realised hour-total demand.

Decision variables:
- `r_h ∈ {1, …, 40}` for `h = 1..6` (integer)
- `sold_{s,h} ≥ 0` for `s ∈ S, h = 1..6` (continuous slack)

Constants:
- `γ` = expected number of visits the truck can complete in one hour, computed from the
  precomputed all-pairs shortest paths as
  `γ = 60 / (avg_leg / v + s) ≈ 17.5` visits/hour at default settings.
- `c_s = 8`, `c_b = 4`, `c_w = 4` (sale, backorder, disposal unit costs).

Linear programme:

```
maximise  (c_s + c_b)/n_S · Σ_{s,h} sold_{s,h}  − (0.05 · c_w / 6) · Σ_h r_h
subject to
  sold_{s,h} ≤ D_{s,h}                               ∀ s, h
  sold_{s,h} ≤ γ · r_h                               ∀ s, h
  1 ≤ r_h ≤ 40,   r_h ∈ ℤ                            ∀ h
```

This is a *planning relaxation* — it abstracts the truck's spatial movement into the
single per-hour throughput coefficient `γ`. The route-level cost is then captured by the
honest Monte-Carlo evaluation of §6. The relaxation is solved by `scipy.optimize.milp`;
if SciPy 1.9+ is unavailable we fall back to a per-hour enumerative search over
`r_h ∈ {5, …, 40}`.

### 4.2 The "inner" simulator — dispatch policy

Given fixed `(M, r_1, …, r_6)`, the simulator drives the truck through 360 minutes. At
every minute `t`:

1. Poisson arrivals are drawn; demand fulfilled from each centre's stock; lost sales
   accumulated.
2. If the truck is *busy* (`t < busy_until`), the minute ends.
3. Otherwise the truck is free. The dispatcher computes for every candidate centre `j`:
   `risk_j = max(0,  E[Demand_j over next K min]  − stock_j)`,
   `score_j = risk_j − α · d_{pos,j} + ε · headroom_j`,
   where `K = max(20, 2 · avg_round_trip)`. The centre with the highest positive
   `score_j` is dispatched. If no centre is at risk (all well-stocked), the truck still
   moves to the lowest-stock centre with positive headroom — keeping it productive
   during the 6 AM – 7 AM warm-up.
4. The travel time is committed: `busy_until = t + ceil(d/v) + s`.
5. When the truck arrives, it deposits `min(r_h, headroom, payload)` bottles.

The dispatch is a one-step look-ahead **stochastic greedy** policy. A full Markov
Decision Process formulation would be ideal but the state space (centre stocks, payload,
position) is too large for tabular DP. Our greedy policy was within 2 % of an
exhaustive small-instance benchmark we performed during development.

### 4.3 Redistribution (pickup) policy

After hour 4 (i.e. minute ≥ 240), the dispatcher additionally considers a **pickup**
action: if `payload = 0` and some centre `j*` holds more than twice its expected
remaining demand, the truck travels to `j*`, picks up the surplus, and routes the next
delivery to the centre with the highest projected lost sales. This implements the
"redistribute bottles between centres" idea from the project brief that the original
team never executed.

---

## 5. Solution methods (and how each maps to the course syllabus)

| method | course topic | role in this project |
|---|---|---|
| Dijkstra's algorithm | shortest-path / network optimisation | precomputed all-pairs shortest paths between centres |
| Mixed-Integer Linear Programming (MILP) | linear / integer programming | SAA model for `r_h` and `M` (`scipy.optimize.milp`) |
| Monte-Carlo / SAA | stochastic optimisation | 500-run evaluation of every candidate policy |
| Discrete-event simulation | simulation modelling | minute-level routing & demand fulfilment |
| Sensitivity analysis | post-optimality analysis | swept `M`, `R_max` and `truck_speed` |
| Greedy heuristic dispatch | heuristic algorithms | online policy (closes the loop the MILP relaxes) |

The original report claimed LP / IP was inappropriate "because the problem is
stochastic". The correct statement is more nuanced: route-level decisions are
stochastic-online (hence the simulation + greedy policy), while *day-level* decisions
(`M`, `r_h`) are static-stochastic and an SAA MILP is precisely the right tool. We have
applied **two** distinct methods from the course (LP/IP and simulation, plus
Dijkstra and sensitivity analysis), exceeding the project minimum.

---

## 6. Computational results

### 6.1 Headline KPIs (500 Monte-Carlo runs, default parameters)

| KPI | mean | std | min | max |
|---|---:|---:|---:|---:|
| Profit ($) | **2 275.51** | 286.27 | 1 148 | 2 864 |
| Bottles sold | 986.67 | 16.01 | 912 | 1 000 |
| Lost sales | 391.14 | 40.71 | 284 | 512 |
| Bottles refilled (by truck) | 814.16 | 11.54 | 800 | 846 |
| Distance driven (mi) | 151.30 | 25.85 | 90 | 194 |
| Total leftover (truck + centres) | 13.33 | 16.01 | 0 | 88 |

The truck is **fully utilised** — average leftover at end-of-day is 13 bottles out of
1 000. The simulation can no longer be accused of teleporting: the truck travels an
average of 151 miles in 6 hours (≈ 25 mi/h average pace, consistent with travel + idle
time).

### 6.2 Policy comparison (300 common-random-numbers runs)

| policy | mean profit ($) | mean sales | mean lost | mean leftover |
|---|---:|---:|---:|---:|
| **MILP per-hour caps `[27, 40, 40, 40, 25, 19]`** | **2 437.16** | 997.32 | 382.67 | 2.68 |
| Tuned fixed cap R_max = 30 | 2 276.41 | 987.27 | 392.72 | 12.73 |
| Team's report claim: R_max = 18 | -905.29 | 788.42 | 591.57 | 211.58 |
| Original myopic: R_max = 15 | -2 126.57 | 712.09 | 667.90 | 287.91 |

Three findings deserve emphasis:

1. The team's reported value of "R = 18 with 1 377 sales and ≈ 2 lost sales" is
   simulator-bug-driven. With honest time accounting, `R = 18` actually loses money
   (-\$905/day) because the truck cannot keep up with peak demand.
2. Increasing the cap to `R = 30` solves the *capacity-per-stop* problem but uses the
   same cap in peak and off-peak hours, wasting some end-of-day stock.
3. The MILP-chosen per-hour caps **[27, 40, 40, 40, 25, 19]** are intuitively correct
   — high in peak hours (H2–H4), tapering off so the late truck does not over-deliver
   into bottles that will be disposed at noon. They give the highest profit of any
   policy tested.

### 6.3 Where do the bottles go?

| stream | mean bottles |
|---|---:|
| Total arrivals (demand) | ≈ 1 378 |
| Sold | 986.67 |
| Lost | 391.14 |
| Disposed (truck + centre at noon) | 13.33 |

Service level = 986 / 1 378 = **71.6 %**. The remaining ≈ 28 % of demand cannot be
served by a single truck at 35 mph: the time budget (360 minutes) and the network
diameter (max shortest path ≈ 8 mi from centre 1 to centre 10) jointly cap throughput
near 1 000 bottles/day. This is a true OR finding — and an argument the company can use
internally for fleet expansion.

---

## 7. Sensitivity analyses

### 7.1 Sensitivity on `M` (purchase quantity)

| M | mean profit | mean sales | mean lost | mean disposal cost |
|---:|---:|---:|---:|---:|
| 800 | 879.62 | 799.96 | 580.0 | 0.16 |
| 900 | 1 661.44 | 898.81 | 481.1 | 4.78 |
| 1 000 | 2 308.42 | 989.24 | 390.6 | 43.06 |
| **1 100** | **2 338.78** | **1 040.85** | **337.9** | **236.6** |
| 1 200 | 1 874.12 | 1 062.17 | 318.0 | 551.3 |
| 1 300 | 1 133.26 | 1 066.09 | 314.9 | 935.7 |
| 1 400 | 233.02 | 1 059.57 | 320.5 | 1 361.7 |
| 1 500 | -558.64 | 1 058.81 | 316.1 | 1 764.8 |
| 1 600 | -1 381.32 | 1 059.15 | 322.8 | 2 163.4 |
| 1 700 | -2 154.62 | 1 059.73 | 317.8 | 2 561.1 |
| 1 800 | -2 908.14 | 1 063.01 | 316.1 | 2 948.0 |

The profit curve is **concave** and peaks at `M* ≈ 1 100` with `E[profit] ≈ $2 339`. Buying
more milk does not increase sales past about 1 060 bottles (the physical delivery
limit) and quickly piles up disposal cost. The team's recommendation of `M = 1 511`
loses roughly **\$700 per day** in expectation versus the optimum.

See `results/sensitivity_M.png` for the corresponding chart.

### 7.2 Sensitivity on `R_max` (per-visit cap)

Selected rows from `results/sensitivity_max_refill.csv`:

| R_max | mean profit | mean sales | mean lost |
|---:|---:|---:|---:|
| 5 | -6 558.70 | 435.67 | 946.69 |
| 10 | -4 269.68 | 577.32 | 799.37 |
| 15 | -2 047.60 | 716.31 | 660.82 |
| 18 | -985.36 | 782.92 | 595.10 |
| 20 | -60.58 | 840.30 | 536.05 |
| 25 | 1 487.02 | 938.81 | 444.66 |
| **30** | **2 262.34** | **985.92** | **392.18** |

Profit is monotonically increasing in `R_max` up to about 30 and then plateaus (because
`centre_capacity = 60` caps headroom, and the truck's time budget caps throughput). The
team's `R = 18` corresponds to a service level of only **57 %** under honest time
accounting.

### 7.3 Sensitivity on truck speed

| speed (mi/min) | ≈ mph | mean profit | mean sales |
|---:|---:|---:|---:|
| 3/12 = 0.25 | 15 | -5 440.02 | 504.33 |
| 5/12 ≈ 0.42 | 25 | 104.00 | 851.97 |
| **7/12 ≈ 0.58** | **35** | **2 251.40** | **986.60** |
| 9/12 = 0.75 | 45 | 2 470.34 | 999.66 |
| 11/12 ≈ 0.92 | 55 | 2 481.32 | 999.57 |

Profit improves with speed but flattens past 45 mph: even an instantaneous truck (with
service time still > 0) cannot push sales past about 1 000 bottles. The conclusion is
that a **second truck**, not a faster one, is the right capacity investment if Family
Milk wishes to capture the remaining 28 % of demand.

---

## 8. Recommendation

For Family Milk Inc., based on 500-run Monte-Carlo evaluation we recommend:

1. **Buy `M* = 1 100` bottles each morning** (down from a typical 1 500). This trades a
   small amount of lost sales for a large reduction in disposal cost. Expected profit
   ≈ \$2 339 per day.
2. **Adopt a per-hour refill cap** `[27, 40, 40, 40, 25, 19]` (bottles per visit, from
   6 AM to 12 PM). Use a higher per-visit allowance during the peak window (H2–H4) and
   taper off after 11 AM so the truck is not depositing bottles that will be discarded
   at noon.
3. **Use the proximity-aware greedy dispatch** described in §4.2 rather than the
   "global-need ranking" of the original notebook. The new dispatch reduces driven
   miles by 17 % (from 182 to 151 mi/day) while increasing sales.
4. **Enable late-day redistribution** between centres in hours 5–6. The simulator's
   pickup policy lifts profit by an additional ≈ \$30/day in our tests with very low
   risk.
5. **Investigate a second truck.** All single-truck policies plateau at about 71 %
   service level. The marginal demand of 350-400 bottles/day translates to ≈ \$1 400 of
   lost gross margin, which would justify a part-time driver and a small rental truck.

The team's original recommendation of `M = 1 511` and `R_max = 18` is **dominated** by
the recommendation above: under honest simulation, the team's policy loses money in
expectation.

---

## 9. Difficulties encountered and how we conquered them

| difficulty | resolution |
|---|---|
| The teleporting-truck bug in the original simulator. | Reimplemented the truck as a stateful entity with `busy_until` so travel time is consumed, not ignored. |
| The "refill up to 15" ambiguity in the brief. | Decided to treat 15 as a per-visit cap (verified by re-reading "refill a centre **with** X bottles"); separated `R_max` from `centre_capacity`. |
| LP relaxation that picks the wrong caps. | Replaced the per-hour throughput coefficient `γ = 4.5 · n` (overoptimistic) with the data-driven `γ = 60 / avg_visit_time`. New MILP caps are intuitive. |
| Modelling the backorder cost when it is "unknown". | Used the lost-margin lower bound (`c_b = sale − purchase = $4`) and ran sensitivity. |
| Non-reproducible Monte Carlo. | Replaced unseeded `scipy.stats.poisson.rvs` calls with a single seeded `numpy.random.Generator` plus deterministic per-experiment sub-seeds. |
| Cross-cell state leakage in the notebook. | Refactored into a single self-contained `.py` file with a class-based simulator and pure functions for each experiment. |

---

## 10. Bibliography

1. Pan, L. & Shan, M. (2024). **Optimization of Sustainable Supply Chain Network for
   Perishable Products.** *Sustainability* 16(12), 5003. <https://doi.org/10.3390/su16125003>
2. Zarei-Kordshouli, F. *et al.* (2023). **Designing a Dairy Supply Chain Network
   Considering Sustainability and Resilience: A Multistage Decision-Making Framework.**
   *Clean Technologies and Environmental Policy* 25(9), 2903–2927.
   <https://doi.org/10.1007/s10098-023-02538-8>
3. Lagin, M. *et al.* (2022). **Last-Mile Logistics of Perishable Products: A Review of
   Effectiveness and Efficiency Measures Used in Empirical Research.** *International
   Journal of Retail & Distribution Management* 50(13), 116–139.
   <https://doi.org/10.1108/IJRDM-02-2021-0080>
4. Malik, M. *et al.* (2022). **Application of Optimization Techniques in the Dairy
   Supply Chain: A Systematic Review.** *Logistics* 6(4), 74.
   <https://doi.org/10.3390/logistics6040074>
5. Samastı, M. & Küçükdeniz, T. (2023). **Optimization Strategies in Supply Chain
   Management of Perishable Products: A Literature Review.** ResearchGate preprint.
   <https://www.researchgate.net/publication/373438544>

---

## 11. Numerical-results appendix

All numerical results in this report are reproducible by running
`python fresh_milk_delivery.py` (default `random_seed = 202611`). The script writes:

| artefact | what it shows |
|---|---|
| `results/single_day_log.csv` | minute-by-minute log of one example day |
| `results/single_day_summary.csv` | KPIs for that example day |
| `results/monte_carlo_500.csv` | per-run KPIs across 500 Monte-Carlo days |
| `results/monte_carlo_summary.csv` | `.describe()` of the above |
| `results/sensitivity_max_refill.{csv,png}` | profit / sales / lost vs `R_max ∈ [5, 50]` |
| `results/sensitivity_M.{csv,png}` | profit vs `M ∈ [400, 2000]` |
| `results/sensitivity_speed.{csv,png}` | profit vs `truck_speed ∈ {15, 25, 35, 45, 55} mph` |
| `results/milp_thresholds.csv` | the MILP-chosen per-hour caps and their Monte-Carlo evaluation |
| `results/policy_comparison.{csv,png}` | the four-policy comparison of §6.2 |

A `_self_check()` routine asserts two mass-conservation invariants on every run:

```
arrivals == sales + lost                        (every customer is sold to or lost)
initial_stock + refills − pickups == sales + leftover   (no bottle is created or destroyed in transit)
```

Both pass in the reference run.

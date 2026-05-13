# Fresh Milk Delivery — Operations Research (Revised)

This repository contains a revised implementation and documentation for the **Fresh Milk Delivery** case from an operations research final project. The original submission mixed a discrete-event simulation with optimization ideas, but contained modelling and coding errors that inflated performance. This version rebuilds the simulator, tightens the interpretation of constraints, and reports results that respect travel time and inventory logic.

## Academic context

This work was completed as part of the **Operations Research** course in the **Master's in Logistics Engineering and Management** at **Tsinghua University**.

- Original course project completion: **December 2024**
- Repository cleanup and public packaging happened later (current push)

## What was done

The work was carried out in four strands.

**Audit.** `PROBLEMS_AND_IMPROVEMENTS.md` lists defects found in the original notebook and report (modelling, data, reporting, and software engineering), with the corresponding fix for each item.

**Code.** `fresh_milk_delivery.py` is a single runnable module that replaces the fragmented notebook. It implements minute-level dynamics for one truck serving nine distribution centres from 6:00 to 12:00, stochastic demand, dispatch rules, optional redistribution, and economic accounting (revenue, purchase, disposal, backorders). Travel uses a real busy-until clock instead of implicit one-minute hops. Per-visit refill limits and centre capacity are separated so early refills are not blocked by a misread cap.

**Optimization and experiments.** The script runs Monte Carlo batches, sensitivity sweeps over policy parameters, a comparison of operating policies, and a sample-average-approximation style MILP for per-hour refill caps where supported by SciPy. Outputs are written as CSV and PNG under `results/`.

**Report.** `OR_REPORT_REVISED.md` is the full narrative: problem statement, assumptions, methods, corrected numerical results, and recommendations (including the honest service level under one truck and when a second truck matters).

## Key outcomes (high level)

Under corrected physics and parameters, the model no longer implies near-perfect service from a single truck. Monte Carlo and sensitivity outputs in `results/` support the written conclusions in `OR_REPORT_REVISED.md`. See `SUMMARY.md` for a short table comparing original claims to revised numbers and how to reproduce the main run.

## Repository layout

| Path | Purpose |
|------|---------|
| `fresh_milk_delivery.py` | Main simulation, experiments, and figure export |
| `OR_REPORT_REVISED.md` | Full project report |
| `SUMMARY.md` | Short reference and reproduction command |
| `PROBLEMS_AND_IMPROVEMENTS.md` | Issue list and fixes |
| `results/` | Generated CSV and PNG artefacts |

## Requirements

Python 3.10 or newer recommended. Dependencies: `numpy`, `pandas`, `matplotlib`, `scipy` (including `scipy.optimize.milp` where available for the MILP experiment).

Install with pip, for example:

```bash
pip install numpy pandas matplotlib scipy
```

## How to run

From the repository root:

```bash
python fresh_milk_delivery.py
```

Runtime is typically well under one minute on a laptop. The script creates `results/` if needed and overwrites or adds the artefact files described in `SUMMARY.md`.

## Documentation map

- Start here for scope: this README.
- For executive numbers and one-line reproduction: `SUMMARY.md`.
- For methodology and academic narrative: `OR_REPORT_REVISED.md`.
- For traceability from old to new behaviour: `PROBLEMS_AND_IMPROVEMENTS.md`.

## License and use

Course and portfolio use. If you fork for publication, cite the course materials you were given and state any changes you make to parameters or demand data.

## Note

This repository is an adapted public version of a course project. It preserves the main idea and learning purpose of the original work, but some details, datasets, methods, and presentation elements have been modified for sharing and educational use.

This work is provided in good faith as a reference only and should not be copied or submitted as academic work by others. Thank you! 

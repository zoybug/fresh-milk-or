from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import dijkstra

try:
    from scipy.optimize import Bounds, LinearConstraint, milp

    HAS_SCIPY_MILP = True
except Exception:
    HAS_SCIPY_MILP = False
    Bounds = Any
    LinearConstraint = Any
    milp = None


@dataclass(slots=True)
class Params:
    # knobs for the family milk day-ahead sim (see OR_REPORT_REVISED for story)
    # r_max = max bottles dropped off per visit, not "fill to this level"
    # truck_speed default 7/12 mi/min ~ 35 mph, same ballpark as the course notebook
    # centre_capacity = fridge cap; we only ship up to headroom

    truck_speed: float = 7.0 / 12.0
    R_max: int = 30
    M: int = 1000
    centre_capacity: int = 60
    sale_price: float = 8.0
    purchase_price: float = 4.0
    backorder_cost: float = 4.0
    disposal_cost: float = 4.0
    n_centres: int = 10
    hours: int = 6
    minutes_per_hour: int = 60
    initial_per_centre: int = 20
    depot: int = 0
    allow_redistribution: bool = True
    random_seed: int = 202611
    service_time_min: int = 1


HOURLY_LAMBDAS = np.array(
    [
        [18, 10, 10, 9, 14, 17, 15, 11, 8, 19],
        [35, 30, 28, 23, 37, 37, 34, 26, 38, 51],
        [40, 29, 31, 25, 40, 41, 35, 29, 47, 53],
        [36, 27, 27, 22, 32, 34, 29, 26, 40, 46],
        [18, 9, 10, 8, 13, 14, 14, 10, 8, 17],
        [14, 7, 9, 7, 11, 13, 12, 7, 7, 12],
    ],
    dtype=float,
)


def build_distance_matrix() -> np.ndarray:
    n = 10
    dist = np.full((n, n), np.inf, dtype=float)
    np.fill_diagonal(dist, 0.0)
    edges = [
        (1, 2, 2),
        (1, 3, 4),
        (1, 10, 8),
        (2, 3, 3),
        (3, 4, 5),
        (3, 10, 6),
        (10, 7, 2),
        (4, 7, 1),
        (7, 8, 1),
        (4, 5, 1),
        (5, 8, 1),
        (6, 5, 1),
        (8, 9, 1),
        (6, 9, 1),
    ]
    for u, v, w in edges:
        i = u - 1
        j = v - 1
        dist[i, j] = float(w)
        dist[j, i] = float(w)
    return dist


def _results_dir() -> Path:
    # keep outputs next to this file so it works from any clone path
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


class MilkDeliverySim:
    def __init__(
        self,
        params: Params,
        refill_caps_by_hour: list[int] | None = None,
        alpha: float = 0.05,
        enforce_end_of_day_reserve: bool = True,
    ) -> None:
        self.params = params
        self.alpha = alpha
        self.enforce_end_of_day_reserve = enforce_end_of_day_reserve
        self.distance_matrix = build_distance_matrix()
        self.shortest_paths = dijkstra(csgraph=self.distance_matrix, directed=False)
        if refill_caps_by_hour is None:
            self.refill_caps_by_hour = [int(params.R_max)] * params.hours
        else:
            if len(refill_caps_by_hour) != params.hours:
                raise ValueError("refill_caps_by_hour must have length equal to params.hours.")
            self.refill_caps_by_hour = [int(x) for x in refill_caps_by_hour]
        self._lookahead_K = self._compute_lookahead_K()

    @property
    def total_minutes(self) -> int:
        return self.params.hours * self.params.minutes_per_hour

    def _hour_idx(self, minute: int) -> int:
        return min(self.params.hours - 1, minute // self.params.minutes_per_hour)

    def _hour_lambda(self, minute: int) -> np.ndarray:
        return HOURLY_LAMBDAS[self._hour_idx(minute)]

    def _expected_remaining_demand(self, centre: int, minute: int) -> float:
        total = 0.0
        t = minute
        while t < self.total_minutes:
            hour_idx = self._hour_idx(t)
            hour_end = min((hour_idx + 1) * self.params.minutes_per_hour, self.total_minutes)
            minutes_here = hour_end - t
            total += HOURLY_LAMBDAS[hour_idx, centre] * (minutes_here / self.params.minutes_per_hour)
            t = hour_end
        return total

    def _compute_lookahead_K(self) -> int:
        # how many minutes ahead we care about when scoring "risky" centres (dispatch heuristic)
        n = self.params.n_centres
        sp = self.shortest_paths
        iu = np.triu_indices(n, k=1)
        legs = sp[iu]
        legs = legs[np.isfinite(legs) & (legs > 0)]
        avg_dist = float(np.mean(legs)) if legs.size > 0 else 2.0
        leg_time = int(ceil(avg_dist / self.params.truck_speed)) + self.params.service_time_min
        avg_round_trip = 2 * leg_time
        return max(20, 2 * avg_round_trip)

    def _expected_demand_over_minutes(self, centre: int, start_min: int, duration_min: float) -> float:
        total = 0.0
        t = float(start_min)
        end = min(float(start_min) + float(duration_min), float(self.total_minutes))
        while t < end - 1e-9:
            hi = self._hour_idx(int(t))
            hour_end = min(float((hi + 1) * self.params.minutes_per_hour), end)
            lam_per_min = HOURLY_LAMBDAS[hi, centre] / float(self.params.minutes_per_hour)
            total += lam_per_min * (hour_end - t)
            t = hour_end
        return total

    def _should_refill(self, centre: int, minute: int, stock_at_centre: int) -> bool:
        if not self.enforce_end_of_day_reserve:
            return True
        if self.total_minutes - minute > 30:
            return True
        rem = self._expected_remaining_demand(centre, minute)
        return bool(stock_at_centre < rem)

    def _per_visit_cap(self, minute: int) -> int:
        return int(self.refill_caps_by_hour[self._hour_idx(minute)])

    def _delivery_refill_amount(self, dest: int, arrival_minute: int, stock: np.ndarray, payload: int) -> int:
        if payload <= 0:
            return 0
        cap = self.params.centre_capacity
        if int(stock[dest]) >= cap:
            return 0
        if not self._should_refill(dest, arrival_minute, int(stock[dest])):
            return 0
        per_visit = self._per_visit_cap(arrival_minute)
        headroom = cap - int(stock[dest])
        headroom = max(0, headroom)
        return int(min(per_visit, headroom, payload))

    def _choose_delivery_target(
        self,
        minute: int,
        truck_position: int,
        stock: np.ndarray,
        payload: int,
        forced_target: int | None = None,
    ) -> tuple[int, int, float] | None:
        if payload <= 0:
            return None

        cap_lim = self.params.centre_capacity
        minutes_left = self.total_minutes - minute
        K = float(self._lookahead_K)

        def finish_ok(travel: int) -> int:
            return minute + travel + self.params.service_time_min

        if forced_target is not None:
            d_forced = float(self.shortest_paths[truck_position, forced_target])
            travel_forced = int(ceil(d_forced / self.params.truck_speed))
            fin = finish_ok(travel_forced)
            if (
                np.isfinite(d_forced)
                and fin <= self.total_minutes
                and int(stock[forced_target]) < cap_lim
                and self._delivery_refill_amount(
                    forced_target, fin, stock, payload
                )
                > 0
            ):
                return forced_target, travel_forced, d_forced

        best_risk: tuple[float, int, int, float] | None = None
        for j in range(self.params.n_centres):
            dist = float(self.shortest_paths[truck_position, j])
            if not np.isfinite(dist):
                continue
            travel = int(ceil(dist / self.params.truck_speed))
            fin = finish_ok(travel)
            if travel + self.params.service_time_min > minutes_left or fin > self.total_minutes:
                continue
            if int(stock[j]) >= cap_lim:
                continue
            if self._delivery_refill_amount(j, fin, stock, payload) <= 0:
                continue
            exp_k = self._expected_demand_over_minutes(j, minute, K)
            risk_j = max(0.0, exp_k - float(stock[j]))
            score = risk_j - self.alpha * dist + 1e-6 * float(cap_lim - int(stock[j]))
            if best_risk is None or score > best_risk[0]:
                best_risk = (score, j, travel, dist)

        if best_risk is not None and best_risk[0] > 1e-12:
            return best_risk[1], best_risk[2], best_risk[3]

        best_idle: tuple[int, int, int, float] | None = None
        for j in range(self.params.n_centres):
            dist = float(self.shortest_paths[truck_position, j])
            if not np.isfinite(dist):
                continue
            travel = int(ceil(dist / self.params.truck_speed))
            fin = finish_ok(travel)
            if travel + self.params.service_time_min > minutes_left or fin > self.total_minutes:
                continue
            headroom = cap_lim - int(stock[j])
            if headroom <= 0:
                continue
            if self._delivery_refill_amount(j, fin, stock, payload) <= 0:
                continue
            score0 = -float(stock[j]) + 1e-6 * float(headroom)
            if best_idle is None or score0 > best_idle[0]:
                best_idle = (score0, j, travel, dist)

        if best_idle is None:
            return None
        return best_idle[1], best_idle[2], best_idle[3]

    def _choose_pickup_pair(
        self,
        minute: int,
        stock: np.ndarray,
    ) -> tuple[int, int, int] | None:
        if not self.params.allow_redistribution:
            return None
        if minute < 4 * self.params.minutes_per_hour:
            return None

        rem_exp = np.array(
            [self._expected_remaining_demand(j, minute) for j in range(self.params.n_centres)],
            dtype=float,
        )
        surplus = stock - rem_exp
        donor_candidates = np.where(stock > 2.0 * rem_exp)[0]
        if donor_candidates.size == 0:
            return None
        receiver_need = np.maximum(0.0, rem_exp - stock)
        if receiver_need.max() < 1.0:
            return None

        donor = int(donor_candidates[np.argmax(surplus[donor_candidates])])
        receiver = int(np.argmax(receiver_need))
        if receiver == donor:
            return None
        qty = int(max(1.0, min(surplus[donor], receiver_need[receiver])))
        if qty <= 0:
            return None
        return donor, receiver, qty

    def simulate(self, seed: int) -> dict[str, Any]:
        rng = np.random.default_rng(int(seed))
        n = self.params.n_centres
        total_minutes = self.total_minutes

        stock = np.full(n, int(self.params.initial_per_centre), dtype=np.int64)
        sales = np.zeros(n, dtype=np.int64)
        lost = np.zeros(n, dtype=np.int64)
        arrivals_total = np.zeros(n, dtype=np.int64)

        truck_position = int(self.params.depot)
        truck_busy_until = 0
        truck_payload = int(max(0, self.params.M - self.params.initial_per_centre * n))
        pending_action: dict[str, Any] | None = None
        pending_drop_target: int | None = None

        distance_travelled = 0.0
        refill_log: list[dict[str, Any]] = []

        for t in range(total_minutes):
            lam_per_min = self._hour_lambda(t) / self.params.minutes_per_hour
            arrivals = rng.poisson(lam=lam_per_min).astype(np.int64)
            arrivals_total += arrivals
            sold = np.minimum(arrivals, stock)
            sales += sold
            stock -= sold
            lost += arrivals - sold

            if pending_action is not None and t >= truck_busy_until:
                dest = int(pending_action["destination"])
                action = str(pending_action["action"])
                qty = 0
                if action == "deliver":
                    refill = self._delivery_refill_amount(dest, t, stock, truck_payload)
                    if refill > 0:
                        stock[dest] += refill
                        truck_payload -= refill
                        qty = int(refill)
                elif action == "pickup":
                    planned = int(pending_action["planned_qty"])
                    pickup = min(planned, int(stock[dest]))
                    if pickup > 0:
                        stock[dest] -= pickup
                        truck_payload += pickup
                        qty = -int(pickup)
                    pending_drop_target = int(pending_action["drop_target"])
                refill_log.append(
                    {
                        "Time (minute)": int(t),
                        "From": int(pending_action["from"]) + 1,
                        "To": dest + 1,
                        "Refill Quantity": int(qty),
                        "Action": action,
                        "Truck Payload After": int(truck_payload),
                        "Distance": float(pending_action["distance"]),
                    }
                )
                truck_position = dest
                pending_action = None

            if pending_action is None and t >= truck_busy_until:
                if truck_payload == 0:
                    pickup_choice = self._choose_pickup_pair(t, stock)
                    if pickup_choice is not None:
                        donor, receiver, qty = pickup_choice
                        dist = float(self.shortest_paths[truck_position, donor])
                        travel = int(ceil(dist / self.params.truck_speed))
                        finish = t + travel + self.params.service_time_min
                        if np.isfinite(dist) and finish <= total_minutes:
                            pending_action = {
                                "action": "pickup",
                                "from": truck_position,
                                "destination": donor,
                                "planned_qty": qty,
                                "drop_target": receiver,
                                "distance": dist,
                            }
                            truck_busy_until = finish
                            distance_travelled += dist
                            continue
                dispatch = self._choose_delivery_target(
                    minute=t,
                    truck_position=truck_position,
                    stock=stock,
                    payload=truck_payload,
                    forced_target=pending_drop_target,
                )
                if dispatch is not None:
                    target, travel, dist = dispatch
                    pending_action = {
                        "action": "deliver",
                        "from": truck_position,
                        "destination": target,
                        "distance": dist,
                    }
                    truck_busy_until = t + travel + self.params.service_time_min
                    distance_travelled += dist
                    if pending_drop_target is not None and target == pending_drop_target:
                        pending_drop_target = None

        if pending_action is not None:
            dest = int(pending_action["destination"])
            action = str(pending_action["action"])
            qty = 0
            if action == "deliver":
                refill = self._delivery_refill_amount(dest, total_minutes - 1, stock, truck_payload)
                if refill > 0:
                    stock[dest] += refill
                    truck_payload -= refill
                    qty = int(refill)
            else:
                planned = int(pending_action["planned_qty"])
                pickup = min(planned, int(stock[dest]))
                if pickup > 0:
                    stock[dest] -= pickup
                    truck_payload += pickup
                    qty = -int(pickup)
            refill_log.append(
                {
                    "Time (minute)": int(total_minutes),
                    "From": int(pending_action["from"]) + 1,
                    "To": dest + 1,
                    "Refill Quantity": int(qty),
                    "Action": action,
                    "Truck Payload After": int(truck_payload),
                    "Distance": float(pending_action["distance"]),
                }
            )
            truck_position = dest

        if np.isfinite(self.shortest_paths[truck_position, self.params.depot]):
            distance_travelled += float(self.shortest_paths[truck_position, self.params.depot])

        total_sales = int(sales.sum())
        total_lost = int(lost.sum())
        total_revenue = self.params.sale_price * total_sales
        total_purchase_cost = self.params.purchase_price * self.params.M
        disposal_qty = int(stock.sum() + truck_payload)
        total_disposal_cost = self.params.disposal_cost * disposal_qty
        total_backorder_cost = self.params.backorder_cost * total_lost
        profit = total_revenue - total_purchase_cost - total_disposal_cost - total_backorder_cost

        return {
            "sales_per_centre": sales.astype(int).tolist(),
            "lost_sales_per_centre": lost.astype(int).tolist(),
            "leftover_per_centre": stock.astype(int).tolist(),
            "arrivals_per_centre": arrivals_total.astype(int).tolist(),
            "refill_log": refill_log,
            "distance_travelled": float(distance_travelled),
            "truck_payload_leftover": int(truck_payload),
            "total_revenue": float(total_revenue),
            "total_purchase_cost": float(total_purchase_cost),
            "total_disposal_cost": float(total_disposal_cost),
            "total_backorder_cost": float(total_backorder_cost),
            "profit": float(profit),
        }


def _finalize_plot(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_single_day(params: Params, seed: int = 42) -> dict[str, Any]:
    out_dir = _results_dir()
    sim = MilkDeliverySim(params=params)
    result = sim.simulate(seed=seed)
    log_df = pd.DataFrame(result["refill_log"])
    summary_df = pd.DataFrame(
        [
            {
                "profit": result["profit"],
                "sales": sum(result["sales_per_centre"]),
                "lost": sum(result["lost_sales_per_centre"]),
                "leftover_centres": sum(result["leftover_per_centre"]),
                "leftover_truck": result["truck_payload_leftover"],
                "distance": result["distance_travelled"],
                "total_revenue": result["total_revenue"],
                "purchase_cost": result["total_purchase_cost"],
                "disposal_cost": result["total_disposal_cost"],
                "backorder_cost": result["total_backorder_cost"],
            }
        ]
    )
    log_df.to_csv(out_dir / "single_day_log.csv", index=False)
    summary_df.to_csv(out_dir / "single_day_summary.csv", index=False)
    row = summary_df.iloc[0]
    print(
        f"[single day] profit={row['profit']:.1f}, sales={int(row['sales'])}, "
        f"lost={int(row['lost'])}, leftover={int(row['leftover_centres'])}+{int(row['leftover_truck'])}, "
        f"distance={row['distance']:.1f}"
    )
    return result


def run_monte_carlo(params: Params, n_runs: int = 500, seed: int = 0) -> pd.DataFrame:
    out_dir = _results_dir()
    sim = MilkDeliverySim(params=params)
    root_rng = np.random.default_rng(seed)
    day_seeds = root_rng.integers(0, 2**32 - 1, size=n_runs, dtype=np.uint32)
    rows: list[dict[str, Any]] = []

    for i in range(n_runs):
        res = sim.simulate(seed=int(day_seeds[i]))
        row = {
            "run": i,
            "profit": res["profit"],
            "sales": int(sum(res["sales_per_centre"])),
            "lost": int(sum(res["lost_sales_per_centre"])),
            "leftover": int(sum(res["leftover_per_centre"]) + res["truck_payload_leftover"]),
            "refilled": int(sum(max(0, int(x["Refill Quantity"])) for x in res["refill_log"])),
            "distance": res["distance_travelled"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "monte_carlo_500.csv", index=False)
    desc = df.describe().T
    desc.to_csv(out_dir / "monte_carlo_summary.csv")
    print(
        f"[monte carlo] n={n_runs}, mean profit={df['profit'].mean():.2f}, "
        f"mean sales={df['sales'].mean():.2f}, mean lost={df['lost'].mean():.2f}"
    )
    return df


def sensitivity_max_refill(
    params: Params,
    refill_caps: range = range(5, 51, 2),
    n_runs: int = 200,
) -> pd.DataFrame:
    out_dir = _results_dir()
    rng = np.random.default_rng(params.random_seed + 11)
    records: list[dict[str, Any]] = []

    for cap in refill_caps:
        sim = MilkDeliverySim(params=replace(params, R_max=int(cap)))
        seeds = rng.integers(0, 2**32 - 1, size=n_runs, dtype=np.uint32)
        profits: list[float] = []
        sales: list[int] = []
        lost: list[int] = []
        for s in seeds:
            res = sim.simulate(seed=int(s))
            profits.append(float(res["profit"]))
            sales.append(int(sum(res["sales_per_centre"])))
            lost.append(int(sum(res["lost_sales_per_centre"])))
        records.append(
            {
                "R_max": int(cap),
                "mean_profit": float(np.mean(profits)),
                "mean_sales": float(np.mean(sales)),
                "mean_lost": float(np.mean(lost)),
            }
        )

    df = pd.DataFrame(records).sort_values("R_max")
    df.to_csv(out_dir / "sensitivity_max_refill.csv", index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(df["R_max"], df["mean_sales"], marker="o", label="Mean Sales")
    plt.plot(df["R_max"], df["mean_lost"], marker="s", label="Mean Lost Sales")
    plt.xlabel("R_max")
    plt.ylabel("Bottles")
    plt.title("Sensitivity: Sales and Lost Sales vs R_max")
    plt.grid(True, alpha=0.3)
    plt.legend()
    _finalize_plot(out_dir / "sensitivity_max_refill.png")
    best = df.iloc[df["mean_profit"].idxmax()]
    print(f"[sens R_max] best mean profit at R_max={int(best['R_max'])}: {best['mean_profit']:.2f}")
    return df


def sensitivity_M(
    params: Params,
    M_values: range = range(400, 2001, 100),
    n_runs: int = 200,
) -> pd.DataFrame:
    out_dir = _results_dir()
    rng = np.random.default_rng(params.random_seed + 23)
    records: list[dict[str, Any]] = []

    for m in M_values:
        sim = MilkDeliverySim(params=replace(params, M=int(m)))
        seeds = rng.integers(0, 2**32 - 1, size=n_runs, dtype=np.uint32)
        profits = []
        sales = []
        lost = []
        disposal = []
        for s in seeds:
            res = sim.simulate(seed=int(s))
            profits.append(float(res["profit"]))
            sales.append(int(sum(res["sales_per_centre"])))
            lost.append(int(sum(res["lost_sales_per_centre"])))
            disposal.append(float(res["total_disposal_cost"]))
        records.append(
            {
                "M": int(m),
                "mean_profit": float(np.mean(profits)),
                "mean_sales": float(np.mean(sales)),
                "mean_lost": float(np.mean(lost)),
                "mean_disposal_cost": float(np.mean(disposal)),
            }
        )

    df = pd.DataFrame(records).sort_values("M")
    df.to_csv(out_dir / "sensitivity_M.csv", index=False)
    plt.figure(figsize=(10, 5))
    plt.plot(df["M"], df["mean_profit"], marker="o")
    plt.xlabel("M (initial bottles)")
    plt.ylabel("Mean profit")
    plt.title("Sensitivity: Profit vs M")
    plt.grid(True, alpha=0.3)
    _finalize_plot(out_dir / "sensitivity_M.png")
    best = df.iloc[df["mean_profit"].idxmax()]
    print(f"[sens M] profit-maximising M={int(best['M'])}, mean profit={best['mean_profit']:.2f}")
    return df


def sensitivity_truck_speed(
    params: Params,
    speeds: list[float] = [3 / 12, 5 / 12, 7 / 12, 9 / 12, 11 / 12],
    n_runs: int = 200,
) -> pd.DataFrame:
    out_dir = _results_dir()
    rng = np.random.default_rng(params.random_seed + 37)
    records: list[dict[str, Any]] = []
    for speed in speeds:
        sim = MilkDeliverySim(params=replace(params, truck_speed=float(speed)))
        seeds = rng.integers(0, 2**32 - 1, size=n_runs, dtype=np.uint32)
        profits = []
        sales = []
        lost = []
        dist = []
        for s in seeds:
            res = sim.simulate(seed=int(s))
            profits.append(float(res["profit"]))
            sales.append(int(sum(res["sales_per_centre"])))
            lost.append(int(sum(res["lost_sales_per_centre"])))
            dist.append(float(res["distance_travelled"]))
        records.append(
            {
                "truck_speed_miles_per_min": float(speed),
                "mean_profit": float(np.mean(profits)),
                "mean_sales": float(np.mean(sales)),
                "mean_lost": float(np.mean(lost)),
                "mean_distance": float(np.mean(dist)),
            }
        )

    df = pd.DataFrame(records).sort_values("truck_speed_miles_per_min")
    df.to_csv(out_dir / "sensitivity_speed.csv", index=False)
    plt.figure(figsize=(10, 5))
    plt.plot(df["truck_speed_miles_per_min"], df["mean_profit"], marker="o")
    plt.xlabel("Truck speed (miles/min)")
    plt.ylabel("Mean profit")
    plt.title("Sensitivity: Profit vs Truck Speed")
    plt.grid(True, alpha=0.3)
    _finalize_plot(out_dir / "sensitivity_speed.png")
    return df


def _scenario_demands(n_scenarios: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    scenarios = rng.poisson(lam=HOURLY_LAMBDAS[None, :, :], size=(n_scenarios, 6, 10))
    return scenarios.astype(float)


def _enumerative_hourwise_relaxation(params: Params, scenario_demands: np.ndarray) -> tuple[list[int], float]:
    # cheap hour-by-hour cap search: pretend visits_per_hour ~ f(speed, avg leg), cap r_h, score demand vs supply
    r_grid = np.arange(5, 41)
    n_s = scenario_demands.shape[0]
    mean_hour_demand = scenario_demands.sum(axis=2).mean(axis=0)
    sp = dijkstra(csgraph=build_distance_matrix(), directed=False)
    iu = np.triu_indices(params.n_centres, k=1)
    avg_leg_miles = float(np.mean(sp[iu][np.isfinite(sp[iu]) & (sp[iu] > 0)]))
    minutes_per_visit = avg_leg_miles / params.truck_speed + params.service_time_min
    visits_per_hour = max(1.0, params.minutes_per_hour / minutes_per_visit)
    disposal_penalty = 0.05 * params.disposal_cost
    chosen: list[int] = []
    objective = 0.0
    for h in range(params.hours):
        scores = []
        for r in r_grid:
            supply = visits_per_hour * float(r)
            sold = min(mean_hour_demand[h], supply)
            lost = max(0.0, mean_hour_demand[h] - sold)
            over = max(0.0, supply - mean_hour_demand[h])
            score = params.sale_price * sold - params.backorder_cost * lost - disposal_penalty * over
            scores.append(score)
        idx = int(np.argmax(scores))
        chosen.append(int(r_grid[idx]))
        objective += float(scores[idx]) / n_s
    objective -= params.purchase_price * params.M
    return chosen, objective


def optimize_refill_thresholds_milp(params: Params, n_scenarios: int = 30, seed: int = 0) -> pd.DataFrame:
    # milp-ish relaxation: pick integer r_h per hour, not a full routing model (gamma hacks truck throughput)
    out_dir = _results_dir()
    scenarios = _scenario_demands(n_scenarios=n_scenarios, seed=seed)
    demand_sh = scenarios.sum(axis=2)
    sp = dijkstra(csgraph=build_distance_matrix(), directed=False)
    iu = np.triu_indices(params.n_centres, k=1)
    avg_leg_miles = float(np.mean(sp[iu][np.isfinite(sp[iu]) & (sp[iu] > 0)]))
    minutes_per_visit = avg_leg_miles / params.truck_speed + params.service_time_min
    gamma = max(1.0, params.minutes_per_hour / minutes_per_visit)

    if HAS_SCIPY_MILP and milp is not None:
        n_r = params.hours
        n_aux = n_scenarios * params.hours
        n_var = n_r + n_aux

        c = np.zeros(n_var, dtype=float)
        coeff_sold = -(params.sale_price + params.backorder_cost) / n_scenarios
        c[n_r:] = coeff_sold
        c[:n_r] = (0.05 * params.disposal_cost) / params.hours

        integrality = np.zeros(n_var, dtype=int)
        integrality[:n_r] = 1

        lb = np.zeros(n_var, dtype=float)
        ub = np.full(n_var, np.inf, dtype=float)
        lb[:n_r] = 1.0
        ub[:n_r] = 40.0

        rows: list[np.ndarray] = []
        lower: list[float] = []
        upper: list[float] = []
        for s in range(n_scenarios):
            for h in range(params.hours):
                aux_idx = n_r + s * params.hours + h
                row_a = np.zeros(n_var, dtype=float)
                row_a[aux_idx] = 1.0
                rows.append(row_a)
                lower.append(0.0)
                upper.append(float(demand_sh[s, h]))

                row_b = np.zeros(n_var, dtype=float)
                row_b[aux_idx] = 1.0
                row_b[h] = -gamma
                rows.append(row_b)
                lower.append(-np.inf)
                upper.append(0.0)

        A = np.vstack(rows)
        constraints = LinearConstraint(A=A, lb=np.array(lower), ub=np.array(upper))
        bounds = Bounds(lb=lb, ub=ub)
        res = milp(c=c, integrality=integrality, bounds=bounds, constraints=constraints)
        if res.success and res.x is not None:
            r_hour = [int(round(v)) for v in res.x[: params.hours]]
            method = "scipy_milp"
            relaxed_objective = float(-res.fun - params.purchase_price * params.M)
        else:
            r_hour, relaxed_objective = _enumerative_hourwise_relaxation(params, scenarios)
            method = "enumerative_fallback"
    else:
        r_hour, relaxed_objective = _enumerative_hourwise_relaxation(params, scenarios)
        method = "enumerative_fallback"

    eval_sim = MilkDeliverySim(params=params, refill_caps_by_hour=r_hour)
    eval_rng = np.random.default_rng(params.random_seed + 101)
    eval_seeds = eval_rng.integers(0, 2**32 - 1, size=200, dtype=np.uint32)
    eval_profits = []
    eval_sales = []
    eval_lost = []
    for s in eval_seeds:
        sim_res = eval_sim.simulate(seed=int(s))
        eval_profits.append(sim_res["profit"])
        eval_sales.append(sum(sim_res["sales_per_centre"]))
        eval_lost.append(sum(sim_res["lost_sales_per_centre"]))

    df = pd.DataFrame(
        {
            "hour": [1, 2, 3, 4, 5, 6],
            "refill_cap": r_hour,
            "method": [method] * params.hours,
            "relaxed_objective": [relaxed_objective] * params.hours,
            "eval_mean_profit_200": [float(np.mean(eval_profits))] * params.hours,
            "eval_mean_sales_200": [float(np.mean(eval_sales))] * params.hours,
            "eval_mean_lost_200": [float(np.mean(eval_lost))] * params.hours,
        }
    )
    df.to_csv(out_dir / "milp_thresholds.csv", index=False)
    print(f"[MILP] method={method}, r_h={r_hour}, eval mean profit={np.mean(eval_profits):.2f}")
    return df


def compare_policies(params: Params, n_runs: int = 300) -> pd.DataFrame:
    out_dir = _results_dir()
    milp_path = out_dir / "milp_thresholds.csv"
    if milp_path.exists():
        milp_df = pd.read_csv(milp_path)
    else:
        milp_df = optimize_refill_thresholds_milp(params=params, n_scenarios=30, seed=0)
    milp_caps = milp_df["refill_cap"].astype(int).tolist()

    policies = {
        "original_myopic_R15": {
            "sim": MilkDeliverySim(
                params=replace(params, R_max=15),
                refill_caps_by_hour=[15] * params.hours,
                enforce_end_of_day_reserve=False,
            )
        },
        "team_R18": {
            "sim": MilkDeliverySim(
                params=replace(params, R_max=18),
                refill_caps_by_hour=[18] * params.hours,
                enforce_end_of_day_reserve=False,
            )
        },
        "tuned_R30": {
            "sim": MilkDeliverySim(
                params=replace(params, R_max=30),
                refill_caps_by_hour=[30] * params.hours,
                enforce_end_of_day_reserve=True,
            )
        },
        "new_milp_per_hour": {
            "sim": MilkDeliverySim(
                params=params,
                refill_caps_by_hour=milp_caps,
                enforce_end_of_day_reserve=True,
            )
        },
    }

    rng = np.random.default_rng(params.random_seed + 303)
    common_seeds = rng.integers(0, 2**32 - 1, size=n_runs, dtype=np.uint32)
    rows: list[dict[str, Any]] = []
    for name, payload in policies.items():
        sim = payload["sim"]
        profits = []
        sales = []
        lost = []
        leftover = []
        for s in common_seeds:
            res = sim.simulate(seed=int(s))
            profits.append(float(res["profit"]))
            sales.append(int(sum(res["sales_per_centre"])))
            lost.append(int(sum(res["lost_sales_per_centre"])))
            leftover.append(int(sum(res["leftover_per_centre"]) + res["truck_payload_leftover"]))
        rows.append(
            {
                "policy": name,
                "mean_profit": float(np.mean(profits)),
                "mean_sales": float(np.mean(sales)),
                "mean_lost": float(np.mean(lost)),
                "mean_leftover": float(np.mean(leftover)),
            }
        )

    df = pd.DataFrame(rows).sort_values("mean_profit", ascending=False)
    df.to_csv(out_dir / "policy_comparison.csv", index=False)
    plt.figure(figsize=(10, 5))
    plt.bar(df["policy"], df["mean_profit"])
    plt.ylabel("Mean profit")
    plt.title("Policy Comparison (higher is better)")
    plt.xticks(rotation=15)
    plt.grid(axis="y", alpha=0.3)
    _finalize_plot(out_dir / "policy_comparison.png")
    print("[policy] best:", df.iloc[0]["policy"], f"(mean profit {df.iloc[0]['mean_profit']:.2f})")
    return df


def _self_check() -> None:
    params = Params()
    sim = MilkDeliverySim(params=params)
    result = sim.simulate(seed=123456)
    arrivals = int(sum(result["arrivals_per_centre"]))
    sales = int(sum(result["sales_per_centre"]))
    lost = int(sum(result["lost_sales_per_centre"]))
    lhs1 = arrivals
    rhs1 = sales + lost
    if lhs1 != rhs1:
        raise AssertionError(f"Mass check 1 failed: arrivals={lhs1}, sales+lost={rhs1}")
    print("[self-check] PASS arrivals = sales + lost")

    initial_stock = params.n_centres * params.initial_per_centre
    refills = sum(max(0, int(x["Refill Quantity"])) for x in result["refill_log"])
    pickups = sum(max(0, -int(x["Refill Quantity"])) for x in result["refill_log"])
    leftover = int(sum(result["leftover_per_centre"]))
    lhs2 = initial_stock + refills - pickups
    rhs2 = sales + leftover
    if lhs2 != rhs2:
        raise AssertionError(
            f"Mass check 2 failed: initial+refills-pickups={lhs2}, sales+leftover={rhs2}"
        )
    print("[self-check] PASS centre stock flow conserved")


def _short_print_df(df: pd.DataFrame, rows: int = 8) -> None:
    preview = df.head(rows).copy()
    print(preview.to_string(index=False))


def main() -> None:
    params = Params()
    print(
        f"Fresh Milk Delivery | M={params.M}, R_max={params.R_max}, speed={params.truck_speed:.3f} mi/min, "
        f"seed={params.random_seed}, redistribution={params.allow_redistribution}"
    )
    _self_check()

    run_single_day(params=params, seed=42)

    mc_df = run_monte_carlo(params=params, n_runs=500, seed=0)
    _short_print_df(mc_df[["run", "profit", "sales", "lost", "leftover", "distance"]], rows=6)

    rmax_df = sensitivity_max_refill(params=params, refill_caps=range(2, 31), n_runs=200)
    _short_print_df(rmax_df[["R_max", "mean_profit", "mean_sales", "mean_lost"]], rows=8)

    m_df = sensitivity_M(params=params, M_values=range(800, 2001, 100), n_runs=200)
    best_m = int(m_df.iloc[m_df["mean_profit"].idxmax()]["M"])
    print(f"[summary] profit-maximising M from sensitivity_M: {best_m}")
    _short_print_df(m_df[["M", "mean_profit", "mean_sales", "mean_lost"]], rows=8)

    speed_df = sensitivity_truck_speed(
        params=params, speeds=[3 / 12, 5 / 12, 7 / 12, 9 / 12, 11 / 12], n_runs=200
    )
    _short_print_df(
        speed_df[["truck_speed_miles_per_min", "mean_profit", "mean_sales", "mean_lost"]], rows=5
    )

    milp_df = optimize_refill_thresholds_milp(params=params, n_scenarios=30, seed=0)
    _short_print_df(milp_df[["hour", "refill_cap", "method", "eval_mean_profit_200"]], rows=6)

    policy_df = compare_policies(params=params, n_runs=300)
    _short_print_df(policy_df[["policy", "mean_profit", "mean_sales", "mean_lost"]], rows=3)

    print(f"All outputs saved to: {_results_dir()}")


if __name__ == "__main__":
    main()

#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np
def compute_enzyme_demand(levels: "np.ndarray", kcat: "np.ndarray", km: "np.ndarray") -> "np.ndarray":
    """Reference implementation (backward accumulation of the diluted flux)."""
    import numpy as np

    def _positive_vector(value, name):
        try:
            array = np.array(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric array") from None
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not (np.all(np.isfinite(array)) and np.all(array > 0.0)):
            raise ValueError(f"{name} must hold finite positive numbers")
        return array

    rho = _positive_vector(levels, "levels")
    rate = _positive_vector(kcat, "kcat")
    affinity = _positive_vector(km, "km")
    if not (rho.shape == rate.shape == affinity.shape):
        raise ValueError("levels, kcat and km must have the same length")
    # Flux of enzyme i per unit growth rate: the protein flux (1) plus the
    # dilution of every metabolite downstream of its product, i + 1 .. n - 1.
    downstream = np.concatenate([np.cumsum(rho[::-1])[::-1][1:], [0.0]])
    flux = 1.0 + downstream
    return flux * (1.0 + affinity / rho) / rate

import numpy as np
def compute_metabolite_costs(levels: "np.ndarray", kcat: "np.ndarray", km: "np.ndarray") -> "np.ndarray":
    """Reference implementation (stationarity of c * x + g_i(x) at the given level)."""
    import numpy as np

    def _positive_vector(value, name):
        try:
            array = np.array(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric array") from None
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not (np.all(np.isfinite(array)) and np.all(array > 0.0)):
            raise ValueError(f"{name} must hold finite positive numbers")
        return array

    rho = _positive_vector(levels, "levels")
    rate = _positive_vector(kcat, "kcat")
    affinity = _positive_vector(km, "km")
    if not (rho.shape == rate.shape == affinity.shape):
        raise ValueError("levels, kcat and km must have the same length")
    # g_i(x) = flux_i (1 + km_i / x) / kcat_i with flux_i fixed; the minimiser of
    # c x + g_i(x) is x = sqrt(flux_i km_i / (kcat_i c)), inverted for c here.
    downstream = np.concatenate([np.cumsum(rho[::-1])[::-1][1:], [0.0]])
    flux = 1.0 + downstream
    return affinity * flux / (rate * rho * rho)

import numpy as np
def propagate_optimal_levels(
    anchor: float,
    kcat: np.ndarray,
    km: np.ndarray,
    coordinate: str = "last",
    return_tangent: bool = False,
) -> np.ndarray:
    """Reference implementation (upstream recursion and family inversion)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _positive_vector(value, name):
        try:
            array = np.array(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric array") from None
        if array.ndim != 1 or array.size == 0:
            raise ValueError(f"{name} must be a non-empty one-dimensional array")
        if not (np.all(np.isfinite(array)) and np.all(array > 0.0)):
            raise ValueError(f"{name} must hold finite positive numbers")
        return array

    if not (_is_number(anchor) and anchor > 0.0):
        raise ValueError("anchor must be a finite positive number")
    if coordinate not in ("last", "total") or not isinstance(coordinate, str):
        raise ValueError("coordinate must be 'last' or 'total'")
    if not isinstance(return_tangent, (bool, np.bool_)):
        raise ValueError("return_tangent must be boolean")
    rate = _positive_vector(kcat, "kcat")
    affinity = _positive_vector(km, "km")
    if rate.shape != affinity.shape:
        raise ValueError("kcat and km must have the same length")
    count = rate.size

    def _from_last(last_level):
        rho = np.empty(count)
        tangent = np.empty(count)
        rho[-1] = float(last_level)
        tangent[-1] = 1.0
        below = 0.0        # total level of metabolites downstream of rho[i]
        below_tangent = 0.0
        for i in range(count - 1, 0, -1):
            # Equal derivatives of D with respect to rho[i - 1] and rho[i]
            # leave one positive root of a quadratic for rho[i - 1].
            cost = affinity[i] * (1.0 + below) / (rate[i] * rho[i] ** 2)
            lead = cost - 1.0 / rate[i - 1]
            if not (np.isfinite(lead) and lead > 0.0):
                raise ValueError("no positive optimal levels for this last level")
            lead_tangent = cost * (
                below_tangent / (1.0 + below) - 2.0 * tangent[i] / rho[i]
            )
            below += rho[i]
            below_tangent += tangent[i]
            linear = affinity[i - 1] / rate[i - 1]
            constant = affinity[i - 1] * (1.0 + below) / rate[i - 1]
            constant_tangent = linear * below_tangent
            discriminant = linear * linear + 4.0 * lead * constant
            rho[i - 1] = (linear + np.sqrt(discriminant)) / (2.0 * lead)
            if not (np.isfinite(rho[i - 1]) and rho[i - 1] > 0.0):
                raise ValueError("no finite positive optimal levels for this last level")
            tangent[i - 1] = (
                constant_tangent - lead_tangent * rho[i - 1] ** 2
            ) / (2.0 * lead * rho[i - 1] - linear)
            if not np.isfinite(tangent[i - 1]):
                raise ValueError("stationary-family tangent is not finite")
        return rho, tangent

    if coordinate == "last":
        profile, tangent = _from_last(float(anchor))
        return np.vstack([profile, tangent]) if return_tangent else profile
    if count == 1:
        profile = np.array([float(anchor)])
        return np.vstack([profile, np.ones(1)]) if return_tangent else profile

    # The total pool grows strictly along the feasible branch. The requested
    # last level cannot exceed the total, while an infeasible upper point is
    # also known to lie above the requested member. Find a feasible lower
    # point, then invert the family in log(last_level).
    log_high = float(np.log(anchor))
    log_low = log_high
    lower = None
    log_two = float(np.log(2.0))
    log_tiny = float(np.log(np.nextafter(0.0, 1.0)))
    for _ in range(2048):
        log_low -= log_two
        if log_low <= log_tiny:
            break
        try:
            candidate, _ = _from_last(float(np.exp(log_low)))
        except ValueError:
            continue
        if float(candidate.sum()) < float(anchor):
            lower = candidate
            break
    if lower is None:
        raise ValueError("no positive optimal profile has this total level")

    for _ in range(256):
        if log_high - log_low <= 5.0e-14:
            break
        log_mid = 0.5 * (log_low + log_high)
        try:
            candidate, _ = _from_last(float(np.exp(log_mid)))
        except ValueError:
            log_high = log_mid
            continue
        if float(candidate.sum()) < float(anchor):
            log_low = log_mid
            lower = candidate
        else:
            log_high = log_mid
    log_mid = 0.5 * (log_low + log_high)
    try:
        result, tangent = _from_last(float(np.exp(log_mid)))
    except ValueError:
        result = lower
        _, tangent = _from_last(float(result[-1]))
    if abs(float(result.sum()) / float(anchor) - 1.0) > 1.0e-12:
        raise ValueError("requested total level was not reached")
    total_tangent = float(tangent.sum())
    if not (np.isfinite(total_tangent) and total_tangent > 0.0):
        raise ValueError("stationary-family total tangent is not positive")
    tangent = tangent / total_tangent
    return np.vstack([result, tangent]) if return_tangent else result

import numpy as np
def close_optimal_transporter(
    levels: "np.ndarray",
    kcat: "np.ndarray",
    km: "np.ndarray",
    kcat_t: float,
    km_t: float,
    demand_fn: "Callable[..., np.ndarray]",
    cost_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Reference implementation (the transporter's cost per unit flux closes the costs)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    def _positive_vector(value, name, size=None):
        try:
            array = np.array(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric array") from None
        if array.ndim != 1 or array.size == 0 or (size is not None and array.size != size):
            raise ValueError(f"{name} must be a one-dimensional array of the pathway length")
        if not (np.all(np.isfinite(array)) and np.all(array > 0.0)):
            raise ValueError(f"{name} must hold finite positive numbers")
        return array

    rho = _positive_vector(levels, "levels")
    rate = _positive_vector(kcat, "kcat", rho.size)
    affinity = _positive_vector(km, "km", rho.size)
    if not (_is_number(kcat_t) and kcat_t > 0.0 and _is_number(km_t) and km_t > 0.0):
        raise ValueError("kcat_t and km_t must be finite positive numbers")
    if not (_is_function(demand_fn) and _is_function(cost_fn)):
        raise ValueError("demand_fn and cost_fn must be callable")
    demand = _positive_vector(demand_fn(rho.copy(), rate.copy(), affinity.copy()), "demand", rho.size)
    cost = _positive_vector(cost_fn(rho.copy(), rate.copy(), affinity.copy()), "cost", rho.size)
    # Metabolite 0 is worth exactly the transporter protein that one more unit
    # of supply flux needs: cost[0] = (1 + km_t / nu) / kcat_t.
    excess = float(kcat_t) * cost[0] - 1.0
    if not excess > 0.0:
        raise ValueError("these levels are not optimal at any finite nutrient level")
    nu = float(km_t) / excess
    supply = 1.0 + rho.sum()                  # transporter flux per unit growth rate
    transporter = supply * cost[0]            # transporter amount per unit growth rate
    lam = 1.0 / (demand.sum() + transporter)
    return np.array([nu, lam * transporter, lam])

import numpy as np
def close_regulated_transporter(
    levels: "np.ndarray",
    kcat: "np.ndarray",
    km: "np.ndarray",
    kcat_t: float,
    km_t: float,
    basal_fraction: float,
    max_fold: float,
    demand_fn: "Callable[..., np.ndarray]",
) -> "np.ndarray":
    """Reference implementation (closed-form solution of budget and supply balance)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    def _positive_vector(value, name, size=None):
        try:
            array = np.array(value, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric array") from None
        if array.ndim != 1 or array.size == 0 or (size is not None and array.size != size):
            raise ValueError(f"{name} must be a one-dimensional array of the pathway length")
        if not (np.all(np.isfinite(array)) and np.all(array > 0.0)):
            raise ValueError(f"{name} must hold finite positive numbers")
        return array

    rho = _positive_vector(levels, "levels")
    rate = _positive_vector(kcat, "kcat", rho.size)
    affinity = _positive_vector(km, "km", rho.size)
    if not (_is_number(kcat_t) and kcat_t > 0.0 and _is_number(km_t) and km_t > 0.0):
        raise ValueError("kcat_t and km_t must be finite positive numbers")
    if not (_is_number(basal_fraction) and 0.0 < basal_fraction < 1.0):
        raise ValueError("basal_fraction must lie in (0, 1)")
    if not (_is_number(max_fold) and max_fold >= 1.0 and basal_fraction * max_fold < 1.0):
        raise ValueError("max_fold must be at least 1 with basal_fraction * max_fold < 1")
    if not _is_function(demand_fn):
        raise ValueError("demand_fn must be callable")
    demand = _positive_vector(demand_fn(rho.copy(), rate.copy(), affinity.copy()), "demand", rho.size)
    # Budget: phi_t = 1 - lam * D. Supply: j_t = lam * (1 + S). With
    # u = max_fold * nu / km_t the regulated uptake is kcat_t * basal * u / (1 + u)
    # and phi_t = basal * (u + max_fold) / (1 + u), so the balance is linear in u.
    ratio = (1.0 + rho.sum()) / demand.sum()
    numerator = ratio * (1.0 - basal_fraction * max_fold)
    denominator = float(kcat_t) * basal_fraction - ratio * (1.0 - basal_fraction)
    if not denominator > 0.0:
        raise ValueError("the regulated transporter cannot supply these levels")
    u = numerator / denominator
    nu = u * float(km_t) / float(max_fold)
    phi_t = basal_fraction * (u + max_fold) / (1.0 + u)
    lam = (1.0 - phi_t) / demand.sum()
    return np.array([nu, phi_t, lam])

import numpy as np
def solve_family_member(
    target: float,
    component: int,
    kcat: "np.ndarray",
    km: "np.ndarray",
    propagate_fn: "Callable[..., np.ndarray]",
    close_fn: "Callable[[np.ndarray], np.ndarray]",
    bracket: tuple = (1e-12, 1.0),
    tolerance: float = 1e-13,
) -> "np.ndarray":
    """Reference implementation (bisection in the logarithm of the last level)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_function(value):
        return callable(value)

    if isinstance(component, bool) or component not in (0, 2):
        raise ValueError("component must be 0 or 2")
    if not (_is_number(target) and target > 0.0):
        raise ValueError("target must be a finite positive number")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    try:
        low, high = bracket
    except (TypeError, ValueError):
        raise ValueError("bracket must hold two numbers") from None
    if not (_is_number(low) and _is_number(high) and 0.0 < low < high):
        raise ValueError("bracket must satisfy 0 < x_low < x_high")
    if not (_is_function(propagate_fn) and _is_function(close_fn)):
        raise ValueError("propagate_fn and close_fn must be callable")

    def _state(log_x):
        # Returns (levels, value) inside the family, (None, None) beyond it.
        try:
            levels = propagate_fn(float(np.exp(log_x)), kcat, km)
            closed = close_fn(np.asarray(levels, dtype=float).copy())
        except ValueError:
            return None, None
        closed = np.asarray(closed, dtype=float)
        if closed.shape != (3,) or not np.all(np.isfinite(closed)):
            raise ValueError("close_fn must return three finite numbers")
        return np.asarray(levels, dtype=float), float(closed[component])

    lo, hi = float(np.log(low)), float(np.log(high))
    _, value_lo = _state(lo)
    if value_lo is None or not value_lo < target:
        raise ValueError("the lower bracket end must lie in the family below the target")
    _, value_hi = _state(hi)
    if value_hi is not None and value_hi < target:
        raise ValueError("the upper bracket end must lie above the target")
    hi_inside = value_hi is not None
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        _, value = _state(mid)
        if value is None or value >= target:
            hi, hi_inside = mid, value is not None
        else:
            lo = mid
    if not hi_inside:
        raise ValueError("the target is not reached inside the family")
    levels, _ = _state(0.5 * (lo + hi))
    if levels is None:
        raise ValueError("the selected member lies beyond the family")
    return levels

import numpy as np
def summarize_growth_response(growth_rates: "np.ndarray", amounts: "np.ndarray", growth_hi: float) -> "np.ndarray":
    """Reference implementation (closed-form least-squares lines)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    try:
        rates = np.array(growth_rates, dtype=float)
        table = np.array(amounts, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("growth_rates and amounts must be numeric arrays") from None
    if rates.ndim != 1 or rates.size < 2:
        raise ValueError("growth_rates must be a one-dimensional array of at least two states")
    if not (np.all(np.isfinite(rates)) and np.all(rates > 0.0)):
        raise ValueError("growth_rates must hold finite positive numbers")
    if table.ndim != 2 or table.shape[0] != rates.size or table.shape[1] == 0:
        raise ValueError("amounts must have shape (m, n) with n >= 1")
    if not np.all(np.isfinite(table)):
        raise ValueError("amounts must be finite")
    if not (_is_number(growth_hi) and growth_hi > 0.0):
        raise ValueError("growth_hi must be a finite positive number")
    if np.unique(rates).size < 2:
        raise ValueError("growth_rates must contain at least two distinct values")
    centred = rates - rates.mean()
    slope = centred @ (table - table.mean(axis=0)) / float(np.sum(centred * centred))
    intercept = table.mean(axis=0) - slope * rates.mean()
    at_hi = intercept + slope * float(growth_hi)
    if not np.all(at_hi > 0.0):
        raise ValueError("every fitted amount at growth_hi must be positive")
    response = intercept / at_hi
    weights = at_hi / at_hi.sum()
    mean = float(np.sum(weights * response))
    rms = float(np.sqrt(np.sum(weights * (response - mean) ** 2)))
    return np.concatenate([response, [mean, rms]])

import numpy as np
def estimate_response_spread(
    kcat: tuple = (23.0, 9.3, 36.0, 5.9, 16.8, 12.6, 44.0, 7.3, 28.5, 10.7, 19.5, 8.1),
    km: tuple = (2.4e-4, 6.1e-4, 1.3e-4, 9.0e-4, 3.3e-4, 1.9e-3,
                 7.5e-5, 4.4e-4, 2.8e-4, 1.2e-3, 5.2e-4, 8.3e-4),
    kcat_t: float = 22.5,
    km_t: float = 1.2e-3,
    nutrient_hi: float = 0.6,
    max_fold: float = 3.0,
    growth_fractions: tuple = (1.0, 0.85, 0.7, 0.55, 0.4),
    regulated: bool = True,
) -> float:
    """Reference orchestrator using only the reference function chain."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _is_flag(value):
        return isinstance(value, (bool, np.bool_))

    if not _is_flag(regulated):
        raise ValueError("regulated must be a boolean")
    for value in (kcat_t, km_t, nutrient_hi):
        if not (_is_number(value) and value > 0.0):
            raise ValueError("kcat_t, km_t and nutrient_hi must be finite positive numbers")
    if not (_is_number(max_fold) and max_fold >= 1.0):
        raise ValueError("max_fold must be a finite number of at least 1")
    try:
        fractions = np.array(growth_fractions, dtype=float)
        rate = np.array(kcat, dtype=float)
        affinity = np.array(km, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("kcat, km and growth_fractions must be numeric sequences") from None
    if (rate.ndim != 1 or rate.size == 0 or rate.shape != affinity.shape
            or not np.all(np.isfinite(rate) & (rate > 0.0))
            or not np.all(np.isfinite(affinity) & (affinity > 0.0))):
        raise ValueError("kcat and km must be equal-length sequences of finite positive numbers")
    if (fractions.ndim != 1 or not np.all(np.isfinite(fractions))
            or not np.all((fractions > 0.0) & (fractions <= 1.0))
            or np.unique(fractions).size < 2):
        raise ValueError("growth_fractions must hold at least two distinct values in (0, 1]")

    def _demand_fn(levels, kc, k):
        return compute_enzyme_demand(levels, kc, k)

    def _cost_fn(levels, kc, k):
        return compute_metabolite_costs(levels, kc, k)

    def _propagate_fn(last_level, kc, k):
        return propagate_optimal_levels(last_level, kc, k)

    def _optimal_fn(levels):
        return close_optimal_transporter(levels, rate, affinity, kcat_t, km_t,
                                                 _demand_fn, _cost_fn)

    rich = solve_family_member(float(nutrient_hi), 0, rate, affinity,
                                       _propagate_fn, _optimal_fn)
    rich_branch = propagate_optimal_levels(
        float(rich[-1]), rate, affinity, "last", True
    )
    if (rich_branch.shape != (2, rate.size)
            or not np.allclose(rich_branch[0], rich, rtol=1.0e-12, atol=0.0)
            or not np.all(np.isfinite(rich_branch[1]))):
        raise ValueError("rich state is inconsistent with the stationary family")
    _, transporter_hi, growth_hi = _optimal_fn(rich)
    basal = transporter_hi / (1.0 + (max_fold - 1.0) / (1.0 + max_fold * nutrient_hi / km_t))

    def _regulated_fn(levels):
        return close_regulated_transporter(levels, rate, affinity, kcat_t, km_t,
                                                   basal, max_fold, _demand_fn)

    close_fn = _regulated_fn if regulated else _optimal_fn
    growth, amounts = [], []
    for fraction in fractions:
        levels = solve_family_member(float(fraction * growth_hi), 2, rate, affinity,
                                             _propagate_fn, close_fn)
        lam = float(close_fn(levels)[2])
        growth.append(lam)
        amounts.append(lam * compute_enzyme_demand(levels, rate, affinity))
    summary = summarize_growth_response(np.array(growth), np.array(amounts), growth_hi)
    return float(summary[-1])
SCICODE_GOLD_EOF

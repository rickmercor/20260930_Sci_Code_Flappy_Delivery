"""
01_simulate_reef_cover

Population dynamics on a coral reef are the balance of two opposed processes. Between disturbances the cover of hard coral regrows towards the carrying capacity of the substrate, and the regrowth is fastest at intermediate cover, since a nearly bare reef has few colonies to propagate and a nearly saturated one has no space left to colonise. Against that, three kinds of disturbance remove cover: cyclones, which break and scour colonies, thermal stress, which bleaches them, and a residual class covering predation and disease. The first of these acts over more than one year, because a reef stripped by a cyclone in one year carries rubble and weakened colonies into the next, so the mortality in a given year responds to the cyclone intensity of that year and of the two before it.

The state variable is the mean hard coral cover at a site, a proportion between a floor and the carrying capacity. Writing $C_{s,t}$ for the cover at site $s$ in year $t$, $g$ for the intrinsic growth rate, $Kcap$ for the carrying capacity and $c_{s,t}$, $b_{s,t}$, $o_{s,t}$ for the cyclone, bleaching and other intensities, each in the unit interval, the update is logistic regrowth followed by proportional mortality,

$$C_{s,t} = C_{s,t-1} + g C_{s,t-1} (1 - C_{s,t-1} / Kcap) - C_{s,t-1} \min(1, p_{s,t}),$$

with the disturbance pressure

$$p_{s,t} = w_c (c_{s,t} + r_1 c_{s,t-1} + r_2 c_{s,t-2}) + w_b b_{s,t} + w_o o_{s,t},$$

the result clipped below at the cover floor and above at the carrying capacity. The pressure is capped at unity so that mortality cannot exceed the standing cover. Lagged intensities are taken as zero before the record starts. This is a cyclone-dominated scenario: the cyclone weight exceeds the other two, and the cyclone term alone carries lags.

The disturbance fields themselves are laid down by fixed integer maps, so that the whole record is reproducible in exact arithmetic with no random draw anywhere. With $s$ and $t$ counted from zero,

$$c_{s,t} = \max[0, ((7 s + 11 t + 3 s t) \operatorname{mod} 23) - 17] / 6,$$

$$b_{s,t} = \max[0, ((5 s + 13 t + s^2) \operatorname{mod} 19) - 14] / 5,$$

$$o_{s,t} = \max[0, ((3 s + 17 t + t^2) \operatorname{mod} 29) - 22] / 7.$$

Each map leaves the intensity at zero in most site-years and raises it to a value in the unit interval in the remainder, which reproduces the sparse, spiky character of real disturbance records rather than a smoothly varying field. The initial cover varies across sites as $0.25 + 0.01 (s \operatorname{mod} 11)$, so that sites do not begin in a common state.

Returns
-------
dict holding four float64 arrays of shape (n_sites, n_years) under the keys cover, cyclone, bleaching and other, and the native floats mean_cover, min_cover and final_mean_cover.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_reef_cover(
    n_sites: int,
    n_years: int,
    growth_rate: float,
    carrying_capacity: float,
    disturbance_weights: dict,
    cover_floor: float,
) -> dict:
    """Advance mean hard coral cover at every site under logistic regrowth and lagged disturbance mortality.
 
    Parameters
    ----------
    n_sites : int
        Number of monitoring sites.
    n_years : int
        Number of years in the record.
    growth_rate : float
        Intrinsic regrowth rate.
    carrying_capacity : float
        Upper bound on cover.
    disturbance_weights : dict
        Disturbance weights under the keys cyclone (w_c), cyclone_lag1 (r_1), cyclone_lag2 (r_2), bleaching (w_b) and other (w_o). The lag entries r_1 and r_2 scale the lagged cyclone intensities inside the cyclone term, so the pressure is w_c (c_t + r_1 c_{t-1} + r_2 c_{t-2}) + w_b b_t + w_o o_t.
    cover_floor : float
        Lower bound on cover.
 
    Returns
    -------
    dict
        Under the keys cover, cyclone, bleaching, other, mean_cover, min_cover and final_mean_cover.
 
    Raises
    ------
    ValueError
        When n_sites is not an integer of one or more, when n_years is not an integer of three or more, when growth_rate is not finite and above zero, when carrying_capacity is not finite, above zero and at most one, when one of the five weight keys is absent, when a weight fails to be finite and not below zero, or when cover_floor is not finite, not below zero and below the carrying capacity.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
 
WEIGHT_KEYS = ("cyclone", "cyclone_lag1", "cyclone_lag2", "bleaching", "other")
 
 
def _disturbance_fields(n_sites, n_years):
    """The three intensity fields, laid down by fixed integer maps in exact arithmetic."""
    site = np.arange(n_sites, dtype=np.int64)[:, None]
    year = np.arange(n_years, dtype=np.int64)[None, :]
    cyclone = np.maximum(0, ((7 * site + 11 * year + 3 * site * year) % 23) - 17) / 6.0
    bleaching = np.maximum(0, ((5 * site + 13 * year + site * site) % 19) - 14) / 5.0
    other = np.maximum(0, ((3 * site + 17 * year + year * year) % 29) - 22) / 7.0
    return cyclone, bleaching, other
 
 
def _oracle_simulate_reef_cover(
    n_sites: int,
    n_years: int,
    growth_rate: float,
    carrying_capacity: float,
    disturbance_weights: dict,
    cover_floor: float,
) -> dict:
    """Reference implementation."""
    WEIGHT_KEYS = ("cyclone", "cyclone_lag1", "cyclone_lag2", "bleaching", "other")
    if not isinstance(n_sites, (int, np.integer)) or int(n_sites) < 1:
        raise ValueError("n_sites wants an integer of one or more")
    if not isinstance(n_years, (int, np.integer)) or int(n_years) < 3:
        raise ValueError("n_years wants an integer of three or more, so that two lags exist")
    growth = float(growth_rate)
    capacity = float(carrying_capacity)
    floor = float(cover_floor)
    if not np.isfinite(growth) or growth <= 0.0:
        raise ValueError("growth_rate wants a finite value above zero")
    if not np.isfinite(capacity) or capacity <= 0.0 or capacity > 1.0:
        raise ValueError("carrying_capacity wants a finite value above zero and at most one")
    absent = [key for key in WEIGHT_KEYS if key not in disturbance_weights]
    if absent:
        raise ValueError("disturbance_weights lacks the entry %s" % absent[0])
    weights = np.array([float(disturbance_weights[key]) for key in WEIGHT_KEYS])
    if not np.isfinite(weights).all() or weights.min() < 0.0:
        raise ValueError("each disturbance weight wants a finite value not below zero")
    if not np.isfinite(floor) or floor < 0.0 or floor >= capacity:
        raise ValueError("cover_floor wants a finite value not below zero and below the carrying capacity")
 
    sites, years = int(n_sites), int(n_years)
    cyclone, bleaching, other = _disturbance_fields(sites, years)
    w_c, w_1, w_2, w_b, w_o = weights
    cover = np.zeros((sites, years))
    cover[:, 0] = 0.25 + 0.01 * (np.arange(sites) % 11)
    for year in range(1, years):
        previous = cover[:, year - 1]
        regrowth = growth * previous * (1.0 - previous / capacity)
        lag_one = cyclone[:, year - 1] if year >= 1 else np.zeros(sites)
        lag_two = cyclone[:, year - 2] if year >= 2 else np.zeros(sites)
        pressure = (w_c * (cyclone[:, year] + w_1 * lag_one + w_2 * lag_two)
                    + w_b * bleaching[:, year] + w_o * other[:, year])
        cover[:, year] = np.clip(previous + regrowth - previous * np.minimum(1.0, pressure),
                                 floor, capacity)
    return {
        "cover": cover,
        "cyclone": cyclone,
        "bleaching": bleaching,
        "other": other,
        "mean_cover": float(cover.mean()),
        "min_cover": float(cover.min()),
        "final_mean_cover": float(cover[:, -1].mean()),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-data cases."""
    setup = """import numpy as np
WEIGHTS = {"cyclone": 0.55, "cyclone_lag1": 0.45, "cyclone_lag2": 0.20,
           "bleaching": 0.30, "other": 0.18}
QUIET = {key: 0.0 for key in WEIGHTS}
def digest(out):
    cover = np.asarray(out["cover"], dtype=float)
    return np.concatenate((
        np.asarray([cover.shape[0], cover.shape[1], out["mean_cover"],
                    out["min_cover"], out["final_mean_cover"]], dtype=float),
        cover.ravel(), np.asarray(out["cyclone"], dtype=float).ravel(),
        np.asarray(out["bleaching"], dtype=float).ravel(),
        np.asarray(out["other"], dtype=float).ravel(),
    ))
def invalid_site_count(fn):
    try:
        fn(0, 10, 0.8, 0.80, WEIGHTS, 0.02)
    except ValueError:
        return np.asarray([1.0])
    except Exception:
        return np.asarray([2.0])
    return np.asarray([0.0])
"""
    return [
        {"setup": setup,
         "call": "digest(simulate_reef_cover(20, 22, 0.8, 0.80, WEIGHTS, 0.02))",
         "gold_call": "digest(_oracle_simulate_reef_cover(20, 22, 0.8, 0.80, WEIGHTS, 0.02))",
         "tol": 1e-10},
        {"setup": setup,
         "call": "digest(simulate_reef_cover(1, 3, 0.8, 0.80, WEIGHTS, 0.02))",
         "gold_call": "digest(_oracle_simulate_reef_cover(1, 3, 0.8, 0.80, WEIGHTS, 0.02))",
         "tol": 1e-10},
        {"setup": setup,
         "call": "digest(simulate_reef_cover(3, 30, 0.8, 0.80, QUIET, 0.02))",
         "gold_call": "digest(_oracle_simulate_reef_cover(3, 30, 0.8, 0.80, QUIET, 0.02))",
         "tol": 1e-10},
        {"setup": setup,
         "call": "invalid_site_count(simulate_reef_cover)",
         "gold_call": "invalid_site_count(_oracle_simulate_reef_cover)"},
    ]

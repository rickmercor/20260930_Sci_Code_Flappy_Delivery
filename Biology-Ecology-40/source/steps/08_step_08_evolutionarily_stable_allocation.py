"""
Given the Dirichlet concentration alpha, the number of cycles, the burn-in and a seed, draw the supply sequence as an (n_cycles, 2) array of gamma variates of shape alpha and unit scale in a single call to numpy.random.default_rng(seed).gamma, floor every variate at 1e-300 and divide each row by its sum; locate the singular strategy phi* with the preceding step using its default bracket and tolerance, and evaluate the invasion growth rates against the symmetric resident pair at phi* of mutants with traits phi* exp(x) for 25 values of x evenly spaced from -4 to ln(1 / phi*), together with a pure specialist, phi = 0. Return phi*, the largest invasion growth rate on that grid minus the resident's own rate (the largest over both preference orders), the trait at which it occurs, the curvature and the convergence slope at phi*, the mean share of resident biomass held by the resident preferring resource 0, and the replay error of the resident pair. Return also the two-resource growth rate of a resident at phi* and its lags after the exhaustion of its preferred and of its other resource; the selection gradient at phi* exp(-0.5) and at the smaller of phi* exp(0.5) and exp(-0.05); the mean durations of the primary and secondary temporal niches over the retained cycles; and, for one cycle of the resident pair at phi* started from the total biomass 1 / (D - 1), with D = 100, split between the two residents in the mean resident proportion just returned, under a supply of 0.5 of each resource, the time of the first exhaustion, found directly by the root finder of step 3, and the duration of the secondary niche from the cycle integration of step 4.

This step assembles the whole calculation of the evolutionarily stable secondary allocation of a microbial community in a boom-and-bust environment whose resource supply ratio fluctuates from cycle to cycle. Each 24-hour cycle opens with one unit of resource, split between two equivalent substitutable resources in proportions drawn independently from the symmetric Dirichlet distribution Dir(alpha, alpha), and closes with a 100-fold dilution. Small alpha means strongly imbalanced supply: most cycles deliver almost all of one resource and a trace of the other. The concentration alpha is the only environmental parameter varied, and it sets which strategy evolution selects: specialists under balanced supply, hierarchical utilisers under moderate fluctuations and, as the fluctuations grow, strategies approaching co-utilisation, because the secondary temporal niche, the stretch of each cycle after the first resource has run out, lengthens relative to the primary one.

The chain is as follows. Step 1 gives the metabolic proteome allocation of a strain with secondary allocation phi, its uptake rates and its growth rate on whatever resources are present. Step 2 gives the reallocation lag that follows the exhaustion of a consumed resource, derived from autocatalytic enzyme synthesis: long when the primary resource runs out, short but not zero when a secondary one does. Step 3 finds the earliest resource exhaustion under exponential growth by a bracketed Newton iteration. Step 4 integrates one cycle from event to event. Step 5 carries the symmetric resident pair, one strain preferring each resource, through the supply sequence and replays rare mutants against the resulting exhaustion schedules to obtain their long-run invasion growth rates. Step 6 forms the selection gradient and the curvature of the invasion growth rate in ln phi with common random numbers. Step 7 locates the singular strategy, where the gradient vanishes, by the Illinois method.

The orchestrator also reports the quantities that show why this value is selected. Steps 1 and 2 give the growth rate of a resident at phi* while both resources are present and its two lags, and step 6, evaluated half a unit of ln phi either side of phi*, shows the gradient pointing towards phi* from both sides. The mechanism is the length of the secondary temporal niche: step 5 returns its mean duration under the fluctuating supply, and steps 3 and 4, applied to one cycle of the resident pair at its stationary total biomass, split in the mean resident proportion, under a balanced 1:1 supply, give the lengths of the two niches when the supply ratio does not fluctuate. Under balanced supply the two resources run out within minutes of each other, so the secondary niche all but vanishes and nothing repays a secondary allocation.

The orchestrator draws the supply sequence, locates the singular strategy and then confirms that it is evolutionarily stable in the global sense: no mutant on a wide grid of traits running from four natural-log units (a factor of about fifty) below phi* up to phi = 1 has an invasion growth rate above that of the resident's own trait. The draws are made as normalised pairs of independent gamma variates of shape alpha, which is the Dirichlet distribution, and both amounts are kept strictly positive so that both resources are present at the start of every cycle, however small the drawn amount. A gamma variate of shape alpha falls below the smallest double-precision magnitude with probability of about 10 to the power -300 alpha; at alpha = 0.03 that is about one in a billion, but below alpha = 0.01 pairs in which both variates underflow become common and would be turned into balanced supplies, so the direct draw is used only for alpha of at least 0.01.

Returns
-------
dict holding the floats phi_star, invasion_excess, excess_trait, curvature, convergence_slope, resident_share, replay_error, growth_rate, lag_primary, lag_secondary, gradient_below, gradient_above, primary_niche, secondary_niche, balanced_primary_niche and balanced_secondary_niche, all times in hours.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evolutionarily_stable_allocation(alpha: float, n_cycles: int, burn_in: int, seed: int) -> dict:
    """Evolutionarily stable secondary allocation under Dirichlet-fluctuating supply ratios.

    Parameters
    ----------
    alpha : float
        Dirichlet concentration of the supply split.
    n_cycles : int
        Number of serial-dilution cycles; an integer at least 100.
    burn_in : int
        Cycles discarded before averaging; an integer satisfying 0 <= burn_in < n_cycles / 2.
    seed : int
        Seed of the supply draws.

    Returns
    -------
    dict
        Under the keys phi_star, invasion_excess, excess_trait, curvature, convergence_slope,
        resident_share, replay_error, growth_rate, lag_primary, lag_secondary, gradient_below,
        gradient_above, primary_niche, secondary_niche, balanced_primary_niche and
        balanced_secondary_niche.

    Raises
    ------
    ValueError
        When alpha is not finite or is below 0.01, the cycle count, burn-in or seed is not of
        integer type, the cycle count or burn-in is out of range, or the seed is negative.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _dirichlet_supply(alpha, n_cycles, seed):
    """Per-cycle supplies of two resources, one unit in total, split by Dir(alpha, alpha)."""
    rng = np.random.default_rng(seed)
    draws = rng.gamma(alpha, 1.0, size=(n_cycles, 2))
    draws = np.maximum(draws, 1e-300)
    return draws / draws.sum(axis=1, keepdims=True)


def _oracle_evolutionarily_stable_allocation(alpha: float, n_cycles: int, burn_in: int, seed: int) -> dict:
    """Reference implementation."""
    alpha = float(alpha)
    if not math.isfinite(alpha) or alpha < 0.01:
        raise ValueError("alpha must be finite and at least 0.01")
    for name, v in (("n_cycles", n_cycles), ("burn_in", burn_in), ("seed", seed)):
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            raise ValueError(name + " must be an integer")
    if n_cycles < 100 or burn_in < 0 or 2 * burn_in >= n_cycles or seed < 0:
        raise ValueError("need n_cycles >= 100, 0 <= burn_in < n_cycles / 2 and seed >= 0")

    supply = _dirichlet_supply(alpha, int(n_cycles), int(seed))
    star = _oracle_singular_strategy(supply, int(burn_in))  # noqa: F821
    phi_star = star["phi_star"]
    grid = np.concatenate([[0.0], phi_star * np.exp(np.linspace(-4.0, math.log(1.0 / phi_star), 25))])
    grid = np.minimum(grid, 1.0)
    check = _oracle_invasion_growth_rates(phi_star, np.concatenate([[phi_star], grid]), supply, int(burn_in))  # noqa: F821
    rates = check["invasion_rates"]
    excess = rates[:, 1:] - rates[:, :1]
    j, q = np.unravel_index(int(np.argmax(excess)), excess.shape)
    pref = np.array([[0, 1], [1, 0]])
    both = np.array([True, True])
    growth = _oracle_proteome_allocation(phi_star, pref[0], both)["growth_rate"]  # noqa: F821
    lag_primary = _oracle_reallocation_lag(phi_star, pref[0], 0, np.array([False, True]))["lag"]  # noqa: F821
    lag_secondary = _oracle_reallocation_lag(phi_star, pref[0], 1, np.array([True, False]))["lag"]  # noqa: F821
    below = _oracle_selection_gradient(phi_star * math.exp(-0.5), supply, int(burn_in))["gradient"]  # noqa: F821
    above = _oracle_selection_gradient(min(phi_star * math.exp(0.5), math.exp(-0.05)), supply, int(burn_in))["gradient"]  # noqa: F821

    balanced = np.array([0.5, 0.5])
    share = check["mean_share"]
    start = np.array([share, 1.0 - share]) / (100.0 - 1.0)
    tables = [_oracle_proteome_allocation(phi_star, pref[i], both) for i in range(2)]  # noqa: F821
    first = _oracle_first_exhaustion(balanced, start, np.array([t["growth_rate"] for t in tables]),  # noqa: F821
                                     np.array([t["uptake"] for t in tables]), 24.0)["time"]
    cycle = _oracle_grow_one_cycle(np.array([phi_star, phi_star]), pref, start, balanced)  # noqa: F821
    ends = np.minimum(cycle["exhaustion_times"], 24.0)
    return {"phi_star": float(phi_star),
            "invasion_excess": float(excess[j, q]),
            "excess_trait": float(grid[q]),
            "curvature": star["curvature"],
            "convergence_slope": star["convergence_slope"],
            "resident_share": check["mean_share"],
            "replay_error": check["replay_error"],
            "growth_rate": float(growth),
            "lag_primary": float(lag_primary),
            "lag_secondary": float(lag_secondary),
            "gradient_below": float(below),
            "gradient_above": float(above),
            "primary_niche": check["primary_niche"],
            "secondary_niche": check["secondary_niche"],
            "balanced_primary_niche": float(first),
            "balanced_secondary_niche": float(ends.max() - ends.min())}

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the declared test cases for this step."""
    SETUP = """import math
import numpy as np

def isolated(fn, *args, **kwargs):
    return fn(*(x.copy() if isinstance(x, np.ndarray) else x for x in args),
              **{k: (x.copy() if isinstance(x, np.ndarray) else x) for k, x in kwargs.items()})

def num(v, k=6):
    return float(round(float(v), k))

def pack(d):
    return (num(d["phi_star"]), num(d["invasion_excess"], 7) + 0.0, num(d["excess_trait"]), num(d["curvature"]),
            num(d["convergence_slope"]), num(d["resident_share"]), num(d["replay_error"], 9) + 0.0,
            num(d["growth_rate"]), num(d["lag_primary"]), num(d["lag_secondary"]), num(d["gradient_below"]),
            num(d["gradient_above"]), num(d["primary_niche"]), num(d["secondary_niche"]),
            num(d["balanced_primary_niche"]), num(d["balanced_secondary_niche"]))
"""

    return [
        {
            "setup": SETUP + """
def digest(fn):
    return pack(isolated(fn, 0.03, 1500, 100, 7))
""",
            "call": "digest(evolutionarily_stable_allocation)",
            "gold_call": "digest(_oracle_evolutionarily_stable_allocation)",
        },
        {
            "setup": SETUP + """
def trend(fn):
    a = isolated(fn, 0.03, 1000, 100, 3)["phi_star"]
    b = isolated(fn, 0.1, 1000, 100, 3)["phi_star"]
    return (num(b), bool(b < a))
""",
            "call": "trend(evolutionarily_stable_allocation)",
            "gold_call": "trend(_oracle_evolutionarily_stable_allocation)",
        },
        {
            "setup": SETUP + """
def boundary(fn):
    return pack(isolated(fn, 0.03, 100, 49, 12))
""",
            "call": "boundary(evolutionarily_stable_allocation)",
            "gold_call": "boundary(_oracle_evolutionarily_stable_allocation)",
        },
        {
            "setup": SETUP + """
def rejects(fn):
    bad = [
        (0.0, 200, 10, 1),
        (0.005, 200, 10, 1),
        (float("inf"), 200, 10, 1),
        (0.03, 50, 10, 1),
        (0.03, 200, 100, 1),
        (0.03, 200, -1, 1),
        (0.03, 200, 10, -3),
        (0.03, 200.0, 10, 1),
    ]
    count = 0
    for args in bad:
        try:
            isolated(fn, *args)
        except ValueError:
            count += 1
    return count
""",
            "call": "rejects(evolutionarily_stable_allocation)",
            "gold_call": "rejects(_oracle_evolutionarily_stable_allocation)",
        },
    ]

"""
Given the resident trait phi, an array of mutant trait values and a sequence of per-cycle supplies of the two resources, integrate the symmetric resident pair through the sequence (resident 0 prefers resource 0, resident 1 prefers resource 1; after each cycle every biomass is divided by the dilution factor). In every cycle, replay each mutant, for both preference orders, against the residents' exhaustion schedule, and return the long-run invasion growth rates averaged over the cycles after the burn-in. Also return the same average for the residents themselves, obtained by the replay, the largest absolute difference, over all retained cycles and both residents, between a resident's replayed and integrated log fold changes, and the mean durations over the retained cycles of the primary temporal niche (from the start of the cycle to the first exhaustion, capped at the cycle length) and of the secondary temporal niche (from the first exhaustion to the second, both capped at the cycle length).

Evolutionary stability is decided by invasion. A mutant that arises in a resident community is rare, so it does not change when the resources run out; it only experiences the schedule of exhaustions that the residents impose, cycle after cycle, and either grows faster than the 100-fold dilution removes it or does not. Its long-run invasion growth rate is the average over cycles of the logarithm of its within-cycle fold change minus ln D, with D = 100 the dilution factor, taken along the stationary sequence of resident states driven by the random supply. A strategy that no rare mutant can invade, whose invasion growth rate is positive for no alternative trait value, is evolutionarily stable.

The resident community that evolution produces under symmetric fluctuating supply of two equivalent resources holds two strains with opposite preference orders. A single strain is not a resting point: a strain preferring the other resource, even with the same trait, invades it at once, because in every cycle in which the second resource is the abundant one it grows at the full rate after only a short lag while the resident lags long and then grows at half the rate. The symmetric pair, one strain preferring each resource and both carrying the same trait phi, is therefore the resident against which invasion is assessed, and frequency dependence keeps both members present: whichever is commoner draws down its own primary resource faster and so hands the secondary niche to the other.

A rare mutant's fold change in a cycle follows from the resident schedule alone. With first exhaustion at time t1 and second at t2 (either may fall beyond the end of the cycle), the mutant grows at its two-resource rate until t1; if it was consuming the resource that ran out it then sits out its own reallocation lag, and it grows on the remaining resource alone from the end of that lag until t2, or not at all if the lag outlasts the remaining resource. Because the resident's own growth obeys the same rule, replaying a resident's trait through it must reproduce the resident's integrated fold change, which checks the bookkeeping. The residents are carried from cycle to cycle by the dilution alone, starting from equal biomasses of 1 / (2 (D - 1)), the stationary total when all supplied resource, normalised to one per cycle, is converted to biomass; an initial stretch of cycles is discarded before averaging.

Returns
-------
dict holding the np.ndarray invasion_rates of shape (2, k), row j for mutants preferring resource j first; the np.ndarray resident_rates of shape (2,); the float replay_error; the float mean_share, the mean fraction of resident biomass held by resident 0 at the start of the retained cycles; and the floats primary_niche and secondary_niche, in hours.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def invasion_growth_rates(phi_resident: float, phi_mutants: np.ndarray, supply: np.ndarray, burn_in: int,
                          dilution: float = 100.0, lag_scale: float = 0.3, cycle_length: float = 24.0) -> dict:
    """Long-run invasion growth rates of rare mutants into the symmetric resident pair.

    Parameters
    ----------
    phi_resident : float
        Resident secondary allocation.
    phi_mutants : np.ndarray
        Mutant secondary allocations, shape (k,).
    supply : np.ndarray
        Per-cycle supplies of the two resources, shape (n_cycles, 2).
    burn_in : int
        Cycles discarded before averaging.
    dilution : float
        Dilution factor.
    lag_scale : float
        Autocatalytic timescale tau0, hours.
    cycle_length : float
        Cycle duration, hours.

    Returns
    -------
    dict
        Under the keys invasion_rates, resident_rates, replay_error, mean_share, primary_niche
        and secondary_niche.

    Raises
    ------
    ValueError
        When a trait lies outside its range, the supply is not a positive finite (n_cycles, 2)
        array, the burn-in leaves no cycle, the dilution factor does not exceed one, or a time
        parameter is not positive and finite.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _strain_table(phi, lag_scale):
    """Rates and lags of a strain of trait phi for both preference orders.

    Entry j holds the two-resource growth rate, the single-resource rate on each resource and
    the lag after the exhaustion of each resource, for the strain preferring resource j.
    """
    _PREFS = (np.array([0, 1]), np.array([1, 0]))
    _BOTH = np.array([True, True])
    _ALONE = (np.array([True, False]), np.array([False, True]))
    table = []
    for pref in _PREFS:
        both = _oracle_proteome_allocation(phi, pref, _BOTH)  # noqa: F821
        single = [_oracle_proteome_allocation(phi, pref, _ALONE[k])["growth_rate"] for k in (0, 1)]  # noqa: F821
        lags = []
        for k in (0, 1):
            if both["uptake"][k] > 0.0:
                lags.append(_oracle_reallocation_lag(phi, pref, k, _ALONE[1 - k], lag_scale)["lag"])  # noqa: F821
            else:
                lags.append(0.0)
        table.append((both["growth_rate"], single, lags))
    return table


def _replay(entry, first, t1, t2, cycle_length):
    """Log fold change of a rare strain under the exhaustion schedule (first, t1, t2)."""
    rate_both, single, lags = entry
    if first < 0 or t1 >= cycle_length:
        return rate_both * cycle_length
    remaining = 1 - first
    start = t1 + lags[first]
    stop = min(t2, cycle_length)
    return rate_both * t1 + (single[remaining] * (stop - start) if stop > start else 0.0)


def _oracle_invasion_growth_rates(phi_resident: float, phi_mutants: np.ndarray, supply: np.ndarray, burn_in: int,
                                  dilution: float = 100.0, lag_scale: float = 0.3, cycle_length: float = 24.0) -> dict:
    """Reference implementation."""
    phi_resident = float(phi_resident)
    if not math.isfinite(phi_resident) or phi_resident <= 0.0 or phi_resident > 1.0:
        raise ValueError("phi_resident must lie in (0, 1]")
    mutants = np.asarray(phi_mutants, dtype=float)
    if mutants.ndim != 1 or mutants.size < 1 or not np.all(np.isfinite(mutants)) \
            or np.any(mutants < 0.0) or np.any(mutants > 1.0):
        raise ValueError("phi_mutants must be a nonempty one-dimensional array in [0, 1]")
    s = np.asarray(supply, dtype=float)
    if s.ndim != 2 or s.shape[1] != 2 or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("supply must be a positive finite array of shape (n_cycles, 2)")
    if isinstance(burn_in, bool) or not isinstance(burn_in, (int, np.integer)) or burn_in < 0 or burn_in >= s.shape[0]:
        raise ValueError("burn_in must be an integer in [0, n_cycles)")
    dilution = float(dilution)
    if not math.isfinite(dilution) or dilution <= 1.0:
        raise ValueError("dilution must exceed one")

    resident_table = _strain_table(phi_resident, lag_scale)
    mutant_tables = [_strain_table(p, lag_scale) for p in mutants]
    phis = np.array([phi_resident, phi_resident])
    prefs = np.array([[0, 1], [1, 0]])
    b = np.full(2, 1.0 / (2.0 * (dilution - 1.0)))
    log_d = math.log(dilution)
    sums = np.zeros((2, mutants.size))
    res_sums = np.zeros(2)
    share = 0.0
    error = 0.0
    niche_one = 0.0
    niche_two = 0.0
    kept = 0
    for c in range(s.shape[0]):
        start = b
        out = _oracle_grow_one_cycle(phis, prefs, start, s[c], lag_scale, cycle_length)  # noqa: F821
        first = out["first_exhausted"]
        times = out["exhaustion_times"]
        t1 = float(times[first]) if first >= 0 else math.inf
        t2 = float(times[1 - first]) if first >= 0 else math.inf
        if c >= burn_in:
            kept += 1
            share += start[0] / start.sum()
            niche_one += min(t1, cycle_length)
            niche_two += min(t2, cycle_length) - min(t1, cycle_length)
            for j in (0, 1):
                g = _replay(resident_table[j], first, t1, t2, cycle_length)
                res_sums[j] += g - log_d
                error = max(error, abs(g - math.log(out["biomass"][j] / start[j])))
                for q, table in enumerate(mutant_tables):
                    sums[j, q] += _replay(table[j], first, t1, t2, cycle_length) - log_d
        b = out["biomass"] / dilution
    return {"invasion_rates": sums / kept,
            "resident_rates": res_sums / kept,
            "replay_error": float(error),
            "mean_share": float(share / kept),
            "primary_niche": float(niche_one / kept),
            "secondary_niche": float(niche_two / kept)}

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

def num(v, k=10):
    return float(round(float(v), k))

def supply_sequence(n, alpha, seed):
    rng = np.random.default_rng(seed)
    g = rng.gamma(alpha, 1.0, size=(n, 2))
    g = np.maximum(g, 1e-300)
    return g / g.sum(axis=1, keepdims=True)

def pack(d):
    return (tuple(num(x) for x in d["invasion_rates"].ravel()) + tuple(num(x) for x in d["resident_rates"])
            + (num(d["replay_error"], 9) + 0.0, num(d["mean_share"]), num(d["primary_niche"]), num(d["secondary_niche"])))
"""

    return [
        {
            "setup": SETUP + """
def digest(fn):
    s = supply_sequence(300, 0.03, 11)
    return pack(isolated(fn, 0.3, np.array([0.1, 0.3, 0.6, 1.0]), s, 50))
""",
            "call": "digest(invasion_growth_rates)",
            "gold_call": "digest(_oracle_invasion_growth_rates)",
        },
        {
            "setup": SETUP + """
def identities(fn):
    s = supply_sequence(200, 0.3, 5)
    a = isolated(fn, 0.15, np.array([0.15, 0.05]), s, 0)
    m = isolated(fn, 0.15, np.array([0.15, 0.05]), s[:, ::-1].copy(), 0)
    same = float(np.max(np.abs(a["invasion_rates"][:, 0] - a["resident_rates"])))
    mirror = float(np.max(np.abs(a["invasion_rates"] - m["invasion_rates"][::-1])))
    return (num(a["replay_error"], 9) + 0.0, num(same, 12) + 0.0, num(mirror, 9) + 0.0,
            num(a["resident_rates"][0], 6), num(a["mean_share"] + m["mean_share"], 9))
""",
            "call": "identities(invasion_growth_rates)",
            "gold_call": "identities(_oracle_invasion_growth_rates)",
        },
        {
            "setup": SETUP + """
def balanced(fn):
    s = np.full((40, 2), 0.5)
    return pack(isolated(fn, 0.05, np.array([0.0, 0.01, 0.05, 0.5]), s, 10))
""",
            "call": "balanced(invasion_growth_rates)",
            "gold_call": "balanced(_oracle_invasion_growth_rates)",
        },
        {
            "setup": SETUP + """
def rejects(fn):
    s = np.full((5, 2), 0.5)
    bad = [
        (0.0, np.array([0.1]), s, 0),
        (1.2, np.array([0.1]), s, 0),
        (0.3, np.array([1.1]), s, 0),
        (0.3, np.array([]), s, 0),
        (0.3, np.array([0.1]), np.full((5, 3), 0.3), 0),
        (0.3, np.array([0.1]), np.array([[0.5, 0.0]] * 5), 0),
        (0.3, np.array([0.1]), s, 5),
        (0.3, np.array([0.1]), s, -1),
    ]
    count = 0
    for args in bad:
        try:
            isolated(fn, *args)
        except ValueError:
            count += 1
    try:
        isolated(fn, 0.3, np.array([0.1]), s, 0, 1.0)
    except ValueError:
        count += 1
    return count
""",
            "call": "rejects(invasion_growth_rates)",
            "gold_call": "rejects(_oracle_invasion_growth_rates)",
        },
    ]

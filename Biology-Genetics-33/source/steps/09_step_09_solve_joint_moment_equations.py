"""
Fit all three working-model parameters jointly from the first three limiting trace moments of the between-family sum-of-squares matrix, selecting the unique admissible root of the resulting system.

Matching as many moments as there are free parameters determines the fit, but the map from parameters to moments is polynomial and is not inverted in closed form, so the estimate is the root of a system rather than a formula. Only one root has all three parameters positive with the genetic spectrum inside the residual one.

Returns
-------
np.ndarray of shape (3,), float: the fitted genetic eigenvalue, its occupied fraction, and the fitted residual eigenvalue.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_joint_moment_equations(design_functionals: np.ndarray,
                                 moments: np.ndarray,
                                 residual_fraction: float) -> np.ndarray:
    """Fit all three working-model parameters from three trace moments at once.

    A candidate parameter triple implies a covariance functional for every word,
    and those together with ``design_functionals`` imply a limiting value for
    each of the first three trace moments of the between-family sum-of-squares
    matrix. The returned triple is the one whose implied moments equal
    ``moments``.

    Among the solutions of that system exactly one has a positive genetic level,
    a positive residual level and a genetic fraction strictly inside
    ``(0, residual_fraction)``; that is the one returned. It is located to a
    relative accuracy of 1e-12 or better in the residual level, which fixes the
    returned triple to at least ten significant figures.

    ``design_functionals`` indexes words first by length and then
    lexicographically, so the word ``(w_1, ..., w_s)`` sits at position
    ``2**s - 2 + sum_t w_t * 2**(s - t)``; letter zero is the family level and
    letter one the individual level.

    Parameters
    ----------
    design_functionals : np.ndarray
        One-dimensional table of normalised traces of design-operator words for
        the between-family increment, tabulated to word length three, so of
        length at least fourteen.
    moments : np.ndarray
        Array of shape ``(3,)`` holding the limiting first, second and third
        normalised trace moments of the between-family sum-of-squares matrix.
    residual_fraction : float
        Known fraction of trait directions carrying the nonzero residual
        eigenvalue, in ``(0, 1]``.

    Returns
    -------
    estimates : np.ndarray
        Array of shape ``(3,)`` holding the fitted genetic eigenvalue, the
        fitted fraction of trait directions carrying it, and the fitted residual
        eigenvalue, in that order.

    Raises
    ------
    ValueError
        If ``design_functionals`` is not a one-dimensional array of at least
        fourteen finite entries, if ``moments`` is not a one-dimensional array
        of exactly three finite entries, if ``residual_fraction`` is not a
        finite number in ``(0, 1]``, if the family-level design functional is
        zero, or if the system has no admissible solution.
    """
    return estimates  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_solve_joint_moment_equations(design_functionals: np.ndarray,
                                         moments: np.ndarray,
                                         residual_fraction: float) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and np.isfinite(value))

    design = np.asarray(design_functionals, dtype=float)
    if design.ndim != 1 or design.size < 14:
        raise ValueError("design_functionals must be a one-dimensional array of at least fourteen entries")
    if not np.all(np.isfinite(design)):
        raise ValueError("design_functionals must contain only finite entries")
    observed = np.asarray(moments, dtype=float)
    if observed.ndim != 1 or observed.size != 3 or not np.all(np.isfinite(observed)):
        raise ValueError("moments must be a one-dimensional array of three finite entries")
    if not _is_number(residual_fraction) or not (0.0 < float(residual_fraction) <= 1.0):
        raise ValueError("residual_fraction must be a finite number in the interval (0, 1]")
    rho_residual = float(residual_fraction)

    family, individual = float(design[0]), float(design[1])
    family_family, family_individual = float(design[2]), float(design[3])
    individual_family, individual_individual = float(design[4]), float(design[5])
    if family == 0.0:
        raise ValueError("the family-level design functional must not vanish")

    block_table = _oracle_compute_kreweras_block_table(
        _oracle_enumerate_noncrossing_pairings(3))

    def _profile_moments(residual_level):
        # The first moment is linear in the genetic profile mean, and the second
        # is linear in the genetic profile's second moment once that mean and the
        # residual level are fixed, so both follow without iteration.
        mean = (observed[0] - individual * rho_residual * residual_level) / family
        residual_mean = rho_residual * residual_level
        known = (family_family * mean ** 2
                 + (family_individual + individual_family) * mean * residual_mean
                 + 2.0 * family * individual * mean * residual_level
                 + individual ** 2 * residual_mean * residual_level
                 + individual_individual * residual_mean ** 2)
        second = (observed[1] - known) / family ** 2
        return mean, second

    def _third_moment_residual(residual_level):
        mean, second = _profile_moments(residual_level)
        if not (mean > 0.0 and second > 0.0):
            return float("nan")
        surrogate = _oracle_build_surrogate_spectral_functionals(
            second / mean, mean ** 2 / second, residual_level, rho_residual, 3)
        return _oracle_evaluate_moment_expansion(design, surrogate, block_table) - observed[2]

    # The genetic profile mean is positive only below this residual level, which
    # bounds the admissible interval from above.
    upper = observed[0] / (individual * rho_residual) if individual * rho_residual > 0.0 else np.inf
    if not np.isfinite(upper) or upper <= 0.0:
        raise ValueError("the moment equations admit no admissible solution")
    low, high = upper * 1e-9, upper * (1.0 - 1e-9)
    f_low, f_high = _third_moment_residual(low), _third_moment_residual(high)

    if not (np.isfinite(f_low) and np.isfinite(f_high) and f_low * f_high < 0.0):
        # Fall back to a deterministic scan for the sign change; the root is
        # unique, so any bracket containing it gives the same answer.
        grid = np.linspace(low, high, 512)
        values = [_third_moment_residual(t) for t in grid]
        low = high = None
        for i in range(len(grid) - 1):
            a, b = values[i], values[i + 1]
            if np.isfinite(a) and np.isfinite(b) and a * b < 0.0:
                low, high, f_low = grid[i], grid[i + 1], a
                break
        if low is None:
            raise ValueError("the moment equations admit no admissible solution")

    for _ in range(200):
        middle = 0.5 * (low + high)
        f_middle = _third_moment_residual(middle)
        if not np.isfinite(f_middle):
            raise ValueError("the moment equations admit no admissible solution")
        if f_middle == 0.0 or (high - low) <= 1e-15 * max(1.0, abs(middle)):
            break
        if f_low * f_middle < 0.0:
            high = middle
        else:
            low, f_low = middle, f_middle

    residual_level = 0.5 * (low + high)
    mean, second = _profile_moments(residual_level)
    if not (mean > 0.0 and second > 0.0):
        raise ValueError("the moment equations admit no admissible solution")
    genetic_level = second / mean
    genetic_fraction = mean ** 2 / second
    if not (0.0 < genetic_fraction < rho_residual and genetic_level > 0.0
            and residual_level > 0.0):
        raise ValueError("the moment equations admit no admissible solution")

    return np.array([genetic_level, genetic_fraction, residual_level], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: the testbed configuration (normal scenario) ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
design_functionals = np.array([2.490277777778, 1.245833333333, 5.806018518519,
                               2.490277777778, 2.490277777778, 1.245833333333,
                               14.917901234568, 5.806018518518, 5.806018518519,
                               2.490277777778, 5.806018518519, 2.490277777778,
                               2.490277777778, 1.245833333333])
moments = np.array([1.818236111111, 9.017816515426, 58.639776199234])
""",
            "call": "sig(solve_joint_moment_equations(design_functionals, moments, 0.8), 1.0)",
            "gold_call": "sig(_oracle_solve_joint_moment_equations(design_functionals, moments, 0.8), 1.0)",
        },
        # --- Valid: a strongly unbalanced design with a different spectrum and a
        #     different known residual fraction ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
design_functionals = np.array([0.612373737374, 0.270833333333, 2.079621977349,
                               0.612373737374, 0.612373737374, 0.270833333333,
                               8.873117075252, 2.079621977349, 2.079621977349,
                               0.612373737374, 2.079621977349, 0.612373737374,
                               0.612373737374, 0.270833333333])
moments = np.array([0.432725694444, 1.403615686195, 6.263766282512])
""",
            "call": "sig(solve_joint_moment_equations(design_functionals, moments, 0.7), 1.0)",
            "gold_call": "sig(_oracle_solve_joint_moment_equations(design_functionals, moments, 0.7), 1.0)",
        },
        # --- Valid: a quarter-turn eigenbasis mismatch, the most adversarial
        #     alignment, on a two-size design ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
design_functionals = np.array([0.990322580645, 0.400000000000, 2.547554630593,
                               0.990322580645, 0.990322580645, 0.400000000000,
                               6.786019267564, 2.547554630593, 2.547554630593,
                               0.990322580645, 2.547554630593, 0.990322580645,
                               0.990322580645, 0.400000000000])
moments = np.array([0.741741935484, 2.174903194589, 8.054489468425])
""",
            "call": "sig(solve_joint_moment_equations(design_functionals, moments, 0.85), 1.0)",
            "gold_call": "sig(_oracle_solve_joint_moment_equations(design_functionals, moments, 0.85), 1.0)",
        },
        # --- Boundary: moments generated by the working model itself, so the fit
        #     must return the generating triple 1.75, 0.4, 0.55 exactly ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
design_functionals = np.array([1.419491525424, 0.725000000000, 3.129603562195,
                               1.419491525424, 1.419491525424, 0.725000000000,
                               7.510052877850, 3.129603562195, 3.129603562195,
                               1.419491525424, 3.129603562195, 1.419491525424,
                               1.419491525424, 0.725000000000])
moments = np.array([1.292706567797, 5.856629032269, 34.202420167607])
""",
            "call": "sig(solve_joint_moment_equations(design_functionals, moments, 0.75), 1.0)",
            "gold_call": "sig(_oracle_solve_joint_moment_equations(design_functionals, moments, 0.75), 1.0)",
        },
        # --- Edge: a small study whose design functionals are far from the
        #     testbed values, checking the bracket is not tuned to one scale ---
        {
            "setup": """import numpy as np
def sig(a, s):
    a = np.asarray(a, dtype=float).ravel()
    k = np.arange(a.size, dtype=float) + 1.0
    return round(float((np.abs(a).sum() + a @ np.cos(k)) / s), 9)
design_functionals = np.array([1.019943019943, 0.527777777778, 2.296369347651,
                               1.019943019943, 1.019943019943, 0.527777777778,
                               5.724283581614, 2.296369347651, 2.296369347651,
                               1.019943019943, 2.296369347651, 1.019943019943,
                               1.019943019943, 0.527777777778])
moments = np.array([0.770730452675, 2.291551994763, 8.948791141703])
""",
            "call": "sig(solve_joint_moment_equations(design_functionals, moments, 0.8), 1.0)",
            "gold_call": "sig(_oracle_solve_joint_moment_equations(design_functionals, moments, 0.8), 1.0)",
        },
        # --- Invalid: moments admitting no triple with a positive genetic
        #     spectrum inside the residual one ---
        {
            "setup": """import numpy as np
design_functionals = np.array([2.490277777778, 1.245833333333, 5.806018518519,
                               2.490277777778, 2.490277777778, 1.245833333333,
                               14.917901234568, 5.806018518518, 5.806018518519,
                               2.490277777778, 5.806018518519, 2.490277777778,
                               2.490277777778, 1.245833333333])
moments = np.array([0.02, 0.005, 0.001])
def run_model():
    try:
        solve_joint_moment_equations(design_functionals, moments, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_joint_moment_equations(design_functionals, moments, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a design table tabulated only to word length two ---
        {
            "setup": """import numpy as np
design_functionals = np.array([2.490277777778, 1.245833333333, 5.806018518519,
                               2.490277777778, 2.490277777778, 1.245833333333])
moments = np.array([1.818236111111, 9.017816515426, 58.639776199234])
def run_model():
    try:
        solve_joint_moment_equations(design_functionals, moments, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_solve_joint_moment_equations(design_functionals, moments, 0.8)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]

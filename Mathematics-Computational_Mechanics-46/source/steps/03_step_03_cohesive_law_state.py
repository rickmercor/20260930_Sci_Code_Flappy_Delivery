"""
Evaluates the stiffness-capped Camacho-Ortiz traction-separation law for a set of interfaces given their current opening and opening history: it updates the monotone damage history, locates the crossover damage where the damage-secant stiffness first drops below the setup-time cap, and returns the traction each interface carries in whichever of the two regimes it currently occupies.

Capped Camacho-Ortiz traction-separation law.

An interface's damage is a one-way ratchet: it is defined from the largest opening the interface has ever sustained, so it can only grow, and any interface that is currently compressed or has never opened stays undamaged. On monotonic opening, the underlying damage-secant law reduces to a linear softening envelope running from the cohesive strength at zero damage down to zero traction at full damage, and the area under that envelope equals the interface's fracture energy by construction, which is what ties the critical opening to the cohesive strength and the fracture energy. Because the raw secant stiffness diverges as damage approaches zero, every interface is given a stiffness cap at setup, and there is a unique damage value at which the secant stiffness exactly equals that cap; below this crossover damage the interface is treated as contributing no stiffness at all and instead carries a constant cap traction opposing any positive opening, while at or above the crossover damage the interface contributes the true damage-secant spring. Damage must also be clamped at full decohesion, because pushing the ratio defining the secant stiffness past that point would flip its sign and turn the spring unstable.

Returns
-------
result : dict -- delta_c, envelope_slope, d_tilde, k_d0, cap_ratio, damage, dmax_open, traction (see docstring).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def cohesive_law_state(delta, dmax_open, n_e: int = 500, length: float = 1e-3, youngs_modulus: float = 370e9, alpha: float = 10.0, sigma_c: float = 262e6, Gc: float = 50.0) -> dict:
    """Evaluate the capped Camacho-Ortiz law for a set of interfaces.

    Parameters
    ----------
    delta : array_like
        Current signed opening of each interface, in metres.
    dmax_open : array_like
        Largest opening previously sustained by each interface, in metres
        (same shape as delta).
    n_e : int
        Number of bulk elements used only to size the element length that
        sets the interface stiffness cap (must be an even integer >= 2).
    length : float
        Total bar length, in metres (must be > 0).
    youngs_modulus : float
        Bulk Young's modulus, in pascals (must be > 0).
    alpha : float
        Interface stiffness cap ratio, dimensionless (must be > 0).
    sigma_c : float
        Cohesive strength, in pascals (must be > 0).
    Gc : float
        Specific fracture energy, in J/m^2 (must be > 0).

    Returns
    -------
    result : dict
        delta_c : float, critical opening 2*Gc/sigma_c.
        envelope_slope : float, sigma_c/delta_c.
        d_tilde : float, crossover damage where the secant stiffness equals
            the setup-time cap.
        k_d0 : float, secant stiffness evaluated at damage 1e-6.
        cap_ratio : float, k_d0 divided by the setup-time cap.
        damage : ndarray, updated damage of each interface, clamped to 1.
        dmax_open : ndarray, updated opening history of each interface.
        traction : ndarray, interface traction in whichever regime each
            interface currently occupies.

    Raises
    ------
    ValueError
        If delta and dmax_open do not share the same shape, if n_e is not
        an even integer >= 2, or if length, youngs_modulus, alpha, sigma_c,
        or Gc is not a positive real number.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cohesive_law_state(delta, dmax_open, n_e: int = 500, length: float = 1e-3, youngs_modulus: float = 370e9, alpha: float = 10.0, sigma_c: float = 262e6, Gc: float = 50.0) -> dict:
    """Reference implementation of cohesive_law_state."""
    if not isinstance(n_e, (int, np.integer)) or int(n_e) < 2 or int(n_e) % 2 != 0:
        raise ValueError("n_e must be an even integer >= 2")
    if not (length > 0.0 and youngs_modulus > 0.0 and alpha > 0.0 and sigma_c > 0.0 and Gc > 0.0):
        raise ValueError("length, youngs_modulus, alpha, sigma_c, and Gc must be positive")
    delta = np.asarray(delta, dtype=float)
    dmax_open = np.asarray(dmax_open, dtype=float)
    if delta.shape != dmax_open.shape:
        raise ValueError("delta and dmax_open must share the same shape")

    n_e = int(n_e)
    h_e = length / n_e
    delta_c = 2.0 * Gc / sigma_c
    envelope_slope = sigma_c / delta_c
    k_tilde = alpha * youngs_modulus / h_e
    d_tilde = sigma_c / (sigma_c + k_tilde * delta_c)

    def _k_of_d(d):
        d = np.asarray(d, dtype=float)
        safe = np.maximum(d, 1e-250)
        return (1.0 - safe) / safe * envelope_slope

    k_d0 = float(_k_of_d(1e-6))
    cap_ratio = k_d0 / k_tilde

    dmax_new = np.maximum(dmax_open, np.maximum(delta, 0.0))
    damage = np.minimum(dmax_new / delta_c, 1.0)
    cap_regime = damage < d_tilde
    cap_traction = np.where(delta > 0.0, sigma_c * (1.0 - damage), 0.0)
    traction = np.where(cap_regime, cap_traction, _k_of_d(damage) * delta)

    return {
        "delta_c": delta_c,
        "envelope_slope": envelope_slope,
        "d_tilde": d_tilde,
        "k_d0": k_d0,
        "cap_ratio": cap_ratio,
        "damage": damage,
        "dmax_open": dmax_new,
        "traction": traction,
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications.

    >= 3 cases: a normal scenario mixing a compressed, a cap-regime, and a
    secant-regime interface; a compressed cap-regime case whose zero-traction
    result is compared against the oracle (gold_call mirrors the call with the
    oracle name); standalone boundary checks for the constitutive constants at
    the setup-time damage; a full-decohesion edge case that exercises the
    damage clamp; and two invalid-input edge cases (shape mismatch,
    non-positive Gc).
    """
    return [
        {
            # regression: compression in the cap regime carries no traction.
            "setup": "import numpy as np",
            "call": (
                "float(cohesive_law_state(np.array([-1e-8]), "
                "np.array([0.0]))['traction'][0])"
            ),
            "gold_call": (
                "float(_oracle_cohesive_law_state(np.array([-1e-8]), "
                "np.array([0.0]))['traction'][0])"
            ),
        },
        {
            # normal: compression (no damage growth), a low-damage opening
            # (cap regime), and a large opening past the crossover damage
            # (secant regime) -- exposes the traction sum across regimes.
            "setup": "import numpy as np",
            "call": (
                "float(np.sum(cohesive_law_state("
                "np.array([-1e-8, 1e-8, 2e-7]), np.array([0.0, 1e-8, 1e-7]))['traction'])) "
                "+ float(np.max(cohesive_law_state("
                "np.array([-1e-8, 1e-8, 2e-7]), np.array([0.0, 1e-8, 1e-7]))['damage']))"
            ),
            "gold_call": (
                "float(np.sum(_oracle_cohesive_law_state("
                "np.array([-1e-8, 1e-8, 2e-7]), np.array([0.0, 1e-8, 1e-7]))['traction'])) "
                "+ float(np.max(_oracle_cohesive_law_state("
                "np.array([-1e-8, 1e-8, 2e-7]), np.array([0.0, 1e-8, 1e-7]))['damage']))"
            ),
        },
        {
            # boundary: benchmark critical separation.
            "setup": "import numpy as np",
            "call": (
                "float(cohesive_law_state(np.array([3.816793893129771e-13]), "
                "np.array([3.816793893129771e-13]))['delta_c'])"
            ),
            "gold_call": (
                "float(_oracle_cohesive_law_state(np.array([3.816793893129771e-13]), "
                "np.array([3.816793893129771e-13]))['delta_c'])"
            ),
        },
        {
            # boundary: benchmark cap-to-secant crossover damage.
            "setup": "import numpy as np",
            "call": (
                "float(cohesive_law_state(np.array([3.816793893129771e-13]), "
                "np.array([3.816793893129771e-13]))['d_tilde'])"
            ),
            "gold_call": (
                "float(_oracle_cohesive_law_state(np.array([3.816793893129771e-13]), "
                "np.array([3.816793893129771e-13]))['d_tilde'])"
            ),
        },
        {
            # boundary: secant stiffness at the setup-time damage d0 = 1e-6.
            "setup": "import numpy as np",
            "call": (
                "float(cohesive_law_state(np.array([3.816793893129771e-13]), "
                "np.array([3.816793893129771e-13]))['k_d0'])"
            ),
            "gold_call": (
                "float(_oracle_cohesive_law_state(np.array([3.816793893129771e-13]), "
                "np.array([3.816793893129771e-13]))['k_d0'])"
            ),
        },
        {
            # boundary: setup secant stiffness relative to the interface cap.
            "setup": "import numpy as np",
            "call": (
                "float(cohesive_law_state(np.array([3.816793893129771e-13]), "
                "np.array([3.816793893129771e-13]))['cap_ratio'])"
            ),
            "gold_call": (
                "float(_oracle_cohesive_law_state(np.array([3.816793893129771e-13]), "
                "np.array([3.816793893129771e-13]))['cap_ratio'])"
            ),
        },
        {
            # edge: an opening far beyond delta_c must clamp damage at 1
            # and drive the traction to zero (full decohesion).
            "setup": "import numpy as np",
            "call": (
                "float(cohesive_law_state(np.array([1.0]), np.array([1.0]))['damage'][0]) "
                "+ float(cohesive_law_state(np.array([1.0]), np.array([1.0]))['traction'][0])"
            ),
            "gold_call": (
                "float(_oracle_cohesive_law_state(np.array([1.0]), np.array([1.0]))['damage'][0]) "
                "+ float(_oracle_cohesive_law_state(np.array([1.0]), np.array([1.0]))['traction'][0])"
            ),
        },
        {
            # edge: mismatched shapes are invalid
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: cohesive_law_state(np.array([1e-8, 2e-8]), np.array([1e-8])))",
            "gold_call": "_guard(lambda: _oracle_cohesive_law_state(np.array([1e-8, 2e-8]), np.array([1e-8])))",
        },
        {
            # edge: a non-positive fracture energy is invalid
            "setup": """import numpy as np
def _guard(thunk):
    try:
        thunk()
        return 0
    except ValueError:
        return 2
    except Exception:
        return 1""",
            "call": "_guard(lambda: cohesive_law_state(np.array([1e-8]), np.array([1e-8]), Gc=0.0))",
            "gold_call": "_guard(lambda: _oracle_cohesive_law_state(np.array([1e-8]), np.array([1e-8]), Gc=0.0))",
        },
    ]

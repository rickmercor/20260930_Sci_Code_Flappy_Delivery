"""
Find the nominal stress and the two coexisting stretches at which a neck propagates steadily along the stripe.

A propagating neck converts un-necked tissue into necked tissue at constant load, so the load must supply exactly the work stored and released as each material slice crosses the neck front.

Returns
-------
np.ndarray: [propagation stress, un-necked stretch, necked stretch].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_propagation_state(
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
    tolerance: float,
) -> "np.ndarray":
    """Return the steady necking-propagation stress and the coexisting stretches.

    With the constitutive law of ``compute_nominal_stress``, find the
    nominal stress ``s`` at which an un-necked region at stretch ``u`` in
    ``[1, transition_stretch]`` and a necked region at stretch
    ``m > transition_stretch`` coexist in steady propagation: both carry
    ``s``, and the work ``s * (m - u)`` done on unit reference area passing
    from ``u`` to ``m`` equals ``compute_stretching_work`` from ``u`` to
    ``m``. The admissible ``s`` lies between the larger of the stresses at
    stretch 1 and immediately above ``transition_stretch``, and the stress at
    ``transition_stretch``. The upper end is the smaller of that transition
    stress and the largest stress reached before the connected admissible
    necked branch ends; over this range the stress rises with the stretch on
    each branch. Return ``s`` with absolute accuracy ``tolerance`` and the
    stretches ``u`` and ``m`` carrying ``s`` on the two branches.

    Parameters
    ----------
    transition_stretch : float
        Stretch of the rearrangement; must exceed 1.
    edge_scale : float
        Positive edge-length scale of the reference regular hexagon.
    kappa : float
        Positive perimeter rigidity ratio.
    chi : float
        Positive preferred perimeter.
    tolerance : float
        Positive absolute accuracy of ``s``, at most ``1e-6``.

    Returns
    -------
    np.ndarray
        Float array ``[s, u, m]``.

    Raises
    ------
    ValueError
        If any argument is not a finite positive real number (booleans are
        rejected), if ``tolerance`` exceeds ``1e-6``, if the stress
        immediately above ``transition_stretch`` or at stretch 1 is not below
        the stress at ``transition_stretch``, if no admissible ``s`` balances
        the work, or if an earlier step rejects a state it visits.
    """
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_solve_propagation_state(
    transition_stretch: float,
    edge_scale: float,
    kappa: float,
    chi: float,
    tolerance: float,
) -> "np.ndarray":
    """Reference implementation (Brent roots nested in a bracketed work balance)."""
    import numpy as np
    from scipy.optimize import brentq

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    arguments = (("transition_stretch", transition_stretch), ("edge_scale", edge_scale),
                 ("kappa", kappa), ("chi", chi), ("tolerance", tolerance))
    for name, value in arguments:
        if not (_is_number(value) and value > 0.0):
            raise ValueError(f"{name} must be a finite positive number")
    if tolerance > 1e-6:
        raise ValueError("tolerance must not exceed 1e-6")
    lam_t = float(transition_stretch)
    k = float(edge_scale)

    def _stress(lam):
        return _oracle_compute_nominal_stress(lam, lam_t, k, kappa, chi)

    peak = _stress(lam_t)
    above = float(np.nextafter(lam_t, np.inf))
    floor = max(_stress(1.0), _stress(above))
    if not floor < peak:
        raise ValueError("the stress does not drop below its value at the transition")

    # Find either a valid point reaching the pre-transition peak or the end
    # of the connected admissible necked branch, whichever comes first.
    necked_limit = above
    necked_limit_stress = _stress(above)
    trial = 1.05 * above
    for _ in range(500):
        try:
            trial_stress = _stress(trial)
        except ValueError:
            valid = necked_limit
            invalid = trial
            valid_stress = necked_limit_stress
            for _ in range(200):
                middle = 0.5 * (valid + invalid)
                if middle <= valid or middle >= invalid:
                    break
                try:
                    middle_stress = _stress(middle)
                except ValueError:
                    invalid = middle
                else:
                    valid = middle
                    valid_stress = middle_stress
            necked_limit = valid
            necked_limit_stress = valid_stress
            break
        necked_limit = trial
        necked_limit_stress = trial_stress
        if trial_stress >= peak:
            break
        trial *= 1.05
        if trial > 1e3:
            raise ValueError("the admissible necked branch could not be bounded")
    else:
        raise ValueError("the admissible necked branch could not be bounded")

    ceiling = min(peak, necked_limit_stress)
    if not floor < ceiling:
        raise ValueError("the necked branch has no common increasing stress range")

    def _unnecked(level):
        return brentq(lambda lam: _stress(lam) - level, 1.0, lam_t, xtol=1e-15, rtol=1e-15)

    def _necked(level):
        return brentq(lambda lam: _stress(lam) - level, above, necked_limit,
                      xtol=1e-15, rtol=1e-15)

    def _balance(level):
        u, m = _unnecked(level), _necked(level)
        return _oracle_compute_stretching_work(u, m, lam_t, k, kappa, chi) - level * (m - u)

    if not (_balance(floor) > 0.0 > _balance(ceiling)):
        raise ValueError("no admissible stress balances the work")
    level = brentq(_balance, floor, ceiling, xtol=float(tolerance), rtol=1e-15)
    return np.array([float(level), float(_unnecked(level)), float(_necked(level))])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    triple = (
        "import numpy as np\n"
        "def _triple(a):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (3,):\n"
        "        return -1.0\n"
        "    return float(a[0] + 2.0 * a[1] + 3.0 * a[2])\n"
    )
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": triple,
            "call": "_triple(solve_propagation_state(1.45, 0.9, 0.3, 2.9, 1e-12))",
            "gold_call": "_triple(_oracle_solve_propagation_state(1.45, 0.9, 0.3, 2.9, 1e-12))",
        },
        {
            "setup": triple,
            "call": "float(solve_propagation_state(1.53, 0.99784776, 0.16, 3.7, 1e-13)[0]) * 10.0",
            "gold_call": "float(_oracle_solve_propagation_state(1.53, 0.99784776, 0.16, 3.7, 1e-13)[0]) * 10.0",
        },
        {
            "setup": triple,
            "call": "_triple(solve_propagation_state(1.2, 0.62, 0.9, 1.6, 1e-12))",
            "gold_call": "_triple(_oracle_solve_propagation_state(1.2, 0.62, 0.9, 1.6, 1e-12))",
        },
        {
            "setup": triple,
            "call": "float(solve_propagation_state(1.6, 0.9, 0.3, 2.9, 1e-12)[2])",
            "gold_call": "float(_oracle_solve_propagation_state(1.6, 0.9, 0.3, 2.9, 1e-12)[2])",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_propagation_state(1.45, 0.9, 0.3, 2.9, 1e-3))",
            "gold_call": "_status(lambda: _oracle_solve_propagation_state(1.45, 0.9, 0.3, 2.9, 1e-3))",
        },
        {
            "setup": status,
            "call": "_status(lambda: solve_propagation_state(1.0, 0.9, 0.3, 2.9, 1e-12))",
            "gold_call": "_status(lambda: _oracle_solve_propagation_state(1.0, 0.9, 0.3, 2.9, 1e-12))",
        },
    ]

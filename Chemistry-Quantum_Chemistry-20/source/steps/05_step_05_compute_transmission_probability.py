"""
Compute the quantum transmission probability through the barrier for a level below the barrier top from the scattering states matched at its inner turning points.

The fraction of right-moving probability that crosses the barrier follows from scattering states built out of the right- and left-moving parts of the exact stationary state, joined smoothly across the classically forbidden region between the two inner turning points.

Returns
-------
float: transmission probability |T_q|**2 of the level at its inner turning points.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_transmission_probability(m: int, p: float, q: int, mu2: float) -> float:
    """Return the transmission probability |T_q|**2 of a level below the barrier top.

    ``psi_q``, ``F``, ``chi``, ``alpha``, ``psi_r``, ``psi_l`` and ``x_min``
    are defined as in ``compute_right_moving_flux``, with the constant
    ``mu**2`` set to ``mu2`` (``compute_right_moving_flux`` supplies the
    regularizer of the level, but any positive value is accepted). The phase
    is odd, ``chi(-x) = -chi(x)``. Let ``a_q`` be the inner turning point,
    ``0 < a_q < x_min`` with ``U(a_q) = eps_q``. Matching the scattering
    states in value and slope at ``x = -a_q`` and ``x = a_q``, with
    reflection amplitude ``R_q = 1 - T_q``, gives::

        T_q = psi_q(a_q) / (psi_r(a_q) - psi_l(-a_q))          for odd q
        T_q = psi_q'(a_q) / (psi_r'(a_q) - psi_l'(-a_q))       for even q

    where primes denote ``d/dx``. Return ``|T_q|**2``.

    Parameters
    ----------
    m : int
        Order of the potential, a positive integer.
    p : float
        Potential parameter with ``p**2 > m**2 - 1/4``.
    q : int
        Index of a level with ``eps_q < 0``.
    mu2 : float
        Positive finite value of ``mu**2``.

    Returns
    -------
    float
        The transmission probability ``|T_q|**2``.

    Raises
    ------
    ValueError
        If ``m``, ``p`` or ``q`` is invalid as in ``evaluate_well_eigenfunction``,
        if ``eps_q >= 0``, or if ``mu2`` is not a finite positive real number
        (booleans are rejected).
    """
    return transmission

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_transmission_probability(m: int, p: float, q: int, mu2: float) -> float:
    """Reference implementation: closed forms of the matched amplitude in terms of tan(chi) and its slope."""
    import math
    import numpy as np
    from scipy.optimize import brentq

    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError("q must be a nonnegative integer")
    if isinstance(mu2, bool) or not isinstance(mu2, (int, float, np.integer, np.floating)):
        raise ValueError("mu2 must be a real number")
    if not (math.isfinite(mu2) and mu2 > 0.0):
        raise ValueError("mu2 must be finite and positive")
    energy = float(_oracle_compute_well_energy_levels(m, p, int(q) + 1)[q])
    if not energy < 0.0:
        raise ValueError("level q does not lie below the barrier top")
    x_min = math.acos(((m * m - 0.25) / (p * p)) ** 0.25)
    turning = brentq(lambda x: _well_potential(x, m, p) - energy, 0.0, x_min,
                     xtol=1.0e-15, rtol=4.0 * np.finfo(float).eps, maxiter=500)
    values = _oracle_evaluate_well_eigenfunction(m, p, q, np.array([x_min, turning]))
    (psi_min, psi), (dpsi_min, dpsi) = values
    rate = math.sqrt(2.0) * abs(dpsi_min) / math.sqrt(psi_min * psi_min + mu2)
    alpha = math.exp(rate * (turning - x_min))
    d2psi = (_well_potential(turning, m, p) - energy) * psi  # zero at a turning point
    numerator, d_numerator = dpsi * alpha, d2psi * alpha + dpsi * rate * alpha
    denominator = psi * (psi * psi + mu2)
    d_denominator = dpsi * (3.0 * psi * psi + mu2)
    tan_chi = -numerator / denominator
    d_tan_chi = -(d_numerator * denominator - numerator * d_denominator) / denominator ** 2
    if q % 2 == 1:
        return float(1.0 / (1.0 + tan_chi * tan_chi))
    lead = dpsi * (1.0 + tan_chi * tan_chi) + psi * tan_chi * d_tan_chi
    return float(dpsi * dpsi * (1.0 + tan_chi * tan_chi)
                 / (lead * lead + psi * psi * d_tan_chi * d_tan_chi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    return [
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_transmission_probability(18, 27.089280949903493, 1, 5.080934285923759))",
            "gold_call": "float(_oracle_compute_transmission_probability(18, 27.089280949903493, 1, 5.080934285923759))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_transmission_probability(18, 27.089280949903493, 2, 1.3038290874147116))",
            "gold_call": "float(_oracle_compute_transmission_probability(18, 27.089280949903493, 2, 1.3038290874147116))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_transmission_probability(18, 27.089280949903493, 0, 0.05))",
            "gold_call": "float(_oracle_compute_transmission_probability(18, 27.089280949903493, 0, 0.05))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_transmission_probability(2, 7.82971, 1, 3.1328994169027675))",
            "gold_call": "float(_oracle_compute_transmission_probability(2, 7.82971, 1, 3.1328994169027675))",
        },
        {
            "setup": "import numpy as np\n",
            "call": "float(compute_transmission_probability(38, 47.9421, 2, 1.6940766100506346))",
            "gold_call": "float(_oracle_compute_transmission_probability(38, 47.9421, 2, 1.6940766100506346))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_transmission_probability(18, 27.089280949903493, 1, 0.0))",
            "gold_call": "_status(lambda: _oracle_compute_transmission_probability(18, 27.089280949903493, 1, 0.0))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_transmission_probability(18, 27.089280949903493, 3, 1.0))",
            "gold_call": "_status(lambda: _oracle_compute_transmission_probability(18, 27.089280949903493, 3, 1.0))",
        },
    ]

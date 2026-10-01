"""
Compute the regularizing constant and the right-moving probability flux at the product-well minimum for a level below the barrier top of the trigonometric double well.

A stationary state is split into right- and left-moving parts through an amplitude and a phase, and when the exact wave function is known, the phase can be closed self-consistently at the well bottom without the divergent integrals over its nodes that the usual construction produces.

Returns
-------
np.ndarray: float array [mu_q**2, J_q] for a level below the barrier top.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_right_moving_flux(m: int, p: float, q: int) -> "np.ndarray":
    """Return the flux regularizer mu_q**2 and the right-moving flux J_q of a level below the barrier top.

    Let ``eps_q < 0`` be the level of ``compute_well_energy_levels`` (the
    barrier top is ``U(0) = 0``), ``psi_q`` its eigenfunction from
    ``evaluate_well_eigenfunction`` and ``x_min > 0`` the minimum of
    ``U(x) = (m**2 - 1/4) tan(x)**2 - p**2 sin(x)**2``. Write
    ``psi_q = F cos(chi) = psi_r + psi_l`` with
    ``psi_r = (F / 2) exp(i chi)`` and ``psi_l = (F / 2) exp(-i chi)``,
    where the phase is fixed by::

        tan(chi(x)) = -(psi_q'(x) / psi_q(x)) * alpha(x) / (psi_q(x)**2 + mu**2)
        alpha(x) = exp(sqrt(2) |psi_q'(x_min)| (|x| - x_min) / s),
        s = sqrt(psi_q(x_min)**2 + mu**2)

    and ``mu**2 > 0`` solves::

        (eps_q - U(x_min)) (2 s**3 - 3 sqrt(2) psi_q(x_min) s**2)
            + 2 psi_q'(x_min)**2 (s - 2 sqrt(2) psi_q(x_min)) = 0,

    taking the largest real root ``s > |psi_q(x_min)|``. Primes denote
    ``d/dx``. The flux is the probability current of the right-moving part,
    ``J_q = Im(conj(psi_r) d psi_r / dx)``, evaluated at ``x = x_min`` in the
    reduced units of the model.

    Parameters
    ----------
    m : int
        Order of the potential, a positive integer.
    p : float
        Potential parameter with ``p**2 > m**2 - 1/4``.
    q : int
        Index of a level with ``eps_q < 0``.

    Returns
    -------
    np.ndarray
        Float array ``[mu_q**2, J_q]``.

    Raises
    ------
    ValueError
        If ``m``, ``p`` or ``q`` is invalid as in ``evaluate_well_eigenfunction``,
        if ``eps_q >= 0``, or if the equation for ``s`` has no real root
        with ``s > |psi_q(x_min)|``.
    """
    return flux

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_right_moving_flux(m: int, p: float, q: int) -> "np.ndarray":
    """Reference implementation: J = (1/4) psi^2 d tan(chi)/dx at x_min with psi'' = (U - eps) psi."""
    import math
    import numpy as np

    if isinstance(q, bool) or not isinstance(q, (int, np.integer)) or q < 0:
        raise ValueError("q must be a nonnegative integer")
    energy = float(_oracle_compute_well_energy_levels(m, p, int(q) + 1)[q])
    if not energy < 0.0:
        raise ValueError("level q does not lie below the barrier top")
    x_min = math.acos(((m * m - 0.25) / (p * p)) ** 0.25)
    psi, dpsi = _oracle_evaluate_well_eigenfunction(m, p, q, np.array([x_min]))[:, 0]
    kinetic = energy - _well_potential(x_min, m, p)
    root2 = math.sqrt(2.0)
    roots = np.roots([2.0 * kinetic, -3.0 * root2 * kinetic * psi,
                      2.0 * dpsi * dpsi, -4.0 * root2 * dpsi * dpsi * psi])
    admissible = [r.real for r in roots
                  if abs(r.imag) <= 1.0e-9 * abs(r) and r.real > abs(psi) * (1.0 + 1.0e-12)]
    if not admissible:
        raise ValueError("no admissible root for the flux regularizer")
    s = max(admissible)
    mu2 = s * s - psi * psi
    dalpha = root2 * abs(dpsi) / s  # alpha(x_min) = 1
    # With D = psi^2 + mu^2: psi^2 (tan chi)' = [alpha psi'^2 - psi psi' (alpha' - 2 psi psi' alpha / D)
    # + alpha psi^2 (eps - U)] / D.
    bracket = dpsi * dpsi - psi * dpsi * (dalpha - 2.0 * psi * dpsi / (s * s)) + psi * psi * kinetic
    return np.array([mu2, bracket / (4.0 * s * s)])


def _well_potential(x: float, m: int, p: float) -> float:
    """Trigonometric double-well potential U(x) = (m^2 - 1/4) tan^2 x - p^2 sin^2 x."""
    import math

    return (m * m - 0.25) * math.tan(x) ** 2 - p * p * math.sin(x) ** 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    reduce = (
        "import numpy as np\n"
        "def _sig(v):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    if v.shape != (2,):\n"
        "        return -1.0\n"
        "    return float(v.size + 10.0 * v[0] + 100.0 * v[1])\n"
    )
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
            "setup": reduce,
            "call": "_sig(compute_right_moving_flux(18, 27.089280949903493, 0))",
            "gold_call": "_sig(_oracle_compute_right_moving_flux(18, 27.089280949903493, 0))",
        },
        {
            "setup": reduce,
            "call": "_sig(compute_right_moving_flux(18, 27.089280949903493, 2))",
            "gold_call": "_sig(_oracle_compute_right_moving_flux(18, 27.089280949903493, 2))",
        },
        {
            "setup": reduce,
            "call": "_sig(compute_right_moving_flux(2, 7.82971, 1))",
            "gold_call": "_sig(_oracle_compute_right_moving_flux(2, 7.82971, 1))",
        },
        {
            "setup": reduce,
            "call": "_sig(compute_right_moving_flux(38, 47.9421, 2))",
            "gold_call": "_sig(_oracle_compute_right_moving_flux(38, 47.9421, 2))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_right_moving_flux(18, 27.089280949903493, 3))",
            "gold_call": "_status(lambda: _oracle_compute_right_moving_flux(18, 27.089280949903493, 3))",
        },
        {
            "setup": status,
            "call": "_status(lambda: compute_right_moving_flux(18, 27.089280949903493, -1))",
            "gold_call": "_status(lambda: _oracle_compute_right_moving_flux(18, 27.089280949903493, -1))",
        },
    ]

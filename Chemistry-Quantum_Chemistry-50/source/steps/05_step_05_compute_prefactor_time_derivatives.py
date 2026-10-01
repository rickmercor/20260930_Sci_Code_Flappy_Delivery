"""
Evaluate the first and second total derivatives of the logarithm of the flux-flux prefactor with respect to the imaginary-time split, following the stationary path.

These derivatives characterize the prefactor's dependence on the time split and are used by the downstream time-integration calculation.

Returns
-------
np.ndarray: shape (2,), first and second total tau derivatives of ln|A|.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_prefactor_time_derivatives(surface: "np.ndarray", beta: float, tau: float,
                                       n_beads: int) -> "np.ndarray":
    """Return the first and second total tau derivatives of the log prefactor.

    Derivatives are total with respect to ``tau`` along the stationary-path
    family, with the bead numbers of the two halves held fixed.

    Parameters
    ----------
    surface : np.ndarray
        The seven surface parameters ``(V0, a, m, omega, chi_inf, chi_0, sigma)``.
    beta : float
        Total imaginary time.
    tau : float
        Duration of the first half, ``0 < tau < beta``.
    n_beads : int
        Number of beads ``N``, even and at least 4.

    Returns
    -------
    np.ndarray
        Array ``[d ln|A| / dtau, d^2 ln|A| / dtau^2]`` of shape ``(2,)``.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    return derivatives

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_prefactor_time_derivatives(surface: "np.ndarray", beta: float, tau: float,
                                               n_beads: int) -> "np.ndarray":
    """Reference implementation: Jacobi's formula for the determinant and bilinear flux derivatives."""
    import numpy as np

    s = np.asarray(surface, dtype=float)
    X, V1, V2 = _oracle_locate_flux_instanton(s, beta, tau, n_beads)
    n, half, m = X.shape[0], X.shape[0] // 2, s[2]
    k = 1 + 2 * n + 4 * n * n
    hess, third, fourth = [], [], []
    for order in range(3):
        flat = _oracle_assemble_action_derivatives(X, s, beta, tau, order)
        hess.append(flat[1 + 2 * n:k].reshape(2 * n, 2 * n))
        third.append(flat[k:k + 8 * n].reshape(n, 2, 2, 2))
        fourth.append(flat[k + 8 * n:].reshape(n, 2, 2, 2, 2))

    def _bead_blocks(tensors):
        out = np.zeros((2 * n, 2 * n))
        for i in range(n):
            out[2 * i:2 * i + 2, 2 * i:2 * i + 2] = tensors[i]
        return out

    free = np.ones(2 * n, bool)
    free[[0, 2 * half]] = False
    sel = np.ix_(free, free)
    inverse = np.linalg.inv(hess[0][sel])
    # Total tau derivatives of the Hessian along the stationary family.
    dJ = hess[1] + _bead_blocks(np.einsum("iabc,ic->iab", third[0], V1))
    ddJ = (hess[2] + 2.0 * _bead_blocks(np.einsum("iabc,ic->iab", third[1], V1))
           + _bead_blocks(np.einsum("iabcd,ic,id->iab", fourth[0], V1, V1))
           + _bead_blocks(np.einsum("iabc,ic->iab", third[0], V2)))
    K = inverse @ dJ[sel]
    log1 = n / tau - n / (beta - tau) + np.trace(K)
    log2 = -n / tau**2 - n / (beta - tau) ** 2 + np.trace(inverse @ ddJ[sel]) - np.sum(K * K.T)
    d_a, d_b = 2.0 * tau / n, 2.0 * (beta - tau) / n
    # Each end momentum is (scale) x (bead difference); scale_a ~ 1/tau and scale_b ~ 1/(beta - tau).
    ends = ((0, 1, m / d_a, -1.0 / tau), (half - 1, half, m / d_a, -1.0 / tau),
            (half, half + 1, m / d_b, 1.0 / (beta - tau)), (n - 1, 0, m / d_b, 1.0 / (beta - tau)))
    p, q1, q2 = np.zeros(4), np.zeros(4), np.zeros(4)
    for j, (i0, i1, scale, rate) in enumerate(ends):
        gap, gap1, gap2 = X[i1, 0] - X[i0, 0], V1[i1, 0] - V1[i0, 0], V2[i1, 0] - V2[i0, 0]
        p[j] = scale * gap
        q1[j] = scale * (rate * gap + gap1)
        q2[j] = scale * (2.0 * rate**2 * gap + 2.0 * rate * gap1 + gap2)
    pair = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=float)
    phi = 0.5 * p @ pair @ p
    r1 = (p @ pair @ q1) / phi
    r2 = (q1 @ pair @ q1 + p @ pair @ q2) / phi
    return np.array([-0.5 * log1 + r1, -0.5 * log2 + r2 - r1**2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    common = (
        "import math\n"
        "import numpy as np\n"
        "def _surface(v0, wb, m, om, ci, c0, sg):\n"
        "    return np.array([v0, math.sqrt(2.0 * v0 / m) / wb, m, om, ci, c0, sg])\n"
        "def _logs(v, picks):\n"
        "    v = np.asarray(v, dtype=float)\n"
        "    if v.shape != (2,):\n"
        "        return np.array([-1.0])\n"
        "    return v\n"
    )
    status = (
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
            "setup": common,
            "call": "_logs(compute_prefactor_time_derivatives(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 12), (1,))",
            "gold_call": "_logs(_oracle_compute_prefactor_time_derivatives(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 12), (1,))",
        },
        {
            "setup": common,
            "call": "_logs(compute_prefactor_time_derivatives(_surface(0.009, 0.0049, 1836.0, 0.0027, 0.02, 0.07, 0.3), 3300.0, 1320.0, 10), (0, 1))",
            "gold_call": "_logs(_oracle_compute_prefactor_time_derivatives(_surface(0.009, 0.0049, 1836.0, 0.0027, 0.02, 0.07, 0.3), 3300.0, 1320.0, 10), (0, 1))",
        },
        {
            "setup": common,
            "call": "_logs(compute_prefactor_time_derivatives(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 1900.0, 1045.0, 4), (0, 1))",
            "gold_call": "_logs(_oracle_compute_prefactor_time_derivatives(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 1900.0, 1045.0, 4), (0, 1))",
        },
        {
            "setup": common,
            "call": "_logs(compute_prefactor_time_derivatives(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.012, 0.5), 2600.0, 1300.0, 16), (1,))",
            "gold_call": "_logs(_oracle_compute_prefactor_time_derivatives(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.012, 0.5), 2600.0, 1300.0, 16), (1,))",
        },
        {
            "setup": common + status,
            "call": "_status(lambda: compute_prefactor_time_derivatives(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 0.0, 12))",
            "gold_call": "_status(lambda: _oracle_compute_prefactor_time_derivatives(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 0.0, 12))",
        },
    ]

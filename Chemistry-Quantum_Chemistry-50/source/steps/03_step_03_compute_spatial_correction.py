"""
Compute the first-order hbar correction coefficient of the discretized flux-flux path integral at a fixed imaginary-time split.

This coefficient is the fixed-split contribution that is combined downstream with the time-integration and reactant-partition-function contributions.

Returns
-------
float: first-order hbar correction coefficient of the fixed-split flux-flux path integral.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_spatial_correction(surface: "np.ndarray", beta: float, tau: float, n_beads: int) -> float:
    """Return the fixed-split first-order correction coefficient of the flux-flux path integral.

    Atomic units with hbar = 1 are used throughout.

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
    float
        The first-order correction coefficient at the specified imaginary-time split.

    Raises
    ------
    ValueError
        If the inputs are invalid.
    """
    return correction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_spatial_correction(surface: "np.ndarray", beta: float, tau: float, n_beads: int) -> float:
    """Reference implementation: Green's-function contractions of the bead-local derivative tensors."""
    import numpy as np

    s = np.asarray(surface, dtype=float)
    X = _oracle_locate_flux_instanton(s, beta, tau, n_beads)[0]
    n, half, m = X.shape[0], X.shape[0] // 2, s[2]
    flat = _oracle_assemble_action_derivatives(X, s, beta, tau, 0)
    k = 1 + 2 * n + 4 * n * n
    hess = flat[1 + 2 * n:k].reshape(2 * n, 2 * n)
    third = flat[k:k + 8 * n].reshape(n, 2, 2, 2)
    fourth = flat[k + 8 * n:].reshape(n, 2, 2, 2, 2)
    free = np.ones(2 * n, bool)
    free[[0, 2 * half]] = False
    # The inverse Hessian over the free coordinates, padded with zeros on the pinned ones.
    G = np.zeros_like(hess)
    G[np.ix_(free, free)] = np.linalg.inv(hess[np.ix_(free, free)])
    blocks = G.reshape(n, 2, n, 2).transpose(0, 2, 1, 3)
    local = blocks[np.arange(n), np.arange(n)]
    quartic = -0.125 * np.einsum("iabcd,iab,icd->", fourth, local, local)
    tadpole = np.einsum("iabc,ibc->ia", third, local).ravel()
    cubic = 0.125 * tadpole @ G @ tadpole
    cubic += sum(np.einsum("abc,jad,jbe,jcf,jdef->", third[i], blocks[i], blocks[i], blocks[i], third)
                 for i in range(n)) / 12.0
    d_a, d_b = 2.0 * tau / n, 2.0 * (beta - tau) / n
    ends = ((0, 1, m / d_a), (half - 1, half, m / d_a), (half, half + 1, m / d_b), (n - 1, 0, m / d_b))
    p = np.zeros(4)
    dp = np.zeros((4, 2 * n))
    for j, (i0, i1, scale) in enumerate(ends):
        p[j] = scale * (X[i1, 0] - X[i0, 0])
        dp[j, 2 * i1] += scale
        dp[j, 2 * i0] -= scale
    pair = np.array([[0, 1, 1, 0], [1, 0, 0, 1], [1, 0, 0, 1], [0, 1, 1, 0]], dtype=float)
    phi = 0.5 * p @ pair @ p
    dphi = dp.T @ pair @ p
    ddphi = dp.T @ pair @ dp
    flux = -0.5 * (dphi @ G @ tadpole) / phi + 0.5 * np.sum(ddphi * G) / phi
    return float(quartic + cubic + flux)

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
        "def _wrap(value):\n"
        "    return float(value)\n"
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
            "call": "_wrap(compute_spatial_correction(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 12))",
            "gold_call": "_wrap(_oracle_compute_spatial_correction(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 12))",
        },
        {
            "setup": common,
            "call": "_wrap(compute_spatial_correction(_surface(0.009, 0.0049, 1836.0, 0.0027, 0.02, 0.07, 0.3), 3300.0, 1320.0, 10))",
            "gold_call": "_wrap(_oracle_compute_spatial_correction(_surface(0.009, 0.0049, 1836.0, 0.0027, 0.02, 0.07, 0.3), 3300.0, 1320.0, 10))",
        },
        {
            "setup": common,
            "call": "_wrap(compute_spatial_correction(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.012, 0.5), 2600.0, 1300.0, 8))",
            "gold_call": "_wrap(_oracle_compute_spatial_correction(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.012, 0.5), 2600.0, 1300.0, 8))",
        },
        {
            "setup": common,
            "call": "_wrap(compute_spatial_correction(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 1900.0, 950.0, 4))",
            "gold_call": "_wrap(_oracle_compute_spatial_correction(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 1900.0, 950.0, 4))",
        },
        {
            "setup": common + status,
            "call": "_status(lambda: compute_spatial_correction(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 3200.0, 12))",
            "gold_call": "_status(lambda: _oracle_compute_spatial_correction(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 3200.0, 12))",
        },
    ]

"""
Compute the first-order hbar correction coefficient of the discretized reactant partition function.

This coefficient supplies the reactant contribution to the corrected rate and uses the task's specified ring-polymer discretization.

Returns
-------
float: first-order hbar correction coefficient of the discretized reactant partition function.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_reactant_correction(surface: "np.ndarray", beta: float, n_beads: int) -> float:
    """Return the first-order correction coefficient of the discretized reactant partition function.

    Atomic units with hbar = 1 are used throughout.

    Parameters
    ----------
    surface : np.ndarray
        The seven surface parameters ``(V0, a, m, omega, chi_inf, chi_0, sigma)``.
    beta : float
        Total imaginary time, positive and finite.
    n_beads : int
        Number of beads ``N``, even and at least 4.

    Returns
    -------
    float
        The first-order reactant correction coefficient ``G_r``.

    Raises
    ------
    ValueError
        If ``n_beads`` is not an even integer of at least 4, if ``beta`` is not
        positive and finite, or if ``surface`` does not hold seven finite
        positive numbers.
    """
    return correction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_reactant_correction(surface: "np.ndarray", beta: float, n_beads: int) -> float:
    """Reference implementation: Green's-function contractions on the collapsed reactant path."""
    import math
    import numpy as np

    if isinstance(n_beads, bool) or not isinstance(n_beads, (int, np.integer)) or n_beads < 4 or n_beads % 2:
        raise ValueError("n_beads must be an even integer of at least 4")
    if not (math.isfinite(beta) and beta > 0.0):
        raise ValueError("beta must be positive and finite")
    s = np.asarray(surface, dtype=float)
    if s.shape != (7,) or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("surface must hold seven finite positive numbers")
    n = int(n_beads)
    X = np.zeros((n, 2))
    # Fifty ranges or widths out, the Eckart term is below 1e-40 of its height and the Gaussian bump has vanished.
    X[:, 0] = -50.0 * max(s[1], s[6])
    flat = _oracle_assemble_action_derivatives(X, s, beta, 0.5 * beta, 0)
    k = 1 + 2 * n + 4 * n * n
    hess = flat[1 + 2 * n:k].reshape(2 * n, 2 * n)
    third = flat[k:k + 8 * n].reshape(n, 2, 2, 2)
    fourth = flat[k + 8 * n:].reshape(n, 2, 2, 2, 2)
    free = np.ones(2 * n, bool)
    free[0] = False
    G = np.zeros_like(hess)
    G[np.ix_(free, free)] = np.linalg.inv(hess[np.ix_(free, free)])
    blocks = G.reshape(n, 2, n, 2).transpose(0, 2, 1, 3)
    local = blocks[np.arange(n), np.arange(n)]
    quartic = -0.125 * np.einsum("iabcd,iab,icd->", fourth, local, local)
    tadpole = np.einsum("iabc,ibc->ia", third, local).ravel()
    cubic = 0.125 * tadpole @ G @ tadpole
    cubic += sum(np.einsum("abc,jad,jbe,jcf,jdef->", third[i], blocks[i], blocks[i], blocks[i], third)
                 for i in range(n)) / 12.0
    return float(quartic + cubic)

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
            "call": "_wrap(compute_reactant_correction(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 12))",
            "gold_call": "_wrap(_oracle_compute_reactant_correction(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 12))",
        },
        {
            "setup": common,
            "call": "_wrap(compute_reactant_correction(_surface(0.009, 0.0049, 1836.0, 0.0027, 0.002, 0.07, 0.3), 3300.0, 10))",
            "gold_call": "_wrap(_oracle_compute_reactant_correction(_surface(0.009, 0.0049, 1836.0, 0.0027, 0.002, 0.07, 0.3), 3300.0, 10))",
        },
        {
            "setup": common,
            "call": "_wrap(compute_reactant_correction(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 400.0, 4))",
            "gold_call": "_wrap(_oracle_compute_reactant_correction(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 400.0, 4))",
        },
        {
            "setup": common + status,
            "call": "_status(lambda: compute_reactant_correction(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), -3000.0, 12))",
            "gold_call": "_status(lambda: _oracle_compute_reactant_correction(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), -3000.0, 12))",
        },
    ]

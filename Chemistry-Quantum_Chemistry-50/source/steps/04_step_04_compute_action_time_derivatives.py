"""
Evaluate the stationary action as a function of the imaginary-time split together with its first four total derivatives with respect to that split.

These derivatives characterize the stationary action's dependence on the time split and are used by the downstream time-integration calculation.

Returns
-------
np.ndarray: shape (5,), the stationary action and its first four total tau derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_action_time_derivatives(surface: "np.ndarray", beta: float, tau: float,
                                    n_beads: int) -> "np.ndarray":
    """Return the stationary action and its first four total derivatives with respect to tau.

    Let ``X(tau)`` be the stationary path of ``locate_flux_instanton`` and
    ``W(tau) = S(X(tau); tau)`` the action of ``assemble_action_derivatives``
    evaluated on it, as a function of the split ``tau`` with the bead numbers
    of the two halves fixed and the two pinned coordinates at zero.

    Parameters
    ----------
    surface : np.ndarray
        The seven surface parameters ``(V0, a, m, omega, chi_inf, chi_0, sigma)``.
    beta : float
        Total imaginary time, as in ``locate_flux_instanton``.
    tau : float
        Duration of the first half, ``0 < tau < beta``.
    n_beads : int
        Number of beads ``N``, even and at least 4.

    Returns
    -------
    np.ndarray
        Array ``[W, dW/dtau, d2W/dtau2, d3W/dtau3, d4W/dtau4]`` of shape ``(5,)``.

    Raises
    ------
    ValueError
        If the inputs are invalid in the sense of ``locate_flux_instanton``.
    """
    return derivatives

# EXPECTED RETURN
#

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_action_time_derivatives(surface: "np.ndarray", beta: float, tau: float,
                                            n_beads: int) -> "np.ndarray":
    """Reference implementation: chain rule with the stationarity condition eliminating higher path derivatives."""
    import numpy as np

    s = np.asarray(surface, dtype=float)
    X, V1, V2 = _oracle_locate_flux_instanton(s, beta, tau, n_beads)
    n = X.shape[0]
    k = 1 + 2 * n + 4 * n * n
    parts = []
    for order in range(5):
        flat = _oracle_assemble_action_derivatives(X, s, beta, tau, order)
        parts.append((flat[0], flat[1:1 + 2 * n], flat[1 + 2 * n:k].reshape(2 * n, 2 * n),
                      flat[k:k + 8 * n].reshape(n, 2, 2, 2), flat[k + 8 * n:].reshape(n, 2, 2, 2, 2)))
    v1, v2 = V1.ravel(), V2.ravel()

    def _cube(t3, A, B, C):
        return np.einsum("iabc,ia,ib,ic->", t3, A, B, C)

    # Terms carrying the third and fourth path derivatives cancel because the path is stationary.
    w1 = parts[1][0]
    w2 = parts[2][0] + parts[1][1] @ v1
    w3 = (parts[3][0] + 3.0 * parts[2][1] @ v1 + 3.0 * v1 @ parts[1][2] @ v1
          + _cube(parts[0][3], V1, V1, V1))
    w4 = (parts[4][0] + 4.0 * parts[3][1] @ v1 + 6.0 * v1 @ parts[2][2] @ v1 + 6.0 * parts[2][1] @ v2
          + 4.0 * _cube(parts[1][3], V1, V1, V1) + 12.0 * v2 @ parts[1][2] @ v1
          + np.einsum("iabcd,ia,ib,ic,id->", parts[0][4], V1, V1, V1, V1)
          + 6.0 * _cube(parts[0][3], V2, V1, V1) + 3.0 * v2 @ parts[0][2] @ v2)
    return np.array([parts[0][0], w1, w2, w3, w4])

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
        "    if v.shape != (5,):\n"
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
            "call": "_logs(compute_action_time_derivatives(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 12), (0, 2, 4))",
            "gold_call": "_logs(_oracle_compute_action_time_derivatives(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 12), (0, 2, 4))",
        },
        {
            "setup": common,
            "call": "_logs(compute_action_time_derivatives(_surface(0.009, 0.0049, 1836.0, 0.0027, 0.02, 0.07, 0.3), 3300.0, 1320.0, 10), (0, 1, 2, 3, 4))",
            "gold_call": "_logs(_oracle_compute_action_time_derivatives(_surface(0.009, 0.0049, 1836.0, 0.0027, 0.02, 0.07, 0.3), 3300.0, 1320.0, 10), (0, 1, 2, 3, 4))",
        },
        {
            "setup": common,
            "call": "_logs(compute_action_time_derivatives(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 1900.0, 1045.0, 4), (0, 1, 2, 3, 4))",
            "gold_call": "_logs(_oracle_compute_action_time_derivatives(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 1900.0, 1045.0, 4), (0, 1, 2, 3, 4))",
        },
        {
            "setup": common,
            "call": "_logs(compute_action_time_derivatives(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 2600.0, 1300.0, 16), (0, 2, 4))",
            "gold_call": "_logs(_oracle_compute_action_time_derivatives(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 2600.0, 1300.0, 16), (0, 2, 4))",
        },
        {
            "setup": common + status,
            "call": "_status(lambda: compute_action_time_derivatives(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 2))",
            "gold_call": "_status(lambda: _oracle_compute_action_time_derivatives(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 2))",
        },
    ]

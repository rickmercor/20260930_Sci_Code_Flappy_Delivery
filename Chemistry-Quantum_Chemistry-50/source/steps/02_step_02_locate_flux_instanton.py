"""
Locate the delocalized stationary ring-polymer path of the flux-side action with one bead of each half fixed on the dividing surface x = 0, and differentiate that path twice with respect to the imaginary-time split.

In the deep-tunnelling regime, the stationary path of the flux-flux correlation function is a periodic tunnelling orbit that crosses the dividing surface at the two junctions of its halves, and the whole semiclassical expansion is built on it.

Returns
-------
np.ndarray: shape (3, N, 2), stationary beads followed by their first and second tau derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def locate_flux_instanton(surface: "np.ndarray", beta: float, tau: float, n_beads: int) -> "np.ndarray":
    """Return the stationary flux-side ring-polymer path and its first two tau derivatives.

    The surface, the ``N``-bead action ``S(X; tau)`` and the coordinate
    layout are those of ``assemble_action_derivatives``. The x coordinates of
    beads ``0`` and ``N/2`` (the junctions of the two halves) are fixed on the
    dividing surface ``x = 0``; every bead lies on ``y = 0``; all other
    coordinates are free. The stationary path ``X(tau)`` is the minimum of
    ``S(X; tau)`` over the free coordinates in which beads ``1, ..., N/2 - 1``
    have ``x > 0`` and beads ``N/2 + 1, ..., N - 1`` have ``x < 0``. It must
    be converged to full double precision. Its derivatives with respect to
    ``tau`` are total derivatives along the family of stationary paths, with
    the bead numbers of the two halves fixed and the two pinned coordinates
    remaining at zero.

    Parameters
    ----------
    surface : np.ndarray
        The seven surface parameters ``(V0, a, m, omega, chi_inf, chi_0, sigma)``.
    beta : float
        Total imaginary time; it must exceed ``2 pi / omega_b`` with
        ``omega_b = sqrt(2 V0 / (m a^2))``.
    tau : float
        Duration of the first half, ``0 < tau < beta``.
    n_beads : int
        Number of beads ``N``, even and at least 4.

    Returns
    -------
    np.ndarray
        Array of shape ``(3, N, 2)``: the stationary bead positions, their
        first derivatives with respect to ``tau`` and their second
        derivatives with respect to ``tau``.

    Raises
    ------
    ValueError
        If ``n_beads`` is not an even integer of at least 4, if ``beta`` does
        not exceed ``2 pi / omega_b``, if the inputs are invalid in the sense
        of ``assemble_action_derivatives``, or if no stationary path with the
        stated sign pattern is found.
    """
    return path

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_locate_flux_instanton(surface: "np.ndarray", beta: float, tau: float, n_beads: int) -> "np.ndarray":
    """Reference implementation: Newton search from the continuum Eckart orbit, then implicit differentiation."""
    import math
    import numpy as np

    if isinstance(n_beads, bool) or not isinstance(n_beads, (int, np.integer)) or n_beads < 4 or n_beads % 2:
        raise ValueError("n_beads must be an even integer of at least 4")
    s = np.asarray(surface, dtype=float)
    if s.shape != (7,) or not np.all(np.isfinite(s)) or np.any(s <= 0.0):
        raise ValueError("surface must hold seven finite positive numbers")
    n, half = int(n_beads), int(n_beads) // 2
    v0, a, m = s[0], s[1], s[2]
    gamma = math.pi * math.sqrt(2.0 * m * a**2 * v0)
    if not (math.isfinite(beta) and beta * v0 > gamma):
        raise ValueError("beta must exceed 2 pi / omega_b")

    def _pieces(order, Y):
        flat = _oracle_assemble_action_derivatives(Y, s, beta, tau, order)
        k = 1 + 2 * n + 4 * n * n
        return flat[1:1 + 2 * n], flat[1 + 2 * n:k].reshape(2 * n, 2 * n), flat[k:k + 8 * n].reshape(n, 2, 2, 2)

    # The continuum periodic orbit of the Eckart barrier is sinh(x/a) proportional to sin(2 pi t / beta).
    energy = gamma**2 / (v0 * beta**2)
    X = np.zeros((n, 2))
    X[:, 0] = a * np.arcsinh(math.sqrt((v0 - energy) / energy) * np.sin(2.0 * math.pi * np.arange(n) / n))
    X[0, 0] = X[half, 0] = 0.0
    free = np.ones(2 * n, bool)
    free[[0, 2 * half]] = False
    size = np.inf
    for _ in range(60):
        grad, hess, _ = _pieces(0, X)
        move = np.zeros(2 * n)
        move[free] = np.linalg.solve(hess[np.ix_(free, free)], -grad[free])
        X = X + move.reshape(n, 2)
        previous, size = size, float(np.max(np.abs(move)))
        if size < 1e-15 * a or (size < 1e-11 * a and size >= previous):
            break
    else:
        raise ValueError("stationary path search did not converge")
    if not (np.all(X[1:half, 0] > 0.0) and np.all(X[half + 1:, 0] < 0.0)):
        raise ValueError("no stationary path with the required sign pattern")
    _, hess0, third = _pieces(0, X)
    grad1, hess1, _ = _pieces(1, X)
    grad2, _, _ = _pieces(2, X)
    inverse = np.linalg.inv(hess0[np.ix_(free, free)])
    first = np.zeros(2 * n)
    first[free] = -inverse @ grad1[free]
    V1 = first.reshape(n, 2)
    drive = np.einsum("iabc,ib,ic->ia", third, V1, V1).ravel() + 2.0 * hess1 @ first + grad2
    second = np.zeros(2 * n)
    second[free] = -inverse @ drive[free]
    return np.stack([X, V1, second.reshape(n, 2)])

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
        "def _digest(v):\n"
        "    blocks = np.asarray(v, dtype=float)\n"
        "    if blocks.ndim != 3 or blocks.shape[0] != 3 or blocks.shape[2] != 2:\n"
        "        return np.array([-1.0])\n"
        "    return blocks\n"
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
            "call": "_digest(locate_flux_instanton(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 12))",
            "gold_call": "_digest(_oracle_locate_flux_instanton(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 12))",
        },
        {
            "setup": common,
            "call": "_digest(locate_flux_instanton(_surface(0.009, 0.0049, 1836.0, 0.0027, 0.02, 0.07, 0.3), 3300.0, 1320.0, 10))",
            "gold_call": "_digest(_oracle_locate_flux_instanton(_surface(0.009, 0.0049, 1836.0, 0.0027, 0.02, 0.07, 0.3), 3300.0, 1320.0, 10))",
        },
        {
            "setup": common,
            "call": "_digest(locate_flux_instanton(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 1900.0, 950.0, 4))",
            "gold_call": "_digest(_oracle_locate_flux_instanton(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 1900.0, 950.0, 4))",
        },
        {
            "setup": common,
            "call": "_digest(locate_flux_instanton(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 2000.0, 1150.0, 16))",
            "gold_call": "_digest(_oracle_locate_flux_instanton(_surface(0.01, 0.005, 1836.0, 0.003, 0.012, 0.09, 0.5), 2000.0, 1150.0, 16))",
        },
        {
            "setup": common + status,
            "call": "_status(lambda: locate_flux_instanton(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 9))",
            "gold_call": "_status(lambda: _oracle_locate_flux_instanton(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 3000.0, 1500.0, 9))",
        },
        {
            "setup": common + status,
            "call": "_status(lambda: locate_flux_instanton(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 900.0, 450.0, 12))",
            "gold_call": "_status(lambda: _oracle_locate_flux_instanton(_surface(0.011, 0.0058, 1836.0, 0.0031, 0.015, 0.08, 0.4), 900.0, 450.0, 12))",
        },
    ]

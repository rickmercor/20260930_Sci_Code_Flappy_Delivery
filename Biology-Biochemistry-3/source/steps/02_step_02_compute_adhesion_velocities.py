"""
Evaluate the overdamped velocity of every cell under polarity-dependent pairwise adhesion with a finite interaction range.

Adhesion is strongest between neighbours whose apico-basal polarities are parallel to each other and perpendicular to the line joining them, so the pair potential depends on the direction of that line as well as on the distance.

Returns
-------
np.ndarray: velocity of every cell, shape (n, 2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_adhesion_velocities(
    positions: np.ndarray,
    angles: np.ndarray,
    magnitudes: np.ndarray,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
) -> np.ndarray:
    """Return the overdamped velocity of every cell.

    Cell ``i`` sits at ``positions[i]`` and carries the polarity vector
    ``p_i = magnitudes[i] * (cos(angles[i]), sin(angles[i]))``. Two distinct
    cells ``i`` and ``j`` are partners when ``r_ij = |r_j - r_i| < cutoff``,
    and a partner pair has the potential ``V_ij = S_ij * U(r_ij)`` with
    ``U(r) = exp(-r) - exp(-r / kernel_range)``,
    ``S_ij = (p_i x e_ij) * (p_j x e_ij) + 1``, ``e_ij = (r_j - r_i) / r_ij``
    and ``a x b = a[0] * b[1] - a[1] * b[0]``. With ``V_i`` the sum of
    ``V_ij`` over the partners of cell ``i``, the velocity of cell ``i`` is
    ``-tau_v`` times the gradient of ``V_i`` with respect to ``r_i``, taken
    at fixed angles and with the partner set held fixed.

    Parameters
    ----------
    positions : np.ndarray
        Float array with shape ``(n, 2)``.
    angles : np.ndarray
        Polarity angles in radians, shape ``(n,)``.
    magnitudes : np.ndarray
        Nonnegative polarity magnitudes, shape ``(n,)``.
    kernel_range : float
        Positive range parameter of ``U``.
    cutoff : float
        Positive interaction range; ``np.inf`` makes every pair a partner.
    tau_v : float
        Positive mobility constant.

    Returns
    -------
    np.ndarray
        Float array with shape ``(n, 2)``: row ``i`` is ``d r_i / d t``.

    Raises
    ------
    ValueError
        If ``positions`` is not a non-empty ``(n, 2)`` array, if ``angles``
        or ``magnitudes`` does not have shape ``(n,)``, if any of these holds
        a non-finite entry or a magnitude is negative, if ``kernel_range``
        or ``tau_v`` is not a finite positive number, if ``cutoff`` is not a
        positive number (``np.inf`` allowed), or if two cells occupy the same
        position.
    """
    return velocities

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_adhesion_velocities(
    positions: np.ndarray,
    angles: np.ndarray,
    magnitudes: np.ndarray,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
) -> np.ndarray:
    """Reference implementation (vectorised analytic gradient)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(positions, dtype=float)
    theta = np.asarray(angles, dtype=float)
    strength = np.asarray(magnitudes, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("positions must have shape (n, 2) with n >= 1")
    count = points.shape[0]
    if theta.shape != (count,) or strength.shape != (count,):
        raise ValueError("angles and magnitudes must have shape (n,)")
    if not (np.all(np.isfinite(points)) and np.all(np.isfinite(theta))
            and np.all(np.isfinite(strength))):
        raise ValueError("positions, angles and magnitudes must be finite")
    if np.any(strength < 0.0):
        raise ValueError("magnitudes must be nonnegative")
    if not (_is_number(kernel_range) and kernel_range > 0.0):
        raise ValueError("kernel_range must be a finite positive number")
    if not (_is_number(tau_v) and tau_v > 0.0):
        raise ValueError("tau_v must be a finite positive number")
    if not ((cutoff == np.inf or _is_number(cutoff)) and cutoff > 0.0):
        raise ValueError("cutoff must be a positive number or np.inf")

    beta = float(kernel_range)
    offset = points[None, :, :] - points[:, None, :]          # r_j - r_i
    distance = np.sqrt(np.sum(offset * offset, axis=2))
    distinct = ~np.eye(count, dtype=bool)
    if np.any(distance[distinct] == 0.0):
        raise ValueError("two cells occupy the same position")
    partner = distinct & (distance < cutoff)
    safe = np.where(partner, distance, 1.0)
    unit = offset / safe[:, :, None]
    # p_hat x e = normal . e with normal = (-sin, cos) of the polarity angle.
    normal = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    cross_i = strength[:, None] * np.einsum("ik,ijk->ij", normal, unit)
    cross_j = strength[None, :] * np.einsum("jk,ijk->ij", normal, unit)
    factor = cross_i * cross_j + 1.0
    kernel = np.exp(-safe) - np.exp(-safe / beta)
    slope = -np.exp(-safe) + np.exp(-safe / beta) / beta
    # d e / d r_i = -(I - e e^T) / r, hence d(m n . e) / d r_i below.
    d_cross_i = -(strength[:, None, None] * normal[:, None, :]
                  - cross_i[:, :, None] * unit) / safe[:, :, None]
    d_cross_j = -(strength[None, :, None] * normal[None, :, :]
                  - cross_j[:, :, None] * unit) / safe[:, :, None]
    d_factor = d_cross_i * cross_j[:, :, None] + cross_i[:, :, None] * d_cross_j
    gradient = -(factor * slope)[:, :, None] * unit + kernel[:, :, None] * d_factor
    gradient = np.where(partner[:, :, None], gradient, 0.0)
    return -float(tau_v) * gradient.sum(axis=1)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _vsig(a, n):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n, 2):\n"
        "        return -1.0\n"
        "    flat = a.ravel()\n"
        "    weights = np.cos(np.arange(1, flat.size + 1, dtype=float))\n"
        "    return float(np.sum(np.abs(flat)) + np.sum(flat * weights))\n"
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
    cluster = (
        "X = np.array([[0.0, 0.0], [1.9, 0.3], [0.8, 1.7], [2.6, 1.9]])\n"
        "T = np.array([0.3, 1.2, 2.5, -0.7])\n"
        "M = np.array([0.9, 0.5, 1.2, 0.7])\n"
    )
    finite_difference = (
        "def _energy(i, xi, X, T, M, b, c):\n"
        "    X = X.copy(); X[i] = xi\n"
        "    total = 0.0\n"
        "    for j in range(len(X)):\n"
        "        if j == i:\n"
        "            continue\n"
        "        d = X[j] - X[i]; r = np.hypot(d[0], d[1])\n"
        "        if not r < c:\n"
        "            continue\n"
        "        e = d / r\n"
        "        ci = M[i] * (np.cos(T[i]) * e[1] - np.sin(T[i]) * e[0])\n"
        "        cj = M[j] * (np.cos(T[j]) * e[1] - np.sin(T[j]) * e[0])\n"
        "        total += (ci * cj + 1.0) * (np.exp(-r) - np.exp(-r / b))\n"
        "    return total\n"
        "def _fd_agrees(fn, X, T, M, b, c, tv):\n"
        "    v = np.asarray(fn(X, T, M, b, c, tv), dtype=float)\n"
        "    if v.shape != X.shape:\n"
        "        return 0.0\n"
        "    h = 1e-5\n"
        "    worst = 0.0\n"
        "    for i in range(len(X)):\n"
        "        for k in range(2):\n"
        "            up = X[i].copy(); up[k] += h\n"
        "            dn = X[i].copy(); dn[k] -= h\n"
        "            g = (_energy(i, up, X, T, M, b, c) - _energy(i, dn, X, T, M, b, c)) / (2 * h)\n"
        "            worst = max(worst, abs(v[i, k] + tv * g))\n"
        "    return 1.0 if worst < 1e-6 else 0.0\n"
    )
    return [
        {
            "setup": helpers + cluster,
            "call": "_vsig(compute_adhesion_velocities(X, T, M, 3.2, 2.6, 1.5), 4)",
            "gold_call": "_vsig(_oracle_compute_adhesion_velocities(X, T, M, 3.2, 2.6, 1.5), 4)",
        },
        {
            "setup": helpers + cluster,
            "call": "float(compute_adhesion_velocities(X, T, M, 3.2, 2.6, 1.5)[2, 1])",
            "gold_call": "float(_oracle_compute_adhesion_velocities(X, T, M, 3.2, 2.6, 1.5)[2, 1])",
        },
        {
            "setup": helpers + (
                "b = 3.2\n"
                "rest = b * np.log(b) / (b - 1.0)\n"
                "X = np.array([[0.0, 0.0], [rest * np.cos(0.4), rest * np.sin(0.4)]])\n"
                "T = np.array([0.9, 2.1]); M = np.array([1.1, 0.8])\n"
            ),
            "call": "_vsig(compute_adhesion_velocities(X, T, M, b, 3.0, 2.0), 2)",
            "gold_call": "_vsig(_oracle_compute_adhesion_velocities(X, T, M, b, 3.0, 2.0), 2)",
        },
        {
            "setup": helpers + finite_difference + cluster,
            "call": "_fd_agrees(compute_adhesion_velocities, X, T, M, 5.8, 2.6, 1.5)",
            "gold_call": "_fd_agrees(_oracle_compute_adhesion_velocities, X, T, M, 5.8, 2.6, 1.5)",
        },
        {
            "setup": helpers + (
                "X = np.array([[0.0, 0.0], [2.5, 0.0], [1.2, 1.1]])\n"
                "T = np.array([1.0, -0.4, 2.2]); M = np.array([0.6, 1.3, 0.9])\n"
            ),
            "call": "_vsig(compute_adhesion_velocities(X, T, M, 4.1, 2.5, 1.0), 3)",
            "gold_call": "_vsig(_oracle_compute_adhesion_velocities(X, T, M, 4.1, 2.5, 1.0), 3)",
        },
        {
            "setup": helpers + (
                "X = np.array([[0.0, 0.0], [2.5, 0.0]])\n"
                "T = np.array([0.7, 2.3]); M = np.array([1.1, 0.6])\n"
            ),
            "call": "float(np.sum(np.abs(compute_adhesion_velocities(X, T, M, 4.1, 2.5, 1.0))))",
            "gold_call": "float(np.sum(np.abs(_oracle_compute_adhesion_velocities(X, T, M, 4.1, 2.5, 1.0))))",
        },
        {
            "setup": helpers + (
                "X = np.array([[0.0, 0.0], [3.5, 0.0], [1.0, 3.2]])\n"
                "T = np.array([0.2, 1.9, -2.6]); M = np.array([0.7, 0.4, 1.0])\n"
            ),
            "call": "_vsig(compute_adhesion_velocities(X, T, M, 5.8, np.inf, 3.0), 3)",
            "gold_call": "_vsig(_oracle_compute_adhesion_velocities(X, T, M, 5.8, np.inf, 3.0), 3)",
        },
        {
            "setup": helpers + (
                "X = np.array([[0.0, 0.0], [1.7, 0.0]])\n"
                "T = np.array([0.5, 2.0]); M = np.zeros(2)\n"
            ),
            "call": "float(compute_adhesion_velocities(X, T, M, 2.6, 2.4, 1.8)[0, 0])",
            "gold_call": "float(_oracle_compute_adhesion_velocities(X, T, M, 2.6, 2.4, 1.8)[0, 0])",
        },
        {
            "setup": helpers + status + (
                "X = np.array([[0.0, 0.0], [1.0, 1.0], [1.0, 1.0]])\n"
                "T = np.zeros(3); M = np.ones(3)\n"
            ),
            "call": "_status(lambda: compute_adhesion_velocities(X, T, M, 3.0, 2.5, 1.0))",
            "gold_call": "_status(lambda: _oracle_compute_adhesion_velocities(X, T, M, 3.0, 2.5, 1.0))",
        },
        {
            "setup": helpers + status + cluster,
            "call": "_status(lambda: compute_adhesion_velocities(X, T[:3], M, 3.2, 2.6, 1.5))",
            "gold_call": "_status(lambda: _oracle_compute_adhesion_velocities(X, T[:3], M, 3.2, 2.6, 1.5))",
        },
    ]

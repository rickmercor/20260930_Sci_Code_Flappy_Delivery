"""
Evaluate the rotation rate of every cell's polarity under adhesion torques and adhesion-driven polarity regulation.

Besides relaxing the adhesion energy, each cell regulates its apico-basal polarity so that the polarity turns away from the cells it touches, at a rate set by a regulation time constant.

Returns
-------
np.ndarray: rotation rate of every polarity angle, shape (n,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
def compute_polarity_rotation_rates(
    positions: np.ndarray,
    angles: np.ndarray,
    magnitudes: np.ndarray,
    regulation_times: np.ndarray,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
) -> np.ndarray:
    """Return the rotation rate of every cell's polarity angle.

    Cells, polarity vectors ``p_i = magnitudes[i] * (cos(angles[i]),
    sin(angles[i]))``, partners (``r_ij < cutoff``) and the potential
    ``V_i = sum_j S_ij * U(r_ij)`` over the partners of cell ``i`` are
    defined as in ``compute_adhesion_velocities``: ``U(r) = exp(-r) -
    exp(-r / kernel_range)``, ``S_ij = (p_i x e_ij) * (p_j x e_ij) + 1``,
    ``e_ij = (r_j - r_i) / r_ij`` and ``a x b = a[0] * b[1] - a[1] * b[0]``.
    With ``u_i = (cos(angles[i]), sin(angles[i]))`` the unit polarity, return
    ``d angles[i] / d t = -tau_v * dV_i/d angles[i]
    - regulation_times[i] * sum_j d(u_i . e_ij)/d angles[i]``, the sum running
    over the partners of cell ``i`` and positions being held fixed.

    Parameters
    ----------
    positions : np.ndarray
        Float array with shape ``(n, 2)``.
    angles : np.ndarray
        Polarity angles in radians, shape ``(n,)``.
    magnitudes : np.ndarray
        Nonnegative polarity magnitudes, shape ``(n,)``.
    regulation_times : np.ndarray
        Nonnegative polarity-regulation constant of each cell, shape ``(n,)``.
    kernel_range : float
        Positive range parameter of ``U``.
    cutoff : float
        Positive interaction range; ``np.inf`` makes every pair a partner.
    tau_v : float
        Positive mobility constant.

    Returns
    -------
    np.ndarray
        Float array with shape ``(n,)``: entry ``i`` is ``d angles[i] / d t``.

    Raises
    ------
    ValueError
        If ``positions`` is not a non-empty ``(n, 2)`` array, if ``angles``,
        ``magnitudes`` or ``regulation_times`` does not have shape ``(n,)``,
        if any of these holds a non-finite entry or a negative magnitude or
        regulation time, if ``kernel_range`` or ``tau_v`` is not a finite
        positive number, if ``cutoff`` is not a positive number (``np.inf``
        allowed), or if two cells occupy the same position.
    """
    return rates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_polarity_rotation_rates(
    positions: np.ndarray,
    angles: np.ndarray,
    magnitudes: np.ndarray,
    regulation_times: np.ndarray,
    kernel_range: float,
    cutoff: float,
    tau_v: float,
) -> np.ndarray:
    """Reference implementation (vectorised analytic angular derivatives)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(positions, dtype=float)
    theta = np.asarray(angles, dtype=float)
    strength = np.asarray(magnitudes, dtype=float)
    regulation = np.asarray(regulation_times, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("positions must have shape (n, 2) with n >= 1")
    count = points.shape[0]
    for values in (theta, strength, regulation):
        if values.shape != (count,):
            raise ValueError("angles, magnitudes and regulation_times must have shape (n,)")
    for values in (points, theta, strength, regulation):
        if not np.all(np.isfinite(values)):
            raise ValueError("inputs must be finite")
    if np.any(strength < 0.0) or np.any(regulation < 0.0):
        raise ValueError("magnitudes and regulation_times must be nonnegative")
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
    polar = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    normal = np.stack([-np.sin(theta), np.cos(theta)], axis=1)
    along_i = np.einsum("ik,ijk->ij", polar, unit)            # u_i . e_ij
    across_i = np.einsum("ik,ijk->ij", normal, unit)          # u_i x e_ij
    cross_j = strength[None, :] * np.einsum("jk,ijk->ij", normal, unit)
    kernel = np.exp(-safe) - np.exp(-safe / beta)
    # d(u_i x e)/d theta_i = -(u_i . e) and d(u_i . e)/d theta_i = u_i x e.
    d_factor = -strength[:, None] * along_i * cross_j
    torque = np.where(partner, kernel * d_factor, 0.0).sum(axis=1)
    alignment = np.where(partner, across_i, 0.0).sum(axis=1)
    return -float(tau_v) * torque - regulation * alignment

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _rsig(a, n):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n,):\n"
        "        return -1.0\n"
        "    weights = np.cos(np.arange(1, n + 1, dtype=float))\n"
        "    return float(np.sum(np.abs(a)) + np.sum(a * weights))\n"
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
        "B = np.array([0.15, 0.4, 0.05, 0.25])\n"
    )
    finite_difference = (
        "def _energy(i, ti, X, T, M, b, c):\n"
        "    T = T.copy(); T[i] = ti\n"
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
        "    w = np.asarray(fn(X, T, M, np.zeros(len(X)), b, c, tv), dtype=float)\n"
        "    if w.shape != (len(X),):\n"
        "        return 0.0\n"
        "    h = 1e-5\n"
        "    worst = 0.0\n"
        "    for i in range(len(X)):\n"
        "        g = (_energy(i, T[i] + h, X, T, M, b, c) - _energy(i, T[i] - h, X, T, M, b, c)) / (2 * h)\n"
        "        worst = max(worst, abs(w[i] + tv * g))\n"
        "    return 1.0 if worst < 1e-6 else 0.0\n"
    )
    return [
        {
            "setup": helpers + cluster,
            "call": "_rsig(compute_polarity_rotation_rates(X, T, M, B, 3.2, 2.6, 1.5), 4)",
            "gold_call": "_rsig(_oracle_compute_polarity_rotation_rates(X, T, M, B, 3.2, 2.6, 1.5), 4)",
        },
        {
            "setup": helpers + cluster,
            "call": "float(compute_polarity_rotation_rates(X, T, M, B, 3.2, 2.6, 1.5)[1])",
            "gold_call": "float(_oracle_compute_polarity_rotation_rates(X, T, M, B, 3.2, 2.6, 1.5)[1])",
        },
        {
            "setup": helpers + finite_difference + cluster,
            "call": "_fd_agrees(compute_polarity_rotation_rates, X, T, M, 4.4, 2.6, 2.0)",
            "gold_call": "_fd_agrees(_oracle_compute_polarity_rotation_rates, X, T, M, 4.4, 2.6, 2.0)",
        },
        {
            "setup": helpers + (
                "X = np.array([[0.0, 0.0], [1.8, 0.0]])\n"
                "T = np.array([2.0, np.pi - 2.0]); M = np.zeros(2); B = np.array([0.3, 0.3])\n"
            ),
            "call": "float(compute_polarity_rotation_rates(X, T, M, B, 3.0, 2.5, 1.0)[0])",
            "gold_call": "float(_oracle_compute_polarity_rotation_rates(X, T, M, B, 3.0, 2.5, 1.0)[0])",
        },
        {
            "setup": helpers + (
                "X = np.array([[0.0, 0.0], [2.5, 0.0], [1.2, 1.1]])\n"
                "T = np.array([1.0, -0.4, 2.2]); M = np.array([0.6, 1.3, 0.9])\n"
                "B = np.array([0.2, 0.0, 0.35])\n"
            ),
            "call": "_rsig(compute_polarity_rotation_rates(X, T, M, B, 4.1, 2.5, 1.0), 3)",
            "gold_call": "_rsig(_oracle_compute_polarity_rotation_rates(X, T, M, B, 4.1, 2.5, 1.0), 3)",
        },
        {
            "setup": helpers + (
                "X = np.array([[0.0, 0.0], [2.5, 0.0]])\n"
                "T = np.array([0.7, 2.3]); M = np.array([1.1, 0.6]); B = np.array([0.4, 0.2])\n"
            ),
            "call": "float(np.sum(np.abs(compute_polarity_rotation_rates(X, T, M, B, 4.1, 2.5, 1.0))))",
            "gold_call": "float(np.sum(np.abs(_oracle_compute_polarity_rotation_rates(X, T, M, B, 4.1, 2.5, 1.0))))",
        },
        {
            "setup": helpers + (
                "X = np.array([[0.0, 0.0], [3.5, 0.0], [1.0, 3.2]])\n"
                "T = np.array([0.2, 1.9, -2.6]); M = np.array([0.7, 0.4, 1.0])\n"
                "B = np.array([0.1, 0.1, 0.1])\n"
            ),
            "call": "_rsig(compute_polarity_rotation_rates(X, T, M, B, 5.8, np.inf, 3.0), 3)",
            "gold_call": "_rsig(_oracle_compute_polarity_rotation_rates(X, T, M, B, 5.8, np.inf, 3.0), 3)",
        },
        {
            "setup": helpers + status + cluster,
            "call": "_status(lambda: compute_polarity_rotation_rates(X, T, M, -B, 3.2, 2.6, 1.5))",
            "gold_call": "_status(lambda: _oracle_compute_polarity_rotation_rates(X, T, M, -B, 3.2, 2.6, 1.5))",
        },
        {
            "setup": helpers + status + cluster,
            "call": "_status(lambda: compute_polarity_rotation_rates(X, T, M, B[:2], 3.2, 2.6, 1.5))",
            "gold_call": "_status(lambda: _oracle_compute_polarity_rotation_rates(X, T, M, B[:2], 3.2, 2.6, 1.5))",
        },
    ]

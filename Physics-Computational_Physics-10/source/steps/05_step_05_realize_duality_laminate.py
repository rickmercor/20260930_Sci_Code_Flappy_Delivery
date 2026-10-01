"""
Realize the anisotropic transformation medium of every triangle as a laminate of two isotropic materials with prescribed layer fractions, giving the two constituent conductivities and the direction along which the layers run.

Thin layers of isotropic materials provide a local realization of the anisotropic transformation medium. Constituent choice and layer orientation must reproduce the target principal response.

Returns
-------
np.ndarray: constituent conductivities and layer direction $(a,b,\theta)$ of every triangle, shape (e, 3).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def realize_duality_laminate(
    jacobians: "np.ndarray",
    background_conductivity: float,
    high_fraction: "float | np.ndarray" = 0.5,
) -> "np.ndarray":
    r"""Return the prescribed-fraction bilayer that realizes each transformation medium.

    The target of triangle $e$ is the transformation medium prescribed by its
    virtual-reference-to-physical Jacobian ``jacobians[e]`` and isotropic
    reference conductivity ``background_conductivity``. It is to be built
    from two isotropic materials with conductivities $a\ge b>0$, with
    material $a$ occupying fraction $f$ = ``high_fraction`` of each local period
    and material $b$ occupying $1-f$. The layers run along direction $\theta$
    (angle from the $x$ axis, $0\le\theta<\pi$). The effective laminate must
    reproduce the target tensor. When the two principal values of the target
    differ by at most $10^{-9}$ times their sum, treat it as isotropic:
    $a=b$ equal to their mean and $\theta=0$.

    Parameters
    ----------
    jacobians : np.ndarray
        Float array of shape (e, 2, 2) with positive determinants.
        Resolve positive material values
        even when the principal-conductivity ratio is as high as $10^{10}$;
        tested inputs have representable finite target values.
    background_conductivity : float
        Positive background conductivity $\kappa_0$.
    high_fraction : float or np.ndarray
        Scalar or shape (e,) array, with finite entries strictly between
        zero and one. Fraction of the higher-conductivity constituent.
        The default 0.5 is the equal-thickness benchmark.

    Returns
    -------
    np.ndarray
        Float array of shape (e, 3); row $e$ holds $(a,b,\theta)$.

    Raises
    ------
    ValueError
        If ``jacobians`` is not a non-empty finite (e, 2, 2) array, if some
        determinant is not positive, or if ``background_conductivity`` is not
        a finite positive number, or if high_fraction has an invalid
        shape, non-finite entry, or an entry outside the open interval (0, 1).
    """
    return laminate

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_realize_duality_laminate(
    jacobians: "np.ndarray",
    background_conductivity: float,
    high_fraction: "float | np.ndarray" = 0.5,
) -> "np.ndarray":
    """Reference implementation (closed-form principal axes of the transformation medium)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    jac = np.asarray(jacobians, dtype=float)
    if (jac.ndim != 3 or jac.shape[1:] != (2, 2) or jac.shape[0] == 0
            or not np.all(np.isfinite(jac))):
        raise ValueError("jacobians must be a non-empty finite (e, 2, 2) array")
    if not (_is_number(background_conductivity) and background_conductivity > 0.0):
        raise ValueError("background_conductivity must be a finite positive number")
    f = np.asarray(high_fraction, dtype=float)
    if f.ndim == 0:
        f = np.full(jac.shape[0], float(f))
    if (f.shape != (jac.shape[0],) or not np.all(np.isfinite(f))
            or np.any((f <= 0.0) | (f >= 1.0))):
        raise ValueError("high_fraction must be scalar or (e,) with 0 < f < 1")

    # The 2D transformation is independent of each Jacobian's scalar scale.
    scale = np.max(np.abs(jac), axis=(1, 2))
    if np.any(scale == 0.0):
        raise ValueError("every Jacobian must have a positive determinant")
    reduced = jac / scale[:, None, None]
    det = reduced[:, 0, 0] * reduced[:, 1, 1] - reduced[:, 0, 1] * reduced[:, 1, 0]
    if np.any(det <= 0.0):
        raise ValueError("every Jacobian must have a positive determinant")

    kappa_0 = float(background_conductivity)
    a11, a12 = reduced[:, 0, 0], reduced[:, 0, 1]
    a21, a22 = reduced[:, 1, 0], reduced[:, 1, 1]
    p = kappa_0 * (a11 * a11 + a12 * a12) / det
    r = kappa_0 * (a21 * a21 + a22 * a22) / det
    q = kappa_0 * (a11 * a21 + a12 * a22) / det
    centre = 0.5 * (p + r)
    radius = np.hypot(0.5 * (p - r), q)
    large = centre + radius
    # det(K_D) = kappa_0**2 avoids subtracting nearly equal eigenvalues.
    small = kappa_0 * (kappa_0 / large)
    theta = np.mod(0.5 * np.arctan2(2.0 * q, p - r), np.pi)
    theta = np.where(theta >= np.pi, theta - np.pi, theta)
    isotropic = 2.0 * radius <= 1e-9 * (2.0 * centre)
    theta = np.where(isotropic, 0.0, theta)

    # Solve the weighted arithmetic/harmonic matching problem. Normalize
    # by the larger eigenvalue and rationalize the low-material root.
    ratio = small / large
    gap = np.maximum(1.0 - ratio, 0.0)
    disc = np.sqrt(gap * (gap + 4.0 * f * (1.0 - f) * ratio))
    first = large * (1.0 - ratio * (1.0 - 2.0 * f) + disc) / (2.0 * f)
    second = small * (2.0 * (1.0 - f)) / (1.0 - ratio * (2.0 * f - 1.0) + disc)
    first = np.where(isotropic, centre, first)
    second = np.where(isotropic, centre, second)
    return np.column_stack([first, second, theta])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return independent numerical test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _duality(scale, gamma, kp, k0, phi0):\n"
        "    out = []\n"
        "    for s, g, k in zip(scale, gamma, kp):\n"
        "        r = np.array([[np.cos(g), -np.sin(g)], [np.sin(g), np.cos(g)]])\n"
        "        q = np.array([[np.cos(phi0), -np.sin(phi0)], [np.sin(phi0), np.cos(phi0)]])\n"
        "        out.append(s * r @ np.diag([1.0, k0 / k]) @ q.T)\n"
        "    return np.array(out)\n"
        "def _lsig(out, e):\n"
        "    out = np.asarray(out, dtype=float)\n"
        "    if out.shape != (e, 3):\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(1, e + 1, dtype=float))\n"
        "    return float(np.sum(out[:, 0] * w) + np.sum(out[:, 1] * w[::-1])\n"
        "                 + np.sum(np.cos(2.0 * out[:, 2]) * w)\n"
        "                 + np.sum(np.sin(2.0 * out[:, 2]) * w[::-1]))\n"
        "SC = np.array([0.7, 1.3, 2.2, 0.4, 1.0, 3.1])\n"
        "GA = np.array([0.3, 1.1, 2.0, 2.9, -0.7, 4.0])\n"
        "KP = np.array([2.4, 0.3, 1.0, 0.05, 5.0, 0.8])\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    cases = [
        {
            "setup": helpers + "J = _duality(SC, GA, KP, 1.0, 0.0)\n",
            "call": "_lsig(realize_duality_laminate(J.copy(), 1.0), 6)",
            "gold_call": "_lsig(_oracle_realize_duality_laminate(J.copy(), 1.0), 6)",
        },
        {
            "setup": helpers + "J = _duality(SC, GA, KP, 1.0, 0.0)\n",
            "call": "float(realize_duality_laminate(J.copy(), 1.0)[1, 2])",
            "gold_call": "float(_oracle_realize_duality_laminate(J.copy(), 1.0)[1, 2])",
        },
        {
            "setup": helpers + "J = _duality(SC, GA + 0.25, KP, 1.0, -0.6)\n",
            "call": "float(np.sum(realize_duality_laminate(J.copy(), 1.0)[:, 2]))",
            "gold_call": "float(np.sum(_oracle_realize_duality_laminate(J.copy(), 1.0)[:, 2]))",
        },
        {
            "setup": helpers + "J = _duality(SC, GA, KP * 2.0, 2.0, 0.9)\n",
            "call": "_lsig(realize_duality_laminate(J.copy(), 2.0), 6)",
            "gold_call": "_lsig(_oracle_realize_duality_laminate(J.copy(), 2.0), 6)",
        },
        {
            "setup": helpers + "J = _duality(SC, GA, KP, 1.0, 0.0)\n",
            "call": "float(1e3 * realize_duality_laminate(J.copy(), 1.0)[3, 1])",
            "gold_call": "float(1e3 * _oracle_realize_duality_laminate(J.copy(), 1.0)[3, 1])",
        },
        {
            "setup": helpers + (
                "J = np.array([[[1.3, -0.4], [0.9, 0.8]], [[0.5, 0.2], [-0.1, 2.0]],\n"
                "              [[2.0, 1.5], [0.3, 1.1]], [[0.9, 0.0], [0.0, 0.9]]])\n"
            ),
            "call": "_lsig(realize_duality_laminate(J.copy(), 1.5), 4)",
            "gold_call": "_lsig(_oracle_realize_duality_laminate(J.copy(), 1.5), 4)",
        },
        {
            "setup": helpers + (
                "J = _duality(SC, GA, np.full(6, 1.4), 1.4, 0.4)\n"
                "def _iso(fn):\n"
                "    out = np.asarray(fn(J.copy(), 1.4), dtype=float)\n"
                "    if out.shape != (6, 3):\n"
                "        return -1.0\n"
                "    return float(np.sum(out[:, 0]) + 2.0 * np.sum(out[:, 1]) + 10.0 * np.sum(np.abs(out[:, 2])))\n"
            ),
            "call": "_iso(realize_duality_laminate)",
            "gold_call": "_iso(_oracle_realize_duality_laminate)",
        },
        {
            "setup": helpers + status + (
                "J = _duality(SC, GA, KP, 1.0, 0.0)\n"
                "J[2] = J[2][:, ::-1]\n"
            ),
            "call": "_status(lambda: realize_duality_laminate(J.copy(), 1.0))",
            "gold_call": "_status(lambda: _oracle_realize_duality_laminate(J.copy(), 1.0))",
        },
    ]
    weighted = helpers + """
def _material_values(out, e):
    out = np.asarray(out, dtype=float)
    if out.shape != (e, 3) or not np.all(np.isfinite(out)):
        raise AssertionError('expected finite (e, 3) laminate')
    if np.any(out[:, 1] <= 0) or np.any(out[:, 0] < out[:, 1]):
        raise AssertionError('materials must satisfy a >= b > 0')
    if np.any((out[:, 2] < 0) | (out[:, 2] >= np.pi)):
        raise AssertionError('layer direction must be in [0, pi)')
    return np.column_stack([np.log(out[:, 0]), np.log(out[:, 1]),
                            np.cos(2*out[:, 2]), np.sin(2*out[:, 2])]).ravel()
"""
    for fraction in [0.08, 0.83]:
        setup = weighted + f"F = {fraction!r}\nJ = _duality(SC, GA, KP, 1.0, 0.7)\n"
        cases.append({"setup": setup,
                      "call": "_material_values(realize_duality_laminate(J.copy(), 1.0, F), 6)",
                      "gold_call": "_material_values(_oracle_realize_duality_laminate(J.copy(), 1.0, F), 6)",
                      "tol": 2e-7})
    cases.append({
        "setup": weighted + "F = np.array([0.03, 0.9, 0.2, 0.75, 0.4, 0.6])\nJ = _duality(SC, GA, KP, 1.0, -1.2)\n",
        "call": "_material_values(realize_duality_laminate(J.copy(), 1.0, F.copy()), 6)",
        "gold_call": "_material_values(_oracle_realize_duality_laminate(J.copy(), 1.0, F.copy()), 6)",
        "tol": 2e-7,
    })
    cases.append({
        "setup": weighted + "F = np.array([0.05, 0.2, 0.9, 0.6, 0.7, 0.02])\nKC = np.array([1e-5, 1e4, 0.1, 5.0, 1.0, 2.0])\nJ = _duality(np.array([1e-100, 1e100, 0.01, 1e40, 1e-50, 3.0]), GA, KC, 1.0, 0.0)\n",
        "call": "_material_values(realize_duality_laminate(J.copy(), 1.0, F.copy()), 6)",
        "gold_call": "_material_values(_oracle_realize_duality_laminate(J.copy(), 1.0, F.copy()), 6)",
        "tol": 2e-7,
    })
    for invalid in ['0.0', '1.0', 'np.full((6, 1), 0.4)']:
        cases.append({
            "setup": weighted + status + "J = _duality(SC, GA, KP, 1.0, 0.0)\n",
            "call": f"_status(lambda: realize_duality_laminate(J.copy(), 1.0, {invalid}))",
            "gold_call": f"_status(lambda: _oracle_realize_duality_laminate(J.copy(), 1.0, {invalid}))",
        })
    return cases

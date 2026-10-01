"""
Construct, triangle by triangle, the Jacobian of the coordinate transformation that is fixed by the unidirectional design's own conductivity and temperature field.

In the physical-geometric duality the fields of a unidirectional isotropic design are read as the output of a coordinate transformation of the homogeneous background, so the local Jacobian is dictated by the design's temperature gradient and conductivity rather than by an analytic mapping.

Returns
-------
np.ndarray: transformation Jacobian of every triangle, shape (e, 2, 2).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_duality_jacobian(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    temperature: "np.ndarray",
    conductivity_values: "np.ndarray",
    background_conductivity: float,
    design_gradient: "np.ndarray",
) -> "np.ndarray":
    r"""Return the transformation Jacobian of every triangle.

    The Jacobian maps virtual-reference coordinate increments into physical
    coordinate increments in the stated Cartesian axes, as prescribed by the
    physical-geometric duality. The local physical temperature field is the
    P1 interpolant of ``temperature`` and its local isotropic design
    conductivity is ``conductivity_values[e]``. The reference medium has
    ``background_conductivity`` and uniform temperature gradient
    ``design_gradient``.

    Parameters
    ----------
    nodes : np.ndarray
        Float array of shape (n, 2) with node coordinates.
    triangles : np.ndarray
        Integer array of shape (e, 3) with node indices; either orientation.
    temperature : np.ndarray
        Float array of shape (n,) with the design's nodal temperatures.
    conductivity_values : np.ndarray
        Float array of shape (e,) with the positive isotropic design
        conductivity of every triangle.
    background_conductivity : float
        Positive background conductivity $\kappa_0$.
    design_gradient : np.ndarray
        Float array of shape (2,), the non-zero design gradient $G_0$.

    Returns
    -------
    np.ndarray
        Float array of shape (e, 2, 2); entry [e, r, c] is row $r$, column $c$
        of $\Lambda$ for triangle $e$.

    Raises
    ------
    ValueError
        If an array has the wrong shape or a non-finite entry, if a triangle
        index is invalid or a triangle has zero area, if a conductivity is
        not positive, if ``design_gradient`` is zero, or if the local physical temperature
        gradient of some triangle is exactly zero.
    """
    return jacobians

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_duality_jacobian(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    temperature: "np.ndarray",
    conductivity_values: "np.ndarray",
    background_conductivity: float,
    design_gradient: "np.ndarray",
) -> "np.ndarray":
    """Reference implementation (closed form Lambda = lambda R S)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(nodes, dtype=float)
    tri = np.asarray(triangles)
    field = np.asarray(temperature, dtype=float)
    kappa_p = np.asarray(conductivity_values, dtype=float)
    g0 = np.asarray(design_gradient, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
        raise ValueError("nodes must be a finite (n, 2) array")
    if (tri.ndim != 2 or tri.shape[1] != 3 or tri.shape[0] == 0
            or not np.issubdtype(tri.dtype, np.integer)
            or tri.min() < 0 or tri.max() >= points.shape[0]):
        raise ValueError("triangles must be a non-empty integer (e, 3) index array")
    if field.shape != (points.shape[0],) or not np.all(np.isfinite(field)):
        raise ValueError("temperature must be a finite (n,) array")
    if (kappa_p.shape != (tri.shape[0],) or not np.all(np.isfinite(kappa_p))
            or np.any(kappa_p <= 0.0)):
        raise ValueError("conductivity_values must be a positive finite (e,) array")
    if not (_is_number(background_conductivity) and background_conductivity > 0.0):
        raise ValueError("background_conductivity must be a finite positive number")
    if g0.shape != (2,) or not np.all(np.isfinite(g0)) or not np.any(g0 != 0.0):
        raise ValueError("design_gradient must be a finite non-zero (2,) array")

    corner = points[tri]
    edge1 = corner[:, 1] - corner[:, 0]
    edge2 = corner[:, 2] - corner[:, 0]
    jac = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    if np.any(jac == 0.0):
        raise ValueError("triangles must have non-zero area")
    rise1 = field[tri[:, 1]] - field[tri[:, 0]]
    rise2 = field[tri[:, 2]] - field[tri[:, 0]]
    grad = np.column_stack([(rise1 * edge2[:, 1] - rise2 * edge1[:, 1]) / jac,
                            (rise2 * edge1[:, 0] - rise1 * edge2[:, 0]) / jac])
    grad_sq = np.sum(grad * grad, axis=1)
    if np.any(grad_sq == 0.0):
        raise ValueError("the temperature gradient vanishes on some triangle")

    # Lambda = (|G0| / |G|^2) [G u^T + (kappa_0 / kappa_P) G_perp u_perp^T],
    # i.e. lambda R S with lambda = |G0|/|G|, R the rotation onto G and
    # S = diag(1, kappa_0 / kappa_P) in the frame of the design direction u.
    g0_norm = float(np.hypot(g0[0], g0[1]))
    unit = g0 / g0_norm
    unit_perp = np.array([-unit[1], unit[0]])
    grad_perp = np.column_stack([-grad[:, 1], grad[:, 0]])
    stretch = float(background_conductivity) / kappa_p
    jacobians = (np.einsum("ei,j->eij", grad, unit)
                 + stretch[:, None, None] * np.einsum("ei,j->eij", grad_perp, unit_perp))
    return jacobians * (g0_norm / grad_sq)[:, None, None]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return independent numerical test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _grid(n):\n"
        "    axis = np.linspace(-1.0, 1.0, n)\n"
        "    gx, gy = np.meshgrid(axis, axis)\n"
        "    nodes = np.column_stack([gx.ravel(), gy.ravel()])\n"
        "    tris = []\n"
        "    for j in range(n - 1):\n"
        "        for i in range(n - 1):\n"
        "            p = j * n + i\n"
        "            tris.append([p, p + 1, p + n + 1])\n"
        "            tris.append([p, p + n + 1, p + n])\n"
        "    return nodes, np.array(tris, dtype=int)\n"
        "X, TR = _grid(7)\n"
        "T = 0.8 * X[:, 0] + 0.2 * np.sin(3.0 * X[:, 1]) + 0.1 * X[:, 0] * X[:, 1]\n"
        "C = X[TR].mean(axis=1)\n"
        "KP = np.where(np.hypot(C[:, 0], C[:, 1]) < 0.5, 0.25, 1.0)\n"
        "KP = np.where(np.hypot(C[:, 0], C[:, 1]) < 0.3, 3.0, KP)\n"
        "def _jsig(a, e):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (e, 2, 2):\n"
        "        return -1.0\n"
        "    flat = a.ravel()\n"
        "    w = np.cos(np.arange(1, flat.size + 1, dtype=float))\n"
        "    return float(np.sum(np.abs(flat)) / e + np.sum(flat * w) / 10.0)\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    design_x = "np.array([1.0, 0.0])"
    cases = [
        {
            "setup": helpers,
            "call": f"_jsig(compute_duality_jacobian(X.copy(), TR.copy(), T.copy(), KP.copy(), 1.0, {design_x}), 72)",
            "gold_call": f"_jsig(_oracle_compute_duality_jacobian(X.copy(), TR.copy(), T.copy(), KP.copy(), 1.0, {design_x}), 72)",
        },
        {
            "setup": helpers,
            "call": f"float(compute_duality_jacobian(X.copy(), TR.copy(), T.copy(), KP.copy(), 1.0, {design_x})[37, 1, 1])",
            "gold_call": f"float(_oracle_compute_duality_jacobian(X.copy(), TR.copy(), T.copy(), KP.copy(), 1.0, {design_x})[37, 1, 1])",
        },
        {
            "setup": helpers + "TN = -T\n",
            "call": "_jsig(compute_duality_jacobian(X.copy(), TR.copy(), TN.copy(), KP.copy(), 1.0, np.array([-1.0, 0.0])), 72)",
            "gold_call": "_jsig(_oracle_compute_duality_jacobian(X.copy(), TR.copy(), TN.copy(), KP.copy(), 1.0, np.array([-1.0, 0.0])), 72)",
        },
        {
            "setup": helpers + "TR2 = TR[:, ::-1].copy()\n",
            "call": "_jsig(compute_duality_jacobian(X.copy(), TR2.copy(), T.copy(), KP.copy(), 2.5, np.array([1.2, 1.6])), 72)",
            "gold_call": "_jsig(_oracle_compute_duality_jacobian(X.copy(), TR2.copy(), T.copy(), KP.copy(), 2.5, np.array([1.2, 1.6])), 72)",
        },
        {
            "setup": helpers + (
                "def _cauchy_riemann(fn):\n"
                "    a = np.asarray(fn(X.copy(), TR.copy(), T.copy(), np.full(72, 1.7), 1.7, np.array([0.0, 2.0])), dtype=float)\n"
                "    if a.shape != (72, 2, 2):\n"
                "        return -1.0\n"
                "    mismatch = np.abs(a[:, 0, 0] - a[:, 1, 1]) + np.abs(a[:, 0, 1] + a[:, 1, 0])\n"
                "    return float(1e3 * np.max(mismatch) + np.sum(a[:, 1, 0]))\n"
            ),
            "call": "_cauchy_riemann(compute_duality_jacobian)",
            "gold_call": "_cauchy_riemann(_oracle_compute_duality_jacobian)",
        },
        {
            "setup": helpers + (
                "def _det_mean(fn):\n"
                "    a = np.asarray(fn(X.copy(), TR.copy(), T.copy(), KP.copy(), 1.0, np.array([0.6, -0.8])), dtype=float)\n"
                "    if a.shape != (72, 2, 2):\n"
                "        return -1.0\n"
                "    return float(np.mean(a[:, 0, 0] * a[:, 1, 1] - a[:, 0, 1] * a[:, 1, 0]))\n"
            ),
            "call": "_det_mean(compute_duality_jacobian)",
            "gold_call": "_det_mean(_oracle_compute_duality_jacobian)",
        },
        {
            "setup": helpers + status + "TU = np.full(49, 0.3)\n",
            "call": f"_status(lambda: compute_duality_jacobian(X.copy(), TR.copy(), TU.copy(), KP.copy(), 1.0, {design_x}))",
            "gold_call": f"_status(lambda: _oracle_compute_duality_jacobian(X.copy(), TR.copy(), TU.copy(), KP.copy(), 1.0, {design_x}))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: compute_duality_jacobian(X.copy(), TR.copy(), T.copy(), KP.copy(), 1.0, np.zeros(2)))",
            "gold_call": "_status(lambda: _oracle_compute_duality_jacobian(X.copy(), TR.copy(), T.copy(), KP.copy(), 1.0, np.zeros(2)))",
        },
    ]
    full = helpers + """
xx, yy = X.T.copy()
factor = (1-xx*xx)*(1-yy*yy)
X[:,0] += .06*factor*np.sin(2*yy)
X[:,1] += .04*factor*np.cos(2*xx)
T = .7*X[:,0] - .4*X[:,1] + .08*X[:,0]*X[:,1] + 17.0
TR[::2] = TR[::2,::-1]
KP = np.exp(np.linspace(-3.,3.,len(TR)))
def _jacobian_values(j):
    j = np.asarray(j,dtype=float)
    if j.shape != (72,2,2) or not np.all(np.isfinite(j)):
        raise AssertionError('expected finite per-triangle Jacobians')
    return j.ravel()
"""
    for g0 in ['np.array([-.6, .8])', 'np.array([2.4, -3.2])']:
        expr = f"(X.copy(),TR.copy(),T.copy(),KP.copy(),1.3,{g0})"
        cases.append({"setup": full,
                      "call": "_jacobian_values(compute_duality_jacobian"+expr+")",
                      "gold_call": "_jacobian_values(_oracle_compute_duality_jacobian"+expr+")",
                      "tol": 2e-8})
    return cases

"""
Calibrate the isotropic conductivity of the outer compensation shell so that the core-shell design leaves no dipolar disturbance in the background under the design drive along $x$.

A unidirectional cloak built from isotropic materials hides an insulating shell by choosing a surrounding compensation shell that restores the external temperature field for one direction of the applied gradient, which is a physical-equivalence condition on the background field rather than a coordinate transformation.

Returns
-------
float: calibrated outer-shell conductivity.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def calibrate_compensation_shell(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    labels: "np.ndarray",
    inner_conductivity: float,
    background_conductivity: float,
    bracket: tuple[float, float],
    tolerance: float,
) -> float:
    r"""Return the outer-shell conductivity that cancels the exterior dipole residual.

    Triangles labelled 0 (background) and 3 (core) have the isotropic
    conductivity ``background_conductivity``, triangles labelled 2 (inner
    shell) have ``inner_conductivity`` and triangles labelled 1 (outer shell)
    have the isotropic trial value $k$. For a given $k$, $T$ is the steady
    temperature of ``solve_steady_conduction`` with drive angle 0, so every
    boundary node of the square is held at $T_0=x$. The exterior nodes are
    the nodes off the square's boundary all of whose incident triangles are
    labelled 0, and the residual is
    $m(k)=\operatorname{mean}\left[x\,(T-x)\right]$ over the exterior nodes.
    Use bisection on ``bracket``, stop when its width is at most
    ``tolerance``, and return its final midpoint. If a midpoint residual
    is exactly zero, retain that midpoint as the upper endpoint and continue
    until the width criterion is met.

    Parameters
    ----------
    nodes : np.ndarray
        Float array of shape (n, 2) with node coordinates in $[-1,1]^2$.
    triangles : np.ndarray
        Integer array of shape (e, 3) with node indices.
    labels : np.ndarray
        Integer array of shape (e,) with region labels 0, 1, 2 or 3.
    inner_conductivity : float
        Positive conductivity of the inner shell.
    background_conductivity : float
        Positive conductivity of the background and of the core.
    bracket : tuple[float, float]
        Positive search interval $(\mathrm{lo},\mathrm{hi})$ with $\mathrm{lo}<\mathrm{hi}$.
    tolerance : float
        Positive bracket width at which the bisection stops.

    Returns
    -------
    float
        The calibrated outer-shell conductivity.

    Raises
    ------
    ValueError
        If ``labels`` does not have shape (e,) with values in {0, 1, 2, 3},
        if a conductivity, a bracket end or ``tolerance`` is not a finite
        positive number, if $\mathrm{lo}\ge\mathrm{hi}$, if there is no exterior node, if
        $m(\mathrm{lo})$ and $m(\mathrm{hi})$ are not of strictly opposite signs, or if
        ``solve_steady_conduction`` rejects the mesh.
    """
    return outer_conductivity

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_calibrate_compensation_shell(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    labels: "np.ndarray",
    inner_conductivity: float,
    background_conductivity: float,
    bracket: tuple[float, float],
    tolerance: float,
) -> float:
    """Reference implementation (bisection over the reference solver)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(nodes, dtype=float)
    tri = np.asarray(triangles)
    region = np.asarray(labels)
    if (region.ndim != 1 or tri.ndim != 2 or region.shape[0] != tri.shape[0]
            or not np.issubdtype(region.dtype, np.integer)
            or np.any((region < 0) | (region > 3))):
        raise ValueError("labels must be an (e,) integer array with values 0 to 3")
    k_inner, k_background = inner_conductivity, background_conductivity
    if not (_is_number(k_inner) and k_inner > 0.0
            and _is_number(k_background) and k_background > 0.0):
        raise ValueError("conductivities must be finite positive numbers")
    if len(bracket) != 2 or not all(_is_number(v) and v > 0.0 for v in bracket):
        raise ValueError("bracket must hold two finite positive numbers")
    lo, hi = float(bracket[0]), float(bracket[1])
    if not lo < hi:
        raise ValueError("bracket must satisfy lo < hi")
    if not (_is_number(tolerance) and tolerance > 0.0):
        raise ValueError("tolerance must be a finite positive number")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("nodes must have shape (n, 2)")

    on_boundary = ((np.abs(points[:, 0]) >= 1.0 - 1e-12)
                   | (np.abs(points[:, 1]) >= 1.0 - 1e-12))
    touched = np.zeros(points.shape[0], dtype=bool)
    touched[tri[region != 0].ravel()] = True
    exterior = ~touched & ~on_boundary
    if not np.any(exterior):
        raise ValueError("there is no exterior node")
    x_ext = points[exterior, 0]

    def _residual(k_outer):
        values = np.array([k_background, k_outer, k_inner, k_background])[region]
        tensors = values[:, None, None] * np.eye(2)
        temperature = _oracle_solve_steady_conduction(points, tri, tensors, 0.0)
        return float(np.mean(x_ext * (temperature[exterior] - x_ext)))

    m_lo, m_hi = _residual(lo), _residual(hi)
    if not ((m_lo < 0.0 < m_hi) or (m_hi < 0.0 < m_lo)):
        raise ValueError("the residual does not change sign across the bracket")
    while hi - lo > tolerance:
        mid = 0.5 * (lo + hi)
        m_mid = _residual(mid)
        if np.sign(m_mid) == np.sign(m_lo):
            lo, m_lo = mid, m_mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return independent numerical test specifications."""
    helpers = (
        "import numpy as np\n"
        "def _device(n, a, b, c, e):\n"
        "    axis = np.linspace(-1.0, 1.0, n)\n"
        "    gx, gy = np.meshgrid(axis, axis)\n"
        "    nodes = np.column_stack([gx.ravel(), gy.ravel()])\n"
        "    tris = []\n"
        "    for j in range(n - 1):\n"
        "        for i in range(n - 1):\n"
        "            p = j * n + i\n"
        "            tris.append([p, p + 1, p + n + 1])\n"
        "            tris.append([p, p + n + 1, p + n])\n"
        "    tris = np.array(tris, dtype=int)\n"
        "    cen = nodes[tris].mean(axis=1)\n"
        "    rho = np.hypot(cen[:, 0], cen[:, 1])\n"
        "    stretch = 1.0 + e * np.cos(2.0 * np.arctan2(cen[:, 1], cen[:, 0]))\n"
        "    lab = np.zeros(len(tris), dtype=int)\n"
        "    lab[rho < c * stretch] = 1\n"
        "    lab[rho < b * stretch] = 2\n"
        "    lab[rho < a * stretch] = 3\n"
        "    return nodes, tris, lab\n"
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
            "setup": helpers + "X, TR, LAB = _device(21, 0.20, 0.32, 0.50, 0.2)\n",
            "call": "calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.1, 1.0, (1.0, 20.0), 1e-10)",
            "gold_call": "_oracle_calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.1, 1.0, (1.0, 20.0), 1e-10)",
        },
        {
            "setup": helpers + "X, TR, LAB = _device(25, 0.18, 0.28, 0.48, -0.15)\n",
            "call": "calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.5, 2.0, (2.0, 30.0), 1e-9)",
            "gold_call": "_oracle_calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.5, 2.0, (2.0, 30.0), 1e-9)",
        },
        {
            "setup": helpers + "X, TR, LAB = _device(17, 0.20, 0.34, 0.55, 0.1)\n",
            "call": "calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.05, 1.0, (1.0, 20.0), 0.3)",
            "gold_call": "_oracle_calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.05, 1.0, (1.0, 20.0), 0.3)",
        },
        {
            "setup": helpers + (
                "X, TR, LAB = _device(21, 0.20, 0.32, 0.50, 0.2)\n"
                "LAB[LAB == 3] = 0\n"
            ),
            "call": "calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.1, 1.0, (1.0, 20.0), 1e-10)",
            "gold_call": "_oracle_calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.1, 1.0, (1.0, 20.0), 1e-10)",
        },
        {
            "setup": helpers + status + "X, TR, LAB = _device(21, 0.20, 0.32, 0.50, 0.2)\n",
            "call": "_status(lambda: calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.1, 1.0, (4.0, 20.0), 1e-10))",
            "gold_call": "_status(lambda: _oracle_calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.1, 1.0, (4.0, 20.0), 1e-10))",
        },
        {
            "setup": helpers + status + (
                "X, TR, LAB = _device(9, 0.20, 0.32, 0.50, 0.2)\n"
                "LAB[LAB == 0] = 1\n"
            ),
            "call": "_status(lambda: calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.1, 1.0, (1.0, 20.0), 1e-6))",
            "gold_call": "_status(lambda: _oracle_calibrate_compensation_shell(X.copy(), TR.copy(), LAB.copy(), 0.1, 1.0, (1.0, 20.0), 1e-6))",
        },
    ]
    cases.append({
        "setup": helpers + "X,TR,LAB = _device(32,.17,.31,.52,-.28)\nX = X[:,::-1].copy()\nTR[::2] = TR[::2,::-1]\n",
        "call": "calibrate_compensation_shell(X.copy(),TR.copy(),LAB.copy(),.025,1.,(1.,30.),1e-9)",
        "gold_call": "_oracle_calibrate_compensation_shell(X.copy(),TR.copy(),LAB.copy(),.025,1.,(1.,30.),1e-9)",
        "tol": 1e-8,
    })
    return cases

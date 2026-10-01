"""
Fill every triangle with its prescribed-fraction laminate and return the largest root-mean-square background disturbance of the temperature field over all directions of the applied gradient.

A device is omnidirectional only if the background field stays undisturbed whatever the direction of the applied gradient, so its performance is the worst case over drive directions rather than the value for the direction it was designed in.

Returns
-------
float: worst-direction root-mean-square far-field disturbance.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_worst_direction_disturbance(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    laminate: "np.ndarray",
    far_threshold: float,
    high_fraction: "float | np.ndarray" = 0.5,
) -> float:
    r"""Return the largest far-field disturbance over all drive directions.

    Triangle $e$ is filled with the laminate ``laminate[e]`` $=(a,b,\theta)$:
    material $a$ occupies fraction $f$ = ``high_fraction`` of each local period
    and $b$ occupies $1-f$. Use the effective conductivity of
    this local isotropic laminate. For a drive angle $\varphi$ let $T_\varphi$ be the nodal steady
    temperature returned by ``solve_steady_conduction`` for this medium,
    whose boundary nodes are held at $T_0=x\cos\varphi+y\sin\varphi$, and let
    $D(\varphi)$ be the root mean square of $T_\varphi-T_0$ over the far nodes, the
    nodes off the square's boundary ($|x|<1-10^{-12}$ and $|y|<1-10^{-12}$)
    with $\max(|x|,|y|)>$ ``far_threshold``. Return the maximum of $D(\varphi)$ over
    all real $\varphi$.

    Parameters
    ----------
    nodes : np.ndarray
        Float array of shape (n, 2) with node coordinates in [-1, 1]^2.
    triangles : np.ndarray
        Integer array of shape (e, 3) with node indices.
    laminate : np.ndarray
        Float array of shape (e, 3) with rows $(a,b,\theta)$, $a\ge b>0$.
    far_threshold : float
        Number in $[0,1)$ selecting the far nodes.
    high_fraction : float or np.ndarray
        Scalar or shape (e,) array of finite fractions strictly between
        zero and one, specifying the share occupied by material $a$.
        The default 0.5 is the equal-thickness benchmark.

    Returns
    -------
    float
        The worst-direction root-mean-square disturbance $\max_\varphi D(\varphi)$.

    Raises
    ------
    ValueError
        If ``laminate`` is not a finite (e, 3) array with $a\ge b>0$ in every
        row, if ``far_threshold`` is not a finite number in $[0,1)$, if there
        is no far node, if high_fraction has an invalid shape or an entry
        outside $(0,1)$ or non-finite, or if ``solve_steady_conduction``
        rejects the mesh.
    """
    return disturbance

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_worst_direction_disturbance(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    laminate: "np.ndarray",
    far_threshold: float,
    high_fraction: "float | np.ndarray" = 0.5,
) -> float:
    """Reference implementation (two drives and the largest eigenvalue)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(nodes, dtype=float)
    layers = np.asarray(laminate, dtype=float)
    tri = np.asarray(triangles)
    if (layers.ndim != 2 or layers.shape[1] != 3 or tri.ndim != 2
            or layers.shape[0] != tri.shape[0] or not np.all(np.isfinite(layers))):
        raise ValueError("laminate must be a finite (e, 3) array")
    a, b, theta = layers[:, 0], layers[:, 1], layers[:, 2]
    if np.any(b <= 0.0) or np.any(a < b):
        raise ValueError("every laminate row must satisfy a >= b > 0")
    if not (_is_number(far_threshold) and 0.0 <= far_threshold < 1.0):
        raise ValueError("far_threshold must be a finite number in [0, 1)")
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("nodes must have shape (n, 2)")

    f = np.asarray(high_fraction, dtype=float)
    if f.ndim == 0:
        f = np.full(layers.shape[0], float(f))
    if (f.shape != (layers.shape[0],) or not np.all(np.isfinite(f))
            or np.any((f <= 0.0) | (f >= 1.0))):
        raise ValueError("high_fraction must be scalar or (e,) with 0 < f < 1")
    along = f * a + (1.0 - f) * b
    across = 1.0 / (f / a + (1.0 - f) / b)
    c, s = np.cos(theta), np.sin(theta)
    tensors = np.empty((layers.shape[0], 2, 2))
    tensors[:, 0, 0] = along * c * c + across * s * s
    tensors[:, 1, 1] = along * s * s + across * c * c
    tensors[:, 0, 1] = (along - across) * c * s
    tensors[:, 1, 0] = tensors[:, 0, 1]

    inside = ((np.abs(points[:, 0]) < 1.0 - 1e-12)
              & (np.abs(points[:, 1]) < 1.0 - 1e-12))
    far = inside & (np.maximum(np.abs(points[:, 0]), np.abs(points[:, 1]))
                    > far_threshold)
    if not np.any(far):
        raise ValueError("there is no far node")
    # The problem is linear in the boundary data, so T_phi = cos(phi) T_0 + sin(phi) T_90
    # and D(phi)^2 is a quadratic form whose maximum is its largest eigenvalue.
    dev_x = _oracle_solve_steady_conduction(points, tri, tensors, 0.0) - points[:, 0]
    dev_y = _oracle_solve_steady_conduction(points, tri, tensors, 0.5 * np.pi) - points[:, 1]
    dx, dy = dev_x[far], dev_y[far]
    count = float(dx.size)
    g_xx, g_yy, g_xy = dx @ dx / count, dy @ dy / count, dx @ dy / count
    largest = 0.5 * (g_xx + g_yy) + np.hypot(0.5 * (g_xx - g_yy), g_xy)
    return float(np.sqrt(max(largest, 0.0)))

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
        "X, TR = _grid(21)\n"
        "C = X[TR].mean(axis=1)\n"
        "R = np.hypot(C[:, 0], C[:, 1] * 1.4)\n"
        "E = len(TR)\n"
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
            "setup": helpers + (
                "k = np.where(R < 0.45, 4.0, 1.0)\n"
                "L = np.column_stack([k, k, np.zeros(E)])\n"
            ),
            "call": "1e2 * compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.52)",
            "gold_call": "1e2 * _oracle_compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.52)",
        },
        {
            "setup": helpers + (
                "a = np.where(R < 0.5, 6.0, 1.0)\n"
                "b = np.where(R < 0.5, 0.4, 1.0)\n"
                "L = np.column_stack([a, b, np.full(E, 0.6)])\n"
            ),
            "call": "1e2 * compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.55)",
            "gold_call": "1e2 * _oracle_compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.55)",
        },
        {
            "setup": helpers + (
                "a = np.where(R < 0.5, 3.0, 1.0)\n"
                "b = np.where(R < 0.5, 0.2, 1.0)\n"
                "th = np.mod(np.arctan2(C[:, 1], C[:, 0]) + 1.2, np.pi)\n"
                "L = np.column_stack([a, b, th])\n"
            ),
            "call": "1e2 * compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.33)",
            "gold_call": "1e2 * _oracle_compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.33)",
        },
        {
            "setup": helpers + (
                "a = np.where(R < 0.5, 2.5, 1.0)\n"
                "b = np.where(R < 0.5, 0.5, 1.0)\n"
                "L = np.column_stack([a, b, np.full(E, 2.2)])\n"
            ),
            "call": "1e2 * compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.05)",
            "gold_call": "1e2 * _oracle_compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.05)",
        },
        {
            "setup": helpers + (
                "L = np.column_stack([np.full(E, 2.0), np.full(E, 2.0), np.full(E, 0.4)])\n"
            ),
            "call": "1e3 * compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.25)",
            "gold_call": "1e3 * _oracle_compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.25)",
        },
        {
            "setup": helpers + status + (
                "L = np.column_stack([np.full(E, 0.5), np.full(E, 2.0), np.zeros(E)])\n"
            ),
            "call": "_status(lambda: compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.52))",
            "gold_call": "_status(lambda: _oracle_compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.52))",
        },
        {
            "setup": helpers + status + (
                "L = np.column_stack([np.full(E, 2.0), np.full(E, 1.0), np.zeros(E)])\n"
            ),
            "call": "_status(lambda: compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.96))",
            "gold_call": "_status(lambda: _oracle_compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.96))",
        },
    ]
    for fraction in [0.07, 0.81]:
        setup = helpers + f"F = {fraction!r}\n" + (
            "a = np.where(R < 0.55, 15.0, 1.0)\n"
            "b = np.where(R < 0.55, 0.04, 1.0)\n"
            "th = np.mod(0.6 + 0.9*C[:, 0] - 0.4*C[:, 1], np.pi)\n"
            "L = np.column_stack([a, b, th])\n")
        cases.append({"setup": setup,
                      "call": "100*compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.58, F)",
                      "gold_call": "100*_oracle_compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.58, F)",
                      "tol": 1e-7})
    cases.append({
        "setup": helpers + "F = 0.1 + 0.8*(1+np.sin(5*C[:, 0]-3*C[:, 1]))/2\na = np.where(R < 0.55, 8.0, 1.0)\nb = np.where(R < 0.55, 0.1, 1.0)\nL = np.column_stack([a,b, np.mod(C[:, 0]+C[:, 1],np.pi)])\n",
        "call": "100*compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.51, F.copy())",
        "gold_call": "100*_oracle_compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.51, F.copy())",
        "tol": 1e-7,
    })
    cases.append({
        "setup": helpers + status + "L = np.column_stack([np.full(E,2.), np.ones(E), np.zeros(E)])\n",
        "call": "_status(lambda: compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.5, 1.0))",
        "gold_call": "_status(lambda: _oracle_compute_worst_direction_disturbance(X.copy(), TR.copy(), L.copy(), 0.5, 1.0))",
    })
    return cases

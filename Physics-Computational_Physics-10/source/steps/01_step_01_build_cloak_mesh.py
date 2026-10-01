"""
Build the uniform triangular finite-element mesh of the square domain and label every triangle by the region of the freeform core-shell device that contains its centroid.

The device is a core wrapped by an inner and an outer shell whose boundaries are smooth star-shaped curves $r(\theta)$ about the origin, written as truncated Fourier series and embedded in a homogeneous background.

Returns
-------
tuple: nodes (n_nodes**2, 2), triangles (2 (n_nodes - 1)**2, 3), labels (2 (n_nodes - 1)**2,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_cloak_mesh(
    n_nodes: int,
    core_boundary: "np.ndarray",
    inner_boundary: "np.ndarray",
    outer_boundary: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    r"""Return the node coordinates, the triangles and the triangle region labels.

    The mesh covers the square $[-1,1]\times[-1,1]$ with $N$ = ``n_nodes``
    equally spaced nodes per side. Node $(i,j)$, with $i,j=0,\dots,N-1$,
    sits at $x=-1+2i/(N-1)$, $y=-1+2j/(N-1)$ and
    has index $jN+i$. Grid square $(i,j)$, with
    $i,j=0,\dots,N-2$, has index $q=j(N-1)+i$ and is
    split by its diagonal from node $(i,j)$ to node $(i+1,j+1)$ into
    triangle $2q$ with nodes $(i,j)$, $(i+1,j)$, $(i+1,j+1)$ and triangle
    $2q+1$ with nodes $(i,j)$, $(i+1,j+1)$, $(i,j+1)$, listed in that
    order.

    Each boundary array has its own shape (m, 2); the three arrays may have
    different lengths m. Row $k$ holds $(A_k,B_k)$ and the
    boundary radius at polar angle $\theta$ is
    $r(\theta)=\sum_k\left[A_k\cos(k\theta)+B_k\sin(k\theta)\right]$. With $\rho$ and
    $\theta$ the polar radius and angle of a triangle's centroid (the mean of
    its three nodes), the label is 3 (core) if $\rho<r_{\mathrm{core}}(\theta)$,
    otherwise 2 (inner shell) if $\rho<r_{\mathrm{inner}}(\theta)$, otherwise 1 (outer
    shell) if $\rho<r_{\mathrm{outer}}(\theta)$, otherwise 0 (background).

    Parameters
    ----------
    n_nodes : int
        Number of nodes per side of the square, at least 3.
    core_boundary : np.ndarray
        Fourier coefficients of the core boundary, shape (m, 2).
    inner_boundary : np.ndarray
        Fourier coefficients of the outer edge of the inner shell, shape (m, 2).
    outer_boundary : np.ndarray
        Fourier coefficients of the outer edge of the outer shell, shape (m, 2).

    Returns
    -------
    nodes : np.ndarray
        Float array of shape (n_nodes**2, 2) with the node coordinates.
    triangles : np.ndarray
        Integer array of shape (2 (n_nodes - 1)**2, 3) with node indices.
    labels : np.ndarray
        Integer array of shape (2 (n_nodes - 1)**2,) with values 0 to 3.

    Raises
    ------
    ValueError
        If ``n_nodes`` is not an integer of at least 3, if a boundary is not
        a non-empty finite array of shape (m, 2), or if at the centroid
        angle of some triangle the radii fail
        $0<r_{\mathrm{core}}<r_{\mathrm{inner}}<r_{\mathrm{outer}}$.
    """
    return nodes, triangles, labels

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_build_cloak_mesh(
    n_nodes: int,
    core_boundary: "np.ndarray",
    inner_boundary: "np.ndarray",
    outer_boundary: "np.ndarray",
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray"]:
    """Reference implementation (vectorised centroid classification)."""
    import numpy as np

    def _is_integer(value):
        return isinstance(value, (int, np.integer)) and not isinstance(value, bool)

    def _radius(coefficients, theta):
        order = np.arange(coefficients.shape[0], dtype=float)[:, None]
        return (coefficients[:, :1] * np.cos(order * theta)
                + coefficients[:, 1:] * np.sin(order * theta)).sum(axis=0)

    if not (_is_integer(n_nodes) and n_nodes >= 3):
        raise ValueError("n_nodes must be an integer of at least 3")
    curves = []
    for boundary in (core_boundary, inner_boundary, outer_boundary):
        coefficients = np.asarray(boundary, dtype=float)
        if (coefficients.ndim != 2 or coefficients.shape[1] != 2
                or coefficients.shape[0] == 0
                or not np.all(np.isfinite(coefficients))):
            raise ValueError("each boundary must be a non-empty finite (m, 2) array")
        curves.append(coefficients)

    n = int(n_nodes)
    axis = np.linspace(-1.0, 1.0, n)
    grid_x, grid_y = np.meshgrid(axis, axis)
    nodes = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    col, row = np.meshgrid(np.arange(n - 1), np.arange(n - 1))
    corner = (row * n + col).ravel()
    triangles = np.empty((2 * corner.size, 3), dtype=int)
    triangles[0::2] = np.column_stack([corner, corner + 1, corner + n + 1])
    triangles[1::2] = np.column_stack([corner, corner + n + 1, corner + n])

    centroid = nodes[triangles].mean(axis=1)
    rho = np.hypot(centroid[:, 0], centroid[:, 1])
    theta = np.arctan2(centroid[:, 1], centroid[:, 0])
    r_core, r_inner, r_outer = (_radius(curve, theta) for curve in curves)
    if not (np.all(r_core > 0.0) and np.all(r_core < r_inner)
            and np.all(r_inner < r_outer)):
        raise ValueError("the boundaries must satisfy 0 < r_core < r_inner < r_outer")
    labels = np.zeros(triangles.shape[0], dtype=int)
    labels[rho < r_outer] = 1
    labels[rho < r_inner] = 2
    labels[rho < r_core] = 3
    return nodes, triangles, labels

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return independent numerical test specifications."""
    helpers = (
        "import numpy as np\n"
        "CORE = np.array([[0.20, 0.0], [0.0, 0.0], [0.04, 0.0]])\n"
        "INNER = np.array([[0.30, 0.0], [0.0, 0.0], [0.06, 0.0], [0.0, 0.0], [0.015, 0.0]])\n"
        "OUTER = np.array([[0.45, 0.0], [0.0, 0.0], [0.09, 0.0], [0.02, 0.0]])\n"
        "def _mesh_sig(out, n):\n"
        "    nodes, tris, labels = (np.asarray(a) for a in out)\n"
        "    e = 2 * (n - 1) ** 2\n"
        "    if nodes.shape != (n * n, 2) or tris.shape != (e, 3) or labels.shape != (e,):\n"
        "        return -1.0\n"
        "    wn = np.cos(np.arange(1, nodes.size + 1, dtype=float))\n"
        "    wt = np.cos(np.arange(1, tris.size + 1, dtype=float))\n"
        "    wl = np.cos(np.arange(1, labels.size + 1, dtype=float))\n"
        "    return float(np.sum(nodes.ravel() * wn) + np.sum(tris.ravel() * wt) / 100.0\n"
        "                 + np.sum(labels * wl) + np.sum(labels) / 10.0)\n"
        "def _label_sig(out, n):\n"
        "    labels = np.asarray(out[2])\n"
        "    if labels.shape != (2 * (n - 1) ** 2,):\n"
        "        return -1.0\n"
        "    counts = np.bincount(labels.astype(int), minlength=4)\n"
        "    w = np.cos(np.arange(1, labels.size + 1, dtype=float))\n"
        "    return float((counts[1] + 10.0 * counts[2] + 100.0 * counts[3]\n"
        "                  + np.sum(labels * w)) / 1000.0)\n"
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
            "setup": helpers,
            "call": "_mesh_sig(build_cloak_mesh(9, CORE.copy(), INNER.copy(), OUTER.copy()), 9)",
            "gold_call": "_mesh_sig(_oracle_build_cloak_mesh(9, CORE.copy(), INNER.copy(), OUTER.copy()), 9)",
        },
        {
            "setup": helpers,
            "call": "_label_sig(build_cloak_mesh(41, CORE.copy(), INNER.copy(), OUTER.copy()), 41)",
            "gold_call": "_label_sig(_oracle_build_cloak_mesh(41, CORE.copy(), INNER.copy(), OUTER.copy()), 41)",
        },
        {
            "setup": helpers,
            "call": "float(build_cloak_mesh(41, CORE.copy(), INNER.copy(), OUTER.copy())[1][1523, 2])",
            "gold_call": "float(_oracle_build_cloak_mesh(41, CORE.copy(), INNER.copy(), OUTER.copy())[1][1523, 2])",
        },
        {
            "setup": helpers + (
                "C2 = np.array([[0.25, 0.0], [0.05, 0.06]])\n"
                "I2 = np.array([[0.40, 0.0], [0.08, 0.09], [0.0, -0.03]])\n"
                "O2 = np.array([[0.62, 0.0], [0.10, 0.12], [0.03, 0.0], [0.0, 0.04]])\n"
            ),
            "call": "_label_sig(build_cloak_mesh(33, C2.copy(), I2.copy(), O2.copy()), 33)",
            "gold_call": "_label_sig(_oracle_build_cloak_mesh(33, C2.copy(), I2.copy(), O2.copy()), 33)",
        },
        {
            "setup": helpers + (
                "C3 = np.array([[0.05, 0.0]])\n"
                "I3 = np.array([[0.30, 0.0]])\n"
                "O3 = np.array([[0.90, 0.0]])\n"
            ),
            "call": "_mesh_sig(build_cloak_mesh(3, C3.copy(), I3.copy(), O3.copy()), 3)",
            "gold_call": "_mesh_sig(_oracle_build_cloak_mesh(3, C3.copy(), I3.copy(), O3.copy()), 3)",
        },
        {
            "setup": helpers,
            "call": "float(build_cloak_mesh(21, CORE.copy(), INNER.copy(), OUTER.copy())[0][237, 1])",
            "gold_call": "float(_oracle_build_cloak_mesh(21, CORE.copy(), INNER.copy(), OUTER.copy())[0][237, 1])",
        },
        {
            "setup": helpers + status + (
                "BAD = np.array([[0.32, 0.0], [0.0, 0.0], [0.06, 0.0]])\n"
            ),
            "call": "_status(lambda: build_cloak_mesh(21, BAD.copy(), INNER.copy(), OUTER.copy()))",
            "gold_call": "_status(lambda: _oracle_build_cloak_mesh(21, BAD.copy(), INNER.copy(), OUTER.copy()))",
        },
        {
            "setup": helpers + status,
            "call": "_status(lambda: build_cloak_mesh(2, CORE.copy(), INNER.copy(), OUTER.copy()))",
            "gold_call": "_status(lambda: _oracle_build_cloak_mesh(2, CORE.copy(), INNER.copy(), OUTER.copy()))",
        },
    ]
    packed = helpers + """
def _mesh_values(out, n):
    points, tri, labels = (np.asarray(v) for v in out)
    e = 2*(n-1)**2
    if points.shape != (n*n,2) or tri.shape != (e,3) or labels.shape != (e,):
        raise AssertionError('incorrect mesh output shapes')
    return np.concatenate([points.ravel(), tri.ravel(), labels])
"""
    cases.append({
        "setup": packed + "C = np.array([[.14,0],[.02,-.01],[.01,.02]])\nI = np.array([[.29,0],[.02,-.01],[.035,.02],[.01,-.02]])\nO = np.array([[.53,0],[.04,-.03],[.035,.02],[.02,.025],[0,-.01]])\n",
        "call": "_mesh_values(build_cloak_mesh(24,C.copy(),I.copy(),O.copy()),24)",
        "gold_call": "_mesh_values(_oracle_build_cloak_mesh(24,C.copy(),I.copy(),O.copy()),24)",
        "tol": 1e-9,
    })
    return cases

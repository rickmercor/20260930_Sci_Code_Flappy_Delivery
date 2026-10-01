"""
Solve two-dimensional steady heat conduction with piecewise-constant anisotropic conductivity by linear finite elements under a uniform applied temperature gradient imposed on the boundary of the square.

Steady temperature fields in heterogeneous thermal media depend on local conductivity and the imposed boundary temperatures.

Returns
-------
np.ndarray: nodal temperatures, shape (n,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_steady_conduction(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    conductivity: "np.ndarray",
    drive_angle: float,
) -> "np.ndarray":
    r"""Return the nodal steady temperature of the linear finite-element problem.

    The temperature is continuous and linear on every triangle (P1 Lagrange
    elements). Triangle $e$ carries the constant conductivity tensor
    ``conductivity[e]``. The nodal temperatures satisfy the P1 Galerkin discretization of
    the steady conduction problem at every nonboundary node. Every boundary node, one with $|x|\ge 1-10^{-12}$ or
    $|y|\ge 1-10^{-12}$, is held at $T_0=x\cos\varphi+y\sin\varphi$, where $\varphi$ is ``drive_angle``.
    Triangles may be listed in either orientation. Solve the linear system
    with a direct sparse solver.

    Parameters
    ----------
    nodes : np.ndarray
        Float array of shape (n, 2) with node coordinates inside the square.
    triangles : np.ndarray
        Integer array of shape (e, 3) with node indices.
    conductivity : np.ndarray
        Float array of shape (e, 2, 2): the symmetric positive-definite
        conductivity tensor of every triangle.
    drive_angle : float
        Direction $\varphi$ (radians) of the applied unit temperature gradient.

    Returns
    -------
    temperature : np.ndarray
        Float array of shape (n,) with the nodal temperatures.

    Raises
    ------
    ValueError
        If ``nodes`` is not a finite (n, 2) array with $n\ge 3$, if
        ``triangles`` is not a non-empty integer (e, 3) array of valid node
        indices, if a triangle has zero area, if ``conductivity`` does not
        have shape (e, 2, 2), holds a non-finite entry, is not symmetric
        ($|K_{01}-K_{10}|>10^{-12}\max(1,\max|K|)$) or is not positive
        definite, or if ``drive_angle`` is not a finite number.
    """
    return temperature

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_solve_steady_conduction(
    nodes: "np.ndarray",
    triangles: "np.ndarray",
    conductivity: "np.ndarray",
    drive_angle: float,
) -> "np.ndarray":
    """Reference implementation (vectorised P1 assembly, sparse LU)."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    points = np.asarray(nodes, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("nodes must have shape (n, 2) with n >= 3")
    if not np.all(np.isfinite(points)):
        raise ValueError("nodes must be finite")
    tri = np.asarray(triangles)
    if (tri.ndim != 2 or tri.shape[1] != 3 or tri.shape[0] == 0
            or not np.issubdtype(tri.dtype, np.integer)):
        raise ValueError("triangles must be a non-empty integer (e, 3) array")
    if tri.min() < 0 or tri.max() >= points.shape[0]:
        raise ValueError("triangle node index out of range")
    kappa = np.asarray(conductivity, dtype=float)
    if kappa.shape != (tri.shape[0], 2, 2) or not np.all(np.isfinite(kappa)):
        raise ValueError("conductivity must be a finite (e, 2, 2) array")
    scale = np.maximum(1.0, np.abs(kappa).max(axis=(1, 2)))
    if np.any(np.abs(kappa[:, 0, 1] - kappa[:, 1, 0]) > 1e-12 * scale):
        raise ValueError("conductivity tensors must be symmetric")
    determinant = kappa[:, 0, 0] * kappa[:, 1, 1] - kappa[:, 0, 1] * kappa[:, 1, 0]
    if np.any(kappa[:, 0, 0] <= 0.0) or np.any(determinant <= 0.0):
        raise ValueError("conductivity tensors must be positive definite")
    if not _is_number(drive_angle):
        raise ValueError("drive_angle must be a finite number")

    corner = points[tri]
    edge1 = corner[:, 1] - corner[:, 0]
    edge2 = corner[:, 2] - corner[:, 0]
    jac = edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0]
    if np.any(jac == 0.0):
        raise ValueError("triangles must have non-zero area")
    grad = np.empty((tri.shape[0], 2, 3))
    grad[:, 0, 1] = edge2[:, 1] / jac
    grad[:, 1, 1] = -edge2[:, 0] / jac
    grad[:, 0, 2] = -edge1[:, 1] / jac
    grad[:, 1, 2] = edge1[:, 0] / jac
    grad[:, :, 0] = -grad[:, :, 1] - grad[:, :, 2]
    area = 0.5 * np.abs(jac)
    local = np.einsum("eai,eab,ebj->eij", grad, kappa, grad) * area[:, None, None]
    size = points.shape[0]
    rows = np.repeat(tri, 3, axis=1).ravel()
    cols = np.tile(tri, (1, 3)).ravel()
    stiffness = sp.csr_matrix((local.ravel(), (rows, cols)), shape=(size, size))

    on_boundary = ((np.abs(points[:, 0]) >= 1.0 - 1e-12)
                   | (np.abs(points[:, 1]) >= 1.0 - 1e-12))
    temperature = (np.cos(drive_angle) * points[:, 0]
                   + np.sin(drive_angle) * points[:, 1])
    free = ~on_boundary
    if np.any(free):
        block = stiffness[free][:, free].tocsc()
        rhs = -(stiffness[free][:, on_boundary] @ temperature[on_boundary])
        temperature[free] = spla.spsolve(block, rhs)
    return temperature

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
        "def _field(nodes, tris, seed):\n"
        "    c = nodes[tris].mean(axis=1)\n"
        "    kxx = 1.0 + 0.6 * np.sin(3.0 * c[:, 0] + seed) ** 2\n"
        "    kyy = 0.5 + 0.8 * np.cos(2.0 * c[:, 1] - seed) ** 2\n"
        "    kxy = 0.3 * np.sin(c[:, 0] * c[:, 1] * 5.0 + seed)\n"
        "    K = np.empty((len(tris), 2, 2))\n"
        "    K[:, 0, 0] = kxx; K[:, 1, 1] = kyy\n"
        "    K[:, 0, 1] = kxy; K[:, 1, 0] = kxy\n"
        "    return K\n"
        "def _tsig(t, n):\n"
        "    t = np.asarray(t, dtype=float)\n"
        "    if t.shape != (n,):\n"
        "        return -1.0\n"
        "    w = np.cos(np.arange(1, n + 1, dtype=float))\n"
        "    return float(np.sum(np.abs(t)) / n + np.sum(t * w))\n"
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
            "setup": helpers + "X, TR = _grid(9)\nK = _field(X, TR, 0.4)\n",
            "call": "_tsig(solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 0.7), 81)",
            "gold_call": "_tsig(_oracle_solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 0.7), 81)",
        },
        {
            "setup": helpers + "X, TR = _grid(13)\nK = _field(X, TR, 1.9)\n",
            "call": "float(solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 2.3)[84])",
            "gold_call": "float(_oracle_solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 2.3)[84])",
        },
        {
            "setup": helpers + (
                "X, TR = _grid(3)\n"
                "K = np.array([[[2.0, 0.5], [0.5, 1.0]]] * 8)\n"
                "K[1] = [[5.0, -1.0], [-1.0, 3.0]]\n"
                "K[6] = [[0.2, 0.0], [0.0, 4.0]]\n"
            ),
            "call": "float(solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 0.3)[4])",
            "gold_call": "float(_oracle_solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 0.3)[4])",
        },
        {
            "setup": helpers + (
                "X, TR = _grid(11)\n"
                "c = X[TR].mean(axis=1)\n"
                "k = np.where(np.hypot(c[:, 0], c[:, 1]) < 0.5, 8.0, 1.0)\n"
                "K = k[:, None, None] * np.eye(2)\n"
            ),
            "call": "_tsig(solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 0.0), 121)",
            "gold_call": "_tsig(_oracle_solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 0.0), 121)",
        },
        {
            "setup": helpers + (
                "X, TR = _grid(9)\n"
                "TR = TR[:, ::-1].copy()\n"
                "K = _field(X, TR, -0.8)\n"
            ),
            "call": "_tsig(solve_steady_conduction(X.copy(), TR.copy(), K.copy(), -1.1), 81)",
            "gold_call": "_tsig(_oracle_solve_steady_conduction(X.copy(), TR.copy(), K.copy(), -1.1), 81)",
        },
        {
            "setup": helpers + (
                "X, TR = _grid(7)\n"
                "K = np.tile(3.0 * np.eye(2), (len(TR), 1, 1))\n"
                "def _linear_error(fn):\n"
                "    t = np.asarray(fn(X.copy(), TR.copy(), K.copy(), 1.2), dtype=float)\n"
                "    exact = np.cos(1.2) * X[:, 0] + np.sin(1.2) * X[:, 1]\n"
                "    return float(1e3 * np.max(np.abs(t - exact)) + np.sum(t))\n"
            ),
            "call": "_linear_error(solve_steady_conduction)",
            "gold_call": "_linear_error(_oracle_solve_steady_conduction)",
        },
        {
            "setup": helpers + status + (
                "X, TR = _grid(5)\n"
                "K = _field(X, TR, 0.1)\n"
                "K[3, 0, 1] += 0.2\n"
            ),
            "call": "_status(lambda: solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 0.0))",
            "gold_call": "_status(lambda: _oracle_solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 0.0))",
        },
        {
            "setup": helpers + status + (
                "X, TR = _grid(5)\n"
                "K = _field(X, TR, 0.1)\n"
                "K[7] = [[1.0, 2.0], [2.0, 1.0]]\n"
            ),
            "call": "_status(lambda: solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 0.0))",
            "gold_call": "_status(lambda: _oracle_solve_steady_conduction(X.copy(), TR.copy(), K.copy(), 0.0))",
        },
    ]
    warped = helpers + """
X, TR = _grid(10)
xx, yy = X.T.copy()
envelope = (1-xx*xx)*(1-yy*yy)
X[:,0] += .055*envelope*np.sin(2.4*yy+xx)
X[:,1] += .06*envelope*np.cos(1.7*xx-yy)
CEN = X[TR].mean(axis=1)
angle = .4 + .8*CEN[:,0] - .6*CEN[:,1]
v = np.column_stack([np.cos(angle),np.sin(angle)])
w = np.column_stack([-np.sin(angle),np.cos(angle)])
large = np.where(np.hypot(CEN[:,0],CEN[:,1]) < .6, 12., 1.)
small = np.where(np.hypot(CEN[:,0],CEN[:,1]) < .6, .07, 1.)
K = large[:,None,None]*v[:,:,None]*v[:,None,:] + small[:,None,None]*w[:,:,None]*w[:,None,:]
# Mixed orientations, permuted vertex IDs, and shuffled element order.
TR[::3] = TR[::3, ::-1]
permutation = (37*np.arange(len(X))+13) % len(X)
inverse = np.argsort(permutation)
X = X[permutation]
TR = inverse[TR][::-1].copy()
K = K[::-1].copy()
def _temperatures(t):
    t = np.asarray(t,dtype=float)
    if t.shape != (100,) or not np.all(np.isfinite(t)):
        raise AssertionError('expected finite nodal temperatures')
    return t
"""
    for angle in [0.43, -1.17]:
        expr = f"(X.copy(),TR.copy(),K.copy(),{angle})"
        cases.append({"setup": warped,
                      "call": "_temperatures(solve_steady_conduction" + expr + ")",
                      "gold_call": "_temperatures(_oracle_solve_steady_conduction" + expr + ")",
                      "tol": 2e-8})
    return cases

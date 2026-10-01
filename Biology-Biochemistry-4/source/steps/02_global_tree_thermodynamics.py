"""
Evaluate the global thermodynamics of the spanning-tree ensemble of the weighted contact graph: the log partition function ln Z from Kirchhoff's matrix-tree theorem, the free energy F = -kT ln Z, the mean energy <E>, the entropy S = (<E> - F)/kT and the heat capacity C = Var(E)/(kT)^2, where the energy of a tree is the total contact length of its edges and averages run over all spanning trees with their Boltzmann weights. Use exact identities of the framework for the ensemble averages; do not sample or finite-difference.

Each spanning tree of the contact graph is a topological microstate whose weight is the product of its edge weights. The matrix-tree theorem collapses the exponentially large sum into one determinant, and the moments of the tree energy follow from the same linear algebra without enumerating a single tree.

Returns
-------
ndarray of float64, shape (5,), [ln Z, F, <E>, S, C] in angstrom units (C dimensionless, in units of k).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def global_tree_thermodynamics(L: "np.ndarray", kT: float) -> "np.ndarray":
    """Evaluate the global thermodynamics of the spanning-tree ensemble of the weighted contact graph: the log partition function ln Z from Kirchhoff's matrix-tree theorem, the free energy F = -kT ln Z, the mean energy <E>, the entropy S = (<E> - F)/kT and the heat capacity C = Var(E)/(kT)^2, where the energy of a tree is the total contact length of its edges and averages run over all spanning trees with their Boltzmann weights. Use exact identities of the framework for the ensemble averages; do not sample or finite-difference.

    Parameters
    ----------
    L : np.ndarray
        Weighted Laplacian of a connected contact graph, shape (N, N).
    kT : float
        Effective temperature in angstrom.

    Returns
    -------
    thermo : np.ndarray
        Array [ln Z, F, <E>, S, C]. Graded at tolerance 1e-9 against the exact identities: <E> = sum_e d_e P(e in T) with P(e in T) = w_e R_e, and Var(E) = sum_e d_e^2 P(e)(1 - P(e)) + sum_(e != f) d_e d_f Cov(e, f) with Cov(e, f) = -K[e, f]^2 from the edge response matrix, where the generalised inverse behind R_e and K is the grounded inverse inv(L[1:, 1:]) padded with zeros (an SVD pseudoinverse of the singular L can miss the tolerance).

    Raises
    ------
    ValueError
        If L is not a finite symmetric Laplacian of size >= 3 with zero row sums and at least one edge, kT is not positive, or the graph is disconnected.
    """
    return thermo

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.linalg import det, slogdet


def _check_positive(x, name):
    x = float(x)
    if not np.isfinite(x) or x <= 0.0:
        raise ValueError(name + " must be a finite positive number")
    return x


def _check_laplacian(L):
    L = np.asarray(L, dtype=np.float64)
    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):
        raise ValueError("L must be a finite square matrix of size >= 3")
    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):
        raise ValueError("L must be a symmetric Laplacian with zero row sums")
    return L


def _edges_from_laplacian(L):
    """Lexicographic (i < j) edge list, lengths and weights recovered from the weighted Laplacian."""
    n = L.shape[0]
    ii, jj = np.where(np.triu(L, 1) < 0.0)
    edges = list(zip(ii.tolist(), jj.tolist()))
    if not edges:
        raise ValueError("L has no edges")
    w = -L[ii, jj]
    return edges, w


def _transfer_current(L, edges, w):
    """Symmetric transfer-current matrix K_ab = sqrt(w_a w_b) Y(e_a, e_b), eqs. (8)-(9). The edge vectors chi_e are
    orthogonal to the constant null vector of L, so any generalised inverse gives the same Y as the Moore-Penrose
    pseudoinverse; the grounded inverse of the reduced Laplacian is used because it is well conditioned and exact."""
    n = L.shape[0]
    Kp = np.zeros((n, n))
    Kp[1:, 1:] = np.linalg.inv(L[1:, 1:])
    chi = np.zeros((len(edges), n))
    for a, (i, j) in enumerate(edges):
        chi[a, i] = 1.0
        chi[a, j] = -1.0
    Y = chi @ Kp @ chi.T
    return Y * np.sqrt(np.outer(w, w))


def _oracle_global_tree_thermodynamics(L: "np.ndarray", kT: float) -> "np.ndarray":
    """Eqs. (3)-(6) and 2.1.7: ln Z from the reduced Laplacian; <E> and Var(E) over the spanning-tree ensemble
    through the marginal inclusion P(e in T) = w_e R_e and the transfer-current covariances Cov(1_e, 1_f) = -K_ef^2."""
    L = _check_laplacian(L)
    kT = _check_positive(kT, "kT")
    edges, w = _edges_from_laplacian(L)
    sign, lnZ = slogdet(L[1:, 1:])
    if sign <= 0:
        raise ValueError("the contact graph is disconnected")
    d = -kT * np.log(w)                                   # edge lengths recovered from eq. (1)
    K = _transfer_current(L, edges, w)
    p = np.diag(K)
    cov = -K ** 2
    np.fill_diagonal(cov, p * (1.0 - p))
    U = float(d @ p)
    varE = float(d @ cov @ d)
    F = -kT * lnZ
    S = (U - F) / kT
    C = varE / kT ** 2
    return np.array([lnZ, F, U, S, C], dtype=np.float64)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT = 7.8, 1.0\nL = _oracle_contact_laplacian(X, r_c, kT)\n",
            "call": "np.asarray(global_tree_thermodynamics(L, kT))",
            "gold_call": "np.asarray(_oracle_global_tree_thermodynamics(L, kT))",
        },
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT = 7.0, 1.2\nL = _oracle_contact_laplacian(X, r_c, kT)\n",
            "call": "np.asarray(global_tree_thermodynamics(L, kT))",
            "gold_call": "np.asarray(_oracle_global_tree_thermodynamics(L, kT))",
        },
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[0.0, 0.0, 0.0], [3.8, 0.0, 0.0], [5.5, 3.4, 0.0], [3.8, 6.8, 0.0], [0.0, 6.8, 0.5], [-2.5, 3.4, 1.0], [1.9, 3.4, 4.0]])\nr_c, kT = 5.5, 0.8\nL = _oracle_contact_laplacian(X, r_c, kT)\n",
            "call": "np.asarray(global_tree_thermodynamics(L, kT))",
            "gold_call": "np.asarray(_oracle_global_tree_thermodynamics(L, kT))",
        },
        {
            "setup": "import numpy as np\nimport numpy as np\ndef _fx_check_laplacian(L):\n    L = np.asarray(L, dtype=np.float64)\n    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):\n        raise ValueError(\"L must be a finite square matrix of size >= 3\")\n    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):\n        raise ValueError(\"L must be a symmetric Laplacian with zero row sums\")\n    return L\ndef _fx_check_positive(x, name):\n    x = float(x)\n    if not np.isfinite(x) or x <= 0.0:\n        raise ValueError(name + \" must be a finite positive number\")\n    return x\ndef _fx_edges_from_laplacian(L):\n    n = L.shape[0]\n    ii, jj = np.where(np.triu(L, 1) < 0.0)\n    edges = list(zip(ii.tolist(), jj.tolist()))\n    if not edges:\n        raise ValueError(\"L has no edges\")\n    w = -L[ii, jj]\n    return edges, w\ndef _fx_transfer_current(L, edges, w):\n    n = L.shape[0]\n    Kp = np.zeros((n, n))\n    Kp[1:, 1:] = np.linalg.inv(L[1:, 1:])\n    chi = np.zeros((len(edges), n))\n    for a, (i, j) in enumerate(edges):\n        chi[a, i] = 1.0\n        chi[a, j] = -1.0\n    Y = chi @ Kp @ chi.T\n    return Y * np.sqrt(np.outer(w, w))\ndef _fx_contact_laplacian(coords, r_c, kT):\n    X = np.asarray(coords, dtype=np.float64)\n    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 3 or not np.all(np.isfinite(X)):\n        raise ValueError(\"coords must be a finite (N, 3) array with N >= 3\")\n    r_c, kT = float(r_c), float(kT)\n    if not (np.isfinite(r_c) and r_c > 0.0 and np.isfinite(kT) and kT > 0.0):\n        raise ValueError(\"r_c and kT must be finite positive numbers\")\n    n = X.shape[0]\n    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))\n    W = np.where((D <= r_c) & ~np.eye(n, dtype=bool), np.exp(-D / kT), 0.0)\n    L = np.diag(W.sum(axis=1)) - W\n    return L\ndef _fx_edge_transfer_current(L, kT):\n    L = np.asarray(L, dtype=np.float64)\n    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):\n        raise ValueError(\"L must be a finite square matrix of size >= 3\")\n    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):\n        raise ValueError(\"L must be a symmetric Laplacian with zero row sums\")\n    if not (np.isfinite(float(kT)) and float(kT) > 0.0):\n        raise ValueError(\"kT must be a finite positive number\")\n    edges, w = _fx_edges_from_laplacian(L)\n    return _fx_transfer_current(L, edges, w)\ndef _fx_channel_paths(L, s, t, max_nodes):\n    L = _fx_check_laplacian(L)\n    n = L.shape[0]\n    for name, v in ((\"s\", s), (\"t\", t), (\"max_nodes\", max_nodes)):\n        if isinstance(v, bool) or int(v) != v:\n            raise ValueError(name + \" must be an integer\")\n    s, t, max_nodes = int(s), int(t), int(max_nodes)\n    if not (0 <= s < n and 0 <= t < n) or s == t or max_nodes < 2:\n        raise ValueError(\"s and t must be distinct residue indices and max_nodes >= 2\")\n    adj = [np.where(L[v] < 0.0)[0].tolist() for v in range(n)]\n    out = []\n    stack = [(s, (s,))]\n    while stack:\n        v, path = stack.pop()\n        if v == t:\n            out.append(path)\n            continue\n        if len(path) >= max_nodes:\n            continue\n        for u in adj[v]:\n            if u not in path:\n                stack.append((u, path + (u,)))\n    out.sort()\n    P = -np.ones((len(out), max_nodes), dtype=np.int64)\n    for r, path in enumerate(out):\n        P[r, :len(path)] = path\n    return P\ndef _fx_path_energy_table(L, kT, K, paths):\n    L = _fx_check_laplacian(L)\n    kT = _fx_check_positive(kT, \"kT\")\n    edges, w = _fx_edges_from_laplacian(L)\n    K = np.asarray(K, dtype=np.float64)\n    P = np.asarray(paths)\n    if K.shape != (len(edges), len(edges)) or P.ndim != 2 or P.shape[0] == 0:\n        raise ValueError(\"K must be E x E for the edges of L and paths must be a non-empty (m, max_nodes) array\")\n    d = -kT * np.log(w)\n    eidx = {}\n    for a, (i, j) in enumerate(edges):\n        eidx[(i, j)] = a\n        eidx[(j, i)] = a\n    out = np.zeros((P.shape[0], 3), dtype=np.float64)\n    for r in range(P.shape[0]):\n        nodes = [int(v) for v in P[r] if v >= 0]\n        if len(nodes) < 2:\n            raise ValueError(\"every path needs at least two nodes\")\n        try:\n            ea = [eidx[(nodes[k], nodes[k + 1])] for k in range(len(nodes) - 1)]\n        except KeyError:\n            raise ValueError(\"path uses a pair that is not an edge of L\")\n        Kb = K[np.ix_(ea, ea)]\n        Yb = Kb / np.sqrt(np.outer(w[ea], w[ea]))\n        out[r, 0] = np.linalg.det(Kb)\n        out[r, 1] = d[ea].sum()\n        out[r, 2] = -kT * np.log(np.linalg.det(Yb))\n    return out\ndef _fx_active_channel_weights(P, eta):\n    P = np.asarray(P, dtype=np.float64).ravel()\n    eta = float(eta)\n    if P.size == 0 or not np.all(np.isfinite(P)) or np.any(P <= 0.0):\n        raise ValueError(\"P must be a non-empty array of finite positive path probabilities\")\n    if not (0.0 < eta <= 1.0):\n        raise ValueError(\"eta must lie in (0, 1]\")\n    q = P / P.sum()\n    order = np.argsort(-q, kind=\"stable\")\n    cum = np.cumsum(q[order])\n    m = int(np.searchsorted(cum, eta, side=\"left\")) + 1\n    m = min(m, P.size)\n    act = order[:m]\n    p = np.zeros_like(P)\n    p[act] = P[act] / P[act].sum()\n    return p\ndef _fx_allosteric_importance(paths, p, n_res):\n    P = np.asarray(paths)\n    p = np.asarray(p, dtype=np.float64).ravel()\n    if isinstance(n_res, bool) or int(n_res) != n_res or int(n_res) < 3:\n        raise ValueError(\"n_res must be an integer >= 3\")\n    n_res = int(n_res)\n    if P.ndim != 2 or P.shape[0] != p.size or np.any(p < 0) or not np.isclose(p.sum(), 1.0) or P.max() >= n_res:\n        raise ValueError(\"paths (m, max_nodes) must align with the probability vector p and index residues < n_res\")\n    I = np.zeros(n_res, dtype=np.float64)\n    for r in range(P.shape[0]):\n        if p[r] > 0.0:\n            nodes = [int(v) for v in P[r] if v >= 0]\n            for v in nodes[1:-1]:\n                I[v] += p[r]\n    return I\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT = 7.8, 1.0\nL = _fx_contact_laplacian(X, r_c, kT)\ndef run_model():\n    try:\n        global_tree_thermodynamics(L, 0.0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_global_tree_thermodynamics(L, 0.0)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

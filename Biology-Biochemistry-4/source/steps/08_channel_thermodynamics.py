"""
From the occupancy weights of the active channel and the per-path table, compute the channel entropy S = -sum p ln p, the participation ratio, the channel heat capacity C = Var(E)/(kT)^2 of the effective path energy E = E_E + E_T, and its decomposition into the energetic component from E_E, the topological component from E_T and the cross-coupling component from their covariance, all averaged over the active distribution.

The heat capacity of a channel measures the breadth of energies its routes sample; splitting the effective energy into a physical and a topological part decomposes that breadth into contact rigidity, path multiplicity and the coupling between them, the signature the framework uses to read how a mutation reroutes a signal.

Returns
-------
ndarray of float64, shape (6,), [S, PR, C, C_E, C_T, C_X] with the capacities in units of k.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def channel_thermodynamics(p: "np.ndarray", table: "np.ndarray", kT: float) -> "np.ndarray":
    """From the occupancy weights of the active channel and the per-path table, compute the channel entropy S = -sum p ln p, the participation ratio, the channel heat capacity C = Var(E)/(kT)^2 of the effective path energy E = E_E + E_T, and its decomposition into the energetic component from E_E, the topological component from E_T and the cross-coupling component from their covariance, all averaged over the active distribution.

    Parameters
    ----------
    p : np.ndarray
        Occupancy weights of shape (m,) summing to one (zeros for inactive paths).
    table : np.ndarray
        Per-path table of shape (m, 3) with columns [P, E_E, E_T].
    kT : float
        Effective temperature in angstrom.

    Returns
    -------
    thermo : np.ndarray
        Array [S, PR, C, C_E, C_T, C_X].

    Raises
    ------
    ValueError
        If p is not a probability vector aligned with the rows of table, or kT is not positive.
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


def _oracle_channel_thermodynamics(p: "np.ndarray", table: "np.ndarray", kT: float) -> "np.ndarray":
    """Eqs. (13), (16)-(18) and 2.3.2: [S, PR, C, C_E, C_T, C_X] over the active channel distribution."""
    p = np.asarray(p, dtype=np.float64).ravel()
    T = np.asarray(table, dtype=np.float64)
    kT = _check_positive(kT, "kT")
    if T.ndim != 2 or T.shape[1] != 3 or T.shape[0] != p.size or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("p must be a probability vector aligned with the rows of table (m, 3)")
    act = p > 0.0
    pa = p[act]
    EE = T[act, 1]
    ET = T[act, 2]
    mean = lambda x: float(pa @ x)
    var = lambda x: float(pa @ (x - mean(x)) ** 2)
    cov = float(pa @ ((EE - mean(EE)) * (ET - mean(ET))))
    S = -float(pa @ np.log(pa))
    PR = 1.0 / float(pa @ pa)
    E = EE + ET
    return np.array([S, PR, var(E) / kT ** 2, var(EE) / kT ** 2, var(ET) / kT ** 2, 2.0 * cov / kT ** 2])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nimport numpy as np\ndef _fx_check_laplacian(L):\n    L = np.asarray(L, dtype=np.float64)\n    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):\n        raise ValueError(\"L must be a finite square matrix of size >= 3\")\n    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):\n        raise ValueError(\"L must be a symmetric Laplacian with zero row sums\")\n    return L\ndef _fx_check_positive(x, name):\n    x = float(x)\n    if not np.isfinite(x) or x <= 0.0:\n        raise ValueError(name + \" must be a finite positive number\")\n    return x\ndef _fx_edges_from_laplacian(L):\n    n = L.shape[0]\n    ii, jj = np.where(np.triu(L, 1) < 0.0)\n    edges = list(zip(ii.tolist(), jj.tolist()))\n    if not edges:\n        raise ValueError(\"L has no edges\")\n    w = -L[ii, jj]\n    return edges, w\ndef _fx_transfer_current(L, edges, w):\n    n = L.shape[0]\n    Kp = np.zeros((n, n))\n    Kp[1:, 1:] = np.linalg.inv(L[1:, 1:])\n    chi = np.zeros((len(edges), n))\n    for a, (i, j) in enumerate(edges):\n        chi[a, i] = 1.0\n        chi[a, j] = -1.0\n    Y = chi @ Kp @ chi.T\n    return Y * np.sqrt(np.outer(w, w))\ndef _fx_contact_laplacian(coords, r_c, kT):\n    X = np.asarray(coords, dtype=np.float64)\n    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 3 or not np.all(np.isfinite(X)):\n        raise ValueError(\"coords must be a finite (N, 3) array with N >= 3\")\n    r_c, kT = float(r_c), float(kT)\n    if not (np.isfinite(r_c) and r_c > 0.0 and np.isfinite(kT) and kT > 0.0):\n        raise ValueError(\"r_c and kT must be finite positive numbers\")\n    n = X.shape[0]\n    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))\n    W = np.where((D <= r_c) & ~np.eye(n, dtype=bool), np.exp(-D / kT), 0.0)\n    L = np.diag(W.sum(axis=1)) - W\n    return L\ndef _fx_edge_transfer_current(L, kT):\n    L = np.asarray(L, dtype=np.float64)\n    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):\n        raise ValueError(\"L must be a finite square matrix of size >= 3\")\n    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):\n        raise ValueError(\"L must be a symmetric Laplacian with zero row sums\")\n    if not (np.isfinite(float(kT)) and float(kT) > 0.0):\n        raise ValueError(\"kT must be a finite positive number\")\n    edges, w = _fx_edges_from_laplacian(L)\n    return _fx_transfer_current(L, edges, w)\ndef _fx_channel_paths(L, s, t, max_nodes):\n    L = _fx_check_laplacian(L)\n    n = L.shape[0]\n    for name, v in ((\"s\", s), (\"t\", t), (\"max_nodes\", max_nodes)):\n        if isinstance(v, bool) or int(v) != v:\n            raise ValueError(name + \" must be an integer\")\n    s, t, max_nodes = int(s), int(t), int(max_nodes)\n    if not (0 <= s < n and 0 <= t < n) or s == t or max_nodes < 2:\n        raise ValueError(\"s and t must be distinct residue indices and max_nodes >= 2\")\n    adj = [np.where(L[v] < 0.0)[0].tolist() for v in range(n)]\n    out = []\n    stack = [(s, (s,))]\n    while stack:\n        v, path = stack.pop()\n        if v == t:\n            out.append(path)\n            continue\n        if len(path) >= max_nodes:\n            continue\n        for u in adj[v]:\n            if u not in path:\n                stack.append((u, path + (u,)))\n    out.sort()\n    P = -np.ones((len(out), max_nodes), dtype=np.int64)\n    for r, path in enumerate(out):\n        P[r, :len(path)] = path\n    return P\ndef _fx_path_energy_table(L, kT, K, paths):\n    L = _fx_check_laplacian(L)\n    kT = _fx_check_positive(kT, \"kT\")\n    edges, w = _fx_edges_from_laplacian(L)\n    K = np.asarray(K, dtype=np.float64)\n    P = np.asarray(paths)\n    if K.shape != (len(edges), len(edges)) or P.ndim != 2 or P.shape[0] == 0:\n        raise ValueError(\"K must be E x E for the edges of L and paths must be a non-empty (m, max_nodes) array\")\n    d = -kT * np.log(w)\n    eidx = {}\n    for a, (i, j) in enumerate(edges):\n        eidx[(i, j)] = a\n        eidx[(j, i)] = a\n    out = np.zeros((P.shape[0], 3), dtype=np.float64)\n    for r in range(P.shape[0]):\n        nodes = [int(v) for v in P[r] if v >= 0]\n        if len(nodes) < 2:\n            raise ValueError(\"every path needs at least two nodes\")\n        try:\n            ea = [eidx[(nodes[k], nodes[k + 1])] for k in range(len(nodes) - 1)]\n        except KeyError:\n            raise ValueError(\"path uses a pair that is not an edge of L\")\n        Kb = K[np.ix_(ea, ea)]\n        Yb = Kb / np.sqrt(np.outer(w[ea], w[ea]))\n        out[r, 0] = np.linalg.det(Kb)\n        out[r, 1] = d[ea].sum()\n        out[r, 2] = -kT * np.log(np.linalg.det(Yb))\n    return out\ndef _fx_active_channel_weights(P, eta):\n    P = np.asarray(P, dtype=np.float64).ravel()\n    eta = float(eta)\n    if P.size == 0 or not np.all(np.isfinite(P)) or np.any(P <= 0.0):\n        raise ValueError(\"P must be a non-empty array of finite positive path probabilities\")\n    if not (0.0 < eta <= 1.0):\n        raise ValueError(\"eta must lie in (0, 1]\")\n    q = P / P.sum()\n    order = np.argsort(-q, kind=\"stable\")\n    cum = np.cumsum(q[order])\n    m = int(np.searchsorted(cum, eta, side=\"left\")) + 1\n    m = min(m, P.size)\n    act = order[:m]\n    p = np.zeros_like(P)\n    p[act] = P[act] / P[act].sum()\n    return p\ndef _fx_allosteric_importance(paths, p, n_res):\n    P = np.asarray(paths)\n    p = np.asarray(p, dtype=np.float64).ravel()\n    if isinstance(n_res, bool) or int(n_res) != n_res or int(n_res) < 3:\n        raise ValueError(\"n_res must be an integer >= 3\")\n    n_res = int(n_res)\n    if P.ndim != 2 or P.shape[0] != p.size or np.any(p < 0) or not np.isclose(p.sum(), 1.0) or P.max() >= n_res:\n        raise ValueError(\"paths (m, max_nodes) must align with the probability vector p and index residues < n_res\")\n    I = np.zeros(n_res, dtype=np.float64)\n    for r in range(P.shape[0]):\n        if p[r] > 0.0:\n            nodes = [int(v) for v in P[r] if v >= 0]\n            for v in nodes[1:-1]:\n                I[v] += p[r]\n    return I\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT = 7.8, 1.0\ns, t, max_nodes = 4, 22, 6\nL = _fx_contact_laplacian(X, r_c, kT)\nK = _fx_edge_transfer_current(L, kT)\npaths = _fx_channel_paths(L, s, t, max_nodes)\ntable = _fx_path_energy_table(L, kT, K, paths)\np = _fx_active_channel_weights(table[:, 0], 0.99)\n",
            "call": "np.asarray(channel_thermodynamics(p, table, kT))",
            "gold_call": "np.asarray(_oracle_channel_thermodynamics(p, table, kT))",
        },
        {
            "setup": "import numpy as np\nimport numpy as np\ndef _fx_check_laplacian(L):\n    L = np.asarray(L, dtype=np.float64)\n    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):\n        raise ValueError(\"L must be a finite square matrix of size >= 3\")\n    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):\n        raise ValueError(\"L must be a symmetric Laplacian with zero row sums\")\n    return L\ndef _fx_check_positive(x, name):\n    x = float(x)\n    if not np.isfinite(x) or x <= 0.0:\n        raise ValueError(name + \" must be a finite positive number\")\n    return x\ndef _fx_edges_from_laplacian(L):\n    n = L.shape[0]\n    ii, jj = np.where(np.triu(L, 1) < 0.0)\n    edges = list(zip(ii.tolist(), jj.tolist()))\n    if not edges:\n        raise ValueError(\"L has no edges\")\n    w = -L[ii, jj]\n    return edges, w\ndef _fx_transfer_current(L, edges, w):\n    n = L.shape[0]\n    Kp = np.zeros((n, n))\n    Kp[1:, 1:] = np.linalg.inv(L[1:, 1:])\n    chi = np.zeros((len(edges), n))\n    for a, (i, j) in enumerate(edges):\n        chi[a, i] = 1.0\n        chi[a, j] = -1.0\n    Y = chi @ Kp @ chi.T\n    return Y * np.sqrt(np.outer(w, w))\ndef _fx_contact_laplacian(coords, r_c, kT):\n    X = np.asarray(coords, dtype=np.float64)\n    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 3 or not np.all(np.isfinite(X)):\n        raise ValueError(\"coords must be a finite (N, 3) array with N >= 3\")\n    r_c, kT = float(r_c), float(kT)\n    if not (np.isfinite(r_c) and r_c > 0.0 and np.isfinite(kT) and kT > 0.0):\n        raise ValueError(\"r_c and kT must be finite positive numbers\")\n    n = X.shape[0]\n    D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1))\n    W = np.where((D <= r_c) & ~np.eye(n, dtype=bool), np.exp(-D / kT), 0.0)\n    L = np.diag(W.sum(axis=1)) - W\n    return L\ndef _fx_edge_transfer_current(L, kT):\n    L = np.asarray(L, dtype=np.float64)\n    if L.ndim != 2 or L.shape[0] != L.shape[1] or L.shape[0] < 3 or not np.all(np.isfinite(L)):\n        raise ValueError(\"L must be a finite square matrix of size >= 3\")\n    if not np.allclose(L, L.T) or not np.allclose(L.sum(axis=1), 0.0, atol=1e-9):\n        raise ValueError(\"L must be a symmetric Laplacian with zero row sums\")\n    if not (np.isfinite(float(kT)) and float(kT) > 0.0):\n        raise ValueError(\"kT must be a finite positive number\")\n    edges, w = _fx_edges_from_laplacian(L)\n    return _fx_transfer_current(L, edges, w)\ndef _fx_channel_paths(L, s, t, max_nodes):\n    L = _fx_check_laplacian(L)\n    n = L.shape[0]\n    for name, v in ((\"s\", s), (\"t\", t), (\"max_nodes\", max_nodes)):\n        if isinstance(v, bool) or int(v) != v:\n            raise ValueError(name + \" must be an integer\")\n    s, t, max_nodes = int(s), int(t), int(max_nodes)\n    if not (0 <= s < n and 0 <= t < n) or s == t or max_nodes < 2:\n        raise ValueError(\"s and t must be distinct residue indices and max_nodes >= 2\")\n    adj = [np.where(L[v] < 0.0)[0].tolist() for v in range(n)]\n    out = []\n    stack = [(s, (s,))]\n    while stack:\n        v, path = stack.pop()\n        if v == t:\n            out.append(path)\n            continue\n        if len(path) >= max_nodes:\n            continue\n        for u in adj[v]:\n            if u not in path:\n                stack.append((u, path + (u,)))\n    out.sort()\n    P = -np.ones((len(out), max_nodes), dtype=np.int64)\n    for r, path in enumerate(out):\n        P[r, :len(path)] = path\n    return P\ndef _fx_path_energy_table(L, kT, K, paths):\n    L = _fx_check_laplacian(L)\n    kT = _fx_check_positive(kT, \"kT\")\n    edges, w = _fx_edges_from_laplacian(L)\n    K = np.asarray(K, dtype=np.float64)\n    P = np.asarray(paths)\n    if K.shape != (len(edges), len(edges)) or P.ndim != 2 or P.shape[0] == 0:\n        raise ValueError(\"K must be E x E for the edges of L and paths must be a non-empty (m, max_nodes) array\")\n    d = -kT * np.log(w)\n    eidx = {}\n    for a, (i, j) in enumerate(edges):\n        eidx[(i, j)] = a\n        eidx[(j, i)] = a\n    out = np.zeros((P.shape[0], 3), dtype=np.float64)\n    for r in range(P.shape[0]):\n        nodes = [int(v) for v in P[r] if v >= 0]\n        if len(nodes) < 2:\n            raise ValueError(\"every path needs at least two nodes\")\n        try:\n            ea = [eidx[(nodes[k], nodes[k + 1])] for k in range(len(nodes) - 1)]\n        except KeyError:\n            raise ValueError(\"path uses a pair that is not an edge of L\")\n        Kb = K[np.ix_(ea, ea)]\n        Yb = Kb / np.sqrt(np.outer(w[ea], w[ea]))\n        out[r, 0] = np.linalg.det(Kb)\n        out[r, 1] = d[ea].sum()\n        out[r, 2] = -kT * np.log(np.linalg.det(Yb))\n    return out\ndef _fx_active_channel_weights(P, eta):\n    P = np.asarray(P, dtype=np.float64).ravel()\n    eta = float(eta)\n    if P.size == 0 or not np.all(np.isfinite(P)) or np.any(P <= 0.0):\n        raise ValueError(\"P must be a non-empty array of finite positive path probabilities\")\n    if not (0.0 < eta <= 1.0):\n        raise ValueError(\"eta must lie in (0, 1]\")\n    q = P / P.sum()\n    order = np.argsort(-q, kind=\"stable\")\n    cum = np.cumsum(q[order])\n    m = int(np.searchsorted(cum, eta, side=\"left\")) + 1\n    m = min(m, P.size)\n    act = order[:m]\n    p = np.zeros_like(P)\n    p[act] = P[act] / P[act].sum()\n    return p\ndef _fx_allosteric_importance(paths, p, n_res):\n    P = np.asarray(paths)\n    p = np.asarray(p, dtype=np.float64).ravel()\n    if isinstance(n_res, bool) or int(n_res) != n_res or int(n_res) < 3:\n        raise ValueError(\"n_res must be an integer >= 3\")\n    n_res = int(n_res)\n    if P.ndim != 2 or P.shape[0] != p.size or np.any(p < 0) or not np.isclose(p.sum(), 1.0) or P.max() >= n_res:\n        raise ValueError(\"paths (m, max_nodes) must align with the probability vector p and index residues < n_res\")\n    I = np.zeros(n_res, dtype=np.float64)\n    for r in range(P.shape[0]):\n        if p[r] > 0.0:\n            nodes = [int(v) for v in P[r] if v >= 0]\n            for v in nodes[1:-1]:\n                I[v] += p[r]\n    return I\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT = 7.8, 1.0\ns, t, max_nodes = 0, 23, 5\nL = _fx_contact_laplacian(X, r_c, kT)\nK = _fx_edge_transfer_current(L, kT)\npaths = _fx_channel_paths(L, s, t, max_nodes)\ntable = _fx_path_energy_table(L, kT, K, paths)\np = _fx_active_channel_weights(table[:, 0], 0.9)\n",
            "call": "np.asarray(channel_thermodynamics(p, table, kT))",
            "gold_call": "np.asarray(_oracle_channel_thermodynamics(p, table, kT))",
        },
        {
            "setup": "import numpy as np\nimport numpy as np\np = np.array([0.5, 0.3, 0.2, 0.0])\ntable = np.array([[0.2, 10.0, -3.0], [0.12, 12.5, -2.0], [0.08, 15.0, -1.0], [0.01, 20.0, 2.0]])\nkT = 1.0\n",
            "call": "np.asarray(channel_thermodynamics(p, table, kT))",
            "gold_call": "np.asarray(_oracle_channel_thermodynamics(p, table, kT))",
        },
        {
            "setup": "import numpy as np\nimport numpy as np\np = np.array([0.5, 0.3])\ntable = np.array([[0.2, 10.0, -3.0], [0.12, 12.5, -2.0]])\ndef run_model():\n    try:\n        channel_thermodynamics(p, table, 1.0)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_channel_thermodynamics(p, table, 1.0)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

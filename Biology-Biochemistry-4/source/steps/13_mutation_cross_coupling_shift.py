"""
Orchestrate the full pipeline for the wild-type and the mutant coordinates: locate the wild type's heat-capacity peak temperature kT* in (kT_lo, kT_hi) and evaluate both structures at that temperature; build each contact Laplacian, its global thermodynamics and edge response matrix, the channel path table between residues s and t inside the node-length envelope, the per-path energies, the active channel weights at threshold eta, the channel thermodynamics, the residue importances, the convergence ratio of each channel at the envelope, the divergence between the two channels and the table of importance shifts, and return the shift of the cross-coupling component of the channel heat capacity, mutant minus wild type. Call the earlier step functions rather than reimplementing them.

The cross-coupling term of a channel's heat capacity records how local contact-length fluctuations and the global topological reservoir move together; its shift under a mutation is the framework's fingerprint of compensatory rewiring, and evaluating it at the wild type's heat-capacity peak places the comparison at the temperature where the tree ensemble is most perturbable.

Returns
-------
float, Delta C_X = C_X(mutant) - C_X(wild type) at kT*, in units of k.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def mutation_cross_coupling_shift(coords_wt: "np.ndarray", coords_mut: "np.ndarray", r_c: float, kT_lo: float, kT_hi: float,
                                          s: int, t: int, max_nodes: int, eta: float) -> float:
    """Orchestrate the full pipeline for the wild-type and the mutant coordinates: locate the wild type's heat-capacity peak temperature kT* in (kT_lo, kT_hi) and evaluate both structures at that temperature; build each contact Laplacian, its global thermodynamics and edge response matrix, the channel path table between residues s and t inside the node-length envelope, the per-path energies, the active channel weights at threshold eta, the channel thermodynamics, the residue importances, the convergence ratio of each channel at the envelope, the divergence between the two channels and the table of importance shifts, and return the shift of the cross-coupling component of the channel heat capacity, mutant minus wild type. Call the earlier step functions rather than reimplementing them.

    Parameters
    ----------
    coords_wt : np.ndarray
        Wild-type C-alpha coordinates, shape (N, 3).
    coords_mut : np.ndarray
        Mutant C-alpha coordinates, shape (N, 3).
    r_c : float
        Contact cutoff in angstrom.
    kT_lo : float
        Lower end of the peak-temperature search interval (angstrom).
    kT_hi : float
        Upper end of the peak-temperature search interval (angstrom).
    s : int
        Source residue index.
    t : int
        Target residue index.
    max_nodes : int
        Maximum number of nodes in a channel path.
    eta : float
        Cumulative-probability threshold of the active channel.

    Returns
    -------
    delta_cx : float
        Shift of the cross-coupling heat-capacity component at kT*.

    Raises
    ------
    ValueError
        If the coordinate arrays are not finite (N, 3) arrays of the same shape, or any parameter is invalid for the steps above.
    """
    return delta_cx

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.linalg import det, slogdet


def _check_coords(coords):
    X = np.asarray(coords, dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != 3 or X.shape[0] < 3 or not np.all(np.isfinite(X)):
        raise ValueError("coords must be a finite (N, 3) array with N >= 3")
    return X


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


def _path_distribution(P, p):
    """Map each active path (node tuple) to its occupancy weight; validates alignment and normalisation."""
    P = np.asarray(P)
    p = np.asarray(p, dtype=np.float64).ravel()
    if P.ndim != 2 or P.shape[0] != p.size or np.any(p < 0) or not np.isclose(p.sum(), 1.0):
        raise ValueError("each path table must align with its probability vector")
    return {tuple(int(v) for v in P[r] if v >= 0): float(p[r]) for r in range(P.shape[0]) if p[r] > 0.0}


def _effective_resistance(L, a, b):
    """Eq. (7) from the grounded generalised inverse of a connected Laplacian."""
    n = L.shape[0]
    G = np.zeros((n, n))
    G[1:, 1:] = np.linalg.inv(L[1:, 1:])
    return G[a, a] + G[b, b] - 2.0 * G[a, b]


def _tree_energy_cumulants(X, r_c, kT):
    """Exact (Var(E), kappa_3) of the tree energy at temperature kT from the determinantal edge-indicator structure:
    Var = d^T Cov d with Cov = -K*K off the diagonal and p(1-p) on it; kappa_3 = sum d^3 p - 3 sum d_e^2 d_f K_ef^2
    + 2 sum d_e d_f d_g K_ef K_fg K_ge (= -dVar/dbeta)."""
    L = _oracle_contact_laplacian(X, r_c, kT)
    edges, w = _edges_from_laplacian(L)
    d = -kT * np.log(w)
    K = _transfer_current(L, edges, w)
    p = np.diag(K)
    K2 = K ** 2
    cov = -K2
    np.fill_diagonal(cov, p * (1.0 - p))
    var = float(d @ cov @ d)
    k3 = float(np.sum(d ** 3 * p) - 3.0 * np.sum((d ** 2)[:, None] * d[None, :] * K2)
               + 2.0 * (d @ (K * (K @ np.diag(d) @ K)) @ d))
    return var, k3


def _peak_residual(X, r_c, kT):
    """g(kT) = kappa_3 - 2 kT Var(E): positive below the heat-capacity peak, negative above it (dC/dbeta = beta^2 g / kT)."""
    var, k3 = _tree_energy_cumulants(X, r_c, kT)
    return k3 - 2.0 * kT * var


def _oracle_mutation_cross_coupling_shift(coords_wt: "np.ndarray", coords_mut: "np.ndarray", r_c: float, kT_lo: float, kT_hi: float,
                                          s: int, t: int, max_nodes: int, eta: float) -> float:
    """Delta C_X = C_X(mutant) - C_X(wild type) for the channel s -> t (Table 2 of the source), both structures evaluated
    at the wild type's heat-capacity peak temperature kT* found in (kT_lo, kT_hi)."""
    Xw = _check_coords(coords_wt)
    Xm = _check_coords(coords_mut)
    if Xw.shape != Xm.shape:
        raise ValueError("wild-type and mutant coordinates must have the same shape")
    kT = _oracle_heat_capacity_peak_temperature(Xw, r_c, kT_lo, kT_hi)
    ensembles = []
    for X in (Xw, Xm):
        L = _oracle_contact_laplacian(X, r_c, kT)
        thermo = _oracle_global_tree_thermodynamics(L, kT)          # also certifies a connected graph
        if not np.all(np.isfinite(thermo)):
            raise ValueError("global tree thermodynamics are not finite")
        K = _oracle_edge_transfer_current(L, kT)
        paths = _oracle_channel_paths(L, s, t, max_nodes)
        table = _oracle_path_energy_table(L, kT, K, paths)
        p = _oracle_active_channel_weights(table[:, 0], eta)
        channel = _oracle_channel_thermodynamics(p, table, kT)
        importance = _oracle_allosteric_importance(paths, p, X.shape[0])
        if importance[int(s)] != 0.0 or importance[int(t)] != 0.0 or np.any(importance > 1.0 + 1e-12):
            raise ValueError("allosteric importance must vanish at the endpoints and never exceed one")
        ratio = _oracle_channel_convergence_ratio(L, s, t, max_nodes)
        if ratio < 1.0 - 1e-9:                                      # Rayleigh monotonicity: R_st(G(L)) >= R_st(G)
            raise ValueError("the truncated channel cannot have a lower resistance than the full graph")
        ensembles.append((paths, p, channel, importance))
    d_js = _oracle_channel_divergence(ensembles[0][0], ensembles[0][1], ensembles[1][0], ensembles[1][1])
    if not (-1e-12 <= d_js <= np.log(2.0) + 1e-12):
        raise ValueError("the channel divergence must lie in [0, ln 2]")
    shifts = _oracle_importance_shift_table(ensembles[0][3], ensembles[1][3])
    if not np.all(np.isfinite(shifts)):
        raise ValueError("importance shifts must be finite")
    return float(ensembles[1][2][5] - ensembles[0][2][5])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return the list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nXm = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [8.0, -0.3, 7.9], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT_lo, kT_hi = 7.8, 0.2, 5.0\ns, t, max_nodes, eta = 4, 22, 7, 0.99\n",
            "call": "mutation_cross_coupling_shift(X, Xm, r_c, kT_lo, kT_hi, s, t, max_nodes, eta)",
            "gold_call": "_oracle_mutation_cross_coupling_shift(X, Xm, r_c, kT_lo, kT_hi, s, t, max_nodes, eta)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nXm = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [8.0, -0.3, 7.9], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT_lo, kT_hi = 7.8, 0.2, 5.0\ns, t, max_nodes, eta = 0, 23, 6, 0.95\n",
            "call": "mutation_cross_coupling_shift(X, Xm, r_c, kT_lo, kT_hi, s, t, max_nodes, eta)",
            "gold_call": "_oracle_mutation_cross_coupling_shift(X, Xm, r_c, kT_lo, kT_hi, s, t, max_nodes, eta)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nXm = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [8.0, -0.3, 7.9], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nr_c, kT_lo, kT_hi = 7.0, 0.3, 3.0\ns, t, max_nodes, eta = 4, 22, 6, 0.99\n",
            "call": "mutation_cross_coupling_shift(X, Xm, r_c, kT_lo, kT_hi, s, t, max_nodes, eta)",
            "gold_call": "_oracle_mutation_cross_coupling_shift(X, Xm, r_c, kT_lo, kT_hi, s, t, max_nodes, eta)",
            "tol": 1e-08,
        },
        {
            "setup": "import numpy as np\nimport numpy as np\nX = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [7.2, -0.0, 7.5], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\nXm = np.array([[2.3, 0.0, 0.0], [-0.4, 2.27, 1.5], [-2.16, -0.79, 3.0], [1.15, -1.99, 4.5], [1.76, 1.48, 6.0], [-1.76, 1.48, 7.5], [-1.15, -1.99, 9.0], [2.16, -0.79, 10.5], [0.4, 2.27, 12.0], [-2.3, 0.0, 13.5], [0.4, -2.27, 15.0], [2.38, 1.63, 17.41], [4.75, 2.3, 18.0], [7.12, 1.63, 17.41], [11.26, 1.48, 15.0], [7.74, 1.48, 13.5], [8.35, -1.99, 12.0], [11.66, -0.78, 10.5], [9.9, 2.27, 9.0], [8.0, -0.3, 7.9], [9.9, -2.26, 6.0], [11.66, 0.79, 4.5], [8.35, 1.99, 3.0], [7.74, -1.48, 1.5], [11.26, -1.48, 0.0], [10.65, 1.99, -1.5]])\ndef run_model():\n    try:\n        mutation_cross_coupling_shift(X, Xm[:-1], 7.8, 0.2, 5.0, 4, 22, 6, 0.99)\n        return 0\n    except ValueError:\n        return 1\ndef run_oracle():\n    try:\n        _oracle_mutation_cross_coupling_shift(X, Xm[:-1], 7.8, 0.2, 5.0, 4, 22, 6, 0.99)\n        return 0\n    except ValueError:\n        return 1\n",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

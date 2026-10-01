"""
When explorers are fast compared with the turnover of settled individuals, their densities sit at the quasi-stationary balance F x = b / lambda for the current production b, so what the settled population experiences is the resolvent F^(-1). The operator L does not depend on f or g, so it is diagonalised once, L = V Omega V^(-1), and the resolvent for any exploration efficiency and mortality ratio follows from the same eigenbasis,

F^(-1) = V diag(1 / (1 + g + f omega_l)) V^(-1),

which is how the effective kernel is written as an explicit function of the dispersal dynamics and the topology. For a directed network the eigenvalues and eigenvectors are in general complex, the inverse eigenvector matrix is not the transpose of the eigenvector matrix, and the resolvent is real only after the conjugate pairs are summed. An operator whose eigenvector matrix is numerically singular cannot be inverted this way.

Explorers are the mobile stage of the species. An explorer in patch i moves to patch j at rate D w_ij, dies at rate gamma, and attempts to colonise a site of its own patch at rate lambda; the attempt consumes the explorer whether or not the chosen site was empty. Measured as a density per site, x_i = [X_i] / M_i, the explorer population therefore relaxes under a linear operator that combines loss by death and colonisation attempts with movement along the network. Movement out of patch i removes density at rate D q_i x_i, and movement into patch i from patch j adds D w_ji [X_j] / M_i = D (M_j / M_i) w_ji x_j, so in units of the attempt rate lambda the operator is

F = (1 + g) I + f L,   L_ij = delta_ij q_i - (M_j / M_i) w_ji,

with the exploration efficiency f = D / lambda, the number of moves an explorer makes per colonisation attempt, and the mortality ratio g = gamma / lambda. The size-weighted generalised Laplacian L is not symmetric: the network is directed and the site numbers differ from patch to patch. Movement conserves the number of explorers, which appears as the left null vector of L: sum_i M_i L_ij = 0 for every j.

Returns
-------
dict, the explorer operator and its quasi-stationary resolvent, keyed by operator (the N by N L), eigenvalues_real and eigenvalues_imag (the parts of the eigenvalues of L, sorted by ascending real part and then by ascending imaginary part), resolvent (the real F^(-1)), conservation_residual (the largest modulus of sum_i M_i L_ij over the largest M_i q_i), inversion_residual (the largest modulus of F F^(-1) - I), and discarded_imaginary (the largest modulus of the imaginary part of the eigenbasis sum).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def explorer_resolvent(
    weights: np.ndarray,
    sites: np.ndarray,
    exploration_efficiency: float,
    mortality_ratio: float,
) -> dict:
    """Assemble the size-weighted explorer operator and invert the quasi-stationary balance in its eigenbasis.

    Parameters
    ----------
    weights : np.ndarray
        Link weights w_ij, non-negative with a zero diagonal.
    sites : np.ndarray
        Site numbers M_i, above zero.
    exploration_efficiency : float
        f = D / lambda, not below zero.
    mortality_ratio : float
        g = gamma / lambda, not below zero.

    Returns
    -------
    dict
        Under the keys operator, eigenvalues_real, eigenvalues_imag, resolvent, conservation_residual, inversion_residual and discarded_imaginary; the eigenvalues are sorted by ascending real part and then by ascending imaginary part.

    Raises
    ------
    ValueError
        When the weights fail to be a finite, non-negative square array with a zero diagonal, when the site numbers fail to be finite, above zero and of matching length, when f or g fails to be finite and not below zero, or when the operator is not diagonalisable to working precision.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _check_network(weights, sites):
    w = np.asarray(weights, dtype=float)
    if w.ndim != 2 or w.shape[0] != w.shape[1] or w.shape[0] < 2:
        raise ValueError("weights must be a square array of at least two patches")
    if not np.all(np.isfinite(w)) or np.any(w < 0.0) or np.any(np.diag(w) != 0.0):
        raise ValueError("weights must be finite, non-negative and zero on the diagonal")
    m = np.asarray(sites, dtype=float)
    if m.shape != (w.shape[0],) or not np.all(np.isfinite(m)) or np.any(m <= 0.0):
        raise ValueError("sites must be finite, above zero and one per patch")
    return w, m


def _oracle_explorer_resolvent(
    weights: np.ndarray,
    sites: np.ndarray,
    exploration_efficiency: float,
    mortality_ratio: float,
) -> dict:
    """Reference implementation."""
    w, m = _check_network(weights, sites)
    f = float(exploration_efficiency)
    g = float(mortality_ratio)
    if not (math.isfinite(f) and f >= 0.0 and math.isfinite(g) and g >= 0.0):
        raise ValueError("exploration_efficiency and mortality_ratio must be finite and not below zero")
    n = w.shape[0]
    q = w.sum(axis=1)
    operator = np.diag(q) - (m[None, :] / m[:, None]) * w.T

    omega, vectors = np.linalg.eig(operator)
    if np.linalg.cond(vectors) > 1e10:
        raise ValueError("the explorer operator is not diagonalisable to working precision")
    inverse_vectors = np.linalg.inv(vectors)
    summed = (vectors / (1.0 + g + f * omega)[None, :]) @ inverse_vectors
    resolvent = summed.real

    full = (1.0 + g) * np.eye(n) + f * operator
    order = np.lexsort((omega.imag, np.round(omega.real, 12)))
    return {
        "operator": operator,
        "eigenvalues_real": omega.real[order],
        "eigenvalues_imag": omega.imag[order],
        "resolvent": resolvent,
        "conservation_residual": float(np.max(np.abs(m @ operator)) / np.max(m * np.maximum(q, 1e-300))),
        "inversion_residual": float(np.max(np.abs(full @ resolvent - np.eye(n)))),
        "discarded_imaginary": float(np.max(np.abs(summed.imag))),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
    def flat(x):
        # flatten to a tuple of plain numeric terminals
        if isinstance(x, (tuple, list)):
            out = []
            for v in x:
                out.extend(flat(v))
            return tuple(out)
        if hasattr(x, "tolist"):
            return flat(x.tolist())
        if isinstance(x, bool):
            return (int(x),)
        return (x,)
    """

    SETUP = """
    import numpy as np
    DOWN = (-1, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 6, 10)
    W = np.zeros((13, 13))
    for i, p in enumerate(DOWN):
        if p >= 0:
            W[i, p] = 1.0
            W[p, i] = 0.3
    W[7, 12] = 0.6
    W[11, 4] = 0.5
    DRAIN = np.array([13, 5, 7, 3, 1, 2, 4, 1, 1, 1, 2, 1, 1], dtype=float)
    M = 40.0 * DRAIN
    def digest(out):
        small = lambda v: 1 if v < 1e-9 else 0
        return (np.round(out["operator"], 10), np.round(out["eigenvalues_real"], 8), np.round(np.abs(out["eigenvalues_imag"]), 8),
                np.round(out["resolvent"], 8), small(out["conservation_residual"]), small(out["inversion_residual"]),
                small(out["discarded_imaginary"]))
    """
    import textwrap as _textwrap
    FLAT = _textwrap.dedent(FLAT)
    SETUP = _textwrap.dedent(SETUP)
    return [
        {
            # the river landscape at f = 1.6 and g = 0.25: complex spectrum, real resolvent
            "setup": SETUP + FLAT,
            "call": "flat(digest(explorer_resolvent(W, M, 1.6, 0.25)))",
            "gold_call": "flat(digest(_oracle_explorer_resolvent(W, M, 1.6, 0.25)))",
        },
        {
            # boundary f = 0: no movement, the resolvent is 1 / (1 + g) times the identity whatever the
            # network; and uniform sites on a symmetric ring give the combinatorial Laplacian
            "setup": SETUP + """
R = np.zeros((6, 6))
for i in range(6):
    R[i, (i + 1) % 6] = 0.7
    R[(i + 1) % 6, i] = 0.7
def boundary(fn):
    a = fn(W, M, 0.0, 0.25)
    b = fn(R, np.full(6, 5.0), 2.0, 0.0)
    return (np.round(a["resolvent"], 12), np.round(b["operator"], 12), np.round(b["eigenvalues_real"], 10),
            np.round(b["resolvent"], 10), int(np.max(np.abs(b["eigenvalues_imag"])) < 1e-12))
""" + FLAT,
            "call": "flat(boundary(explorer_resolvent))",
            "gold_call": "flat(boundary(_oracle_explorer_resolvent))",
        },
        {
            # conservation: every column of M-weighted F^(-1) sums to M_j / (1 + g), for any f,
            # and scaling every site number by a common factor leaves the operator unchanged
            "setup": SETUP + """
def conserve(fn):
    out = fn(W, M, 7.5, 0.4)
    cols = (M @ out["resolvent"]) / M
    scaled = fn(W, 3.0 * M, 7.5, 0.4)
    return (np.round(cols, 12), float(np.max(np.abs(scaled["operator"] - out["operator"]))) < 1e-12, np.round(out["resolvent"][0], 9))
""" + FLAT,
            "call": "flat(conserve(explorer_resolvent))",
            "gold_call": "flat(conserve(_oracle_explorer_resolvent))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(weights=W, sites=M, exploration_efficiency=1.6, mortality_ratio=0.25)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
BADDIAG = W.copy(); BADDIAG[2, 2] = 0.1
NEG = W.copy(); NEG[0, 1] = -0.1
DEFECT = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
""",
            "call": "(verdict(explorer_resolvent, weights=BADDIAG), verdict(explorer_resolvent, weights=NEG), verdict(explorer_resolvent, weights=W[:, :5]), verdict(explorer_resolvent, sites=M[:-1]), verdict(explorer_resolvent, sites=0.0 * M), verdict(explorer_resolvent, exploration_efficiency=-1.0), verdict(explorer_resolvent, mortality_ratio=float('inf')), verdict(explorer_resolvent, weights=DEFECT, sites=np.ones(3)), verdict(explorer_resolvent))",
            "gold_call": "(verdict(_oracle_explorer_resolvent, weights=BADDIAG), verdict(_oracle_explorer_resolvent, weights=NEG), verdict(_oracle_explorer_resolvent, weights=W[:, :5]), verdict(_oracle_explorer_resolvent, sites=M[:-1]), verdict(_oracle_explorer_resolvent, sites=0.0 * M), verdict(_oracle_explorer_resolvent, exploration_efficiency=-1.0), verdict(_oracle_explorer_resolvent, mortality_ratio=float('inf')), verdict(_oracle_explorer_resolvent, weights=DEFECT, sites=np.ones(3)), verdict(_oracle_explorer_resolvent))",
        },
    ]

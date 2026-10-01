"""
Build the community matrix and return the mass of the leading reactivity mode on the apparent-competition motif with the largest reactivity.

A large block score only names a strong subgraph bound. The mass of the leading short-time mode on that subgraph says whether the early jump of the full web is already sitting there.

Returns
-------
float, squared mass of the leading reactivity mode on the winning AC motif
"""

import math
import numpy
import scipy.special

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reactivity_mode_mass(
    A: "np.ndarray",
    phi: "np.ndarray",
    gamma: "np.ndarray",
    psi: "np.ndarray",
    mu: "np.ndarray",
    lam: "np.ndarray",
    R: float = 42.0,
) -> float:
    '''Return the mass of the leading reactivity mode on the most reactive AC motif.

    Parameters
    ----------
    A : np.ndarray
        Square 0-1 feeding matrix, A[i, j] = 1 if i eats j.
    phi, gamma, psi, mu : np.ndarray
        Length-N elasticities.
    lam : np.ndarray
        Shape (N, N) diet elasticities.
    R : float
        Metabolic base for row timescales. Default 42.

    Returns
    -------
    mass : float
        Sum of squared entries of the unit leading reactivity mode
        on the winning apparent-competition motif.

    Raises
    ------
    ValueError
        If inputs are invalid, if Levine levels do not converge,
        if there is no apparent-competition motif, or if the symmetric part is not finite.
    '''
    return mass

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_reactivity_mode_mass(
    A: "np.ndarray",
    phi: "np.ndarray",
    gamma: "np.ndarray",
    psi: "np.ndarray",
    mu: "np.ndarray",
    lam: "np.ndarray",
    R: float = 42.0,
) -> float:
    rho, sigma, chi, beta = _oracle_feeding_branching(A)
    alpha = _oracle_row_timescales(A, R)
    J = _oracle_community_matrix(A, phi, gamma, psi, mu, lam, alpha, rho, sigma, chi, beta)
    _ = _oracle_reactivity(J)
    motifs = _oracle_apparent_competition_motifs(A)
    S = 0.5 * J + 0.5 * J.T
    if not np.all(np.isfinite(S)):
        raise ValueError("symmetric part is not finite")
    r_motif = _oracle_max_motif_reactivity(S, motifs)
    best_rows = []
    for row in motifs:
        idx = np.asarray(row, dtype=int)
        val = float(np.linalg.eigvalsh(S[np.ix_(idx, idx)])[-1])
        if np.isclose(val, r_motif, rtol=0.0, atol=1e-12):
            best_rows.append([int(x) for x in row])
    best_rows.sort()
    idx = np.array(best_rows[0], dtype=int)
    v = np.linalg.eigh(S)[1][:, -1]
    nrm = float(np.linalg.norm(v))
    if nrm == 0.0:
        raise ValueError("leading reactivity mode has norm 0")
    v = v / nrm
    return float(np.sum(v[idx] ** 2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np
A = np.array([[0,0,0,0,0,0],[0,0,0,0,0,0],[1,1,0,0,0,0],[0,1,0,0,0,0],[1,0,1,0,0,0],[0,1,0,1,1,0]], dtype=float)
phi = np.array([0.22,0.38,0,0,0,0], dtype=float)
gamma = np.array([0.95,1.10,0.85,1.25,0.70,1.05], dtype=float)
psi = np.array([0.90,0.80,0.60,0.50,0.85,1.00], dtype=float)
mu = np.array([1.60,1.45,1.70,1.80,1.30,2.00], dtype=float)
lam = np.ones((6, 6), dtype=float); lam[5, 4] = 0.75; lam[4, 0] = 1.20
R = 42.0
""",
            "call": "reactivity_mode_mass(A, phi, gamma, psi, mu, lam, R)",
            "gold_call": "_oracle_reactivity_mode_mass(A, phi, gamma, psi, mu, lam, R)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,0,0],[0,0,0],[1,1,0]], dtype=float)
phi = np.array([0.2, 0.4, 0.0]); gamma = np.array([1.0, 1.0, 1.3])
psi = np.array([1.0, 1.0, 0.7]); mu = np.array([1.6, 1.6, 1.9])
lam = np.ones((3, 3)); R = 42.0
""",
            "call": "reactivity_mode_mass(A, phi, gamma, psi, mu, lam, R)",
            "gold_call": "_oracle_reactivity_mode_mass(A, phi, gamma, psi, mu, lam, R)",
        },
        {
            "setup": """import numpy as np
A = np.array([[0,0],[1,0]], dtype=float)
phi = np.array([0.2, 0.0]); gamma = np.array([1.0, 1.0])
psi = np.array([1.0, 0.8]); mu = np.array([1.5, 1.8])
lam = np.ones((2, 2))
def run_model():
    try:
        reactivity_mode_mass(A, phi, gamma, psi, mu, lam, 42.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_reactivity_mode_mass(A, phi, gamma, psi, mu, lam, 42.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]

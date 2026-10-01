"""
The logarithmic energy velocity along the trajectory, evaluated analytically rather than by differencing.

Optimising the CAP strength means balancing two errors that pull in opposite directions. A finite basis describes the continuum incompletely, which argues for larger eta so that resonance and continuum separate; but the absorbing potential is itself an artificial perturbation, whose damage grows with eta. The quantity that measures the residual first-order sensitivity is the logarithmic energy velocity

  v(eta) = | eta dE/deta |

and the trade-off is struck where v is stationary.

The derivative need not be taken numerically. For the generalised complex symmetric problem H(eta) C = E S C the Hellmann-Feynman theorem still holds, provided it is written in the bilinear c-product and the eigenvector is normalised as c^T S c = 1. In that form the derivative of the eigenvalue with respect to the CAP strength is a plain expectation value of the derivative of the Hamiltonian with respect to eta, so it costs nothing beyond the eigenvector already in hand, it is exact at every grid point, and it introduces none of the truncation error that differencing E(eta) along the grid would.

The velocity also inherits the branch: the vector used here is the one accepted by the tracking procedure, so v refers throughout to the same physical state rather than to whichever root happens to sit at a given position in a sorted list.

Returns
-------
numpy.ndarray of shape (n_eta,), real: the logarithmic energy velocity |eta dE/deta| at each grid point
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def branch_velocity(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                    eta_seed: float, e_max: float) -> "np.ndarray":
    '''Logarithmic energy velocity of the followed resonance branch.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    etas : numpy.ndarray
        Strictly increasing grid of non-negative CAP strengths, at least 3 points.
    eta_seed : float
        CAP strength at which the resonance is identified.
    e_max : float
        Top of the barrier, the upper edge of the seed search window.

    Returns
    -------
    velocity : numpy.ndarray
        Real array of shape (n_eta,) holding |eta dE/deta| along the branch, with
        dE/deta obtained analytically from the Hellmann-Feynman theorem in the
        c-product rather than by differencing E(eta).

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have n
        rows, if etas is not a strictly increasing array of at least 3 non-negative
        points, if e_max is not positive, or if no eigenvalue at the seed lies
        strictly between zero and e_max.
    '''
    return velocity

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_branch_velocity(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                            eta_seed: float, e_max: float) -> "np.ndarray":
    import numpy as np
    g = np.asarray(etas, dtype=float)
    I = np.asarray(integrals, dtype=float)
    W = I[2]
    # Step 5 supplies the energy of the tracked state at every grid point.
    # Match those energies to the eigensystems and evaluate the analytic
    # Hellmann-Feynman derivative for the selected state.
    energies = np.asarray(
        _oracle_branch_energies(integrals, transform, etas, eta_seed, e_max),
        dtype=complex,
    )
    D = np.empty(g.size, dtype=complex)
    for i, eta in enumerate(g):
        vals, C = _cap_solve(integrals, transform, float(eta))
        k = int(np.argmin(np.abs(vals - energies[i])))
        D[i] = -1j * (C[:, k] @ W @ C[:, k])
    return np.abs(g * D)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the full production grid ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'branch_velocity(I_model.copy(), X_model.copy(), g.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_branch_velocity(I_gold.copy(), X_gold.copy(), g.copy(), 0.15, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Normal: a coarse grid ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'branch_velocity(I_model.copy(), X_model.copy(), gs.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_branch_velocity(I_gold.copy(), X_gold.copy(), gs.copy(), 0.15, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Boundary: a grid that starts at eta = 0, where the velocity must vanish ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
g0 = np.concatenate(([0.0], np.geomspace(1e-3, 3.0, 89)))
""",
            "call": 'branch_velocity(I_model.copy(), X_model.copy(), g0.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_branch_velocity(I_gold.copy(), X_gold.copy(), g0.copy(), 0.15, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Edge: seeded at the far end of the grid ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'branch_velocity(I_model.copy(), X_model.copy(), gs.copy(), 3.0, 0.88291066)',
            "gold_call": '_oracle_branch_velocity(I_gold.copy(), X_gold.copy(), gs.copy(), 3.0, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Edge: the alternative basis and potential ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.05, 2.5, 9, 2.0, 0.8, 3.5)
I_gold = _oracle_basis_integrals(0.05, 2.5, 9, 2.0, 0.8, 3.5)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
gb = np.geomspace(1e-3, 2.0, 70)
""",
            "call": 'branch_velocity(I_model.copy(), X_model.copy(), gb.copy(), 0.3, 1.8394)',
            "gold_call": '_oracle_branch_velocity(I_gold.copy(), X_gold.copy(), gb.copy(), 0.3, 1.8394)',
            "tol": 1e-09,
        },
        # --- Edge: a different potential and a tighter, larger basis ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.01, 1.9, 24, 2.0, 0.8, 1.5)
I_gold = _oracle_basis_integrals(0.01, 1.9, 24, 2.0, 0.8, 1.5)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g2 = np.geomspace(1e-4, 4.0, 1200)
""",
            "call": 'branch_velocity(I_model.copy(), X_model.copy(), g2.copy(), 0.15, 0.9197005)',
            "gold_call": '_oracle_branch_velocity(I_gold.copy(), X_gold.copy(), g2.copy(), 0.15, 0.9197005)',
            "tol": 1e-09,
        },
        # --- Edge: CAP onset inside the barrier, which leaves only two solutions ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 1.2)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 1.2)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g3 = np.geomspace(1e-4, 4.0, 1200)
""",
            "call": 'branch_velocity(I_model.copy(), X_model.copy(), g3.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_branch_velocity(I_gold.copy(), X_gold.copy(), g3.copy(), 0.15, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Invalid: a two-point grid is too short to carry a trajectory ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
short = np.array([0.1, 0.2])
def run_model():
    try:
        branch_velocity(I_model.copy(), X_model.copy(), short.copy(), 0.15, 0.88291066)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_branch_velocity(I_gold.copy(), X_gold.copy(), short.copy(), 0.15, 0.88291066)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": 'run_model()',
            "gold_call": 'run_oracle()',
        },
    ]

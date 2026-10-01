"""
The first-order corrected resonance width at every CAP solution.

Evaluating the trajectory at a stationary point gives the zeroth-order resonance energy E(eta_opt), which still carries the artificial energy contribution of the absorbing potential. The standard remedy is to remove that contribution to first order, defining the deperturbed, or first-order corrected, energy

  U = E - eta dE/deta

again with the derivative taken analytically from the c-product Hellmann-Feynman expression. Writing U = E_R - i Gamma / 2 gives the corrected width as Gamma = -2 Im U. The deperturbation is often expected to suppress the sensitivity to the CAP, and with it the multiplicity of solutions; whether it actually does so is a question to be settled by computation rather than assumed.

Because the refined CAP strength no longer lies on the grid, the eigenproblem is solved once more at eta_opt and the branch is re-identified there by taking the eigenvector of largest c-product overlap magnitude with the vector already accepted at the bracketing grid point. Widths are returned in increasing order of CAP strength, matching the ordering of the solutions themselves.

Returns
-------
numpy.ndarray of shape (n_solutions,), real: the first-order corrected width Gamma = -2 Im U in hartree at each CAP solution, ordered by increasing CAP strength
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def deperturbed_widths(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                       eta_seed: float, e_max: float) -> "np.ndarray":
    '''First-order corrected resonance widths at every CAP solution.

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
    widths : numpy.ndarray
        Real array of shape (n_solutions,) holding Gamma = -2 Im(E - eta dE/deta)
        in hartree at each CAP solution, ordered by increasing CAP strength. Empty
        if the velocity has no interior local minimum on the grid.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have n
        rows, if etas is not a strictly increasing array of at least 3 non-negative
        points, if e_max is not positive, or if no eigenvalue at the seed lies
        strictly between zero and e_max.
    '''
    return widths

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_deperturbed_widths(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                               eta_seed: float, e_max: float) -> "np.ndarray":
    import numpy as np
    g = np.asarray(etas, dtype=float)
    I = np.asarray(integrals, dtype=float)
    S, W = I[0], I[2]
    _, D, vecs = _track_branch(integrals, transform, etas, eta_seed, e_max)
    v = np.abs(g * D)

    widths = []
    for i in _velocity_minima(v):
        eta_opt = _parabolic_eta(g, v, i)
        vals, C = _cap_solve(integrals, transform, eta_opt)
        idx = int(np.argmax(np.abs(vecs[i] @ S @ C)))
        c = C[:, idx]
        energy = vals[idx]
        deriv = -1j * (c @ W @ c)
        widths.append(-2.0 * (energy - eta_opt * deriv).imag)
    return np.array(widths, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the production grid ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'deperturbed_widths(I_model.copy(), X_model.copy(), g.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_deperturbed_widths(I_gold.copy(), X_gold.copy(), g.copy(), 0.15, 0.88291066)',
            "tol": 1e-08,
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
            "call": 'deperturbed_widths(I_model.copy(), X_model.copy(), gs.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_deperturbed_widths(I_gold.copy(), X_gold.copy(), gs.copy(), 0.15, 0.88291066)',
            "tol": 1e-08,
        },
        # --- Boundary: a grid truncated before the outermost solution ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
gt = np.geomspace(1e-4, 0.5, 400)
""",
            "call": 'deperturbed_widths(I_model.copy(), X_model.copy(), gt.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_deperturbed_widths(I_gold.copy(), X_gold.copy(), gt.copy(), 0.15, 0.88291066)',
            "tol": 1e-08,
        },
        # --- Edge: a window with no interior minimum, so nothing is returned ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
gn = np.geomspace(0.3, 0.6, 12)
""",
            "call": 'deperturbed_widths(I_model.copy(), X_model.copy(), gn.copy(), 0.4, 0.88291066)',
            "gold_call": '_oracle_deperturbed_widths(I_gold.copy(), X_gold.copy(), gn.copy(), 0.4, 0.88291066)',
            "tol": 1e-08,
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
            "call": 'deperturbed_widths(I_model.copy(), X_model.copy(), gb.copy(), 0.3, 1.8394)',
            "gold_call": '_oracle_deperturbed_widths(I_gold.copy(), X_gold.copy(), gb.copy(), 0.3, 1.8394)',
            "tol": 1e-08,
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
            "call": 'deperturbed_widths(I_model.copy(), X_model.copy(), g2.copy(), 0.15, 0.9197005)',
            "gold_call": '_oracle_deperturbed_widths(I_gold.copy(), X_gold.copy(), g2.copy(), 0.15, 0.9197005)',
            "tol": 1e-08,
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
            "call": 'deperturbed_widths(I_model.copy(), X_model.copy(), g3.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_deperturbed_widths(I_gold.copy(), X_gold.copy(), g3.copy(), 0.15, 0.88291066)',
            "tol": 1e-08,
        },
        # --- Invalid: a one-dimensional requirement violated by a 2-D grid ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
bad = np.geomspace(1e-3, 1.0, 30).reshape(5, 6)
def run_model():
    try:
        deperturbed_widths(I_model.copy(), X_model.copy(), bad.copy(), 0.15, 0.88291066)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_deperturbed_widths(I_gold.copy(), X_gold.copy(), bad.copy(), 0.15, 0.88291066)
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

"""
The slope of the logarithmic energy velocity with respect to ln(eta) along the followed branch, from analytic first and second derivatives of the resonance energy.

A CAP solution found on a grid is only as sharp as the grid. To say where a stationary point of the velocity actually lies, and above all to follow what happens to it when the Hamiltonian is changed until it disappears, the velocity has to be treated as a smooth function of the CAP strength and its slope has to be available at any strength, on or off the grid. The natural variable is the logarithm of the strength, the same variable in which the solutions are refined, so the object computed here is

  dv/d ln(eta),  with  v = | eta dE/deta |

evaluated for the tracked resonance. A local minimum of the velocity is a point where this slope crosses zero from below, a local maximum a point where it crosses from above.

The slope involves the second derivative of the complex eigenvalue with respect to eta as well as the first. Neither is to be obtained by differencing. The first is the c-product Hellmann-Feynman expectation value already used for the velocity. The second follows from second-order perturbation theory for the complex symmetric generalised problem, written in the same bilinear c-product with every eigenvector normalised as c^T S c = 1; within the retained subspace the sum over the other eigenstates is complete, so the result is exact there rather than approximate. The derivation is left to the implementation; the point to respect is that the metric is the c-product throughout, with no complex conjugation anywhere, including in the couplings between the resonance and the other states.

The slope refers to the same physical state at every grid point, the one accepted by the seed-and-overlap tracking procedure. Because v vanishes at eta = 0, where its logarithmic slope is undefined, the grid must be strictly positive. The Hamiltonian block of the integrals array is used exactly as supplied, so a Hamiltonian that already carries an additional real one-body term is handled without any change.

Returns
-------
numpy.ndarray of shape (n_eta,), real: dv/d ln(eta) of the logarithmic energy velocity of the followed branch at each grid point
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def branch_velocity_slope(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                          eta_seed: float, e_max: float) -> "np.ndarray":
    '''Slope dv/d ln(eta) of the logarithmic energy velocity along the followed branch.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2; the second
        block is used as the Hamiltonian exactly as supplied.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    etas : numpy.ndarray
        Strictly increasing grid of strictly positive CAP strengths, at least 3
        points.
    eta_seed : float
        CAP strength at which the resonance is identified; snapped to the nearest
        grid point.
    e_max : float
        Top of the barrier, the upper edge of the seed search window; must be
        positive.

    Returns
    -------
    slope : numpy.ndarray
        Real array of shape (n_eta,) holding d|eta dE/deta| / d ln(eta) along the
        branch, with dE/deta from the c-product Hellmann-Feynman theorem and
        d2E/deta2 from c-product second-order perturbation theory over all
        retained eigenstates, neither obtained by differencing.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have n
        rows, if etas is not a strictly increasing array of at least 3 strictly
        positive points, if e_max is not positive, or if no eigenvalue at the seed
        lies strictly between zero and e_max.
    '''
    return slope

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _slope_at(integrals, transform, eta, reference):
    """dv/dln(eta) of the state of largest c-product overlap with reference, and its vector."""
    import numpy as np
    I = np.asarray(integrals, dtype=float)
    S, W = I[0], I[2]
    vals, C = _cap_solve(integrals, transform, eta)
    k = int(np.argmax(np.abs(reference @ S @ C)))
    Wm = C.T @ W @ C
    d1 = -1j * Wm[k, k]
    gaps = vals[k] - vals
    gaps[k] = np.inf
    d2 = -2.0 * np.sum(Wm[k, :] ** 2 / gaps)
    q = eta * d1
    dq = eta * d1 + eta * eta * d2
    return float((np.conj(q) * dq).real / abs(q)), C[:, k]


def _oracle_branch_velocity_slope(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                                  eta_seed: float, e_max: float) -> "np.ndarray":
    import numpy as np
    g = np.asarray(etas, dtype=float)
    if g.ndim == 1 and g.size >= 1 and np.all(np.isfinite(g)) and g[0] <= 0.0:
        raise ValueError("etas must be strictly positive")
    _, _, vecs = _track_branch(integrals, transform, etas, eta_seed, e_max)
    return np.array([_slope_at(integrals, transform, float(g[i]), vecs[i])[0]
                     for i in range(g.size)], dtype=float)

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
""",
            "call": 'branch_velocity_slope(I_model.copy(), X_model.copy(), g.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_branch_velocity_slope(I_gold.copy(), X_gold.copy(), g.copy(), 0.15, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Normal: a coarse grid, whose points sit far from the fine-grid stationary points ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'branch_velocity_slope(I_model.copy(), X_model.copy(), gs.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_branch_velocity_slope(I_gold.copy(), X_gold.copy(), gs.copy(), 0.15, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Normal: a Hamiltonian block that already carries an extra real multiple of W ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
J_model = I_model.copy()
J_model[1] = I_model[1] + 0.16 * I_model[2]
J_gold = I_gold.copy()
J_gold[1] = I_gold[1] + 0.16 * I_gold[2]
g = np.geomspace(1e-4, 4.0, 1200)
""",
            "call": 'branch_velocity_slope(J_model.copy(), X_model.copy(), g.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_branch_velocity_slope(J_gold.copy(), X_gold.copy(), g.copy(), 0.15, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Edge: seeded at the far end of a coarse grid ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'branch_velocity_slope(I_model.copy(), X_model.copy(), gs.copy(), 3.0, 0.88291066)',
            "gold_call": '_oracle_branch_velocity_slope(I_gold.copy(), X_gold.copy(), gs.copy(), 3.0, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Edge: a different potential and a larger, tighter basis ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.01, 1.9, 24, 2.0, 0.8, 1.5)
I_gold = _oracle_basis_integrals(0.01, 1.9, 24, 2.0, 0.8, 1.5)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g2 = np.geomspace(1e-4, 4.0, 700)
""",
            "call": 'branch_velocity_slope(I_model.copy(), X_model.copy(), g2.copy(), 0.15, 0.9197005)',
            "gold_call": '_oracle_branch_velocity_slope(I_gold.copy(), X_gold.copy(), g2.copy(), 0.15, 0.9197005)',
            "tol": 1e-09,
        },
        # --- Edge: CAP onset inside the barrier ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 1.2)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 1.2)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g3 = np.geomspace(1e-4, 4.0, 500)
""",
            "call": 'branch_velocity_slope(I_model.copy(), X_model.copy(), g3.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_branch_velocity_slope(I_gold.copy(), X_gold.copy(), g3.copy(), 0.15, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Invalid: a grid that starts at eta = 0, where the logarithmic slope is undefined ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g0 = np.concatenate(([0.0], np.geomspace(1e-3, 3.0, 89)))
def run_model():
    try:
        branch_velocity_slope(I_model.copy(), X_model.copy(), g0.copy(), 0.15, 0.88291066)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_branch_velocity_slope(I_gold.copy(), X_gold.copy(), g0.copy(), 0.15, 0.88291066)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": 'run_model()',
            "gold_call": 'run_oracle()',
        },
        # --- Invalid: a two-point grid is too short to carry a trajectory ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
short = np.array([0.1, 0.2])
def run_model():
    try:
        branch_velocity_slope(I_model.copy(), X_model.copy(), short.copy(), 0.15, 0.88291066)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_branch_velocity_slope(I_gold.copy(), X_gold.copy(), short.copy(), 0.15, 0.88291066)
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

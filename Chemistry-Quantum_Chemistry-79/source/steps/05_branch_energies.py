"""
The complex energy of the resonance followed continuously along the whole CAP-strength grid.

Sweeping eta and collecting the eigenvalue of the resonance traces out the eta-trajectory in the complex plane, the central object of the CAP method. Building it requires following one state, not one index: as eta grows the resonance exchanges character with the discretised continuum states around it, so an eigenvalue picked by position in a sorted list jumps between physically different states and the trajectory acquires spurious structure.

The branch is therefore identified once and then propagated. At a seed strength eta_seed, chosen in the region where the resonance is already well separated, the resonance is the state of smallest spatial extent among those whose real part lies between the detachment threshold at zero and the top of the barrier; states above the barrier are not trapped and states below zero are not in the continuum at all. From that seed the branch is continued outwards in both directions, to smaller and to larger eta, by choosing at each new grid point the eigenvector of largest c-product overlap magnitude |c_prev^T S c| with the one already accepted. Because neighbouring grid points differ only slightly, this overlap is close to one for the correct partner and much smaller for every other state.

The seed strength is snapped to the nearest point of the supplied grid, so the sweep is carried out entirely on grid points.

Returns
-------
numpy.ndarray of shape (n_eta,), complex: the resonance eigenvalue E(eta) at each grid point, in the order of the supplied grid
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def branch_energies(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                    eta_seed: float, e_max: float) -> "np.ndarray":
    '''Resonance energy along the CAP-strength grid.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    etas : numpy.ndarray
        Strictly increasing grid of non-negative CAP strengths, at least 3 points.
    eta_seed : float
        CAP strength at which the resonance is identified; snapped to the nearest
        grid point.
    e_max : float
        Upper edge of the search window for the seed, the top of the barrier;
        must be positive.

    Returns
    -------
    energies : numpy.ndarray
        Complex array of shape (n_eta,) holding E(eta) along the followed branch.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have n
        rows, if etas is not a strictly increasing array of at least 3 non-negative
        points, if e_max is not positive, or if no eigenvalue at the seed lies
        strictly between zero and e_max.
    '''
    return energies

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _check_grid(etas, e_max):
    import numpy as np
    g = np.asarray(etas, dtype=float)
    if g.ndim != 1 or g.size < 3:
        raise ValueError("etas must be a one-dimensional grid of at least 3 points")
    if not np.all(np.isfinite(g)) or g[0] < 0.0:
        raise ValueError("etas must be finite and non-negative")
    if not np.all(np.diff(g) > 0.0):
        raise ValueError("etas must be strictly increasing")
    if not np.isfinite(e_max) or e_max <= 0.0:
        raise ValueError("e_max must be a positive finite number")
    return g


def _track_branch(integrals, transform, etas, eta_seed, e_max):
    """Follow the resonance outwards from the seed; return E, dE/deta and the vectors."""
    import numpy as np
    g = _check_grid(etas, e_max)
    I = np.asarray(integrals, dtype=float)
    S, W = I[0], I[2]

    js = int(np.argmin(np.abs(g - float(eta_seed))))
    vals, C = _cap_solve(integrals, transform, g[js])
    r2 = _oracle_resonance_extents(integrals, transform, g[js])
    window = np.where((vals.real > 0.0) & (vals.real < float(e_max)))[0]
    if window.size == 0:
        raise ValueError("no eigenvalue at the seed lies between zero and e_max")
    k = int(window[int(np.argmin(r2[window]))])

    n = g.size
    E = np.empty(n, dtype=complex)
    D = np.empty(n, dtype=complex)
    vecs = [None] * n

    def _record(i, vals_i, C_i, idx):
        c = C_i[:, idx]
        E[i] = vals_i[idx]
        D[i] = -1j * (c @ W @ c)
        vecs[i] = c
        return c

    prev = _record(js, vals, C, k)
    for i in range(js - 1, -1, -1):
        vals_i, C_i = _cap_solve(integrals, transform, g[i])
        idx = int(np.argmax(np.abs(prev @ S @ C_i)))
        prev = _record(i, vals_i, C_i, idx)
    prev = vecs[js]
    for i in range(js + 1, n):
        vals_i, C_i = _cap_solve(integrals, transform, g[i])
        idx = int(np.argmax(np.abs(prev @ S @ C_i)))
        prev = _record(i, vals_i, C_i, idx)
    return E, D, vecs


def _oracle_branch_energies(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                            eta_seed: float, e_max: float) -> "np.ndarray":
    import numpy as np
    return _track_branch(integrals, transform, etas, eta_seed, e_max)[0]

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
            "call": 'branch_energies(I_model.copy(), X_model.copy(), g.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_branch_energies(I_gold.copy(), X_gold.copy(), g.copy(), 0.15, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Normal: a coarse grid seeded at the same place ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'branch_energies(I_model.copy(), X_model.copy(), gs.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_branch_energies(I_gold.copy(), X_gold.copy(), gs.copy(), 0.15, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Boundary: seeded at the very last grid point, so the sweep runs one way ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'branch_energies(I_model.copy(), X_model.copy(), gs.copy(), 3.0, 0.88291066)',
            "gold_call": '_oracle_branch_energies(I_gold.copy(), X_gold.copy(), gs.copy(), 3.0, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Edge: a seed far below the trajectory structure, where mixing is strongest ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'branch_energies(I_model.copy(), X_model.copy(), gs.copy(), 0.001, 0.88291066)',
            "gold_call": '_oracle_branch_energies(I_gold.copy(), X_gold.copy(), gs.copy(), 0.001, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Edge: a narrower window that excludes the upper continuum states ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'branch_energies(I_model.copy(), X_model.copy(), gs.copy(), 0.15, 0.60)',
            "gold_call": '_oracle_branch_energies(I_gold.copy(), X_gold.copy(), gs.copy(), 0.15, 0.60)',
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
            "call": 'branch_energies(I_model.copy(), X_model.copy(), gb.copy(), 0.3, 1.8394)',
            "gold_call": '_oracle_branch_energies(I_gold.copy(), X_gold.copy(), gb.copy(), 0.3, 1.8394)',
            "tol": 1e-09,
        },
        # --- Invalid: a grid that is not strictly increasing ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
bad = np.array([0.1, 0.1, 0.2])
def run_model():
    try:
        branch_energies(I_model.copy(), X_model.copy(), bad.copy(), 0.15, 0.88291066)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_branch_energies(I_gold.copy(), X_gold.copy(), bad.copy(), 0.15, 0.88291066)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": 'run_model()',
            "gold_call": 'run_oracle()',
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
            "call": 'branch_energies(I_model.copy(), X_model.copy(), g2.copy(), 0.15, 0.9197005)',
            "gold_call": '_oracle_branch_energies(I_gold.copy(), X_gold.copy(), g2.copy(), 0.15, 0.9197005)',
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
            "call": 'branch_energies(I_model.copy(), X_model.copy(), g3.copy(), 0.15, 0.88291066)',
            "gold_call": '_oracle_branch_energies(I_gold.copy(), X_gold.copy(), g3.copy(), 0.15, 0.88291066)',
            "tol": 1e-09,
        },
        # --- Invalid: a non-positive barrier top ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
def run_model():
    try:
        branch_energies(I_model.copy(), X_model.copy(), gs.copy(), 0.15, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_branch_energies(I_gold.copy(), X_gold.copy(), gs.copy(), 0.15, 0.0)
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

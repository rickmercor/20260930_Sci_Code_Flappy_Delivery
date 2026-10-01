"""
Spatial extent of every CAP eigenstate, used to tell the compact resonance apart from the diffuse discretised continuum.

In a finite basis the electronic continuum is represented by a handful of square-integrable states spread over the whole box, while a shape resonance is trapped behind the barrier and stays compact. The cleanest discriminator is therefore the second moment of the state,

  <x^2> = c^T X2 c

evaluated with the same bilinear c-product used to normalise the eigenvectors, c^T S c = 1, and with the real part taken because the c-product expectation value of a complex symmetric problem is complex in general. At a CAP strength where the trajectory is well developed, the resonance of this model sits near <x^2> of order one while every continuum-like partner lies one to two orders of magnitude higher, so the separation is unambiguous.

The ordering of the returned extents matches the ordering of the eigenvalues from the previous step, namely increasing real part with ties broken by increasing imaginary part, so that the two arrays can be read side by side.

Returns
-------
numpy.ndarray of shape (n_kept,), real: the expectation value <x^2> of each CAP eigenstate, in the same order as the eigenvalues
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def resonance_extents(integrals: "np.ndarray", transform: "np.ndarray", eta: float) -> "np.ndarray":
    '''Second spatial moment of each CAP eigenstate.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    eta : float
        CAP strength; must not be negative.

    Returns
    -------
    extents : numpy.ndarray
        Real array of shape (n_kept,) holding Re(c^T X2 c) for each eigenvector,
        normalised by c^T S c = 1 and ordered exactly as the eigenvalues.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have
        n rows, or if eta is negative.
    '''
    return extents

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_resonance_extents(integrals: "np.ndarray", transform: "np.ndarray", eta: float) -> "np.ndarray":
    import numpy as np
    I = np.asarray(integrals, dtype=float)
    # Step 3 defines the public ordering of the CAP states. The eigensolver
    # also supplies the vectors needed here, so align those vectors to the
    # eigenvalues returned by Step 3 before forming the extents.
    target = np.asarray(_oracle_cap_eigenvalues(integrals, transform, eta), dtype=complex)
    vals, C_raw = _cap_solve(integrals, transform, eta)
    remaining = list(range(vals.size))
    order = []
    for value in target:
        j = min(remaining, key=lambda k: abs(vals[k] - value))
        order.append(j)
        remaining.remove(j)
    C = C_raw[:, order]
    return np.einsum('ij,jk,ki->i', C.T, I[3], C).real

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: the seed point, where the resonance is the most compact state ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'resonance_extents(I_model.copy(), X_model.copy(), 0.15)',
            "gold_call": '_oracle_resonance_extents(I_gold.copy(), X_gold.copy(), 0.15)',
            "tol": 1e-08,
        },
        # --- Normal: a weaker CAP, where the compact state is still distinguishable ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'resonance_extents(I_model.copy(), X_model.copy(), 0.05)',
            "gold_call": '_oracle_resonance_extents(I_gold.copy(), X_gold.copy(), 0.05)',
            "tol": 1e-08,
        },
        # --- Boundary: eta = 0, all states real and the continuum maximally diffuse ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'resonance_extents(I_model.copy(), X_model.copy(), 0.0)',
            "gold_call": '_oracle_resonance_extents(I_gold.copy(), X_gold.copy(), 0.0)',
            "tol": 1e-08,
        },
        # --- Edge: a strong CAP that compresses every state ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'resonance_extents(I_model.copy(), X_model.copy(), 3.5)',
            "gold_call": '_oracle_resonance_extents(I_gold.copy(), X_gold.copy(), 3.5)',
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
            "call": 'resonance_extents(I_model.copy(), X_model.copy(), 0.4)',
            "gold_call": '_oracle_resonance_extents(I_gold.copy(), X_gold.copy(), 0.4)',
            "tol": 1e-08,
        },
        # --- Invalid: a negative CAP strength ---
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
        resonance_extents(I_model.copy(), X_model.copy(), -1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_resonance_extents(I_gold.copy(), X_gold.copy(), -1.0)
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

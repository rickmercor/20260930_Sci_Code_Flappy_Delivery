"""
Complex eigenvalues of the CAP-augmented Hamiltonian at one value of the absorbing-potential strength.

The absorbing potential is switched on by perturbing the field-free Hamiltonian with a purely imaginary one-body term,

  H(eta) = H0 - i eta W

where eta is the CAP strength. The resulting operator is complex symmetric rather than Hermitian, so its eigenvalues leave the real axis in conjugate-asymmetric fashion and a metastable state appears as E = E_R - i Gamma / 2, with E_R the resonance position relative to the detachment threshold and Gamma the width, whose reciprocal is the lifetime.

In the non-orthogonal primitive basis the problem to solve is generalised, H(eta) C = E S C. Applying the canonical transformation X reduces it to the ordinary complex-symmetric problem X^T H(eta) X C' = E C', after which the primitive-basis coefficients are recovered as C = X C'. A complex symmetric operator is not normal, so the relevant inner product is the bilinear c-product u^T v, taken without complex conjugation; the eigenvectors are accordingly normalised as c^T S c = 1 rather than by the Hermitian norm. Using the conjugated product here does not merely rescale the result, it destroys the analytic structure the method depends on.

Eigenvalues are returned sorted by increasing real part, ties broken by increasing imaginary part.

Returns
-------
numpy.ndarray of shape (n_kept,), complex: eigenvalues of H(eta) in the retained subspace, sorted by increasing real part and then by increasing imaginary part
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import scipy.linalg as sla

def cap_eigenvalues(integrals: "np.ndarray", transform: "np.ndarray", eta: float) -> "np.ndarray":
    '''Complex eigenvalues of H(eta) = H0 - i eta W.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2 as produced
        by basis_integrals.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    eta : float
        CAP strength; must not be negative.

    Returns
    -------
    eigenvalues : numpy.ndarray
        Complex array of shape (n_kept,), sorted by increasing real part with
        ties broken by increasing imaginary part.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have
        n rows, or if eta is negative.
    '''
    return eigenvalues

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _check_cap_inputs(integrals, transform, eta):
    import numpy as np
    I = np.asarray(integrals, dtype=float)
    X = np.asarray(transform, dtype=float)
    if I.ndim != 3 or I.shape[0] != 4 or I.shape[1] != I.shape[2]:
        raise ValueError("integrals must have shape (4, n, n)")
    if X.ndim != 2 or X.shape[0] != I.shape[1]:
        raise ValueError("transform must have as many rows as the primitive basis size")
    if not np.isfinite(eta) or eta < 0.0:
        raise ValueError("eta must be a non-negative finite number")
    return I, X, float(eta)


def _cap_solve(integrals, transform, eta):
    """Sorted eigenvalues and c-product normalised primitive-basis eigenvectors."""
    import numpy as np
    import scipy.linalg as sla
    I, X, e = _check_cap_inputs(integrals, transform, eta)
    S, H0, W = I[0], I[1], I[2]
    vals, Cp = sla.eig(X.T @ (H0 - 1j * e * W) @ X)
    C = X @ Cp
    norms = np.einsum('ij,jk,ki->i', C.T, S, C)
    C = C / np.sqrt(norms)
    order = np.lexsort((vals.imag, vals.real))
    return vals[order], C[:, order]


def _oracle_cap_eigenvalues(integrals: "np.ndarray", transform: "np.ndarray", eta: float) -> "np.ndarray":
    import numpy as np
    return _cap_solve(integrals, transform, eta)[0]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: a CAP strength in the region where the resonance is well separated ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'cap_eigenvalues(I_model.copy(), X_model.copy(), 0.15)',
            "gold_call": '_oracle_cap_eigenvalues(I_gold.copy(), X_gold.copy(), 0.15)',
            "tol": 1e-09,
        },
        # --- Normal: a strong CAP, near the outermost solution of the trajectory ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'cap_eigenvalues(I_model.copy(), X_model.copy(), 2.15)',
            "gold_call": '_oracle_cap_eigenvalues(I_gold.copy(), X_gold.copy(), 2.15)',
            "tol": 1e-09,
        },
        # --- Boundary: eta = 0, where the spectrum must be exactly real ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'cap_eigenvalues(I_model.copy(), X_model.copy(), 0.0)',
            "gold_call": '_oracle_cap_eigenvalues(I_gold.copy(), X_gold.copy(), 0.0)',
            "tol": 1e-10,
        },
        # --- Edge: a very weak CAP, where resonance and continuum are still mixed ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
I_gold = _oracle_basis_integrals(0.02, 2.0, 20, 1.2, 0.5, 2.0)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
g = np.geomspace(1e-4, 4.0, 1200)
gs = np.geomspace(1e-3, 3.0, 90)
""",
            "call": 'cap_eigenvalues(I_model.copy(), X_model.copy(), 1e-4)',
            "gold_call": '_oracle_cap_eigenvalues(I_gold.copy(), X_gold.copy(), 1e-4)',
            "tol": 1e-10,
        },
        # --- Edge: a different basis and potential entirely ---
        {
            "setup": """import numpy as np
I_model = basis_integrals(0.05, 2.5, 9, 2.0, 0.8, 3.5)
I_gold = _oracle_basis_integrals(0.05, 2.5, 9, 2.0, 0.8, 3.5)
X_model = canonical_transform(I_model[0].copy(), 1e-8)
X_gold = _oracle_canonical_transform(I_gold[0].copy(), 1e-8)
gb = np.geomspace(1e-3, 2.0, 70)
""",
            "call": 'cap_eigenvalues(I_model.copy(), X_model.copy(), 0.4)',
            "gold_call": '_oracle_cap_eigenvalues(I_gold.copy(), X_gold.copy(), 0.4)',
            "tol": 1e-10,
        },
        # --- Invalid: a negative CAP strength has no physical meaning ---
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
        cap_eigenvalues(I_model.copy(), X_model.copy(), -0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_cap_eigenvalues(I_gold.copy(), X_gold.copy(), -0.1)
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

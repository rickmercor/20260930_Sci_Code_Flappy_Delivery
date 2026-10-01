"""
Step 05: Complex-rotated L = 1 triplet levels and coefficients. Complex-rotated L = 1 triplet levels and expansion coefficients of a two-electron atom in a correlated-Gaussian basis of one parity.

A level that lies above a dissociation or ionization threshold but cannot break up sits in the same energy range as the continuum of the opposite parity, and the final states of its radiative decay run from discrete bound levels through that continuum. Both kinds of final state have to come out of one calculation. Rotating the electron coordinates into the complex plane by a real angle turns the continuum into a branch that leaves the real axis while bound levels stay where they are, so a single square-integrable basis produces both, at the price of a Hamiltonian that is complex symmetric rather than Hermitian.



A complex symmetric matrix has no orthonormal eigenbasis in the usual sense. Eigenvectors are instead paired with their own transposes rather than their conjugate transposes, the so-called c-product, under which the eigenvectors of distinct eigenvalues are orthogonal and a state can be scaled to unit c-norm. Rotated bound-state eigenvalues are real to the accuracy of the basis and independent of the angle, while the rotated continuum eigenvalues fan out below the real axis; the physical results extracted from them should be stationary against the angle over a plateau, which is the standard check that the basis supports the rotated states.



Correlated Gaussians with exponents spread over several decades become nearly linearly dependent long before they become complete, so the overlap matrix is badly conditioned and some directions of the span carry no usable information in double precision. The angle enters only through the Hamiltonian, since the basis functions themselves are real.

Returns
-------
numpy.ndarray of shape (k, n + 1): complex-rotated eigenvalue in hartree and the coefficients of that eigenstate, one row per retained state
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rotated_levels(exps: "np.ndarray", kind: str, Z: float, theta: float, cut: float) -> "np.ndarray":
    '''Complex-rotated L = 1 triplet eigenvalues and expansion coefficients for a two-electron atom.

    Parameters
    ----------
    exps : np.ndarray
        Array of shape (n, 3), n >= 1, of exponents (a, b, c) in bohr^-2; finite, non-negative, a b + c (a + b) > 0.
    kind : str
        "even" for the basis phi = (1 - P12)[(r1_vec x r2_vec)_z g] or "odd" for phi = (1 - P12)[z1 g], with
        g = exp(-a r1^2 - b r2^2 - c r12^2).
    Z : float
        Nuclear charge, Z > 0, of an infinitely heavy nucleus.
    theta : float
        Rotation angle in radians, 0 <= theta < pi/4. The electron coordinates are scaled by exp(i theta), so the
        kinetic and potential parts of H = -(nabla_1^2 + nabla_2^2)/2 - Z/r1 - Z/r2 + 1/r12 are multiplied by
        exp(-2 i theta) and exp(-i theta) respectively. theta = 0 gives the ordinary real variational problem.
    cut : float
        Relative threshold, 0 < cut < 1. Eigenvectors of the overlap matrix of the unnormalized basis functions (as
        defined for ecg_overlap) with eigenvalue s <= cut * max(s) are removed before the rotated Hamiltonian is
        diagonalized in the remaining space.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (k, n + 1), one row per retained eigenstate, where k is the number of retained overlap
        eigenvectors. Element [j, 0] is the eigenvalue E_j in hartree and elements [j, 1:] are the coefficients of
        eigenstate j in the original unnormalized basis. Rows are sorted by ascending real part of E_j, ties broken by
        ascending imaginary part. Each coefficient vector is scaled so that c^T S c = 1 with S the overlap matrix of
        the unnormalized functions, and the remaining sign is fixed by requiring the coefficient of largest modulus to
        have a positive real part, or a positive imaginary part if that real part is zero.

    Raises
    ------
    ValueError
        If the exponents are malformed or non-integrable, kind is not "even" or "odd", Z is not positive, theta is
        outside [0, pi/4), or cut is not strictly between 0 and 1.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import scipy.linalg as spl


def _oracle_rotated_levels(exps: "np.ndarray", kind: str, Z: float, theta: float, cut: float) -> "np.ndarray":
    charge = float(Z)
    angle = float(theta)
    drop = float(cut)
    if not charge > 0.0:
        raise ValueError("Z must be positive")
    if not np.isfinite(angle) or not 0.0 <= angle < 0.25 * np.pi:
        raise ValueError("theta must lie in [0, pi/4)")
    if not 0.0 < drop < 1.0:
        raise ValueError("cut must lie strictly between 0 and 1")
    overlap = _oracle_ecg_overlap(exps, exps, kind)
    hamiltonian = (np.exp(-2.0j * angle) * _oracle_ecg_kinetic(exps, exps, kind)
                   + np.exp(-1.0j * angle) * _oracle_ecg_coulomb(exps, exps, kind, charge))
    s_val, s_vec = np.linalg.eigh(0.5 * (overlap + overlap.T))
    keep = s_val > drop * s_val.max()
    proj = s_vec[:, keep] / np.sqrt(s_val[keep])
    small = proj.T @ (0.5 * (hamiltonian + hamiltonian.T)) @ proj
    values, vectors = spl.eig(0.5 * (small + small.T))
    vectors = vectors / np.sqrt(np.sum(vectors * vectors, axis=0))
    order = np.lexsort((values.imag, values.real))
    values, vectors = values[order], vectors[:, order]
    coeff = proj @ vectors
    lead = np.argmax(np.abs(coeff), axis=0)
    pivot = coeff[lead, np.arange(coeff.shape[1])]
    flip = np.where(pivot.real != 0.0, np.sign(pivot.real), np.sign(pivot.imag))
    flip = np.where(flip == 0.0, 1.0, flip)
    coeff = coeff * flip
    return np.column_stack([values, coeff.T]).astype(complex)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    even_setup = """import numpy as np
g = [0.08, 0.25, 0.8, 2.5]
e = np.array([(g[i], g[j], c) for i in range(4) for j in range(i, 4) for c in (0.0, 0.15)])
def parts(table):
    t = np.asarray(table, dtype=complex)
    return np.concatenate([t[:, 0].real, t[:, 0].imag, np.abs(t[:, 1:]).ravel()])
"""
    odd_setup = """import numpy as np
inner, outer = [0.6, 2.4, 9.0, 36.0], [0.015, 0.05, 0.15, 0.5]
rows = [(a, b, 0.0) for a in inner for b in outer] + [(b, a, 0.0) for a in inner for b in outer]
rows += [(a, a, 0.2) for a in (0.1, 0.5)]
e = np.array(rows)
def parts(table):
    t = np.asarray(table, dtype=complex)[:8]
    return np.concatenate([t[:, 0].real, t[:, 0].imag, np.abs(t[:, 1:]).ravel()])
"""
    return [
        # --- Normal: helium 2p^2-like 3Pe manifold, unrotated, eigenvalues and coefficient moduli ---
        {"setup": even_setup, "call": "parts(rotated_levels(e.copy(), 'even', 2.0, 0.0, 1e-12))",
         "gold_call": "parts(_oracle_rotated_levels(e.copy(), 'even', 2.0, 0.0, 1e-12))", "tol": 1e-7},
        # --- Normal: helium 1snp 3Po manifold rotated by 0.25 rad, the paper's angle ---
        {"setup": odd_setup, "call": "parts(rotated_levels(e.copy(), 'odd', 2.0, 0.25, 1e-12))",
         "gold_call": "parts(_oracle_rotated_levels(e.copy(), 'odd', 2.0, 0.25, 1e-12))", "tol": 1e-7},
        # --- Normal: the signed coefficients themselves, so the phase convention is graded ---
        {"setup": even_setup, "call": "np.asarray(rotated_levels(e.copy(), 'even', 2.0, 0.18, 1e-12), dtype=complex)[:4, 1:].real",
         "gold_call": "np.asarray(_oracle_rotated_levels(e.copy(), 'even', 2.0, 0.18, 1e-12), dtype=complex)[:4, 1:].real",
         "tol": 1e-7},
        # --- Boundary: bound levels of the rotated problem stay real, which is the theta-independence check ---
        {"setup": odd_setup, "call": "float(np.abs(np.asarray(rotated_levels(e.copy(), 'odd', 2.0, 0.3, 1e-10), dtype=complex)[0, 0].imag))",
         "gold_call": "float(np.abs(np.asarray(_oracle_rotated_levels(e.copy(), 'odd', 2.0, 0.3, 1e-10), dtype=complex)[0, 0].imag))",
         "tol": 1e-9},
        # --- Boundary: a loose threshold discards most of the span (Li+ scaling) ---
        {"setup": odd_setup + "e = e * 2.25\n", "call": "parts(rotated_levels(e.copy(), 'odd', 3.0, 0.2, 1e-4))",
         "gold_call": "parts(_oracle_rotated_levels(e.copy(), 'odd', 3.0, 0.2, 1e-4))", "tol": 1e-7},
        # --- Edge: a single basis function ---
        {"setup": "import numpy as np\ne = np.array([[0.3, 0.3, 0.05]])\n",
         "call": "np.asarray(rotated_levels(e.copy(), 'even', 2.0, 0.1, 1e-10), dtype=complex)[0, 0].real",
         "gold_call": "np.asarray(_oracle_rotated_levels(e.copy(), 'even', 2.0, 0.1, 1e-10), dtype=complex)[0, 0].real",
         "tol": 1e-9},
        # --- Error: rotation angle outside [0, pi/4) ---
        {"setup": "import numpy as np\ne = np.array([[0.3, 0.3, 0.05]])\n"
                  "def _probe(fn):\n    try:\n        fn(e, 'even', 2.0, 0.9, 1e-10)\n    except ValueError:\n        return 1\n    return 0\n",
         "call": "_probe(rotated_levels)", "gold_call": "_probe(_oracle_rotated_levels)"},
    ]

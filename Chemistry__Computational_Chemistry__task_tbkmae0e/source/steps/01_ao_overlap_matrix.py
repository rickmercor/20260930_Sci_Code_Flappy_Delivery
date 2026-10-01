"""
Implement ao_overlap_matrix, which returns the exact overlap matrix of a basis of normalized s-type primitive Gaussians.

Each basis function is a spherical Gaussian of the given exponent about the given center, scaled to unit self-overlap. Entry (mu, nu) of the overlap matrix is the integral over all space of the product of primitives mu and nu. For Gaussians this integral has a closed form, and the matrix is to be evaluated from it rather than by numerical quadrature.



The overlap matrix is the elementary one-electron matrix of a basis set. It is symmetric, has unit diagonal for normalized primitives, and is positive definite whenever the primitives are linearly independent; an even-tempered set of primitives sharing a center produces strongly off-diagonal overlaps and therefore a poorly conditioned matrix. Because this matrix collects the exact integrals of every product of two basis functions, it is the natural reference against which a real-space quadrature rule for those same products can be judged.

Returns
-------
np.ndarray of shape (n_ao, n_ao), float: the exact overlap matrix of the normalized s-type primitives, with unit diagonal
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def ao_overlap_matrix(ao_centers: "np.ndarray", ao_exponents: "np.ndarray") -> "np.ndarray":
    '''Exact analytic overlap matrix of normalized s-type primitive Gaussians.

    Parameters
    ----------
    ao_centers : np.ndarray
        Array of shape (n_ao, 3) holding the center of each primitive, in bohr.
        Must contain at least one row.
    ao_exponents : np.ndarray
        Array of shape (n_ao,) holding the Gaussian exponent of each primitive.
        Every exponent must be strictly positive.

    Returns
    -------
    overlap : np.ndarray
        Symmetric array of shape (n_ao, n_ao) of dtype float whose entry (mu, nu) is the
        exact overlap integral between primitives mu and nu. The diagonal is exactly one.

    Raises
    ------
    ValueError
        If ao_centers is not a two-dimensional array with three columns and at least one
        row, if ao_exponents is not a one-dimensional array of matching length, or if any
        exponent is not strictly positive.
    '''
    return overlap

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _validated_basis(ao_centers, ao_exponents):
    """Return the basis as float arrays, rejecting malformed or unphysical input."""
    centers = np.asarray(ao_centers, dtype=float)
    exponents = np.asarray(ao_exponents, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 3 or centers.shape[0] < 1:
        raise ValueError("ao_centers must have shape (n_ao, 3) with n_ao >= 1")
    if exponents.ndim != 1 or exponents.shape[0] != centers.shape[0]:
        raise ValueError("ao_exponents must have shape (n_ao,) matching ao_centers")
    if not np.all(exponents > 0.0):
        raise ValueError("every Gaussian exponent must be strictly positive")
    return centers, exponents


def _oracle_ao_overlap_matrix(ao_centers: "np.ndarray",
                              ao_exponents: "np.ndarray") -> "np.ndarray":
    centers, exponents = _validated_basis(ao_centers, ao_exponents)
    separation2 = np.sum((centers[:, None, :] - centers[None, :, :]) ** 2, axis=-1)
    total = exponents[:, None] + exponents[None, :]
    product = exponents[:, None] * exponents[None, :]
    overlap = (2.0 * np.sqrt(product) / total) ** 1.5 * np.exp(-product / total * separation2)
    return overlap

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    common = """import numpy as np
"""
    return [
        # --- Typical: three primitives on three distinct centers ---
        {
            "setup": common + """
ao_centers = np.array([[0.0, 0.0, 0.0],
                       [1.3, 0.0, 0.0],
                       [0.2, 1.4, 0.5]], dtype=float)
ao_exponents = np.array([0.7, 1.9, 0.45], dtype=float)
""",
            "call": "ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())",
            "gold_call": "_oracle_ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())",
        },
        # --- Boundary: coincident centers, where the exponential factor is exactly one ---
        {
            "setup": common + """
ao_centers = np.zeros((4, 3), dtype=float)
ao_exponents = np.array([0.18, 0.45, 1.125, 2.8125], dtype=float)
""",
            "call": "ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())",
            "gold_call": "_oracle_ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())",
        },
        # --- Edge: a single primitive, whose overlap matrix is exactly [[1.0]] ---
        {
            "setup": common + """
ao_centers = np.array([[0.4, -0.7, 2.0]], dtype=float)
ao_exponents = np.array([3.25], dtype=float)
""",
            "call": "ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())",
            "gold_call": "_oracle_ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())",
        },
        # --- Edge: widely separated primitives drive the overlap to zero ---
        {
            "setup": common + """
ao_centers = np.array([[0.0, 0.0, 0.0],
                       [18.0, 0.0, 0.0]], dtype=float)
ao_exponents = np.array([1.0, 1.0], dtype=float)
""",
            "call": "ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())",
            "gold_call": "_oracle_ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())",
        },
        # --- Invalid: a non-positive exponent ---
        {
            "setup": common + """
ao_centers = np.zeros((2, 3), dtype=float)
ao_exponents = np.array([1.0, 0.0], dtype=float)
def run_model():
    try:
        ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        _oracle_ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Invalid: mismatched number of centers and exponents ---
        {
            "setup": common + """
ao_centers = np.zeros((3, 3), dtype=float)
ao_exponents = np.array([1.0, 2.0], dtype=float)
def run_model():
    try:
        ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())
        return 0.0
    except ValueError:
        return 1.0
def run_oracle():
    try:
        _oracle_ao_overlap_matrix(ao_centers.copy(), ao_exponents.copy())
        return 0.0
    except ValueError:
        return 1.0
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

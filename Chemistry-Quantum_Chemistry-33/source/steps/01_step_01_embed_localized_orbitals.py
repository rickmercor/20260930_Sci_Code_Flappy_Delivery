"""
Validate coefficients with shape (P,A,R,M), integer shell_counts with shape (A,), and weights with shape (K,R). Zero every radial slot r >= shell_counts[a], then compute embedded[p,a,k,m] = sum_r weights[k,r]*padded[p,a,r,m]. Return padded, embedded, and dot(arange(1,N+1), embedded.ravel())/N in C order.

The source determines why element-dependent coefficient blocks are padded and why one shared equivariant map is applied to every orbital graph. The checksum is a disclosed benchmark convention.

Returns
-------
tuple : padded coefficients, embedded orbital states, and embedding checksum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def embed_localized_orbitals(coefficients: "np.ndarray", shell_counts: "np.ndarray", weights: "np.ndarray") -> tuple:
    """Return padded coefficients, embedded orbital graphs, and a checksum.

    Parameters
    ----------
    coefficients : np.ndarray
        Float array with shape (P, A, R, M). Entries at radial indices greater
        than or equal to shell_counts[a] are ignored as element-dependent padding.
    shell_counts : np.ndarray
        Integer array with shape (A,) and values in [1, R].
    weights : np.ndarray
        Shared equivariant radial map with shape (K, R).

    Returns
    -------
    padded : np.ndarray
        Coefficients after invalid radial slots are set to zero, shape (P,A,R,M).
    embedded : np.ndarray
        Shared-weight orbital graph states with shape (P,A,K,M).
    checksum : float
        Weighted checksum of embedded in C order.

    Raises
    ------
    ValueError
        If an input has an invalid shape, non-finite values, or an invalid shell count.
    """
    return None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_embed_localized_orbitals(coefficients: "np.ndarray", shell_counts: "np.ndarray", weights: "np.ndarray") -> tuple:
    coefficients = np.asarray(coefficients, dtype=float)
    shell_counts = np.asarray(shell_counts)
    weights = np.asarray(weights, dtype=float)
    if coefficients.ndim != 4 or min(coefficients.shape) < 1:
        raise ValueError("coefficients must have shape (P,A,R,M) with nonzero axes")
    p, a, r, m = coefficients.shape
    if shell_counts.shape != (a,) or not np.issubdtype(shell_counts.dtype, np.integer):
        raise ValueError("shell_counts must be an integer array of shape (A,)")
    if np.any(shell_counts < 1) or np.any(shell_counts > r):
        raise ValueError("each shell count must lie in [1,R]")
    if weights.ndim != 2 or weights.shape[1] != r or weights.shape[0] < 1:
        raise ValueError("weights must have shape (K,R), K>=1")
    if not np.all(np.isfinite(coefficients)) or not np.all(np.isfinite(weights)):
        raise ValueError("numeric inputs must be finite")
    mask = np.arange(r)[None, :] < shell_counts[:, None]
    padded = coefficients * mask[None, :, :, None]
    embedded = np.einsum("kr,parm->pakm", weights, padded, optimize=True)
    flat = embedded.ravel(order="C")
    checksum = float(np.dot(np.arange(1, flat.size + 1, dtype=float), flat) / flat.size)
    return padded, embedded, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    pack = lambda call: "(lambda z: [len(z),*z[0].shape,*z[1].shape,*np.concatenate([z[0].ravel(),z[1].ravel(),[z[2]]]).tolist()])(" + call + ")"
    err = "def et(fn,*a):\n try: fn(*a); return 0\n except ValueError: return 1\n"
    return [
        {"tol": 1e-10, "setup": "import numpy as np\nr=np.random.default_rng(3);c=r.normal(size=(5,4,3,3));s=np.array([3,2,1,3]);w=r.normal(size=(4,3))", "call": pack("embed_localized_orbitals(c,s,w)"), "gold_call": pack("_oracle_embed_localized_orbitals(c,s,w)")},
        {"tol": 1e-10, "setup": "import numpy as np\nr=np.random.default_rng(7);c=r.normal(size=(2,1,1,3));s=np.array([1]);w=r.normal(size=(2,1))", "call": pack("embed_localized_orbitals(c,s,w)"), "gold_call": pack("_oracle_embed_localized_orbitals(c,s,w)")},
        {"tol": 1e-10, "setup": "import numpy as np\nr=np.random.default_rng(11);c=r.normal(size=(7,3,4,1));s=np.array([1,4,2]);w=r.normal(size=(3,4))", "call": pack("embed_localized_orbitals(c,s,w)"), "gold_call": pack("_oracle_embed_localized_orbitals(c,s,w)")},
        {"tol": 0.0, "setup": "import numpy as np\n" + err + "c=np.zeros((2,2,3,3));s=np.array([3,0]);w=np.ones((2,3))", "call": "et(embed_localized_orbitals,c,s,w)", "gold_call": "et(_oracle_embed_localized_orbitals,c,s,w)"},
    ]

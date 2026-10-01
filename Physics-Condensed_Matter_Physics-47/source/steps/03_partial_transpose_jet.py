"""
Apply momentum sign reversal on the selected local sites to both covariance slices.

For a chosen subsystem, transposition reverses its momentum quadratures and leaves all position quadratures unchanged. If $T$ is the diagonal matrix of these signs, transform both slices by $\gamma^\Gamma=T\gamma T$ and $\dot\gamma^\Gamma=T\dot\gamma T$. Site indices refer to the local order of the covariance, independent of the original lattice coordinates.

Returns
-------
np.ndarray, shape (2, 2*m, 2*m), the partially transposed covariance and its bias derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def partial_transpose_jet(covariance_jet: "np.ndarray",
                          transpose_sites: "np.ndarray") -> "np.ndarray":
    """Apply a subsystem partial transpose to the value and bias derivative.

    Parameters
    ----------
    covariance_jet : np.ndarray
        Finite real array of shape (2, 2*m, 2*m), with m at least one, in grouped (all q, all p) order. Each slice must be symmetric within absolute tolerance 1e-10 and zero relative tolerance. The symmetric part of the value slice must be strictly positive definite. No quantum uncertainty-condition test is required.
    transpose_sites : np.ndarray
        One-dimensional array of distinct finite integer-valued local site indices in [0, m). Boolean entries are invalid. An empty array and the set of all local sites are both valid.

    Returns
    -------
    transposed_jet : np.ndarray
        Real array of the same shape, with momentum signs reversed on the selected sites in both the covariance and its derivative. The output has independent storage and neither input is modified.

    Raises
    ------
    ValueError
        If the covariance shape, realness, finiteness, symmetry or positive definiteness is invalid, or if the site indices violate their domain.
    """
    return transposed_jet

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_partial_transpose_jet(covariance_jet: "np.ndarray",
                                  transpose_sites: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np

    try:
        raw_jet = np.asarray(covariance_jet)
        raw_sites = np.asarray(transpose_sites)
    except (TypeError, ValueError) as exc:
        raise ValueError("The covariance jet and site indices must be numeric arrays") from exc
    if (raw_jet.ndim != 3 or raw_jet.shape[0] != 2 or
            raw_jet.shape[1] != raw_jet.shape[2] or raw_jet.shape[1] < 2 or
            raw_jet.shape[1] % 2 or raw_jet.dtype.kind not in "iuf"):
        raise ValueError("covariance_jet must be real with shape (2, 2*m, 2*m)")
    jet = np.array(raw_jet, dtype=float, copy=True)
    if (not np.all(np.isfinite(jet)) or
            not np.allclose(jet, jet.transpose(0, 2, 1), rtol=0.0, atol=1e-10)):
        raise ValueError("Both covariance slices must be finite and symmetric")
    try:
        np.linalg.cholesky(jet[0] / 2.0 + jet[0].T / 2.0)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The covariance value must be positive definite") from exc
    m = jet.shape[1] // 2
    if raw_sites.ndim != 1 or raw_sites.dtype.kind not in "iuf":
        raise ValueError("transpose_sites must be a one-dimensional integer-coordinate array")
    if any(isinstance(value, (bool, np.bool_))
           for value in np.asarray(transpose_sites, dtype=object).flat):
        raise ValueError("Boolean transpose indices are invalid")
    sites = np.array(raw_sites, dtype=float, copy=True)
    if (not np.all(np.isfinite(sites)) or np.any(sites != np.floor(sites)) or
            np.any(sites < 0) or np.any(sites >= m) or
            np.unique(sites).size != sites.size):
        raise ValueError("transpose_sites must contain distinct indices in [0, m)")
    signs = np.ones(2 * m, dtype=float)
    signs[m + sites.astype(int)] = -1.0
    transposed_jet = jet * signs[None, :, None] * signs[None, None, :]
    return transposed_jet

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return tests of local subsystem selection and value/tangent conjugation."""
    return [
        {
            "setup": """import numpy as np
symbols = _oracle_build_ness_symbols(0.45, 1.0, 3.2, 0.9, 128)
jet = _oracle_fourier_covariance(symbols, np.arange(-3, 3))
sites = np.array([3, 4, 5])
""",
            "call": "partial_transpose_jet(jet.copy(), sites.copy())",
            "gold_call": "_oracle_partial_transpose_jet(jet.copy(), sites.copy())",
        },
        {
            "setup": """import numpy as np
a = np.array([[1.1, 0.3, -0.2, 0.5], [0.0, 0.8, 0.6, -0.3], [0.0, 0.0, 1.4, 0.7], [0.0, 0.0, 0.0, 0.9]])
tangent = np.arange(16, dtype=float).reshape(4, 4) / 20.0
jet = np.stack((a @ a.T, tangent + tangent.T))
sites = np.array([], dtype=int)
""",
            "call": "partial_transpose_jet(jet.copy(), sites.copy())",
            "gold_call": "_oracle_partial_transpose_jet(jet.copy(), sites.copy())",
        },
        {
            "setup": """import numpy as np
a = np.array([[1.1, 0.3, -0.2, 0.5], [0.0, 0.8, 0.6, -0.3], [0.0, 0.0, 1.4, 0.7], [0.0, 0.0, 0.0, 0.9]])
tangent = np.arange(16, dtype=float).reshape(4, 4) / 20.0
jet = np.stack((a @ a.T, tangent + tangent.T))
sites = np.array([1, 0])
""",
            "call": "partial_transpose_jet(jet.copy(), sites.copy())",
            "gold_call": "_oracle_partial_transpose_jet(jet.copy(), sites.copy())",
        },
        {
            "setup": """import numpy as np
jet = np.array([[[1.3, 0.2], [0.2, 0.7]], [[-0.4, 0.6], [0.6, 0.9]]])
sites = np.array([0])
""",
            "call": "partial_transpose_jet(jet.copy(), sites.copy())",
            "gold_call": "_oracle_partial_transpose_jet(jet.copy(), sites.copy())",
        },
        {
            "setup": """import numpy as np
a = np.reshape(np.sin(np.arange(36)), (6, 6)) / 5.0
value = a @ a.T + 0.6*np.eye(6)
tangent = np.reshape(np.cos(np.arange(36)), (6, 6))
jet = np.stack((value, (tangent + tangent.T)/2.0))
sites = np.array([2.0, 0.0])
""",
            "call": "partial_transpose_jet(jet.copy(), sites.copy())",
            "gold_call": "_oracle_partial_transpose_jet(jet.copy(), sites.copy())",
        },
        {
            "setup": """import numpy as np
jet = np.stack((np.diag([1.0, -0.1]), np.eye(2)))
sites = np.array([0])
def _raises_value_error(fn):
    try:
        fn(jet.copy(), sites.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(partial_transpose_jet)",
            "gold_call": "_raises_value_error(_oracle_partial_transpose_jet)",
        },
        {
            "setup": """import numpy as np
jet = np.stack((np.eye(4), np.ones((4, 4))))
sites = np.array([0, 0])
def _raises_value_error(fn):
    try:
        fn(jet.copy(), sites.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(partial_transpose_jet)",
            "gold_call": "_raises_value_error(_oracle_partial_transpose_jet)",
        },
        {
            "setup": """import numpy as np
jet = np.stack((np.eye(4), np.ones((4, 4))))
jet[1, 0, 1] += 1e-4
sites = np.array([1])
def _raises_value_error(fn):
    try:
        fn(jet.copy(), sites.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(partial_transpose_jet)",
            "gold_call": "_raises_value_error(_oracle_partial_transpose_jet)",
        },
    ]

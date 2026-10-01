"""
Fourier-transform both symbol slices and restrict the resulting covariance jet to the ordered site set.

For integer site differences $d=x_i-x_j$ and the midpoint grid from step 01, form $Q_{ij}=\langle\cos(qd)s_{qq}\rangle$, $P_{ij}=\langle\cos(qd)s_{pp}\rangle$, and $R_{ij}=-\langle\sin(qd)s_{\mathrm{mixed}}\rangle$. Here the brackets denote the arithmetic mean over the supplied grid. The grouped canonical ordering is $(q_1,\ldots,q_m,p_1,\ldots,p_m)$, so $\gamma=\left(\begin{smallmatrix}Q\&R\\R^T\&P\end{smallmatrix}\right)$. Apply the same linear transform to the bias tangent. The symbol normalization already uses the symmetrized covariance convention; add no extra factor of one half.

Returns
-------
np.ndarray, shape (2, 2*m, 2*m), covariance and bias tangent in grouped q-then-p canonical order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def fourier_covariance(symbols: "np.ndarray", positions: "np.ndarray") -> "np.ndarray":
    """Fourier transform a symbol jet onto an ordered selection of lattice sites.

    Parameters
    ----------
    symbols : np.ndarray
        Finite real array of shape (2, nq, 3), with even nq at least eight. Axes contain (value, bias derivative), midpoint momentum, and (qq, pp, mixed_imag), respectively. The qq and pp value channels must be strictly positive; derivative channels may have either sign.
    positions : np.ndarray
        Nonempty one-dimensional array of distinct finite integer-valued site coordinates, in the desired output order. Their span must be strictly less than nq. Boolean entries are invalid.

    Returns
    -------
    covariance_jet : np.ndarray
        Finite real array of shape (2, 2*m, 2*m), where m is the number of positions. The first slice is the symmetrized covariance and the second is its bias derivative, both in grouped (all q, all p) order. No input is modified.

    Raises
    ------
    ValueError
        If symbols or positions violate the stated shape, finiteness, positivity, integer-coordinate, uniqueness or span conditions, or if the resulting covariance is not representable as finite floats.
    """
    return covariance_jet

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fourier_covariance(symbols: "np.ndarray", positions: "np.ndarray") -> "np.ndarray":
    """Reference implementation."""
    import numpy as np

    try:
        raw_symbols = np.asarray(symbols)
        raw_positions = np.asarray(positions)
    except (TypeError, ValueError) as exc:
        raise ValueError("symbols and positions must be numeric arrays") from exc
    if (raw_symbols.ndim != 3 or raw_symbols.shape[0] != 2 or
            raw_symbols.shape[2] != 3 or raw_symbols.shape[1] < 8 or
            raw_symbols.shape[1] % 2 or raw_symbols.dtype.kind not in "iuf"):
        raise ValueError("symbols must be real with shape (2, even nq>=8, 3)")
    symbol_values = np.array(raw_symbols, dtype=float, copy=True)
    if not np.all(np.isfinite(symbol_values)) or np.any(symbol_values[0, :, :2] <= 0):
        raise ValueError("symbols must be finite with strictly positive qq and pp values")
    if (raw_positions.ndim != 1 or raw_positions.size < 1 or
            raw_positions.dtype.kind not in "iuf"):
        raise ValueError("positions must be a nonempty real integer-coordinate vector")
    # Inspect object views too, so booleans in a mixed Python sequence are not
    # silently converted to integer coordinates during numeric coercion.
    if any(isinstance(value, (bool, np.bool_))
           for value in np.asarray(positions, dtype=object).flat):
        raise ValueError("Boolean site positions are invalid")
    if (not np.all(np.isfinite(raw_positions)) or
            np.any(raw_positions != np.floor(raw_positions))):
        raise ValueError("positions must contain distinct finite integer coordinates")
    coordinates = [int(value) for value in raw_positions]
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("positions must contain distinct finite integer coordinates")
    nq = symbol_values.shape[1]
    if max(coordinates) - min(coordinates) >= nq:
        raise ValueError("The position span must be strictly less than nq")

    q = -np.pi + (2.0 * np.pi / nq) * (np.arange(nq) + 0.5)
    differences = np.array([[left - right for right in coordinates]
                            for left in coordinates], dtype=float)
    cosine = np.cos(q[:, None, None] * differences)
    sine = np.sin(q[:, None, None] * differences)
    m = len(coordinates)
    covariance_jet = np.empty((2, 2 * m, 2 * m), dtype=float)
    with np.errstate(over="ignore", invalid="ignore"):
        for order in range(2):
            qq = np.einsum("kij,k->ij", cosine, symbol_values[order, :, 0] / nq)
            pp = np.einsum("kij,k->ij", cosine, symbol_values[order, :, 1] / nq)
            mixed = -np.einsum("kij,k->ij", sine, symbol_values[order, :, 2] / nq)
            covariance_jet[order, :m, :m] = qq
            covariance_jet[order, m:, m:] = pp
            covariance_jet[order, :m, m:] = mixed
            covariance_jet[order, m:, :m] = mixed.T
    if not np.all(np.isfinite(covariance_jet)):
        raise ValueError("The requested covariance is not finite in double precision")
    return covariance_jet

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return tests of Fourier normalization, current orientation and site order."""
    return [
        {
            "setup": """import numpy as np
symbols = _oracle_build_ness_symbols(0.45, 1.0, 3.2, 0.9, 128)
positions = np.arange(-3, 3)
""",
            "call": "fourier_covariance(symbols.copy(), positions.copy())",
            "gold_call": "_oracle_fourier_covariance(symbols.copy(), positions.copy())",
        },
        {
            "setup": """import numpy as np
symbols = _oracle_build_ness_symbols(0.8, 0.4, 2.0, 0.0, 8)
positions = np.array([17])
""",
            "call": "fourier_covariance(symbols.copy(), positions.copy())",
            "gold_call": "_oracle_fourier_covariance(symbols.copy(), positions.copy())",
        },
        {
            "setup": """import numpy as np
symbols = _oracle_build_ness_symbols(0.23, 1.7, 2.3, -1.4, 30)
positions = np.array([5, -2, 1, 0])
""",
            "call": "fourier_covariance(symbols.copy(), positions.copy())",
            "gold_call": "_oracle_fourier_covariance(symbols.copy(), positions.copy())",
        },
        {
            "setup": """import numpy as np
q = -np.pi + (2.0 * np.pi / 12) * (np.arange(12) + 0.5)
symbols = np.empty((2, 12, 3))
symbols[0] = np.column_stack((1.1 + 0.2*np.cos(q), 1.6 - 0.3*np.cos(2*q), 0.4*np.sin(q)))
symbols[1] = np.column_stack((0.3*np.cos(2*q), -0.2*np.cos(q), -0.7*np.sin(3*q)))
positions = np.array([0, 2, 5])
""",
            "call": "fourier_covariance(symbols.copy(), positions.copy())",
            "gold_call": "_oracle_fourier_covariance(symbols.copy(), positions.copy())",
        },
        {
            "setup": """import numpy as np
symbols = _oracle_build_ness_symbols(0.3, 0.9, 1.5, 0.7, 8)
positions = np.array([1000000, 1000007], dtype=np.int64)
""",
            "call": "fourier_covariance(symbols.copy(), positions.copy())",
            "gold_call": "_oracle_fourier_covariance(symbols.copy(), positions.copy())",
        },
        {
            "setup": """import numpy as np
symbols = np.ones((2, 8, 3))
positions = np.array([0, 0])
def _raises_value_error(fn):
    try:
        fn(symbols.copy(), positions.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(fourier_covariance)",
            "gold_call": "_raises_value_error(_oracle_fourier_covariance)",
        },
        {
            "setup": """import numpy as np
symbols = np.ones((2, 8, 3))
positions = np.array([0.0, 1.5])
def _raises_value_error(fn):
    try:
        fn(symbols.copy(), positions.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(fourier_covariance)",
            "gold_call": "_raises_value_error(_oracle_fourier_covariance)",
        },
        {
            "setup": """import numpy as np
symbols = np.ones((2, 8, 3))
symbols[0, 3, 0] = 0.0
positions = np.array([0, 1])
def _raises_value_error(fn):
    try:
        fn(symbols.copy(), positions.copy())
    except ValueError:
        return 1
    return 0
""",
            "call": "_raises_value_error(fourier_covariance)",
            "gold_call": "_raises_value_error(_oracle_fourier_covariance)",
        },
    ]

"""
Construct the reciprocal-permittivity convolution matrix of the periodic binary layer specified by harmonics, fill, offset, and epsilon_high, together with its derivative with respect to this call's fill argument.

Use the physical profile and Fourier convention specified in the main problem and preserve the supplied harmonic ordering. Offset is periodic modulo one, including for large finite representable offsets. The derivative holds offset and epsilon_high fixed; the final orchestrator separately sets the second layer's tangent to zero. Harmonics must be a nonempty one-dimensional integer array, 0<fill<1, and epsilon_high must be positive with a finite representable reciprocal. Raise ValueError for invalid inputs or an unrepresentable reciprocal.

Returns
-------
Return (coefficients,tangent), two finite complex128 arrays of shape (N,N), where N is the number of supplied harmonics. coefficients is the reciprocal-permittivity convolution matrix and tangent is its fill derivative. Both axes preserve the supplied harmonic order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def reciprocal_fourier(
    harmonics: np.ndarray,
    fill: float,
    offset: float,
    epsilon_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a reciprocal-permittivity convolution matrix and fill tangent.

    Parameters
    ----------
    harmonics : np.ndarray
        Nonempty one-dimensional integer harmonic ordering.
    fill : float
        Layer fill fraction satisfying 0 < fill < 1.
    offset : float
        Finite periodic displacement. Interpret the value modulo one.
    epsilon_high : float
        Positive relative permittivity with a finite float64 reciprocal.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The complex128 coefficient matrix C and its derivative dC/df,
        both with shape (N, N).

    Raises
    ------
    ValueError
        If the harmonic array or layer parameters are invalid, or if
        1/epsilon_high is not representable as a finite float64 value.
    """
    return (None, None)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_reciprocal_fourier(
    harmonics: np.ndarray,
    fill: float,
    offset: float,
    epsilon_high: float,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    harmonics = np.asarray(harmonics)

    if (
        harmonics.ndim != 1
        or harmonics.size == 0
        or not np.issubdtype(harmonics.dtype, np.integer)
    ):
        raise ValueError("harmonics must be a nonempty integer vector")

    parameters = np.asarray([fill, offset, epsilon_high])

    if (
        not np.isrealobj(parameters)
        or not np.all(np.isfinite(parameters))
        or not 0 < fill < 1
        or epsilon_high <= 0
    ):
        raise ValueError("invalid layer parameters")

    offset = float(np.remainder(offset, 1.0))

    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        reciprocal_high = np.divide(1.0, np.float64(epsilon_high))

    if not np.isfinite(reciprocal_high):
        raise ValueError(
            "epsilon_high is outside the representable reciprocal domain"
        )

    orders = harmonics[:, None] - harmonics[None, :]
    contrast = reciprocal_high - 1.0

    phase = np.exp(-2j * np.pi * orders * (offset + fill / 2))

    coefficients = (
        (orders == 0)
        + contrast * fill * np.sinc(orders * fill) * phase
    )

    tangent = contrast * np.exp(
        -2j * np.pi * orders * (offset + fill)
    )

    return (
        coefficients.astype(np.complex128),
        tangent.astype(np.complex128),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": (
                "reciprocal_fourier("
                "np.arange(-3, 4), 0.43, 0.0, 4.0)"
            ),
            "gold_call": (
                "_oracle_reciprocal_fourier("
                "np.arange(-3, 4), 0.43, 0.0, 4.0)"
            ),
        },
        {
            "setup": "import numpy as np",
            "call": (
                "reciprocal_fourier("
                "np.arange(-3, 4), 0.31, 0.19, 2.89)"
            ),
            "gold_call": (
                "_oracle_reciprocal_fourier("
                "np.arange(-3, 4), 0.31, 0.19, 2.89)"
            ),
        },
        {
            "setup": "import numpy as np",
            "call": (
                "reciprocal_fourier("
                "np.arange(-3, 4), 0.43, "
                "1099511627776.0, 4.0)"
            ),
            "gold_call": (
                "_oracle_reciprocal_fourier("
                "np.arange(-3, 4), 0.43, 0.0, 4.0)"
            ),
        },
        {
            "setup": (
                "import numpy as np\n"
                "\n"
                "def capture_value_error(function, *arguments):\n"
                "    try:\n"
                "        function(*arguments)\n"
                "    except ValueError:\n"
                "        return 1\n"
                "    return 0"
            ),
            "call": (
                "capture_value_error("
                "reciprocal_fourier, "
                "np.arange(-3, 4), 0.43, 0.0, 1e-320)"
            ),
            "gold_call": (
                "capture_value_error("
                "_oracle_reciprocal_fourier, "
                "np.arange(-3, 4), 0.43, 0.0, 1e-320)"
            ),
        },
    ]

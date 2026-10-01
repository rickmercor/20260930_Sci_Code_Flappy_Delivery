"""
Build the propagator tensor for symmetric finite-difference shifts $j=-J,\ldots,J$, where $J=\texttt{degree}$. For row index $k'$ and column index $k$, use the signed sample index $m=\texttt{stride}\,(k-k')+j$ and set the corresponding entry to $g_m$. Read $g_m$ from the supplied nonnegative-time scalar sequence for $m\ge 0$ and use $g_m=\overline{g_{-m}}$ for $m<0$.

A time-evolution Krylov basis makes overlap and shifted-propagator matrices Toeplitz in the basis-index difference. Unitarity relates a negative-time expectation value to the complex conjugate of its positive-time counterpart. Combining those two facts reconstructs every matrix element from a one-dimensional measured sequence while preserving the mirror relation between positive and negative finite-difference shifts.

Returns
-------
np.ndarray of shape (2 * degree + 1, krylov_dimension, krylov_dimension), ordered by j = -degree,...,degree and stored as complex128
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from numbers import Integral

import numpy as np

def build_shifted_propagators(
    g_nonnegative: np.ndarray,
    krylov_dimension: int,
    degree: int,
    stride: int,
) -> np.ndarray:
    """Build Toeplitz propagator matrices for symmetric shifts.

    Parameters
    ----------
    g_nonnegative : np.ndarray
        Complex sequence ``g[m]`` for nonnegative integer indices starting at 0.
    krylov_dimension : int
        Positive matrix dimension ``n``.
    degree : int
        Positive shift half-width ``J``.
    stride : int
        Positive integer ratio ``tau / delta_t``.

    Returns
    -------
    propagators : np.ndarray
        Complex128 array of shape ``(2 * degree + 1, n, n)`` ordered by
        ``j = -degree, ..., degree``.

    Raises
    ------
    ValueError
        If dimensions are invalid, samples are non-finite, or the sequence is
        too short for every required signed index.
    """
    return np.empty(
        (2 * degree + 1, krylov_dimension, krylov_dimension),
        dtype=np.complex128,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_build_shifted_propagators(
    g_nonnegative: np.ndarray,
    krylov_dimension: int,
    degree: int,
    stride: int,
) -> np.ndarray:
    from numbers import Integral

    import numpy as np

    for value, name in ((krylov_dimension, "krylov_dimension"), (degree, "degree"), (stride, "stride")):
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 1:
            raise ValueError(f"{name} must be an integer >= 1")
    n = int(krylov_dimension)
    degree = int(degree)
    stride = int(stride)

    try:
        samples = np.asarray(g_nonnegative, dtype=np.complex128)
    except (TypeError, ValueError) as exc:
        raise ValueError("g_nonnegative must be a one-dimensional numeric array") from exc
    if samples.ndim != 1 or samples.size < 1:
        raise ValueError("g_nonnegative must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(samples.real)) or not np.all(np.isfinite(samples.imag)):
        raise ValueError("g_nonnegative must contain finite complex values")

    largest_index = stride * (n - 1) + degree
    if samples.size <= largest_index:
        raise ValueError("g_nonnegative is too short for the requested matrix indices")

    result = np.empty((2 * degree + 1, n, n), dtype=np.complex128)
    for shift_position, shift in enumerate(range(-degree, degree + 1)):
        for row in range(n):
            for column in range(n):
                signed_index = stride * (column - row) + shift
                if signed_index >= 0:
                    value = samples[signed_index]
                else:
                    value = np.conjugate(samples[-signed_index])
                result[shift_position, row, column] = value
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
g = np.array([
    1.0+0.0j,
    0.99429356005856029-0.025590728300645980j,
    0.97730685925486915-0.050590494877476529j,
    0.94943860829937488-0.074423809017078665j,
    0.91133730900167487-0.096544241297493519j,
    0.86388714249963516-0.116449649912794870j,
    0.80818080703044082-0.133693203305266430j,
    0.74548930367185484-0.147895607047029390j,
    0.67722726275148382-0.158754282232542730j,
    0.60491163012666282-0.166049446589216800j,
    0.53012217788209992-0.169649646725830630j,
    0.45445491616160788-0.169514288978398760j,
    0.37948284322780795-0.165693225887655530j,
    0.30671045822235593-0.158324864969491470j,
    0.23753537514773893-0.147629912812195420j,
    0.17321244307405181-0.133906119606736620j,
], dtype=np.complex128)
""",
            "call": "build_shifted_propagators(g, 4, 3, 4)",
            "gold_call": "_oracle_build_shifted_propagators(g, 4, 3, 4)",
        },
        {
            "setup": """import numpy as np
g = np.array([1+0j, 0.8-0.1j, 0.5-0.2j], dtype=complex)
""",
            "call": "build_shifted_propagators(g, 1, 2, 1)",
            "gold_call": "_oracle_build_shifted_propagators(g, 1, 2, 1)",
        },
        {
            "setup": """import numpy as np
g = np.linspace(1.0, 0.1, 8).astype(complex)
""",
            "call": "build_shifted_propagators(g, 3, 1, 2)",
            "gold_call": "_oracle_build_shifted_propagators(g, 3, 1, 2)",
        },
        {
            "setup": """import numpy as np
g = np.ones(5, dtype=complex)
def run_model():
    try:
        build_shifted_propagators(g, 4, 2, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_build_shifted_propagators(g, 4, 2, 2)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
g = np.array([1+0j, np.nan+0j, 0.5+0j], dtype=complex)
def run_model():
    try:
        build_shifted_propagators(g, 1, 1, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_build_shifted_propagators(g, 1, 1, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]

"""
Evaluate how much of a log partition-function change is omitted by a finite substitution-order series.

This compares the nonlinear observable evaluated from the exact library polynomial with the Taylor truncation inherited from lower-order substitution effects.

Returns
-------
float: relative absolute remainder of the supplied log-series truncation.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_log_remainder_share(
    normalized_coefficients: "np.ndarray",
    log_coefficients: "np.ndarray",
    mutation_rate: float,
) -> float:
    """Return the relative absolute remainder of a truncated log series.

    Let ``A(mu)`` be the polynomial stored in increasing power order by
    ``normalized_coefficients`` and let ``G_k(mu)`` be the polynomial stored
    by ``log_coefficients``.  The constant terms must represent a normalized
    observable: ``A(0) = 1`` and ``G_k(0) = 0``.  Return

    ``abs(log(A(mu)) - G_k(mu)) / abs(log(A(mu)))``.

    Parameters
    ----------
    normalized_coefficients : np.ndarray
        Finite one-dimensional vector with at least two entries and constant
        coefficient one to absolute tolerance ``1e-10``.
    log_coefficients : np.ndarray
        Finite one-dimensional vector with between two and
        ``len(normalized_coefficients)`` entries and zero constant coefficient
        to absolute tolerance ``1e-10``.
    mutation_rate : float
        Finite scalar in ``[0, 1]``.

    Returns
    -------
    float
        Nonnegative relative absolute remainder.

    Raises
    ------
    ValueError
        If an input is invalid, ``A(mu)`` is not positive, or its logarithm
        is zero at the requested rate.
    """
    return share

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _remainder_inputs(normalized_coefficients, log_coefficients, mutation_rate):
    """Validate inputs shared by evaluation and root location."""
    import numpy as np

    try:
        normalized = np.array(normalized_coefficients, dtype=float)
        logarithm = np.array(log_coefficients, dtype=float)
    except (TypeError, ValueError):
        raise ValueError("coefficient arrays must be numeric") from None
    if normalized.ndim != 1 or normalized.size < 2 or not np.all(np.isfinite(normalized)):
        raise ValueError("normalized_coefficients must be a finite one-dimensional vector")
    if logarithm.ndim != 1 or logarithm.size < 2 or logarithm.size > normalized.size:
        raise ValueError("log_coefficients has an invalid shape")
    if not np.all(np.isfinite(logarithm)):
        raise ValueError("log_coefficients must be finite")
    if not np.isclose(normalized[0], 1.0, rtol=0.0, atol=1e-10):
        raise ValueError("normalized_coefficients must have constant one")
    if not np.isclose(logarithm[0], 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("log_coefficients must have constant zero")
    if isinstance(mutation_rate, bool):
        raise ValueError("mutation_rate must be a finite scalar in [0, 1]")
    try:
        rate = float(mutation_rate)
    except (TypeError, ValueError):
        raise ValueError("mutation_rate must be a finite scalar in [0, 1]") from None
    if not np.isfinite(rate) or rate < 0.0 or rate > 1.0:
        raise ValueError("mutation_rate must be a finite scalar in [0, 1]")
    return normalized, logarithm, rate


def _oracle_evaluate_log_remainder_share(
    normalized_coefficients: "np.ndarray",
    log_coefficients: "np.ndarray",
    mutation_rate: float,
) -> float:
    """Reference polynomial evaluation and relative log remainder."""
    import numpy as np

    normalized, logarithm, rate = _remainder_inputs(
        normalized_coefficients, log_coefficients, mutation_rate
    )
    exact_polynomial = float(np.polynomial.polynomial.polyval(rate, normalized))
    if exact_polynomial <= 0.0 or not np.isfinite(exact_polynomial):
        raise ValueError("the normalized partition polynomial must be positive at mutation_rate")
    exact_log = float(np.log(exact_polynomial))
    if exact_log == 0.0:
        raise ValueError("the exact log change is zero at mutation_rate")
    truncated_log = float(np.polynomial.polynomial.polyval(rate, logarithm))
    return float(abs(exact_log - truncated_log) / abs(exact_log))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only normal, boundary, edge and validation cases."""
    status = (
        "import numpy as np\n"
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
    )
    return [
        {
            "setup": "import numpy as np\nC = np.array([1.0, 2.0])\nG = np.array([0.0, 2.0])\n",
            "call": "float(evaluate_log_remainder_share(C.copy(), G.copy(), .2))",
            "gold_call": "float(_oracle_evaluate_log_remainder_share(C.copy(), G.copy(), .2))",
        },
        {
            "setup": "import numpy as np\nC = np.array([1.0, -1.2, 2.3, -.7, .1])\nG = np.array([0.0, -1.2, 1.58, 1.484])\n",
            "call": "float(evaluate_log_remainder_share(C.copy(), G.copy(), .35))",
            "gold_call": "float(_oracle_evaluate_log_remainder_share(C.copy(), G.copy(), .35))",
        },
        {
            "setup": "import numpy as np\nC = np.array([1.0, .5, .2])\nG = np.array([0.0, .5, .075])\n",
            "call": "float(evaluate_log_remainder_share(C.copy(), G.copy(), 1.0))",
            "gold_call": "float(_oracle_evaluate_log_remainder_share(C.copy(), G.copy(), 1.0))",
        },
        {
            "setup": status + "C = np.array([1.0, 2.0])\nG = np.array([0.0, 2.0])\n",
            "call": "_status(lambda: evaluate_log_remainder_share(C, G, 0.0))",
            "gold_call": "_status(lambda: _oracle_evaluate_log_remainder_share(C, G, 0.0))",
        },
        {
            "setup": status + "C = np.array([1.0, -3.0])\nG = np.array([0.0, -3.0])\n",
            "call": "_status(lambda: evaluate_log_remainder_share(C, G, .5))",
            "gold_call": "_status(lambda: _oracle_evaluate_log_remainder_share(C, G, .5))",
        },
        {
            "setup": status + "C = np.array([.9, 2.0])\nG = np.array([0.0, 2.0])\n",
            "call": "_status(lambda: evaluate_log_remainder_share(C, G, .2))",
            "gold_call": "_status(lambda: _oracle_evaluate_log_remainder_share(C, G, .2))",
        },
    ]

"""
Convert a gene's unspliced and spliced abundances and its fitted kinetic rates into the abundance used for signalling and its rate of change in every cell.

Transcription, splicing and degradation define a coupled RNA-velocity balance. The signalling state and its instantaneous change must be derived from that balance rather than treating either measured RNA species alone as the state.

Returns
-------
np.ndarray: (n, 2) array of [signalling expression state, instantaneous rate].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_total_expression_rates(
    unspliced: "np.ndarray",
    spliced: "np.ndarray",
    transcription_rate: float,
    splicing_rate: float,
    degradation_rate: float,
) -> "np.ndarray":
    """Return each cell's total mRNA abundance of a gene and its time derivative.

    Apply the transcription-splicing-degradation balance to the supplied RNA
    species. For every cell, return the expression state used by the
    signalling framework and the instantaneous derivative of that same state.
    Derive both columns from the coupled kinetic model and the input
    abundances; do not substitute a single RNA species for the framework's
    expression state.

    Parameters
    ----------
    unspliced : np.ndarray
        Nonnegative float array of shape ``(n,)``, ``n >= 1``.
    spliced : np.ndarray
        Nonnegative float array of shape ``(n,)``.
    transcription_rate : float
        Nonnegative transcription rate.
    splicing_rate : float
        Positive splicing rate constant.
    degradation_rate : float
        Positive degradation rate constant.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n, 2)``: signalling abundance, then its rate
        of change under the fitted kinetics.

    Raises
    ------
    ValueError
        If ``unspliced`` or ``spliced`` is not a finite nonnegative numeric
        array of shape ``(n,)`` with ``n >= 1``, if their lengths differ, if
        ``transcription_rate`` is not a finite nonnegative number, or if
        ``splicing_rate`` or ``degradation_rate`` is not a finite positive
        number (booleans are rejected for the rates).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_compute_total_expression_rates(
    unspliced: "np.ndarray",
    spliced: "np.ndarray",
    transcription_rate: float,
    splicing_rate: float,
    degradation_rate: float,
) -> "np.ndarray":
    """Reference implementation (sum of the unspliced and spliced balances)."""
    import numpy as np

    def _is_number(value):
        return (not isinstance(value, bool)
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _abundance(values, name):
        try:
            array = np.array(values, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric array") from None
        if array.ndim != 1 or array.size < 1:
            raise ValueError(f"{name} must have shape (n,) with n >= 1")
        if not (np.all(np.isfinite(array)) and np.all(array >= 0.0)):
            raise ValueError(f"{name} must be finite and nonnegative")
        return array

    u = _abundance(unspliced, "unspliced")
    s = _abundance(spliced, "spliced")
    if u.shape != s.shape:
        raise ValueError("unspliced and spliced must have the same length")
    if not (_is_number(transcription_rate) and transcription_rate >= 0.0):
        raise ValueError("transcription_rate must be a finite nonnegative number")
    if not (_is_number(splicing_rate) and splicing_rate > 0.0):
        raise ValueError("splicing_rate must be a finite positive number")
    if not (_is_number(degradation_rate) and degradation_rate > 0.0):
        raise ValueError("degradation_rate must be a finite positive number")
    unspliced_rate = float(transcription_rate) - float(splicing_rate) * u
    spliced_rate = float(splicing_rate) * u - float(degradation_rate) * s
    return np.column_stack([u + s, unspliced_rate + spliced_rate])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    head = (
        "import numpy as np\n"
        "def _sig(a, n):\n"
        "    a = np.asarray(a, dtype=float)\n"
        "    if a.shape != (n, 2):\n"
        "        return -1.0\n"
        "    k = np.arange(a.size, dtype=float)\n"
        "    return float(np.sum(np.abs(a)) + np.sum(a.ravel() * np.cos(0.7 * k)))\n"
    )
    status = (
        "def _status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "        return 0\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
    )
    counts = (
        "u = np.array([0.42, 1.10, 0.05, 0.77, 0.30])\n"
        "s = np.array([3.90, 0.65, 2.40, 1.15, 0.00])\n"
    )
    return [
        {
            "setup": head + counts,
            "call": "_sig(compute_total_expression_rates(u, s, 1.3, 0.7, 0.45), 5)",
            "gold_call": "_sig(_oracle_compute_total_expression_rates(u, s, 1.3, 0.7, 0.45), 5)",
        },
        {
            "setup": head + counts,
            "call": "float(np.sum(compute_total_expression_rates(u, s, 1.3, 25.0, 0.45)[:, 1] * np.arange(1.0, 6.0)))",
            "gold_call": "float(np.sum(_oracle_compute_total_expression_rates(u, s, 1.3, 25.0, 0.45)[:, 1] * np.arange(1.0, 6.0)))",
        },
        {
            "setup": head,
            "call": "_sig(compute_total_expression_rates(np.zeros(3), np.array([0.5, 2.0, 4.0]), 0.0, 1.0, 0.2), 3)",
            "gold_call": "_sig(_oracle_compute_total_expression_rates(np.zeros(3), np.array([0.5, 2.0, 4.0]), 0.0, 1.0, 0.2), 3)",
        },
        {
            "setup": head,
            "call": "_sig(compute_total_expression_rates(np.array([2.0]), np.array([1.5]), 0.9, 0.3, 0.6), 1)",
            "gold_call": "_sig(_oracle_compute_total_expression_rates(np.array([2.0]), np.array([1.5]), 0.9, 0.3, 0.6), 1)",
        },
        {
            "setup": head + counts + status,
            "call": "_status(lambda: compute_total_expression_rates(u, -s, 1.3, 0.7, 0.45))",
            "gold_call": "_status(lambda: _oracle_compute_total_expression_rates(u, -s, 1.3, 0.7, 0.45))",
        },
        {
            "setup": head + counts + status,
            "call": "_status(lambda: compute_total_expression_rates(u, s[:4], 1.3, 0.7, 0.45))",
            "gold_call": "_status(lambda: _oracle_compute_total_expression_rates(u, s[:4], 1.3, 0.7, 0.45))",
        },
        {
            "setup": head + counts + status,
            "call": "_status(lambda: compute_total_expression_rates(u, s, 1.3, 0.7, 0.0))",
            "gold_call": "_status(lambda: _oracle_compute_total_expression_rates(u, s, 1.3, 0.7, 0.0))",
        },
    ]

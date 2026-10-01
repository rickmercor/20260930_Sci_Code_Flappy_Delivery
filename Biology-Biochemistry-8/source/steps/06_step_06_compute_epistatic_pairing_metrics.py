"""
Convert total and event-specific endpoint log weights into the normalized finite interaction and double-mutant pairing share.

A two-site finite interaction is a signed inclusion-exclusion contrast of four partition-function endpoints. An event share is a ratio of an event-specific endpoint to its total endpoint; log inputs keep both calculations stable over a wide dynamic range.

Returns
-------
np.ndarray: [normalized finite interaction, double-mutant p-as-5'-pair weight share].
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_epistatic_pairing_metrics(log_endpoints: "np.ndarray") -> "np.ndarray":
    """Return the normalized interaction and double-mutant event share.

    ``log_endpoints`` has shape (2, 4) and the layout returned by
    ``evaluate_ordered_double_endpoints``. From total endpoints
    ``[Z, Z_p, Z_q, Z_pq]``, compute ``(Z_pq - Z_p - Z_q + Z) / Z``.
    Compute the pairing share as the double-mutant event endpoint divided by
    ``Z_pq``. Event endpoints may be ``-inf`` for zero weight and must not
    exceed their corresponding total endpoints.

    Returns
    -------
    np.ndarray
        Float array ``[normalized_interaction, pairing_share]``.

    Raises
    ------
    ValueError
        If the input has the wrong shape, a total endpoint is non-finite, an
        event endpoint is NaN or positive infinity, an event exceeds its total,
        or either returned metric is non-finite or the share lies outside [0, 1].
    """
    return metrics  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _signed_log_difference(positive_logs: "np.ndarray",
                           negative_logs: "np.ndarray") -> float:
    """Return sum(exp(positive_logs)) minus sum(exp(negative_logs)) stably."""
    positive_values = list(np.asarray(positive_logs, dtype=float).ravel())
    negative_values = list(np.asarray(negative_logs, dtype=float).ravel())
    unmatched_positive = []
    for value in positive_values:
        match = next((i for i, other in enumerate(negative_values) if value == other), None)
        if match is None:
            unmatched_positive.append(value)
        else:
            negative_values.pop(match)
    if not unmatched_positive and not negative_values:
        return 0.0
    positive = -np.inf if not unmatched_positive else float(
        np.logaddexp.reduce(np.asarray(unmatched_positive, dtype=float))
    )
    negative = -np.inf if not negative_values else float(
        np.logaddexp.reduce(np.asarray(negative_values, dtype=float))
    )
    if positive == negative:
        return 0.0
    larger, smaller = (positive, negative) if positive > negative else (negative, positive)
    log_magnitude = larger + float(np.log(-np.expm1(smaller - larger)))
    if log_magnitude > np.log(np.finfo(float).max):
        return np.inf if positive > negative else -np.inf
    magnitude = float(np.exp(log_magnitude))
    return magnitude if positive > negative else -magnitude

def _oracle_compute_epistatic_pairing_metrics(log_endpoints: "np.ndarray") -> "np.ndarray":
    logs = np.asarray(log_endpoints, dtype=float)
    if logs.shape != (2, 4) or not np.all(np.isfinite(logs[0])) \
            or np.any(np.isnan(logs[1])) or np.any(np.isposinf(logs[1])) \
            or np.any(logs[1] > logs[0] + 1e-10):
        raise ValueError("log_endpoints has an invalid shape, total, or event endpoint")
    relative = logs[0] - logs[0, 0]
    interaction = _signed_log_difference(
        np.array([relative[3], 0.0]), np.array([relative[1], relative[2]])
    )
    share = 0.0 if np.isneginf(logs[1, 3]) else float(np.exp(logs[1, 3] - logs[0, 3]))
    if not np.isfinite(interaction) or not np.isfinite(share) or not 0.0 <= share <= 1.0 + 1e-12:
        raise ValueError("metrics must be finite and the event share must lie in [0, 1]")
    return np.array([interaction, min(1.0, share)], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    setup = (
        "import numpy as np\n"
        "def _pin(v):\n    a=np.asarray(v,dtype=float); return float(a[0]+np.sqrt(2.0)*a[1]+a[0]*a[1])\n"
        "def _status(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    return [
        {"setup": setup + "E=np.log(np.array([[10.0,14.0,18.0,31.0],[2.0,4.0,3.0,12.0]]))\n",
         "call": "_pin(compute_epistatic_pairing_metrics(E.copy()))",
         "gold_call": "_pin(_oracle_compute_epistatic_pairing_metrics(E.copy()))"},
        {"setup": setup + "E=np.array([[0.0,1000.0,np.log(2.0),1000.0],[-np.inf,-np.inf,-np.inf,-np.inf]])\n",
         "call": "_pin(compute_epistatic_pairing_metrics(E.copy()))",
         "gold_call": "_pin(_oracle_compute_epistatic_pairing_metrics(E.copy()))"},
        {"setup": setup + "E=np.array([[0.0,np.log(2.0),np.log(3.0),np.log(4.5)],[-np.inf,-np.inf,-np.inf,-np.inf]])\n",
         "call": "_pin(compute_epistatic_pairing_metrics(E.copy()))",
         "gold_call": "_pin(_oracle_compute_epistatic_pairing_metrics(E.copy()))"},
        {"setup": setup + "E=np.zeros((2,4)); E[1,3]=0.1\n",
         "call": "_status(lambda: compute_epistatic_pairing_metrics(E))",
         "gold_call": "_status(lambda: _oracle_compute_epistatic_pairing_metrics(E))"},
    ]

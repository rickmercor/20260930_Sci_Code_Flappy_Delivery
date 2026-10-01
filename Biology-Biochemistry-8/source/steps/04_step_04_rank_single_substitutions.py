"""
Rank exact single substitutions from their role-resolved log weights and select the strongest separated pair.

Log-domain endpoints allow relative finite changes to be compared without first materializing large partition functions. The unchanged-base endpoint at every position supplies a common reference normalization.

Returns
-------
np.ndarray: float array [top_score, p, c, gap_score, q, d] with 0-based positions and bases.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def rank_single_substitutions(sequence: str, channel_logs: "np.ndarray",
                              minimum_gap: int) -> "np.ndarray":
    """Return the strongest substitution and the strongest separated one.

    ``channel_logs[p, c, r]`` contains the four role log weights from
    ``evaluate_single_replacement_channels``. Log-sum the role axis to obtain
    each exact mutant endpoint. The endpoint for the unchanged base must give
    one common finite reference weight. Score every changed base by its
    absolute weight change divided by that reference and return
    ``[top_score, p, c, gap_score, q, d]``. Positions and bases are 0-based and
    A, C, G, U ordered. Ties go to the smaller position, then earlier base.

    Returns
    -------
    np.ndarray
        Float array ``[top_score, p, c, gap_score, q, d]``.

    Raises
    ------
    ValueError
        If the sequence is invalid, ``channel_logs`` has the wrong shape or
        contains NaN or positive infinity, the unchanged endpoints disagree,
        ``minimum_gap`` is not a positive integer, a score is non-finite, or
        no substitution is sufficiently separated.
    """
    return ranked  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_rank_single_substitutions(sequence: str, channel_logs: "np.ndarray",
                                      minimum_gap: int) -> "np.ndarray":
    if not isinstance(sequence, str) or not sequence or set(sequence) - set("ACGU"):
        raise ValueError("sequence must be a non-empty RNA string")
    logs = np.asarray(channel_logs, dtype=float)
    n = len(sequence)
    if logs.shape != (n, 4, 4) or np.any(np.isnan(logs)) or np.any(np.isposinf(logs)):
        raise ValueError("channel_logs must have shape (n, 4, 4) with no NaN or positive infinity")
    if isinstance(minimum_gap, bool) or not isinstance(minimum_gap, (int, np.integer)) or minimum_gap < 1:
        raise ValueError("minimum_gap must be a positive integer")
    endpoints = np.logaddexp.reduce(logs, axis=2)
    x = np.fromiter(("ACGU".index(ch) for ch in sequence), dtype=int, count=n)
    reference_entries = endpoints[np.arange(n), x]
    if not np.all(np.isfinite(reference_entries)) \
            or not np.allclose(reference_entries, reference_entries[0], rtol=0.0, atol=1e-10):
        raise ValueError("unchanged-base endpoints must share one finite reference log weight")
    relative_logs = endpoints - reference_entries[0]
    scores = np.empty_like(relative_logs)
    increasing = relative_logs >= 0.0
    scores[increasing] = np.expm1(relative_logs[increasing])
    scores[~increasing] = -np.expm1(relative_logs[~increasing])
    if not np.all(np.isfinite(scores)):
        raise ValueError("relative substitution scores must be finite")
    scores[np.arange(n), x] = -1.0
    p, c = np.unravel_index(np.argmax(scores), scores.shape)
    separated = scores.copy()
    separated[max(0, p - int(minimum_gap) + 1):p + int(minimum_gap), :] = -1.0
    if separated.max() < 0.0:
        raise ValueError("no substitution lies at least minimum_gap positions from the strongest one")
    q, d = np.unravel_index(np.argmax(separated), separated.shape)
    return np.array([scores[p, c], p, c, scores[q, d], q, d], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    base = (
        "import numpy as np\ns='GCAUGGUACGUAGCCAUGGA'; x=np.array(['ACGU'.index(c) for c in s])\n"
        "V=1.0+np.random.default_rng(4).random((len(s),4))*4.0; V[np.arange(len(s)),x]=2.5\n"
        "def _channels(v):\n    return np.repeat(np.log(v/4.0)[:,:,None],4,axis=2)\n"
        "def _pin(v):\n    a=np.asarray(v,dtype=float); r=np.arange(1.0,a.size+1.0); return float(np.sum(np.sin(.43*r)*a)+np.sum(np.cos(.21*r)*a*a))\n"
        "def _status(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    return [
        {"setup": base + "C=_channels(V)\n", "call": "_pin(rank_single_substitutions(s,C.copy(),5))",
         "gold_call": "_pin(_oracle_rank_single_substitutions(s,C.copy(),5))"},
        {"setup": base + "C=_channels(V)\n", "call": "_pin(rank_single_substitutions(s,C.copy(),12))",
         "gold_call": "_pin(_oracle_rank_single_substitutions(s,C.copy(),12))"},
        {"setup": base + "V[:]=2.5; V[7,1]=9.0; V[7,2]=9.0; V[2,1]=6.0; C=_channels(V)\n",
         "call": "_pin(rank_single_substitutions(s,C.copy(),4))",
         "gold_call": "_pin(_oracle_rank_single_substitutions(s,C.copy(),4))"},
        {"setup": base + "C=_channels(V)\n", "call": "_status(lambda: rank_single_substitutions(s,C,20))",
         "gold_call": "_status(lambda: _oracle_rank_single_substitutions(s,C,20))"},
    ]

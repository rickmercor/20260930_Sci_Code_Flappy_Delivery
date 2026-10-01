"""
Select the robust candidate.

A candidate is assessed across the complete perturbation panel. Its performance is the worst surviving scenario rather than a performance measured in a selected favorable state.

Returns
-------
A finite float ndarray of length 3+2C.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_robust_candidate(
    nets: "np.ndarray",
    compatibility: "np.ndarray",
    target: int,
    normalization: float,
) -> "np.ndarray":
    """
    nets is finite (C,P,N), compatibility is finite (C,P,2) with binary flags in its
    first component, target is a zero-based integer reaction index, and normalization is
    positive finite. A candidate is eligible when every scenario is compatible. Its
    score is its minimum signed target net flux divided by normalization. Select the
    largest eligible score; candidates within 1e-10 of the largest score tie, with the
    smallest row index selected. Within the selected candidate, scenarios within 1e-10
    of its minimum score tie, with the smallest scenario index selected. Return one
    vector: selected candidate index, selected limiting scenario index, selected score,
    C eligibility flags, C scores (including ineligible rows). Invalid inputs or no
    eligible candidate raise ValueError.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_select_robust_candidate(
    nets: "np.ndarray",
    compatibility: "np.ndarray",
    target: int,
    normalization: float,
) -> "np.ndarray":
    import numpy as np

    v = np.asarray(nets, dtype=float)
    k = np.asarray(compatibility, dtype=float)
    d = float(normalization)
    if (
        v.ndim != 3
        or min(v.shape) == 0
        or k.shape != v.shape[:-1] + (2,)
        or (not np.isfinite(v).all())
        or (not np.isfinite(k).all())
        or (not np.isin(k[..., 0], [0, 1]).all())
        or (not isinstance(target, (int, np.integer)))
        or (not 0 <= target < v.shape[-1])
        or (not np.isfinite(d))
        or (d <= 0)
    ):
        raise ValueError("Invalid selection inputs")
    eligible = np.all(k[..., 0] == 1, axis=1)
    if not eligible.any():
        raise ValueError("No eligible candidate")
    scores = np.min(v[:, :, target], axis=1) / d
    maximum = np.max(scores[eligible])
    winner = np.flatnonzero(eligible & (scores >= maximum - 1e-10))[0]
    row = v[winner, :, target] / d
    scenario = np.flatnonzero(row <= scores[winner] + 1e-10)[0]
    return np.r_[
        float(winner),
        float(scenario),
        scores[winner],
        eligible.astype(float),
        scores,
    ]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\n"
            "v = np.array([[[5.0], [4.0]], [[6.0], [2.0]], [[4.5], [4.5]]])\n"
            "k = np.zeros((3, 2, 2))\n"
            "k[:, :, 0] = 1.0\n"
            "k[1, 1, 0] = 0.0",
            "call": "select_robust_candidate(v,k,0,5.)",
            "gold_call": "_oracle_select_robust_candidate(v,k,0,5.)",
        },
        {
            "setup": "import numpy as np\n"
            "v = np.array([[[2.0], [2.0]], [[2.0], [3.0]]])\n"
            "k = np.zeros((2, 2, 2))\n"
            "k[:, :, 0] = 1.0",
            "call": "select_robust_candidate(v,k,0,2.)",
            "gold_call": "_oracle_select_robust_candidate(v,k,0,2.)",
        },
        {
            "setup": "import numpy as np\n"
            "v = np.array([[[-2.0], [-3.0]], [[-4.0], [-1.0]]])\n"
            "k = np.zeros((2, 2, 2))\n"
            "k[:, :, 0] = 1.0",
            "call": "select_robust_candidate(v,k,0,2.)",
            "gold_call": "_oracle_select_robust_candidate(v,k,0,2.)",
        },
        {
            "setup": "import numpy as np\n"
            "v = np.ones((1, 1, 1))\n"
            "k = np.zeros((1, 1, 2))\n"
            "\n"
            "def _error_check(fn, args):\n"
            "    try:\n"
            "        fn(*args)\n"
            "    except ValueError:\n"
            "        return 1.0\n"
            "    return 0.0",
            "call": "_error_check(select_robust_candidate, (v,k,0,1.0,))",
            "gold_call": "_error_check(_oracle_select_robust_candidate, (v,k,0,1.0,))",
            "tol": 0.0,
        },
    ]

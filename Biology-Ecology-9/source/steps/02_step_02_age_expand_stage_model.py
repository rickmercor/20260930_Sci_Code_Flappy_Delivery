"""
Given a stage-classified matrix of size s, with s at least two, that is zero everywhere except its first row, its subdiagonal and its southeast corner, and a relative tolerance, build the age-classified expansion of every length n greater than s in increasing order and return the first whose growth rate differs from that of the stage matrix by less than the tolerance times the stage matrix's growth rate. In the expansion of length n the fertility of age class i is the first-row entry of stage i for i up to s and the first-row entry of stage s beyond it; the survival probability from age class j to j + 1 is the subdiagonal entry leaving stage j for j below s and the stasis probability of stage s from j = s onwards. Obtain both growth rates from the Perron triplet. Raise ValueError if no length up to max_classes meets the tolerance.

Many published matrix population models are classified by stage rather than by age. A common and simple form has the pattern of a Leslie matrix, with fertilities in the first row and survival probabilities on the subdiagonal, except that the last stage is a plus-group: its individuals remain in it from one interval to the next with a stasis probability, so the southeast corner of the matrix is positive. Such a model is not age-classified, and methods that merge adjacent age classes into wider ones, which rely on every individual advancing exactly one class per interval, do not apply to it directly.

The plus-group can be unrolled into age classes. An individual that enters the last stage survives each further interval with the stasis probability and keeps reproducing at the last stage's fertility, so the age-classified model that repeats the last stage's fertility and survival in every age beyond it reproduces the stage model exactly when the number of age classes is infinite. A finite Leslie model truncates that tail, and the individuals that would pass beyond the last age class are lost, so the growth rate of the truncated model rises towards that of the stage model from below as age classes are added. The truncation error falls geometrically, by roughly the stasis probability divided by the growth rate per added class.

The length of the expansion is therefore a choice fixed by an accuracy criterion on the growth rate: the shortest age-classified model whose growth rate agrees with that of the stage model to within a stated relative tolerance. That length, and not the number of stages, is the number of age classes that any later aggregation works with.

Returns
-------
dict holding the np.ndarray fertility of shape (n,) and survival of shape (n - 1,) of the chosen expansion; the integer age_classes, its length n; the floats stage_growth_rate and age_growth_rate, the two Perron roots; and the floats relative_error, the relative difference of the two growth rates at length n, and previous_error, the same at length n - 1, which is NaN when the chosen length is the shortest examined, s + 1, because no shorter expansion was evaluated.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def age_expand_stage_model(
    stage_matrix: np.ndarray,
    tolerance: float = 1e-3,
    max_classes: int = 500,
) -> dict:
    """Shortest age-classified expansion of a stage model with a plus-group.

    Parameters
    ----------
    stage_matrix : np.ndarray
        Stage-classified matrix with Leslie pattern plus a positive southeast entry, shape (s, s).
        Its subdiagonal survival probabilities lie in (0, 1], its last stage is fertile, and its
        southeast stasis probability lies strictly between zero and one.
    tolerance : float
        Relative tolerance on the growth rate, above zero.
    max_classes : int
        Largest expansion length examined, greater than s.

    Returns
    -------
    dict
        Under the keys fertility, survival, age_classes, stage_growth_rate, age_growth_rate,
        relative_error and previous_error. previous_error is NaN when the chosen length is
        the shortest examined, s + 1, because no shorter expansion was evaluated.

    Raises
    ------
    ValueError
        When the stage matrix does not have the stated pattern and ranges, when the tolerance is
        not finite and above zero, when max_classes is not an integer greater than s, or when no
        expansion up to max_classes meets the tolerance. A count given as a bool, or as a float
        even when its value is integral, is not accepted as an integer, and a bool or a
        non-numeric value is not accepted as a real number.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numbers

import numpy as np


def _check_stage_matrix(stage_matrix):
    """Validate a stage matrix with Leslie pattern and a plus-group and return it as floats."""
    a = np.asarray(stage_matrix)
    if np.iscomplexobj(a):
        raise ValueError("stage_matrix must be real")
    a = a.astype(float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or a.shape[0] < 2:
        raise ValueError("stage_matrix must be square with at least two stages")
    if not np.all(np.isfinite(a)) or np.any(a < 0.0):
        raise ValueError("stage_matrix must be finite and nonnegative")
    s = a.shape[0]
    pattern = np.zeros((s, s), dtype=bool)
    pattern[0, :] = True
    pattern[np.arange(1, s), np.arange(s - 1)] = True
    pattern[s - 1, s - 1] = True
    if np.any(a[~pattern] != 0.0):
        raise ValueError("stage_matrix must be zero outside its first row, subdiagonal and corner")
    below = np.array([a[j + 1, j] for j in range(s - 1)])
    if np.any(below <= 0.0) or np.any(below > 1.0):
        raise ValueError("subdiagonal survival probabilities must lie in (0, 1]")
    if a[0, s - 1] <= 0.0:
        raise ValueError("the last stage must be fertile")
    if not 0.0 < a[s - 1, s - 1] < 1.0:
        raise ValueError("the stasis probability of the last stage must lie in (0, 1)")
    return a


def _expansion(a, n):
    """Fertilities and survival probabilities of the age expansion of length n."""
    s = a.shape[0]
    fertility = np.array([a[0, i] if i < s else a[0, s - 1] for i in range(n)])
    survival = np.array([a[j + 1, j] if j < s - 1 else a[s - 1, s - 1] for j in range(n - 1)])
    return fertility, survival


def _leslie(fertility, survival):
    """The Leslie matrix with the given fertilities and survival probabilities."""
    n = fertility.size
    matrix = np.zeros((n, n))
    matrix[0, :] = fertility
    matrix[np.arange(1, n), np.arange(n - 1)] = survival
    return matrix


def _oracle_age_expand_stage_model(
    stage_matrix: np.ndarray,
    tolerance: float = 1e-3,
    max_classes: int = 500,
) -> dict:
    """Reference implementation."""
    a = _check_stage_matrix(stage_matrix)
    s = a.shape[0]
    if isinstance(tolerance, bool) or not isinstance(tolerance, numbers.Real):
        raise ValueError("tolerance must be a real number")
    tol = float(tolerance)
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tolerance must be finite and above zero")
    if isinstance(max_classes, bool) or not isinstance(max_classes, numbers.Integral):
        raise ValueError("max_classes must be an integer")
    if int(max_classes) <= s:
        raise ValueError("max_classes must exceed the number of stages")

    target = _oracle_perron_triplet(a)["growth_rate"]  # noqa: F821
    previous = np.nan
    for n in range(s + 1, int(max_classes) + 1):
        fertility, survival = _expansion(a, n)
        rate = _oracle_perron_triplet(_leslie(fertility, survival))["growth_rate"]  # noqa: F821
        error = abs(rate - target) / target
        if error < tol:
            return {"fertility": fertility,
                    "survival": survival,
                    "age_classes": n,
                    "stage_growth_rate": target,
                    "age_growth_rate": rate,
                    "relative_error": error,
                    "previous_error": previous}
        previous = error
    raise ValueError("no expansion up to max_classes meets the tolerance")

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return self-contained, single-invocation numeric test cases."""
    setup = "import numpy as np\n\ndef _flatten(value):\n    if isinstance(value, dict):\n        parts = [_flatten(value[key]) for key in sorted(value)]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    if isinstance(value, (list, tuple)):\n        parts = [_flatten(item) for item in value]\n        return np.concatenate(parts) if parts else np.empty(0, dtype=float)\n    return np.asarray(value, dtype=float).ravel()\n\ndef project(value):\n    raw = _flatten(value)\n    missing = np.isnan(raw)\n    return np.concatenate((np.where(missing, 0.0, raw), missing.astype(float)))\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return np.array([1.0])\n    return np.array([0.0])\n\ndef expect_runtime_error(fn, *args):\n    try:\n        fn(*args)\n    except RuntimeError:\n        return np.array([1.0])\n    return np.array([0.0])\n\nSTAGE = np.array([[0.0, 0.3, 1.0, 2.0],\n                  [0.6, 0.0, 0.0, 0.0],\n                  [0.0, 0.7, 0.0, 0.0],\n                  [0.0, 0.0, 0.85, 0.55]])\n"
    return [
        {
            "setup": setup + "",
            "call": "project(age_expand_stage_model(STAGE))",
            "gold_call": "project(_oracle_age_expand_stage_model(STAGE))",
        },
        {
            "setup": setup + "OTHER = np.array([[0.2, 1.1, 1.6], [0.5, 0.0, 0.0], [0.0, 0.8, 0.3]])\n",
            "call": "project(age_expand_stage_model(OTHER))",
            "gold_call": "project(_oracle_age_expand_stage_model(OTHER))",
        },
        {
            "setup": setup + "QUICK = np.array([[0.0, 3.0], [0.9, 0.01]])\n",
            "call": "project(age_expand_stage_model(QUICK, 1e-2))",
            "gold_call": "project(_oracle_age_expand_stage_model(QUICK, 1e-2))",
        },
        {
            "setup": setup + "",
            "call": "expect_value_error(age_expand_stage_model, STAGE, 1e-12, 6)",
            "gold_call": "expect_value_error(_oracle_age_expand_stage_model, STAGE, 1e-12, 6)",
        },
    ]

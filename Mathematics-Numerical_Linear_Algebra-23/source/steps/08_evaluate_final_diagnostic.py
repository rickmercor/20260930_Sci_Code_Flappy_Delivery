"""
Run the complete Balance Method 2 and Hermite-spline screening pipeline and return the maximum flagged exceedance margin as the final scalar diagnostic.

Use the seven preceding computational steps as one pipeline. First evaluate the Balance Method 2 candidates and construct the balanced root set. Then construct the associated balanced residual polynomial and evaluate its derivative at every real balanced root. Use those real-root derivative values to construct the eligible Hermite spline candidates, apply the Ritz-extrema screening to those candidates, and compute the exact interval maxima of the residual polynomial on the same examined intervals.



The intermediate outputs must be mutually consistent. In particular, if the balanced root set contains $m$ roots, the residual polynomial must contain $m + 1$ coefficients. Exactly one row of the Balance Method 2 candidate table must be marked as selected. The exact-maximum table must have one row per examined interval.



For the screening result, let $q_j$ denote the exceedance-margin column returned for interval $j$. By construction,



$$

q_j =

\\begin{cases}

C_j(\\widehat{x}_j) - 1, & \\text{if interval } j \\text{ is flagged},\\\\

0, & \\text{otherwise}.

\\end{cases}

$$



Define the final diagnostic as



$$

Q = \\max_j q_j

$$



if at least one examined interval exists, and $Q = 0$ if no interval is examined. Because unflagged intervals already have zero margin, this is equivalent to taking the maximum positive exceedance over the flagged intervals.



The implementation must call the seven preceding public functions rather than reproducing their internal formulas.



Invalid input (empty or non-one-dimensional root array, non-finite entries, non-finite Ritz extrema, or $\\widetilde{\\theta}_{\\min} > \\widetilde{\\theta}_{\\max}$) must raise ValueError.

Returns
-------
A finite Python float equal to the largest flagged exceedance margin, or 0.0 when no interval is flagged.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def evaluate_final_diagnostic(
    harmonic_ritz_roots: np.ndarray,
    ritz_min: float,
    ritz_max: float,
) -> float:
    """
    Evaluate the complete balanced-polynomial screening diagnostic.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots with
        complex roots occurring in conjugate pairs.
    ritz_min : float
        Ritz value with the smallest real part.
    ritz_max : float
        Ritz value with the largest real part.

    Returns
    -------
    float
        Maximum flagged spline exceedance margin, or 0.0 if none
        is present.

    Raises
    ------
    ValueError
        If the root array is empty or not one-dimensional, contains
        non-finite entries, or the Ritz extrema are non-finite or
        ritz_min > ritz_max.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_evaluate_final_diagnostic(
    harmonic_ritz_roots,
    ritz_min,
    ritz_max,
):
    roots = np.asarray(
        harmonic_ritz_roots,
        dtype=complex,
    )

    ritz_min = float(ritz_min)
    ritz_max = float(ritz_max)

    if roots.ndim != 1 or roots.size == 0:
        raise ValueError(
            "harmonic_ritz_roots must be a non-empty one-dimensional array."
        )

    if not np.all(
        np.isfinite(roots.real)
        & np.isfinite(roots.imag)
    ):
        raise ValueError(
            "harmonic_ritz_roots must contain only finite values."
        )

    if not np.isfinite(ritz_min) or not np.isfinite(ritz_max):
        raise ValueError(
            "Ritz extrema must be finite."
        )

    if ritz_min > ritz_max:
        raise ValueError(
            "ritz_min must not exceed ritz_max."
        )

    candidate_table = _oracle_evaluate_balance_candidates(
        roots
    )

    selected_count = np.count_nonzero(
        candidate_table[:, 6] == 1.0
    )

    if selected_count != 1:
        raise ValueError(
            "Exactly one Balance Method 2 candidate must be selected."
        )

    balanced_roots = _oracle_apply_balance_method2(
        roots
    )

    coefficients = _oracle_build_balanced_residual_polynomial(
        roots
    )

    if coefficients.size != balanced_roots.size + 1:
        raise ValueError(
            "Polynomial degree is inconsistent with the balanced root count."
        )

    real_root_derivatives = _oracle_evaluate_real_root_derivatives(
        roots
    )

    spline_candidates = _oracle_construct_hermite_spline_candidates(
        real_root_derivatives
    )

    screening = _oracle_apply_ritz_interval_screening(
        spline_candidates,
        ritz_min,
        ritz_max,
    )

    exact_maxima = _oracle_evaluate_exact_interval_maxima(
        roots
    )

    if exact_maxima.shape[0] != screening.shape[0]:
        raise ValueError(
            "The exact-maximum table must have one row per examined interval."
        )

    if screening.shape[0] == 0:
        return 0.0

    margins = screening[:, 8]

    if not np.all(np.isfinite(margins)):
        raise ValueError(
            "Screening margins must be finite."
        )

    Q = float(
        np.max(margins)
    )

    if Q < 0.0:
        raise ValueError(
            "Exceedance margins must be nonnegative."
        )

    return Q

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    -5.0,
    -1.3,
    0.9,
    2.0 + 1.5j,
    2.0 - 1.5j
], dtype=complex)

ritz_min = -4.6
ritz_max = 0.75
""",
            "call": """evaluate_final_diagnostic(
    harmonic_ritz_roots,
    ritz_min,
    ritz_max
)""",
            "gold_call": """_oracle_evaluate_final_diagnostic(
    harmonic_ritz_roots,
    ritz_min,
    ritz_max
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    1.0,
    -1.1111111111111112
], dtype=complex)

ritz_min = -5.0
ritz_max = 5.0
""",
            "call": """evaluate_final_diagnostic(
    harmonic_ritz_roots,
    ritz_min,
    ritz_max
)""",
            "gold_call": """_oracle_evaluate_final_diagnostic(
    harmonic_ritz_roots,
    ritz_min,
    ritz_max
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    1.0,
    -1.1111111111111112,
    -20.0 + 20.0j,
    -20.0 - 20.0j
], dtype=complex)

ritz_min = -5.0
ritz_max = 5.0
""",
            "call": """evaluate_final_diagnostic(
    harmonic_ritz_roots,
    ritz_min,
    ritz_max
)""",
            "gold_call": """_oracle_evaluate_final_diagnostic(
    harmonic_ritz_roots,
    ritz_min,
    ritz_max
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    -3.0,
    -1.0,
    1.0678380296485983,
    1.6094271405406886,
    2.6866715986911203,
    4.287768888388513,
    6.395177052675791,
    8.985806886701518,
    12.03127490787994,
    15.498214331265899,
    19.348640643033054,
    23.540367766756177,
    28.027470262892436,
    32.76078649750764,
    37.68845726742523,
    42.75649398050857,
    47.90937016597429,
    53.09062983402571,
    58.243506019491434,
    63.31154273257478,
    68.23921350249236,
    72.97252973710758,
    77.45963223324384,
    81.65135935696695,
    85.5017856687341,
    88.96872509212005,
    92.01419311329849,
    94.60482294732421,
    96.71223111161149,
    98.31332840130888,
    99.39057285945933,
    99.9321619703514
], dtype=complex)

ritz_min = -2.4
ritz_max = 98.5
""",
            "call": """evaluate_final_diagnostic(
    harmonic_ritz_roots, ritz_min, ritz_max
)""",
            "gold_call": """_oracle_evaluate_final_diagnostic(
    harmonic_ritz_roots, ritz_min, ritz_max
)""",
        },
        {
            "setup": """import numpy as np


def _run_invalid(fn, *args):
    try:
        fn(*args)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

harmonic_ritz_roots = np.array([
    -5.0,
    -1.3,
    0.9,
    2.0 + 1.5j,
    2.0 - 1.5j
], dtype=complex)

ritz_min = 0.75
ritz_max = -4.6
""",
            "call": """_run_invalid(
    evaluate_final_diagnostic, harmonic_ritz_roots, ritz_min, ritz_max
)""",
            "gold_call": """_run_invalid(
    _oracle_evaluate_final_diagnostic, harmonic_ritz_roots, ritz_min, ritz_max
)""",
        },
    ]

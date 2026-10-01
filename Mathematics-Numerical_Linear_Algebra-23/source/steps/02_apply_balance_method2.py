"""
Apply the Balance Method 2 remove-or-keep decision to the harmonic Ritz roots and construct the balanced root set by appending the corresponding balancing root.

Use the Balance Method 2 candidate information defined from the harmonic Ritz roots. Let



$$

\\varphi'(0) = \\sum_i \\frac{1}{\\theta_i},

$$



and let $\\xi$ denote the reciprocal contribution of the candidate selected by the minimum-difference rule.



Remove the selected candidate only when



$$

\\left|\\varphi'(0) - \\xi\\right| < \\left|\\varphi'(0)\\right|.

$$



If this strict inequality holds, remove the complete selected candidate from the original root set, identified by its source indices. A selected real root removes exactly the one entry at its primary index, even when the same real value occurs elsewhere in the input. A selected complex-conjugate candidate removes exactly the two entries at its primary and partner indices.



For the removal branch, define the retained reciprocal sum



$$

s_{\\text{ret}} = \\varphi'(0) - \\xi

$$



and compute the balancing root



$$

\\eta = -\\frac{1}{s_{\\text{ret}}}.

$$



If the strict inequality does not hold, retain all original roots and instead use $s_{\\text{ret}} = \\varphi'(0)$, so that



$$

\\eta = -\\frac{1}{\\varphi'(0)}.

$$



The balancing root is real for the valid conjugate-symmetric inputs considered here.



Use evaluate_balance_candidates to obtain the candidate table and selected candidate.



Return the retained original harmonic Ritz roots in their original input order, followed by the balancing root $\\eta$ as the final entry.



The balancing denominator must be nonzero to numerical tolerance. Invalid input (empty or non-one-dimensional array, non-finite entries, invalid candidate structure, or a numerically zero balancing denominator) must raise ValueError.

Returns
-------
A one-dimensional complex NumPy array containing the retained original roots in input order followed by the balancing root.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def apply_balance_method2(
    harmonic_ritz_roots: np.ndarray,
) -> np.ndarray:
    """
    Apply Balance Method 2 and construct the balanced root set.

    Parameters
    ----------
    harmonic_ritz_roots : np.ndarray
        One-dimensional array of nonzero harmonic Ritz roots with
        complex roots occurring in conjugate pairs.

    Returns
    -------
    np.ndarray
        Complex-valued one-dimensional array containing the retained
        original roots in input order followed by the real balancing
        root.

    Raises
    ------
    ValueError
        If the array is empty or not one-dimensional, contains non-finite
        entries, or the balancing denominator is numerically zero.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_apply_balance_method2(
    harmonic_ritz_roots,
):
    roots = np.asarray(
        harmonic_ritz_roots,
        dtype=complex,
    )

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

    candidate_table = _oracle_evaluate_balance_candidates(
        roots
    )

    selected_rows = np.flatnonzero(
        candidate_table[:, 6] == 1.0
    )

    if selected_rows.size != 1:
        raise ValueError(
            "Exactly one Balance Method 2 candidate must be selected."
        )

    row = int(selected_rows[0])

    primary_index = int(
        candidate_table[row, 0]
    )

    partner_index = int(
        candidate_table[row, 1]
    )

    multiplicity = int(
        candidate_table[row, 2]
    )

    xi = float(
        candidate_table[row, 3]
    )

    selected_difference = float(
        candidate_table[row, 4]
    )

    phi_prime_0 = float(
        candidate_table[row, 5]
    )

    remove_selected = (
        selected_difference
        <
        abs(phi_prime_0)
    )

    keep = np.ones(
        roots.size,
        dtype=bool,
    )

    if remove_selected:
        keep[primary_index] = False

        if multiplicity == 2:
            if partner_index < 0:
                raise ValueError(
                    "A conjugate-pair candidate must specify its partner."
                )

            keep[partner_index] = False

        elif multiplicity != 1:
            raise ValueError(
                "Candidate multiplicity must be one or two."
            )

        retained_sum = (
            phi_prime_0
            -
            xi
        )

    else:
        retained_sum = phi_prime_0

    tol = 1.0e-12

    if abs(retained_sum) <= tol:
        raise ValueError(
            "The balancing-root denominator is numerically zero."
        )

    eta = -1.0 / retained_sum

    balanced_roots = np.concatenate((
        roots[keep],
        np.array(
            [complex(eta, 0.0)],
            dtype=complex,
        ),
    ))

    return balanced_roots

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
""",
            "call": """apply_balance_method2(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_apply_balance_method2(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    -6.0,
    -2.0,
    0.8,
    3.5
], dtype=complex)
""",
            "call": """apply_balance_method2(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_apply_balance_method2(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    1.0,
    -1.1111111111111112
], dtype=complex)
""",
            "call": """apply_balance_method2(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_apply_balance_method2(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    0.5,
    0.5,
    -3.0,
    -6.0
], dtype=complex)
""",
            "call": """apply_balance_method2(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_apply_balance_method2(
    harmonic_ritz_roots
)""",
        },
        {
            "setup": """import numpy as np

harmonic_ritz_roots = np.array([
    2.0,
    -2.0,
    1.0 + 1.0j,
    1.0 - 1.0j,
    1.0 + 1.0j,
    1.0 - 1.0j
], dtype=complex)
""",
            "call": """apply_balance_method2(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_apply_balance_method2(
    harmonic_ritz_roots
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
""",
            "call": """apply_balance_method2(
    harmonic_ritz_roots
)""",
            "gold_call": """_oracle_apply_balance_method2(
    harmonic_ritz_roots
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

import numpy as np

harmonic_ritz_roots = np.array([
    2.0,
    -2.0
], dtype=complex)
""",
            "call": """_run_invalid(
    apply_balance_method2, harmonic_ritz_roots
)""",
            "gold_call": """_run_invalid(
    _oracle_apply_balance_method2, harmonic_ritz_roots
)""",
        },
    ]

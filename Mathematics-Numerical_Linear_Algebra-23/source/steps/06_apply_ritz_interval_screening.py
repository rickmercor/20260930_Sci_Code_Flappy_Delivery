"""
Apply the Ritz-extrema screening procedure of Algorithm 4 to the supplied Hermite-spline candidates and return the screening result for every examined interval.

Each supplied row already contains the interval endpoints $r_j < r_{j+1}$, the endpoint derivative data, the Hermite cubic coefficients $a_j, b_j, c_j$, the selected interior critical point $\\widehat{x}_j$, and the spline value $C_j(\\widehat{x}_j)$, where for $t = x - r_j$



$$

C_j(x) = \\frac{a_j}{6} t^3 + \\frac{b_j}{2} t^2 + c_j t.

$$



Let $\\widetilde{\\theta}_{\\min}$ and $\\widetilde{\\theta}_{\\max}$ denote the supplied real Ritz extrema. The three source alternatives are evaluated independently of the final interval flag, with exactly the following boundary conventions:



$$

\\text{condition}_1:\\quad

\\widehat{x}_j \\le \\widetilde{\\theta}_{\\min} \\le r_{j+1}

\\ \\text{ and }\\

C_j(\\widetilde{\\theta}_{\\min}) > 1,

$$



$$

\\text{condition}_2:\\quad

r_j \\le \\widetilde{\\theta}_{\\max} \\le \\widehat{x}_j

\\ \\text{ and }\\

C_j(\\widetilde{\\theta}_{\\max}) > 1,

$$



$$

\\text{condition}_3:\\quad

\\widetilde{\\theta}_{\\min} \\notin (\\widehat{x}_j, r_{j+1})

\\ \\text{ and }\\

\\widetilde{\\theta}_{\\max} \\notin (r_j, \\widehat{x}_j),

$$



where the intervals in $\\text{condition}_3$ are open. The condition indicators are properties of the source conditions themselves and must be reported for every examined interval, whether or not the interval is flagged.



An interval is flagged when



$$

C_j(\\widehat{x}_j) > 1

\\quad \\text{and} \\quad

(\\text{condition}_1 \\ \\text{or}\\ \\text{condition}_2 \\ \\text{or}\\ \\text{condition}_3).

$$



Consequently, an interval can have a true condition indicator while still being unflagged.



For a flagged interval, the exceedance margin is $C_j(\\widehat{x}_j) - 1 > 0$. For an unflagged interval, the exceedance margin is $0$.



Return the original interval order together with the selected interior critical point, its spline value, the three condition indicators (as 0 or 1), the final screening flag (0 or 1), and the exceedance margin.



Invalid input (array not of shape (n_examined, 9), non-finite entries, non-finite Ritz extrema, $\\widetilde{\\theta}_{\\min} > \\widetilde{\\theta}_{\\max}$, or a critical point not strictly inside its interval) must raise ValueError.

Returns
-------
A real NumPy array of shape (n_examined, 9) containing each interval, its selected spline maximum, the three Ritz-condition indicators, the final flag, and its exceedance margin.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def apply_ritz_interval_screening(
    spline_candidates: np.ndarray,
    ritz_min: float,
    ritz_max: float,
) -> np.ndarray:
    """
    Apply Ritz-extrema screening to Hermite spline candidates.

    Parameters
    ----------
    spline_candidates : np.ndarray
        Real array of shape (n_examined, 9) with columns
        [left_root, right_root, left_derivative, right_derivative,
         a, b, c, x_hat, C_at_x_hat].
    ritz_min : float
        Ritz value with the smallest real part.
    ritz_max : float
        Ritz value with the largest real part.

    Returns
    -------
    np.ndarray
        Real array with one row per examined interval and columns
        [left_root, right_root, x_hat, C_at_x_hat,
         condition_1, condition_2, condition_3,
         flagged, exceedance_margin].

    Raises
    ------
    ValueError
        If the array is not of shape (n_examined, 9), contains non-finite
        entries, the Ritz extrema are non-finite or ritz_min > ritz_max,
        or a critical point is not strictly inside its interval.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max,
):
    table = np.asarray(
        spline_candidates,
        dtype=float,
    )

    ritz_min = float(ritz_min)
    ritz_max = float(ritz_max)

    if table.ndim != 2 or table.shape[1] != 9:
        raise ValueError(
            "spline_candidates must have shape (n_examined, 9)."
        )

    if not np.all(np.isfinite(table)):
        raise ValueError(
            "spline_candidates must contain only finite values."
        )

    if not np.isfinite(ritz_min) or not np.isfinite(ritz_max):
        raise ValueError(
            "Ritz extrema must be finite."
        )

    if ritz_min > ritz_max:
        raise ValueError(
            "ritz_min must not exceed ritz_max."
        )

    rows = []

    for candidate in table:
        left = float(candidate[0])
        right = float(candidate[1])

        a = float(candidate[4])
        b = float(candidate[5])
        c = float(candidate[6])

        x_hat = float(candidate[7])
        c_hat = float(candidate[8])

        if not left < x_hat < right:
            raise ValueError(
                "Each selected critical point must lie strictly inside its interval."
            )

        above_one = (
            c_hat > 1.0
        )

        def _spline_value(x):
            t = x - left

            return (
                (a / 6.0) * t**3
                +
                (b / 2.0) * t**2
                +
                c * t
            )

        condition_1 = False

        if (
            x_hat <= ritz_min <= right
        ):
            condition_1 = (
                _spline_value(ritz_min)
                >
                1.0
            )

        condition_2 = False

        if (
            left <= ritz_max <= x_hat
        ):
            condition_2 = (
                _spline_value(ritz_max)
                >
                1.0
            )

        condition_3 = (
            not (
                x_hat
                <
                ritz_min
                <
                right
            )
            and
            not (
                left
                <
                ritz_max
                <
                x_hat
            )
        )

        flagged = (
            above_one
            and
            (
                condition_1
                or
                condition_2
                or
                condition_3
            )
        )

        margin = (
            c_hat - 1.0
            if flagged
            else 0.0
        )

        rows.append([
            left,
            right,
            x_hat,
            c_hat,
            float(condition_1),
            float(condition_2),
            float(condition_3),
            float(flagged),
            margin,
        ])

    if not rows:
        return np.empty(
            (0, 9),
            dtype=float,
        )

    return np.asarray(
        rows,
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """import numpy as np

spline_candidates = np.array([
    [
        -7.048192771084337,
        -5.0,
         2.269535106753861,
        -1.084403535685587,
         1.695025202717511,
        -3.373380403049850,
         2.269535106753861,
        -6.190670517647341,
         0.884018499411247
    ],
    [
        -1.3,
         0.9,
         1.134807217473884,
        -2.502127547666009,
        -1.695025202717510,
         0.211375557016583,
         1.134807217473884,
        -0.011450895049290,
         1.033329760465761
    ]
], dtype=float)

ritz_min = -4.6
ritz_max = 0.75
""",
            "call": """apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
)""",
            "gold_call": """_oracle_apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
)""",
        },
        {
            "setup": """import numpy as np

spline_candidates = np.array([
    [
        0.0,
        2.0,
        5.0,
       -1.0,
        6.0,
       -9.0,
        5.0,
        0.736237384174027,
        1.641056385130302
    ]
], dtype=float)

ritz_min = 1.0
ritz_max = 3.0
""",
            "call": """apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
)""",
            "gold_call": """_oracle_apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
)""",
        },
        {
            "setup": """import numpy as np

spline_candidates = np.array([
    [
        0.0,
        2.0,
        5.0,
       -1.0,
        6.0,
       -9.0,
        5.0,
        0.736237384174027,
        1.641056385130302
    ]
], dtype=float)

ritz_min = -3.0
ritz_max = 0.5
""",
            "call": """apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
)""",
            "gold_call": """_oracle_apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
)""",
        },
        {
            "setup": """import numpy as np

spline_candidates = np.array([
    [
        0.0,
        2.0,
        5.0,
       -1.0,
        6.0,
       -9.0,
        5.0,
        0.736237384174027,
        1.641056385130302
    ]
], dtype=float)

ritz_min = 0.0
ritz_max = 0.736237384174027
""",
            "call": """apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
)""",
            "gold_call": """_oracle_apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
)""",
        },
        {
            "setup": """import numpy as np

spline_candidates = np.array([
    [
        0.0,
        2.0,
        5.0,
       -1.0,
        6.0,
       -9.0,
        5.0,
        0.736237384174027,
        1.641056385130302
    ]
], dtype=float)

ritz_min = 1.0
ritz_max = 1.5
""",
            "call": """apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
)""",
            "gold_call": """_oracle_apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
)""",
        },
        {
            "setup": """import numpy as np

spline_candidates = np.array([
    [
        -7.048192771084337,
        -5.0,
         2.269535106753861,
        -1.084403535685587,
         1.695025202717511,
        -3.373380403049850,
         2.269535106753861,
        -6.190670517647341,
         0.884018499411247
    ],
    [
        -1.3,
         0.9,
         1.134807217473884,
        -2.502127547666009,
        -1.695025202717510,
         0.211375557016583,
         1.134807217473884,
        -0.011450895049290,
         1.033329760465761
    ]
], dtype=float)

ritz_min = -6.190670517647341
ritz_max = -1.3
""",
            "call": """apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
)""",
            "gold_call": """_oracle_apply_ritz_interval_screening(
    spline_candidates,
    ritz_min,
    ritz_max
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

spline_candidates = np.array([
    [
        0.0,
        2.0,
        5.0,
       -1.0,
        6.0,
       -9.0,
        5.0,
        0.736237384174027,
        1.641056385130302
    ]
], dtype=float)

ritz_min = 3.0
ritz_max = 1.0
""",
            "call": """_run_invalid(
    apply_ritz_interval_screening, spline_candidates, ritz_min, ritz_max
)""",
            "gold_call": """_run_invalid(
    _oracle_apply_ritz_interval_screening, spline_candidates, ritz_min, ritz_max
)""",
        },
    ]

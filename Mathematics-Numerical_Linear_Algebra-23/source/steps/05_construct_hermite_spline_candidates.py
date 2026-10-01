"""
Construct the Hermite cubic-spline candidates on the eligible intervals between consecutive real balanced roots and determine the relevant interior critical point and spline value for each examined interval.

Let the sorted real roots of the balanced residual polynomial be



$$

r_1 < r_2 < \\cdots < r_s,

$$



with corresponding derivative values $m_j = \\pi'(r_j)$.



With $\\tau = 10^{-12}$ and $s_j = \\max(1, |m_j|, |m_{j+1}|)$, a derivative is treated as zero when its magnitude is at most $\\tau s_j$. An interval $(r_j, r_{j+1})$ is examined only when $m_j \\ge -\\tau s_j$ (nonnegative to tolerance) and the two endpoint derivatives are not both zero to tolerance.



For an examined interval, define $h_j = r_{j+1} - r_j$ and $t = x - r_j$. Construct the cubic spline



$$

C_j(x) = \\frac{a_j}{6} t^3 + \\frac{b_j}{2} t^2 + c_j t,

$$



where



$$

a_j = \\frac{6\\,(m_{j+1} + m_j)}{h_j^2},

\\qquad

b_j = -\\frac{2 m_{j+1} + 4 m_j}{h_j},

\\qquad

c_j = m_j.

$$



The critical points of $C_j$ solve $\\tfrac{1}{2} a_j t^2 + b_j t + c_j = 0$. When $|a_j| > \\tau \\max(1, |a_j|, |b_j|, |c_j|)$ this gives the source formula



$$

\\widehat{x} = r_j + \\frac{-b_j \\pm \\sqrt{b_j^2 - 2 a_j c_j}}{a_j},

$$



whose discriminant is nonnegative for every eligible interval; when $a_j$ is zero to that tolerance the cubic degenerates to a parabola and the single critical point is $\\widehat{x} = r_j - c_j / b_j$.



A critical point is interior when $r_j + \\tau h < \\widehat{x} < r_{j+1} - \\tau h$ with $\\tau h = \\tau \\max(1, h_j)$. Among the interior critical points, select the one at which $C_j$ attains a local maximum, that is $C_j''(\\widehat{x}) = a_j (\\widehat{x} - r_j) + b_j < 0$; a cubic has at most one such point. If an examined interval has no interior local maximum (this happens only when $m_j$ is zero to tolerance and $m_{j+1} > 0$, so that $C_j \\le 0$ on the interval), the interval cannot exceed the positivity threshold and is omitted from the returned table.



Return one row for each examined interval, in increasing interval order, with columns



$$

[r_j,\\ r_{j+1},\\ m_j,\\ m_{j+1},\\ a_j,\\ b_j,\\ c_j,\\ \\widehat{x}_j,\\ C_j(\\widehat{x}_j)].

$$



Intervals that fail the eligibility rule, and examined intervals without an interior local maximum, are omitted.



Invalid input (array not of shape (n_real, 2), fewer than two roots, non-finite entries, or roots not strictly increasing) must raise ValueError.

Returns
-------
A real NumPy array of shape (n_examined, 9), with one row per eligible interval containing its endpoint data, spline coefficients, selected interior critical point, and spline value.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def construct_hermite_spline_candidates(
    real_root_derivatives: np.ndarray,
) -> np.ndarray:
    """
    Construct Hermite spline candidates on eligible real-root intervals.

    Parameters
    ----------
    real_root_derivatives : np.ndarray
        Real array of shape (n_real, 2) with columns
        [sorted_real_root, derivative_value].

    Returns
    -------
    np.ndarray
        Real array with one row per eligible interval and columns
        [left_root, right_root, left_derivative, right_derivative,
         a, b, c, x_hat, C_at_x_hat].

    Raises
    ------
    ValueError
        If the array is not of shape (n_real, 2), has fewer than two rows,
        contains non-finite entries, or roots are not strictly increasing.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_construct_hermite_spline_candidates(
    real_root_derivatives,
):
    data = np.asarray(
        real_root_derivatives,
        dtype=float,
    )

    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError(
            "real_root_derivatives must have shape (n_real, 2)."
        )

    if data.shape[0] < 2:
        raise ValueError(
            "At least two real roots are required."
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(
            "real_root_derivatives must contain only finite values."
        )

    roots = data[:, 0]
    derivatives = data[:, 1]

    if np.any(np.diff(roots) <= 0.0):
        raise ValueError(
            "Real roots must be strictly increasing."
        )

    tol = 1.0e-12
    rows = []

    for j in range(roots.size - 1):
        left = float(roots[j])
        right = float(roots[j + 1])

        m_left = float(derivatives[j])
        m_right = float(derivatives[j + 1])

        scale = max(
            1.0,
            abs(m_left),
            abs(m_right),
        )

        if m_left < -tol * scale:
            continue

        if (
            abs(m_left) <= tol * scale
            and
            abs(m_right) <= tol * scale
        ):
            continue

        h = right - left

        a = (
            6.0
            * (m_right + m_left)
            / (h * h)
        )

        b = (
            -(
                2.0 * m_right
                +
                4.0 * m_left
            )
            / h
        )

        c = m_left

        coefficient_scale = max(
            1.0,
            abs(a),
            abs(b),
            abs(c),
        )

        x_tol = tol * max(
            1.0,
            h,
        )

        if abs(a) <= tol * coefficient_scale:
            if abs(b) <= tol * coefficient_scale:
                raise ValueError(
                    "A degenerate spline must have a nonzero linear derivative coefficient."
                )

            offsets = [-c / b]

        else:
            discriminant = (
                b * b
                -
                2.0 * a * c
            )

            disc_scale = max(
                1.0,
                b * b,
                abs(2.0 * a * c),
            )

            if discriminant < -tol * disc_scale:
                raise ValueError(
                    "The eligible spline has no real critical point."
                )

            discriminant = max(
                discriminant,
                0.0,
            )

            sqrt_disc = np.sqrt(
                discriminant
            )

            offsets = [
                (-b - sqrt_disc) / a,
                (-b + sqrt_disc) / a,
            ]

        maxima = []

        for t in offsets:
            if not (
                t > x_tol
                and
                t < h - x_tol
            ):
                continue

            if a * t + b >= 0.0:
                continue

            if not any(
                abs(t - previous) <= x_tol
                for previous in maxima
            ):
                maxima.append(
                    float(t)
                )

        if len(maxima) == 0:
            continue

        if len(maxima) != 1:
            raise ValueError(
                "A cubic spline cannot have two interior local maxima."
            )

        t = maxima[0]

        x_hat = left + t

        c_value = (
            (a / 6.0) * t**3
            +
            (b / 2.0) * t**2
            +
            c * t
        )

        rows.append([
            left,
            right,
            m_left,
            m_right,
            a,
            b,
            c,
            x_hat,
            c_value,
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

real_root_derivatives = np.array([
    [-7.048192771084337,  2.269535106753861],
    [-5.0, -1.084403535685587],
    [-1.3,  1.134807217473884],
    [ 0.9, -2.502127547666009]
], dtype=float)
""",
            "call": """construct_hermite_spline_candidates(
    real_root_derivatives
)""",
            "gold_call": """_oracle_construct_hermite_spline_candidates(
    real_root_derivatives
)""",
        },
        {
            "setup": """import numpy as np

real_root_derivatives = np.array([
    [-6.0, -2.972789115646258],
    [-2.0,  0.922902494331066],
    [ 2.625, -0.316592261904762],
    [ 3.5,  0.414682539682540]
], dtype=float)
""",
            "call": """construct_hermite_spline_candidates(
    real_root_derivatives
)""",
            "gold_call": """_oracle_construct_hermite_spline_candidates(
    real_root_derivatives
)""",
        },
        {
            "setup": """import numpy as np

real_root_derivatives = np.array([
    [-20.0, -8.925000000000000],
    [ -1.111111111111111,  1.697522290809327],
    [  1.0, -2.097243750000000]
], dtype=float)
""",
            "call": """construct_hermite_spline_candidates(
    real_root_derivatives
)""",
            "gold_call": """_oracle_construct_hermite_spline_candidates(
    real_root_derivatives
)""",
        },
        {
            "setup": """import numpy as np

real_root_derivatives = np.array([
    [0.0, 1.0],
    [2.0, 1.0],
    [5.0, -0.3]
], dtype=float)
""",
            "call": """construct_hermite_spline_candidates(
    real_root_derivatives
)""",
            "gold_call": """_oracle_construct_hermite_spline_candidates(
    real_root_derivatives
)""",
        },
        {
            "setup": """import numpy as np

real_root_derivatives = np.array([
    [-1.0,  2.0],
    [ 1.0, -2.0],
    [ 3.0,  0.0],
    [ 6.0,  1.5]
], dtype=float)
""",
            "call": """construct_hermite_spline_candidates(
    real_root_derivatives
)""",
            "gold_call": """_oracle_construct_hermite_spline_candidates(
    real_root_derivatives
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

real_root_derivatives = np.array([
    [-1.3,  1.134807217473884],
    [-5.0, -1.084403535685587],
    [ 0.9, -2.502127547666009]
], dtype=float)
""",
            "call": """_run_invalid(
    construct_hermite_spline_candidates, real_root_derivatives
)""",
            "gold_call": """_run_invalid(
    _oracle_construct_hermite_spline_candidates, real_root_derivatives
)""",
        },
    ]

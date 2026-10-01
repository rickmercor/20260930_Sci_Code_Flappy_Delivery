"""
Evaluate one curvature-based spline constitutive function and its first two

derivatives at a set of arguments, given the interpolation domain and the

unconstrained free variables that parameterise it.

A scalar constitutive function f on the interpolation domain [x_1, x_end] is

represented through its curvature rather than through its values. The second

derivative is expanded in the degree-one B-spline basis of the uniform grid

x_1 = s_1 < ... < s_nc = x_end, so f'' is the piecewise linear interpolant of the

n_c expansion coefficients at those sites. Those coefficients and the slope d of

f at the left endpoint are not themselves the parameters: each is the image of an

unconstrained variable from theta = (theta_0, theta_1, ..., theta_nc) under one

fixed smooth map onto the positive reals, theta_0 supplying the slope and

theta_1 onwards the curvature coefficients. Positivity of both is what makes f

monotone and convex by construction, so the identification needs no inequality

constraints, and which map is used is part of the method rather than a free

choice. The function follows by integrating the curvature twice, the first

constant of integration being d and the second fixed by the constitutive

normalisation f(x_1) = 0. Both integrals are elementary because f'' is piecewise

linear, so f' is piecewise quadratic and f piecewise cubic. Arguments outside

[x_1, x_end] arise whenever the domain has not yet grown to cover the activated

range, and the method continues the representation there by a controlled

extrapolation rule of its own rather than by extending the polynomial pieces or

by clamping the argument to the domain.

Returns
-------
tuple of three np.ndarray of shape (m,), the values, first derivatives and second derivatives as float64 arrays
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def curvature_spline_values(x_1: float, x_end: float, theta, x):
    """Evaluate a curvature-based spline and its first two derivatives.

    Parameters
    ----------
    x_1 : float
        Left endpoint of the interpolation domain.
    x_end : float
        Right endpoint of the interpolation domain, strictly greater than x_1.
    theta : array-like of shape (n_c + 1,)
        Free variables, theta[0] mapping to the left-endpoint slope and
        theta[1:] mapping to the n_c curvature coefficients.
    x : array-like of shape (m,)
        Arguments at which the function is evaluated.

    Returns
    -------
    spline_values : tuple of three np.ndarray of shape (m,)
        The function values normalised so that f(x_1) = 0, the first
        derivatives and the second derivatives, in that order.

    Raises
    ------
    ValueError
        If x_end is not strictly greater than x_1, if theta holds fewer than
        three entries, or if theta or x is not one dimensional.
    """
    return spline_values

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_curvature_spline_values(x_1: float, x_end: float, theta, x):
    """Reference implementation."""
    theta = np.asarray(theta, dtype=float)
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if theta.ndim != 1 or theta.size < 3:
        raise ValueError("theta must be one dimensional with at least three entries")
    if x.ndim != 1:
        raise ValueError("x must be one dimensional")
    if not float(x_end) > float(x_1):
        raise ValueError("x_end must be strictly greater than x_1")

    d = float(np.logaddexp(0.0, theta[0]))
    c = np.logaddexp(0.0, theta[1:])
    n_c = c.size
    h = (float(x_end) - float(x_1)) / (n_c - 1)

    # Slope and value accumulated at the interpolation sites. The quadrature is
    # exact because f'' is piecewise linear and f' piecewise quadratic.
    slope_at = np.zeros(n_c)
    value_at = np.zeros(n_c)
    for j in range(1, n_c):
        slope_at[j] = slope_at[j - 1] + 0.5 * h * (c[j - 1] + c[j])
        value_at[j] = (value_at[j - 1] + h * (d + slope_at[j - 1])
                       + 0.5 * c[j - 1] * h * h + (c[j] - c[j - 1]) * h * h / 6.0)

    f = np.empty_like(x)
    f_prime = np.empty_like(x)
    f_second = np.empty_like(x)
    for k, xk in enumerate(x):
        if xk < x_1:
            u = xk - x_1
            f_second[k] = c[0]
            f_prime[k] = d + c[0] * u
            f[k] = d * u + 0.5 * c[0] * u * u
        elif xk > x_end:
            u = xk - x_end
            slope_end = d + slope_at[-1]
            f_second[k] = c[-1]
            f_prime[k] = slope_end + c[-1] * u
            f[k] = value_at[-1] + slope_end * u + 0.5 * c[-1] * u * u
        else:
            j = min(int((xk - x_1) / h), n_c - 2)
            u = xk - (x_1 + j * h)
            ramp = (c[j + 1] - c[j]) / h
            f_second[k] = c[j] + ramp * u
            f_prime[k] = d + slope_at[j] + c[j] * u + 0.5 * ramp * u * u
            f[k] = (value_at[j] + (d + slope_at[j]) * u + 0.5 * c[j] * u * u
                    + ramp * u ** 3 / 6.0)
    return f, f_prime, f_second

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: the dissipation potential of the prompt, inside
        # the domain and in the constant-curvature continuation beyond it ---
        {
            "setup": """x_1 = 0.0
x_end = 0.10
theta = [-4.0, -5.0, -5.5, -6.0]
x = [0.0, 0.025, 0.05, 0.10, 0.35, 0.9]
""",
            "call": "[[round(v, 12) for v in a.tolist()] for a in curvature_spline_values(x_1, x_end, theta, x)]",
            "gold_call": "[[round(v, 12) for v in a.tolist()] for a in _oracle_curvature_spline_values(x_1, x_end, theta, x)]",
        },
        # --- Boundary case: the smallest admissible coefficient count, with
        # arguments on both endpoints and on either side of the domain ---
        {
            "setup": """x_1 = 3.0
x_end = 3.02
theta = [-1.0, -2.0, -3.0]
x = [2.5, 3.0, 3.02, 3.5]
""",
            "call": "[[round(v, 12) for v in a.tolist()] for a in curvature_spline_values(x_1, x_end, theta, x)]",
            "gold_call": "[[round(v, 12) for v in a.tolist()] for a in _oracle_curvature_spline_values(x_1, x_end, theta, x)]",
        },
        # --- Edge case: a degenerate interpolation domain must raise ValueError ---
        {
            "setup": """theta = [-1.0, -2.0, -2.5, -3.0]
x = [3.0, 3.1]

def run_model():
    try:
        curvature_spline_values(3.0, 3.0, theta, x)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_curvature_spline_values(3.0, 3.0, theta, x)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

"""
Assemble a moving weak-SAT affine system and a runtime-order time jet.

Let $d=\\operatorname{sign}(c)$ and let $X_\\kappa$ contain positive grid-relative magnitudes. Each diagonal block is $-dX_\\kappa D_\\kappa-\\tau_\\kappa X_\\kappa H_\\kappa^{-1}E_{\\rm in}$, where $E_{\\rm in}$ selects the left endpoint for $d>0$ and the right endpoint for $d<0$.



For positive characteristics, the donor chain is left to middle to right and the matrix is block lower triangular; for negative characteristics, it is right to middle to left and the matrix is block upper triangular. The two receiver rows use computational-space donor weights, while only the upstream outer grid receives the exact Gaussian inflow $\\exp[-\\beta(x_{\\rm in}-x_0-ct)^2]$.



Also assemble a runtime-order time jet of the complete affine system. The integer $p$ may range from zero through eight and defaults to six. The middle translation is $A\\sin(\\omega t)$ with $\\omega=2\\pi/T$, so translation derivatives through order $p+1$ must be propagated through the discrete metric, incoming penalty coefficients, and both donor couplings.



Treat every interpolation row as the composition of the inverse cubic donor map with the moving receiver-donor geometry. Request spatial derivatives through order $p$, convert all raw derivatives to Taylor coefficients, compose with the signed receiver-donor displacement jet, and convolve every coefficient-row outer product through degree $p$. The receiver and donor roles, rather than the characteristic sign alone, determine the displacement signs.



Expand the physical inflow $g(t)=\\exp[-\\beta(x_{\\rm in}-x_0-ct)^2]$ as a formal Taylor series and multiply it by its time-dependent penalty coefficient through degree $p$. No finite differences may be used. Return raw derivative channels $d^k[M|b]/dt^k$ for $k=0,\\ldots,p$, in order. Penalties must be at least $1/2$, donor receivers must remain inside their donor intervals, and receiver values are never overwritten.

Returns
-------
shape (p+1, total, total+1): raw derivatives d^k[M|b]/dt^k, k=0,...,p
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def assemble_weak_overset_system(
    t: float,
    counts: np.ndarray,
    sigmas: np.ndarray,
    c: float,
    amplitude: float,
    period: float,
    penalties: np.ndarray,
    beta: float = 80.0,
    x0: float = -0.4,
    jet_order: int = 6,
) -> np.ndarray:
    r"""Assemble $[M(t)\mid b(t)]$ and a runtime-order analytic time jet.

    Parameters
    ----------
    t : float
        Evaluation time.
    counts : np.ndarray
        Integer counts ``[N_L, N_M, N_R]``, each at least eighteen.
    sigmas : np.ndarray
        Finite stretching parameters ``[sigma_L, sigma_M, sigma_R]``, each of
        magnitude below one.
    c : float
        Finite nonzero physical advection speed. Its sign selects the incoming
        side on every grid.
    amplitude : float
        Middle-grid translation amplitude.
    period : float
        Positive translation period.
    penalties : np.ndarray
        Three finite penalties, each at least $1/2$.
    beta : float
        Positive Gaussian scaling parameter.
    x0 : float
        Initial Gaussian centre.
    jet_order : int
        Highest requested raw time derivative, from zero through eight
        inclusive. The default is six.

    Returns
    -------
    np.ndarray
        Array of shape
        ``(jet_order + 1, counts.sum(), counts.sum() + 1)``. Channel ``k`` is
        the raw derivative ``d^k[M | b]/dt^k`` at the supplied time.

    Raises
    ------
    ValueError
        For invalid ``counts`` or ``sigmas``; a penalty below ``0.5``;
        non-finite scalar inputs; zero ``c``; non-positive ``period`` or
        ``beta``; a ``jet_order`` outside zero through eight; inconsistent
        grid-relative characteristic signs; or a receiver that leaves its
        donor grid.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from math import factorial

import numpy as np


def _validated_system_inputs(
    t, counts, sigmas, c, amplitude, period, penalties, beta, x0, jet_order
):
    counts = np.asarray(counts)
    penalties = np.asarray(penalties, dtype=float)
    if counts.shape != (3,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("counts must be an integer array of shape (3,)")
    if np.any(counts < 18):
        raise ValueError("every grid must have at least eighteen nodes")
    if penalties.shape != (3,) or not np.all(np.isfinite(penalties)):
        raise ValueError("penalties must be a finite array of shape (3,)")
    if np.any(penalties < 0.5):
        raise ValueError("every penalty must be at least 0.5")
    values = np.asarray([t, c, amplitude, period, beta, x0], dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("all scalar inputs must be finite")
    if float(c) == 0.0 or float(period) <= 0.0 or float(beta) <= 0.0:
        raise ValueError("c must be nonzero; period and beta must be positive")
    if isinstance(jet_order, (bool, np.bool_)) or not isinstance(
        jet_order, (int, np.integer)
    ):
        raise ValueError("jet_order must be an integer")
    if not 0 <= int(jet_order) <= 8:
        raise ValueError("jet_order must be between zero and eight")
    return (
        float(t),
        counts.astype(int),
        float(c),
        float(amplitude),
        float(period),
        penalties,
        float(beta),
        float(x0),
        int(jet_order),
    )


def _jet_product(first, second):
    """Multiply two truncated Taylor-coefficient jets."""
    order = first.shape[0] - 1
    trailing_shape = np.broadcast_shapes(first.shape[1:], second.shape[1:])
    result = np.zeros((order + 1,) + trailing_shape, dtype=float)
    for total_order in range(order + 1):
        for first_order in range(total_order + 1):
            result[total_order] += (
                first[first_order] * second[total_order - first_order]
            )
    return result


def _jet_compose(outer, inner):
    """Compose a Taylor jet with a scalar zero-constant Taylor jet."""
    result = np.zeros_like(outer, dtype=float)
    for coefficient in outer[::-1]:
        result = _jet_product(result, inner)
        result[0] += coefficient
    return result


def _jet_exponential(exponent):
    """Exponentiate a scalar Taylor-coefficient jet."""
    order = exponent.size - 1
    result = np.zeros(order + 1, dtype=float)
    result[0] = np.exp(exponent[0])
    for total_order in range(1, order + 1):
        result[total_order] = (
            sum(
                exponent_order
                * exponent[exponent_order]
                * result[total_order - exponent_order]
                for exponent_order in range(1, total_order + 1)
            )
            / total_order
        )
    return result


def _jet_outer(first, second):
    """Form the Taylor jet of an outer product."""
    order = first.shape[0] - 1
    result = np.zeros((order + 1, first.shape[1], second.shape[1]), dtype=float)
    for total_order in range(order + 1):
        for first_order in range(total_order + 1):
            result[total_order] += np.outer(
                first[first_order], second[total_order - first_order]
            )
    return result


def _oracle_assemble_weak_overset_system(
    t: float,
    counts: np.ndarray,
    sigmas: np.ndarray,
    c: float,
    amplitude: float,
    period: float,
    penalties: np.ndarray,
    beta: float = 80.0,
    x0: float = -0.4,
    jet_order: int = 6,
) -> np.ndarray:
    """Reference affine-system value and exact runtime-order time jet."""
    (
        t,
        counts,
        c,
        amplitude,
        period,
        penalties,
        beta,
        x0,
        jet_order,
    ) = _validated_system_inputs(
        t,
        counts,
        sigmas,
        c,
        amplitude,
        period,
        penalties,
        beta,
        x0,
        jet_order,
    )
    state = _oracle_compute_moving_grid_state(t, counts, sigmas, amplitude, period)
    nodes = state[1:]
    angular_frequency = 2.0 * np.pi / period
    derivative_indices = np.arange(jet_order + 2)
    shift_derivatives = (
        amplitude
        * angular_frequency**derivative_indices
        * np.sin(angular_frequency * t + 0.5 * np.pi * derivative_indices)
    )
    velocity = float(shift_derivatives[1])
    metric = _oracle_compute_relative_metric_diagonals(c, velocity, counts, nodes)

    direction = 1 if c > 0.0 else -1
    if np.sign(c - velocity) != direction:
        raise ValueError("all grid-relative characteristics must share c's sign")

    starts = np.r_[0, np.cumsum(counts)]
    left = nodes[starts[0] : starts[1]]
    middle = nodes[starts[1] : starts[2]]
    right = nodes[starts[2] : starts[3]]
    factorials = np.array([float(factorial(order)) for order in range(jet_order + 1)])

    metric_series = np.zeros((jet_order + 1, nodes.size), dtype=float)
    metric_series[0] = metric
    middle_derivative = _oracle_build_sbp63_operator(int(counts[1]))[2]
    middle_jacobian = middle_derivative @ middle
    for order in range(1, jet_order + 1):
        metric_series[order, starts[1] : starts[2]] = (
            -direction
            * shift_derivatives[order + 1]
            / (factorials[order] * middle_jacobian)
        )

    if direction > 0:
        if middle[0] < left[0] or middle[0] > left[-1]:
            raise ValueError("the middle receiver has left the left donor grid")
        if right[0] < middle[0] or right[0] > middle[-1]:
            raise ValueError("the right receiver has left the middle donor grid")
        first_data = _oracle_compute_donor_interpolation(left, middle[0], jet_order)
        second_data = _oracle_compute_donor_interpolation(middle, right[0], jet_order)
        first_shift_sign = 1.0
        second_shift_sign = -1.0
    else:
        if left[-1] < middle[0] or left[-1] > middle[-1]:
            raise ValueError("the left receiver has left the middle donor grid")
        if middle[-1] < right[0] or middle[-1] > right[-1]:
            raise ValueError("the middle receiver has left the right donor grid")
        first_data = _oracle_compute_donor_interpolation(middle, left[-1], jet_order)
        second_data = _oracle_compute_donor_interpolation(right, middle[-1], jet_order)
        first_shift_sign = -1.0
        second_shift_sign = 1.0

    first_displacement = np.zeros(jet_order + 1, dtype=float)
    second_displacement = np.zeros(jet_order + 1, dtype=float)
    first_displacement[1:] = (
        first_shift_sign * shift_derivatives[1 : jet_order + 1] / factorials[1:]
    )
    second_displacement[1:] = (
        second_shift_sign * shift_derivatives[1 : jet_order + 1] / factorials[1:]
    )
    first_weight_series = _jet_compose(
        first_data / factorials[:, None], first_displacement
    )
    second_weight_series = _jet_compose(
        second_data / factorials[:, None], second_displacement
    )

    total = int(np.sum(counts))
    matrix_series = np.zeros((jet_order + 1, total, total), dtype=float)
    forcing_series = np.zeros((jet_order + 1, total), dtype=float)
    incoming = 0 if direction > 0 else -1
    operators = [_oracle_build_sbp63_operator(int(counts[grid])) for grid in range(3)]
    corners = []
    for grid in range(3):
        h_matrix = operators[grid][0]
        derivative = operators[grid][2]
        corners.append(float(h_matrix[incoming, incoming]))
        begin, end = starts[grid], starts[grid + 1]
        for order in range(jet_order + 1):
            scale = metric_series[order, begin:end]
            block = -direction * scale[:, None] * derivative
            block[incoming, incoming] -= (
                penalties[grid] * scale[incoming] / h_matrix[incoming, incoming]
            )
            matrix_series[order, begin:end, begin:end] = block

    if direction > 0:
        first_column = np.zeros((jet_order + 1, int(counts[1])), dtype=float)
        first_column[:, 0] = penalties[1] * metric_series[:, starts[1]] / corners[1]
        second_column = np.zeros((jet_order + 1, int(counts[2])), dtype=float)
        second_column[:, 0] = penalties[2] * metric_series[:, starts[2]] / corners[2]
        matrix_series[:, starts[1] : starts[2], starts[0] : starts[1]] = _jet_outer(
            first_column, first_weight_series
        )
        matrix_series[:, starts[2] : starts[3], starts[1] : starts[2]] = _jet_outer(
            second_column, second_weight_series
        )
        inflow_index = starts[0]
        inflow_grid = 0
        inflow_x = -1.0
    else:
        first_column = np.zeros((jet_order + 1, int(counts[0])), dtype=float)
        first_column[:, -1] = (
            penalties[0] * metric_series[:, starts[1] - 1] / corners[0]
        )
        second_column = np.zeros((jet_order + 1, int(counts[1])), dtype=float)
        second_column[:, -1] = (
            penalties[1] * metric_series[:, starts[2] - 1] / corners[1]
        )
        matrix_series[:, starts[0] : starts[1], starts[1] : starts[2]] = _jet_outer(
            first_column, first_weight_series
        )
        matrix_series[:, starts[1] : starts[2], starts[2] : starts[3]] = _jet_outer(
            second_column, second_weight_series
        )
        inflow_index = starts[3] - 1
        inflow_grid = 2
        inflow_x = 1.0

    residual = inflow_x - x0 - c * t
    inflow_exponent = np.zeros(jet_order + 1, dtype=float)
    inflow_exponent[0] = -beta * residual**2
    if jet_order >= 1:
        inflow_exponent[1] = 2.0 * beta * c * residual
    if jet_order >= 2:
        inflow_exponent[2] = -beta * c**2
    inflow_series = _jet_exponential(inflow_exponent)
    coefficient_series = (
        penalties[inflow_grid] * metric_series[:, inflow_index] / corners[inflow_grid]
    )
    forcing_series[:, inflow_index] = _jet_product(coefficient_series, inflow_series)

    augmented_series = np.concatenate(
        (matrix_series, forcing_series[:, :, None]), axis=2
    )
    return augmented_series * factorials[:, None, None]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return moving, static, varied-order, and combined invalid cases."""
    return [
        {
            "setup": """import numpy as np
t = 0.0
counts = np.array([19, 23, 21])
sigmas = np.array([0.25, -0.15, 0.20])
c, amplitude, period = 1.0, 0.1, 1.0
penalties = np.array([0.75, 0.75, 0.75])
jet_order = 6
""",
            "call": "assemble_weak_overset_system(t, counts, sigmas, c, amplitude, period, penalties, jet_order=jet_order)",
            "gold_call": "_oracle_assemble_weak_overset_system(t, counts, sigmas, c, amplitude, period, penalties, jet_order=jet_order)",
        },
        {
            "setup": """import numpy as np
t = 0.17
counts = np.array([20, 25, 22])
sigmas = np.array([-0.2, 0.35, 0.1])
c, amplitude, period = -1.0, 0.05, 0.8
penalties = np.array([0.6, 0.7, 0.8])
beta, x0 = 60.0, 0.35
jet_order = 8
""",
            "call": "assemble_weak_overset_system(t, counts, sigmas, c, amplitude, period, penalties, beta, x0, jet_order)",
            "gold_call": "_oracle_assemble_weak_overset_system(t, counts, sigmas, c, amplitude, period, penalties, beta, x0, jet_order)",
        },
        {
            "setup": """import numpy as np
t = 0.3
counts = np.array([18, 21, 19])
sigmas = np.array([-0.3, 0.4, 0.0])
c, amplitude, period = 1.0, 0.0, 1.0
penalties = np.array([0.5, 0.6, 0.8])
jet_order = 5
""",
            "call": "assemble_weak_overset_system(t, counts, sigmas, c, amplitude, period, penalties, jet_order=jet_order)",
            "gold_call": "_oracle_assemble_weak_overset_system(t, counts, sigmas, c, amplitude, period, penalties, jet_order=jet_order)",
        },
        {
            "setup": """import numpy as np
t = 0.0
counts = np.array([19, 23, 21])
sigmas = np.array([0.25, -0.15, 0.20])
c, amplitude, period = 1.0, 0.1, 1.0
penalties = np.array([0.75, 0.49, 0.75])
def run_model():
    try:
        assemble_weak_overset_system(t, counts, sigmas, c, amplitude, period, penalties)
        return np.array([0, 0])
    except ValueError:
        first = 1
    except Exception:
        first = 2
    try:
        assemble_weak_overset_system(
            t, counts, sigmas, c, amplitude, period,
            np.array([0.75, 0.75, 0.75]), jet_order=9
        )
        second = 0
    except ValueError:
        second = 1
    except Exception:
        second = 2
    return np.array([first, second])
def run_gold():
    try:
        _oracle_assemble_weak_overset_system(t, counts, sigmas, c, amplitude, period, penalties)
        return np.array([0, 0])
    except ValueError:
        first = 1
    except Exception:
        first = 2
    try:
        _oracle_assemble_weak_overset_system(
            t, counts, sigmas, c, amplitude, period,
            np.array([0.75, 0.75, 0.75]), jet_order=9
        )
        second = 0
    except ValueError:
        second = 1
    except Exception:
        second = 2
    return np.array([first, second])
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]

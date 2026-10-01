"""
Invert a stretched cubic donor map and return a runtime-order weight jet.

For a receiver inside donor cell $[x_j,x_{j+1}]$, use the clamped four-node stencil beginning at $s=\\min(\\max(j-1,0),N-4)$. On local computational coordinates $\\zeta=0,1,2,3$, let $\\ell_a$ be the four cubic cardinal polynomials and form the physical map $\\Phi(\\zeta)=\\sum_{a=0}^3x_{s+a}\\ell_a(\\zeta)$.



Start from the cellwise linear estimate of $\\zeta_r$. Maintain a bracket equal to the containing computational cell; accept a tangent update only when its derivative is positive and the candidate stays strictly inside that bracket, otherwise bisect. Stop once $|\\Phi(\\zeta_r)-x_r|\\le10^{-14}\\max(1,|x_r|)$.



The runtime integer $p$ requests derivatives zero through $p$, where $0\\le p\\le8$. Revert the local Taylor series of $\\Phi$ about the converged $\\zeta_r$ so that its composition equals $x_r+\\delta x$ through degree $p$. Compose every cardinal polynomial with that inverse series and convert Taylor coefficients to raw physical-coordinate derivatives. Direct physical-space interpolation, finite differences, and a fixed collection of low-order chain-rule formulas are not equivalent.



For a scalar receiver return a $(p+1)\\times N$ array; for a batch, prepend the receiver axis. The default is $p=6$. Reject extrapolation, nonmonotone donors, nonfinite data, an invalid derivative order, a nonpositive inverse-map derivative, and failure to converge in 100 iterations.

Returns
-------
scalar -> (p+1, N); q receivers -> (q, p+1, N), p=derivative_order
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_donor_interpolation(
    donor_nodes: np.ndarray,
    receiver_x: float | np.ndarray,
    derivative_order: int = 6,
) -> np.ndarray:
    r"""Return cubic weights and a runtime-order physical derivative jet.

    Parameters
    ----------
    donor_nodes : np.ndarray
        Strictly increasing finite one-dimensional donor coordinates, at least
        four of them.
    receiver_x : float or np.ndarray
        One finite receiver coordinate or a nonempty one-dimensional array of
        receiver coordinates inside the closed donor interval.
    derivative_order : int
        Highest requested receiver-coordinate derivative, from zero through
        eight inclusive. The default is six.

    Returns
    -------
    np.ndarray
        For a scalar receiver, an array of shape
        ``(derivative_order + 1, donor_nodes.size)`` whose rows contain raw
        receiver-coordinate derivatives zero through ``derivative_order``.
        For ``q`` receivers, the leading shape is
        ``(q, derivative_order + 1)``. Every derivative row has at most four
        nonzero entries.

    Raises
    ------
    ValueError
        If ``donor_nodes`` is not one-dimensional, has fewer than four entries,
        contains non-finite data, or is not strictly increasing; if the
        receiver input has invalid shape, is non-finite, or leaves the closed
        donor interval; if ``derivative_order`` is not an integer from zero
        through eight; or if a safeguarded inverse-map iteration fails.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from math import factorial

import numpy as np

_LOCAL_NODES = np.array([0.0, 1.0, 2.0, 3.0])


def _cardinal_coefficients():
    coefficients = np.zeros((4, 4), dtype=float)
    for basis_index in range(4):
        polynomial = np.array([1.0])
        for node_index in range(4):
            if node_index != basis_index:
                factor = np.array([-_LOCAL_NODES[node_index], 1.0])
                polynomial = np.convolve(polynomial, factor) / (
                    _LOCAL_NODES[basis_index] - _LOCAL_NODES[node_index]
                )
        coefficients[basis_index] = polynomial
    return coefficients


_CARDINAL_COEFFICIENTS = _cardinal_coefficients()


def _validated_donor_data(donor_nodes, receiver_x, derivative_order):
    donor_nodes = np.asarray(donor_nodes, dtype=float)
    if donor_nodes.ndim != 1 or donor_nodes.size < 4:
        raise ValueError("donor_nodes must be a 1D array with at least four entries")
    receivers = np.asarray(receiver_x, dtype=float)
    scalar_receiver = receivers.ndim == 0
    if receivers.ndim > 1 or (receivers.ndim == 1 and receivers.size == 0):
        raise ValueError("receiver_x must be a finite scalar or nonempty 1D array")
    receivers = np.atleast_1d(receivers)
    if not np.all(np.isfinite(donor_nodes)) or not np.all(np.isfinite(receivers)):
        raise ValueError("donor data must be finite")
    if np.any(np.diff(donor_nodes) <= 0.0):
        raise ValueError("donor_nodes must be strictly increasing")
    if np.any(receivers < donor_nodes[0]) or np.any(receivers > donor_nodes[-1]):
        raise ValueError("every receiver must lie in the closed donor interval")
    if isinstance(derivative_order, (bool, np.bool_)) or not isinstance(
        derivative_order, (int, np.integer)
    ):
        raise ValueError("derivative_order must be an integer")
    if not 0 <= int(derivative_order) <= 8:
        raise ValueError("derivative_order must be between zero and eight")
    return donor_nodes, receivers, scalar_receiver, int(derivative_order)


def _cubic_basis_value_and_slope(z):
    powers = np.array([1.0, z, z * z, z * z * z])
    values = _CARDINAL_COEFFICIENTS @ powers
    slopes = (
        _CARDINAL_COEFFICIENTS[:, 1]
        + 2.0 * z * _CARDINAL_COEFFICIENTS[:, 2]
        + 3.0 * z * z * _CARDINAL_COEFFICIENTS[:, 3]
    )
    return values, slopes


def _series_product(first, second, order):
    return np.convolve(first, second)[: order + 1]


def _compose_polynomial(polynomial, series, order):
    result = np.zeros(order + 1, dtype=float)
    for coefficient in polynomial[::-1]:
        result = _series_product(result, series, order)
        result[0] += coefficient
    return result


def _oracle_compute_donor_interpolation(
    donor_nodes: np.ndarray,
    receiver_x: float | np.ndarray,
    derivative_order: int = 6,
) -> np.ndarray:
    """Reference batched, safeguarded computational-space interpolation."""
    donor_nodes, receivers, scalar_receiver, derivative_order = _validated_donor_data(
        donor_nodes, receiver_x, derivative_order
    )
    size = donor_nodes.size
    result = np.zeros((receivers.size, derivative_order + 1, size), dtype=float)
    for receiver_index, receiver in enumerate(receivers):
        cell = int(np.searchsorted(donor_nodes, receiver, side="right")) - 1
        cell = min(max(cell, 0), size - 2)
        start = min(max(cell - 1, 0), size - 4)
        stencil = donor_nodes[start : start + 4]

        lower = float(cell - start)
        upper = lower + 1.0
        z = lower + (receiver - donor_nodes[cell]) / (
            donor_nodes[cell + 1] - donor_nodes[cell]
        )
        tolerance = 1e-14 * max(1.0, abs(float(receiver)))
        for _ in range(100):
            values, slopes = _cubic_basis_value_and_slope(z)
            residual = float(stencil @ values) - float(receiver)
            if abs(residual) <= tolerance:
                break
            if residual < 0.0:
                lower = z
            else:
                upper = z
            jacobian = float(stencil @ slopes)
            candidate = z - residual / jacobian if jacobian > 0.0 else np.nan
            if not np.isfinite(candidate) or candidate <= lower or candidate >= upper:
                candidate = 0.5 * (lower + upper)
            z = candidate
        else:
            raise ValueError("the donor inverse map did not converge")

        values, slopes = _cubic_basis_value_and_slope(z)
        map_slope = float(stencil @ slopes)
        if not np.isfinite(map_slope) or map_slope <= 0.0:
            raise ValueError("the donor inverse map has a nonpositive derivative")

        map_polynomial = stencil @ _CARDINAL_COEFFICIENTS
        inverse_series = np.zeros(derivative_order + 1, dtype=float)
        inverse_series[0] = z
        for order in range(1, derivative_order + 1):
            known = _compose_polynomial(map_polynomial, inverse_series, order)[order]
            target = 1.0 if order == 1 else 0.0
            inverse_series[order] = (target - known) / map_slope
        for basis_index, polynomial in enumerate(_CARDINAL_COEFFICIENTS):
            composed = _compose_polynomial(polynomial, inverse_series, derivative_order)
            for order in range(derivative_order + 1):
                result[receiver_index, order, start + basis_index] = (
                    factorial(order) * composed[order]
                )
    return result[0] if scalar_receiver else result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three derivative orders and combined invalid cases."""
    return [
        {
            "setup": """import numpy as np
xi = np.linspace(0.0, 1.0, 41)
donor = -1.0 + 0.9 * (xi - (0.25 / np.pi) * np.sin(np.pi * xi))
receiver = -0.35
derivative_order = 8
""",
            "call": "compute_donor_interpolation(donor, receiver, derivative_order)",
            "gold_call": "_oracle_compute_donor_interpolation(donor, receiver, derivative_order)",
        },
        {
            "setup": """import numpy as np
xi = np.linspace(0.0, 1.0, 17)
donor = -0.35 + 0.7 * (xi - (0.78 / np.pi) * np.sin(np.pi * xi))
receiver = np.array([
    donor[0] + 0.03 * (donor[1] - donor[0]),
    0.5 * (donor[7] + donor[8]),
    donor[-2] + 0.97 * (donor[-1] - donor[-2]),
])
derivative_order = 6
""",
            "call": "compute_donor_interpolation(donor, receiver, derivative_order)",
            "gold_call": "_oracle_compute_donor_interpolation(donor, receiver, derivative_order)",
        },
        {
            "setup": """import numpy as np
xi = np.linspace(0.0, 1.0, 13)
donor = 0.1 + 0.9 * (xi - (0.2 / np.pi) * np.sin(np.pi * xi))
receiver = np.array([donor[0], donor[4], donor[-1], donor[4]])
derivative_order = 4
""",
            "call": "compute_donor_interpolation(donor, receiver, derivative_order)",
            "gold_call": "_oracle_compute_donor_interpolation(donor, receiver, derivative_order)",
        },
        {
            "setup": """import numpy as np
donor = np.linspace(-1.0, -0.1, 12)
receiver = 0.0
def run_model():
    try:
        compute_donor_interpolation(donor, receiver)
        return np.array([0, 0])
    except ValueError:
        first = 1
    except Exception:
        first = 2
    try:
        compute_donor_interpolation(np.linspace(0.0, 1.0, 9), 0.5, 9)
        second = 0
    except ValueError:
        second = 1
    except Exception:
        second = 2
    return np.array([first, second])
def run_gold():
    try:
        _oracle_compute_donor_interpolation(donor, receiver)
        return np.array([0, 0])
    except ValueError:
        first = 1
    except Exception:
        first = 2
    try:
        _oracle_compute_donor_interpolation(np.linspace(0.0, 1.0, 9), 0.5, 9)
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

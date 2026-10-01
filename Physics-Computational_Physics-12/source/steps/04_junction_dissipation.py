"""
Registered event frequencies and the dissipation carried by the resolved junctions.

The stationary distribution $\pi$ of the embedded chain solves $\pi = \pi P$ with $\sum_u \pi_u = 1$, the mean time between registered events is $\langle t\rangle = \sum_u \pi_u \sum_v M^{(1)}_{uv}$, and the registered frequency of type $u$ is $\nu_u = \pi_u/\langle t\rangle$.



The entropy production rate carried by junction $L$ is the product of the net rate $J_L$ at which the device traverses that junction in the forward direction and the affinity of the junction,



$$\sigma_{L} = J_L\,\ln\frac{k_{L+}}{k_{L-}}.$$

Returns
-------
A NumPy array of shape `(6,)` holding `[nu_a_plus, nu_a_minus, nu_b_plus, nu_b_minus, sigma_a, sigma_b]`, the frequencies in inverse microseconds and the two dissipation rates in Boltzmann constants per microsecond. Raise `ValueError` for a shape mismatch, a registration probability outside $(0,1]$, a non-positive rate constant, a non-positive mean waiting time, or a non-finite entry.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Registered event frequencies and the dissipation carried by the resolved junctions."""

import numpy as np


def junction_dissipation(moments, detection_and_rates):
    """Convert a semi-Markov kernel and its registration model into junction dissipation.

    Parameters
    ----------
    moments : array_like, shape (3, 4, 4)
        ``[P, M1, M2]`` for the recorded kernel, as returned by the moment stage.
    detection_and_rates : array_like, shape (2, 4)
        Row ``L`` holds ``[eta_plus, eta_minus, k_plus, k_minus]`` for junction ``L``.

    Returns
    -------
    numpy.ndarray, shape (6,)
        ``[nu_a_plus, nu_a_minus, nu_b_plus, nu_b_minus, sigma_a, sigma_b]``:
        the four registered event frequencies in inverse microseconds followed
        by the entropy production rate carried by each resolved junction in
        Boltzmann constants per microsecond.
    """
    return np.zeros(6)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _stationary(matrix):
    import numpy as np
    system = np.vstack([matrix.T - np.eye(4), np.ones(4)])
    target = np.array([0.0, 0.0, 0.0, 0.0, 1.0])
    solution, _, _, _ = np.linalg.lstsq(system, target, rcond=None)
    return solution


def _oracle_junction_dissipation(moments, detection_and_rates):
    import numpy as np
    stack = np.asarray(moments, dtype=float)
    values = np.asarray(detection_and_rates, dtype=float)
    if stack.shape != (3, 4, 4) or not np.all(np.isfinite(stack)):
        raise ValueError("moments must be a finite array of shape (3, 4, 4)")
    if values.shape != (2, 4) or not np.all(np.isfinite(values)):
        raise ValueError("detection_and_rates must be a finite array of shape (2, 4)")
    if np.any(values[:, :2] <= 0.0) or np.any(values[:, :2] > 1.0):
        raise ValueError("registration probabilities must lie in (0, 1]")
    if np.any(values[:, 2:] <= 0.0):
        raise ValueError("every junction rate constant must be positive")
    transition, first = stack[0], stack[1]
    weights = _stationary(transition)
    mean_time = float(weights @ first.sum(axis=1))
    if mean_time <= 0.0:
        raise ValueError("mean registered waiting time must be positive")
    frequencies = weights / mean_time
    dissipation = np.zeros(2)
    for link in range(2):
        eta_plus, eta_minus, k_plus, k_minus = values[link]
        restored = frequencies[2 * link] / eta_plus - frequencies[2 * link + 1] / eta_minus
        dissipation[link] = restored * np.log(k_plus / k_minus)
    return np.concatenate([frequencies, dissipation])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Differential cases for the restored junction dissipation."""
    asymmetric = (
        "import numpy as np\n"
        "P = np.array([[0.05, 0.72, 0.19, 0.04],\n"
        "              [0.17, 0.40, 0.35, 0.08],\n"
        "              [0.10, 0.31, 0.33, 0.26],\n"
        "              [0.06, 0.37, 0.52, 0.05]])\n"
        "P = P / P.sum(axis=1).reshape(4, 1)\n"
        "M1 = np.array([[0.016, 0.147, 0.069, 0.014],\n"
        "               [0.049, 0.193, 0.153, 0.044],\n"
        "               [0.058, 0.212, 0.186, 0.085],\n"
        "               [0.026, 0.140, 0.113, 0.026]])\n"
        "M2 = np.array([[0.013, 0.079, 0.050, 0.014],\n"
        "               [0.038, 0.161, 0.125, 0.041],\n"
        "               [0.053, 0.211, 0.172, 0.066],\n"
        "               [0.022, 0.103, 0.078, 0.024]])\n"
        "moments = np.stack([P, M1, M2])\n"
        "rates = np.array([[0.35, 0.88, 2.9, 6.1], [0.78, 0.42, 6.2, 3.2]])\n"
    )
    complete = (
        "import numpy as np\n"
        "P = np.full((4, 4), 0.25)\n"
        "M1 = np.full((4, 4), 0.2)\n"
        "M2 = np.full((4, 4), 0.11)\n"
        "moments = np.stack([P, M1, M2])\n"
        "rates = np.array([[1.0, 1.0, 4.0, 1.6], [1.0, 1.0, 2.5, 5.0]])\n"
    )
    skewed = (
        "import numpy as np\n"
        "P = np.array([[0.02, 0.90, 0.06, 0.02],\n"
        "              [0.88, 0.03, 0.05, 0.04],\n"
        "              [0.04, 0.05, 0.06, 0.85],\n"
        "              [0.03, 0.06, 0.88, 0.03]])\n"
        "P = P / P.sum(axis=1).reshape(4, 1)\n"
        "M1 = np.array([[0.01, 0.55, 0.04, 0.01],\n"
        "               [0.61, 0.02, 0.03, 0.02],\n"
        "               [0.02, 0.03, 0.04, 0.48],\n"
        "               [0.02, 0.04, 0.52, 0.02]])\n"
        "M2 = np.array([[0.02, 1.40, 0.09, 0.02],\n"
        "               [1.60, 0.04, 0.07, 0.05],\n"
        "               [0.05, 0.07, 0.09, 1.20],\n"
        "               [0.04, 0.09, 1.30, 0.04]])\n"
        "moments = np.stack([P, M1, M2])\n"
        "rates = np.array([[0.15, 0.98, 9.0, 1.2], [0.55, 0.30, 2.0, 7.5]])\n"
    )
    return [
        {
            "setup": asymmetric,
            "call": "junction_dissipation(moments, rates)",
            "gold_call": "_oracle_junction_dissipation(moments, rates)",
        },
        {
            "setup": complete,
            "call": "junction_dissipation(moments, rates)",
            "gold_call": "_oracle_junction_dissipation(moments, rates)",
        },
        {
            "setup": skewed,
            "call": "junction_dissipation(moments, rates)",
            "gold_call": "_oracle_junction_dissipation(moments, rates)",
        },
    ]

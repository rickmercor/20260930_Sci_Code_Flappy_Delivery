"""
Solve the complete bounded inverse prolate-Ewald problem and return the recovered force-error parameter.

Build the reciprocal index cube, transform the interior physical initial guess to logistic coordinates, and repeatedly apply the deterministic damped least-squares iteration. Convergence requires all three scaled residuals to be at most the requested tolerance. The recovered prolate bandwidth is converted to the force-error tolerance with the source paper's operative bandwidth rule (not its asymptotic small-error form).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def solve_inverse_precision(fractional_positions: "np.ndarray", charges: "np.ndarray", box_length: float, mode_count: int, quadrature_order: int, reference_frequency: float, target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", initial_parameters: "np.ndarray", tolerance: float, max_iterations: int) -> float:
    """Return the force-error tolerance epsilon for the recovered c after solving the inverse problem.

    Parameters
    ----------
    fractional_positions : np.ndarray
        Finite fractional coordinates with shape (N,3).
    charges : np.ndarray
        Finite neutral charge vector of length N.
    box_length : float
        Positive finite cubic scale L.
    mode_count : int
        Even centered reciprocal mode count at least 4.
    quadrature_order : int
        Integer Gauss-Legendre order at least 8.
    reference_frequency : float
        Positive finite reference wavenumber kappa_ref of the filter-factor
        diagnostic |xi|^2 S_hat(xi) at |xi| = kappa_ref.
    target_values : np.ndarray
        Target vector [T_ref,E_s,F_0x] of length 3.
    target_scales : np.ndarray
        Positive residual scales of length 3.
    bounds : np.ndarray
        Ordered physical bounds of shape (3,2) for [c,cutoff,shear].
    initial_parameters : np.ndarray
        Strictly interior physical initial guess of length 3.
    tolerance : float
        Positive convergence tolerance for max(abs(scaled residual)).
    max_iterations : int
        Positive integer iteration limit.

    Returns
    -------
    epsilon : float
        Force-error tolerance corresponding to the converged recovered bandwidth c
        under the source paper's operative bandwidth rule. Wherever the PSWF
        itself is evaluated, it is normalized so that
        int_{-1}^{1} psi_0^c(x)^2 dx = 1 and psi_0^c(0) > 0.

    Raises
    ------
    ValueError
        If solver inputs or the initial point violate the stated domain.
    RuntimeError
        If the scaled residual does not converge within max_iterations.
    """
    return epsilon

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_inverse_precision(fractional_positions: "np.ndarray", charges: "np.ndarray", box_length: float, mode_count: int, quadrature_order: int, reference_frequency: float, target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", initial_parameters: "np.ndarray", tolerance: float, max_iterations: int) -> float:
    bounds = np.asarray(bounds, dtype=float)
    initial_parameters = np.asarray(initial_parameters, dtype=float)
    tolerance = float(tolerance)
    if bounds.shape != (3, 2) or initial_parameters.shape != (3,):
        raise ValueError("bounds must be (3,2) and initial_parameters length 3")
    if not np.all(np.isfinite(bounds)) or not np.all(np.isfinite(initial_parameters)):
        raise ValueError("bounds and initial parameters must be finite")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("bounds must be strictly ordered")
    if np.any(initial_parameters <= bounds[:, 0]) or np.any(initial_parameters >= bounds[:, 1]):
        raise ValueError("initial_parameters must lie strictly inside bounds")
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance must be positive and finite")
    if isinstance(max_iterations, bool) or int(max_iterations) != max_iterations or int(max_iterations) < 1:
        raise ValueError("max_iterations must be a positive integer")
    max_iterations = int(max_iterations)

    indices = _oracle_centered_reciprocal_indices(mode_count)
    fraction = (initial_parameters - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    y = np.log(fraction / (1.0 - fraction))
    damping = 1.0e-2
    trust_radius = 1.0

    for _ in range(max_iterations):
        state = _oracle_scaled_inverse_residual(
            y, target_values, target_scales, bounds, fractional_positions, charges,
            indices, box_length, quadrature_order, reference_frequency
        )
        if float(np.max(np.abs(state[:3]))) <= tolerance:
            break

        iteration = _oracle_damped_lm_iteration(
            y, damping, trust_radius, target_values, target_scales, bounds,
            fractional_positions, charges, indices, box_length, quadrature_order,
            reference_frequency
        )
        y = iteration[:3]
        damping = float(iteration[3])
        trust_radius = float(iteration[4])

    state = _oracle_scaled_inverse_residual(
        y, target_values, target_scales, bounds, fractional_positions, charges,
        indices, box_length, quadrature_order, reference_frequency
    )
    if float(np.max(np.abs(state[:3]))) > tolerance:
        raise RuntimeError("inverse solve did not converge within max_iterations")
    return _prolate_edge_value(float(state[12]), quadrature_order)


def _prolate_edge_value(c: float, quadrature_order: int) -> float:
    packed = _oracle_pswf_bandwidth_sensitivity(c, quadrature_order)
    n = int(quadrature_order)
    nodes = packed[:n]
    weights = packed[n:2*n]
    chi = packed[2*n:3*n]
    eigenvalue = float(packed[-2])
    psi = chi / np.sqrt(float(np.dot(weights, chi * chi)))
    return float(np.dot(weights * np.cos(c * nodes), psi) / eigenvalue)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return three inverse problems with widely separated initial guesses."""
    return [
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.23,-0.08,0.05],[0.16,-0.18,-0.06],[0.25,0.14,0.11],[-0.07,0.23,-0.15]],float)\ncharges=np.array([1.0,-1.0,0.75,-0.75])\nbox_length=4.8\nmode_count=10\nquadrature_order=32\nreference_frequency=4.0\ntarget_values=np.array([0.49415969767112566,0.24195924758540227,0.01899987567836107])\ntarget_scales=np.array([0.05,0.20,0.02])\nbounds=np.array([[3.0,11.0],[0.45,1.15],[-0.35,0.35]])\ninitial_parameters=np.array([5.0,0.60,-0.10])\ntolerance=1e-12\nmax_iterations=60","call":"solve_inverse_precision(fractional_positions.copy(), charges.copy(), box_length, mode_count, quadrature_order, reference_frequency, target_values.copy(), target_scales.copy(), bounds.copy(), initial_parameters.copy(), tolerance, max_iterations)","gold_call":"_oracle_solve_inverse_precision(fractional_positions.copy(), charges.copy(), box_length, mode_count, quadrature_order, reference_frequency, target_values.copy(), target_scales.copy(), bounds.copy(), initial_parameters.copy(), tolerance, max_iterations)","tol":1e-10},
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.31,0.12,0.07],[0.18,-0.22,0.15],[0.09,0.27,-0.19],[0.04,-0.05,-0.03]],float)\ncharges=np.array([1.2,-0.7,-0.3,-0.2])\nbox_length=4.4\nmode_count=8\nquadrature_order=24\nreference_frequency=3.5\ntarget_values=np.array([0.6266262701264445,0.17388303201335253,0.00383731584980652])\ntarget_scales=np.array([0.06,0.18,0.025])\nbounds=np.array([[2.8,10.5],[0.40,1.10],[-0.40,0.40]])\ninitial_parameters=np.array([9.2,0.98,0.25])\ntolerance=1e-12\nmax_iterations=60","call":"solve_inverse_precision(fractional_positions.copy(), charges.copy(), box_length, mode_count, quadrature_order, reference_frequency, target_values.copy(), target_scales.copy(), bounds.copy(), initial_parameters.copy(), tolerance, max_iterations)","gold_call":"_oracle_solve_inverse_precision(fractional_positions.copy(), charges.copy(), box_length, mode_count, quadrature_order, reference_frequency, target_values.copy(), target_scales.copy(), bounds.copy(), initial_parameters.copy(), tolerance, max_iterations)","tol":1e-10},
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.20,-0.15,0.10],[0.22,0.18,-0.12],[-0.08,0.26,0.21],[0.11,-0.29,-0.16],[-0.05,0.00,-0.03]],float)\ncharges=np.array([0.8,-1.1,0.6,-0.2,-0.1])\nbox_length=5.2\nmode_count=10\nquadrature_order=28\nreference_frequency=4.5\ntarget_values=np.array([0.3403140251356788,0.17346926227771398,0.00294797222884924])\ntarget_scales=np.array([0.05,0.22,0.018])\nbounds=np.array([[3.2,12.0],[0.50,1.30],[-0.30,0.30]])\ninitial_parameters=np.array([4.0,0.58,-0.22])\ntolerance=1e-12\nmax_iterations=150","call":"solve_inverse_precision(fractional_positions.copy(), charges.copy(), box_length, mode_count, quadrature_order, reference_frequency, target_values.copy(), target_scales.copy(), bounds.copy(), initial_parameters.copy(), tolerance, max_iterations)","gold_call":"_oracle_solve_inverse_precision(fractional_positions.copy(), charges.copy(), box_length, mode_count, quadrature_order, reference_frequency, target_values.copy(), target_scales.copy(), bounds.copy(), initial_parameters.copy(), tolerance, max_iterations)","tol":1e-10},
    ]

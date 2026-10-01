"""
Perform one deterministic damped least-squares iteration in bounded logistic coordinates.

Compute an SVD Levenberg-Marquardt step from the analytic residual Jacobian, enforce a Euclidean trust radius, and backtrack by powers of one half until the scaled least-squares objective decreases. Update damping and trust radius from the ratio of actual to linearized predicted reduction.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def damped_lm_iteration(y: "np.ndarray", damping: float, trust_radius: float, target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray", box_length: float, quadrature_order: int, reference_frequency: float) -> "np.ndarray":
    """Return one deterministic bounded Levenberg-Marquardt iteration.

    Parameters
    ----------
    y : np.ndarray
        Finite unconstrained parameter vector of length 3.
    damping : float
        Positive finite Levenberg-Marquardt damping mu.
    trust_radius : float
        Positive finite Euclidean trust radius in y-space.
    target_values, target_scales, bounds, fractional_positions, charges, indices,
    box_length, quadrature_order, reference_frequency :
        Inputs passed unchanged to scaled_inverse_residual.

    Returns
    -------
    packed : np.ndarray
        Real vector of length 8: updated y[0:3], updated damping, updated trust
        radius, accepted flag (1 or 0), old objective, and new objective,
        where objective = 0.5*sum(r_i**2) over the scaled residual r.
        The SVD step is -V diag(s/(s^2+mu)) U^T r, scaled to the trust radius.
        Try alpha=1,1/2,... for at most 12 trials and accept the first strict
        objective decrease. If accepted, use rho=actual/predicted reduction:
        rho>0.75 halves damping and doubles trust up to 4; rho<0.25 multiplies
        damping by 4 and halves trust down to 1e-6. If no trial is accepted,
        keep y, multiply damping by 10, and halve trust down to 1e-8.

    Raises
    ------
    ValueError
        If y, damping, trust_radius, or downstream residual inputs are invalid.
    """
    return packed

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_damped_lm_iteration(y: "np.ndarray", damping: float, trust_radius: float, target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray", box_length: float, quadrature_order: int, reference_frequency: float) -> "np.ndarray":
    y = np.asarray(y, dtype=float)
    damping = float(damping)
    trust_radius = float(trust_radius)
    if y.shape != (3,) or not np.all(np.isfinite(y)):
        raise ValueError("y must be a finite vector of length 3")
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be positive and finite")
    if not np.isfinite(trust_radius) or trust_radius <= 0.0:
        raise ValueError("trust_radius must be positive and finite")

    current = _oracle_scaled_inverse_residual(
        y, target_values, target_scales, bounds, fractional_positions, charges,
        indices, box_length, quadrature_order, reference_frequency
    )
    residual = current[:3]
    jacobian = current[3:12].reshape(3, 3)
    objective = 0.5 * float(residual @ residual)

    left, singular_values, right_t = np.linalg.svd(jacobian, full_matrices=False)
    delta = -right_t.T @ (
        (singular_values / (singular_values ** 2 + damping))
        * (left.T @ residual)
    )
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > trust_radius:
        delta = delta * (trust_radius / delta_norm)

    new_y = y.copy()
    new_damping = damping
    new_trust = trust_radius
    new_objective = objective
    accepted = 0.0
    alpha = 1.0

    for _ in range(12):
        step = alpha * delta
        trial = _oracle_scaled_inverse_residual(
            y + step, target_values, target_scales, bounds, fractional_positions,
            charges, indices, box_length, quadrature_order, reference_frequency
        )
        trial_residual = trial[:3]
        trial_objective = 0.5 * float(trial_residual @ trial_residual)
        if np.isfinite(trial_objective) and trial_objective < objective:
            predicted = objective - 0.5 * float(
                (residual + jacobian @ step) @ (residual + jacobian @ step)
            )
            actual = objective - trial_objective
            rho = actual / predicted if predicted > 0.0 else -np.inf
            new_y = y + step
            new_objective = trial_objective
            accepted = 1.0
            if rho > 0.75:
                new_trust = min(2.0 * trust_radius, 4.0)
                new_damping = max(0.5 * damping, 1.0e-12)
            elif rho < 0.25:
                new_trust = max(0.5 * trust_radius, 1.0e-6)
                new_damping = min(4.0 * damping, 1.0e12)
            break
        alpha *= 0.5

    if accepted == 0.0:
        new_trust = max(0.5 * trust_radius, 1.0e-8)
        new_damping = min(10.0 * damping, 1.0e12)

    return np.concatenate([
        new_y,
        np.array([new_damping, new_trust, accepted, objective, new_objective], dtype=float),
    ])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return trust-clipped, unclipped, and strongly damped inverse-iteration cases."""
    return [
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.23,-0.08,0.05],[0.16,-0.18,-0.06],[0.25,0.14,0.11],[-0.07,0.23,-0.15]],float)\ncharges=np.array([1.0,-1.0,0.75,-0.75])\naxis=np.arange(-5,5,dtype=int)\nindices=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)\nindices=indices[np.any(indices!=0,axis=1)]\nbox_length=4.8\nquadrature_order=32\nreference_frequency=4.0\ntarget_values=np.array([0.49415969767112566,0.24195924758540227,0.01899987567836107])\ntarget_scales=np.array([0.05,0.20,0.02])\nbounds=np.array([[3.0,11.0],[0.45,1.15],[-0.35,0.35]])\ninitial=np.array([5.0,0.60,-0.10])\nt=(initial-bounds[:,0])/(bounds[:,1]-bounds[:,0])\ny=np.log(t/(1.0-t))\ndamping=1e-2\ntrust_radius=1.0","call":"damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","gold_call":"_oracle_damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","tol":1e-8},
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.31,0.12,0.07],[0.18,-0.22,0.15],[0.09,0.27,-0.19],[0.04,-0.05,-0.03]],float)\ncharges=np.array([1.2,-0.7,-0.3,-0.2])\naxis=np.arange(-4,4,dtype=int)\nindices=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)\nindices=indices[np.any(indices!=0,axis=1)]\nbox_length=4.4\nquadrature_order=24\nreference_frequency=3.5\ntarget_values=np.array([0.6266262701264445,0.17388303201335253,0.00383731584980652])\ntarget_scales=np.array([0.06,0.18,0.025])\nbounds=np.array([[2.8,10.5],[0.40,1.10],[-0.40,0.40]])\ny=np.array([1.5,1.0,-1.2])\ndamping=5e-1\ntrust_radius=0.35","call":"damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","gold_call":"_oracle_damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","tol":1e-8},
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.20,-0.15,0.10],[0.22,0.18,-0.12],[-0.08,0.26,0.21],[0.11,-0.29,-0.16],[-0.05,0.00,-0.03]],float)\ncharges=np.array([0.8,-1.1,0.6,-0.2,-0.1])\naxis=np.arange(-5,5,dtype=int)\nindices=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)\nindices=indices[np.any(indices!=0,axis=1)]\nbox_length=5.2\nquadrature_order=28\nreference_frequency=4.5\ntarget_values=np.array([0.3403140251356788,0.17346926227771398,0.00294797222884924])\ntarget_scales=np.array([0.05,0.22,0.018])\nbounds=np.array([[3.2,12.0],[0.50,1.30],[-0.30,0.30]])\ny=np.array([-2.0,1.8,1.4])\ndamping=3.0\ntrust_radius=0.12","call":"damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","gold_call":"_oracle_damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","tol":1e-8},
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.23,-0.08,0.05],[0.16,-0.18,-0.06],[0.25,0.14,0.11],[-0.07,0.23,-0.15]],float)\ncharges=np.array([1.0,-1.0,0.75,-0.75])\naxis=np.arange(-5,5,dtype=int)\nindices=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)\nindices=indices[np.any(indices!=0,axis=1)]\nbox_length=4.8\nquadrature_order=32\nreference_frequency=4.0\ntarget_values=np.array([0.49415969767112566,0.24195924758540227,0.01899987567836107])\ntarget_scales=np.array([0.05,0.20,0.02])\nbounds=np.array([[3.0,11.0],[0.45,1.15],[-0.35,0.35]])\ninitial=np.array([7.09,0.81,0.13])\nt=(initial-bounds[:,0])/(bounds[:,1]-bounds[:,0])\ny=np.log(t/(1.0-t))\ndamping=2.5e-3\ntrust_radius=4.0","call":"damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","gold_call":"_oracle_damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","tol":1e-8},
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.23,-0.08,0.05],[0.16,-0.18,-0.06],[0.25,0.14,0.11],[-0.07,0.23,-0.15]],float)\ncharges=np.array([1.0,-1.0,0.75,-0.75])\naxis=np.arange(-5,5,dtype=int)\nindices=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)\nindices=indices[np.any(indices!=0,axis=1)]\nbox_length=4.8\nquadrature_order=32\nreference_frequency=4.0\ntarget_values=np.array([0.49415969767112566,0.24195924758540227,0.01899987567836107])\ntarget_scales=np.array([0.05,0.20,0.02])\nbounds=np.array([[3.0,11.0],[0.45,1.15],[-0.35,0.35]])\ny=np.array([-1.17760723739,-0.639749731906,1.0380816831])\ndamping=0.00125\ntrust_radius=4.0","call":"damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","gold_call":"_oracle_damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","tol":1e-8},
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.31,0.12,0.07],[0.18,-0.22,0.15],[0.09,0.27,-0.19],[0.04,-0.05,-0.03]],float)\ncharges=np.array([1.2,-0.7,-0.3,-0.2])\naxis=np.arange(-4,4,dtype=int)\nindices=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)\nindices=indices[np.any(indices!=0,axis=1)]\nbox_length=4.4\nquadrature_order=24\nreference_frequency=3.5\ntarget_values=np.array([0.6266262701264445,0.17388303201335253,0.00383731584980652])\ntarget_scales=np.array([0.06,0.18,0.025])\nbounds=np.array([[2.8,10.5],[0.40,1.10],[-0.40,0.40]])\ny=np.array([1.9000678905,0.762326358706,-1.15265417406])\ndamping=7.8125e-05\ntrust_radius=4.0","call":"damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","gold_call":"_oracle_damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","tol":1e-8},
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.31,0.12,0.07],[0.18,-0.22,0.15],[0.09,0.27,-0.19],[0.04,-0.05,-0.03]],float)\ncharges=np.array([1.2,-0.7,-0.3,-0.2])\naxis=np.arange(-4,4,dtype=int)\nindices=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)\nindices=indices[np.any(indices!=0,axis=1)]\nbox_length=4.4\nquadrature_order=24\nreference_frequency=3.5\ntarget_values=np.array([0.6266262701264445,0.17388303201335253,0.00383731584980652])\ntarget_scales=np.array([0.06,0.18,0.025])\nbounds=np.array([[2.8,10.5],[0.40,1.10],[-0.40,0.40]])\ny=np.array([0.424089831631,0.198587579113,-1.15966304721])\ndamping=7.8125e-05\ntrust_radius=4.0","call":"damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","gold_call":"_oracle_damped_lm_iteration(y.copy(), damping, trust_radius, target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","tol":1e-8},
    ]

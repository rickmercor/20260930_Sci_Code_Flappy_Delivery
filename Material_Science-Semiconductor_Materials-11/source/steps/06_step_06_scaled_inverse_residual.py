"""
Assemble the bounded inverse residual and its full analytic Jacobian.

Three unconstrained logistic coordinates are mapped to c, cutoff, and shear inside supplied bounds. The residual combines the spectral filter factor at a reference wavenumber, the reciprocal spectral energy, and the x force on particle 0. Analytic derivatives propagate PSWF sensitivity, cutoff sensitivity, and reciprocal-geometry shear sensitivity through the entire spectral sum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def scaled_inverse_residual(y: "np.ndarray", target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray", box_length: float, quadrature_order: int, reference_frequency: float) -> "np.ndarray":
    """Return scaled residuals, the Jacobian in logistic coordinates, and physical parameters.

    Parameters
    ----------
    y : np.ndarray
        Finite unconstrained vector of length 3.
    target_values : np.ndarray
        Finite target vector [T_ref, E_s, F_0x] of length 3.
    target_scales : np.ndarray
        Positive finite residual scales of length 3.
    bounds : np.ndarray
        Finite array of shape (3,2), one strict lower/upper bound pair for
        [c, cutoff, shear].
    fractional_positions : np.ndarray
        Finite fractional particle coordinates with shape (N,3).
    charges : np.ndarray
        Finite neutral charge vector of length N, summing to zero within 1e-12.
    indices : np.ndarray
        Nonempty integer reciprocal index array of shape (M,3), excluding zero.
    box_length : float
        Positive finite cubic scale L.
    quadrature_order : int
        Integer Gauss-Legendre order at least 8.
    reference_frequency : float
        Positive finite reference wavenumber kappa_ref. The diagnostic
        observable T_ref is the spectral filter factor |xi|^2 S_hat(xi) at
        |xi| = kappa_ref, where S_hat is the source's Fourier-space long-range kernel.

    Returns
    -------
    packed : np.ndarray
        One-dimensional array of length 15. Entries 0:3 are scaled residuals;
        entries 3:12 are the 3x3 Jacobian dR/dy in row-major order; entries
        12:15 are the physical parameters [c, cutoff, shear].

    Raises
    ------
    ValueError
        If dimensions, finiteness, neutrality, scales, bounds, or scalar domains are invalid.
    """
    return packed

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _inverse_sigmoid_coordinates(y: "np.ndarray", bounds: "np.ndarray") -> tuple:
    sigmoid = np.empty_like(y, dtype=float)
    positive = y >= 0.0
    sigmoid[positive] = 1.0 / (1.0 + np.exp(-y[positive]))
    exp_y = np.exp(y[~positive])
    sigmoid[~positive] = exp_y / (1.0 + exp_y)
    span = bounds[:, 1] - bounds[:, 0]
    parameters = bounds[:, 0] + span * sigmoid
    derivative = span * sigmoid * (1.0 - sigmoid)
    return parameters, derivative


def _oracle_scaled_inverse_residual(y: "np.ndarray", target_values: "np.ndarray", target_scales: "np.ndarray", bounds: "np.ndarray", fractional_positions: "np.ndarray", charges: "np.ndarray", indices: "np.ndarray", box_length: float, quadrature_order: int, reference_frequency: float) -> "np.ndarray":
    y = np.asarray(y, dtype=float)
    target_values = np.asarray(target_values, dtype=float)
    target_scales = np.asarray(target_scales, dtype=float)
    bounds = np.asarray(bounds, dtype=float)
    fractional_positions = np.asarray(fractional_positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    indices_array = np.asarray(indices)
    box_length = float(box_length)
    reference_frequency = float(reference_frequency)

    if y.shape != (3,) or target_values.shape != (3,) or target_scales.shape != (3,):
        raise ValueError("y, target_values, and target_scales must have length 3")
    if bounds.shape != (3, 2):
        raise ValueError("bounds must have shape (3,2)")
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(target_values)) or not np.all(np.isfinite(target_scales)) or not np.all(np.isfinite(bounds)):
        raise ValueError("solver vectors and bounds must be finite")
    if np.any(target_scales <= 0.0) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("target scales must be positive and bounds must be ordered")
    if fractional_positions.ndim != 2 or fractional_positions.shape[1] != 3 or fractional_positions.shape[0] == 0:
        raise ValueError("fractional_positions must have shape (N,3)")
    if charges.ndim != 1 or charges.size != fractional_positions.shape[0]:
        raise ValueError("charges must have length N")
    if not np.all(np.isfinite(fractional_positions)) or not np.all(np.isfinite(charges)):
        raise ValueError("particle data must be finite")
    if abs(float(np.sum(charges))) > 1.0e-12:
        raise ValueError("charges must be neutral within 1e-12")
    if indices_array.ndim != 2 or indices_array.shape[1] != 3 or indices_array.shape[0] == 0:
        raise ValueError("indices must have nonempty shape (M,3)")
    if not np.all(np.isfinite(indices_array)) or not np.all(indices_array == np.rint(indices_array)):
        raise ValueError("indices must be finite and integer-valued")
    integer_indices = indices_array.astype(int)
    if np.any(np.all(integer_indices == 0, axis=1)):
        raise ValueError("indices must exclude zero")
    if not np.isfinite(box_length) or box_length <= 0.0:
        raise ValueError("box_length must be positive and finite")
    if not np.isfinite(reference_frequency) or reference_frequency <= 0.0:
        raise ValueError("reference_frequency must be positive and finite")

    parameters, dp_dy = _inverse_sigmoid_coordinates(y, bounds)
    c, cutoff, shear = [float(value) for value in parameters]
    if not (0.0 < cutoff < box_length / 2.0):
        raise ValueError("transformed cutoff must satisfy 0 < cutoff < box_length/2")
    if abs(shear) >= 0.5:
        raise ValueError("transformed shear must satisfy |shear| < 0.5")

    diagnostic = _oracle_prolate_transform_sensitivity(
        c, quadrature_order, np.array([cutoff * reference_frequency], dtype=float)
    )[0]

    geometry = _oracle_sheared_reciprocal_geometry(integer_indices, box_length, shear)
    xi = geometry[:, 0:3]
    xi_shear = geometry[:, 3:6]
    norms = geometry[:, 6]
    norms_shear = geometry[:, 7]

    transform = _oracle_prolate_transform_sensitivity(
        c, quadrature_order, cutoff * norms
    )
    chi_hat = transform[:, 0]
    chi_hat_c = transform[:, 1]
    chi_hat_s = transform[:, 2]

    structure = _oracle_fractional_structure_factors(
        fractional_positions, charges, integer_indices
    )
    structure_power = np.abs(structure) ** 2
    volume = box_length ** 3

    weights = chi_hat / (volume * norms ** 2)
    weights_c = chi_hat_c / (volume * norms ** 2)
    weights_cutoff = chi_hat_s / (volume * norms)
    weights_shear = (
        chi_hat_s * cutoff * norms_shear / (volume * norms ** 2)
        - 2.0 * chi_hat * norms_shear / (volume * norms ** 3)
    )

    energy = 0.5 * float(np.sum(weights * structure_power))
    energy_c = 0.5 * float(np.sum(weights_c * structure_power))
    energy_cutoff = 0.5 * float(np.sum(weights_cutoff * structure_power))
    energy_shear = 0.5 * float(np.sum(weights_shear * structure_power))

    phase = 2.0 * np.pi * (integer_indices.astype(float) @ fractional_positions.T)
    first_imag = np.imag(np.exp(1j * phase[:, 0]) * np.conj(structure))
    q0 = float(charges[0])
    force_x = q0 * float(np.sum(weights * xi[:, 0] * first_imag))
    force_x_c = q0 * float(np.sum(weights_c * xi[:, 0] * first_imag))
    force_x_cutoff = q0 * float(np.sum(weights_cutoff * xi[:, 0] * first_imag))
    force_x_shear = q0 * float(np.sum(
        (weights_shear * xi[:, 0] + weights * xi_shear[:, 0]) * first_imag
    ))

    observables = np.array([diagnostic[0], energy, force_x], dtype=float)
    physical_jacobian = np.array([
        [diagnostic[1], reference_frequency * diagnostic[2], 0.0],
        [energy_c, energy_cutoff, energy_shear],
        [force_x_c, force_x_cutoff, force_x_shear],
    ], dtype=float)

    residual = (observables - target_values) / target_scales
    jacobian_y = (
        physical_jacobian * dp_dy[None, :]
    ) / target_scales[:, None]

    return np.concatenate([residual, jacobian_y.ravel(), parameters])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential inverse-residual cases with different cells and targets."""
    return [
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.23,-0.08,0.05],[0.16,-0.18,-0.06],[0.25,0.14,0.11],[-0.07,0.23,-0.15]],float)\ncharges=np.array([1.0,-1.0,0.75,-0.75])\naxis=np.arange(-5,5,dtype=int)\nindices=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)\nindices=indices[np.any(indices!=0,axis=1)]\nbox_length=4.8\nquadrature_order=32\nreference_frequency=4.0\ntarget_values=np.array([0.49415969767112566,0.24195924758540227,0.01899987567836107])\ntarget_scales=np.array([0.05,0.20,0.02])\nbounds=np.array([[3.0,11.0],[0.45,1.15],[-0.35,0.35]])\ny=np.array([-0.8,-0.5,0.3])","call":"scaled_inverse_residual(y.copy(), target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","gold_call":"_oracle_scaled_inverse_residual(y.copy(), target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","tol":1e-8},
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.31,0.12,0.07],[0.18,-0.22,0.15],[0.09,0.27,-0.19],[0.04,-0.05,-0.03]],float)\ncharges=np.array([1.2,-0.7,-0.3,-0.2])\naxis=np.arange(-4,4,dtype=int)\nindices=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)\nindices=indices[np.any(indices!=0,axis=1)]\nbox_length=4.4\nquadrature_order=24\nreference_frequency=3.5\ntarget_values=np.array([0.6266262701264445,0.17388303201335253,0.00383731584980652])\ntarget_scales=np.array([0.06,0.18,0.025])\nbounds=np.array([[2.8,10.5],[0.40,1.10],[-0.40,0.40]])\ny=np.array([1.1,-0.9,-1.0])","call":"scaled_inverse_residual(y.copy(), target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","gold_call":"_oracle_scaled_inverse_residual(y.copy(), target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","tol":1e-8},
        {"setup":"import numpy as np\nfractional_positions=np.array([[-0.20,-0.15,0.10],[0.22,0.18,-0.12],[-0.08,0.26,0.21],[0.11,-0.29,-0.16],[-0.05,0.00,-0.03]],float)\ncharges=np.array([0.8,-1.1,0.6,-0.2,-0.1])\naxis=np.arange(-5,5,dtype=int)\nindices=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1).reshape(-1,3)\nindices=indices[np.any(indices!=0,axis=1)]\nbox_length=5.2\nquadrature_order=28\nreference_frequency=4.5\ntarget_values=np.array([0.3403140251356788,0.17346926227771398,0.00294797222884924])\ntarget_scales=np.array([0.05,0.22,0.018])\nbounds=np.array([[3.2,12.0],[0.50,1.30],[-0.30,0.30]])\ny=np.array([-1.4,0.8,1.2])","call":"scaled_inverse_residual(y.copy(), target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","gold_call":"_oracle_scaled_inverse_residual(y.copy(), target_values.copy(), target_scales.copy(), bounds.copy(), fractional_positions.copy(), charges.copy(), indices.copy(), box_length, quadrature_order, reference_frequency)","tol":1e-8},
    ]

"""
Apply the source's signed projection diagnostic. Project the basis effective vector onto the reference effective vector, normalize by the squared reference norm, subtract one, and partition directions by sign. Within each nonempty sign region, compute the quadrature-weighted RMS norm of the full directional error vector and divide by the common spherical mean magnitude of the reference effective-response vectors from source equation 29. If s[n] is the signed projection error, also return the checksum sum_n (n+1)s[n] and each sign region's quadrature weight divided by 4*pi; an empty sign region contributes zero RMS error.

This stage preserves a source-defined scientific quantity used by later parts of the directional convergence audit.

Returns
-------
tuple: Positive-region and negative-region normalized vector-error RMS values, signed checksum, and region weights.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_signed_projection_metrics(beta_basis: np.ndarray, beta_mra: np.ndarray, directions: np.ndarray, weights: np.ndarray) -> tuple:
    """Compute sign-partitioned directional-error metrics on the unit sphere.

    Parameters
    ----------
    beta_basis, beta_mra : np.ndarray
        Approximate and reference tensors, each with shape (3,3,3).
    directions : np.ndarray
        Unit directions with shape (n,3).
    weights : np.ndarray
        Positive quadrature weights of length n that sum to 4*pi.
    Returns
    -------
    tuple
        Positive-region and negative-region normalized vector-error RMS values,
        signed checksum, positive weight fraction, and negative weight fraction.
    Raises
    ------
    ValueError
        If quadrature or tensor contracts fail or a sampled reference response vanishes.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

def _oracle_compute_signed_projection_metrics(
    beta_basis: np.ndarray,
    beta_mra: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
) -> tuple:
    directions = np.asarray(directions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (directions.shape[0],) or np.any(~np.isfinite(weights)):
        raise ValueError("weights must match the direction count and be finite")
    if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 4.0 * np.pi, rtol=0.0, atol=1e-10):
        raise ValueError("weights must be positive and sum to 4*pi")
    basis_eff = _oracle_compute_effective_hyperpolarizability(beta_basis, directions)
    mra_eff = _oracle_compute_effective_hyperpolarizability(beta_mra, directions)
    denom = np.sum(mra_eff * mra_eff, axis=1)
    if np.any(denom <= 1e-14):
        raise ValueError("every sampled MRA effective response must be nonzero")
    signed = np.sum(basis_eff * mra_eff, axis=1) / denom - 1.0
    error_norm_sq = np.sum((basis_eff - mra_eff) ** 2, axis=1)
    mra_scale = float(np.dot(weights, np.linalg.norm(mra_eff, axis=1)) / (4.0 * np.pi))
    if mra_scale <= 1e-14:
        raise ValueError("the MRA response norm must be nonzero")
    positive = signed > 0.0
    negative = signed < 0.0
    positive_weight = float(np.sum(weights[positive]))
    negative_weight = float(np.sum(weights[negative]))
    positive_rms = (
        math.sqrt(float(np.sum(weights[positive] * error_norm_sq[positive]) / positive_weight)) / mra_scale
        if positive_weight > 0.0
        else 0.0
    )
    negative_rms = (
        math.sqrt(float(np.sum(weights[negative] * error_norm_sq[negative]) / negative_weight)) / mra_scale
        if negative_weight > 0.0
        else 0.0
    )
    checksum = float(np.dot(np.arange(1, signed.size + 1, dtype=float), signed))
    return (
        float(positive_rms),
        float(negative_rms),
        checksum,
        float(positive_weight / (4.0 * np.pi)),
        float(negative_weight / (4.0 * np.pi)),
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common='import numpy as np\ndef _fixture(scale=0.08, n=12):\n    j=np.arange(n,dtype=float); z=1-2*(j+.5)/n; r=np.sqrt(1-z*z); a=2*np.pi*j/((1+np.sqrt(5))/2)\n    u=np.column_stack((r*np.cos(a),r*np.sin(a),z)); u/=np.linalg.norm(u,axis=1,keepdims=True)\n    w=np.full(n,4*np.pi/n)\n    ref=np.arange(27,dtype=float).reshape(3,3,3)/31-.35; ref=.5*(ref+ref.swapaxes(1,2)); ref[0,0,0]+=1.2; ref[1,1,1]-=.8; ref[2,2,2]+=.55\n    mode=np.sin(np.arange(27,dtype=float)+.3).reshape(3,3,3); mode=.5*(mode+mode.swapaxes(1,2)); mode/=np.linalg.norm(mode)\n    basis=(1+.035)*ref+scale*mode\n    return basis,ref,u,w\ndef _flat(value):\n    pieces=[]\n    for item in value if isinstance(value,tuple) else (value,): pieces.extend(np.asarray(item,dtype=float).ravel().tolist())\n    return np.asarray(pieces,dtype=float)\ndef _value_error(function,*args):\n    try: function(*args)\n    except ValueError: return 1.0\n    return 0.0\n'
    return [
        {"setup":common+chr(10)+"b,r,u,w=_fixture(.08,12)","call":"_flat(compute_signed_projection_metrics(b,r,u,w))","gold_call":"_flat(_oracle_compute_signed_projection_metrics(b,r,u,w))","tol":1e-10},
        {"setup":common+chr(10)+"b,r,u,w=_fixture(1e-5,24)","call":"_flat(compute_signed_projection_metrics(b,r,u,w))","gold_call":"_flat(_oracle_compute_signed_projection_metrics(b,r,u,w))","tol":1e-10},
        {"setup":common+chr(10)+"b,r,u,w=_fixture(.31,7)","call":"_flat(compute_signed_projection_metrics(b,r,u,w))","gold_call":"_flat(_oracle_compute_signed_projection_metrics(b,r,u,w))","tol":1e-10},
        {"setup":common+chr(10)+"b,r,u,w=_fixture();w=w*.9","call":"_value_error(compute_signed_projection_metrics,b,r,u,w)","gold_call":"_value_error(_oracle_compute_signed_projection_metrics,b,r,u,w)","tol":1e-12}
    ]

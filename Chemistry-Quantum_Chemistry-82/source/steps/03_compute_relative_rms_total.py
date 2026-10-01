"""
Use the source's relative RMS total error. Form the effective basis and reference vector fields, take the 4*pi-normalized spherical RMS norm of their difference, and divide by the spherical mean magnitude of the reference vector field from source equation 29. Report dimensionless fractions, without percentage scaling. If the row-major directional error array is e[n,k], return its checksum sum over n and k of (3n+k+1)e[n,k] together with the ratio and reference scale.

This stage preserves a source-defined scientific quantity used by later parts of the directional convergence audit.

Returns
-------
tuple: Relative RMS fraction, quadrature-weighted spherical mean magnitude of the MRA effective-response vectors, and row-major directional-error checksum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_relative_rms_total(beta_basis: np.ndarray, beta_mra: np.ndarray, directions: np.ndarray, weights: np.ndarray) -> tuple:
    """Compute the source-normalized spherical RMS total error.

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
        Relative RMS error, quadrature-weighted spherical mean magnitude of the MRA effective-response vectors, and directional-error checksum.
    Raises
    ------
    ValueError
        If quadrature or tensor contracts fail or the reference norm vanishes.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

def _oracle_compute_relative_rms_total(
    beta_basis: np.ndarray,
    beta_mra: np.ndarray,
    directions: np.ndarray,
    weights: np.ndarray,
) -> tuple:
    directions = np.asarray(directions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (directions.shape[0],) or not np.all(np.isfinite(weights)):
        raise ValueError("weights must match the direction count and be finite")
    if np.any(weights <= 0.0) or not np.isclose(np.sum(weights), 4.0 * np.pi, rtol=0.0, atol=1e-10):
        raise ValueError("weights must be positive and sum to 4*pi")
    basis_eff = _oracle_compute_effective_hyperpolarizability(beta_basis, directions)
    mra_eff = _oracle_compute_effective_hyperpolarizability(beta_mra, directions)
    error = basis_eff - mra_eff
    mra_scale = float(np.dot(weights, np.linalg.norm(mra_eff, axis=1)) / (4.0 * np.pi))
    if mra_scale <= 1e-14:
        raise ValueError("the MRA response norm must be nonzero")
    rms_error = math.sqrt(float(np.sum(weights[:, None] * error**2) / (4.0 * np.pi)))
    coefficients = np.arange(1, error.size + 1, dtype=float).reshape(error.shape)
    checksum = float(np.sum(coefficients * error))
    return float(rms_error / mra_scale), float(mra_scale), checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common='import numpy as np\ndef _fixture(scale=0.08, n=12):\n    j=np.arange(n,dtype=float); z=1-2*(j+.5)/n; r=np.sqrt(1-z*z); a=2*np.pi*j/((1+np.sqrt(5))/2)\n    u=np.column_stack((r*np.cos(a),r*np.sin(a),z)); u/=np.linalg.norm(u,axis=1,keepdims=True)\n    w=np.full(n,4*np.pi/n)\n    ref=np.arange(27,dtype=float).reshape(3,3,3)/31-.35; ref=.5*(ref+ref.swapaxes(1,2)); ref[0,0,0]+=1.2; ref[1,1,1]-=.8; ref[2,2,2]+=.55\n    mode=np.sin(np.arange(27,dtype=float)+.3).reshape(3,3,3); mode=.5*(mode+mode.swapaxes(1,2)); mode/=np.linalg.norm(mode)\n    basis=(1+.035)*ref+scale*mode\n    return basis,ref,u,w\ndef _flat(value):\n    pieces=[]\n    for item in value if isinstance(value,tuple) else (value,): pieces.extend(np.asarray(item,dtype=float).ravel().tolist())\n    return np.asarray(pieces,dtype=float)\ndef _value_error(function,*args):\n    try: function(*args)\n    except ValueError: return 1.0\n    return 0.0\n'
    return [
        {"setup":common+chr(10)+"b,r,u,w=_fixture(.08,12)","call":"_flat(compute_relative_rms_total(b,r,u,w))","gold_call":"_flat(_oracle_compute_relative_rms_total(b,r,u,w))","tol":1e-10},
        {"setup":common+chr(10)+"b,r,u,w=_fixture(1e-5,24)","call":"_flat(compute_relative_rms_total(b,r,u,w))","gold_call":"_flat(_oracle_compute_relative_rms_total(b,r,u,w))","tol":1e-10},
        {"setup":common+chr(10)+"b,r,u,w=_fixture(.31,7)","call":"_flat(compute_relative_rms_total(b,r,u,w))","gold_call":"_flat(_oracle_compute_relative_rms_total(b,r,u,w))","tol":1e-10},
        {"setup":common+chr(10)+"b,r,u,w=_fixture();w=w*.9","call":"_value_error(compute_relative_rms_total,b,r,u,w)","gold_call":"_value_error(_oracle_compute_relative_rms_total,b,r,u,w)","tol":1e-12}
    ]

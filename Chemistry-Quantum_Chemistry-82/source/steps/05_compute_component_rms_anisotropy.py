"""
Use the source's component-wise RMS analysis. Compute one spherical RMS error c[k] for each Cartesian output component, normalize each by the common spherical mean MRA vector magnitude from source equation 29, then report the three-vector, its maximum-to-minimum ratio, and the checksum c[0]+2c[1]+3c[2]. Report dimensionless fractions, without percentage scaling.

This stage preserves a source-defined scientific quantity used by later parts of the directional convergence audit.

Returns
-------
tuple: Component RMS fractions, anisotropy ratio, and checksum.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_component_rms_anisotropy(beta_basis: np.ndarray, beta_mra: np.ndarray, directions: np.ndarray, weights: np.ndarray) -> tuple:
    """Compute Cartesian RMS errors and their anisotropy.

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
        Three normalized component RMS errors, max-to-min ratio, and component checksum.
    Raises
    ------
    ValueError
        If quadrature or tensor contracts fail or a required norm vanishes.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np

def _oracle_compute_component_rms_anisotropy(
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
    mra_scale = float(np.dot(weights, np.linalg.norm(mra_eff, axis=1)) / (4.0 * np.pi))
    if mra_scale <= 1e-14:
        raise ValueError("the MRA response norm must be nonzero")
    error = basis_eff - mra_eff
    component = np.sqrt(np.sum(weights[:, None] * error**2, axis=0) / (4.0 * np.pi)) / mra_scale
    if np.min(component) <= 1e-14:
        raise ValueError("all component errors must be nonzero for the anisotropy ratio")
    anisotropy = float(np.max(component) / np.min(component))
    checksum = float(np.dot(np.arange(1, 4, dtype=float), component))
    return component.astype(float), anisotropy, checksum

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common='import numpy as np\ndef _fixture(scale=0.08, n=12):\n    j=np.arange(n,dtype=float); z=1-2*(j+.5)/n; r=np.sqrt(1-z*z); a=2*np.pi*j/((1+np.sqrt(5))/2)\n    u=np.column_stack((r*np.cos(a),r*np.sin(a),z)); u/=np.linalg.norm(u,axis=1,keepdims=True)\n    w=np.full(n,4*np.pi/n)\n    ref=np.arange(27,dtype=float).reshape(3,3,3)/31-.35; ref=.5*(ref+ref.swapaxes(1,2)); ref[0,0,0]+=1.2; ref[1,1,1]-=.8; ref[2,2,2]+=.55\n    mode=np.sin(np.arange(27,dtype=float)+.3).reshape(3,3,3); mode=.5*(mode+mode.swapaxes(1,2)); mode/=np.linalg.norm(mode)\n    basis=(1+.035)*ref+scale*mode\n    return basis,ref,u,w\ndef _flat(value):\n    pieces=[]\n    for item in value if isinstance(value,tuple) else (value,): pieces.extend(np.asarray(item,dtype=float).ravel().tolist())\n    return np.asarray(pieces,dtype=float)\ndef _value_error(function,*args):\n    try: function(*args)\n    except ValueError: return 1.0\n    return 0.0\n'
    return [
        {"setup":common+chr(10)+"b,r,u,w=_fixture(.08,12)","call":"_flat(compute_component_rms_anisotropy(b,r,u,w))","gold_call":"_flat(_oracle_compute_component_rms_anisotropy(b,r,u,w))","tol":1e-10},
        {"setup":common+chr(10)+"b,r,u,w=_fixture(1e-5,24)","call":"_flat(compute_component_rms_anisotropy(b,r,u,w))","gold_call":"_flat(_oracle_compute_component_rms_anisotropy(b,r,u,w))","tol":1e-10},
        {"setup":common+chr(10)+"b,r,u,w=_fixture(.31,7)","call":"_flat(compute_component_rms_anisotropy(b,r,u,w))","gold_call":"_flat(_oracle_compute_component_rms_anisotropy(b,r,u,w))","tol":1e-10},
        {"setup":common+chr(10)+"b,r,u,w=_fixture();w=w*.9","call":"_value_error(compute_component_rms_anisotropy,b,r,u,w)","gold_call":"_value_error(_oracle_compute_component_rms_anisotropy,b,r,u,w)","tol":1e-12}
    ]

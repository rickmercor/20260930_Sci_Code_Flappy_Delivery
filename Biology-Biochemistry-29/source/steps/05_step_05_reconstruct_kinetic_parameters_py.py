"""
Recover the four unknown rates from normalized inverse invariants, measured Yp, and the fixed rate constants. Preserve the shared-denominator structure and reject singular or nonphysical reconstructions.

Write u=k3, w=k12, q=k8, D=d0+d1uw, and Yp=Hu/D. The measured Yp constrains w=(H/Yp-d0/u)/d1. The normalized invariants s=S/B, v=C/B, r=beta/B determine u, q, w, and k6 through the coefficient identities. The unknown entries of fixed_rates are placeholders only; their values must not be used as the unknown parameter values.

Returns
-------
np.ndarray of shape (4,), containing finite strictly positive rates in k3,k6,k8,k12 order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def reconstruct_kinetic_parameters(
    normalized: np.ndarray,
    Yp: float,
    fixed_rates: np.ndarray,
) -> np.ndarray:
    """Return [k3,k6,k8,k12] from normalized=[s,v,r].

    Yp is positive and fixed_rates is a length-14 vector.
    Positions 2,5,7,11 (zero-based) contain arbitrary positive
    placeholders; all other entries are fixed. Recover the unknown
    rates using the structural coefficient identities. This
    function does not enforce the calibration bounds, which are
    supplied to Step 6.

    Returns:
        np.ndarray: Four finite, strictly positive kinetic rates
            in k3,k6,k8,k12 order.

    Raises:
        ValueError: If inputs are invalid, an inverse denominator
            is singular, or the reconstructed rates are nonfinite
            or nonpositive.
    """
    return np.empty(4, dtype=float)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _checked_array(value, shape, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != shape or not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must have shape {shape} and be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a

def _oracle_reconstruct_kinetic_parameters(normalized, Yp, fixed_rates):
    import numpy as np

    s, v, r = _checked_array(normalized, (3,), "normalized")
    k = _checked_array(fixed_rates, (14,), "fixed_rates", True)
    yp = float(Yp)

    if (
        not np.isfinite(yp) or yp <= 0
        or s <= 1 or v <= 0 or r <= 0
    ):
        raise ValueError("invalid inverse parameters")

    d0 = k[8]*k[10]*(k[12]+k[13])
    d1 = (
        k[0]*k[13]*(k[9]+k[10])
        / (k[1]*(k[3]+k[4]))
    )
    H = (
        k[0]*k[4]*(k[9]+k[10])*(k[12]+k[13])
        / (k[1]*(k[3]+k[4]))
    )

    a = k[8]*(k[12]+k[13])*yp/H
    a0 = (k[3]+k[4])/k[4]*(1+k[1]/k[0])
    e = yp/(k[4]*(k[12]+k[13]))

    delta = 1-s+v
    if delta <= 0 or s-v <= 0:
        raise ValueError("invalid inverse invariants")

    J = k[4]*(1-(1+a0/a)*delta)
    if J <= 0:
        raise ValueError("invalid inverse invariants")

    u = a*J/delta
    w = (H/yp-d0/u)/d1

    den = s-1-e*J*w
    if den <= 0:
        raise ValueError("invalid inverse invariants")

    B = 1/den
    q = J*B
    k6 = (k[6]+q)/(r*B)

    return _checked_array(
        np.array([u, k6, q, w]),
        (4,),
        "reconstructed rates",
        True
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = """import numpy as np

def capture(fn, *args):
    try:
        fn(*args)
        return 0.0
    except ValueError:
        return 1.0
    except Exception:
        return 2.0

base = np.array([
    1.7, 0.83, 1.29, 0.47, 1.11, 1e13,
    0.62, 0.91, 1.37, 0.58, 1.43, 0.76,
    0.66, 1.21
], dtype=float)

coeffs = np.array([
    0.9489489489489491,
    0.4633103691927221,
    0.2752191290298749,
    0.7207207207207207,
    0.3358980541217181,
    0.5259259259259259,
    0.8715294886755629
], dtype=float)

alt_coeffs = np.array([
    1.1646191646191644,
    0.5686081803728862,
    0.3233095241643865,
    0.8108108108108107,
    0.36170857888010516,
    0.5066666666666667,
    0.834220552423816
], dtype=float)

def invariants(c):
    B = np.sum(c[[0,1,2,3]])
    S = 1 + np.sum(c[:5])
    C = 1 + c[2] + c[4]
    return np.array([S/B, C/B, c[5]/B])
"""

    return [
        {
            "description": "Reconstruct an independent synthetic kinetic-rate vector.",
            "setup": common + "\nnormalized = invariants(coeffs)\n",
            "call": "reconstruct_kinetic_parameters(normalized, coeffs[6], base)",
            "gold_call": "_oracle_reconstruct_kinetic_parameters(normalized, coeffs[6], base)",
            "tol": 1e-9
        },
        {
            "description": "Reconstruct an independent synthetic parameter vector.",
            "setup": common + "\nnormalized = invariants(alt_coeffs)\n",
            "call": "reconstruct_kinetic_parameters(normalized, alt_coeffs[6], base)",
            "gold_call": "_oracle_reconstruct_kinetic_parameters(normalized, alt_coeffs[6], base)",
            "tol": 1e-9
        },
        {
            "description": "Nonphysical normalized invariants are rejected.",
            "setup": common + "\nnormalized = np.array([1.0,0.4,0.2])\n",
            "call": "capture(reconstruct_kinetic_parameters, normalized, coeffs[6], base)",
            "gold_call": "capture(_oracle_reconstruct_kinetic_parameters, normalized, coeffs[6], base)",
            "tol": 1e-9
        }
    ]

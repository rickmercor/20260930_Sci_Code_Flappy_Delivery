"""
Form the site contour and differentiate its normalized second spatial moment about a fixed center.

A local negativity contour assigns a nonnegative contribution to each site. Normalizing its second moment separates the spatial distribution from the total amount of entanglement. Both numerator and normalization change when the state changes.

Returns
-------
float, the right derivative of the normalized second spatial moment as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def negativity_width_response(log_jet: "np.ndarray", positions: "np.ndarray", center: float) -> float:
    """Differentiate the second moment of the normalized site contour.

    Parameters
    ----------
    log_jet : np.ndarray
        Real finite symmetric array of shape (2, 2*m, 2*m), m >= 1, with symmetry absolute tolerance 1e-10. The two slices contain the active logarithm L and its right derivative dL, in all-q-then-all-p order. Site weights are e_i = (L_ii + L_(m+i,m+i))/2, and their derivatives use dL in the same way. The total weight must exceed 1e-14. A scientifically valid active-log jet is a precondition; positivity of each separate site weight is not checked.
    positions : np.ndarray
        Real finite one-dimensional array of m distinct integer-valued site positions, in the same order as the canonical coordinates. Boolean entries are invalid.
    center : float
        Finite real scalar specifying the fixed spatial origin; booleans are invalid. It does not vary with the differentiation parameter. Caller inputs are not mutated.

    Returns
    -------
    response : float
        Native Python float equal to the right derivative of sum((positions-center)**2 * e) / sum(e).

    Raises
    ------
    ValueError
        If inputs are complex or nonfinite, arrays have invalid shapes, log_jet is nonsymmetric, positions are not distinct integer-valued sites, center is not a finite real scalar, total weight <= 1e-14, or the spatial response cannot be represented as a finite float.
    """
    return response

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_negativity_width_response(log_jet: "np.ndarray", positions: "np.ndarray", center: float) -> float:
    import numpy as np
    from numbers import Real

    raw = np.asarray(log_jet)
    sites = np.asarray(positions)
    if (np.iscomplexobj(raw) or sites.dtype.kind not in "iuf"
            or any(isinstance(v, (bool, np.bool_)) for v in np.asarray(positions, dtype=object).flat)):
        raise ValueError("inputs must be real and sites nonboolean")
    try:
        z = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("arrays must be numeric") from exc
    if z.ndim != 3 or z.shape[0] != 2 or z.shape[1] != z.shape[2] or z.shape[1] < 2 or z.shape[1] % 2:
        raise ValueError("log_jet has an invalid shape")
    if not np.isfinite(z).all() or not np.allclose(z, z.transpose(0, 2, 1), atol=1e-10, rtol=0):
        raise ValueError("log_jet must be finite and symmetric")
    m = z.shape[1] // 2
    if sites.shape != (m,) or not np.isfinite(sites).all() or not np.all(sites == np.floor(sites)):
        raise ValueError("positions must be distinct finite integer sites")
    coordinates = [int(v) for v in sites]
    if len(set(coordinates)) != m:
        raise ValueError("positions must be distinct finite integer sites")
    if isinstance(center, (bool, np.bool_)) or not isinstance(center, Real) or not np.isfinite(center):
        raise ValueError("center must be a finite real scalar")
    diagonals = np.diagonal(z, axis1=1, axis2=2)
    e, de = (diagonals[:, :m] + diagonals[:, m:]) / 2
    total = float(e.sum())
    if total <= 1e-14:
        raise ValueError("total negativity must exceed 1e-14")
    origin = int(center) if center == int(center) else float(center)
    with np.errstate(over="ignore", invalid="ignore"):
        r2 = np.array([v - origin for v in coordinates], dtype=float) ** 2
    moment = float(r2 @ e)
    dmoment = float(r2 @ de)
    dtotal = float(de.sum())
    response = float((dmoment * total - moment * dtotal) / total**2)
    if not np.isfinite(response):
        raise ValueError("The spatial response must be representable as a finite float")
    return response

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nz=np.stack((np.diag([.1,.3,.2,.1,.3,.2]),np.diag([.02,-.01,.03,.02,-.01,.03]))); x=np.array([-1,0,2]); c=.5", "call": "negativity_width_response(z.copy(),x.copy(),c)", "gold_call": "_oracle_negativity_width_response(z.copy(),x.copy(),c)"},
        {"setup": "import numpy as np\nz=np.stack((np.diag([.2,.4,.2,.4]),np.diag([.02,.04,.02,.04]))); x=np.array([-2,1]); c=0.", "call": "negativity_width_response(z.copy(),x.copy(),c)", "gold_call": "_oracle_negativity_width_response(z.copy(),x.copy(),c)"},
        {"setup": "import numpy as np\nz=np.stack((np.array([[.3,.1],[.1,.3]]),np.array([[-.02,.04],[.04,-.02]]))); x=np.array([3]); c=-.5", "call": "negativity_width_response(z.copy(),x.copy(),c)", "gold_call": "_oracle_negativity_width_response(z.copy(),x.copy(),c)"},
        {"setup": "import numpy as np\nz=np.stack((np.diag([.1,.2,.4,.5]),np.diag([.03,-.01,.02,.04]))); x=np.array([0,3]); c=1.25", "call": "negativity_width_response(z.copy(),x.copy(),c)", "gold_call": "_oracle_negativity_width_response(z.copy(),x.copy(),c)"},
        {"setup": "import numpy as np\nz=np.stack((np.diag([.2,.4,.2,.4]),np.diag([.03,-.01,.03,-.01]))); x=np.array([1000000,1000007],dtype=np.int64); c=1000000", "call": "negativity_width_response(z.copy(),x.copy(),c)", "gold_call": "_oracle_negativity_width_response(z.copy(),x.copy(),c)"},
        {"setup": "import numpy as np\nz=np.stack((np.eye(4),np.eye(4))); x=[False,2]; c=0.\ndef rejected(fn):\n    try:\n        fn(z.copy(),x.copy(),c)\n    except ValueError:\n        return 1\n    return 0", "call": "rejected(negativity_width_response)", "gold_call": "rejected(_oracle_negativity_width_response)"},
        {"setup": "import numpy as np\nz=np.stack((np.zeros((2,2)),np.eye(2))); x=np.array([0]); c=0.\ndef rejected(fn):\n    try:\n        fn(z.copy(),x.copy(),c)\n    except ValueError:\n        return 1\n    return 0", "call": "rejected(negativity_width_response)", "gold_call": "rejected(_oracle_negativity_width_response)"},
        {"setup": "import numpy as np\nz=np.stack((np.eye(4),np.eye(4))); x=np.array([1,1]); c=0.\ndef rejected(fn):\n    try:\n        fn(z.copy(),x.copy(),c)\n    except ValueError:\n        return 1\n    return 0", "call": "rejected(negativity_width_response)", "gold_call": "rejected(_oracle_negativity_width_response)"}
    ]

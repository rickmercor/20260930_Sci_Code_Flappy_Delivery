"""
Evaluate the active logarithm and its right directional derivative, treating a degenerate threshold subspace jointly.

Only partially transposed symplectic modes below the vacuum threshold contribute to logarithmic negativity. At the threshold the scalar filter has a kink, and its matrix directional derivative retains a spectral positive part inside the degenerate threshold subspace.

Returns
-------
ndarray, shape (2, 2*m, 2*m), containing the active logarithm and its right directional derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def active_log_jet(contour_jet: "np.ndarray") -> "np.ndarray":
    """Apply the negativity filter and its right directional derivative.

    Parameters
    ----------
    contour_jet : np.ndarray
        Real finite array of shape (2, 2*m, 2*m), with m >= 1. The slices are a symmetric positive definite Phi and a symmetric direction H. Symmetry uses absolute tolerance 1e-10. The filter is f(x) = max(0, -log(2*x)), with the natural logarithm. Eigenvalues within absolute tolerance 1e-10 of 0.5 are treated as exactly 0.5 in both slices; this is the numerical threshold convention. Outside that band use ordinary spectral divided differences. Within the full threshold eigenspace use the positive semidefinite part of -2 times the restriction of H. Inputs are not mutated.

    Returns
    -------
    log_jet : np.ndarray
        Real array of shape (2, 2*m, 2*m). Slice 0 is f(Phi); slice 1 is its right directional derivative under the threshold convention.

    Raises
    ------
    ValueError
        If the input is complex, nonfinite, has a wrong shape or nonsymmetric slices, or Phi is not positive definite.
    """
    return log_jet

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_active_log_jet(contour_jet: "np.ndarray") -> "np.ndarray":
    import numpy as np

    raw = np.asarray(contour_jet)
    if np.iscomplexobj(raw):
        raise ValueError("contour_jet must be real")
    try:
        z = np.array(raw, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("contour_jet must be numeric") from exc
    if z.ndim != 3 or z.shape[0] != 2 or z.shape[1] != z.shape[2] or z.shape[1] < 2 or z.shape[1] % 2:
        raise ValueError("contour_jet has an invalid shape")
    if not np.isfinite(z).all() or not np.allclose(z, z.transpose(0, 2, 1), atol=1e-10, rtol=0):
        raise ValueError("contour_jet must be finite and symmetric")
    phi, h = (z + z.transpose(0, 2, 1)) / 2
    lam, u = np.linalg.eigh(phi)
    if np.min(lam) <= 0:
        raise ValueError("Phi must be positive definite")
    boundary = np.abs(lam - 0.5) <= 1e-10
    lam[boundary] = 0.5
    values = np.maximum(0.0, -np.log(2 * lam))
    hhat = u.T @ h @ u
    coeff = np.zeros((len(lam), len(lam)))
    for i, a in enumerate(lam):
        for j, b in enumerate(lam):
            if boundary[i] and boundary[j]:
                continue
            if a < 0.5 and b < 0.5:
                gap = a - b
                if abs(gap) <= 1e-12 * max(a, b):
                    coeff[i, j] = -2 / (a + b)
                else:
                    coeff[i, j] = -np.log1p(gap / b) / gap
            elif a != b:
                coeff[i, j] = (values[i] - values[j]) / (a - b)
    tangent = coeff * hhat
    ids = np.flatnonzero(boundary)
    if len(ids):
        block = -2 * hhat[np.ix_(ids, ids)]
        ev, v = np.linalg.eigh(block)
        tangent[np.ix_(ids, ids)] = (v * np.maximum(ev, 0)) @ v.T
    value = (u * values) @ u.T
    direction = u @ tangent @ u.T
    out = np.stack((value, direction))
    return (out + out.transpose(0, 2, 1)) / 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\np=np.diag([.3,.7,.4,.9]); h=np.array([[.02,.03,.01,0],[.03,-.01,.02,.04],[.01,.02,.05,-.01],[0,.04,-.01,.03]]); z=np.stack((p,h))", "call": "active_log_jet(z.copy())", "gold_call": "_oracle_active_log_jet(z.copy())"},
        {"setup": "import numpy as np\np=.5*np.eye(4); h=np.array([[.1,.2,0,0],[.2,-.1,0,0],[0,0,-.3,.05],[0,0,.05,.2]]); z=np.stack((p,h))", "call": "active_log_jet(z.copy())", "gold_call": "_oracle_active_log_jet(z.copy())"},
        {"setup": "import numpy as np\na=np.array([[1,2,3,4],[2,-1,4,-3],[3,-4,-1,2],[4,3,-2,-1]],dtype=float); u=np.linalg.qr(a)[0]; p=u@np.diag([.3,.5,.5,.8])@u.T; h=u@np.array([[.02,.03,.04,.01],[.03,.1,.2,.05],[.04,.2,-.1,.02],[.01,.05,.02,-.03]])@u.T; z=np.stack((p,h))", "call": "active_log_jet(z.copy())", "gold_call": "_oracle_active_log_jet(z.copy())"},
        {"setup": "import numpy as np\np=np.diag([.2,.2,.8,.8]); h=np.full((4,4),.03); z=np.stack((p,h))", "call": "active_log_jet(z.copy())", "gold_call": "_oracle_active_log_jet(z.copy())"},
        {"setup": "import numpy as np\np=np.diag([.5-5e-11,.5+5e-11]); h=np.array([[-.2,.1],[.1,.3]]); z=np.stack((p,h))", "call": "active_log_jet(z.copy())", "gold_call": "_oracle_active_log_jet(z.copy())"},
        {"setup": "import numpy as np\nz=np.stack((np.diag([.50001,2.0]),np.ones((2,2))))", "call": "active_log_jet(z.copy())", "gold_call": "_oracle_active_log_jet(z.copy())"},
        {"setup": "import numpy as np\nz=np.zeros((2,2,2))\ndef rejected(fn):\n    try:\n        fn(z.copy())\n    except ValueError:\n        return 1\n    return 0", "call": "rejected(active_log_jet)", "gold_call": "rejected(_oracle_active_log_jet)"},
        {"setup": "import numpy as np\nz=np.stack((np.eye(2),np.array([[0.,1.],[0.,0.]])))\ndef rejected(fn):\n    try:\n        fn(z.copy())\n    except ValueError:\n        return 1\n    return 0", "call": "rejected(active_log_jet)", "gold_call": "rejected(_oracle_active_log_jet)"}
    ]

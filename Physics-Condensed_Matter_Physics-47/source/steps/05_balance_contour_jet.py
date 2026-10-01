"""
Remove the Williamson polar metric and propagate its variation to obtain the symmetric matrix defining the spatial contour.

The right polar metric of a Williamson transformation is fixed by the covariance although the transformation itself has mode-basis freedom. Removing this metric gives the symmetric matrix whose spectral functions define the passive-polar spatial contour. Its variation includes the variation of the metric as well as that of the covariance.

Returns
-------
ndarray, shape (2, 2*m, 2*m), the passive-polar contour matrix and its directional derivative.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def balance_contour_jet(covariance_jet: "np.ndarray", absolute_jet: "np.ndarray") -> "np.ndarray":
    """Construct the passive-polar contour matrix and its first variation.

    Parameters
    ----------
    covariance_jet : np.ndarray
        Real symmetric array (2, 2*m, 2*m), containing positive definite G and its direction dG in grouped position/momentum coordinates.
    absolute_jet : np.ndarray
        Real symmetric array (4, 2*m, 2*m), containing C, dC, B, dB from symplectic_absolute_jet for this same covariance and direction. C and B must be positive definite. The caller supplies a consistent pair; consistency with the defining identities is a precondition.

    Returns
    -------
    contour_jet : np.ndarray
        Real symmetric array (2, 2*m, 2*m), containing Phi and dPhi. Phi = K.T (D direct-sum D) K, where G = W.T (D direct-sum D) W is any Williamson decomposition and K is W's orthogonal polar factor. The output must be independent of the mode basis, including repeated symplectic eigenvalues. dPhi is the exact first variation. Caller inputs are not mutated.

    Raises
    ------
    ValueError
        If inputs are not finite real symmetric arrays with the stated matching shapes, or G, C or B is not positive definite. Symmetry uses absolute tolerance 1e-10 and zero relative tolerance.
    """
    return contour_jet

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_balance_contour_jet(covariance_jet: "np.ndarray", absolute_jet: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from scipy.linalg import solve_sylvester

    arrays = []
    for raw, leading in ((covariance_jet, 2), (absolute_jet, 4)):
        raw = np.asarray(raw)
        if np.iscomplexobj(raw):
            raise ValueError("Both jets must be real.")
        try:
            a = np.asarray(raw, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Both jets must be numerical.") from exc
        if (a.ndim != 3 or a.shape[0] != leading or a.shape[1] != a.shape[2]
                or a.shape[1] < 2 or a.shape[1] % 2 or not np.isfinite(a).all()):
            raise ValueError("Invalid jet shape or entries.")
        if not np.allclose(a, a.transpose(0, 2, 1), atol=1e-10, rtol=0):
            raise ValueError("Jet matrices must be symmetric.")
        arrays.append((a + a.transpose(0, 2, 1)) / 2)
    if arrays[0].shape[1:] != arrays[1].shape[1:]:
        raise ValueError("The two jets must have matching matrix dimensions.")
    g, dg = arrays[0]
    c, dc, b, db = arrays[1]
    if any(np.linalg.eigvalsh(a)[0] <= 0 for a in (g, c, b)):
        raise ValueError("G, C and B must be positive definite.")
    x = np.linalg.solve(b, c)
    metric = c @ x
    dmetric = dc @ x + x.T @ dc - x.T @ db @ x
    values, vectors = np.linalg.eigh((metric + metric.T) / 2)
    if values[0] <= 0:
        raise ValueError("The polar metric must be positive definite.")
    s = (vectors * np.sqrt(values)) @ vectors.T
    ds = solve_sylvester(s, s, (dmetric + dmetric.T) / 2)
    r = np.linalg.solve(s, np.eye(s.shape[0]))
    dr = -r @ ds @ r
    phi = r @ g @ r
    dphi = dr @ g @ r + r @ dg @ r + r @ g @ dr
    result = np.stack((phi, dphi))
    return (result + result.transpose(0, 2, 1)) / 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nG=np.array([[1.2,.13,.22,-.08],[.13,.8,.03,.17],[.22,.03,.9,.11],[-.08,.17,.11,1.1]])\nH=np.array([[.2,.03,.1,0],[.03,-.1,.04,.06],[.1,.04,.05,-.02],[0,.06,-.02,.08]])\na=np.stack((G,H))\nfrom scipy.linalg import sqrtm,solve_sylvester\nC=sqrtm(G); dC=solve_sylvester(C,C,H);J=np.block([[np.zeros((2,2)),np.eye(2)],[-np.eye(2),np.zeros((2,2))]])\nA=C@J@C;dA=dC@J@C+C@J@dC;B=sqrtm(-A@A);dB=solve_sylvester(B,B,-dA@A-A@dA)\nb=np.stack((C,dC,B,dB)).real", "call": "balance_contour_jet(a.copy(),b.copy())", "gold_call": "_oracle_balance_contour_jet(a.copy(),b.copy())"},
        {"setup": "import numpy as np\na=np.stack((.5*np.eye(4),.2*np.eye(4)))\nb=np.stack((np.sqrt(.5)*np.eye(4),.1/np.sqrt(.5)*np.eye(4),.5*np.eye(4),.2*np.eye(4)))", "call": "balance_contour_jet(a.copy(),b.copy())", "gold_call": "_oracle_balance_contour_jet(a.copy(),b.copy())"},
        {"setup": "import numpy as np\ng=np.array([.02,18.]);h=np.array([.001,-.2]);v=np.sqrt(np.prod(g));dv=v/2*np.sum(h/g)\na=np.stack((np.diag(g),np.diag(h)))\nb=np.stack((np.diag(np.sqrt(g)),np.diag(h/(2*np.sqrt(g))),v*np.eye(2),dv*np.eye(2)))", "call": "balance_contour_jet(a.copy(),b.copy())", "gold_call": "_oracle_balance_contour_jet(a.copy(),b.copy())"},
        {"setup": "import numpy as np\na=np.stack((np.eye(2),np.zeros((2,2))))\nb=np.stack((np.eye(2),np.zeros((2,2)),np.eye(2),np.zeros((2,2))))", "call": "balance_contour_jet(a.copy(),b.copy())", "gold_call": "_oracle_balance_contour_jet(a.copy(),b.copy())"},
        {"setup": "import numpy as np\na=np.stack((np.eye(2),np.zeros((2,2))))\nb=np.zeros((4,2,2))\ndef check(f):\n    try:\n        f(a.copy(),b.copy())\n    except ValueError:\n        return 1\n    return 0", "call": "check(balance_contour_jet)", "gold_call": "check(_oracle_balance_contour_jet)"},
        {"setup": "import numpy as np\na=np.stack((np.eye(2),np.zeros((2,2))))\nb=np.zeros((4,4,4))\ndef check(f):\n    try:\n        f(a.copy(),b.copy())\n    except ValueError:\n        return 1\n    return 0", "call": "check(balance_contour_jet)", "gold_call": "check(_oracle_balance_contour_jet)"}
    ]

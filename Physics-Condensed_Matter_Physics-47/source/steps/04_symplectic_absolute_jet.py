"""
Construct the covariance square root, the positive absolute matrix of its symplectic congruence, and both directional derivatives.

The square-root congruence of a covariance with the canonical symplectic form is antisymmetric. Its positive absolute value carries the doubled symplectic spectrum. Differentiating positive square roots through their Sylvester equations remains well defined at repeated symplectic eigenvalues.

Returns
-------
ndarray, shape (4, 2*m, 2*m), the covariance square root and symplectic absolute matrix with their directional derivatives.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def symplectic_absolute_jet(covariance_jet: "np.ndarray") -> "np.ndarray":
    """Return the square-root and symplectic-absolute value/tangent pairs.

    Parameters
    ----------
    covariance_jet : np.ndarray
        Real array of shape (2, 2*m, 2*m), m >= 1, containing a positive definite covariance G and a symmetric direction dG. Coordinates are grouped as all positions followed by all momenta. Symmetry is checked with absolute tolerance 1e-10 and zero relative tolerance.

    Returns
    -------
    absolute_jet : np.ndarray
        Real array of shape (4, 2*m, 2*m), ordered C, dC, B, dB, where C is the principal positive square root of G and B is the positive absolute value of C J C, with J = [[0,I],[-I,0]]. The directions are exact first variations along G + epsilon*dG at epsilon=0. All four matrices are symmetric. Repeated eigenvalues are valid. Caller inputs are not mutated.

    Raises
    ------
    ValueError
        If the input is not finite and real, has the wrong shape, contains nonsymmetric matrices, or G is not positive definite.
    """
    return absolute_jet

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_symplectic_absolute_jet(covariance_jet: "np.ndarray") -> "np.ndarray":
    import numpy as np
    from scipy.linalg import solve_sylvester

    raw = np.asarray(covariance_jet)
    if np.iscomplexobj(raw):
        raise ValueError("The covariance jet must be real.")
    try:
        jet = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("The covariance jet must be numerical.") from exc
    if (jet.ndim != 3 or jet.shape[0] != 2 or jet.shape[1] != jet.shape[2]
            or jet.shape[1] < 2 or jet.shape[1] % 2 or not np.isfinite(jet).all()):
        raise ValueError("The covariance jet must have shape (2, 2*m, 2*m).")
    if not np.allclose(jet, jet.transpose(0, 2, 1), atol=1e-10, rtol=0):
        raise ValueError("Both covariance matrices must be symmetric.")
    g, dg = (jet + jet.transpose(0, 2, 1)) / 2
    values, vectors = np.linalg.eigh(g)
    if values[0] <= 0:
        raise ValueError("The covariance must be positive definite.")
    c = (vectors * np.sqrt(values)) @ vectors.T
    dc = solve_sylvester(c, c, dg)
    m = g.shape[0] // 2
    j = np.block([[np.zeros((m, m)), np.eye(m)], [-np.eye(m), np.zeros((m, m))]])
    a = c @ j @ c
    da = dc @ j @ c + c @ j @ dc
    square = a @ a.T
    values, vectors = np.linalg.eigh((square + square.T) / 2)
    if values[0] <= 0:
        raise ValueError("The symplectic absolute square must be positive definite.")
    b = (vectors * np.sqrt(values)) @ vectors.T
    db = solve_sylvester(b, b, da @ a.T + a @ da.T)
    result = np.stack((c, dc, b, db))
    return (result + result.transpose(0, 2, 1)) / 2

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nG=np.array([[1.2,.13,.22,-.08],[.13,.8,.03,.17],[.22,.03,.9,.11],[-.08,.17,.11,1.1]])\nH=np.array([[.2,.03,.1,0],[.03,-.1,.04,.06],[.1,.04,.05,-.02],[0,.06,-.02,.08]])\na=np.stack((G,H))", "call": "symplectic_absolute_jet(a.copy())", "gold_call": "_oracle_symplectic_absolute_jet(a.copy())"},
        {"setup": "import numpy as np\na=np.stack((.5*np.eye(4),np.array([[.2,.1,.3,0],[.1,-.2,0,.1],[.3,0,-.1,.2],[0,.1,.2,.4]])))", "call": "symplectic_absolute_jet(a.copy())", "gold_call": "_oracle_symplectic_absolute_jet(a.copy())"},
        {"setup": "import numpy as np\na=np.stack((np.diag([.025,12.0]),np.array([[.001,.04],[.04,-.3]])))", "call": "symplectic_absolute_jet(a.copy())", "gold_call": "_oracle_symplectic_absolute_jet(a.copy())"},
        {"setup": "import numpy as np\na=np.stack((np.eye(6),np.zeros((6,6))))", "call": "symplectic_absolute_jet(a.copy())", "gold_call": "_oracle_symplectic_absolute_jet(a.copy())"},
        {"setup": "import numpy as np\na=np.stack((np.diag([1.,0.]),np.eye(2)))\ndef check(f):\n    try:\n        f(a.copy())\n    except ValueError:\n        return 1\n    return 0", "call": "check(symplectic_absolute_jet)", "gold_call": "check(_oracle_symplectic_absolute_jet)"},
        {"setup": "import numpy as np\na=np.ones((2,3,3))\ndef check(f):\n    try:\n        f(a.copy())\n    except ValueError:\n        return 1\n    return 0", "call": "check(symplectic_absolute_jet)", "gold_call": "check(_oracle_symplectic_absolute_jet)"},
        {"setup": "import numpy as np\na=np.stack((np.eye(2),np.array([[0.,1.],[0.,0.]])))\ndef check(f):\n    try:\n        f(a.copy())\n    except ValueError:\n        return 1\n    return 0", "call": "check(symplectic_absolute_jet)", "gold_call": "check(_oracle_symplectic_absolute_jet)"}
    ]

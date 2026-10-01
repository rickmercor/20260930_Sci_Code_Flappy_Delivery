"""
Convert measured competition into the niche-conditioned condition panel.

Niche overlap changes the strength of competition, while a difference in niche breadth makes that change directional. Returns
-------
return competition

Returns
-------
return competition
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_competition(base_competition: "numpy.ndarray", geometry: "numpy.ndarray",
                          coupling: "numpy.ndarray") -> "numpy.ndarray":
    """Apply the declared directed ecological bridge.

    For each condition s and off-diagonal pair i,j, multiply the measured
    coefficient by exp(coupling[s,0]*(overlap_ij-mean_overlap) +
    coupling[s,1]*(breadth_j-breadth_i)). The overlap mean is over all
    ordered off-diagonal entries. Diagonal entries remain zero.

    Raises
    ------
    ValueError
        If the arrays are nonfinite or dimensionally misaligned, the
        competition panel is not square and nonnegative with a zero
        diagonal, the geometry matrix is not symmetric and aligned, or
        coupling does not contain one two-value row per condition.
    """
    return competition

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_effective_competition(base_competition: "numpy.ndarray", geometry: "numpy.ndarray",
                                  coupling: "numpy.ndarray") -> "numpy.ndarray":
    B = np.asarray(base_competition, dtype=np.float64)
    G = np.asarray(geometry, dtype=np.float64)
    C = np.asarray(coupling, dtype=np.float64)
    if (B.ndim != 3 or B.shape[0] < 1 or B.shape[1] != B.shape[2]
            or B.shape[1] < 2 or G.shape != B.shape[1:] or C.shape != (B.shape[0], 2)
            or any(not np.all(np.isfinite(x)) for x in (B, G, C))
            or np.any(B < 0.0) or np.max(np.abs(np.diagonal(B, axis1=1, axis2=2))) > 1e-12
            or np.max(np.abs(G-G.T)) > 1e-10):
        raise ValueError("aligned finite competition, symmetric geometry and coupling are required")
    n = len(G)
    beta = np.diag(G)
    beta0 = beta-beta.mean()
    mask = ~np.eye(n, dtype=bool)
    mean_overlap = G[mask].mean()
    centered_overlap = G-mean_overlap
    result = B.copy()
    for s in range(len(B)):
        exponent = (C[s, 0]*centered_overlap
                    + C[s, 1]*(beta0[None, :]-beta0[:, None]))
        result[s] *= np.exp(exponent)
        np.fill_diagonal(result[s], 0.0)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nB=np.array([[[0,.4,.7],[.2,0,.5],[.9,.3,0]],[[0,.6,.4],[.3,0,.8],[.7,.2,0]]],float)\nG=np.array([[.8,.2,.5],[.2,.6,.4],[.5,.4,.7]])\nC=np.array([[.7,-.4],[-.2,.9]])",
            "call": "effective_competition(B,G,C)",
            "gold_call": "_oracle_effective_competition(B,G,C)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nB=np.array([[[0,.4],[.2,0]]],float)\nG=np.array([[.8,.3],[.3,.5]])\nC=np.zeros((1,2))",
            "call": "effective_competition(B,G,C)",
            "gold_call": "_oracle_effective_competition(B,G,C)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nB=np.zeros((3,4,4));B[:,~np.eye(4,dtype=bool)]=.5\nG=np.eye(4)*.7+(1-np.eye(4))*.25\nC=np.array([[3.,-2.],[0.,0.],[-1.,4.]])",
            "call": "effective_competition(B,G,C)",
            "gold_call": "_oracle_effective_competition(B,G,C)",
            "tol": 1e-10,
        },
        {
            "setup": "import numpy as np\nB=np.array([[[1.,.4],[.2,0.]]]);G=np.array([[.8,.3],[.3,.5]]);C=np.zeros((1,2))\ndef run(fn):\n    try: fn(B,G,C)\n    except ValueError: return 1.0\n    return 0.0",
            "call": "run(effective_competition)",
            "gold_call": "run(_oracle_effective_competition)",
            "tol": 0.0,
        },
    ]

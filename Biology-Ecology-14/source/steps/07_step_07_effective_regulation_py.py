"""
Convert plan-level regulation measurements into niche-conditioned slopes.

Each plan has its own response to deviations from the community's mean niche breadth, so the same resource observations alter plans differently.
Returns
-------
return regulation

Returns
-------
return regulation
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def effective_regulation(base_designs: "numpy.ndarray", geometry: "numpy.ndarray",
                         breadth_response: "numpy.ndarray") -> "numpy.ndarray":
    """Multiply plan c, species i by exp(q_c*(beta_i-mean(beta))).

    Raises
    ------
    ValueError
        If the arrays are nonfinite or dimensionally misaligned, any
        regulation slope in base_designs is not strictly positive, the
        geometry matrix is not symmetric and aligned, or breadth_response
        does not contain one value per plan.
    """
    return regulation

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_effective_regulation(base_designs: "numpy.ndarray", geometry: "numpy.ndarray",
                                 breadth_response: "numpy.ndarray") -> "numpy.ndarray":
    D = np.asarray(base_designs, dtype=np.float64)
    G = np.asarray(geometry, dtype=np.float64)
    q = np.asarray(breadth_response, dtype=np.float64)
    if (D.ndim != 2 or D.shape[0] < 1 or D.shape[1] < 2
            or G.shape != (D.shape[1], D.shape[1]) or q.shape != (D.shape[0],)
            or any(not np.all(np.isfinite(x)) for x in (D, G, q))
            or np.any(D <= 0.0) or np.max(np.abs(G-G.T)) > 1e-10):
        raise ValueError("positive plans, aligned symmetric geometry and one response per plan are required")
    beta = np.diag(G)
    return D*np.exp(q[:, None]*(beta-beta.mean())[None, :])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nD=np.array([[1.,2.,3.],[.7,1.4,.9]])\nG=np.array([[.8,.2,.5],[.2,.6,.4],[.5,.4,.7]])\nq=np.array([-1.2,.8])",
            "call": "effective_regulation(D,G,q)",
            "gold_call": "_oracle_effective_regulation(D,G,q)",
            "tol": 1e-11,
        },
        {
            "setup": "import numpy as np\nD=np.array([[1.,2.],[3.,4.]])\nG=np.array([[.5,.2],[.2,.5]])\nq=np.array([8.,-4.])",
            "call": "effective_regulation(D,G,q)",
            "gold_call": "_oracle_effective_regulation(D,G,q)",
            "tol": 1e-12,
        },
        {
            "setup": "import numpy as np\nD=np.ones((3,4))\nG=np.eye(4)*np.array([.2,.4,.7,.9])+(1-np.eye(4))*.1\nq=np.array([0.,2.,-3.])",
            "call": "effective_regulation(D,G,q)",
            "gold_call": "_oracle_effective_regulation(D,G,q)",
            "tol": 1e-11,
        },
        {
            "setup": "import numpy as np\nD=np.array([[1.,0.]]);G=np.eye(2);q=np.array([1.])\ndef run(fn):\n    try: fn(D,G,q)\n    except ValueError: return 1.0\n    return 0.0",
            "call": "run(effective_regulation)",
            "gold_call": "run(_oracle_effective_regulation)",
            "tol": 0.0,
        },
    ]

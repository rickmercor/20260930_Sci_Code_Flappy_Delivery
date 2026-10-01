"""
Evaluate the supplied polynomial potential, its gradient, and its Hessian at one geometry.

The full potential and gradient determine the moving centre. The Hessian at the reference geometry is held fixed during width propagation.

Returns
-------
value, gradient, hessian
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def potential_jet(q, origin, powers, coefficients):
    r"""Evaluate V(q)=sum_r coefficients[r]*product_j (q[j]-origin[j])**powers[r,j].
    Coefficients are ordinary monomial coefficients, with no factorial convention.
    Differentiate analytically; coordinates equal to the expansion origin are valid.

    Parameters
    ----------
    q, origin : real arrays, shape (D,), D >= 1
    powers : numeric integer-valued array, shape (R,D), entries >= 0
    coefficients : real array, shape (R,)
        R=0 is allowed and means the zero potential. Repeated powers add.

    Returns
    -------
    value : float
    gradient : real array, shape (D,)
    hessian : real array, shape (D,D)

    Raises
    ------
    ValueError
        For incompatible shapes, empty q, nonfinite data, or powers that are
        negative or not integer-valued.
    """
    return value, gradient, hessian

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_potential_jet(q, origin, powers, coefficients):
    import numpy as np
    q = np.asarray(q, dtype=float)
    origin = np.asarray(origin, dtype=float)
    raw = np.asarray(powers)
    coefficients = np.asarray(coefficients, dtype=float)
    if (q.ndim != 1 or q.size == 0 or origin.shape != q.shape
            or raw.ndim != 2 or raw.shape[1] != q.size
            or coefficients.shape != (raw.shape[0],)
            or not all(np.all(np.isfinite(a)) for a in (q, origin, raw, coefficients))
            or np.any(raw < 0) or np.any(raw != np.floor(raw))):
        raise ValueError("Invalid polynomial arrays")
    powers = raw.astype(int)
    x = q - origin
    d = q.size
    value = 0.0
    gradient = np.zeros(d)
    hessian = np.zeros((d, d))
    for alpha, weight in zip(powers, coefficients):
        value += weight * np.prod(x ** alpha)
        for i in range(d):
            if alpha[i] == 0:
                continue
            beta = alpha.copy()
            beta[i] -= 1
            gradient[i] += weight * alpha[i] * np.prod(x ** beta)
            for j in range(d):
                if beta[j] == 0:
                    continue
                gamma = beta.copy()
                gamma[j] -= 1
                hessian[i, j] += weight * alpha[i] * beta[j] * np.prod(x ** gamma)
    return float(value), gradient, hessian

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "name": 'case_1',
            "setup": """import numpy as np
data={'origin': [0.1, -0.15, 0.08], 'q_reference': [0.06, -0.04, 0.12], 'potential_powers': [[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1], [3, 0, 0], [0, 3, 0], [0, 0, 3], [2, 1, 0], [1, 1, 1], [4, 0, 0], [0, 4, 0], [0, 0, 4], [2, 2, 0], [0, 2, 2], [2, 0, 2]], 'potential_coefficients': [1.6, 0.5, 0.925, 0.65, 0.21, -0.16, 0.27, 0.06, -0.04, 0.03, 0.08, -0.055, 0.018, 0.024, 0.02, 0.012, 0.016, 0.01]}
""",
            "call": "potential_jet(data['q_reference'], data['origin'], data['potential_powers'], data['potential_coefficients'])",
            "gold_call": "_oracle_potential_jet(data['q_reference'], data['origin'], data['potential_powers'], data['potential_coefficients'])",
        },
        {
            "name": 'case_2',
            "setup": """import numpy as np
q=np.zeros(2);origin=q.copy();powers=np.array([[0,0],[1,0],[0,1],[2,0],[1,1],[0,3]]);a=np.array([2.,3.,-4.,5.,7.,9.])
""",
            "call": 'potential_jet(q, origin, powers, a)',
            "gold_call": '_oracle_potential_jet(q, origin, powers, a)',
        },
        {
            "name": 'case_3',
            "setup": """import numpy as np
q=np.array([0.2]);origin=np.zeros(1);powers=np.empty((0,1),int);a=np.empty(0)
""",
            "call": 'potential_jet(q, origin, powers, a)',
            "gold_call": '_oracle_potential_jet(q, origin, powers, a)',
        },
        {
            "name": 'case_4',
            "setup": """import numpy as np
q=np.array([0.,-0.3,0.8]);origin=np.zeros(3);powers=np.array([[2,1,1],[2,1,1],[0,0,0],[0,2,3]]);a=np.array([.2,-.5,1.,.08])
""",
            "call": 'potential_jet(q, origin, powers, a)',
            "gold_call": '_oracle_potential_jet(q, origin, powers, a)',
        },
        {
            "name": 'case_5',
            "setup": """import numpy as np
q=np.zeros(1);origin=q.copy();powers=np.array([[-1]]);a=np.ones(1)

def _status(fn, *args):
    try:
        fn(*args)
    except ValueError:
        return 1
    raise AssertionError("Expected ValueError")
""",
            "call": '_status(potential_jet, q, origin, powers, a)',
            "gold_call": '_status(_oracle_potential_jet, q, origin, powers, a)',
        },
    ]

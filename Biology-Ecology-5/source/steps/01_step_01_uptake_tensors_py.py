"""
Calculate resource derivatives of saturating uptake through third order.

Species-specific uptake saturation makes both growth gradients and ecological feedback change with supply; third resource derivatives enter a mixed environmental response.

Returns
-------
Shape (4,P,M). Entry [k,i,a] is the kth ordinary derivative in resource a, k=0,1,2,3, of U[i,a]*R[a]/(half[i,a]+R[a]). Derivatives are not divided by factorials; other-resource derivatives vanish.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def uptake_tensors(U: 'np.ndarray', half: 'np.ndarray', R: 'np.ndarray') -> 'np.ndarray':
    """Calculate resource derivatives of saturating uptake through third order.

    Parameters
    ----------
    U : np.ndarray
        Nonnegative maximum uptake rates (P,M), P and M positive.
    half : np.ndarray
        Strictly positive half-saturation constants (P,M).
    R : np.ndarray
        Strictly positive resource abundances (M,).

    Returns
    -------
    result : np.ndarray
        Shape (4,P,M). Entry [k,i,a] is the kth ordinary derivative in
        resource a, k=0,1,2,3, of U[i,a]*R[a]/(half[i,a]+R[a]).
        Derivatives are not divided by factorials; other-resource derivatives vanish.

    Raises
    ------
    ValueError
        If inputs are nonreal, nonfinite, have incompatible or empty shapes,
        any U is negative, any half or R is nonpositive, or a result is nonfinite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _arr(x, ndim=None, shape=None):
    try:
        if np.iscomplexobj(x): raise ValueError('real data required')
        a=np.asarray(x,dtype=float)
    except (ValueError,TypeError,OverflowError) as e: raise ValueError('numeric data required') from e
    if (ndim is not None and a.ndim!=ndim) or (shape is not None and a.shape!=shape) or not np.all(np.isfinite(a)): raise ValueError('invalid shape or finite data')
    return a

def _solve(a,b):
    try: z=np.linalg.solve(a,b)
    except np.linalg.LinAlgError as e: raise ValueError('singular response') from e
    return _arr(z)

def _oracle_uptake_tensors(U: 'np.ndarray', half: 'np.ndarray', R: 'np.ndarray') -> 'np.ndarray':
    U=_arr(U,2);half=_arr(half,2,U.shape);R=_arr(R,1,(U.shape[1],))
    if min(U.shape)<1 or np.any(U<0) or np.any(half<=0) or np.any(R<=0):raise ValueError('invalid uptake domain')
    den=half+R
    return _arr(np.stack([U*R/den,U*half/den**2,-2*U*half/den**3,6*U*half/den**4]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge and declared-invalid test cases."""
    return [{'setup': 'import numpy as np\nimport copy\n# normal\na0=np.array([[1.0, 0.2, 0.1, 0.05], [0.1, 0.9, 0.25, 0.1], [0.15, 0.1, 0.8, 0.3], [0.4, 0.3, 0.2, 0.1], [0.425, 0.28, 0.215, 0.09]], dtype=float)\na1=np.array([[0.4, 0.7, 0.3, 0.8], [0.6, 0.2, 0.9, 0.5], [0.3, 0.8, 0.4, 0.6], [0.5, 0.4, 0.7, 0.3], [0.52, 0.38, 0.72, 0.28]], dtype=float)\na2=np.array([1.0, 0.8, 1.2, 0.9], dtype=float)\n', 'call': 'uptake_tensors(a0.copy(), a1.copy(), a2.copy())', 'gold_call': '_oracle_uptake_tensors(a0.copy(), a1.copy(), a2.copy())', 'tol': 1e-08}, {'setup': 'import numpy as np\nimport copy\n# boundary\na0=np.array([[0.0, 0.0]], dtype=float)\na1=np.array([[1.0, 1.0]], dtype=float)\na2=np.array([1.0, 1.0], dtype=float)\n', 'call': 'uptake_tensors(a0.copy(), a1.copy(), a2.copy())', 'gold_call': '_oracle_uptake_tensors(a0.copy(), a1.copy(), a2.copy())', 'tol': 1e-08}, {'setup': 'import numpy as np\nimport copy\n# edge\na0=np.array([[2.0]], dtype=float)\na1=np.array([[0.001]], dtype=float)\na2=np.array([0.01], dtype=float)\n', 'call': 'uptake_tensors(a0.copy(), a1.copy(), a2.copy())', 'gold_call': '_oracle_uptake_tensors(a0.copy(), a1.copy(), a2.copy())', 'tol': 1e-08}, {'setup': 'import numpy as np\nimport copy\n# invalid_declared_condition\na0=np.array([[1.0, 0.2, 0.1, 0.05], [0.1, 0.9, 0.25, 0.1], [0.15, 0.1, 0.8, 0.3], [0.4, 0.3, 0.2, 0.1], [0.425, 0.28, 0.215, 0.09]], dtype=float)\na1=np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]], dtype=float)\na2=np.array([1.0, 0.8, 1.2, 0.9], dtype=float)\n\ndef expect_value_error(fn, *args):\n    try:\n        fn(*args)\n    except ValueError:\n        return 1\n    return 0\n', 'call': 'expect_value_error(uptake_tensors, copy.deepcopy(a0), copy.deepcopy(a1), copy.deepcopy(a2))', 'gold_call': 'expect_value_error(_oracle_uptake_tensors, copy.deepcopy(a0), copy.deepcopy(a1), copy.deepcopy(a2))', 'tol': 0}]

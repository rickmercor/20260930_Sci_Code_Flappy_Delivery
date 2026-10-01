"""
Continue a simple determinant root and its logarithmic conditioning response.

Polynomial elimination converts coefficient perturbations into a moving root.

The implicit branch and the derivative of the determinant both contribute to

the response of the coefficientwise root condition number. Truncated power

series retain their coupled contributions without subtracting nearby roots.

Returns
-------
A real array of shape $(P+1,2)$ containing root and natural-log-condition Taylor coefficients.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def continue_condition_jets(
    coefficient_jets: np.ndarray, anchor_root: float,
) -> np.ndarray:
    r"""Compute Taylor coefficients of a real simple root and its log condition.
    
    Parameters
    ----------
    coefficient_jets : np.ndarray
        Finite real array of shape $(P+1,K)$, with $0\le P\le6$ and $K\ge2$.
        Entry $(a,n)$ is the Taylor coefficient $c_{a n}$ in
        $D(t,x)=\sum_{a=0}^{P}\sum_{n=0}^{K-1}c_{a n}t^a x^n+O(t^{P+1})$.
        These are Taylor coefficients, not ordinary derivatives. High-degree
        zero padding is retained: the conditioning numerator uses all $K$ powers.
    anchor_root : float
        Finite real approximation $x_0$ to a simple root of $D(0,x)$.
        With $Q=\sum_n|c_{0n}|\max\{1,|x_0|\}^n$, require $Q>0$ and
        $|D(0,x_0)|\le10^{-8}Q$. The nearby simple root determines the branch.
    
    Returns
    -------
    response : np.ndarray
        Finite real array of shape $(P+1,2)$. Column zero contains the Taylor
        coefficients of the branch $x(t)$ satisfying $D(t,x(t))=0$.
        Column one contains the Taylor coefficients of
        $\ell(t)=\log\bigl(\sqrt{\sum_{n=0}^{K-1}x(t)^{2n}}/
        |\partial_xD(t,x(t))|\bigr)$, where $\log$ is natural logarithm.
        Row $a$ contains the ordinary derivative divided by $a!$.
    
    Raises
    ------
    ValueError
        If the array is complex, nonfinite, has invalid shape, or has zero base
        coefficients; the anchor is nonfinite or violates its residual bound;
        the nearby root is numerically multiple; or the response is nonfinite.
        A derivative magnitude no greater than
        $100\epsilon\sum_{n=1}^{K-1}n|c_{0n}|\max\{1,|x|\}^{n-1}$
        is numerically zero, with $\epsilon$ double-precision machine epsilon.
    
    Notes
    -----
    Refine the base approximation within its simple-root neighborhood before
    continuing the branch. Obtain the response from the supplied Taylor data,
    including the implicit motion of the root and the varying determinant scale.
    No residual or interval gate is applied in this step. Constant rescaling of
    all coefficient jets changes only the zeroth log-condition coefficient.
    Do not mutate inputs.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _response_product(a, b, length):
    return np.convolve(a,b)[:length]

def _response_evaluate(coefficients, root):
    length = len(root)
    value = np.zeros(length)
    for degree in range(coefficients.shape[1]-1,-1,-1):
        value = _response_product(value,root,length) + coefficients[:,degree]
    return value

def _response_log_positive(series):
    length = len(series)
    if series[0] <= 0:
        raise ValueError("positive logarithm base required")
    inverse = np.zeros(length)
    inverse[0] = 1.0/series[0]
    for order in range(1,length):
        inverse[order] = -np.dot(series[1:order+1],inverse[order-1::-1])/series[0]
    result = np.zeros(length)
    result[0] = np.log(series[0])
    if length > 1:
        derivative = np.arange(1,length)*series[1:]
        result[1:] = np.convolve(derivative,inverse)[:length-1]/np.arange(1,length)
    return result

def _oracle_continue_condition_jets(
    coefficient_jets: np.ndarray, anchor_root: float,
) -> np.ndarray:
    original = np.asarray(coefficient_jets)
    if (original.ndim != 2 or not 1 <= original.shape[0] <= 7
            or original.shape[1] < 2 or not np.isrealobj(original)
            or not np.all(np.isfinite(original))):
        raise ValueError("invalid real coefficient jets")
    if not np.isscalar(anchor_root) or not np.isrealobj(anchor_root) or not np.isfinite(anchor_root):
        raise ValueError("invalid anchor root")
    scale = float(np.max(np.abs(original[0])))
    if scale == 0:
        raise ValueError("zero base polynomial")
    coefficients = np.asarray(original,dtype=float)/scale
    length, width = coefficients.shape
    root = np.zeros(length)
    root[0] = float(anchor_root)
    base = coefficients[0]
    derivative = coefficients[:,1:]*np.arange(1,width)
    bound = float(np.dot(np.abs(base),max(1.0,abs(root[0]))**np.arange(width)))
    if abs(np.polynomial.polynomial.polyval(root[0],base)) > 1e-8*bound:
        raise ValueError("anchor is not a base root")
    for _ in range(8):
        slope = float(np.polynomial.polynomial.polyval(root[0],derivative[0]))
        slope_bound = float(np.dot(np.abs(derivative[0]),max(1.0,abs(root[0]))**np.arange(width-1)))
        if abs(slope) <= 100*np.finfo(float).eps*slope_bound:
            raise ValueError("multiple base root")
        correction = np.polynomial.polynomial.polyval(root[0],base)/slope
        root[0] -= correction
        if abs(correction) <= 2*np.finfo(float).eps*max(1.0,abs(root[0])):
            break
    slope = float(np.polynomial.polynomial.polyval(root[0],derivative[0]))
    for order in range(1,length):
        root[order] = -_response_evaluate(coefficients,root)[order]/slope
    moving_slope = _response_evaluate(derivative,root)
    numerator_squared = np.zeros(length)
    power = np.zeros(length)
    power[0] = 1.0
    squared_root = _response_product(root,root,length)
    for _ in range(width):
        numerator_squared += power
        power = _response_product(power,squared_root,length)
    log_condition = 0.5*_response_log_positive(numerator_squared)
    log_condition -= _response_log_positive(moving_slope*np.sign(moving_slope[0]))
    log_condition[0] -= np.log(scale)
    result = np.column_stack([root,log_condition])
    if not np.all(np.isfinite(result)):
        raise ValueError("nonfinite continuation")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Scientific and exact continuation invariants."""
    return [{'setup': 'import numpy as np\n'
               'C = np.array([[-2.0, 1.0, 1.0], [-0.30000000000000004, -0.30000000000000004, 0.0], [0.1, '
               '0.04, 0.0], [-0.024, -0.01, 0.0], [0.001, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, '
               '0.0]],dtype=float)\n'
               'x0 = 1.00000000001\n'
               'expected = np.array([[1.0, -0.5493061443340549], [0.2, 0.16666666666666666], [-0.04, '
               '-0.019444444444444445], [0.01, 0.0008765432098765433], [0.0, 0.002970679012345679], [0.0, '
               '-0.0010572427983539094], [0.0, 0.00026680681298582533]],dtype=float)\n',
      'call': 'continue_condition_jets(C.copy(), x0)',
      'gold_call': '_oracle_continue_condition_jets(C.copy(), x0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'C = np.array([[0.0, -2.0, -1.0, 1.0], [0.8, 0.30000000000000004, -0.8, 0.3], '
               '[0.23999999999999996, 0.47999999999999987, -0.22999999999999998, -0.07], '
               '[-0.47800000000000004, -0.20200000000000004, 0.19000000000000003, 0.0], '
               '[0.11640000000000002, 0.0020999999999999977, 0.0004999999999999961, 0.0], '
               '[0.03244000000000001, 0.014700000000000005, -0.025000000000000005, 0.0], '
               '[-0.020110000000000006, -0.0022499999999999985, 0.004900000000000001, 0.0]],dtype=float)\n'
               'x0 = 1e-11\n'
               'expected = np.array([[0.0, -0.6931471805599453], [0.4, -0.25], [0.1, 0.17125], [-0.2, '
               '0.13879166666666667], [0.07, -0.1535734375], [-0.01, 0.0170321875], [0.003, '
               '0.04616648177083333]],dtype=float)\n',
      'call': 'continue_condition_jets(C.copy(), x0)',
      'gold_call': '_oracle_continue_condition_jets(C.copy(), x0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'C = np.array([[2.4, -4.0, -2.6, 2.0, 0.0, 0.0, 0.0], [-1.4, 0.6399999999999998, '
               '0.39000000000000007, -0.3, 0.0, 0.0, 0.0], [-0.22200000000000003, 0.9339999999999999, -0.4, '
               '0.0, 0.0, 0.0, 0.0], [0.26070000000000004, -0.33099999999999996, 0.12, 0.0, 0.0, 0.0, 0.0], '
               '[-0.0436, 0.0345, -0.009, 0.0, 0.0, 0.0, 0.0]],dtype=float)\n'
               'x0 = -1.19999999999\n'
               'expected = np.array([[-1.2, -0.7407285840512312], [0.1, -0.03340688343500852], [0.2, '
               '-0.507298280754369], [-0.03, 0.1292320786415941], [0.0, '
               '0.044435677203793475]],dtype=float)\n',
      'call': 'continue_condition_jets(C.copy(), x0)',
      'gold_call': '_oracle_continue_condition_jets(C.copy(), x0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'C = np.array([[-5.1e+119, 2e+120, -1e+120], [-1.0699999999999994e+119, 1.5e+120, '
               '-6.999999999999999e+119], [1.3299999999999999e+119, 4.700000000000001e+119, '
               '-2.0000000000000002e+119], [9.2e+118, 1.999999999999999e+118, -0.0], '
               '[1.2000000000000002e+118, -0.0, -0.0], [-0.0, -0.0, -0.0], [-0.0, -0.0, '
               '-0.0]],dtype=float)\n'
               'x0 = 0.30000000001\n'
               'expected = np.array([[0.3, -276.59989268909595], [-0.2, -1.1216178594195167], [0.0, '
               '0.13266693160360124], [0.0, 0.009370316520248964], [0.0, -0.014097702366392928], [0.0, '
               '0.0059440613555893015], [0.0, -0.001329085484196892]],dtype=float)\n',
      'call': 'continue_condition_jets(C.copy(), x0)',
      'gold_call': '_oracle_continue_condition_jets(C.copy(), x0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'C = np.array([[-6.0, 3.0]],dtype=float)\n'
               'x0 = 2.00000000001\n'
               'expected = np.array([[2.0, -0.29389333245105953]],dtype=float)\n',
      'call': 'continue_condition_jets(C.copy(), x0)',
      'gold_call': '_oracle_continue_condition_jets(C.copy(), x0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'C = np.array([[-1.0, 1.0, 0.0, 0.0], [-0.3, 0.3, 0.0, 0.0], [0.1, -0.1, 0.0, 0.0], [0.0, '
               '0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, '
               '0.0]],dtype=float)\n'
               'x0 = 1.00000000001\n'
               'expected = np.array([[1.0, 0.6931471805599453], [0.0, -0.3], [0.0, 0.145], [0.0, -0.039], '
               '[0.0, 0.016025], [0.0, -0.006186], [0.0, 0.002614833333333333]],dtype=float)\n',
      'call': 'continue_condition_jets(C.copy(), x0)',
      'gold_call': '_oracle_continue_condition_jets(C.copy(), x0)',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'def checked(fn):\n'
               '    bad = [(np.zeros((2,3)),0.0), (np.array([[1.,0.,1.]]),0.0),\n'
               '           (np.array([[1.,-2.,1.],[0.,1.,0.]]),1.0),\n'
               '           (np.array([[-1.,1.],[np.nan,0.]]),1.0),\n'
               '           (np.ones((8,2)),0.0), (np.array([[-1+0j,1+0j]]),1.0)]\n'
               '    for C,x in bad:\n'
               '        try: fn(C.copy(),x)\n'
               '        except ValueError: continue\n'
               '        return 0\n'
               '    return 1\n',
      'call': 'checked(continue_condition_jets)',
      'gold_call': 'checked(_oracle_continue_condition_jets)',
      'tol': 0},
     {'setup': 'import numpy as np\n'
               'C = np.array([[0.0, -2.0, -1.0, 1.0], [0.8, 0.30000000000000004, -0.8, 0.3], '
               '[0.23999999999999996, 0.47999999999999987, -0.22999999999999998, -0.07], '
               '[-0.47800000000000004, -0.20200000000000004, 0.19000000000000003, 0.0], '
               '[0.11640000000000002, 0.0020999999999999977, 0.0004999999999999961, 0.0], '
               '[0.03244000000000001, 0.014700000000000005, -0.025000000000000005, 0.0], '
               '[-0.020110000000000006, -0.0022499999999999985, 0.004900000000000001, 0.0]],dtype=float)\n'
               'x0 = 1e-11\n'
               'expected = np.array([[0.0, -0.6931471805599453], [0.4, -0.25], [0.1, 0.17125], [-0.2, '
               '0.13879166666666667], [0.07, -0.1535734375], [-0.01, 0.0170321875], [0.003, '
               '0.04616648177083333]],dtype=float)\n'
               'def checked(fn):\n'
               '    local = C.copy()\n'
               '    before = local.copy()\n'
               '    out = fn(local,x0)\n'
               '    return int(np.array_equal(local,before) and out.shape == expected.shape and '
               'np.allclose(out,expected,rtol=1e-8,atol=1e-8))\n',
      'call': 'checked(continue_condition_jets)',
      'gold_call': 'checked(_oracle_continue_condition_jets)',
      'tol': 0},
     {'setup': 'import numpy as np\n'
               'C=np.zeros((7,2))\n'
               'C[0,1]=1; C[1,0]=-1\n'
               'expected=np.zeros((7,2)); expected[1,0]=1; expected[2,1]=0.5; expected[4,1]=-0.25; '
               'expected[6,1]=1/6\n'
               'def checked(fn):\n'
               '    result=fn(C.copy(),0.0)\n'
               '    return int(result.shape==(7,2) and '
               'np.allclose(result,expected,rtol=1e-10,atol=1e-10))\n',
      'call': 'checked(continue_condition_jets)',
      'gold_call': 'checked(_oracle_continue_condition_jets)',
      'tol': 0},
     {'setup': 'import numpy as np\n'
               'C=np.zeros((7,13))\n'
               'C[0,1]=1; C[1,0]=-1\n'
               'expected=np.zeros((7,2)); expected[1,0]=1; expected[2,1]=0.5; expected[4,1]=0.25; '
               'expected[6,1]=1/6\n'
               'def checked(fn):\n'
               '    result=fn(C.copy(),0.0)\n'
               '    return int(result.shape==(7,2) and '
               'np.allclose(result,expected,rtol=1e-10,atol=1e-10))\n',
      'call': 'checked(continue_condition_jets)',
      'gold_call': 'checked(_oracle_continue_condition_jets)',
      'tol': 0}]

"""
Compute global half-power proximal minimizers with entrywise broadcast weights and penalties, preserving branch selection and relative accuracy across numerical scales.

The separable auxiliary objective permits a different regularization weight and quadratic penalty at each entry. This coding extension uses the same scalar minimization as the source; the prescribed ecological reconstruction remains the scalar-coefficient special case. Nonzero stationary points are not necessarily global minimizers. The implementation must compare with zero, select the correct signed branch, and avoid overflow from unnecessary powers at large scales.

Returns
-------
a : ndarray, exactly b.shape. Entrywise global signed minimizer with weight and penalty broadcast to b.shape; zero wins ties, zero weights give identity, inputs remain unchanged. Preserve relative accuracy across the stated scale range.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def half_prox(
    b: 'np.ndarray',
    weight: 'np.ndarray | float',
    penalty: 'np.ndarray | float',
) -> 'np.ndarray':
    """Return the entrywise global minimizer of
    weight*sqrt(abs(a)) + penalty*(a-b)**2/2 over real a.

    b is a nonempty finite real array of any shape, including shape ().
    weight and penalty are finite real scalars or arrays. Each must be
    independently broadcastable TO b.shape without expanding that shape.
    Every weight is nonnegative; every penalty is strictly positive.
    A zero weight gives a=b at that entry. Signs are unrestricted.
    When zero and a nonzero minimizer tie, return zero. Selecting a
    stationary point without comparing its objective with zero is invalid.

    Return a new float ndarray with exactly b.shape; mutate no input.
    The effective ratio weight/penalty and its two-thirds power must be
    finite. A positive weight whose effective ratio underflows to zero
    is outside the numerical contract and must raise ValueError.
    For positive effective ratios from 1e-240 through 1e240 and nonzero
    abs(b) from 1e-200 through 1e200, valid finite outputs must not fail
    merely because an avoidable intermediate power or squared objective
    overflows. On active entries away from the switching threshold,
    preserve relative accuracy of 1e-9 even for outputs below 1e-9.
    At the switching threshold, evaluate its represented floating-point
    value consistently and choose zero for equality. Do not use an
    absolute tolerance that erases genuinely active small-scale entries.

    Raise ValueError for nonnumeric/complex/nonfinite inputs, empty b,
    incompatible broadcasting, invalid coefficient signs, a nonfinite
    effective ratio/threshold/output, or the ratio underflow above.
    """
    return a

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_half_prox(b: 'np.ndarray', weight: 'np.ndarray | float', penalty: 'np.ndarray | float') -> 'np.ndarray':
    try:
        if any(np.iscomplexobj(x) for x in (b, weight, penalty)):
            raise ValueError('real inputs required')
        b = np.asarray(b, dtype=float)
        weight = np.asarray(weight, dtype=float)
        penalty = np.asarray(penalty, dtype=float)
        weight = np.broadcast_to(weight, b.shape)
        penalty = np.broadcast_to(penalty, b.shape)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('numeric inputs broadcastable to b required') from exc
    if (b.size == 0 or any(not np.isfinite(x).all() for x in (b, weight, penalty))
            or np.any(weight < 0) or np.any(penalty <= 0)):
        raise ValueError('invalid inputs')
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        c = weight / penalty
        threshold = 1.5 * np.power(c, 2.0 / 3.0)
    if (not np.isfinite(c).all() or not np.isfinite(threshold).all()
            or np.any((weight > 0) & (c == 0))):
        raise ValueError('effective ratio outside numerical contract')
    flat = b.ravel()
    cf = c.ravel()
    wf = weight.ravel()
    result = np.zeros_like(flat)
    identity = wf == 0
    result[identity] = flat[identity]
    active = (~identity) & (np.abs(flat) > threshold.ravel())
    mag = np.abs(flat[active])
    # Dimensionless cubic: avoid mag**1.5 and squaring a large root.
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        q = (cf[active] / mag) / np.sqrt(mag)
        arg = np.clip(-(3.0 * np.sqrt(3.0) / 4.0) * q, -1.0, 1.0)
        fraction = (4.0 / 3.0) * np.cos(np.arccos(arg) / 3.0)**2
        result[active] = np.sign(flat[active]) * (mag * fraction)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite proximal result')
    return result.reshape(b.shape)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n',
      'call': 'half_prox(np.array([-5.,-.2,0.,.8,4.]), .8, .6)',
      'gold_call': '_oracle_half_prox(np.array([-5.,-.2,0.,.8,4.]), .8, .6)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n',
      'call': 'half_prox(np.array([-1.5,0.,1.5]), 1., 1.)',
      'gold_call': '_oracle_half_prox(np.array([-1.5,0.,1.5]), 1., 1.)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n',
      'call': 'half_prox(np.array([1.5-1e-8,1.5+1e-8,-1.5-1e-8]), 1., 1.)',
      'gold_call': '_oracle_half_prox(np.array([1.5-1e-8,1.5+1e-8,-1.5-1e-8]), 1., 1.)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n',
      'call': 'half_prox(np.array([[-2.,0.],[.1,7.]]), 0., 2.)',
      'gold_call': '_oracle_half_prox(np.array([[-2.,0.],[.1,7.]]), 0., 2.)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n',
      'call': 'half_prox(np.array(-.4), .003, 2.)',
      'gold_call': '_oracle_half_prox(np.array(-.4), .003, 2.)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n',
      'call': 'half_prox(np.array([-1e6,2e5]), 3., .02)',
      'gold_call': '_oracle_half_prox(np.array([-1e6,2e5]), 3., .02)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'Y=np.array([[[0,1,0],[3,5,4],[0,0,0]],[[2,1,3],[0,0,1],[4,3,5]],[[6,4,5],[1,0,1],[0,1,0]],[[0,0,0],[2,4,3],[5,6,4]]],dtype=float)\n'
               'S=Y.sum(axis=2)\n'
               'x=np.array([-.9,-.2,.4,1.]); z=np.array([-.8,.1,.9])\n'
               'Z=np.array([[1+.2*xi,.5+.1*zj] for xi in x for zj in z])\n'
               'alpha=np.array([.4,.3]); weights=np.array([.8,1.1,.9]); rho=np.array([.08,.06,.1])\n'
               'U=np.array([[1.,.4],[.2,1.1],[.8,.3],[.4,.7]])\n'
               'V=np.array([[.6,.8],[1.2,.2],[.3,1.]])\n'
               'P=np.full((4,3),.6)\n'
               'products=(U@U.T,U@V.T,V@V.T)\n'
               'A=tuple(a.copy() for a in products)\n'
               'W=tuple(np.zeros_like(a) for a in products)\n'
               'B=tuple(a+.2 for a in products)\n'
               '\n'
               'def _exception_code(function,*args,**kwargs):\n'
               '    try:\n'
               '        function(*args,**kwargs)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_exception_code(half_prox, np.array([2.]), 1., 0.)',
      'gold_call': '_exception_code(_oracle_half_prox, np.array([2.]), 1., 0.)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n',
      'call': 'bool(np.array_equal(half_prox(np.array([-1.4,1.4]),1.,1.),np.zeros(2)))',
      'gold_call': 'bool(np.array_equal(_oracle_half_prox(np.array([-1.4,1.4]),1.,1.),np.zeros(2)))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n',
      'call': 'bool(np.allclose(half_prox(np.array([-4.5,4.5]),2.,1.),np.array([-4.,4.]),rtol=1e-12,atol=0))',
      'gold_call': 'bool(np.allclose(_oracle_half_prox(np.array([-4.5,4.5]),2.,1.),np.array([-4.,4.]),rtol=1e-12,atol=0))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               's=np.array([1e-160,1e-80,1.,1e80,1e160])\n'
               'b=np.stack((-4.5*s,4.5*s))\n'
               'w=2.*s**1.5\n',
      'call': 'bool(np.allclose(half_prox(b,w,1.)/s,np.array([[-4.], [4.]]),rtol=1e-9,atol=0))',
      'gold_call': 'bool(np.allclose(_oracle_half_prox(b,w,1.)/s,np.array([[-4.], '
                   '[4.]]),rtol=1e-9,atol=0))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'b=np.array([[0.,4.5,-4.5],[1.,4.5,-4.5]])\n'
               'w=np.array([[0.],[2.]])\n'
               'rho=np.array([1.,1.,2.])\n',
      'call': 'half_prox(b,w,rho)',
      'gold_call': '_oracle_half_prox(b,w,rho)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nb=np.array([[-2.,0.,4.5],[-2.,0.,4.5]])\n',
      'call': 'bool(np.array_equal(half_prox(b,np.zeros((2,1)),np.array([1.,2.,3.])),b))',
      'gold_call': 'bool(np.array_equal(_oracle_half_prox(b,np.zeros((2,1)),np.array([1.,2.,3.])),b))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n',
      'call': 'bool(np.allclose(half_prox(np.array([-1e200,1e200]),1.,1.)/1e200,np.array([-1.,1.]),rtol=1e-12,atol=0))',
      'gold_call': 'bool(np.allclose(_oracle_half_prox(np.array([-1e200,1e200]),1.,1.)/1e200,np.array([-1.,1.]),rtol=1e-12,atol=0))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n',
      'call': 'bool(np.allclose(half_prox(np.array([-4.5e-160,4.5e-160]),2e-240,1.)/1e-160,np.array([-4.,4.]),rtol=1e-9,atol=0))',
      'gold_call': 'bool(np.allclose(_oracle_half_prox(np.array([-4.5e-160,4.5e-160]),2e-240,1.)/1e-160,np.array([-4.,4.]),rtol=1e-9,atol=0))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\nb=np.array([-2.,-.2,0.,.2,2.])\n',
      'call': 'bool(np.allclose(half_prox(np.array([-4.5,4.5]),2e120,1e120),np.array([-4.,4.]),rtol=1e-12,atol=0))',
      'gold_call': 'bool(np.allclose(_oracle_half_prox(np.array([-4.5,4.5]),2e120,1e120),np.array([-4.,4.]),rtol=1e-12,atol=0))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _exception_code(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_exception_code(half_prox,np.ones((2,3)),np.ones((2,)),1.)',
      'gold_call': '_exception_code(_oracle_half_prox,np.ones((2,3)),np.ones((2,)),1.)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _exception_code(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_exception_code(half_prox,np.ones((1,3)),np.ones((2,3)),1.)',
      'gold_call': '_exception_code(_oracle_half_prox,np.ones((1,3)),np.ones((2,3)),1.)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _exception_code(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_exception_code(half_prox,np.ones((2,3)),np.array([[1.],[-1.]]),1.)',
      'gold_call': '_exception_code(_oracle_half_prox,np.ones((2,3)),np.array([[1.],[-1.]]),1.)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _exception_code(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_exception_code(half_prox,np.ones((2,3)),1.,np.array([1.,0.,2.]))',
      'gold_call': '_exception_code(_oracle_half_prox,np.ones((2,3)),1.,np.array([1.,0.,2.]))',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _exception_code(function, *args):\n'
               '    try:\n'
               '        function(*args)\n'
               '    except ValueError:\n'
               '        return 1\n'
               '    except Exception:\n'
               '        return 2\n'
               '    return 0\n',
      'call': '_exception_code(half_prox,np.ones(2),1e-300,1e300)',
      'gold_call': '_exception_code(_oracle_half_prox,np.ones(2),1e-300,1e300)',
      'tol': 1e-09},
     {'setup': 'import numpy as np\n'
               'def _check_prox_inputs(fn):\n'
               '    b=np.array([[4.5,-4.5],[2.,-2.]])\n'
               '    w=np.array([[2.],[0.]])\n'
               '    r=np.array([1.,2.])\n'
               '    before=[a.copy() for a in (b,w,r)]\n'
               '    out=fn(b,w,r)\n'
               '    return bool(out.shape==b.shape and not np.shares_memory(out,b)\n'
               '                and all(np.array_equal(a,z) for a,z in zip((b,w,r),before)))\n',
      'call': '_check_prox_inputs(half_prox)',
      'gold_call': '_check_prox_inputs(_oracle_half_prox)',
      'tol': 1e-09}]

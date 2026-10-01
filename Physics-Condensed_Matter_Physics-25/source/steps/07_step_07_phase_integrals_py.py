"""
Integrate local protection over every nontrivial phase interval.

Paper Section II supplies the oriented Pfaffian phase and local gap. The coupling integral is a constructed observable. Split at supplied distinct closures; interval parity is measured at its midpoint, including across tangencies. In each nontrivial interval return its order-point Gauss–Legendre sum of the local gap. This function composes paired_ordering, skew_factor and pfaffian_certificate; every computed node contributes through its physical gap.

Returns
-------
return result  # real ndarray (K+1,4), rows in increasing c order; columns lower endpoint, upper endpoint, oriented interval sign, integrated gap contribution. The final column is zero in trivial intervals. Units are c,c,dimensionless,energy*c.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def phase_integrals(pencil: ArrayLike, crossings: ArrayLike, lower: float, upper: float, orientation: int = 1, order: int = 64) -> np.ndarray:
    'Integrate local protection over every nontrivial phase interval.\n\nParameters\n----------\npencil : real (2,n,n), ordered S0,S1.\ncrossings : real (n/2+1,), count K, K sorted interior closures, zero padding as in pencil_crossings.\nlower, upper : finite increasing bounds.\norientation : +1 or -1, sign of the trivial reference.\norder : positive integer quadrature order, default 64.\n\nRaises\n------\nValueError for incompatible shapes, numeric types, nonfinite values, or the stated domain violations. Numeric array-like inputs are accepted; real inputs have real numeric type, complex input is allowed only for layer, and booleans, strings and object arrays are invalid. Integer parameters and index arrays have integer type; integer entries packed into real factor/crossing records are integer-valued. Both pencil matrices have positive even size and satisfy skew symmetry within relative max-entry tolerance 1e-12. The pencil obeys the regularity and root-separation domain of pencil_crossings. The crossing record has exactly shape (n/2+1,), an integer-valued count in [0,n/2], that many strictly increasing interior roots, and exact zero padding. It contains every distinct interior closure within root tolerance 1e-6.\n\nReturns\n-------\nreal ndarray (K+1,4), rows in increasing c order; columns lower endpoint, upper endpoint, oriented interval sign, integrated gap contribution. The final column is zero in trivial intervals. Units are c,c,dimensionless,energy*c.'
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigvals, eigvalsh

def _oracle_phase_integrals(pencil: ArrayLike, crossings: ArrayLike, lower: float, upper: float, orientation: int = 1, order: int = 64) -> np.ndarray:
    def _checked_interval(lower, upper):
        lo = _checked_scalar(lower, 'lower')
        hi = _checked_scalar(upper, 'upper')
        if lo >= hi:
            raise ValueError('bounds must be strictly increasing')
        return lo, hi

    def _checked_numeric(value, name, real=True):
        try:
            a = np.asarray(value)
            allowed = 'iuf' if real else 'iufc'
            if a.dtype.kind not in allowed or not np.isfinite(a).all():
                raise ValueError(name + ' must contain finite numeric values of the declared type')
            with np.errstate(over='ignore', invalid='ignore'):
                a = a.astype(float if real else complex)
            if not np.isfinite(a).all():
                raise ValueError(name + ' exceeds the supported floating-point range')
            return a
        except (TypeError, OverflowError) as exc:
            raise ValueError(name + ' must be a numeric scalar or array') from exc

    def _checked_orientation(orientation):
        sign = _checked_scalar(orientation, 'orientation', integer=True)
        if sign not in (-1, 1):
            raise ValueError('orientation must be integer +1 or -1')
        return sign

    def _checked_pencil(pencil):
        p = _checked_numeric(pencil, 'pencil')
        if p.ndim != 3 or p.shape[0] != 2:
            raise ValueError('pencil must have shape (2,n,n)')
        for a in p:
            _checked_skew(a)
        return p

    def _checked_scalar(value, name, minimum=None, strict=False, integer=False):
        a = _checked_numeric(value, name)
        if a.shape != ():
            raise ValueError(name + ' must be a scalar')
        x = float(a)
        if integer and (np.asarray(value).dtype.kind not in 'iu' or x != np.floor(x)):
            raise ValueError(name + ' must have integer type')
        if minimum is not None and (x < minimum or (strict and x == minimum)):
            raise ValueError(name + ' is outside its stated domain')
        return int(x) if integer else x

    def _checked_skew(skew):
        a = _checked_numeric(skew, 'skew')
        if a.ndim != 2 or a.shape[0] != a.shape[1] or len(a) == 0 or len(a) % 2:
            raise ValueError('skew must be square with positive even size')
        scale = float(np.max(abs(a)))
        if scale and np.max(abs(a / scale + a.T / scale)) > 1e-12:
            raise ValueError('matrix must be skew-symmetric to relative max-entry tolerance 1e-12')
        return a

    p=_checked_pencil(pencil);lower,upper=_checked_interval(lower,upper)
    orientation=_checked_orientation(orientation);order=_checked_scalar(order,'order',minimum=1,integer=True)
    crossings=_checked_numeric(crossings,'crossings');n=p.shape[1]
    if crossings.shape!=(n//2+1,):raise ValueError('crossings must have shape (n/2+1,)')
    count=crossings[0]
    if count!=np.floor(count) or count<0 or count>n//2:raise ValueError('crossing count must be an integer from 0 through n/2')
    count=int(count);roots=crossings[1:count+1]
    if np.any(crossings[count+1:]!=0) or np.any(roots<=lower) or np.any(roots>=upper) or np.any(np.diff(roots)<=0):
        raise ValueError('crossings must be increasing interior locations followed by zero padding')
    expected=_oracle_pencil_crossings(p,lower,upper)
    if count!=int(expected[0]) or np.any(abs(roots-expected[1:count+1])>1e-6):
        raise ValueError('crossing record must contain all distinct interior closures within 1e-6')
    knots=np.r_[lower,roots,upper]
    nodes,weights=np.polynomial.legendre.leggauss(int(order));out=np.zeros((count+1,4))
    for k,(lo,hi) in enumerate(zip(knots[:-1],knots[1:])):
        mid=(lo+hi)/2;s=p[0]+mid*p[1]
        perm=_oracle_paired_ordering(s);f=_oracle_skew_factor(s,perm)
        sg=_oracle_pfaffian_certificate(s,f,orientation)[0]
        val=0.
        if sg<0:
            for node,weight in zip(nodes,weights):
                c=mid+(hi-lo)*node/2;s=p[0]+c*p[1]
                cert=_oracle_pfaffian_certificate(s,_oracle_skew_factor(s,perm),orientation)
                val+=weight*cert[2]
            val*=(hi-lo)/2
        out[k]=[lo,hi,sg,val]
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, 1.0], [-1.0, 0.0]], [[0.0, 1.0], [-1.0, 0.0]]],dtype=float)\n'
               'r=np.array([0.0, 0.0],dtype=float)',
      'call': 'phase_integrals(p,r,0.0,1.0,1,8)',
      'gold_call': '_oracle_phase_integrals(p,r,0.0,1.0,1,8)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -1.0], [1.0, 0.0]], [[0.0, 0.2], [-0.2, 0.0]]],dtype=float)\n'
               'r=np.array([0.0, 0.0],dtype=float)',
      'call': 'phase_integrals(p,r,0.0,1.0,1,8)',
      'gold_call': '_oracle_phase_integrals(p,r,0.0,1.0,1,8)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -0.4], [0.4, 0.0]], [[0.0, 1.0], [-1.0, 0.0]]],dtype=float)\n'
               'r=np.array([1.0, 0.4],dtype=float)',
      'call': 'phase_integrals(p,r,0.0,1.0,1,12)',
      'gold_call': '_oracle_phase_integrals(p,r,0.0,1.0,1,12)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -0.2, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -0.7], [0.0, '
               '0.0, 0.7, 0.0]], [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], '
               '[0.0, 0.0, -1.0, 0.0]]],dtype=float)\n'
               'r=np.array([2.0, 0.2, 0.7],dtype=float)',
      'call': 'phase_integrals(p,r,0.0,1.0,1,16)',
      'gold_call': '_oracle_phase_integrals(p,r,0.0,1.0,1,16)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -0.4, 0.0, 0.0], [0.4, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, -0.4], [0.0, '
               '0.0, 0.4, 0.0]], [[0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], '
               '[0.0, 0.0, -1.0, 0.0]]],dtype=float)\n'
               'r=np.array([1.0, 0.4, 0.0],dtype=float)',
      'call': 'phase_integrals(p,r,0.0,1.0,1,16)',
      'gold_call': '_oracle_phase_integrals(p,r,0.0,1.0,1,16)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, 0.4, 0.0, 0.0], [-0.4, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0], [0.0, '
               '0.0, -2.0, 0.0]], [[0.0, 0.0, 0.0, 0.0], [-0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], '
               '[0.0, 0.0, -0.0, 0.0]]],dtype=float)\n'
               'r=np.array([0.0, 0.0, 0.0],dtype=float)',
      'call': 'phase_integrals(p,r,0.0,0.7,-1,8)',
      'gold_call': '_oracle_phase_integrals(p,r,0.0,0.7,-1,8)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -0.2, 0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, '
               '0.0, 0.0, -0.7, 0.0, 0.0], [0.0, 0.0, 0.7, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, '
               '1.0], [0.0, 0.0, 0.0, 0.0, -1.0, 0.0]], [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [-1.0, 0.0, '
               '0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0, 0.0, 0.0], '
               '[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, -0.0, 0.0]]],dtype=float)\n'
               'r=np.array([2.0, 0.2, 0.7, 0.0],dtype=float)',
      'call': 'phase_integrals(p,r,0.0,1.0,-1,24)',
      'gold_call': '_oracle_phase_integrals(p,r,0.0,1.0,-1,24)'},
     {'setup': 'import numpy as np\n'
               'p=np.array([[[0.0, -0.3, 0.0, 0.24], [0.3, 0.0, 0.39, 0.0], [0.0, -0.39, 0.0, 1.312], '
               '[-0.24, 0.0, -1.312, 0.0]], [[0.0, 1.0, 0.0, -0.8], [-1.0, 0.0, -1.3, 0.0], [0.0, 1.3, '
               '0.0, -1.04], [0.8, 0.0, 1.04, 0.0]]],dtype=float)\n'
               'r=np.array([1.0, 0.3, 0.0],dtype=float)',
      'call': 'phase_integrals(p,r,-0.2,0.8,1,32)',
      'gold_call': '_oracle_phase_integrals(p,r,-0.2,0.8,1,32)'},
     {'setup': 'import numpy as np\n'
               'z=np.zeros((2,2));i=np.eye(2)\n'
               'm=np.array([[-10000.,-0.0005],[0.0005,-10000.]])\n'
               'p=np.array([np.block([[z,m],[-m.T,z]]),np.block([[z,i],[-i,z]])])\n',
      'call': 'phase_integrals(p,np.array([0.,0.,0.]),9999.,10001.,1,8)',
      'gold_call': '_oracle_phase_integrals(p,np.array([0.,0.,0.]),9999.,10001.,1,8)'},
     {'setup': 'import numpy as np\n'
               'z=np.zeros((2,2));i=np.eye(2)\n'
               'm=np.array([[-10000.,-0.0005],[0.0005,-10000.]])\n'
               'p=np.array([np.block([[z,m],[-m.T,z]]),np.block([[z,i],[-i,z]])])\n'
               'def _raises_value_error(call):\n'
               '    try:\n'
               '        call()\n'
               '    except ValueError:\n'
               '        return 1.\n'
               '    return 0.\n',
      'call': '_raises_value_error(lambda: '
              'phase_integrals(p,np.array([1.,10000.,0.]),9999.,10001.,1,8))',
      'gold_call': '_raises_value_error(lambda: '
                   '_oracle_phase_integrals(p,np.array([1.,10000.,0.]),9999.,10001.,1,8))'},
     {'setup': 'import numpy as np\n'
               'j=np.array([[0.,1.],[-1.,0.]])\n'
               'z=np.zeros((2,2))\n'
               'p=np.array([np.block([[-.2*j,z],[z,-.2000005*j]]),np.block([[j,z],[z,j]])])\n'
               'def _raises_value_error(call):\n'
               '    try:\n'
               '        call()\n'
               '    except ValueError:\n'
               '        return 1.\n'
               '    return 0.\n',
      'call': '_raises_value_error(lambda: phase_integrals(p,np.array([1.,.20000025,0.]),0.,1.,1,8))',
      'gold_call': '_raises_value_error(lambda: '
                   '_oracle_phase_integrals(p,np.array([1.,.20000025,0.]),0.,1.,1,8))'},
     {'setup': 'import numpy as np\n'
               'j=np.array([[0.,1.],[-1.,0.]])\n'
               'z=np.zeros((2,2))\n'
               'p=np.array([np.block([[-5e-9*j,z],[z,-.4*j]]),np.block([[j,z],[z,j]])])\n'
               'def _raises_value_error(call):\n'
               '    try:\n'
               '        call()\n'
               '    except ValueError:\n'
               '        return 1.\n'
               '    return 0.\n',
      'call': '_raises_value_error(lambda: phase_integrals(p,np.array([1.,.4,0.]),0.,1.,1,8))',
      'gold_call': '_raises_value_error(lambda: '
                   '_oracle_phase_integrals(p,np.array([1.,.4,0.]),0.,1.,1,8))'},
     {'setup': 'import numpy as np\n'
               'z=np.zeros((2,2));i=np.eye(2)\n'
               'm=np.array([[-.3,-0.000005],[0.000005,-.3]])\n'
               'p=np.array([np.block([[z,m],[-m.T,z]]),np.block([[z,i],[-i,z]])])\n'
               'def _raises_value_error(call):\n'
               '    try:\n'
               '        call()\n'
               '    except ValueError:\n'
               '        return 1.\n'
               '    return 0.\n',
      'call': '_raises_value_error(lambda: phase_integrals(p,np.array([1.,.3,0.]),0.,1.,1,8))',
      'gold_call': '_raises_value_error(lambda: '
                   '_oracle_phase_integrals(p,np.array([1.,.3,0.]),0.,1.,1,8))'},
     {'setup': 'import numpy as np\n'
               's0=np.array([[0.,0.,.5,1.],[0.,0.,-.5,-.5],[-.5,.5,0.,0.],[-1.,.5,0.,0.]])\n'
               's1=np.array([[0.,0.,1.,0.],[0.,0.,1.,1.],[-1.,-1.,0.,0.],[0.,-1.,0.,0.]])\n'
               'p=np.stack((s0,s1))\n'
               'r=np.array([1.0, 0.5, 0.0],dtype=float)',
      'call': 'phase_integrals(p,r,0.0,1.0,1,16)',
      'gold_call': '_oracle_phase_integrals(p,r,0.0,1.0,1,16)'},
     {'setup': 'import numpy as np\n'
               's0=np.array([[0.,0.,.5,1.],[0.,0.,-.5,-.5],[-.5,.5,0.,0.],[-1.,.5,0.,0.]])\n'
               's1=np.array([[0.,0.,1.,0.],[0.,0.,1.,1.],[-1.,-1.,0.,0.],[0.,-1.,0.,0.]])\n'
               'p=np.stack((s0,s1))\n'
               'b=np.array([[1.,.5,0.,-.25],[0.,1.,.25,0.],[0.,0.,1.,.5],[0.,0.,0.,1.]])\n'
               'p=np.stack([b@s@b.T for s in p])\n'
               'r=np.array([1.0, 0.5, 0.0],dtype=float)',
      'call': 'phase_integrals(p,r,0.0,1.0,1,16)',
      'gold_call': '_oracle_phase_integrals(p,r,0.0,1.0,1,16)'},
     {'setup': 'import numpy as np\n'
               's0=np.array([[0.,0.,.5,1.],[0.,0.,-.5,-.5],[-.5,.5,0.,0.],[-1.,.5,0.,0.]])\n'
               's1=np.array([[0.,0.,1.,0.],[0.,0.,1.,1.],[-1.,-1.,0.,0.],[0.,-1.,0.,0.]])\n'
               'p=np.stack((s0,s1))\n'
               'r=np.array([0.0, 0.0, 0.0],dtype=float)',
      'call': 'phase_integrals(p,r,0.5,1.5,1,16)',
      'gold_call': '_oracle_phase_integrals(p,r,0.5,1.5,1,16)'},
     {'setup': 'import numpy as np\n'
               's0=np.array([[0.,0.,.5,1.],[0.,0.,-.5,-.5],[-.5,.5,0.,0.],[-1.,.5,0.,0.]])\n'
               's1=np.array([[0.,0.,1.,0.],[0.,0.,1.,1.],[-1.,-1.,0.,0.],[0.,-1.,0.,0.]])\n'
               'p=np.stack((s0,s1))\n'
               'j=np.array([[0.,1.],[-1.,0.]])\n'
               'q=np.zeros((2,6,6))\n'
               'q[:,:4,:4]=p\n'
               'q[0,4:,4:]=-.75*j\n'
               'q[1,4:,4:]=j\n'
               'p=q\n'
               'r=np.array([2.0, 0.5, 0.75, 0.0],dtype=float)',
      'call': 'phase_integrals(p,r,0.0,1.0,-1,32)',
      'gold_call': '_oracle_phase_integrals(p,r,0.0,1.0,-1,32)'}]

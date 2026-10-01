"""
Propagate an initially pressure-balanced density perturbation under the

autonomous flux generator of step 4. Initial pressure perturbation is zero;

the density perturbation must have zero arithmetic mean. The dynamics conserve

both means. Output is the induced pressure at the requested elapsed time.

Propagate an initially pressure-balanced density perturbation under the

autonomous flux generator of step 4. Initial pressure perturbation is zero;

the density perturbation must have zero arithmetic mean. The dynamics conserve

both means. Output is the induced pressure at the requested elapsed time.



Inputs: generator is a real (2\*N,2\*N) array in density-then-pressure order,

N>=2; time>=0. Inputs already satisfy the conservative physical construction.

Returns: out, an (N,N) real response matrix acting on an arbitrary density

input after removing its mean, with the output pressure mean also removed.

The final singular norm will therefore optimize only over allowed perturbations.

Returns
-------
out, an (N,N) real response matrix acting on an arbitrary density input after removing its mean, with the output pressure mean also removed. The final singular norm will therefore optimize only over allowed perturbations.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def pressure_response(generator: np.ndarray, time: float) -> np.ndarray:
    """Return the finite-time density-to-pressure response, with both means removed.

    generator is the (2*N,2*N) density-then-pressure evolution matrix and
    time is the nonnegative elapsed time in consistent units. Return an
    (N,N) real matrix from initial density to induced pressure, applying
    the stated zero-mean restrictions and zero initial pressure.

    Raise ValueError for non-square or odd-sized matrices, fewer than four
    rows, nonfinite inputs, negative time or a nonfinite propagated output.
    """
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import expm

def _oracle_pressure_response(generator: np.ndarray, time: float) -> np.ndarray:
    l=np.asarray(generator,dtype=float); t=np.asarray(time,dtype=float)
    if l.ndim!=2 or l.shape[0]!=l.shape[1] or len(l)<4 or len(l)%2 or t.ndim:
        raise ValueError('invalid generator shape')
    if not np.all(np.isfinite(l)) or not np.isfinite(t) or t<0:
        raise ValueError('invalid generator or time')
    n=len(l)//2
    p=np.eye(n)-np.ones((n,n))/n
    out=p@expm(float(t)*l)[n:,:n]@p
    if not np.all(np.isfinite(out)): raise ValueError('nonfinite response')
    return out

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    return [
        {'setup': "import numpy as np\nx,y=np.meshgrid(2*np.pi*np.arange(3)/3,2*np.pi*np.arange(5)/5,indexing='ij')\ncf=np.column_stack([v.ravel() for v in (.04*(1+.2*np.cos(x+y)),.3*np.sin(x),.1*np.cos(y),.5*np.cos(x),np.sin(y),.2*np.sin(2*x+y),.1*np.cos(x-y))])\nL=_oracle_flux_generator(cf,(3,5),(2*np.pi,2*np.pi))\n", 'call': 'pressure_response(L,3.0)', 'gold_call': '_oracle_pressure_response(L,3.0)'},
        {'setup': "import numpy as np\nx,y=np.meshgrid(2*np.pi*np.arange(3)/3,2*np.pi*np.arange(5)/5,indexing='ij')\ncf=np.column_stack([v.ravel() for v in (.04*(1+.2*np.cos(x+y)),.3*np.sin(x),.1*np.cos(y),.5*np.cos(x),np.sin(y),.2*np.sin(2*x+y),.1*np.cos(x-y))])\nL=_oracle_flux_generator(cf,(3,5),(2*np.pi,2*np.pi))\n", 'call': 'pressure_response(L,0.0)', 'gold_call': '_oracle_pressure_response(L,0.0)'},
        {'setup': "import numpy as np\nx,y=np.meshgrid(2*np.pi*np.arange(3)/3,2*np.pi*np.arange(5)/5,indexing='ij')\ncf=np.column_stack([v.ravel() for v in (.04*(1+.2*np.cos(x+y)),.3*np.sin(x),.1*np.cos(y),.5*np.cos(x),np.sin(y),.2*np.sin(2*x+y),.1*np.cos(x-y))])\nL=_oracle_flux_generator(cf,(3,5),(2*np.pi,2*np.pi))\nL[15:,:15]=0.\n", 'call': 'pressure_response(L,2.0)', 'gold_call': '_oracle_pressure_response(L,2.0)'},
        {'setup': "import numpy as np\nx,y=np.meshgrid(2*np.pi*np.arange(3)/3,2*np.pi*np.arange(5)/5,indexing='ij')\ncf=np.column_stack([v.ravel() for v in (.04*(1+.2*np.cos(x+y)),.3*np.sin(x),.1*np.cos(y),.5*np.cos(x),np.sin(y),.2*np.sin(2*x+y),.1*np.cos(x-y))])\nL=_oracle_flux_generator(cf,(3,5),(2*np.pi,2*np.pi))\n", 'call': 'pressure_response(L,17.0)', 'gold_call': '_oracle_pressure_response(L,17.0)'},
        {'setup': "import numpy as np\nx,y=np.meshgrid(2*np.pi*np.arange(3)/3,2*np.pi*np.arange(5)/5,indexing='ij')\ncf=np.column_stack([v.ravel() for v in (.04*(1+.2*np.cos(x+y)),.3*np.sin(x),.1*np.cos(y),.5*np.cos(x),np.sin(y),.2*np.sin(2*x+y),.1*np.cos(x-y))])\nL=_oracle_flux_generator(cf,(3,5),(2*np.pi,2*np.pi))\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n", 'call': '_status(pressure_response, L, -1.0)', 'gold_call': '_status(_oracle_pressure_response, L, -1.0)'},
    ]

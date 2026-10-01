"""
Construct the two-dimensional density-pressure evolution generator from the

tangent fluxes of step 3. Both fields obey their flux-only conservative balance

in the magnetic-surface frame; no source or parallel transport is retained.

Construct the two-dimensional density-pressure evolution generator from the

tangent fluxes of step 3. Both fields obey their flux-only conservative balance

in the magnetic-surface frame; no source or parallel transport is retained.



Inputs: coefficients has (N,7) rows [eta,a_x,a_y,h_x,h_y,c_x,c_y];

shape=(nx,ny), with odd nx,ny>=3 and N=nx\*ny; periods=(Lx,Ly)>0.

The rectangular periodic grid is flattened with index i\*ny+j, so y varies

fastest. Differentiation is that of the periodic tensor-product trigonometric

interpolant of degree (nx-1)/2 in x and (ny-1)/2 in y. Multiplication is

pointwise on the grid, after continuum linearization, without dealiasing.



Returns: out, the real (2\*N,2\*N) generator acting on stacked density then

pressure perturbations, with each field in the stated flattened grid order.

Returns
-------
out, the real (2*N,2*N) generator acting on stacked density then pressure perturbations, with each field in the stated flattened grid order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def flux_generator(coefficients: np.ndarray, shape: tuple, periods: tuple) -> np.ndarray:
    """Return the conservative tensor-grid generator.

    coefficients is (nx*ny,7), shape contains the two odd integer node counts,
    and periods contains the two positive lengths. Return a real square
    (2*nx*ny,2*nx*ny) matrix in density-then-pressure order.
    Raise ValueError for wrong shapes, nonfinite data, invalid node counts,
    negative eta or nonpositive periods.
    """
    return out

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_flux_generator(coefficients: np.ndarray, shape: tuple, periods: tuple) -> np.ndarray:
    cf=np.asarray(coefficients,dtype=float); dims=np.asarray(shape,dtype=float); lengths=np.asarray(periods,dtype=float)
    if dims.shape!=(2,) or lengths.shape!=(2,) or not np.all(np.isfinite(dims)) or not np.all(np.isfinite(lengths)):
        raise ValueError('invalid grid shape')
    if np.any(dims<3) or np.any(dims!=np.floor(dims)) or np.any(dims%2!=1) or np.any(lengths<=0):
        raise ValueError('invalid grid parameters')
    nx,ny=map(int,dims); n=nx*ny
    if cf.shape!=(n,7) or not np.all(np.isfinite(cf)) or np.any(cf[:,0]<0):
        raise ValueError('invalid coefficients')
    derivatives=[]
    for count,length in zip((nx,ny),lengths):
        k=2*np.pi*np.fft.fftfreq(count,d=length/count)
        derivatives.append(np.fft.ifft(1j*k[:,None]*np.fft.fft(np.eye(count),axis=0),axis=0).real)
    dx=np.kron(derivatives[0],np.eye(ny)); dy=np.kron(np.eye(nx),derivatives[1])
    rr=np.zeros((n,n)); pp=np.zeros((n,n)); pr=np.zeros((n,n))
    for axis,d in enumerate((dx,dy)):
        rr+=d@(cf[:,0,None]*(d-np.diag(cf[:,1+axis])))
        pp+=d@(cf[:,0,None]*(d-np.diag(cf[:,3+axis])))
        pr-=d@np.diag(cf[:,0]*cf[:,5+axis])
    out=np.block([[rr,np.zeros((n,n))],[pr,pp]])
    return out

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    return [
        {'setup': "import numpy as np\nx,y=np.meshgrid(2*np.pi*np.arange(3)/3,2*np.pi*np.arange(5)/5,indexing='ij')\ncf=np.column_stack([v.ravel() for v in (.03*(1+.2*np.cos(x+y)),np.sin(x),.3*np.cos(y),.4*np.cos(x-y),np.sin(y),.2*np.sin(x+2*y),.1*np.cos(2*x-y))])\n", 'call': 'flux_generator(cf,(3,5),(6.283185307179586, 6.283185307179586))', 'gold_call': '_oracle_flux_generator(cf,(3,5),(6.283185307179586, 6.283185307179586))'},
        {'setup': "import numpy as np\nx,y=np.meshgrid(2*np.pi*np.arange(5)/5,2*np.pi*np.arange(3)/3,indexing='ij')\ncf=np.column_stack([v.ravel() for v in (.03*(1+.2*np.cos(x+y)),np.sin(x),.3*np.cos(y),.4*np.cos(x-y),np.sin(y),.2*np.sin(x+2*y),.1*np.cos(2*x-y))])\ncf[:,5:]=0.\n", 'call': 'flux_generator(cf,(5,3),(2.3, 4.0))', 'gold_call': '_oracle_flux_generator(cf,(5,3),(2.3, 4.0))'},
        {'setup': "import numpy as np\nx,y=np.meshgrid(2*np.pi*np.arange(3)/3,2*np.pi*np.arange(3)/3,indexing='ij')\ncf=np.column_stack([v.ravel() for v in (.03*(1+.2*np.cos(x+y)),np.sin(x),.3*np.cos(y),.4*np.cos(x-y),np.sin(y),.2*np.sin(x+2*y),.1*np.cos(2*x-y))])\ncf[:,0]=0.\n", 'call': 'flux_generator(cf,(3,3),(1.7, 2.1))', 'gold_call': '_oracle_flux_generator(cf,(3,3),(1.7, 2.1))'},
        {'setup': "import numpy as np\nx,y=np.meshgrid(2*np.pi*np.arange(5)/5,2*np.pi*np.arange(7)/7,indexing='ij')\ncf=np.column_stack([v.ravel() for v in (.03*(1+.2*np.cos(x+y)),np.sin(x),.3*np.cos(y),.4*np.cos(x-y),np.sin(y),.2*np.sin(x+2*y),.1*np.cos(2*x-y))])\ncf[:,1:]=0.\n", 'call': 'flux_generator(cf,(5,7),(2.0, 3.0))', 'gold_call': '_oracle_flux_generator(cf,(5,7),(2.0, 3.0))'},
        {'setup': 'import numpy as np\ndef _status(fn, *args, **kwargs):\n    try:\n        fn(*args, **kwargs)\n    except ValueError:\n        return 1\n    return 0\n', 'call': '_status(flux_generator, np.ones((12, 7)), (3, 4), (2.0, 2.0))', 'gold_call': '_status(_oracle_flux_generator, np.ones((12, 7)), (3, 4), (2.0, 2.0))'},
    ]

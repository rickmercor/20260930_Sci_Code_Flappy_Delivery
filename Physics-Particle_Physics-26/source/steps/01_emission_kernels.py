"""
Construct the single-emission kernels, including the collinear rotation.

Use Eqs. (7), (13), (20)-(21) with the finite quadrature specified in the task. The three channels include their quadrature weights and the NLO factor 1/2. Terms with coincident endpoints or an emitted endpoint are zero.

Returns
-------
return result  # ndarray, shape (3,n,n,n), dimensionless; entries [channel,i,j,k]. Channels are weighted K0, weighted L, and weighted one-emission K1, in this order.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def emission_kernels(points: "np.ndarray", weights: "np.ndarray", mu2: float, nf: int = 3, nc: int = 3) -> "np.ndarray":
    """points : ndarray, shape (n,2), distinct transverse coordinates in GeV^-1, n >= 3.
    weights : ndarray, shape (n,), nonnegative weights for d^2x/(2*pi), in GeV^-2.
    mu2 : positive float, collinear scale squared in GeV^2.
    nf : nonnegative integer flavor count.
    nc : integer color count >= 2.

    Returns
    -------
    ndarray, shape (3,n,n,n), dimensionless; entries [channel,i,j,k].
    Channels are weighted K0, weighted L, and weighted one-emission K1, in this order.
    Inputs must be finite. Invalid dimensions, scales, or prescribed domains raise ValueError. 
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _geometry(points, weights):
    points=np.asarray(points,dtype=float);weights=np.asarray(weights,dtype=float)
    if points.ndim!=2 or points.shape[1]!=2 or len(points)<3 or weights.shape!=(len(points),):
        raise ValueError('Expected points (n,2) and weights (n,), n >= 3.')
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(weights)) or np.any(weights<0):
        raise ValueError('Coordinates and nonnegative quadrature weights must be finite.')
    d=np.sum((points[:,None,:]-points[None,:,:])**2,axis=2)
    if np.any(d[np.triu_indices(len(points),1)]<=0):raise ValueError('Nodes must be distinct.')
    return points,weights,d


def _oracle_emission_kernels(points: "np.ndarray", weights: "np.ndarray", mu2: float, nf: int = 3, nc: int = 3) -> "np.ndarray":
    """Weighted K_LO, L_LO and one-emission K_NLO; Eqs. 7, 13 and 21."""
    import numpy as np
    p,w,d=_geometry(points,weights);n=len(p)
    if not np.isfinite(mu2) or mu2<=0 or nc<2 or nf<0 or nc!=int(nc) or nf!=int(nf):raise ValueError('Invalid scale or color multiplicity.')
    out=np.zeros((3,n,n,n));constant=67/18-np.pi**2/6-5*nf/(9*nc)
    for i in range(n):
      for j in range(n):
       if i==j:continue
       for k in range(n):
        if k in (i,j):continue
        r,a,b=d[i,j],d[i,k],d[k,j];c=w[k]*r/(a*b)
        out[:,i,j,k]=c,c*np.log(mu2*np.sqrt(a*b)),.5*c*(constant-np.log(a/r)*np.log(b/r))
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\np=np.array([[-1.1, -0.35], [-0.55, 0.8], [0.15, -0.9], [0.65, 0.4], [1.25, -0.2], [-0.25, -0.1], [0.05, 1.25]], dtype=float)\nw=np.array([0.13, 0.11, 0.16, 0.12, 0.14, 0.09, 0.15], dtype=float)\n', 'call': 'emission_kernels(p,w,1.3,3,3)', 'gold_call': '_oracle_emission_kernels(p,w,1.3,3,3)', 'tol': 2e-08}, {'setup': 'import numpy as np\np=np.array([[0, 0], [2, 0], [1, 1], [1, -1]], dtype=float)\nw=np.array([0.1, 0.2, 0.15, 0.13], dtype=float)\n', 'call': 'emission_kernels(p,w,1.3,3,3)', 'gold_call': '_oracle_emission_kernels(p,w,1.3,3,3)', 'tol': 2e-08}, {'setup': 'import numpy as np\np=np.array([[0.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.00000002, -1.0]], dtype=float)\nw=np.array([0.11, 0.15, 0.12, 0.14], dtype=float)\n', 'call': 'emission_kernels(p,w,1.3,3,3)', 'gold_call': '_oracle_emission_kernels(p,w,1.3,3,3)', 'tol': 2e-08}, {'setup': 'import numpy as np\np=np.array([[0.0, 0.0], [0.3, 0.0], [0.9, 0.0], [1.7, 0.0], [2.4, 0.0]], dtype=float)\nw=np.array([0.02, 0.025, 0.03, 0.02, 0.025], dtype=float)\n', 'call': 'emission_kernels(p,w,0.1,3,3)', 'gold_call': '_oracle_emission_kernels(p,w,0.1,3,3)', 'tol': 2e-08}, {'setup': 'import numpy as np\np=np.array([[0.0, 0.0], [0.045, 0.02], [0.9, 0.4], [-0.4, 1.2], [1.5, -0.5]], dtype=float)\nw=np.array([0.0002, 0.0003, 0.002, 0.0015, 0.0025], dtype=float)\n', 'call': 'emission_kernels(p,w,1.3,3,3)', 'gold_call': '_oracle_emission_kernels(p,w,1.3,3,3)', 'tol': 2e-08}, {'setup': 'import numpy as np\np=np.array([[0.0, 0.0], [2.0, 0.0], [0.8, 0.7], [0.81, 0.705], [1.5, -0.4]], dtype=float)\nw=np.array([2e-05, 3e-05, 2.5e-05, 1.8e-05, 2e-05], dtype=float)\n', 'call': 'emission_kernels(p,w,1.3,3,3)', 'gold_call': '_oracle_emission_kernels(p,w,1.3,3,3)', 'tol': 2e-08}, {'setup': 'import numpy as np\np=np.array([[-1.1, -0.35], [-0.55, 0.8], [0.15, -0.9], [0.65, 0.4], [1.25, -0.2], [-0.25, -0.1], [0.05, 1.25]], dtype=float)\nw=np.array([0.13, 0.0, 0.16, 0.12, 0.14, 0.09, 0.15], dtype=float)\n', 'call': 'emission_kernels(p,w,1.3,3,3)', 'gold_call': '_oracle_emission_kernels(p,w,1.3,3,3)', 'tol': 2e-08}, {'setup': 'import numpy as np\np=np.array([[0.0, 0.0], [1.0, 0.0], [0.3, 0.8]], dtype=float)\nw=np.array([0.06, 0.04, 0.05], dtype=float)\n', 'call': 'emission_kernels(p,w,1.3,0,3)', 'gold_call': '_oracle_emission_kernels(p,w,1.3,0,3)', 'tol': 2e-08}, {'setup': 'import numpy as np\np=np.array([[0.0, 0.0], [0.7, 0.1], [8.0, 4.0], [-6.0, 9.0], [12.0, -3.0]], dtype=float)\nw=np.array([0.03, 0.02, 0.2, 0.3, 0.4], dtype=float)\n', 'call': 'emission_kernels(p,w,1.3,3,3)', 'gold_call': '_oracle_emission_kernels(p,w,1.3,3,3)', 'tol': 2e-08}, {'setup': 'import numpy as np\np=np.array([[1.75, -2.3], [2.9000000000000004, -2.8499999999999996], [1.2000000000000002, -3.55], [2.5, -4.05], [1.9000000000000001, -4.65], [2.0, -3.15], [3.35, -3.4499999999999997]], dtype=float)\nw=np.array([0.13, 0.11, 0.16, 0.12, 0.14, 0.09, 0.15], dtype=float)\n', 'call': 'emission_kernels(p,w,1.3,5,3)', 'gold_call': '_oracle_emission_kernels(p,w,1.3,5,3)', 'tol': 2e-08}, {'setup': 'import numpy as np\np=np.array([[-1.87, -0.595], [-0.935, 1.36], [0.255, -1.53], [1.105, 0.68], [2.125, -0.34], [-0.425, -0.17], [0.085, 2.125]], dtype=float)\nw=np.array([0.3757, 0.31789999999999996, 0.4624, 0.34679999999999994, 0.4046, 0.26009999999999994, 0.43349999999999994], dtype=float)\n', 'call': 'emission_kernels(p,w,0.4498269896193772,3,3)', 'gold_call': '_oracle_emission_kernels(p,w,0.4498269896193772,3,3)', 'tol': 2e-08}, {'setup': 'import numpy as np\np=np.array([[1.25, -0.2], [-0.55, 0.8], [0.05, 1.25], [-1.1, -0.35], [-0.25, -0.1], [0.15, -0.9], [0.65, 0.4]], dtype=float)\nw=np.array([0.14, 0.11, 0.15, 0.13, 0.09, 0.16, 0.12], dtype=float)\n', 'call': 'emission_kernels(p,w,1.3,3,5)', 'gold_call': '_oracle_emission_kernels(p,w,1.3,3,5)', 'tol': 2e-08}]

"""
Evaluate the projected thermal exponential through second order.

Two coordinate derivatives produce both ordered operator insertions and a direct Hamiltonian-curvature term. Degenerate eigenvalues require continuous divided differences; numerical energy shifts must not generate physical derivatives

Returns
-------
Return (e, de, d2e, log_scale), with shapes (d,d), (r,d,d), (r,r,d,d), and scalar.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sector_propagator_jet(h: "np.ndarray", dh: "np.ndarray", d2h: "np.ndarray", projector: "np.ndarray", b: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    """Return a projected exponential and its scaled first and second derivatives.
    
    Parameters
    ----------
    h : finite real symmetric array (d, d)
    dh : finite real symmetric arrays (r, d, d), r>=1
    d2h : finite real array (r, r, d, d)
        Hamiltonian coordinate derivatives. d2h[i,j]=d2h[j,i], and each
        matrix is symmetric. dh and d2h need not commute with h.
    projector : finite real orthogonal projector (d, d)
        Nonzero rank, constant in the coordinates, commuting with h,
        dh and d2h. Its eigenvalues greater than 0.5 define its range.
    b : positive float
        beta/p. The restricted spectral width of b*h is at most 500.
    
    Returns
    -------
    e : float array (d, d)
    de : float array (r, d, d)
    d2e : float array (r, r, d, d)
    log_scale : float
        On the projector range, let h=U diag(lambda) U.T, lambda0=min(lambda),
        c=-b*lambda0, and x_a=-b*(lambda_a-lambda0). Define A_i=-b*U.T*dh[i]*U
        and A_ij=-b*U.T*d2h[i,j]*U, using matrix products.
        In this eigenbasis e_ab=delta_ab*exp(x_a),
        de[i]_ab=exp[x_a,x_b]*(A_i)_ab, and
        d2e[i,j]_ab=exp[x_a,x_b]*(A_ij)_ab
          +sum_c exp[x_a,x_c,x_b]*((A_i)_ac*(A_j)_cb+(A_j)_ac*(A_i)_cb).
        exp[...] denotes a symmetric divided difference of exp. For
        distinct end knots f[x0,...,xk]=(f[x1,...,xk]-f[x0,...,xk-1])/(xk-x0),
        starting from f[x]=exp(x); coincident knots use continuous Hermite
        limits, including f[x,x]=exp(x) and f[x,x,x]=exp(x)/2.
        Evaluate near-coincident knots without cancellation. Return all
        matrices transformed back to the original coordinates, zero on
        the projector complement, and log_scale=c.
        de and d2e are exp(-c) times derivatives of the ORIGINAL exp(-b*h).
        Do not differentiate the coordinate-dependent numerical shift c.
        Include both ordered mixed insertions and the d2h contribution.
        Use analytic matrix derivatives, not coordinate finite differences.
        NumPy is the only numerical dependency. Inputs are not mutated.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sector_propagator_jet(h: "np.ndarray", dh: "np.ndarray", d2h: "np.ndarray", projector: "np.ndarray", b: float) -> "tuple[np.ndarray, np.ndarray, np.ndarray, float]":
    """Return a projected exponential and its scaled first and second derivatives.

    Parameters
    ----------
    h : finite real symmetric array (d, d)
    dh : finite real symmetric arrays (r, d, d), r>=1
    d2h : finite real array (r, r, d, d)
        Hamiltonian coordinate derivatives. d2h[i,j]=d2h[j,i], and each
        matrix is symmetric. dh and d2h need not commute with h.
    projector : finite real orthogonal projector (d, d)
        Nonzero rank, constant in the coordinates, commuting with h,
        dh and d2h. Its eigenvalues greater than 0.5 define its range.
    b : positive float
        beta/p. The restricted spectral width of b*h is at most 500.

    Returns
    -------
    e : float array (d, d)
    de : float array (r, d, d)
    d2e : float array (r, r, d, d)
    log_scale : float
        On the projector range, let h=U diag(lambda) U.T, lambda0=min(lambda),
        c=-b*lambda0, and x_a=-b*(lambda_a-lambda0). Define A_i=-b*U.T*dh[i]*U
        and A_ij=-b*U.T*d2h[i,j]*U, using matrix products.
        In this eigenbasis e_ab=delta_ab*exp(x_a),
        de[i]_ab=exp[x_a,x_b]*(A_i)_ab, and
        d2e[i,j]_ab=exp[x_a,x_b]*(A_ij)_ab
          +sum_c exp[x_a,x_c,x_b]*((A_i)_ac*(A_j)_cb+(A_j)_ac*(A_i)_cb).
        exp[...] denotes a symmetric divided difference of exp. For
        distinct end knots f[x0,...,xk]=(f[x1,...,xk]-f[x0,...,xk-1])/(xk-x0),
        starting from f[x]=exp(x); coincident knots use continuous Hermite
        limits, including f[x,x]=exp(x) and f[x,x,x]=exp(x)/2.
        Evaluate near-coincident knots without cancellation. Return all
        matrices transformed back to the original coordinates, zero on
        the projector complement, and log_scale=c.
        de and d2e are exp(-c) times derivatives of the ORIGINAL exp(-b*h).
        Do not differentiate the coordinate-dependent numerical shift c.
        Include both ordered mixed insertions and the d2h contribution.
        Use analytic matrix derivatives, not coordinate finite differences.
        NumPy is the only numerical dependency. Inputs are not mutated.
    """
    pv, pu = np.linalg.eigh(projector)
    v = pu[:, pv > .5]
    lam, rot = np.linalg.eigh(v.T @ h @ v)
    u = v @ rot
    x = -b*(lam-lam[0])

    def _first(a, z):
        gap = np.abs(a-z)
        factor = np.ones(np.broadcast_shapes(np.shape(a), np.shape(z)))
        np.divide(-np.expm1(-gap), gap, out=factor, where=gap != 0)
        return np.exp(np.maximum(a,z))*factor

    # For a short knot interval use the entire homogeneous-polynomial
    # series. The separated-knot recurrence otherwise has no small gap.
    a, c, z = np.broadcast_arrays(x[:,None,None],x[None,:,None],x[None,None,:])
    knots = np.sort(np.stack((a,c,z)),axis=0)
    low, mid, high = knots
    width = high-low
    second = np.empty_like(width)
    wide = width > .5
    second[wide] = (_first(mid[wide],high[wide])-_first(low[wide],mid[wide]))/width[wide]
    aa, bb = low[~wide]-high[~wide], mid[~wide]-high[~wide]
    homogeneous = np.ones_like(aa)
    a_power = np.ones_like(aa)
    inverse_factorial = .5
    series = .5*np.ones_like(aa)
    for degree in range(1,32):
        a_power *= aa
        homogeneous = bb*homogeneous+a_power
        inverse_factorial /= degree+2
        series += inverse_factorial*homogeneous
    second[~wide] = np.exp(high[~wide])*series
    kernel = _first(x[:,None],x[None,:])
    ai = -b*np.einsum('ka,ikl,lb->iab',u,dh,u)
    aij = -b*np.einsum('ka,ijkl,lb->ijab',u,d2h,u)
    de_reduced = kernel[None,:,:]*ai
    ordered = np.einsum('acb,iac,jcb->ijab',second,ai,ai)
    d2e_reduced = kernel[None,None,:,:]*aij + ordered + ordered.swapaxes(0,1)
    e = (u*np.exp(x)) @ u.T
    de = np.einsum('ak,ikl,bl->iab',u,de_reduced,u)
    d2e = np.einsum('ak,ijkl,bl->ijab',u,d2e_reduced,u)
    return e, de, d2e, float(-b*lam[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(743); m=3; d=3; r=2; b=1.3\n'
               'w,_=np.linalg.qr(rng.normal(size=(d,d)))\n'
               'v=w[:,:m]\n'
               'h=v@np.diag((np.array([0.0, 0.4, 1.3])+(0.0))/b)@v.T\n'
               'projector=v@v.T\n'
               'x=rng.normal(size=(r,m,m)); a=(x+x.swapaxes(-1,-2))/2\n'
               'dh=np.einsum("ak,ikl,bl->iab",v,a,v)\n'
               'x=rng.normal(size=(r,r,m,m)); x=(x+x.swapaxes(0,1))/2; x=(x+x.swapaxes(2,3))/2\n'
               'd2h=.2*np.einsum("ak,ijkl,bl->ijab",v,x,v)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in sector_propagator_jet(h, dh, d2h, projector, b)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_sector_propagator_jet(h, dh, d2h, projector, b)])',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(743); m=3; d=3; r=2; b=1.3\n'
               'w,_=np.linalg.qr(rng.normal(size=(d,d)))\n'
               'v=w[:,:m]\n'
               'h=v@np.diag((np.array([0.0, 0.0, 0.0])+(0.0))/b)@v.T\n'
               'projector=v@v.T\n'
               'x=rng.normal(size=(r,m,m)); a=(x+x.swapaxes(-1,-2))/2\n'
               'dh=np.einsum("ak,ikl,bl->iab",v,a,v)\n'
               'x=rng.normal(size=(r,r,m,m)); x=(x+x.swapaxes(0,1))/2; x=(x+x.swapaxes(2,3))/2\n'
               'd2h=.2*np.einsum("ak,ijkl,bl->ijab",v,x,v)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in sector_propagator_jet(h, dh, d2h, projector, b)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_sector_propagator_jet(h, dh, d2h, projector, b)])',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(743); m=3; d=3; r=2; b=1.3\n'
               'w,_=np.linalg.qr(rng.normal(size=(d,d)))\n'
               'v=w[:,:m]\n'
               'h=v@np.diag((np.array([0.0, 1e-12, 1.3])+(0.0))/b)@v.T\n'
               'projector=v@v.T\n'
               'x=rng.normal(size=(r,m,m)); a=(x+x.swapaxes(-1,-2))/2\n'
               'dh=np.einsum("ak,ikl,bl->iab",v,a,v)\n'
               'x=rng.normal(size=(r,r,m,m)); x=(x+x.swapaxes(0,1))/2; x=(x+x.swapaxes(2,3))/2\n'
               'd2h=.2*np.einsum("ak,ijkl,bl->ijab",v,x,v)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in sector_propagator_jet(h, dh, d2h, projector, b)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_sector_propagator_jet(h, dh, d2h, projector, b)])',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(743); m=3; d=3; r=2; b=1.3\n'
               'w,_=np.linalg.qr(rng.normal(size=(d,d)))\n'
               'v=w[:,:m]\n'
               'h=v@np.diag((np.array([0.0, 1e-10, 3e-10])+(0.0))/b)@v.T\n'
               'projector=v@v.T\n'
               'x=rng.normal(size=(r,m,m)); a=(x+x.swapaxes(-1,-2))/2\n'
               'dh=np.einsum("ak,ikl,bl->iab",v,a,v)\n'
               'x=rng.normal(size=(r,r,m,m)); x=(x+x.swapaxes(0,1))/2; x=(x+x.swapaxes(2,3))/2\n'
               'd2h=.2*np.einsum("ak,ijkl,bl->ijab",v,x,v)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in sector_propagator_jet(h, dh, d2h, projector, b)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_sector_propagator_jet(h, dh, d2h, projector, b)])',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(743); m=2; d=4; r=2; b=1.3\n'
               'w,_=np.linalg.qr(rng.normal(size=(d,d)))\n'
               'v=w[:,:m]\n'
               'h=v@np.diag((np.array([0.0, 0.6])+(0.0))/b)@v.T\n'
               'h=h+w[:,m:]@np.diag(np.array([-40.,-80.])+(0.0)/b)@w[:,m:].T\n'
               'projector=v@v.T\n'
               'x=rng.normal(size=(r,m,m)); a=(x+x.swapaxes(-1,-2))/2\n'
               'dh=np.einsum("ak,ikl,bl->iab",v,a,v)\n'
               'x=rng.normal(size=(r,r,m,m)); x=(x+x.swapaxes(0,1))/2; x=(x+x.swapaxes(2,3))/2\n'
               'd2h=.2*np.einsum("ak,ijkl,bl->ijab",v,x,v)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in sector_propagator_jet(h, dh, d2h, projector, b)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_sector_propagator_jet(h, dh, d2h, projector, b)])',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(743); m=3; d=3; r=2; b=1.3\n'
               'w,_=np.linalg.qr(rng.normal(size=(d,d)))\n'
               'v=w[:,:m]\n'
               'h=v@np.diag((np.array([0.0, 0.2, 1.3])+(-1000.0))/b)@v.T\n'
               'projector=v@v.T\n'
               'x=rng.normal(size=(r,m,m)); a=(x+x.swapaxes(-1,-2))/2\n'
               'dh=np.einsum("ak,ikl,bl->iab",v,a,v)\n'
               'x=rng.normal(size=(r,r,m,m)); x=(x+x.swapaxes(0,1))/2; x=(x+x.swapaxes(2,3))/2\n'
               'd2h=.2*np.einsum("ak,ijkl,bl->ijab",v,x,v)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in sector_propagator_jet(h, dh, d2h, projector, b)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_sector_propagator_jet(h, dh, d2h, projector, b)])',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(743); m=3; d=3; r=2; b=1.3\n'
               'w,_=np.linalg.qr(rng.normal(size=(d,d)))\n'
               'v=w[:,:m]\n'
               'h=v@np.diag((np.array([0.0, 130.0, 500.0])+(1000.0))/b)@v.T\n'
               'projector=v@v.T\n'
               'x=rng.normal(size=(r,m,m)); a=(x+x.swapaxes(-1,-2))/2\n'
               'dh=np.einsum("ak,ikl,bl->iab",v,a,v)\n'
               'x=rng.normal(size=(r,r,m,m)); x=(x+x.swapaxes(0,1))/2; x=(x+x.swapaxes(2,3))/2\n'
               'd2h=.2*np.einsum("ak,ijkl,bl->ijab",v,x,v)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in sector_propagator_jet(h, dh, d2h, projector, b)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_sector_propagator_jet(h, dh, d2h, projector, b)])',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(743); m=3; d=3; r=2; b=1.3\n'
               'w,_=np.linalg.qr(rng.normal(size=(d,d)))\n'
               'v=w[:,:m]\n'
               'h=v@np.diag((np.array([0.0, 0.4, 1.3])+(0.0))/b)@v.T\n'
               'projector=v@v.T\n'
               'x=rng.normal(size=(r,m,m)); a=(x+x.swapaxes(-1,-2))/2\n'
               'dh=np.einsum("ak,ikl,bl->iab",v,a,v)\n'
               'x=rng.normal(size=(r,r,m,m)); x=(x+x.swapaxes(0,1))/2; x=(x+x.swapaxes(2,3))/2\n'
               'd2h=.2*np.einsum("ak,ijkl,bl->ijab",v,x,v)\n'
               'dh=np.zeros_like(dh)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in sector_propagator_jet(h, dh, d2h, projector, b)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_sector_propagator_jet(h, dh, d2h, projector, b)])',
      'tol': 1e-08},
     {'setup': 'import numpy as np\n'
               'rng=np.random.default_rng(743); m=1; d=3; r=2; b=1.3\n'
               'w,_=np.linalg.qr(rng.normal(size=(d,d)))\n'
               'v=w[:,:m]\n'
               'h=v@np.diag((np.array([0.0])+(-1.7))/b)@v.T\n'
               'h=h+w[:,m:]@np.diag(np.array([-40.,-80.])+(-1.7)/b)@w[:,m:].T\n'
               'projector=v@v.T\n'
               'x=rng.normal(size=(r,m,m)); a=(x+x.swapaxes(-1,-2))/2\n'
               'dh=np.einsum("ak,ikl,bl->iab",v,a,v)\n'
               'x=rng.normal(size=(r,r,m,m)); x=(x+x.swapaxes(0,1))/2; x=(x+x.swapaxes(2,3))/2\n'
               'd2h=.2*np.einsum("ak,ijkl,bl->ijab",v,x,v)\n',
      'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in sector_propagator_jet(h, dh, d2h, projector, b)])',
      'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in _oracle_sector_propagator_jet(h, dh, d2h, projector, b)])',
      'tol': 1e-08}]

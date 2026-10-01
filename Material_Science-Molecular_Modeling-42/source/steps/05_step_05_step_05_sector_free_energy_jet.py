"""
Coordinates on one bead differentiate the same exponential twice. Coordinates on different beads replace two ordered factors. Taking the logarithm subtracts the disconnected product of first derivatives.

The cyclic thermal trace couples imaginary-time slices. Each bead force replaces one factor by its derivative; the remaining factors retain their cyclic order. Common exponential scales cancel from the force ratio.

Returns
-------
Return (log_q, force, hessian), with shapes scalar, (p,n), and (p*n,p*n).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sector_free_energy_jet(e: "np.ndarray", de: "np.ndarray", d2e: "np.ndarray", log_scales: "np.ndarray", b: float) -> "tuple[float, np.ndarray, np.ndarray]":
    """Contract the ordered bead product into sector forces and force constants.
    
    Parameters
    ----------
    e : finite real array (p, d, d)
    de : finite real array (p, n, d, d)
    d2e : finite real array (p, n, n, d, d)
    log_scales : finite real array (p,)
        Outputs of sector_propagator_jet for one common spin sector.
        p=1,...,8. The scaled ordered trace z=Tr(e[0]@...@e[p-1]) is
        positive and at least 1e-250; all required ratios remain finite.
    b : positive float
        beta/p. Coordinate a=t*n+i uses bead-major order.
    
    Returns
    -------
    log_q : float
        sum(log_scales)+log(z), without exponentiating the scale sum.
    force : float array (p, n)
    hessian : float array (p*n, p*n)
        Define z_a by replacing e[t] by de[t,i] in the ordered trace.
        For a=t*n+i, bcoord=t*n+j on the same bead, z_ab replaces that
        one factor by d2e[t,i,j]. For different beads, replace both
        corresponding factors by their de matrices, retaining bead order.
        Then force_a=z_a/(b*z) and
        hessian_ab=-(z_ab/z-z_a*z_b/z**2)/b.
        The Hessian is d2(-log_q/b)/dq_a dq_b, hence minus the force
        Jacobian. No spin multiplicity is included here. Empty products
        are identity, including p=1. Do not symmetrize a bead product or
        replace its trace by a product of traces. Input arrays and their
        numerical scales are not differentiated or mutated.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_sector_free_energy_jet(e: "np.ndarray", de: "np.ndarray", d2e: "np.ndarray", log_scales: "np.ndarray", b: float) -> "tuple[float, np.ndarray, np.ndarray]":
    """Contract the ordered bead product into sector forces and force constants.

    Parameters
    ----------
    e : finite real array (p, d, d)
    de : finite real array (p, n, d, d)
    d2e : finite real array (p, n, n, d, d)
    log_scales : finite real array (p,)
        Outputs of sector_propagator_jet for one common spin sector.
        p=1,...,8. The scaled ordered trace z=Tr(e[0]@...@e[p-1]) is
        positive and at least 1e-250; all required ratios remain finite.
    b : positive float
        beta/p. Coordinate a=t*n+i uses bead-major order.

    Returns
    -------
    log_q : float
        sum(log_scales)+log(z), without exponentiating the scale sum.
    force : float array (p, n)
    hessian : float array (p*n, p*n)
        Define z_a by replacing e[t] by de[t,i] in the ordered trace.
        For a=t*n+i, bcoord=t*n+j on the same bead, z_ab replaces that
        one factor by d2e[t,i,j]. For different beads, replace both
        corresponding factors by their de matrices, retaining bead order.
        Then force_a=z_a/(b*z) and
        hessian_ab=-(z_ab/z-z_a*z_b/z**2)/b.
        The Hessian is d2(-log_q/b)/dq_a dq_b, hence minus the force
        Jacobian. No spin multiplicity is included here. Empty products
        are identity, including p=1. Do not symmetrize a bead product or
        replace its trace by a product of traces. Input arrays and their
        numerical scales are not differentiated or mutated.
    """
    p,n = de.shape[:2]
    d = e.shape[1]
    product = np.eye(d)
    for a in e:
        product = product @ a
    z = float(np.trace(product))
    first = np.empty((p,n))
    second = np.zeros((p*n,p*n))
    for t in range(p):
        cyclic = np.eye(d)
        for offset in range(1,p):
            cyclic = cyclic @ e[(t+offset)%p]
        first[t] = np.einsum('iab,ba->i',de[t],cyclic)/z
        sl = slice(t*n,(t+1)*n)
        second[sl,sl] = np.einsum('ijab,ba->ij',d2e[t],cyclic)/z
        for other in range(t+1,p):
            middle = np.eye(d)
            for k in range(t+1,other):
                middle = middle @ e[k]
            tail = np.eye(d)
            for k in list(range(other+1,p))+list(range(t)):
                tail = tail @ e[k]
            block = np.einsum('iab,jba->ij',de[t]@middle,de[other]@tail)/z
            sr = slice(other*n,(other+1)*n)
            second[sl,sr] = block
            second[sr,sl] = block.T
    gradient = first.reshape(-1)
    hessian = -(second-np.outer(gradient,gradient))/b
    return float(np.sum(log_scales)+np.log(z)), first/b, hessian

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
          'sector_free_energy_jet(e, de, d2e, scales, b)])',
  'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
               '_oracle_sector_free_energy_jet(e, de, d2e, scales, b)])',
  'setup': 'import numpy as np\n'
           'def _fx_sector_propagator_jet(h, dh, d2h, projector, b):\n'
           '    pv, pu = np.linalg.eigh(projector)\n'
           '    v = pu[:, pv > .5]\n'
           '    lam, rot = np.linalg.eigh(v.T @ h @ v)\n'
           '    u = v @ rot\n'
           '    x = -b*(lam-lam[0])\n'
           '\n'
           '    def _first(a, z):\n'
           '        gap = np.abs(a-z)\n'
           '        factor = np.ones(np.broadcast_shapes(np.shape(a), np.shape(z)))\n'
           '        np.divide(-np.expm1(-gap), gap, out=factor, where=gap != 0)\n'
           '        return np.exp(np.maximum(a,z))*factor\n'
           '\n'
           '    a, c, z = np.broadcast_arrays(x[:,None,None],x[None,:,None],x[None,None,:])\n'
           '    knots = np.sort(np.stack((a,c,z)),axis=0)\n'
           '    low, mid, high = knots\n'
           '    width = high-low\n'
           '    second = np.empty_like(width)\n'
           '    wide = width > .5\n'
           '    second[wide] = '
           '(_first(mid[wide],high[wide])-_first(low[wide],mid[wide]))/width[wide]\n'
           '    aa, bb = low[~wide]-high[~wide], mid[~wide]-high[~wide]\n'
           '    homogeneous = np.ones_like(aa)\n'
           '    a_power = np.ones_like(aa)\n'
           '    inverse_factorial = .5\n'
           '    series = .5*np.ones_like(aa)\n'
           '    for degree in range(1,32):\n'
           '        a_power *= aa\n'
           '        homogeneous = bb*homogeneous+a_power\n'
           '        inverse_factorial /= degree+2\n'
           '        series += inverse_factorial*homogeneous\n'
           '    second[~wide] = np.exp(high[~wide])*series\n'
           '    kernel = _first(x[:,None],x[None,:])\n'
           "    ai = -b*np.einsum('ka,ikl,lb->iab',u,dh,u)\n"
           "    aij = -b*np.einsum('ka,ijkl,lb->ijab',u,d2h,u)\n"
           '    de_reduced = kernel[None,:,:]*ai\n'
           "    ordered = np.einsum('acb,iac,jcb->ijab',second,ai,ai)\n"
           '    d2e_reduced = kernel[None,None,:,:]*aij + ordered + ordered.swapaxes(0,1)\n'
           '    e = (u*np.exp(x)) @ u.T\n'
           "    de = np.einsum('ak,ikl,bl->iab',u,de_reduced,u)\n"
           "    d2e = np.einsum('ak,ijkl,bl->ijab',u,d2e_reduced,u)\n"
           '    return e, de, d2e, float(-b*lam[0])\n'
           'p=4; n=2; b=.7\n'
           'h=np.array([[[.2*t,.1+.13*t],[.1+.13*t,-.3-.08*t]] for t in range(p)])\n'
           'dh=np.array([[[.3+.2*i,.4-.1*i],[.4-.1*i,-.5+.1*i]] for i in range(n)])\n'
           'd2h=np.array([[np.array([[.1*(i+j+1),.05],[.05,-.07]]) for j in range(n)] for i in '
           'range(n)])\n'
           'jets=[_fx_sector_propagator_jet(a,dh,d2h,np.eye(2),b) for a in h]\n'
           'e,de,d2e=[np.stack([a[k] for a in jets]) for k in range(3)]\n'
           'scales=np.array([a[3] for a in jets])\n',
  'tol': 1e-08},
 {'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
          'sector_free_energy_jet(e, de, d2e, scales, b)])',
  'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
               '_oracle_sector_free_energy_jet(e, de, d2e, scales, b)])',
  'setup': 'import numpy as np\n'
           'def _fx_sector_propagator_jet(h, dh, d2h, projector, b):\n'
           '    pv, pu = np.linalg.eigh(projector)\n'
           '    v = pu[:, pv > .5]\n'
           '    lam, rot = np.linalg.eigh(v.T @ h @ v)\n'
           '    u = v @ rot\n'
           '    x = -b*(lam-lam[0])\n'
           '\n'
           '    def _first(a, z):\n'
           '        gap = np.abs(a-z)\n'
           '        factor = np.ones(np.broadcast_shapes(np.shape(a), np.shape(z)))\n'
           '        np.divide(-np.expm1(-gap), gap, out=factor, where=gap != 0)\n'
           '        return np.exp(np.maximum(a,z))*factor\n'
           '\n'
           '    a, c, z = np.broadcast_arrays(x[:,None,None],x[None,:,None],x[None,None,:])\n'
           '    knots = np.sort(np.stack((a,c,z)),axis=0)\n'
           '    low, mid, high = knots\n'
           '    width = high-low\n'
           '    second = np.empty_like(width)\n'
           '    wide = width > .5\n'
           '    second[wide] = '
           '(_first(mid[wide],high[wide])-_first(low[wide],mid[wide]))/width[wide]\n'
           '    aa, bb = low[~wide]-high[~wide], mid[~wide]-high[~wide]\n'
           '    homogeneous = np.ones_like(aa)\n'
           '    a_power = np.ones_like(aa)\n'
           '    inverse_factorial = .5\n'
           '    series = .5*np.ones_like(aa)\n'
           '    for degree in range(1,32):\n'
           '        a_power *= aa\n'
           '        homogeneous = bb*homogeneous+a_power\n'
           '        inverse_factorial /= degree+2\n'
           '        series += inverse_factorial*homogeneous\n'
           '    second[~wide] = np.exp(high[~wide])*series\n'
           '    kernel = _first(x[:,None],x[None,:])\n'
           "    ai = -b*np.einsum('ka,ikl,lb->iab',u,dh,u)\n"
           "    aij = -b*np.einsum('ka,ijkl,lb->ijab',u,d2h,u)\n"
           '    de_reduced = kernel[None,:,:]*ai\n'
           "    ordered = np.einsum('acb,iac,jcb->ijab',second,ai,ai)\n"
           '    d2e_reduced = kernel[None,None,:,:]*aij + ordered + ordered.swapaxes(0,1)\n'
           '    e = (u*np.exp(x)) @ u.T\n'
           "    de = np.einsum('ak,ikl,bl->iab',u,de_reduced,u)\n"
           "    d2e = np.einsum('ak,ijkl,bl->ijab',u,d2e_reduced,u)\n"
           '    return e, de, d2e, float(-b*lam[0])\n'
           'p=1; n=2; b=.7\n'
           'h=np.array([[[.2*t,.1+.13*t],[.1+.13*t,-.3-.08*t]] for t in range(p)])\n'
           'dh=np.array([[[.3+.2*i,.4-.1*i],[.4-.1*i,-.5+.1*i]] for i in range(n)])\n'
           'd2h=np.array([[np.array([[.1*(i+j+1),.05],[.05,-.07]]) for j in range(n)] for i in '
           'range(n)])\n'
           'jets=[_fx_sector_propagator_jet(a,dh,d2h,np.eye(2),b) for a in h]\n'
           'e,de,d2e=[np.stack([a[k] for a in jets]) for k in range(3)]\n'
           'scales=np.array([a[3] for a in jets])\n',
  'tol': 1e-08},
 {'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
          'sector_free_energy_jet(e, de, d2e, scales, b)])',
  'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
               '_oracle_sector_free_energy_jet(e, de, d2e, scales, b)])',
  'setup': 'import numpy as np\n'
           'def _fx_sector_propagator_jet(h, dh, d2h, projector, b):\n'
           '    pv, pu = np.linalg.eigh(projector)\n'
           '    v = pu[:, pv > .5]\n'
           '    lam, rot = np.linalg.eigh(v.T @ h @ v)\n'
           '    u = v @ rot\n'
           '    x = -b*(lam-lam[0])\n'
           '\n'
           '    def _first(a, z):\n'
           '        gap = np.abs(a-z)\n'
           '        factor = np.ones(np.broadcast_shapes(np.shape(a), np.shape(z)))\n'
           '        np.divide(-np.expm1(-gap), gap, out=factor, where=gap != 0)\n'
           '        return np.exp(np.maximum(a,z))*factor\n'
           '\n'
           '    a, c, z = np.broadcast_arrays(x[:,None,None],x[None,:,None],x[None,None,:])\n'
           '    knots = np.sort(np.stack((a,c,z)),axis=0)\n'
           '    low, mid, high = knots\n'
           '    width = high-low\n'
           '    second = np.empty_like(width)\n'
           '    wide = width > .5\n'
           '    second[wide] = '
           '(_first(mid[wide],high[wide])-_first(low[wide],mid[wide]))/width[wide]\n'
           '    aa, bb = low[~wide]-high[~wide], mid[~wide]-high[~wide]\n'
           '    homogeneous = np.ones_like(aa)\n'
           '    a_power = np.ones_like(aa)\n'
           '    inverse_factorial = .5\n'
           '    series = .5*np.ones_like(aa)\n'
           '    for degree in range(1,32):\n'
           '        a_power *= aa\n'
           '        homogeneous = bb*homogeneous+a_power\n'
           '        inverse_factorial /= degree+2\n'
           '        series += inverse_factorial*homogeneous\n'
           '    second[~wide] = np.exp(high[~wide])*series\n'
           '    kernel = _first(x[:,None],x[None,:])\n'
           "    ai = -b*np.einsum('ka,ikl,lb->iab',u,dh,u)\n"
           "    aij = -b*np.einsum('ka,ijkl,lb->ijab',u,d2h,u)\n"
           '    de_reduced = kernel[None,:,:]*ai\n'
           "    ordered = np.einsum('acb,iac,jcb->ijab',second,ai,ai)\n"
           '    d2e_reduced = kernel[None,None,:,:]*aij + ordered + ordered.swapaxes(0,1)\n'
           '    e = (u*np.exp(x)) @ u.T\n'
           "    de = np.einsum('ak,ikl,bl->iab',u,de_reduced,u)\n"
           "    d2e = np.einsum('ak,ijkl,bl->ijab',u,d2e_reduced,u)\n"
           '    return e, de, d2e, float(-b*lam[0])\n'
           'p=2; n=2; b=.7\n'
           'h=np.array([[[.2*t,.1+.13*t],[.1+.13*t,-.3-.08*t]] for t in range(p)])\n'
           'dh=np.array([[[.3+.2*i,.4-.1*i],[.4-.1*i,-.5+.1*i]] for i in range(n)])\n'
           'd2h=np.array([[np.array([[.1*(i+j+1),.05],[.05,-.07]]) for j in range(n)] for i in '
           'range(n)])\n'
           'jets=[_fx_sector_propagator_jet(a,dh,d2h,np.eye(2),b) for a in h]\n'
           'e,de,d2e=[np.stack([a[k] for a in jets]) for k in range(3)]\n'
           'scales=np.array([a[3] for a in jets])\n',
  'tol': 1e-08},
 {'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
          'sector_free_energy_jet(e, de, d2e, scales, b)])',
  'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
               '_oracle_sector_free_energy_jet(e, de, d2e, scales, b)])',
  'setup': 'import numpy as np\n'
           'def _fx_sector_propagator_jet(h, dh, d2h, projector, b):\n'
           '    pv, pu = np.linalg.eigh(projector)\n'
           '    v = pu[:, pv > .5]\n'
           '    lam, rot = np.linalg.eigh(v.T @ h @ v)\n'
           '    u = v @ rot\n'
           '    x = -b*(lam-lam[0])\n'
           '\n'
           '    def _first(a, z):\n'
           '        gap = np.abs(a-z)\n'
           '        factor = np.ones(np.broadcast_shapes(np.shape(a), np.shape(z)))\n'
           '        np.divide(-np.expm1(-gap), gap, out=factor, where=gap != 0)\n'
           '        return np.exp(np.maximum(a,z))*factor\n'
           '\n'
           '    a, c, z = np.broadcast_arrays(x[:,None,None],x[None,:,None],x[None,None,:])\n'
           '    knots = np.sort(np.stack((a,c,z)),axis=0)\n'
           '    low, mid, high = knots\n'
           '    width = high-low\n'
           '    second = np.empty_like(width)\n'
           '    wide = width > .5\n'
           '    second[wide] = '
           '(_first(mid[wide],high[wide])-_first(low[wide],mid[wide]))/width[wide]\n'
           '    aa, bb = low[~wide]-high[~wide], mid[~wide]-high[~wide]\n'
           '    homogeneous = np.ones_like(aa)\n'
           '    a_power = np.ones_like(aa)\n'
           '    inverse_factorial = .5\n'
           '    series = .5*np.ones_like(aa)\n'
           '    for degree in range(1,32):\n'
           '        a_power *= aa\n'
           '        homogeneous = bb*homogeneous+a_power\n'
           '        inverse_factorial /= degree+2\n'
           '        series += inverse_factorial*homogeneous\n'
           '    second[~wide] = np.exp(high[~wide])*series\n'
           '    kernel = _first(x[:,None],x[None,:])\n'
           "    ai = -b*np.einsum('ka,ikl,lb->iab',u,dh,u)\n"
           "    aij = -b*np.einsum('ka,ijkl,lb->ijab',u,d2h,u)\n"
           '    de_reduced = kernel[None,:,:]*ai\n'
           "    ordered = np.einsum('acb,iac,jcb->ijab',second,ai,ai)\n"
           '    d2e_reduced = kernel[None,None,:,:]*aij + ordered + ordered.swapaxes(0,1)\n'
           '    e = (u*np.exp(x)) @ u.T\n'
           "    de = np.einsum('ak,ikl,bl->iab',u,de_reduced,u)\n"
           "    d2e = np.einsum('ak,ijkl,bl->ijab',u,d2e_reduced,u)\n"
           '    return e, de, d2e, float(-b*lam[0])\n'
           'p=3; n=1; b=.7\n'
           'h=np.array([[[.2*t,.1+.13*t],[.1+.13*t,-.3-.08*t]] for t in range(p)])\n'
           'dh=np.array([[[.3+.2*i,.4-.1*i],[.4-.1*i,-.5+.1*i]] for i in range(n)])\n'
           'd2h=np.array([[np.array([[.1*(i+j+1),.05],[.05,-.07]]) for j in range(n)] for i in '
           'range(n)])\n'
           'jets=[_fx_sector_propagator_jet(a,dh,d2h,np.eye(2),b) for a in h]\n'
           'e,de,d2e=[np.stack([a[k] for a in jets]) for k in range(3)]\n'
           'scales=np.array([a[3] for a in jets])\n',
  'tol': 1e-08},
 {'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
          'sector_free_energy_jet(e, de, d2e, scales, b)])',
  'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
               '_oracle_sector_free_energy_jet(e, de, d2e, scales, b)])',
  'setup': 'import numpy as np\n'
           'def _fx_sector_propagator_jet(h, dh, d2h, projector, b):\n'
           '    pv, pu = np.linalg.eigh(projector)\n'
           '    v = pu[:, pv > .5]\n'
           '    lam, rot = np.linalg.eigh(v.T @ h @ v)\n'
           '    u = v @ rot\n'
           '    x = -b*(lam-lam[0])\n'
           '\n'
           '    def _first(a, z):\n'
           '        gap = np.abs(a-z)\n'
           '        factor = np.ones(np.broadcast_shapes(np.shape(a), np.shape(z)))\n'
           '        np.divide(-np.expm1(-gap), gap, out=factor, where=gap != 0)\n'
           '        return np.exp(np.maximum(a,z))*factor\n'
           '\n'
           '    a, c, z = np.broadcast_arrays(x[:,None,None],x[None,:,None],x[None,None,:])\n'
           '    knots = np.sort(np.stack((a,c,z)),axis=0)\n'
           '    low, mid, high = knots\n'
           '    width = high-low\n'
           '    second = np.empty_like(width)\n'
           '    wide = width > .5\n'
           '    second[wide] = '
           '(_first(mid[wide],high[wide])-_first(low[wide],mid[wide]))/width[wide]\n'
           '    aa, bb = low[~wide]-high[~wide], mid[~wide]-high[~wide]\n'
           '    homogeneous = np.ones_like(aa)\n'
           '    a_power = np.ones_like(aa)\n'
           '    inverse_factorial = .5\n'
           '    series = .5*np.ones_like(aa)\n'
           '    for degree in range(1,32):\n'
           '        a_power *= aa\n'
           '        homogeneous = bb*homogeneous+a_power\n'
           '        inverse_factorial /= degree+2\n'
           '        series += inverse_factorial*homogeneous\n'
           '    second[~wide] = np.exp(high[~wide])*series\n'
           '    kernel = _first(x[:,None],x[None,:])\n'
           "    ai = -b*np.einsum('ka,ikl,lb->iab',u,dh,u)\n"
           "    aij = -b*np.einsum('ka,ijkl,lb->ijab',u,d2h,u)\n"
           '    de_reduced = kernel[None,:,:]*ai\n'
           "    ordered = np.einsum('acb,iac,jcb->ijab',second,ai,ai)\n"
           '    d2e_reduced = kernel[None,None,:,:]*aij + ordered + ordered.swapaxes(0,1)\n'
           '    e = (u*np.exp(x)) @ u.T\n'
           "    de = np.einsum('ak,ikl,bl->iab',u,de_reduced,u)\n"
           "    d2e = np.einsum('ak,ijkl,bl->ijab',u,d2e_reduced,u)\n"
           '    return e, de, d2e, float(-b*lam[0])\n'
           'p=4; n=2; b=.7\n'
           'h=np.array([[[.2*t,.1+.13*t],[.1+.13*t,-.3-.08*t]] for t in range(p)])\n'
           'dh=np.array([[[.3+.2*i,.4-.1*i],[.4-.1*i,-.5+.1*i]] for i in range(n)])\n'
           'd2h=np.array([[np.array([[.1*(i+j+1),.05],[.05,-.07]]) for j in range(n)] for i in '
           'range(n)])\n'
           'jets=[_fx_sector_propagator_jet(a,dh,d2h,np.eye(2),b) for a in h]\n'
           'e,de,d2e=[np.stack([a[k] for a in jets]) for k in range(3)]\n'
           'scales=np.array([a[3] for a in jets])\n'
           'scales=scales+1000.\n',
  'tol': 1e-08},
 {'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
          'sector_free_energy_jet(e, de, d2e, scales, b)])',
  'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
               '_oracle_sector_free_energy_jet(e, de, d2e, scales, b)])',
  'setup': 'import numpy as np\n'
           'def _fx_sector_propagator_jet(h, dh, d2h, projector, b):\n'
           '    pv, pu = np.linalg.eigh(projector)\n'
           '    v = pu[:, pv > .5]\n'
           '    lam, rot = np.linalg.eigh(v.T @ h @ v)\n'
           '    u = v @ rot\n'
           '    x = -b*(lam-lam[0])\n'
           '\n'
           '    def _first(a, z):\n'
           '        gap = np.abs(a-z)\n'
           '        factor = np.ones(np.broadcast_shapes(np.shape(a), np.shape(z)))\n'
           '        np.divide(-np.expm1(-gap), gap, out=factor, where=gap != 0)\n'
           '        return np.exp(np.maximum(a,z))*factor\n'
           '\n'
           '    a, c, z = np.broadcast_arrays(x[:,None,None],x[None,:,None],x[None,None,:])\n'
           '    knots = np.sort(np.stack((a,c,z)),axis=0)\n'
           '    low, mid, high = knots\n'
           '    width = high-low\n'
           '    second = np.empty_like(width)\n'
           '    wide = width > .5\n'
           '    second[wide] = '
           '(_first(mid[wide],high[wide])-_first(low[wide],mid[wide]))/width[wide]\n'
           '    aa, bb = low[~wide]-high[~wide], mid[~wide]-high[~wide]\n'
           '    homogeneous = np.ones_like(aa)\n'
           '    a_power = np.ones_like(aa)\n'
           '    inverse_factorial = .5\n'
           '    series = .5*np.ones_like(aa)\n'
           '    for degree in range(1,32):\n'
           '        a_power *= aa\n'
           '        homogeneous = bb*homogeneous+a_power\n'
           '        inverse_factorial /= degree+2\n'
           '        series += inverse_factorial*homogeneous\n'
           '    second[~wide] = np.exp(high[~wide])*series\n'
           '    kernel = _first(x[:,None],x[None,:])\n'
           "    ai = -b*np.einsum('ka,ikl,lb->iab',u,dh,u)\n"
           "    aij = -b*np.einsum('ka,ijkl,lb->ijab',u,d2h,u)\n"
           '    de_reduced = kernel[None,:,:]*ai\n'
           "    ordered = np.einsum('acb,iac,jcb->ijab',second,ai,ai)\n"
           '    d2e_reduced = kernel[None,None,:,:]*aij + ordered + ordered.swapaxes(0,1)\n'
           '    e = (u*np.exp(x)) @ u.T\n'
           "    de = np.einsum('ak,ikl,bl->iab',u,de_reduced,u)\n"
           "    d2e = np.einsum('ak,ijkl,bl->ijab',u,d2e_reduced,u)\n"
           '    return e, de, d2e, float(-b*lam[0])\n'
           'p=4; n=2; b=.7\n'
           'h=np.array([[[.2*t,.1+.13*t],[.1+.13*t,-.3-.08*t]] for t in range(p)])\n'
           'dh=np.array([[[.3+.2*i,.4-.1*i],[.4-.1*i,-.5+.1*i]] for i in range(n)])\n'
           'd2h=np.array([[np.array([[.1*(i+j+1),.05],[.05,-.07]]) for j in range(n)] for i in '
           'range(n)])\n'
           'jets=[_fx_sector_propagator_jet(a,dh,d2h,np.eye(2),b) for a in h]\n'
           'e,de,d2e=[np.stack([a[k] for a in jets]) for k in range(3)]\n'
           'scales=np.array([a[3] for a in jets])\n'
           'e=np.roll(e,1,axis=0); de=np.roll(de,1,axis=0); d2e=np.roll(d2e,1,axis=0); '
           'scales=np.roll(scales,1)\n',
  'tol': 1e-08},
 {'call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
          'sector_free_energy_jet(e, de, d2e, scales, b)])',
  'gold_call': 'np.concatenate([np.asarray(component, dtype=float).ravel() for component in '
               '_oracle_sector_free_energy_jet(e, de, d2e, scales, b)])',
  'setup': 'import numpy as np\n'
           'def _fx_sector_propagator_jet(h, dh, d2h, projector, b):\n'
           '    pv, pu = np.linalg.eigh(projector)\n'
           '    v = pu[:, pv > .5]\n'
           '    lam, rot = np.linalg.eigh(v.T @ h @ v)\n'
           '    u = v @ rot\n'
           '    x = -b*(lam-lam[0])\n'
           '\n'
           '    def _first(a, z):\n'
           '        gap = np.abs(a-z)\n'
           '        factor = np.ones(np.broadcast_shapes(np.shape(a), np.shape(z)))\n'
           '        np.divide(-np.expm1(-gap), gap, out=factor, where=gap != 0)\n'
           '        return np.exp(np.maximum(a,z))*factor\n'
           '\n'
           '    a, c, z = np.broadcast_arrays(x[:,None,None],x[None,:,None],x[None,None,:])\n'
           '    knots = np.sort(np.stack((a,c,z)),axis=0)\n'
           '    low, mid, high = knots\n'
           '    width = high-low\n'
           '    second = np.empty_like(width)\n'
           '    wide = width > .5\n'
           '    second[wide] = '
           '(_first(mid[wide],high[wide])-_first(low[wide],mid[wide]))/width[wide]\n'
           '    aa, bb = low[~wide]-high[~wide], mid[~wide]-high[~wide]\n'
           '    homogeneous = np.ones_like(aa)\n'
           '    a_power = np.ones_like(aa)\n'
           '    inverse_factorial = .5\n'
           '    series = .5*np.ones_like(aa)\n'
           '    for degree in range(1,32):\n'
           '        a_power *= aa\n'
           '        homogeneous = bb*homogeneous+a_power\n'
           '        inverse_factorial /= degree+2\n'
           '        series += inverse_factorial*homogeneous\n'
           '    second[~wide] = np.exp(high[~wide])*series\n'
           '    kernel = _first(x[:,None],x[None,:])\n'
           "    ai = -b*np.einsum('ka,ikl,lb->iab',u,dh,u)\n"
           "    aij = -b*np.einsum('ka,ijkl,lb->ijab',u,d2h,u)\n"
           '    de_reduced = kernel[None,:,:]*ai\n'
           "    ordered = np.einsum('acb,iac,jcb->ijab',second,ai,ai)\n"
           '    d2e_reduced = kernel[None,None,:,:]*aij + ordered + ordered.swapaxes(0,1)\n'
           '    e = (u*np.exp(x)) @ u.T\n'
           "    de = np.einsum('ak,ikl,bl->iab',u,de_reduced,u)\n"
           "    d2e = np.einsum('ak,ijkl,bl->ijab',u,d2e_reduced,u)\n"
           '    return e, de, d2e, float(-b*lam[0])\n'
           'p=8; n=2; b=.7\n'
           'h=np.array([[[.2*t,.1+.13*t],[.1+.13*t,-.3-.08*t]] for t in range(p)])\n'
           'dh=np.array([[[.3+.2*i,.4-.1*i],[.4-.1*i,-.5+.1*i]] for i in range(n)])\n'
           'd2h=np.array([[np.array([[.1*(i+j+1),.05],[.05,-.07]]) for j in range(n)] for i in '
           'range(n)])\n'
           'jets=[_fx_sector_propagator_jet(a,dh,d2h,np.eye(2),b) for a in h]\n'
           'e,de,d2e=[np.stack([a[k] for a in jets]) for k in range(3)]\n'
           'scales=np.array([a[3] for a in jets])\n',
  'tol': 1e-08}]

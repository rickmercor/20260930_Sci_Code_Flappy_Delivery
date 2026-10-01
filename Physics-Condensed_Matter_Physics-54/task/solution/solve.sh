#!/bin/bash
# Oracle: writes the gold solution so the verifier scores reward 1.0.
cat > /app/solution.py <<'SCICODE_GOLD_EOF'
import numpy as np
try:
    import scipy, scipy.linalg, scipy.sparse, scipy.optimize, scipy.spatial, scipy.integrate, scipy.stats, scipy.special, scipy.signal
except Exception:
    pass
import math

import numpy as np

def magnetic_ribbon(k: np.ndarray, delta: np.ndarray, flux: float = 0.0,
                            center: float = 0.0) -> np.ndarray:
    """Construct a magnetic Haldane ribbon with two orbitals per cell.

    Parameters
    ----------
    k : numpy.ndarray
        Nonempty finite real vector of dimensionless y momenta.
    delta : numpy.ndarray
        Finite real vector, length nx>=4, scalar on-site stripe disorder.
    flux : float
        Finite real magnetic flux per primitive cell, in flux-quantum units.
    center : float
        Finite real gauge origin x0 in physical length units.

    Returns
    -------
    numpy.ndarray
        Complex stack (len(k),2*nx,2*nx), basis (x,A),(x,B).
        Geometry: a1=(sqrt(3)/2,1/2), a2=(0,1), rA=(0,0),
        rB=(1/sqrt(3),0), open x and infinite periodic y. On-sites are
        delta[x]+0.2 on A and delta[x]-0.2 on B. NN directed bonds
        A(x,n)->B(x+dx,n+m): (dx,m)=(0,0),(-1,0),(-1,1), t=1.
        NNN directed same-sublattice bonds have (dx,m)=(1,0),(0,-1),
        (-1,1), amplitude i/3 on A and -i/3 on B. Each directed bond
        contributes to H[target,source] and its h.c.; truncate only at x edges.
        Its Bloch multiplier is exp(-i*k*m). Additionally multiply by
        exp[-2*pi*i*flux/Acell*((Xs+Xt)/2-center)*(m+Yt-Ys)],
        Acell=sqrt(3)/2, X=x*Acell+(alpha==B)/sqrt(3), Y=x/2.
        Use physical bond displacement, not just the integer m, in Peierls.

    Raises
    ------
    ValueError
        If k, delta, flux or center violates the stated domain.
    """
    k,delta=np.asarray(k),np.asarray(delta)
    if k.ndim!=1 or k.size<1 or not np.isrealobj(k) or not np.all(np.isfinite(k)):
        raise ValueError('invalid momentum vector')
    if delta.ndim!=1 or delta.size<4 or not np.isrealobj(delta) or not np.all(np.isfinite(delta)):
        raise ValueError('invalid disorder vector')
    for a in (flux,center):
        if not np.isscalar(a) or not np.isrealobj(a) or not np.isfinite(a):
            raise ValueError('invalid flux or center')
    nx=len(delta); area=np.sqrt(3)/2
    X=np.repeat(np.arange(nx)*area,2)+np.tile([0,1/np.sqrt(3)],nx)
    Y=np.repeat(np.arange(nx)/2,2)
    h=np.zeros((len(k),2*nx,2*nx),complex)
    h[:,np.arange(2*nx),np.arange(2*nx)]=np.repeat(delta,2)+np.tile([.2,-.2],nx)
    bonds=[]
    for x in range(nx):
        for dx,m in [(0,0),(-1,0),(-1,1)]:
            if 0<=x+dx<nx: bonds.append((2*x,2*(x+dx)+1,m,1.))
        for alpha in [0,1]:
            for dx,m in [(1,0),(0,-1),(-1,1)]:
                if 0<=x+dx<nx:
                    bonds.append((2*x+alpha,2*(x+dx)+alpha,m,(1 if alpha==0 else -1)*1j/3))
    for s,t,m,amp in bonds:
        phase=-2*np.pi*flux/area*((X[s]+X[t])/2-center)*(m+Y[t]-Y[s])
        v=amp*np.exp(-1j*k*m+1j*phase)
        h[:,t,s]+=v
        h[:,s,t]+=v.conj()
    return h

import numpy as np

def fixed_mu_projectors(h: np.ndarray, mu: float) -> np.ndarray:
    """Construct zero-temperature occupied projectors at fixed chemical potential.

    Parameters
    ----------
    h : numpy.ndarray
        Finite Hermitian stack (m,d,d), m>=1, even d>=2, Hermiticity
        absolute error <=1e-10. Occupied rank may differ between blocks.
    mu : float
        Finite real chemical potential, held fixed across fluxes.

    Returns
    -------
    numpy.ndarray
        Complex stack matching h, spectral projector onto eigenvalues E<mu.
        E=mu is excluded. Empty and full occupied subspaces are allowed.
        Return projectors, not eigenvectors; never force rank d/2.

    Raises
    ------
    ValueError
        Invalid shape, nonfinite/non-Hermitian h or invalid mu.
    """
    h=np.asarray(h,complex)
    if h.ndim!=3 or h.shape[0]<1 or h.shape[1]!=h.shape[2] or h.shape[1]<2 or h.shape[1]%2:
        raise ValueError('invalid Hamiltonian shape')
    if not np.all(np.isfinite(h)) or np.max(np.abs(h-h.swapaxes(-1,-2).conj()))>1e-10:
        raise ValueError('invalid Hermitian matrices')
    if not np.isscalar(mu) or not np.isrealobj(mu) or not np.isfinite(mu):
        raise ValueError('invalid chemical potential')
    e,v=np.linalg.eigh(h)
    return (v*(e<mu)[:,None,:])@v.swapaxes(-1,-2).conj()

import numpy as np

def fermi_partition(delta: np.ndarray, flux: float, mu: float,
                             scan: int = 128) -> np.ndarray:
    """Locate momentum intervals of fixed occupancy by bandwise root bracketing.

    Parameters
    ----------
    delta, flux : numpy.ndarray, float
        Disorder and flux accepted by magnetic_ribbon; gauge center is zero.
    mu : float
        Finite real fixed chemical potential.
    scan : int
        Integer >=8, no booleans; scan+1 equally spaced nodes on [0,2*pi].

    Returns
    -------
    numpy.ndarray
        Sorted real partition including 0 and 2*pi. At scan nodes sort the
        eigenvalues ascending. For each band and adjacent node pair whose
        E-mu values have strictly opposite signs, bisect that ordered band
        until bracket width <=1e-11, then record its midpoint. Also record
        interior scan nodes with abs(E-mu)<=1e-12. Merge interior roots
        separated by <=1e-9, keeping the smaller root; discard roots within
        1e-9 of either endpoint. The task prescribes this detector; it does
        not promise to find tangent roots or multiple roots inside one bin.

    Raises
    ------
    ValueError
        Invalid scan or mu, or invalid magnetic_ribbon parameters.
    """
    if isinstance(scan,(bool,np.bool_)) or not isinstance(scan,(int,np.integer)) or scan<8:
        raise ValueError('invalid scan count')
    if not np.isscalar(mu) or not np.isrealobj(mu) or not np.isfinite(mu):
        raise ValueError('invalid mu')
    grid=np.linspace(0,2*np.pi,scan+1)
    values=np.linalg.eigvalsh(magnetic_ribbon(grid,delta,flux))-mu
    roots=[]
    for b in range(values.shape[1]):
        for i in range(scan):
            if i>0 and abs(values[i,b])<=1e-12: roots.append(grid[i])
            if values[i,b]*values[i+1,b]<0:
                lo,hi=grid[i],grid[i+1]; flo=values[i,b]
                while hi-lo>1e-11:
                    mid=(lo+hi)/2
                    fm=np.linalg.eigvalsh(magnetic_ribbon(np.array([mid]),delta,flux))[0,b]-mu
                    if flo*fm<=0: hi=mid
                    else: lo=mid; flo=fm
                roots.append((lo+hi)/2)
    unique=[]
    for r in sorted(roots):
        if 1e-9<r<2*np.pi-1e-9 and (not unique or r-unique[-1]>1e-9): unique.append(r)
    return np.array([0.]+unique+[2*np.pi])

import numpy as np

def partition_density(delta: np.ndarray, flux: float, mu: float,
                               cuts: np.ndarray, order: int) -> np.ndarray:
    """Integrate the occupied cell density separately on each momentum interval.

    Parameters
    ----------
    delta, flux, mu : numpy.ndarray, float, float
        Model inputs accepted by earlier functions; gauge center zero.
    cuts : numpy.ndarray
        Finite real strictly increasing vector of length>=2, endpoints
        0 and 2*pi to absolute tolerance 1e-10, used as supplied.
    order : int
        Gauss-Legendre order >=4 per interval; booleans excluded.

    Returns
    -------
    numpy.ndarray
        Real vector (nx,), integral of sum_alpha P_(x alpha,x alpha)(k)
        over 0..2*pi, divided by 2*pi. Map an order-point Gauss-Legendre
        rule independently to every interval; use E<mu at every node.
        Do not enforce a fixed electron count or sample across a cut.

    Raises
    ------
    ValueError
        Invalid cuts/order or invalid earlier-function inputs.
    """
    cuts=np.asarray(cuts)
    if cuts.ndim!=1 or len(cuts)<2 or not np.isrealobj(cuts) or not np.all(np.isfinite(cuts)) or np.any(np.diff(cuts)<=0) or abs(cuts[0])>1e-10 or abs(cuts[-1]-2*np.pi)>1e-10:
        raise ValueError('invalid partition')
    if isinstance(order,(bool,np.bool_)) or not isinstance(order,(int,np.integer)) or order<4:
        raise ValueError('invalid quadrature order')
    q,w=np.polynomial.legendre.leggauss(order)
    ks=[]; weights=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        ks.extend((a+b)/2+(b-a)*q/2); weights.extend((b-a)*w/(4*np.pi))
    h=magnetic_ribbon(np.array(ks),delta,flux)
    p=fixed_mu_projectors(h,mu)
    n=np.diagonal(p,axis1=-2,axis2=-1).real.reshape(len(ks),-1,2).sum(axis=2)
    return np.array(weights)@n

import numpy as np

def refine_density(delta: np.ndarray, flux: float, mu: float, cuts: np.ndarray,
                            initial: np.ndarray, order: int = 8,
                            tol: float = 1e-10, max_order: int = 256) -> np.ndarray:
    """Refine partition quadrature with the supplied first density estimate.

    Parameters
    ----------
    delta, flux, mu, cuts : numpy.ndarray, float, float, numpy.ndarray
        Same domains as partition_density. Cuts remain fixed.
    initial : numpy.ndarray
        Finite real vector matching delta, the density at the given order.
        Treat this as an input estimate; do not discard or recompute it.
    order, max_order : int
        Integers, order>=4 and max_order>=2*order; booleans excluded.
    tol : float
        Finite positive real maximum-component convergence threshold.

    Returns
    -------
    numpy.ndarray
        Refined real density (nx,). Double order, recompute the integral,
        and return the new density at the first max(abs(new-old))<=tol.
        Test the candidate at max_order if reached by doubling. Never use
        an order larger than max_order; if none converges raise ValueError.

    Raises
    ------
    ValueError
        Invalid parameters/estimate, nonconvergence, or earlier input errors.
    """
    for n in (order,max_order):
        if isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)):
            raise ValueError('orders must be integers')
    if order<4 or max_order<2*order: raise ValueError('invalid order limits')
    if not np.isscalar(tol) or not np.isrealobj(tol) or not np.isfinite(tol) or tol<=0:
        raise ValueError('invalid tolerance')
    prev=np.asarray(initial)
    if prev.shape!=np.asarray(delta).shape or prev.ndim!=1 or not np.isrealobj(prev) or not np.all(np.isfinite(prev)):
        raise ValueError('invalid initial density')
    n=2*order
    while n<=max_order:
        current=partition_density(delta,flux,mu,cuts,n)
        if np.max(np.abs(current-prev))<=tol: return current
        prev=current; n*=2
    raise ValueError('density quadrature did not converge')

import numpy as np

def embedded_chern(p: np.ndarray) -> np.ndarray:
    """Evaluate the open-x marker in the cell-periodic Bloch gauge.

    Parameters
    ----------
    p : numpy.ndarray
        Finite complex stack (ny,2*nx,2*nx), ny>=5, nx>=4.
        Samples at k_j=2*pi*j/ny; general matrices allowed.

    Returns
    -------
    numpy.ndarray
        Real marker per cell (nx,), including orbital embedding and area.
        X_(x,A)=x*sqrt(3)/2, X_(x,B)=X_(x,A)+1/sqrt(3);
        Y_(x,alpha)=x/2; Acell=sqrt(3)/2. Differentiate the sampled P in k
        with the source method's prescription for this hybrid marker.
        D=partial_k P-i[Y,P], A=-i[X,P], and return
        Re[(2*pi*i/(ny*Acell))*sum_(j,alpha) diag(P_j[A_j,D_j])].
        X differences are ordinary open-ribbon differences, never wrapped.

    Raises
    ------
    ValueError
        Invalid shape or nonfinite input.
    """
    p=np.asarray(p,complex)
    if p.ndim!=3 or p.shape[0]<5 or p.shape[1]!=p.shape[2] or p.shape[1]<8 or p.shape[1]%2 or not np.all(np.isfinite(p)):
        raise ValueError('invalid projector samples')
    ny,dim,_=p.shape; nx=dim//2; area=np.sqrt(3)/2
    X=np.repeat(np.arange(nx)*area,2)+np.tile([0,1/np.sqrt(3)],nx)
    Y=np.repeat(np.arange(nx)/2,2)
    modes=np.fft.fftfreq(ny,1/ny)
    if ny%2==0: modes[ny//2]=0
    deriv=np.fft.ifft((1j*modes)[:,None,None]*np.fft.fft(p,axis=0),axis=0)
    A=-1j*(X[:,None]-X[None,:])*p
    D=deriv-1j*(Y[:,None]-Y[None,:])*p
    diag=np.einsum('kij,kji->ki',p,A@D-D@A)
    return (2j*np.pi/area*diag.reshape(ny,nx,2).sum(axis=2).mean(axis=0)).real

import numpy as np

def boundary_discrepancy(c: np.ndarray, nplus: np.ndarray, nminus: np.ndarray,
                                 flux: float, edge: int = 2) -> np.ndarray:
    """Compare local Chern and symmetric finite-flux Streda markers.

    Parameters
    ----------
    c, nplus, nminus : numpy.ndarray
        Equal finite real vectors (nx,). Densities correspond to +flux/-flux
        at identical chemical potential and geometric gauge convention.
    flux : float
        Finite positive real flux magnitude in flux-quantum units per cell.
    edge : int
        Positive integer, 2*edge<nx; booleans excluded.

    Returns
    -------
    numpy.ndarray
        Real vector [mean(cS), R_edge, R_bulk, R_edge-R_bulk], where
        cS=(nplus-nminus)/(2*flux). R is RMS of cS-c over the first/last
        edge cells or the remaining cells respectively. Count both edges.
        Do not sum or average the two densities before differencing.

    Raises
    ------
    ValueError
        Invalid vectors, flux or edge width.
    """
    c,nplus,nminus=map(np.asarray,(c,nplus,nminus))
    if c.ndim!=1 or c.shape!=nplus.shape or c.shape!=nminus.shape or any(not np.isrealobj(a) or not np.all(np.isfinite(a)) for a in (c,nplus,nminus)):
        raise ValueError('invalid vectors')
    if isinstance(edge,(bool,np.bool_)) or not isinstance(edge,(int,np.integer)) or edge<1 or 2*edge>=len(c):
        raise ValueError('invalid edge width')
    if not np.isscalar(flux) or not np.isrealobj(flux) or not np.isfinite(flux) or flux<=0:
        raise ValueError('invalid flux')
    cs=(nplus-nminus)/(2*flux); d=cs-c
    re=np.sqrt(np.mean(np.r_[d[:edge],d[-edge:]]**2))
    rb=np.sqrt(np.mean(d[edge:-edge]**2))
    return np.array([cs.mean(),re,rb,re-rb])

import numpy as np

def run_ribbon_comparison(nx: int = 14, ny: int = 129, seed: int = 54,
                                   flux: float = .003, mu: float = .08,
                                   edge: int = 2) -> float:
    """Run all seven earlier steps for the ribbon marker comparison.

    Parameters
    ----------
    nx, ny : int
        Cell count>=6 and marker momentum count>=5, booleans excluded.
    seed : int
        Unsigned 32-bit seed, booleans excluded. For each x, update
        s=(1664525*s+1013904223) mod 2**32 then delta[x]=.4*(s/2**32-.5).
    flux, mu, edge : float, float, int
        Positive flux, finite mu, and edge width accepted by earlier steps.

    Returns
    -------
    float
        R_edge-R_bulk. Compute the zero-flux marker on k_j=2*pi*j/ny.
        For each signed flux, obtain cuts with scan=128, start partition
        quadrature at order 8, then refine with tol=1e-10,max_order=256.
        Density integrates continuous ky; marker retains the stated ny grid.
        All seven earlier oracles must be called directly and their outputs
        used. Keep mu fixed at all fluxes; do not impose half filling.

    Raises
    ------
    ValueError
        Invalid parameters, nonconvergent integration or earlier input errors.
    """
    for v,lo in ((nx,6),(ny,5)):
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<lo:
            raise ValueError('invalid grid size')
    if isinstance(seed,(bool,np.bool_)) or not isinstance(seed,(int,np.integer)) or not 0<=seed<2**32:
        raise ValueError('invalid seed')
    if not np.isscalar(flux) or not np.isrealobj(flux) or not np.isfinite(flux) or flux<=0:
        raise ValueError('invalid flux')
    s=int(seed); delta=np.empty(nx)
    for x in range(nx):
        s=(1664525*s+1013904223)%2**32
        delta[x]=.4*(s/2**32-.5)
    h=magnetic_ribbon(2*np.pi*np.arange(ny)/ny,delta,0.)
    p=fixed_mu_projectors(h,mu)
    c=embedded_chern(p)
    densities=[]
    for f in (flux,-flux):
        cuts=fermi_partition(delta,f,mu,128)
        first=partition_density(delta,f,mu,cuts,8)
        densities.append(refine_density(delta,f,mu,cuts,first,8,1e-10,256))
    diagnostics=boundary_discrepancy(c,densities[0],densities[1],flux,edge)
    return float(diagnostics[3])
SCICODE_GOLD_EOF

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

def spin_operators(n: int) -> "tuple[np.ndarray, np.ndarray]":
    """Return (bonds, total_s2) in the sorted zero-magnetization bit basis.

    Parameters
    ----------
    n : int
        Even number of spin-1/2 sites, 4 <= n <= 8. Site i is bit i;
        bit 1 has Sz=+1/2. Basis integers ascend and have n/2 set bits.

    Returns
    -------
    bonds : float array (n, d, d), d = binomial(n, n/2)
        bonds[i] = S_i dot S_(i+1 mod n), with hbar=1. A pair contributes
        +1/4 on parallel bits, -1/4 on antiparallel bits, and an off-diagonal
        spin-exchange matrix element +1/2 on antiparallel bits.
    total_s2 : float array (d, d)
        Total-spin Casimir 3*n/4*I + 2*sum_(i<j) S_i dot S_j.

    Notes
    -----
    Use each periodic nearest-neighbor bond once. All inputs are valid;
    no input is mutated. Return real numerical arrays, not basis labels.
    """
    basis = [x for x in range(1 << n) if x.bit_count() == n // 2]
    index = {x: i for i, x in enumerate(basis)}
    d = len(basis)

    def _pair(i, j):
        a = np.zeros((d, d))
        for c, x in enumerate(basis):
            opposite = ((x >> i) & 1) != ((x >> j) & 1)
            a[c, c] = -0.25 if opposite else 0.25
            if opposite:
                a[index[x ^ (1 << i) ^ (1 << j)], c] = 0.5
        return a

    bonds = np.stack([_pair(i, (i + 1) % n) for i in range(n)])
    s2 = 0.75 * n * np.eye(d)
    for i in range(n):
        for j in range(i + 1, n):
            s2 += 2.0 * _pair(i, j)
    return bonds, s2

import numpy as np

def spin_projectors(total_s2: "np.ndarray", n: int) -> "np.ndarray":
    """Return orthogonal projectors onto each total-spin sector in Sz=0.

    Parameters
    ----------
    total_s2 : finite real symmetric array (d, d)
        Casimir from spin_operators(n); d=binomial(n,n/2).
    n : int
        Even site count, 4 <= n <= 8.

    Returns
    -------
    projectors : float array (n/2+1, d, d)
        Index S=0,...,n/2 projects onto eigenvalue S*(S+1) of total_s2.
        Assign an eigenvalue to that sector when its absolute distance
        from S*(S+1) is below 1e-7. Projectors sum to I. They retain the
        original bit-basis coordinates and do not depend on eigenvector
        signs or rotations within a degenerate eigenspace. Inputs are
        not mutated. The trace of a projector is its multiplicity-space
        dimension; the thermal spin degeneracy is separately 2*S+1.
    """
    values, vectors = np.linalg.eigh(total_s2)
    result = []
    for spin in range(n // 2 + 1):
        u = vectors[:, np.abs(values - spin * (spin + 1)) < 1e-7]
        result.append(u @ u.T)
    return np.stack(result)

import numpy as np

def bead_hamiltonian_jet(q: "np.ndarray", j0: float, g: float, alpha: float, bonds: "np.ndarray") -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Return each bead Hamiltonian and its first two coordinate derivatives.

    Parameters
    ----------
    q : finite real array (p, n)
        Reduced displacements, p=1,...,8; n=4,6,8. Site and bead indices
        start at zero. Sites are periodic. These are displacements, so
        do not apply a minimum-image convention to site differences.
    j0 : positive float
    g, alpha : nonnegative floats
        Exchange parameters, with g in inverse length and alpha in
        inverse length squared. All exchanges in this input are positive.
    bonds : finite real array (n, d, d)
        bonds[k]=S_k dot S_(k+1 mod n) in the common Sz=0 basis.

    Returns
    -------
    h : float array (p, d, d)
    dh : float array (p, n, d, d)
    d2h : float array (n, n, d, d)
        Set v[k,i]=delta_(i,(k+1)%n)-delta_(i,k), x[t,k]=sum_i v[k,i]q[t,i],
        and J[t,k]=j0*(1-g*x[t,k]+alpha*x[t,k]**2).
        h[t]=sum_k J[t,k]*bonds[k].
        dh[t,i]=sum_k j0*(-g+2*alpha*x[t,k])*v[k,i]*bonds[k].
        d2h[i,j]=2*j0*alpha*sum_k v[k,i]*v[k,j]*bonds[k].
        The second derivative is bead independent. Any derivative
        involving coordinates on different beads is zero at this stage.
        Both coordinate indices of d2h are ordinary derivatives, with
        no factorial scaling. All inputs are valid and are not mutated.
    """
    q = np.asarray(q, dtype=float)
    n = q.shape[1]
    incidence = np.roll(np.eye(n), 1, axis=1) - np.eye(n)
    x = q @ incidence.T
    exchange = j0 * (1.0 - g*x + alpha*x*x)
    h = np.einsum('tk,kab->tab', exchange, bonds)
    dh = np.einsum('tk,ki,kab->tiab', j0*(-g+2*alpha*x), incidence, bonds)
    d2h = 2*j0*alpha*np.einsum('ki,kj,kab->ijab', incidence, incidence, bonds)
    return h, dh, d2h

import numpy as np

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

import numpy as np

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

import numpy as np

def rped_force_constants(q: "np.ndarray", beta: float, j0: float, g: float, alpha: float, bonds: "np.ndarray", projectors: "np.ndarray") -> "tuple[float, np.ndarray, np.ndarray, np.ndarray]":
    """Return exact finite-bead spin free energy, forces and analytic Hessian.

    Parameters
    ----------
    q : finite real array (p, n), p=1,...,8; n=4,6,8
    beta, j0 : positive floats
    g, alpha : nonnegative floats
        Units have hbar=kB=1. J=j0*(1-g*Delta q+alpha*Delta q**2)>0.
        Every scaled sector trace satisfies sector_free_energy_jet's
        domain, and each projected b*h spectral width is at most 500.
    bonds : float array (n, d, d)
    projectors : float array (n/2+1, d, d)
        Consistent zero-magnetization operators and total-spin projectors
        from spin_operators and spin_projectors, ordered S=0,...,n/2.

    Returns
    -------
    free_energy : float
    forces : float array (p, n)
    hessian : float array (p*n, p*n)
    probabilities : float array (n/2+1,)
        Set b=beta/p. For each spin S obtain log Q_S, force f_S and
        Hessian K_S from sector_free_energy_jet. The complete partition
        factor Z=sum_S(2*S+1)*Q_S defines F_p=-log(Z)/b and weights
        w_S=(2*S+1)*Q_S/Z. Combine logarithms by log-sum-exp.
        The full force is f=sum_S w_S*f_S. In bead-major coordinates,
        K=sum_S w_S*K_S-b*(sum_S w_S*outer(f_S,f_S)-outer(f,f)).
        Return F_p,f,K,w. The covariance term comes from differentiating
        the thermal sector weights and must be retained.
        Compute the analytic whole-chain Hessian, including same-bead
        d2h and cross-bead derivative insertions. No finite differences,
        subsystem corrections or trace factorization are part of this
        calculation. NumPy only; inputs are not mutated.
    """
    q = np.asarray(q,dtype=float)
    b = beta/q.shape[0]
    h,dh,d2h = bead_hamiltonian_jet(q,j0,g,alpha,bonds)
    logs,fs,ks = [],[],[]
    for spin,projector in enumerate(projectors):
        jets = [sector_propagator_jet(h[t],dh[t],d2h,projector,b) for t in range(len(q))]
        e,de,d2e = [np.stack([a[k] for a in jets]) for k in range(3)]
        scales = np.array([a[3] for a in jets])
        logq,force,k = sector_free_energy_jet(e,de,d2e,scales,b)
        logs.append(logq+np.log(2*spin+1))
        fs.append(force)
        ks.append(k)
    logs = np.array(logs)
    raw = np.exp(logs-logs.max())
    weights = raw/raw.sum()
    logz = float(logs.max()+np.log(raw.sum()))
    fs = np.stack(fs)
    flat = fs.reshape(len(fs),-1)
    force = np.einsum('s,sti->ti',weights,fs)
    mean = force.reshape(-1)
    covariance = np.einsum('s,sa,sb->ab',weights,flat,flat)-np.outer(mean,mean)
    hessian = np.einsum('s,sab->ab',weights,np.stack(ks))-b*covariance
    return float(-logz/b),force,hessian,weights

import numpy as np

def internal_curvature(hessian: "np.ndarray", p: int, n: int, beta: float, mass: float) -> "tuple[float, np.ndarray]":
    """Find the Einstein stiffness needed for nonnegative internal curvature.

    Parameters
    ----------
    hessian : finite real symmetric array (p*n, p*n)
        Spin free-energy Hessian in bead-major order; symmetry holds to
        absolute error 1e-9. p=1,...,8 and n=4,6,8.
    p, n : int
    beta, mass : positive floats
        Equal reduced mass at every site; hbar=1.

    Returns
    -------
    k_star : float
    eigenvalues : float array (p*(n-1),)
        Add the bead spring Hessian gamma*kron(L_p,I_n), where
        gamma=mass*(p/beta)**2 and
        L_p=sum_(t=0..p-1)(e_t-e_((t+1)%p))(e_t-e_((t+1)%p)).T.
        Thus L_1=0; for p=2 both cyclic edges are counted, giving
        L_2=[[2,-2],[-2,2]]. This is the Hessian of
        mass/(2*(beta/p)**2)*sum_(t,i)(q[t,i]-q[(t+1)%p,i])**2.
        Restrict to displacements whose site sum is zero on every bead.
        One valid orthonormal basis B of shape (n,n-1) has, in column j,
        entries 1/sqrt((j+1)*(j+2)) in rows 0,...,j, entry
        -(j+1)/sqrt((j+1)*(j+2)) in row j+1, and zero otherwise.
        With C=kron(I_p,B), return the ascending eigenvalues of
        C.T @ (hessian+gamma*kron(L_p,I_n)) @ C and
        k_star=max(0,-eigenvalues[0]). Average hessian with its transpose
        before diagonalizing to remove roundoff asymmetry only.
        Adding Einstein energy k/2*sum q**2 shifts these eigenvalues by k.
        The result concerns local curvature at fixed q, not an equilibrium
        transition or a phonon frequency at a stationary structure.
        Inputs are not mutated.
    """
    laplacian = np.zeros((p,p))
    for t in range(p):
        edge = np.zeros(p)
        edge[t] += 1
        edge[(t+1)%p] -= 1
        laplacian += np.outer(edge,edge)
    basis = np.zeros((n,n-1))
    for j in range(n-1):
        norm = np.sqrt((j+1)*(j+2))
        basis[:j+1,j] = 1/norm
        basis[j+1,j] = -(j+1)/norm
    c = np.kron(np.eye(p),basis)
    total = .5*(hessian+hessian.T)+mass*(p/beta)**2*np.kron(laplacian,np.eye(n))
    values = np.linalg.eigvalsh(c.T@total@c)
    return float(max(0.,-values[0])),values

import numpy as np

def solve(q: "np.ndarray", beta: float, j0: float, g: float, alpha: float, mass: float) -> float:
    """Return the exact local internal-curvature Einstein threshold.

    Parameters
    ----------
    q : finite real array (p, n), p=1,...,8; n=4,6,8
    beta, j0, mass : positive floats
    g, alpha : nonnegative floats
        Same domain and reduced units as rped_force_constants.

    Returns
    -------
    k_star : float
        Construct the periodic Sz=0 spin operators and all total-spin
        projectors. Obtain the analytic Hessian of F_p=-log(Z)/(beta/p)
        for quadratic exchange J=j0*(1-g*Delta q+alpha*Delta q**2),
        retaining thermal multiplicities and all mixed bead derivatives.
        Add the physical bead-spring Hessian and restrict to the
        per-bead zero-site-sum space using internal_curvature.
        Return max(0,-lambda_min) before adding the Einstein k*I term.
        Use the full chain at the supplied p; no finite stencil, fitting,
        relaxation or continuum limit. NumPy only. Inputs are not mutated.
    """
    n = np.shape(q)[1]
    bonds,s2 = spin_operators(n)
    projectors = spin_projectors(s2,n)
    _,_,hessian,_ = rped_force_constants(q,beta,j0,g,alpha,bonds,projectors)
    return internal_curvature(hessian,np.shape(q)[0],n,beta,mass)[0]
SCICODE_GOLD_EOF

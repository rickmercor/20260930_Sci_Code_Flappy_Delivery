# Material_Science-Molecular_Modeling-42

## Background

Geometry-dependent exchange couples the spin trace to nuclear displacements on imaginary-time beads. Its second derivatives couple different beads and include changes in thermal spin-sector weights. The resulting force constants determine the local curvature of the finite ring-polymer potential.

## Problem

Compute the local Einstein stiffness needed for nonnegative internal curvature at the fixed configuration below, using the exact whole-chain spin–phonon construction in the attached paper. Retain all spin states at the stated bead count, with no subsystem correction or bead-count extrapolation; the quadratic exchange and analytic Hessian are benchmark extensions of the paper's geometry-dependent Hamiltonian.

The periodic six-site spin-$1/2$ chain has $P=4$, $\beta=3.7$, $J_0=1$, $g=0.8$, $\alpha=0.30$, $m=0.20$, $\hbar=k_B=1$, and
$$
\widehat h(q_\tau)=\sum_{i=0}^{5}J_{\tau i}\widehat{\mathbf S}_i\cdot\widehat{\mathbf S}_{i+1},\qquad
J_{\tau i}=J_0\left[1-g\Delta q_{\tau i}+\alpha(\Delta q_{\tau i})^2\right],\qquad
\Delta q_{\tau i}=q_{\tau,i+1}-q_{\tau i}.
$$
Site indices are modulo six; displacements use no minimum-image wrapping, with rows in imaginary-time order and columns in site order:
$$
q^0=\begin{pmatrix}
 0.36&-0.24& 0.12&-0.33& 0.21&-0.12\\
-0.09& 0.30&-0.18& 0.27&-0.36& 0.06\\
 0.24&-0.06&-0.30& 0.15& 0.09&-0.12\\
-0.21& 0.12& 0.33&-0.09&-0.27& 0.12
\end{pmatrix}.
$$
Inputs use consistent reduced energy, length and mass units; $g$ has inverse-length units and $\alpha$ has inverse-length-squared units.

Let $K_{\rm spin}$ be the full analytic Hessian of the paper's ring-polymer spin free energy $F_P$ at $q^0$, in bead-major coordinates $a=6\tau+i$, including same-bead and cross-bead derivatives; let $K_{\rm spring}$ be the Hessian of its periodic harmonic bead-spring term. For $C$ with orthonormal columns spanning $\mathcal V=\{u\in\mathbb R^{24}:\sum_i u_{\tau i}=0\text{ for every }\tau\}$, define
$$
\lambda_0\leq\lambda_1\leq\cdots=\operatorname{eig}\left[C^{\mathsf T}(K_{\rm spin}+K_{\rm spring})C\right],\qquad
k_* = \max(0,-\lambda_0).
$$
The Einstein term $\tfrac{k}{2}\sum_{\tau i}q_{\tau i}^2$ shifts each retained curvature by $k$; this is a local analytic curvature threshold at fixed geometry, not an equilibrium phase boundary or a finite-stencil value.

Recover the free-energy normalization, bead-spring coefficient and exact noncommuting exponential derivative from the paper, and report $k_*$ to eight decimal places in reduced energy per squared reduced length. In the reasoning, cite the source equation numbers, explain the full spin trace, both ordered second-derivative insertions, the direct Hamiltonian curvature and the sector-weight derivative term; give $\log Z$ for the complete partition factor, $F_P$, $K_{{\rm spin},0,1}$, $K_{{\rm spin},0,6}$, the two lowest constrained eigenvalues, and $\gamma$ in $K_{\rm spring}=\gamma L_P\otimes I_6$, where $L_P$ sums all $P$ directed periodic bead links as quadratic spring energies.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules:

- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.

Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_step_01_spin_operators

Goal
----
Build periodic spin-1/2 bond operators and the total-spin Casimir.

```python
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
    return
```

### Step 2

02_step_02_spin_projectors.py

Goal
----
Resolve the zero-magnetization space into total-spin sectors.

```python
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
    return
```

### Step 3

03_step_03_bead_hamiltonians.

Goal
----
Build bead Hamiltonians and their first two displacement derivatives.

```python
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
    return
```

### Step 4

04_step_04_sector_propagator

Goal
----
Evaluate the projected thermal exponential through second order.

```python
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
```

### Step 5

05_step_05_sector_free_energy_jet.

Goal
----
Coordinates on one bead differentiate the same exponential twice. Coordinates on different beads replace two ordered factors. Taking the logarithm subtracts the disconnected product of first derivatives.

```python
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
```

### Step 6

06_step_06_rped_force_constants

Goal
----
Assemble the whole-chain spin free energy, forces and Hessian.

```python
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
    return
```

### Step 7

07_step_07_internal_curvature

Goal
----
Resolve the least-curved internal ring-polymer displacement.

```python
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
    return
```

### Step 8

08_step_08_solve

Goal
----
The threshold is the extra Einstein stiffness needed for nonnegative curvature in the specified internal space. The geometry is fixed; the calculation does not locate an equilibrium phase transition.

```python
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
        Finite stiffness in reduced energy per squared reduced length.
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
    return
```

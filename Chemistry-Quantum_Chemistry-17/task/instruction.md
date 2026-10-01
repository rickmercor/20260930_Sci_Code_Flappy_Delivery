# Chemistry-Quantum_Chemistry-17

## Background

Strong electronic correlation can require many electronic configurations to describe a molecular wavefunction. Projection methods use a tractable trial state to guide evolution through a larger determinant space. Complex phases and nonorthogonal determinants make the evaluation of mixed matrix elements a central numerical problem.

## Problem

Compute a finite-block mixed-energy increment for a four-orbital electronic Hamiltonian using an orbital-relaxed DOCI trial and phaseless auxiliary-field projection. Use the recent construction that combines a seniority-zero multiconfigurational trial with auxiliary-field projection, its original complex importance-sampling formulation, and a generalized nonorthogonal matrix-element treatment that remains valid when individual determinant overlaps vanish. Combine this with a coupled orbital–configuration response treatment from modular multiconfigurational orbital optimization, and identify and cite the four methodological sources. The integrals, four orbital-relaxation updates, initial walkers and seven auxiliary-field rows below define the calculation completely. This is a deterministic replay of one projection block; it is not a request for a converged Monte Carlo ground-state energy.
In the methodological discussion, retrieve four details from the cited sources: the standard MR-trial CI coefficient truncation threshold and the exception for the orbital-optimized DOCI calculation with 12 electrons in 28 orbitals; the alternative additional phase constraint tested within the original importance-sampled phase-free random walk and its reported variance comparison with the cosine constraint; the generalized nonorthogonal two-body contraction for two zero-overlap spin-orbital pairs, together with its rule for more than two such pairs; and the quantities exchanged between the orbital optimizer and active-space solver in the modular one-step multiconfigurational scheme. Define the reduced overlap and zero-overlap co-density used in that contraction. Distinguish the published truncation choices from the complete six-component trial required here, distinguish the alternative phase constraint from the prescribed replay, and apply the zero-overlap rule to the separate trial-column walker. These source comparisons belong in the reasoning and do not change the numerical replay specified below.
Use an orthonormal spatial basis, two alpha and two beta electrons, and atomic units. The Hamiltonian is
\[
\hat H=\sum_{pq\sigma}h_{pq}a^\dagger_{p\sigma}a_{q\sigma}
+\frac12\sum_{\ell,pqrs,\sigma\tau}L^\ell_{pq}L^\ell_{rs}
a^\dagger_{p\sigma}a^\dagger_{r\tau}a_{s\tau}a_{q\sigma}.
\]There is no nuclear constant. The matrices are fixed benchmark integrals, not data extracted from a molecular calculation in the paper. The entries of h are in Eh; those of the factors are in the square root of Eh.
\[
h=\begin{pmatrix}
-1.084145&0.12&-0.08&0.05\\
0.12&-0.885553&0.11&-0.06\\
-0.08&0.11&-0.763931&0.09\\
0.05&-0.06&0.09&-0.742439
\end{pmatrix}.
\]\[
L^0=\begin{pmatrix}.58&.196&-.084&.112\\.196&.43&.224&-.056\\-.084&.224&.36&.168\\.112&-.056&.168&.29\end{pmatrix},\quad
L^1=\begin{pmatrix}.16&-.308&.112&.084\\-.308&-.22&.140&.196\\.112&.140&.19&-.224\\.084&.196&-.224&-.14\end{pmatrix},
\]\[
L^2=\begin{pmatrix}-.12&.084&.252&-.140\\.084&.17&-.168&.056\\.252&-.168&.08&.308\\-.140&.056&.308&-.10\end{pmatrix}.
\]For each real orthogonal spatial frame U, build the complete paired basis in lexicographic order of occupied frame-column tuples and project the Hamiltonian into this basis. Within a determinant, alpha creation operators precede beta creation operators, with increasing column indices inside each spin sector. Let E_D(U) be the lowest projected eigenvalue and c(U) its normalized eigenvector, with its largest-magnitude component positive and the smallest index breaking a magnitude tie. All six coefficients are retained.
Use the following deterministic orbital-relaxation protocol, whose constants and fixed iteration count are benchmark conventions. Begin with U_0 equal to the four-dimensional identity and perform exactly four macro updates. The local rotation coordinates are ordered (0,1), (0,2), (0,3), (1,2), (1,3), (2,3), with
\[
K^{pq}_{pq}=1,\qquad K^{pq}_{qp}=-1,
\qquad K^{pq}_{ij}=0\text{ otherwise},\qquad
U(\theta)=U\exp\!\left(\sum_{p<q}\theta_{pq}K^{pq}\right).
\]At each current frame, use the exact gradient and Hessian of E_D(U(theta)) at theta=0, including the response of the re-diagonalized CI ground state. If the Euclidean gradient norm is at most 1e-10 Eh, hold the frame unchanged for that update. Otherwise define
\[
\delta=\max\!\left(0,0.05\,E_h-\lambda_{\min}(\mathcal H)\right),\qquad
 d=-(\mathcal H+\delta I)^{-1}\boldsymbol g,\qquad
 d\leftarrow d\min\!\left(1,\frac{0.25}{\|d\|_2}\right),
\]where the coordinates are dimensionless rotation angles, bold g is the local energy gradient and script H is its Hessian. After clipping, also hold the frame unchanged when
\[
-\boldsymbol g^T d\leq10^{-14}E_h.
\]Otherwise choose the first alpha in the ordered sequence 1, 1/2, ..., 2^-20 satisfying
\[
E_D\!\left(U\exp\!\left[\alpha\sum_{p<q}d_{pq}K^{pq}\right]\right)
\leq E_D(U)+10^{-4}\alpha\,\boldsymbol g^T d,
\]and accept that frame; failure to find such an alpha is an error. Re-diagonalize at the accepted frame, and after four updates fix the trial to U_4 and c(U_4), with reference energy E_trial=E_D(U_4). The specified finite relaxation is not a claim of convergence to an orbital-optimized minimum. Keep the Hamiltonian, walkers and propagators in the original input basis throughout the projection replay; only the trial orbitals use U_4. The walkers remain unrestricted determinants.
There are four walkers. Let the spin index be zero for alpha and one for beta. The real base matrices are
\[
A^0=\begin{pmatrix}1&.07\\-.04&.92\\.18&-.14\\-.12&.21\end{pmatrix},\qquad
A^1=\begin{pmatrix}.95&-.03\\.06&1.02\\-.13&.17\\.16&.09\end{pmatrix}.
\]For zero-based indices w=0,...,3, p=0,...,3 and j=0,1, initialize the orbital matrices by
\[
\Phi^\sigma_{w,pj}=A^\sigma_{pj}
+0.035w\cos[(p+1)(j+2)]
+0.020i(w+\sigma)\sin[(p+2)(j+1)].
\]The initial importance weights are [1.0, 0.8, 1.2, 0.9]. They are not orbital normalization factors. Use a time step of 0.08 inverse Eh and an energy shift of -2.1 Eh. For t=0,...,6 and ell=0,1,2, the prescribed fields are
\[
x_{tw\ell}=0.85\sin[(t+1)(w+2)(\ell+1)]
+0.35\cos[(t+2)(\ell+2)+w].
\]All trigonometric arguments are in radians. Do not draw additional fields or apply a Gaussian quadrature weight to these samples.
Use the following finite-step convention. For each live walker, evaluate its total DOCI overlap O, local energy and complex force expectation on the old orbitals. An individual DOCI determinant may have zero overlap; this does not remove its one- or two-body numerator from the trial contraction. Only the summed overlap belongs in the estimator denominator.
\[
E_L=\frac{\langle\Psi_T|\hat H|\Phi\rangle}{O},\qquad
v_\ell=\frac{\langle\Psi_T|i\sum_{pq\sigma}L^\ell_{pq}a^\dagger_{p\sigma}a_{q\sigma}|\Phi\rangle}{O},\qquad
\bar x=-\sqrt{\Delta\tau}\,v.
\]\[
h_0=h-\frac12\sum_\ell L^\ell L^\ell,\qquad
B=e^{-\Delta\tau h_0/2}
 e^{i\sqrt{\Delta\tau}\sum_\ell(x_\ell-\bar x_\ell)L^\ell}
 e^{-\Delta\tau h_0/2},\qquad
\Lambda=x\cdot\bar x-\frac12\bar x\cdot\bar x.
\]The two dot products in Lambda are bilinear. Apply B to both spin matrices. A thin QR factorization may stabilize each propagated matrix. Make each R diagonal positive real and retain the product g of the two R determinants. If O' is measured on the two Q matrices, use
\[
S=g\frac{O'}{O},\qquad
I=S\exp(\Lambda+\Delta\tau E_{\rm shift}),\qquad
w'=w|I|\max[0,\cos(\arg S)].
\]Use the phase of S in the cosine. Measure the local energies after updating all walkers and weights. Do not replace the finite-step importance factor by a local-energy exponential. Do not subtract a mean field, cap the force, clip the local energy, renormalize the weights, discard an initial burn-in period or reconfigure the population. Skip walkers already at zero weight.
For the initial ensemble and after each completed step, form the real mixed energy. Return the increment after step seven in mEh:
\[
E_t=\frac{\sum_w w_{tw}\,\operatorname{Re}E_{L,tw}}{\sum_w w_{tw}},\qquad
Z=1000(E_7-E_{\rm trial}).
\]In the reasoning report: the five orbital-relaxation energies E_D(U_0) through E_D(U_4), the six initial orbital-gradient components, the six initial Hessian eigenvalues in ascending order, the final 4 by 4 frame U_4 and its six signed DOCI coefficients; the complex overlap, local energy and three force expectations for walker zero before propagation; Lambda and g for its first update; the eight mixed energies E_0 through E_7; the four final absolute weights; and the four physical overlap angles from the first and last updates. Also evaluate the overlap, local energy and three forces for the separate walker whose two spin matrices both equal columns 0 and 1 of U_4. Explain why zero component overlaps do not justify dropping that walker's nonzero matrix-element contributions. State the role and limitation of the seniority-zero trial, the force-bias sign, and the distinction between the overlap phase and the full importance-factor phase. Put Z alone inside <final_answer> tags, followed by <reasoning> containing the requested derivation and diagnostics. Report each diagnostic component with absolute error at most 2e-8, and Z with absolute error at most 2e-6 mEh.

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

01_pair_occupations

Goal
----
Enumerate the paired determinant space.

```python
def pair_occupations(norb: int, npair: int) -> "np.ndarray":
    """Enumerate the paired determinant space.
    
    Parameters
    ----------
    norb : int
        Number of spatial orbitals, 1 through 6.
    npair : int
        Number of doubly occupied orbitals, 1 through norb.
    Returns
    -------
    occupations : integer ndarray, shape (binom(norb,npair),npair)
        Strictly increasing zero-based tuples, in lexicographic order.
    Raises
    ------
    ValueError
        If either argument is not an integer or 1 <= npair <= norb <= 6 fails.
    """
    return occupations
```

### Step 2

02_doci_matrix

Goal
----
Project the factorized electronic Hamiltonian into the paired determinant basis.

```python
def doci_matrix(h: "np.ndarray", factors: "np.ndarray", occupations: "np.ndarray") -> "np.ndarray":
    """Project the factorized electronic Hamiltonian into the paired determinant basis.
    
    All inputs are real and finite and must not be modified. NumPy and SciPy are available.
    
    The spatial basis is orthonormal. For occupation tuple I, |D_I> has
    alpha creation operators in increasing spatial order followed by beta
    creation operators in the same order. The Hamiltonian is
    H = sum_{pq,s} h[p,q] a^dagger[p,s] a[q,s]
        + (1/2) sum_{l,pqrs,sigma,tau} L[l,p,q] L[l,r,s]
          a^dagger[p,sigma] a^dagger[r,tau] a[s,tau] a[q,sigma].
    No nuclear constant is included. Project this full Hamiltonian into
    the listed seniority-zero determinant subspace.

    Parameters
    ----------
    h : real ndarray, shape (n,n)
        Symmetric one-electron integrals in Eh.
    factors : real ndarray, shape (r,n,n)
        Symmetric factors in sqrt(Eh); the two-electron tensor is their unweighted outer-product sum.
    occupations : integer ndarray, shape (nd,k)
        Distinct, strictly increasing orbital tuples from pair_occupations, or a nonempty subset in any row order.
    Returns
    -------
    matrix : real ndarray, shape (nd,nd)
        DOCI Hamiltonian in Eh, matrix[I,J] = <D_I|H|D_J>, with the
        same row and column order as occupations.
    Raises
    ------
    ValueError
        If h/factors have incompatible shapes or are not symmetric, or an occupation row is out of range or not strictly increasing.
    """
    return matrix
```

### Step 3

03_doci_orbital_response

Goal
----
Evaluate the DOCI ground-state energy and its relaxed orbital gradient and Hessian.

```python
def doci_orbital_response(h: "np.ndarray", factors: "np.ndarray", occupations: "np.ndarray", orbitals: "np.ndarray") -> "tuple[float, np.ndarray, np.ndarray, np.ndarray]":
    """Differentiate the lowest DOCI eigenvalue with respect to orbital rotations.

    All arrays are finite and real. Inputs must not be modified. NumPy and
    SciPy are available. The electronic Hamiltonian and paired determinant
    convention are those of doci_matrix. At a real orthogonal orbital frame
    U, define M(U) = doci_matrix(U.T h U, {U.T L_l U}, occupations).
    For every p < q in lexicographic order, K_(p,q) has entry +1 at (p,q),
    -1 at (q,p), and zero elsewhere. If A(theta) = sum_a theta_a K_a,
    E(theta) is the lowest eigenvalue of M(U exp(A(theta))). The requested
    gradient and Hessian are the first and second derivatives of E(theta)
    at theta=0. The configuration coefficients are rediagonalized for every
    theta; derivatives holding them fixed do not define this energy surface.

    Parameters
    ----------
    h : real ndarray, shape (n,n)
        Symmetric one-electron matrix in Eh, 1 <= n <= 6.
    factors : real ndarray, shape (r,n,n)
        Symmetric two-electron factors in sqrt(Eh), 1 <= r <= 8.
    occupations : integer ndarray, shape (nd,k)
        Nonempty selection of distinct increasing orbital tuples, in any
        row order, with 1 <= k <= n. Orbital indices lie in [0,n).
    orbitals : real ndarray, shape (n,n)
        Orthogonal U; its columns are the spatial orbitals of the trial.
        Orthogonality is accepted within absolute tolerance 1e-10.

    Returns
    -------
    energy : float
        Lowest projected eigenvalue E(0), in Eh.
    coefficients : real ndarray, shape (nd,)
        Normalized ground eigenvector in the supplied occupation order.
        Its largest-magnitude component is positive, using the lowest index
        if several components have exactly equal maximum magnitude.
    gradient : real ndarray, shape (m,)
        Relaxed energy gradient in Eh per radian, m=n(n-1)/2.
    hessian : real ndarray, shape (m,m)
        Relaxed energy Hessian in Eh per radian squared, in the same order.
        For n=1, gradient and Hessian have shapes (0,) and (0,0).

    Raises
    ------
    ValueError
        For incompatible matrix/occupation shapes, nonsymmetric h or factors,
        invalid occupations, nonorthogonal orbitals, or a gap between the
        lowest two DOCI eigenvalues <= 1e-10 Eh. No gap is required when nd=1.
    """
    return energy, coefficients, gradient, hessian
```

### Step 4

04_relax_doci_trial

Goal
----
Construct a DOCI trial by a fixed number of damped orbital-relaxation updates.

```python
def relax_doci_trial(h: "np.ndarray", factors: "np.ndarray", npair: int, n_updates: int) -> "tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Relax a paired trial using the specified deterministic orbital schedule.

    Inputs are finite and real and must not be modified. NumPy and SciPy are
    available. Use the complete paired occupation basis from pair_occupations.
    Start with orbital frame U=I and rediagonalize the projected Hamiltonian
    after each accepted rotation. At every update, E, c, g and B are the
    relaxed orbital energy, coefficients, gradient and Hessian defined by
    doci_orbital_response, including configuration-coefficient response.

    The following numerical schedule defines this task's finite relaxation:
    if ||g||_2 <= 1e-10 Eh, retain U and record scale zero. Otherwise set
    mu=max(0,0.05 Eh-lambda_min(B)), solve (B+mu I)p=-g and shorten p, if
    necessary, to ||p||_2=0.25 radians. If -g.T p <= 1e-14 Eh, also retain U
    and record zero. Otherwise try alpha=2**(-j) for j=0,...,20 in that order.
    Accept the first U_new=U exp(alpha*sum_a p_a K_a) satisfying
    E(U_new) <= E(U)+1e-4*alpha*g.T p, with the generator ordering of
    doci_orbital_response. Each trial energy is rediagonalized. Perform
    exactly n_updates scheduled updates, including any zero-scale updates.
    The constants specify this finite replay, not a convergence criterion
    or a claim that the final orbitals are globally optimal.

    Parameters
    ----------
    h : real ndarray, shape (n,n)
        Symmetric one-electron matrix in Eh, 1 <= n <= 6.
    factors : real ndarray, shape (r,n,n)
        Symmetric two-electron factors in sqrt(Eh), 1 <= r <= 8.
    npair : int
        Number of alpha-beta electron pairs, 1 <= npair <= n.
    n_updates : int
        Number of scheduled orbital updates, 0 <= n_updates <= 6.

    Returns
    -------
    energy : float
        Final DOCI trial energy in Eh.
    coefficients : real ndarray, shape (nd,)
        Normalized final coefficients in lexicographic occupation order,
        with the sign convention of doci_orbital_response.
    orbitals : real ndarray, shape (n,n)
        Final real orthogonal trial orbital frame U.
    energy_history : real ndarray, shape (n_updates+1,)
        Energy before the first update and after each scheduled update.
    accepted_scales : real ndarray, shape (n_updates,)
        Accepted alpha for each update, or zero for a stationary update.

    Raises
    ------
    ValueError
        For incompatible or nonsymmetric integrals, npair or n_updates
        outside its integer range, a projected ground-state gap <= 1e-10 Eh
        at a retained frame with more than one configuration, or failure of
        all 21 trial scales to satisfy the stated decrease condition.
    """
    return energy, coefficients, orbitals, energy_history, accepted_scales
```

### Step 5

05_advance_walker

Goal
----
Advance an unrestricted walker through one force-biased imaginary-time step with a selected DOCI trial, retaining mixed estimators and the determinant normalization.

```python
def advance_walker(occupations: "np.ndarray", coefficients: "np.ndarray", phi_alpha: "np.ndarray", phi_beta: "np.ndarray", h: "np.ndarray", factors: "np.ndarray", field: "np.ndarray", dt: float, trial_orbitals: "np.ndarray | None" = None) -> "tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]":
    """Return the propagated walker and the quantities for its weight update.

    The Hamiltonian spatial basis is orthonormal. The trial ket is the sum
    of the listed paired determinants with the supplied complex coefficients.
    Each determinant occupies columns U[:, occupations[d]] in both spins;
    alpha creation operators precede beta creation operators, with increasing
    occupied frame indices within each spin. U is fixed and defaults to the
    identity. No orbital or coefficient optimization is requested.

    The Hamiltonian is
    H = sum_{pq,s} h[p,q] a^dagger[p,s] a[q,s]
        + (1/2) sum_{l,pqrs,sigma,tau} L[l,p,q] L[l,r,s]
          a^dagger[p,sigma] a^dagger[r,tau] a[s,tau] a[q,sigma],
    without a nuclear constant. Define
    v_l = i sum_{pq,s} L[l,p,q] a^dagger[p,s] a[q,s].
    For any walker |W>, O(W)=<Psi_T|W>, E(W)=<Psi_T|H|W>/O(W),
    and f_l(W)=<Psi_T|v_l|W>/O(W). Only the summed trial overlap
    is an estimator denominator; zero overlaps of individual components
    can coexist with nonzero one- or two-body matrix elements.

    The force-biased one-body propagator is
    B = exp(-dt*h0/2)
        @ exp(i*sqrt(dt)*sum_l (field[l]-bar_x[l])*L[l])
        @ exp(-dt*h0/2),
    with h0=h-(1/2)*sum_l L[l]@L[l] and
    bar_x=-sqrt(dt)*f(Phi) evaluated on the input walker.
    For each spin, B@phi=Q@R is the thin QR factorization whose R diagonal
    is real positive. |B Phi>=gauge*|Q_alpha Q_beta>, with
    gauge=det(R_alpha)*det(R_beta). No weight update is performed here.

    Parameters
    ----------
    occupations : integer ndarray, shape (nd,k)
        Distinct, strictly increasing paired occupation tuples with indices
        in [0,n), 1 <= k <= min(n,6), 1 <= n <= 28, and 1 <= nd <= 32.
        A selected subset in any row order is permitted.
    coefficients : real ndarray, shape (nd,2)
        Packed complex ket coefficients, not necessarily normalized.
    phi_alpha, phi_beta : real ndarrays, shape (n,k,2)
        Packed complex full-column-rank walker orbitals. The spin sectors
        are independent and need not be orthonormal or paired.
    h : real ndarray, shape (n,n)
        Symmetric one-electron integrals in Eh.
    factors : real ndarray, shape (r,n,n)
        Symmetric two-electron factors in sqrt(Eh), 1 <= r <= 8.
    field : real ndarray, shape (r,)
        Prescribed standard-normal coordinates, one per factor.
    dt : float
        Nonnegative imaginary-time step in inverse Eh. At dt=0 the
        propagator is the identity, but the input walker is still normalized.
    trial_orbitals : real ndarray, shape (n,n,2), or None
        Packed complex unitary matrix U in the Hamiltonian basis.
        U^dagger U=I is a precondition. None means U=I. Complex conjugation
        of this ket frame and the coefficients defines the trial bra.
        All inputs are finite and must not be modified. Packed arrays have
        final axis [real, imaginary]. NumPy and SciPy are available;
        earlier public functions may be called.

    Returns
    -------
    q_alpha, q_beta : real ndarrays, shape (n,k,2)
        Packed normalized spin orbitals Q with the positive-diagonal QR gauge.
    old_overlap, new_overlap : real ndarrays, shape (2,)
        Packed O(Phi) and O(Q), respectively. new_overlap excludes gauge.
    old_energy, new_energy : real ndarrays, shape (2,)
        Packed E(Phi) and E(Q) in Eh, retaining their imaginary parts.
    old_force : real ndarray, shape (r,2)
        Packed f(Phi) in sqrt(Eh), before multiplication by -sqrt(dt).
    log_gaussian : real ndarray, shape (2,)
        Packed field dot bar_x - (bar_x dot bar_x)/2. Both dots are bilinear.
    gauge : real ndarray, shape (2,)
        Packed det(R_alpha)*det(R_beta) for the positive-diagonal convention.

    Raises
    ------
    ValueError
        If packed shapes or trial/walker/Hamiltonian/field dimensions
        disagree, the dimensional bounds fail, an occupation row is invalid,
        h or a factor is not symmetric, or dt<0. Also raised if the magnitude
        of either total trial overlap is <=1e-12, or any propagated thin-QR
        diagonal magnitude is <=1e-12.
    """
    return q_alpha, q_beta, old_overlap, new_overlap, old_energy, new_energy, old_force, log_gaussian, gauge
```

### Step 6

06_phaseless_weight

Goal
----
Update the positive importance weight using the physical overlap phase.

```python
def phaseless_weight(old_overlap: "np.ndarray", new_overlap: "np.ndarray", gauge: "np.ndarray", log_gaussian: "np.ndarray", weight: float, dt: float, energy_shift: float) -> "tuple[float, float]":
    """Update the positive importance weight using the physical overlap phase.
    
    Packed complex arrays have a final axis [real, imaginary]. All inputs are finite and must not be modified. NumPy and SciPy are available; earlier public functions may be called.
    
    Parameters
    ----------
    old_overlap, new_overlap, gauge, log_gaussian : real ndarrays, shape (2,)
        Packed complex scalars. new_overlap is measured on normalized Q walkers;
        gauge converts Q to the raw propagated determinant and may be complex for this standalone function.
    weight : float
        Current nonnegative weight.
    dt : float
        Nonnegative time step, inverse Eh.
    energy_shift : float
        Real constant in Eh.
    Returns
    -------
    updated_weight : float
        weight*abs(I)*max(0,cos(theta)), where S=gauge*new_overlap/old_overlap,
        I=S*exp(log_gaussian+dt*energy_shift), theta=arg(S).
    phase : float
        theta in radians, the principal argument from numpy.angle.
    Raises
    ------
    ValueError
        If abs(old_overlap)<=1e-12, weight<0, or dt<0.
    """
    return updated_weight, phase
```

### Step 7

07_projection_block

Goal
----
Replay a finite orbital-relaxed DOCI-guided phaseless projection block.

```python
def projection_block(h: "np.ndarray", factors: "np.ndarray", npair: int, walkers: "np.ndarray", weights: "np.ndarray", fields: "np.ndarray", dt: float, energy_shift: float, n_updates: int = 4) -> "tuple[float, np.ndarray, np.ndarray, np.ndarray]":
    """Replay a finite DOCI-guided phaseless projection block.
    
    Packed complex arrays have a final axis [real, imaginary]. All inputs are finite and must not be modified. NumPy and SciPy are available; earlier public functions may be called.
    
    h is a real symmetric (n,n) matrix in Eh, factors is a real symmetric (r,n,n) array in sqrt(Eh), with 1<=n<=6, 1<=r<=5 and 1<=npair<=n. The spatial basis is orthonormal; each spin sector has npair electrons. H=sum(h[p,q] a†[p,s]a[q,s]) + 1/2 sum_l,p,q,r,s,spin,spin' L_l[p,q]L_l[r,s] a†[p,spin]a†[r,spin']a[s,spin']a[q,spin]. No nuclear constant is included. The DOCI basis is the complete set of paired occupations. At each retained trial frame, the lowest DOCI eigenvalue is isolated by more than 1e-10 Eh when the paired space has more than one configuration. A walker is two complex (n,npair) orbital matrices. walkers has packed shape (nw,2,n,npair,2), weights is real (nw,), and fields is a real (nt,nw,r) prescribed replay array with 1<=nw<=6 and 0<=nt<=10. dt>=0 is in inverse Eh and energy_shift in Eh. All live total trial overlaps exceed 1e-12 and initial and propagated QR diagonal magnitudes exceed 1e-12, except dedicated error tests. Do not sample new fields, constrain walkers to seniority zero, reconfigure the population, clip local energies, clip force bias, subtract a mean field, or renormalize weights.
    Parameters
    ----------
    h, factors, npair, walkers, weights, fields, dt, energy_shift
        As specified in the common model contract above. The orbitals and weights describe the initial importance-sampled ensemble.
    n_updates : int, default 4
        Number of orbital-relaxation updates, 0 <= n_updates <= 6,
        using exactly the protocol of relax_doci_trial. Construct this
        trial once before the replay. Its columns remain in the input
        Hamiltonian basis; use them in every overlap, force and local
        energy contraction. The Hamiltonian and walkers retain their
        input basis. Zero updates gives the fixed-frame DOCI trial.
    Returns
    -------
    trial_energy : float
        DOCI variational energy after n_updates orbital updates, Eh.
    energies : real ndarray, shape (nt+1,)
        E_t=sum_w weight_tw*Re(E_local_tw)/sum_w weight_tw, including t=0 and each completed time step.
    weight_history : real ndarray, shape (nt+1,nw)
        Absolute importance weights, including the unchanged initial weights; no normalization/reconfiguration.
    phase_history : real ndarray, shape (nt,nw)
        Physical overlap angles in time-major order. A walker already at zero weight is skipped and its later angle is zero.
        At each step evaluate force on the old walker, propagate, normalize by QR, evaluate the new overlap,
        update the weight, and then measure the energy of the new weighted ensemble.
    Raises
    ------
    ValueError
        If block shapes disagree, a weight or dt is negative, initial total weight is nonpositive,
        total surviving weight is <=1e-14, or an earlier step raises a documented ValueError.
    """
    return trial_energy, energies, weight_history, phase_history
```

### Step 8

08_solve_doci_projection

Goal
----
Return the finite-block mixed-energy increment relative to the orbital-relaxed DOCI trial.

```python
def solve_doci_projection(h: "np.ndarray", factors: "np.ndarray", npair: int, walkers: "np.ndarray", weights: "np.ndarray", fields: "np.ndarray", dt: float, energy_shift: float, n_updates: int = 4) -> float:
    """Return the finite-block mixed-energy increment relative to the DOCI trial.
    
    Packed complex arrays have a final axis [real, imaginary]. All inputs are finite and must not be modified. NumPy and SciPy are available; earlier public functions may be called.
    
    h is a real symmetric (n,n) matrix in Eh, factors is a real symmetric (r,n,n) array in sqrt(Eh), with 1<=n<=6, 1<=r<=5 and 1<=npair<=n. The spatial basis is orthonormal; each spin sector has npair electrons. H=sum(h[p,q] a†[p,s]a[q,s]) + 1/2 sum_l,p,q,r,s,spin,spin' L_l[p,q]L_l[r,s] a†[p,spin]a†[r,spin']a[s,spin']a[q,spin]. No nuclear constant is included. The DOCI basis is the complete set of paired occupations. At each retained trial frame, the lowest DOCI eigenvalue is isolated by more than 1e-10 Eh when the paired space has more than one configuration. A walker is two complex (n,npair) orbital matrices. walkers has packed shape (nw,2,n,npair,2), weights is real (nw,), and fields is a real (nt,nw,r) prescribed replay array with 1<=nw<=6 and 0<=nt<=10. dt>=0 is in inverse Eh and energy_shift in Eh. All live total trial overlaps exceed 1e-12 and initial and propagated QR diagonal magnitudes exceed 1e-12, except dedicated error tests. Do not sample new fields, constrain walkers to seniority zero, reconfigure the population, clip local energies, clip force bias, subtract a mean field, or renormalize weights.
    Parameters
    ----------
    h, factors, npair, walkers, weights, fields, dt, energy_shift
        Same contract and replay prescription as projection_block.
    n_updates : int, default 4
        Number of orbital-relaxation updates, 0 <= n_updates <= 6,
        using exactly the protocol of relax_doci_trial. Construct this
        trial once before the replay. Its columns remain in the input
        Hamiltonian basis; use them in every overlap, force and local
        energy contraction. The Hamiltonian and walkers retain their
        input basis. Zero updates gives the fixed-frame DOCI trial.
    Returns
    -------
    increment : float
        1000*(energies[-1]-trial_energy), in mEh. Use the full pipeline.
    Raises
    ------
    ValueError
        Propagate every documented input, overlap, rank, spectral-gap or all-walkers-dead error from projection_block.
    """
    return increment
```

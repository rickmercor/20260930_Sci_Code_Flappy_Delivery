# Chemistry-Computational_Chemistry-46

## Background

In a QM/MM simulation, the system splits into a quantum (QM) region treated with electronic structure theory and a classical (MM) region described by a potential energy function. If the MM region is fixed or only carries static charges, it cannot respond to the field it receives or generates, so any electrostatic feedback between the two regions is lost. A polarizable embedding (PE) treatment removes that limitation: both subsystems carry moments — permanent plus induced — that adjust self-consistently to each other's field, so the total energy becomes a functional of the full, converged moment set rather than a fixed charge distribution. Extending this consistently to a system that is periodic in two dimensions, such as a solid/liquid interface, means the induced moments must also account for the field from every periodic image of every site, not just the sites in one unit cell.

Each classical site (e.g. a water molecule) can be assigned permanent multipole moments about its center of mass — dipole, quadrupole, and higher orders — plus polarizability tensors that couple different multipole ranks (dipole responding to the local field, quadrupole responding to the field gradient, and so on). The total classical electrostatic energy then decomposes into a permanent-plus-polarized interaction term and a polarization self-energy term, the latter accounting for the energetic cost of inducing a moment against a site's own restoring polarizability. At self-consistency, the induced moment at each site is exactly the linear response of that site's polarizability to the total on-site field — summed over every other site's permanent and induced moments, including periodic images — so solving for these moments requires an iterative procedure, directly analogous to a self-consistent field cycle in electronic structure theory.

The bare Coulomb interaction tensor between point multipoles diverges as two sites approach each other, and can drive the induced-moment iteration itself to diverge — a polarization catastrophe. Any workable scheme must therefore damp, or regularize, the interaction tensor at short range while leaving it essentially unchanged at long range. Two different situations call for two different constructions. When both interacting partners are themselves point multipoles with no independent spatial extent, the natural fix is to damp each rank of the interaction tensor directly — a family of damping functions, one per tensor rank, that recursively regularize the monopole, dipole, quadrupole, and higher-order terms so that none of them diverges as the separation goes to zero. When one partner is instead a spatially extended charge distribution (such as a QM electron density) interacting with a point multipole, damping the interaction pointwise, grid point by grid point, makes the damping depend on the orientation of the classical site relative to the QM density rather than on separation alone — an inconsistency, since the physical screening of an overlapping charge distribution should depend only on how close the two centers are. The consistent fix is an isotropic damping envelope defined purely by the distance between the two centers of mass: it screens the interaction toward zero as the centers coincide and recovers the full, undamped interaction once they are well separated, independent of how either object is oriented.

A 2D-periodic system requires summing interactions over infinitely many periodic images of every site, and explicitly evaluating each individual image multipole becomes wasteful once the images are far enough away that fine angular detail no longer affects the result appreciably — at long range, a group of nearby image sites is electrostatically almost
indistinguishable from a single point multipole located at their combined center. This motivates a compression step: distant image sites are grouped spatially (for example, with the K-means algorithm, alternating between assigning each site to its nearest center and recomputing each center as the mean of its assigned sites) into a small number of cluster centers, each contributing site's multipole moments are analytically translated from its own origin to the assigned cluster centroid, and the translated moments are summed at each center. The compressed centers then interact with the sites of interest through the ordinary (undamped) long-range multipole expansion, trading a large number of near-exact terms for a small number of compressed ones with minimal loss of accuracy in the converged electrostatics.

## Problem

Point-multipole polarizable-embedding QM/MM schemes must damp the electrostatic coupling
between the quantum (QM) region and the surrounding classical (MM) point-multipole sites at
short range, or the induced-moment iteration diverges as a "polarization catastrophe" when
QM electron density and an MM site come into close contact. Prior schemes damp this coupling
per real-space grid point, which makes the damping depend on molecular orientation rather
than separation alone; the method evaluated here instead damps the QM–MM boundary
isotropically, as a function of the center-of-mass distance between the QM region and each
nearby MM site, while MM–MM near-field pairs keep the standard (orientation-independent)
Gaussian tensor damping used within the classical sublattice, and electrostatically distant
periodic images of both the QM region and the MM sites are compressed together into a
K-means-clustered, origin-shifted multipole expansion. The primary inputs are the QM
region's and MM sites' positions and permanent multipole moments, the MM polarizability
tensors, the isotropic and tensor damping parameters, and the 2D lattice vectors of one
unit cell; the required output is a single scalar, the converged total
electrostatic-plus-induction energy of the coupled system.

For this problem the QM region is represented, as in the method's own far-field treatment,
by a single fixed point multipole (it does not carry a polarizability and its moments are
not updated); only the MM sites respond self-consistently. Near-field QM–MM pairs (within
the specified lattice-translation cutoff) contribute their bare multipole interaction energy
multiplied by the isotropic center-of-mass damping factor of the source method, evaluated at
the given damping parameter; near-field MM–MM pairs instead use the source method's Gaussian
tensor damping of the interaction tensor (rank 0 through rank 4, since permanent moments run
up to quadrupole) at its own damping width. Given the total on-site field and field gradient
at each MM site — from every other MM site, from the fixed QM multipole, and from all of
their near-field periodic images — the induced dipole and induced quadrupole at that site
follow from its polarizability tensors and must be found by iterating to self-consistency
with linear mixing between updates.

Periodic images of both the QM region and the MM sites lying beyond the near-field cutoff
are grouped together in the xy-plane by K-means clustering; each contributing image's
permanent multipole moments are analytically translated from its own origin to its assigned
cluster centroid, and the resulting cluster moments interact with the home-cell sites
through the bare (undamped) multipole tensor. Using the geometry, moments, polarizabilities,
damping parameters, near/far cutoff, and clustering parameters specified below, determine
the fully converged total electrostatic-plus-induction energy of the coupled QM–MM system,
in atomic units (Hartree).

 Data:

Lattice (2D-periodic in x, y; z non-periodic): a₁ = (4.20, 0.00, 0.00) Å, a₂ = (0.00,
4.20, 0.00) Å.

QM site (fixed point multipole, no polarizability; Cartesian COM position, Å;
charge-neutral, q = 0; moments in atomic units, quadrupole traceless):

| x | y | z | μ_x | μ_y | μ_z | Θ_xx | Θ_yy | Θ_xy | Θ_xz | Θ_yz |
|---|---|---|---|---|---|---|---|---|---|---|
| 2.10 | 2.10 | 1.50 | 0.10 | 0.05 | −0.60 | 0.20 | 0.15 | −0.05 | 0.10 | −0.08 |
MM sites (Cartesian COM positions, Å; all sites charge-neutral, q = 0):

| Site | x | y | z |
|---|---|---|---|
| 1 | 0.55 | 0.40 | 0.00 |
| 2 | 2.95 | 1.35 | 0.30 |
| 3 | 1.65 | 3.10 | −0.25 |

*MM permanent moments (atomic units; quadrupole given as independent Cartesian
components, traceless so Θ_zz = −(Θ_xx + Θ_yy)):

| Site | μ_x | μ_y | μ_z | Θ_xx | Θ_yy | Θ_xy | Θ_xz | Θ_yz |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.42 | −0.28 | 0.09 | 0.75 | −0.45 | 0.22 | −0.12 | 0.05 |
| 2 | −0.18 | 0.52 | −0.06 | −0.38 | 0.85 | −0.09 | 0.18 | −0.22 |
| 3 | 0.06 | −0.38 | 0.55 | 0.28 | 0.12 | 0.06 | −0.28 | 0.14 |

MM polarizabilities (atomic units, identical at every MM site, isotropic): dipole–dipole
α = 9.85; dipole–quadrupole coupling A = 0 (neglected for this problem); quadrupole–
quadrupole C = 24.5, acting diagonally on the traceless quadrupole response.

QM–MM isotropic damping: use the source method's isotropic real-space damping factor for
the QM/MM boundary, evaluated at the QM-site–MM-site center-of-mass separation, with
damping parameter β = 0.291 Å⁻¹.

MM–MM tensor damping: use the source method's Gaussian-type damping of the interaction
tensor, with width parameter g = 0.32 Å, applied to every near-field MM–MM pair.

Near/far split: near field = the QM site plus the 3 MM sites, together with their
periodic images at lattice translations n_x, n_y ∈ {−1, 0, 1} (Nc = [1,1,0]), excluding
self-pairs, interacting via the damping rules above (isotropic for QM–MM pairs, tensor for
MM–MM pairs). Far field = periodic images (of both the QM site and the MM sites) at
n_x, n_y ∈ {−2, −1, 0, 1, 2} excluding the near-field cells already counted, interacting via
the undamped tensor after K-means compression.

Self-image convention: in the near field an MM target omits only its own zero-translation
term but still interacts with its own periodic near images, whereas the QM target omits its
entire near-field self-lattice — its home instance and all of its near-field periodic
images. The far-field K-means clusters include every site's far images and contribute to every target, QM and MM alike, through the compressed cluster moments; so the QM target does see its own far-field images, but only in compressed form.

Clustering: K = 3 cluster expansion centers, K-means run on the in-plane (x, y)
coordinates of all far-field image sites (QM and MM images pooled together), k-means++
initialization, n_init = 10, random_state = 0, tol = 1e-6 (scikit-learn `KMeans`
convention). The z coordinate of each 3D cluster center is the mean z of the far-field images assigned to it (the clustering itself uses only the x, y coordinates).

SCF: linear mixing parameter t = 0.35 between successive induced-moment updates (MM sites
only), initial guess Δμ = Δθ = 0 at every MM site, converged when the largest absolute
change in any Δμ or Δθ component between iterations is below 1e-8 a.u.

Output Format Requirements:

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before
the tags.

You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if
the value is approximate or you are unsure.

Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not
  Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.

Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the
final number. Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

Isotropic qmmm damping

Goal
----
Compute the isotropic real-space damping factor S^G for one or more QM-site--MM-site centre-of-mass separations, given the damping parameter beta.  The result is a dimensionless scalar in [0, 1] for each separation; downstream it multiplies the bare QM--MM multipole interaction (and the field / field gradient it produces) for every near-field QM--MM pair, including periodic images.

```python
def isotropic_qmmm_damping(r_com, beta: float) -> np.ndarray:
    '''Isotropic real-space QM-MM boundary damping factor S^G.

    Parameters
    ----------
    r_com : array_like
        QM-site--MM-site centre-of-mass separations, shape (...,).  Must be finite and
        non-negative.  Same length unit as ``1 / beta``.
    beta : float
        Isotropic damping parameter (strictly positive, inverse length).

    Returns
    -------
    S_G : np.ndarray
        Array with the same shape as ``r_com`` holding the scalar damping prefactor
        S^G(d) = erf(beta d) - (2/sqrt(pi)) beta d exp(-(beta d)^2), clipped to [0, 1].

    Raises
    ------
    ValueError
        If ``r_com`` contains a non-finite value or a negative separation, or if
        ``beta`` is not finite or is not strictly positive.
    '''
    return np.zeros_like(np.asarray(r_com, dtype=float))
```

### Step 2

Gaussian tensor damping kernels

Goal
----
Build the Gaussian-damped Cartesian multipole interaction tensors of ranks 0 through 4 for a given displacement between two MM centres of mass and a Gaussian width g.  These are used for every near-field MM--MM pair by the multipole potential calculator

```python
import numpy as np
from scipy.special import erf

_SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 13), 3: slice(13, 40), 4: slice(40, 121)}
_TOTAL = 121

def gaussian_damped_tensors(r_vec, g=None) -> np.ndarray:
    # Flat layout of a "tensor set": [T0 | T1(3) | T2(3,3) | T3(3,3,3) | T4(3,3,3,3)]
    '''Gaussian-damped Cartesian interaction tensors T^{ij,d} of ranks 0..4.

    Parameters
    ----------
    r_vec : array_like, shape (3,)
        Displacement between the two MM centres of mass (field point minus source).
        Must be a finite, non-zero 3-vector.
    g : float or None
        Gaussian damping width (strictly positive, same length unit as ``r_vec``).
        ``g = None`` returns the bare (undamped) tensor set -- the g -> 0 limit in which
        every kernel lambda_k -> 1.

    Returns
    -------
    tensors : np.ndarray, shape (121,)
        The five (damped) tensors flattened row-major and concatenated in the order
        [T0 (1), T1 (3), T2 (9), T3 (27), T4 (81)].

    Raises
    ------
    ValueError
        If ``r_vec`` is not a finite 3-vector or is the zero vector, or if ``g`` is
        given (not ``None``) but is not finite or is not strictly positive.
    '''
    return np.zeros(_TOTAL, dtype=float)
```

### Step 3

Multipole potential calculator

Goal
----
Compute the electrostatic potential (rank 0), field (rank 1) and field gradient (rank 2)produced at a target point by a source carrying a permanent charge q, dipole mu and traceless quadrupole Theta (Buckingham normalisation, Theta_ab = 1/2 sum q (3 r_a r_b - r^2 delta_ab))

```python
import numpy as np

# Flat layout of a rank-0..4 "tensor set": [T0 | T1(3) | T2(3,3) | T3(3,3,3) | T4(3,3,3,3)]
_SLICES = {0: slice(0, 1), 1: slice(1, 4), 2: slice(4, 13), 3: slice(13, 40), 4: slice(40, 121)}
_TOTAL = 121

def multipole_potential_field(r_vec, q: float, mu, theta, tensors=None, iso: float = 1.0) -> np.ndarray:
    '''Potential, field and field gradient at a target point from a permanent multipole.

    Parameters
    ----------
    r_vec : array_like, shape (3,)
        Displacement r_target - r_source (finite, non-zero).
    q : float
        Source permanent charge.
    mu : array_like, shape (3,)
        Source permanent dipole.
    theta : array_like
        Source permanent quadrupole, either a (3,3) traceless matrix or the 5-vector
        [Theta_xx, Theta_yy, Theta_xy, Theta_xz, Theta_yz].
    tensors : array_like, shape (121,), optional
        Pre-built damped interaction-tensor set (flat layout [T0(1) | T1(3) | T2(9) |
        T3(27) | T4(81)], as produced by step 2).  If ``None`` the bare (undamped)
        tensors are built internally from ``r_vec``.
    iso : float, optional
        Scalar isotropic prefactor applied to the whole result (default 1.0); this is
        where the QM--MM envelope S^G multiplies in.

    Returns
    -------
    out : np.ndarray, shape (13,)
        [phi, V_x, V_y, V_z, V_xx, V_xy, V_xz, V_yx, V_yy, V_yz, V_zx, V_zy, V_zz],
        with V_a = -d phi/d r_a and V_ab = -d^2 phi/d r_a d r_b.

    Raises
    ------
    ValueError
        If ``r_vec`` is not a finite 3-vector or is the zero vector; if ``mu`` is not a
        finite 3-vector or ``q`` / ``iso`` is not finite; if ``theta`` is not a (3, 3)
        matrix or a 5-vector, or is not traceless; or if ``tensors`` is supplied with a
        length other than 121.
    '''
    return np.zeros(13, dtype=float)
```

### Step 4

Multipole origin translation

Goal
----
Re-express a site's permanent dipole and quadrupole about a new origin, given the shift vector R = R_new - R_i (new origin minus the site's own origin) and the site charge q.

```python
import numpy as np

def translate_multipole(mu, theta, R, q: float = 0.0) -> np.ndarray:
    '''Translate a site's permanent dipole and quadrupole to a new origin.

    Parameters
    ----------
    mu : array_like, shape (3,)
        Dipole about the site's own origin.
    theta : array_like
        Quadrupole about the site's own origin, either a (3,3) traceless matrix or the
        5-vector [Theta_xx, Theta_yy, Theta_xy, Theta_xz, Theta_yz].
    R : array_like, shape (3,)
        Shift vector R = R_new - R_i (new common origin minus this site's origin).
    q : float, optional
        Site charge (default 0.0).

    Returns
    -------
    shifted : np.ndarray, shape (12,)
        [mu'_x, mu'_y, mu'_z] followed by the row-major (3,3) shifted quadrupole
        Theta'_ab.

    Raises
    ------
    ValueError
        If ``mu`` or ``R`` is not a 3-vector; if ``mu`` / ``R`` / ``q`` is not finite;
        or if ``theta`` is not a (3, 3) matrix or a 5-vector.   
    '''
    return np.zeros(12, dtype=float)
```

### Step 5

K-means far-field centre compression

Goal
----
Partition the pooled far-field periodic image sites, by their in-plane (x, y) coordinates only, into K expansion centres by K-means with k-means++ initialisation and Lloyd iteration, implemented with numpy alone so the step is self-contained.  The reference runs a fixed floor of 64 random restarts and keeps the lowest-inertia clustering.  The prompt states the clustering parameters as K = 3, n_init = 10, random_state = 0, tol = 1e-6 (the scikit-learn ``KMeans`` convention); this reference uses `$random_state$` to seed the restarts and ``tol`$to bound the per-step centre shift as stated, but treats$$n_init$` only as a lower bound on the restart count -- not the exact restart budget -- because plain k-means++ needs more restarts than scikit-learn's greedy variant to reach the same optimum.  At the benchmark K = 3 the 64-restart clustering equals the global optimum ``sklearn.cluster.KMeans(n_init=10)`` finds; at other K the two implementations may settle on different local optima.  Each centre's z coordinate is the mean z of the sites assigned to it.  This step produces only the clustering; the orchestrator then origin-shifts each image's permanent moments onto its assigned centre and sums them.

```python
import numpy as np

_MIN_RESTARTS = 64

def kmeans_expansion_centers(xy, z, n_clusters: int = 3, random_state: int = 0,
                             n_init: int = 10, tol: float = 1e-6) -> np.ndarray:
    '''K-means expansion centres for the pooled far-field image sites.

    Parameters
    ----------
    xy : array_like, shape (N, 2)
        In-plane (x, y) coordinates of the pooled far-field image sites.
    z : array_like, shape (N,)
        z coordinates of the same sites.
    n_clusters : int
        Number of expansion centres K (default 3).
    random_state : int
        Seed for ``numpy.random.default_rng`` -- fixes the k-means++ seeding.
    n_init : int
        Requested number of random restarts (the reference uses at least a fixed floor).
    tol : float
        Convergence threshold on the summed centre shift per Lloyd step.

    Returns
    -------
    packed : np.ndarray, shape (3*n_clusters + N,)
        The K centre positions [cx, cy, cz] (cz = mean z of assigned sites), ordered by
        ascending cx then cy, followed by the N per-site cluster labels as floats --
        each an index into that ordered centre list.

    Raises
    ------
    ValueError
        If ``xy`` is not shape (N, 2) or ``z`` is not shape (N,); if any input is not
        finite; or if ``n_clusters`` is not an integer in [1, N].
    '''
    return np.zeros(3 * n_clusters, dtype=float)
```

### Step 6

MM inner SCF loop

Goal
----
Solve self-consistently for the induced dipole and induced quadrupole at each of the three MM sites (the QM site is a fixed point multipole and does not respond), then return the converged total electrostatic-plus-induction energy of the coupled system.

```python
import numpy as np
from scipy.special import erf
 
def mm_scf_energy(perm_mu, perm_th, v_const, g_const, mm_terms, alpha: float, C: float,
                  mix: float = 0.35, tol: float = 1e-8, max_iter: int = 20000) -> float:
    '''Solve the MM induced-moment SCF and return the coupled electrostatic+induction energy.
 
    Parameters
    ----------
    perm_mu : array_like, shape (4, 3)
        Permanent dipole of each target -- row 0 QM, rows 1-3 the MM sites (a.u.).
    perm_th : array_like, shape (4, 3, 3) or (4, 5)
        Permanent traceless quadrupole of each target, same ordering.
    v_const, g_const : array_like, shape (4, 3) and (4, 3, 3)
        SCF-independent on-site field and field gradient at each target (fixed QM
        multipole plus far-field expansion centres).
    mm_terms : list of length 4
        ``mm_terms[t]`` is a list of ``(j, T2, T3, T4, iso)`` MM-source contributions to
        target ``t`` (``j`` in 0..2 the MM site index; ``T2`` (3,3), ``T3`` (3,3,3),
        ``T4`` (3,3,3,3); ``iso`` a scalar prefactor).
    alpha, C : float
        Isotropic MM dipole-dipole and quadrupole-quadrupole polarizabilities (a.u.).
        The induced dipole is Delta_mu = alpha * V and the induced quadrupole is
        C times the traceless part of the assembled field gradient,
        Delta_Theta = C * (V_grad - (1/3) tr(V_grad) I).
    mix : float
        Linear mixing parameter t in (0, 1) (default 0.35):
        Delta_M_new = (1 - t) * Delta_M_fresh + t * Delta_M_old.
    tol : float
        SCF convergence threshold (default 1e-8) on the largest absolute change of any
        induced component between the mixed update Delta_M_new and Delta_M_old.
    max_iter : int
        Safety cap on SCF iterations.
 
    Returns
    -------
    energy : float
        Converged total electrostatic-plus-induction energy of the coupled QM--MM
        system, in Hartree.
 
    Raises
    ------
    ValueError
        If ``perm_mu`` is not shape (4, 3), ``v_const`` not (4, 3) or ``g_const`` not
        (4, 3, 3); if ``perm_th`` is not (4, 3, 3) or (4, 5); if ``mm_terms`` does not
        have length 4; if ``mix`` is not strictly inside (0, 1); if ``tol`` is not
        finite and positive; or if ``alpha`` / ``C`` is not finite or ``alpha`` is
        negative.
    RuntimeError
        If the SCF does not converge within ``max_iter`` iterations.
 
    '''
    return energy
```

### Step 7

End-to-end coupled QM/MM polarizable-embedding energy

Goal
----
It assembles the concrete system, converts all lengths to Bohr, performs the near/far lattice split, compresses the far field with K-means and runs the MM induced-moment SCF to the converged coupled electrostatic-plus-induction energy, it returns that single scalar, in Hartree.

```python
import numpy as np
 
ANGSTROM_TO_BOHR = 1.8897261246
_TOTAL = 121
 
def coupled_qmmm_energy(beta_inv_angstrom: float = 0.291, g_angstrom: float = 0.32,
                        n_clusters: int = 3, mixing: float = 0.35, scf_tol: float = 1e-8) -> float:
    '''Full pipeline: converged coupled QM/MM electrostatic-plus-induction energy.
 
    Parameters
    ----------
    beta_inv_angstrom : float
        Isotropic QM--MM damping parameter beta, in A^-1 (default 0.291).
    g_angstrom : float
        Gaussian MM--MM tensor damping width, in A (default 0.32).
    n_clusters : int
        Number of K-means far-field expansion centres (default 3).
    mixing : float
        SCF linear mixing parameter t (default 0.35).
    scf_tol : float
        SCF convergence threshold on the largest induced-component change (default 1e-8).
 
    Returns
    -------
    energy : float
        The single requested scalar: the fully converged total
        electrostatic-plus-induction energy of the coupled QM/MM system, in Hartree.
        
    Raises
    ------
    ValueError
        If ``beta_inv_angstrom`` or ``g_angstrom`` is not finite or is not strictly
        positive, or if ``n_clusters`` is not a positive integer.
    '''
    return energy
```

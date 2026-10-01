# Physics-Condensed_Matter_Physics-17

## Background

Localized electronic orbitals are not generally orthogonal, so particle populations and force estimates must account for their overlap geometry. A sparse density alone is insufficient for the overlap contribution to the force; an energy-weighted kernel is also required. Signed spatial probing and compact spectral projections offer a way to obtain these operators without manipulating every full-space matrix function. In a finite charge-feedback calculation, compression, element selection, population control and the changing potential interact, making consistent operator-action representations essential.

## Problem

Compute the frozen-potential electronic force on orbital $a=0$, $F_a=-\operatorname{Tr}(\rho\,\partial_aH)+\operatorname{Tr}(\rho_E\,\partial_aS)$ in eV/bohr, after the finite charge-feedback calculation below; this is the specified one-spin compressed-density force estimator, not a converged total force or a finite-difference derivative of the numerical orbital energy. Use $n=18$, $L=24$ bohr, 0-based fixed orbital order and positions `[0.00,0.72,1.83,3.05,4.12,5.55,6.44,7.90,9.15,10.02,11.48,12.73,14.05,15.22,16.87,18.31,20.12,22.01]`, signed images $\Delta_{ij}=x_i-x_j-L\lfloor(x_i-x_j)/L+1/2\rfloor$, and distances $d_{ij}=|\Delta_{ij}|$, holding image branches and support masks fixed when differentiating each upper-triangle smooth matrix element and mirroring its derivative. The real symmetric overlap is $S_{ii}=1+0.03\cos(2\pi x_i/L)$ and $S_{ij}=0.16e^{-d_{ij}/0.85}[1+0.05\cos(2\pi(x_i+x_j)/L)]$ for $i\ne j$ with $d_{ij}\le2.25$, while the bare Hamiltonian in eV is $H^0_{ii}=-0.25+0.35\cos(2\pi x_i/L)+0.08\sin(4\pi x_i/L)+0.025(-1)^i$ and $H^0_{ij}=-0.90e^{-d_{ij}/1.25}[1+0.10\cos(2\pi(x_i+x_j)/L)]$ for $i\ne j$ with $d_{ij}\le3.25$, with other offdiagonal entries zero.

Use distance-separated signed color probes with fixed-order greedy coloring, the smallest available nonnegative label, strict conflicts $d_{ij}<d_{\rm excl}$, increasing-color column order and unnormalized unit-magnitude entries: overlap parameters are $d_{\rm excl}^S=4.9$, $v_S=3$, signs `[1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,1,-1,1,-1]`, and density parameters are $d_{\rm excl}^{\rho}=7.3$, $v_\rho=2$, signs `[-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1]`. A depth-$v$ Krylov space is represented by two-pass modified Gram-Schmidt on the raw blocks $[B,AB,\ldots,A^{v-1}B]$ in block order then column order, dropping residual norms at most $10^{-12}\max(1,\|B\|_F)$ and orienting each normalized retained vector so its first largest-magnitude component is positive. Define symmetric extraction from an action $Z$ by $\mathcal E(Z)_{ij}=\tfrac12[s_jZ_{i,c_j}+s_iZ_{j,c_i}]$ followed by an inclusive spatial mask and then an offdiagonal magnitude mask that retains equality and always exempts the diagonal: the $(d_{\rm keep},\tau)$ settings are $(2.4,0.006)$ for $M\approx S^{-1/2}$, $(3.4,0.135\ \mathrm{eV})$ for the compressed orthogonal Hamiltonian, and $(3.6,0)$ for both density kernels. For the energy-weighted kernel, represent inverse overlap by two applications of the same extracted map $M$. Using the supplied distance-colored probe sets, finite-depth Krylov spaces, and extraction conventions, compute the overlap map, orthogonally transformed Hamiltonian, density, and energy-weighted density needed for the specified force.

The frozen interaction matrix is $\Gamma_{ij}=1.1\delta_{ij}+0.15e^{-d_{ij}/3}$ eV, the initial excess-electron population is $q_i^{(0)}=0.12\cos(2\pi i/18)+0.04\sin(4\pi i/18)$ minus its arithmetic mean, and the finite feedback specification is $V^{(k)}=\Gamma q^{(k)}$, $H^{(k)}_{ij}=H^0_{ij}+\tfrac12S_{ij}(V_i^{(k)}+V_j^{(k)})$, $q^{(k+1)}=(1-\alpha_k)q^{(k)}+\alpha_k[\operatorname{diag}(\rho^{(k)}S)-8/18]$ for $k=0,1,2$ with $(\alpha_0,\alpha_1,\alpha_2)=(0.6,0.35,0.75)$, followed by a fresh Hamiltonian and density evaluation at $q^{(3)}$ without another charge update. At each evaluation impose $\operatorname{Tr}(S\rho(\mu))=8$ at $\beta=32\ \mathrm{eV}^{-1}$ using the spatially extracted finite-depth density, with nonnegative projected-mode particle weights and their positive total; the chemical-potential convention is 64 bisections of $[\epsilon_{\min}-\max(2,50/\beta),\epsilon_{\max}+\max(2,50/\beta)]$ in eV, moving the lower bound when the count is below eight and the upper bound otherwise and taking the final midpoint, where the $\epsilon$ are the projected Hamiltonian eigenvalues. For the final force hold $q^{(3)}$ and $\Gamma$ fixed, use $\partial_aH_{ij}=\partial_aH^0_{ij}+\tfrac12\partial_aS_{ij}(V_i^{(3)}+V_j^{(3)})$, retain binary64 values with overflow-safe occupations and no intermediate rounding, round only $F_a$ to 10 decimal places with ties to even, and in `<reasoning>` give the compressed-action identities and particle-weight relation plus the decisive scalars $\|M\|_F$, $\mu^{(0)}$, $\mu^{(1)}$, the four-bit retained-support pattern of bond $(3,5)$ in the compressed orthogonal Hamiltonian at $k=0,1,2,3$, $\|q^{(3)}\|_2$, $\mu^{(3)}$, the final fixed-Hamiltonian $\partial N/\partial\mu$, $\|\rho^{(3)}\|_F$, $\operatorname{Tr}(\rho_E^{(3)})$, $\|\rho_E^{(3)}\|_F$, and the two force contributions; algebraically equivalent identities and numerical results at the stated precision are accepted.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

build_periodic_chain_tangent

Goal
----
Construct the periodic distances, overlap, bare Hamiltonian, and their frozen-support derivative with respect to one orbital position.

```python
from numbers import Integral, Real
import numpy as np

def build_periodic_chain_tangent(positions: np.ndarray, length: float, force_index: int) -> np.ndarray:
    """Construct the ring operators and the displacement tangent.

    Parameters
    ----------
    positions : np.ndarray
        Finite, strictly increasing real vector x of length n >= 2 in [0, length).
    length : float
        Finite positive ring length, in bohr.
    force_index : int
        Orbital index a in [0,n); Python and NumPy integers are accepted, not bool.

    Returns
    -------
    operators : np.ndarray
        Real array of shape (5,n,n), ordered [D,S,H0,dS,dH0]. D is in bohr,
        S is dimensionless, H0 is in eV, dS in bohr**(-1), and dH0 in eV/bohr.

    Raises
    ------
    ValueError
        If positions are not a finite strictly increasing real vector in the
        stated interval, length is not a positive finite scalar, or the index
        is not an integer in the stated range.

    Notes
    -----
    Set k=2*pi/length, delta_ij=x_i-x_j-length*floor((x_i-x_j)/length+1/2),
    D_ij=abs(delta_ij), and phase_ij=k*(x_i+x_j). The negative image is chosen
    at a half-ring tie. Differentiate each i<j pair once and mirror its
    derivative to j,i; this defines the one-sided tangent at a half-ring tie.
    For i != j:
    S_ij=0.16*exp(-D_ij/0.85)*(1+0.05*cos(phase_ij)) for D_ij<=2.25;
    H0_ij=-0.90*exp(-D_ij/1.25)*(1+0.10*cos(phase_ij)) for D_ij<=3.25.
    Other offdiagonal entries vanish. S_ii=1+0.03*cos(k*x_i) and
    H0_ii=-0.25+0.35*cos(k*x_i)+0.08*sin(2*k*x_i)+0.025*(-1)**i.
    The tangents differentiate these smooth expressions with respect to x_a,
    holding image integers and support masks fixed, including at a mask tie.
    No derivative of a cutoff or of a minimum-image switch is included.
    """
    return np.zeros((5,np.asarray(positions).size,np.asarray(positions).size),dtype=float)
```

### Step 2

build_colored_signed_probes

Goal
----
Greedily color a distance graph in orbital order and form its unnormalized signed probe block.

```python
from numbers import Integral, Real
import numpy as np

def build_colored_signed_probes(distances: np.ndarray, exclusion: float, signs: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Construct fixed-order distance colors and their signed probes.

    Parameters
    ----------
    distances : np.ndarray
        Finite real, symmetric, nonnegative (n,n) distances; n>=1 and zero diagonal.
        Symmetry and zero-diagonal tolerance is absolute 1e-12.
    exclusion : float
        Nonnegative finite distance; conflicts occur strictly below this value.
    signs : np.ndarray
        Real vector of shape (n,) with values exactly +1 or -1. Integer and
        floating dtypes are both valid; no normalization is performed.

    Returns
    -------
    colors : np.ndarray
        Integer vector (n,) in orbital order. Assign the smallest nonnegative
        label absent from all earlier conflicting vertices.
    probes : np.ndarray
        Real (n,m) block, m=1+max(colors), whose columns follow increasing labels.
        Row i has signs[i] at column colors[i] and zero elsewhere.

    Raises
    ------
    ValueError
        If the distance, exclusion, or sign constraints fail.
    """
    return np.zeros(np.asarray(signs).size,dtype=int),np.zeros((np.asarray(signs).size,0),dtype=float)
```

### Step 3

canonical_block_krylov_basis

Goal
----
Build a deterministic orthonormal block-Krylov basis with two-pass rank deflation.

```python
from numbers import Integral, Real
import numpy as np

def canonical_block_krylov_basis(
    operator: np.ndarray,
    block: np.ndarray,
    depth: int,
    rank_tolerance: float = 1e-12,
) -> np.ndarray:
    """Construct a canonical two-pass block-Krylov basis.

    Parameters
    ----------
    operator : np.ndarray
        Finite real symmetric matrix of shape (n, n).
    block : np.ndarray
        Finite starting block of shape (n, m), with at least one nonzero direction.
    depth : int
        Positive number of blocks in [B, A B, ..., A**(depth-1) B].
    rank_tolerance : float
        Positive relative threshold multiplying max(1, ||B||_F).

    Returns
    -------
    basis : np.ndarray
        Canonical orthonormal basis of shape (n, r). Symmetry tolerance is 1e-12.
        Process the raw blocks [B,AB,...] in block order then column order.
        For each candidate perform two sequential modified Gram-Schmidt passes
        against previously retained columns, discarding it when its residual
        norm <= rank_tolerance*max(1,||B||_F). Normalize accepted columns and
        flip their sign if the first largest-magnitude component is negative.

    Raises
    ------
    ValueError
        If shapes, symmetry, finiteness, depth, tolerance, or numerical rank are invalid.
    """
    return np.empty((operator.shape[0], 0), dtype=float)
```

### Step 4

projected_inverse_sqrt_action

Goal
----
Evaluate the compressed overlap inverse-square-root action on arbitrary right-hand sides in a supplied orthonormal subspace.

```python
from numbers import Integral, Real
import numpy as np

def projected_inverse_sqrt_action(
    overlap: np.ndarray,
    probes: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Apply the Galerkin inverse square root in a supplied subspace.

    Parameters
    ----------
    overlap : np.ndarray
        Finite real symmetric strictly positive-definite matrix (n, n), n>=1.
        Positive definiteness applies also outside the supplied subspace.
    probes : np.ndarray
        Finite real right-hand-side block (n, m), m>=1. Zero or dependent
        columns and components outside the span of basis are valid inputs.
    basis : np.ndarray
        Finite real orthonormal frame (n, r), 1<=r<=n. It may be rectangular
        and need not contain probes or be an invariant subspace of overlap.
        Its Gram matrix equals identity within rtol=1e-10, atol=1e-12.
        Overlap symmetry uses rtol=0, atol=1e-12.

    Returns
    -------
    action : np.ndarray
        Real (n, m) action: compress overlap to basis, apply its spectral
        inverse square root to the projected probes, then lift back. The
        output lies in the supplied span; no action on its orthogonal
        complement is appended. Preserve original probe amplitudes.
        Retain every strictly positive projected eigenvalue without a
        cutoff or regularization. Scalar overlap rescaling by a>0 must
        scale the action by a**(-1/2); no absolute eigenvalue floor is used.
        Return finite values whenever the specified action is representable.

    Raises
    ------
    ValueError
        If shapes, realness, finiteness, symmetry, full-overlap or projected
        positive definiteness, or orthonormality fail, or the final action
        is not finite in binary64.

    Notes
    -----
    The supplied frame is the entire finite-projection model: do not
    enlarge it from the probes. Uniformly small/large overlaps and weak
    positive spectral directions are supported. Use stable scalar
    equilibration where needed, without changing the mathematical map.
    Do not mutate inputs. Equivalent matrix-function evaluations are valid.
    """
    return np.empty_like(probes, dtype=float)
```

### Step 5

extract_sparse_operator

Goal
----
Reconstruct a symmetric operator from a signed probe-action block, then apply the spatial and offdiagonal-magnitude masks.

```python
from numbers import Integral, Real
import numpy as np

def extract_sparse_operator(action: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, keep_radius: float, magnitude_threshold: float=0.0) -> np.ndarray:
    """Extract one real symmetric locally supported operator.

    Parameters
    ----------
    action : np.ndarray
        Finite real action block (n,m), with m equal to the number of colors.
    distances : np.ndarray
        Finite nonnegative symmetric (n,n) distances, diagonal zero; n>=1.
        Absolute symmetry and zero-diagonal tolerance is 1e-12.
    colors : np.ndarray
        Integer-valued (n,) labels with all labels 0,...,m-1 represented.
        Integer-valued floating inputs are also valid.
    signs : np.ndarray
        Real (n,) values exactly +1 or -1; integer and float dtypes accepted.
    keep_radius : float
        Nonnegative finite inclusive spatial support radius.
    magnitude_threshold : float
        Nonnegative finite threshold applied to offdiagonal entries AFTER
        symmetrization. Equality is retained; diagonals are exempt.

    Returns
    -------
    matrix : np.ndarray
        Real symmetric (n,n) matrix. Before masking,
        T_ij=(signs[j]*action[i,colors[j]]+signs[i]*action[j,colors[i]])/2.
        Entries with distance>keep_radius vanish. Offdiagonal entries also
        vanish when abs(T_ij)<magnitude_threshold. No normalization or
        element thresholding of action precedes the symmetric reconstruction.

    Raises
    ------
    ValueError
        If shapes, finite-real data, distance conditions, labels, signs,
        keep_radius, or magnitude_threshold violate these constraints.
    """
    return np.zeros(np.asarray(distances).shape,dtype=float)
```

### Step 6

solve_fixed_population_density

Goal
----
Construct the finite-depth nonorthogonal density action and determine its chemical potential from the particle count of the spatially extracted density.

```python
from numbers import Integral, Real
import numpy as np

def solve_fixed_population_density(overlap: np.ndarray, overlap_map: np.ndarray, transformed_hamiltonian: np.ndarray, probes: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, beta: float, electron_count: float, depth: int, keep_radius: float) -> tuple[float,np.ndarray,np.ndarray]:
    """Solve the fixed-population compressed density problem.

    Parameters
    ----------
    overlap : np.ndarray
        Finite real symmetric positive-definite S of shape (n,n).
    overlap_map : np.ndarray
        Finite real symmetric reconstructed inverse-square-root map M, (n,n).
    transformed_hamiltonian : np.ndarray
        Finite real symmetric, already extracted orthogonal Hamiltonian K, (n,n).
    probes : np.ndarray
        Unnormalized signed probe block R, (n,m), matching colors and signs exactly.
    distances : np.ndarray
        Finite nonnegative symmetric (n,n) distance matrix, zero diagonal.
    colors : np.ndarray
        Contiguous nonnegative integer-valued labels (n,), covering all m columns.
    signs : np.ndarray
        Real (n,) values +/-1; integer or float dtype accepted.
    beta : float
        Positive finite inverse temperature in eV**(-1). The finite-temperature
        root is required even in a spectral gap: evaluate occupations without
        overflow and the count residual without losing small electron/hole
        contributions to saturation, cancellation, or underflow.
    electron_count : float
        Requested one-spin particle count strictly between zero and the total
        spectral weight, with a strictly bracketing chemical-potential interval.
    depth : int
        Positive depth for the canonical block-Krylov space starting at M R.
    keep_radius : float
        Finite nonnegative inclusive density extraction radius; no magnitude
        threshold is applied to density or to its spectral components.

    Returns
    -------
    chemical_potential : float
        Mu in eV solving the finite-temperature Tr(S rho(mu))=electron_count
        relation to absolute error 1e-10. An edge of a rounded count plateau
        is not an alternative root; no zero-temperature midpoint convention
        or population-residual-only stopping rule replaces this requirement.
    density : np.ndarray
        Symmetric extracted density rho of shape (n,n).
    density_action : np.ndarray
        Unextracted original-coordinate density action X of shape (n,m).

    Raises
    ------
    ValueError
        If shapes, finite-real data, symmetry, positive definiteness of S,
        probe consistency, distance/color/sign constraints, depth, beta,
        radius, spectral weights, or the root bracket are invalid. Spectral
        weights must be nonnegative, have positive total, and bracket the count.

    Notes
    -----
    Use canonical_block_krylov_basis for the depth-limited subspace of K
    starting at M R, and extract_sparse_operator for spatial reconstruction.
    Both sides of the particle constraint refer to the same extracted
    original-coordinate density. Spectral particle weights are the
    overlap-traces of its extracted spectral components, not unit mode
    multiplicities or Euclidean norms. Negative weights are outside the
    monotone-root domain; zero weights are permitted.
    Use [min(eps)-max(2,50/beta), max(eps)+max(2,50/beta)] and 64 bisections,
    moving the lower bound when the mathematical count is below target and
    the upper bound otherwise, then take the midpoint. An equivalent root
    solve agreeing within 1e-10 is valid. Test the strict bracket and each
    comparison with a cancellation-resistant finite-temperature residual;
    rounding a sum of saturated occupations before subtracting the target
    does not define the comparison. Occupations in the returned arrays may
    round to 0 or 1, but their tiny tails still determine the root.
    Build the unextracted action before symmetric spatial extraction. Do not
    apply a magnitude threshold to density or spectral components. Matrix
    symmetry tolerance is absolute 1e-12. Do not mutate the inputs.
    """
    return 0.0,np.zeros(np.asarray(overlap).shape),np.zeros(np.asarray(probes).shape)
```

### Step 7

compute_frozen_potential_force

Goal
----
Reconstruct the energy-weighted density from the unextracted density action and return the Hamiltonian, overlap-Pulay, and total electronic force contributions.

```python
from numbers import Integral, Real
import numpy as np

def compute_frozen_potential_force(overlap_map: np.ndarray, hamiltonian: np.ndarray, density_action: np.ndarray, density: np.ndarray, overlap_tangent: np.ndarray, hamiltonian_tangent: np.ndarray, distances: np.ndarray, colors: np.ndarray, signs: np.ndarray, keep_radius: float) -> np.ndarray:
    """Compute a frozen-potential nonorthogonal electronic force.

    Parameters
    ----------
    overlap_map : np.ndarray
        Reconstructed real symmetric inverse-square-root M, shape (n,n).
        The inverse-overlap approximation is the ordered product M M.
    hamiltonian : np.ndarray
        Physical real symmetric H in the original basis, shape (n,n), not the
        sparse orthogonal Hamiltonian used to construct the density Krylov space.
    density_action : np.ndarray
        Unextracted original-coordinate density action X, shape (n,m).
    density : np.ndarray
        Already extracted real symmetric density rho, shape (n,n).
    overlap_tangent : np.ndarray
        Real symmetric dS/dx_a, shape (n,n), in inverse bohr.
    hamiltonian_tangent : np.ndarray
        Real symmetric dH/dx_a at frozen charge and interaction kernel, (n,n),
        in eV/bohr.
    distances : np.ndarray
        Nonnegative finite symmetric (n,n) distances with zero diagonal.
    colors : np.ndarray
        Contiguous integer-valued (n,) color labels covering the m action columns.
    signs : np.ndarray
        Real (n,) values +/-1; both integer and floating dtypes are accepted.
    keep_radius : float
        Finite nonnegative inclusive radius for the energy-weighted extraction.

    Returns
    -------
    force_terms : np.ndarray
        Real length-3 vector [F_H,F_S,F_H+F_S] in eV/bohr, where
        F_H=-Tr(rho dH) and F_S=Tr(rho_E dS). Form the original-coordinate
        energy-weighted action by inverse-overlap times H times X, then use
        extract_sparse_operator with zero magnitude threshold for rho_E.
        There is no spin factor, repulsive term, or charge derivative.

    Raises
    ------
    ValueError
        If any matrix is nonfinite, complex, nonsymmetric where required, or
        shape-incompatible, or any extraction constraint fails. Symmetry uses
        absolute tolerance 1e-12.
    """
    return np.zeros(3,dtype=float)
```

### Step 8

run_chromatic_charge_force

Goal
----
Orchestrate the finite-SCF force benchmark, rebuilding the compressed Hamiltonian and fixed-population density after each charge update and once at the final charge.

```python
from numbers import Integral, Real
import numpy as np

def run_chromatic_charge_force(positions: np.ndarray | None=None, beta: float=32.0, electron_count: float=8.0, mixing: tuple[float,...]=(0.6,0.35,0.75), hamiltonian_threshold: float=0.135, density_depth: int=2, force_index: int=0) -> float:
    """Run the deterministic charge-feedback force calculation.

    Parameters
    ----------
    positions : np.ndarray or None
        Exactly 18 finite strictly increasing positions in [0,24) bohr. None uses
        [0,.72,1.83,3.05,4.12,5.55,6.44,7.90,9.15,10.02,11.48,12.73,
         14.05,15.22,16.87,18.31,20.12,22.01]. Orbital order is fixed.
    beta : float
        Positive finite inverse temperature, eV**(-1).
    electron_count : float
        One-spin particle count in (0,18), also within every density-root bracket.
    mixing : tuple of float
        Finite mixing factors in [0,1]. Each factor specifies one update;
        the empty tuple is valid and makes no updates. Recompute the density
        once after the last update; do not make a further charge update.
    hamiltonian_threshold : float
        Nonnegative finite offdiagonal magnitude cutoff for the reconstructed
        orthogonal Hamiltonian; equality is retained after symmetrization.
    density_depth : int
        Positive canonical Krylov depth for each density evaluation.
    force_index : int
        Orbital displaced for the force tangent, in [0,18).

    Returns
    -------
    force : float
        Final frozen-potential electronic force in eV/bohr, rounded only at
        the end to 10 decimal places with ties to even.

    Raises
    ------
    ValueError
        If any argument violates its contract, the initial overlap is not
        positive definite, or any fixed-population density problem has invalid
        spectral weights or fails to bracket its requested population.

    Notes
    -----
    Call the preceding public step functions; do not reimplement their work.
    Length is 24, overlap exclusion/keep radii are 4.9/2.4, overlap depth is 3,
    overlap magnitude cutoff is .006, density exclusion/keep radii are 7.3/3.6,
    and transformed-Hamiltonian keep radius is 3.4. Both density kernels use
    zero magnitude threshold. The canonical rank tolerance is 1e-12.
    Overlap signs: [1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1,1,-1,1,-1].
    Density signs: [-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,1,-1,1,1,-1,1].
    First obtain the inverse-square-root action Y and its extracted map M.
    Gamma=1.1*I+.15*exp(-D/3) in eV is elementwise and remains fixed.
    Initial q_i=.12*cos(2*pi*i/18)+.04*sin(4*pi*i/18), minus its arithmetic mean.
    At charge q use V=Gamma@q and H=H0+.5*S*(V[:,None]+V[None,:]).
    Reconstruct the transformed Hamiltonian from the compressed action M@H@Y,
    using the overlap colors/signs, radius 3.4 and hamiltonian_threshold.
    Obtain mu,rho,X through solve_fixed_population_density. For each supplied
    alpha update q=(1-alpha)*q+alpha*(diag(rho@S)-electron_count/18).
    At the final q hold q and Gamma fixed, use dH=dH0+.5*dS*(V_i+V_j),
    and call compute_frozen_potential_force using the physical H and X.
    Retain binary64 values throughout and do not mutate input arrays.
    """
    return 0.0
```

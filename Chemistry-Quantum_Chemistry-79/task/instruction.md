# Chemistry-Quantum_Chemistry-79

## Background

Resonances as complex eigenvalues

A bound state is square-integrable and has a real energy. A resonance is not: it is embedded in the continuum and decays by ejecting the particle, so it is represented by a Siegert state obeying purely outgoing boundary conditions, whose energy is complex,

E = E_R - i Gamma / 2.

The real part places the state relative to the detachment threshold and the imaginary part fixes the lifetime tau = 1 / Gamma. The outgoing tail means the Siegert function diverges at large distance, which is why a resonance cannot be obtained by the variational machinery that works for bound states. The methods that do reach it, complex scaling among them, share one idea: deform the problem analytically so that the divergent state becomes square-integrable, at the price of a non-Hermitian operator.

The absorbing potential

The complex absorbing potential takes the deformation into the potential rather than into the coordinate. One adds a negative imaginary term,

H(eta) = H0 - i eta W,

with W a positive one-body function that is zero over the region where the physics happens and grows outside it. The absorber damps the diverging tail, so the resonance appears as a discrete eigenvalue of a matrix problem built from ordinary square-integrable functions. In a complete basis the exact resonance is recovered in the limit eta -> 0.

That limit is unreachable in practice. A finite basis represents the continuum by a handful of discretised pseudostates, so at very small eta the resonance is not yet separated from them and the calculation returns continuum-like roots instead. Increasing eta separates the resonance, but the absorber is a fictitious perturbation and displaces the very eigenvalue one is trying to measure. Every CAP calculation is therefore a compromise between an error that falls with eta and an error that grows with it, and the whole method rests on locating that compromise.

Non-Hermitian structure and the c-product

Because eta W is real and W is symmetric, H(eta) is complex symmetric, H^T = H. Such an operator is not normal: its eigenvectors are not orthogonal in the Hermitian sense and the Hermitian norm of a resonance state is not conserved under the deformation. The structure that is conserved is bilinear. Using the c-product u^T v, defined without complex conjugation, the eigenvectors of a complex symmetric operator are orthogonal and can be normalised as c^T S c = 1 in a basis of overlap S. Expectation values then become complex quantities whose real parts carry the physical content. Every derivative identity of ordinary quantum mechanics survives in this metric, so the Hellmann-Feynman theorem still holds and gives dE/deta as the c-product expectation value of the perturbation. Replacing the c-product by the Hermitian inner product does not merely rescale intermediate quantities, it breaks the analytic continuation the whole construction depends on.

Optimising the absorber, and the corrected energy

The trade-off is quantified by the logarithmic energy velocity v = |eta dE/deta|, the leading term of a Taylor expansion of the energy in the absorber strength. A weakly perturbed resonance is one for which this sensitivity is stationary, and the working criterion is that the optimum sits at a local minimum of v. The distinction between a local and a global minimum matters here rather than being pedantic: for a purely imaginary absorber the velocity vanishes identically at eta = 0, so the global minimiser is always the trivial point at which nothing has been computed. Graphically the stationary point shows up as an inflection, or a cusp, of the eta-trajectory traced by E(eta) in the complex plane.

A common refinement is to subtract the first-order energy contribution of the absorber, giving the deperturbed or first-order corrected energy U = E - eta dE/deta, from which the corrected width follows as Gamma = -2 Im U. The expectation is that removing the leading artificial contribution should reduce the sensitivity to the absorber.

Multiple solutions

Nothing in the formalism guarantees that the stationarity condition has a unique root, and in finite bases it frequently does not. The usual practice has been to elect one of the roots as the physical solution, on grounds that are rarely stated, and to discard the others as spurious artefacts. If instead each root is read at face value, as one particular balance between an incompletely described continuum and the perturbation introduced to compensate for it, then the range of resonance parameters spanned by the roots is not noise but an uncertainty intrinsic to the method, and one that no amount of care within a single root can remove. Whether that range is negligible next to the familiar error sources, the basis and the treatment of correlation, is a quantitative question, and the answer decides whether the multiplicity is a curiosity or a real limitation on what a CAP calculation can claim.

Removing solutions with a real potential

One proposed way out is to add to the absorber a real multiple of the same function, so that the perturbation becomes (kappa - i eta) W with kappa >= 0. This continuum remover was introduced to eliminate the stabilisation points that belong to the discretised continuum while leaving the physical one, which would turn the choice between solutions into a computation. The claim is testable, and testing it needs every solution rather than one. Stationary points of a smooth function do not fade away: a local minimum of the velocity disappears only by meeting a neighbouring maximum and annihilating with it, at a definite strength of the real term. The order in which the solutions go, and whether the last to survive is the one a stability argument would have kept, decides whether the remover selects anything. Where the absorber is switched on also matters: the onset fixes how much of the outer region is absorbed, and molecular calculations place it from the spatial extent of the parent state rather than from the shape of a model barrier.

## Problem

A single particle moves on a line in atomic units (hbar = m = 1) in the potential V(x) = A x^2 exp(-lambda x^2) with A = 1.2 and lambda = 0.5. This potential vanishes at the origin and at large distance and rises to a maximum of A / (lambda e) in between, so it traps a particle behind a finite barrier and supports a metastable state above the dissociation threshold at zero energy. Such a state is characterised by a complex energy E = E_R - i Gamma / 2, and the standard way to reach it with square-integrable functions is to add an absorbing potential to the Hamiltonian, H(eta) = H0 - i eta W, and to look for a strength eta at which the result is least sensitive to the absorber. Determine how much the answer for the width depends on which such strength is chosen, and whether a proposed cure for that ambiguity, a real addition to the absorber, retires the solutions selectively or merely retires them all.

Represent the problem in the non-orthogonal even-tempered Gaussian set phi_i(x) = exp(-alpha_i x^2) with alpha_i = 0.02 * 2.0^i for i = 0 to 19, and take the absorbing potential to be the box form W(x) = (|x| - c)^2 for |x| >= c and zero inside, with onset c = 2.0. Build the overlap, the field-free Hamiltonian and the absorber in that basis, then remove near-linear dependence by canonical orthogonalisation, discarding every direction whose overlap eigenvalue is not greater than 1.0e-8 and keeping the survivors ordered by increasing overlap eigenvalue. The operator H(eta) is complex symmetric rather than Hermitian, so throughout the calculation use the bilinear c-product u^T v, formed without complex conjugation, and normalise every eigenvector by c^T S c = 1.

Sweep the absorber strength over the 1200 geometrically spaced values from eta = 1.0e-4 to eta = 4.0 inclusive. Identify the resonance once, at the grid point nearest eta = 0.15, as the state of smallest Re(c^T X2 c) among those whose eigenvalue has a real part strictly between zero and the barrier maximum, then follow that same state outwards to smaller and to larger eta by selecting at each new grid point the eigenvector of largest |c_prev^T S c|. At every grid point form the logarithmic energy velocity v = |eta dE/deta|, taking the derivative analytically from the Hellmann-Feynman relation dE/deta = -i c^T W c rather than by differencing. A solution is any interior grid point at which v is strictly smaller than at both of its neighbours; sharpen each one by fitting a parabola to the three bracketing points in the variable ln(eta) and taking its vertex, then solve once more at that refined strength, re-identifying the state by largest |c^T S c| overlap with the vector accepted at the bracketing grid point.

From each solution take the first-order corrected energy U = E - eta dE/deta and its width Gamma = -2 Im U, and measure the ambiguity the multiplicity carries by the spread (max Gamma - min Gamma) over all solutions, expressed in meV using 1 hartree = 27211.386245988 meV. Then add the continuum remover, a real multiple of the same absorber, so that the Hamiltonian becomes H0 + (kappa - i eta) W with kappa >= 0, and keep every rule above unchanged with H0 + kappa W in place of H0, including the seed window bounded by the barrier maximum of the unshifted potential. Follow every solution of the unshifted problem separately as kappa grows, and do not recover a followed solution from the seed rule at kappa > 0: over some intervals of kappa the state of smallest extent inside the seed window belongs to the discretised continuum rather than to the branch being followed, so the identity is carried in kappa instead. The eigenvector accepted at the previous strength, taken at the position its own minimum occupied there, fixes the state at the next strength by largest |c^T S c| overlap, and the branch is tracked outwards in eta from that position by the same rule as before. Judge existence on the velocity as a smooth function of ln(eta) rather than on the grid, with dE/deta and d2E/deta2 both obtained analytically in the c-product, the second from second-order perturbation theory over all retained states, and read it inside the window spanned by the solution and the velocity maximum immediately below it in eta at the largest strength where that solution was still present. A solution vanishes when that pair coalesces. Locate the coalescence strength of every solution to an absolute accuracy of 1.0e-10, and report as your final answer the largest of them, the remover strength beyond which the velocity criterion selects nothing at all in this model.

Molecular practice does not place the absorber where this model does. Look up the box CAP onset adopted along the two Cartesian directions perpendicular to the internuclear axis for the N2 and CO shape resonances in the benchmark complex absorbing potential literature, a single value shared by both molecules, and repeat the whole analysis with the onset of this one-dimensional model set to that number in bohr and everything else unchanged. Report that onset, how many solutions the model carries there, the spread of their first-order corrected widths in meV, and the remover strength at which the last of them goes. The final answer stays the value for the onset of 2.0 bohr.

The scalars that determine the final number, and that your reasoning should set out alongside it, are the barrier maximum; the smallest overlap eigenvalue and how many basis directions survive the cut; for every solution at onset 2.0 the refined absorber strength and the first-order corrected width in meV; the spread of the first-order widths and of the zeroth-order widths in meV; the coalescence strength of each solution and the absorber strength at which the last pair coalesces; which solution survives longest and whether it is the one carrying the largest width; and, for the looked-up onset, the four quantities named in the previous paragraph.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

basis_integrals

Goal
----
All one-body integrals of the model in an even-tempered Gaussian basis: overlap, field-free Hamiltonian, complex absorbing potential, and the second spatial moment.

```python
import numpy as np
from scipy.special import erfc

def basis_integrals(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                    potential_width: float, cap_onset: float) -> "np.ndarray":
    '''Overlap, field-free Hamiltonian, CAP and second-moment matrices.

    Parameters
    ----------
    alpha0 : float
        Exponent of the most diffuse Gaussian; must be positive.
    beta : float
        Even-tempered ratio, alpha_i = alpha0 * beta**i; must exceed 1.
    n_basis : int
        Number of Gaussians; must be at least 2.
    barrier_strength : float
        Coefficient A of the potential V(x) = A x^2 exp(-lambda x^2); must not be negative.
    potential_width : float
        Exponent lambda of the potential; must be positive.
    cap_onset : float
        Onset c of the box CAP W(x) = (|x| - c)^2 for |x| >= c; must not be negative.

    Returns
    -------
    integrals : numpy.ndarray
        Real array of shape (4, n_basis, n_basis) holding, in order, the overlap
        matrix S, the field-free Hamiltonian T + V, the CAP matrix W and the
        second-moment matrix X2, each in the primitive (non-orthogonal) basis.

    Raises
    ------
    ValueError
        If alpha0 is not positive, if beta is not greater than 1, if n_basis is not an
        integer of at least 2, if barrier_strength is negative, if potential_width is
        not positive, or if cap_onset is negative.
    '''
    return integrals
```

### Step 2

canonical_transform

Goal
----
Canonical orthogonalisation of the non-orthogonal Gaussian basis, with the near-linear-dependent directions removed and a fixed column sign.

```python
import numpy as np

def canonical_transform(overlap: "np.ndarray", threshold: float) -> "np.ndarray":
    '''Canonical orthogonalisation matrix of a non-orthogonal basis.

    Parameters
    ----------
    overlap : numpy.ndarray
        Real symmetric overlap matrix S of shape (n_basis, n_basis).
    threshold : float
        Overlap eigenvalues less than or equal to this value are discarded;
        must be positive.

    Returns
    -------
    transform : numpy.ndarray
        Real array of shape (n_basis, n_kept) whose column k is u_k / sqrt(s_k),
        ordered by increasing s_k and sign-fixed so that the entry of largest
        absolute value in each column is positive.

    Raises
    ------
    ValueError
        If overlap is not a square two-dimensional array, if threshold is not
        positive, or if no overlap eigenvalue exceeds the threshold.
    '''
    return transform
```

### Step 3

cap_eigenvalues

Goal
----
Complex eigenvalues of the CAP-augmented Hamiltonian at one value of the absorbing-potential strength.

```python
import numpy as np
import scipy.linalg as sla

def cap_eigenvalues(integrals: "np.ndarray", transform: "np.ndarray", eta: float) -> "np.ndarray":
    '''Complex eigenvalues of H(eta) = H0 - i eta W.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2 as produced
        by basis_integrals.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    eta : float
        CAP strength; must not be negative.

    Returns
    -------
    eigenvalues : numpy.ndarray
        Complex array of shape (n_kept,), sorted by increasing real part with
        ties broken by increasing imaginary part.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have
        n rows, or if eta is negative.
    '''
    return eigenvalues
```

### Step 4

resonance_extents

Goal
----
Spatial extent of every CAP eigenstate, used to tell the compact resonance apart from the diffuse discretised continuum.

```python
import numpy as np

def resonance_extents(integrals: "np.ndarray", transform: "np.ndarray", eta: float) -> "np.ndarray":
    '''Second spatial moment of each CAP eigenstate.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    eta : float
        CAP strength; must not be negative.

    Returns
    -------
    extents : numpy.ndarray
        Real array of shape (n_kept,) holding Re(c^T X2 c) for each eigenvector,
        normalised by c^T S c = 1 and ordered exactly as the eigenvalues.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have
        n rows, or if eta is negative.
    '''
    return extents
```

### Step 5

branch_energies

Goal
----
The complex energy of the resonance followed continuously along the whole CAP-strength grid.

```python
import numpy as np

def branch_energies(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                    eta_seed: float, e_max: float) -> "np.ndarray":
    '''Resonance energy along the CAP-strength grid.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    etas : numpy.ndarray
        Strictly increasing grid of non-negative CAP strengths, at least 3 points.
    eta_seed : float
        CAP strength at which the resonance is identified; snapped to the nearest
        grid point.
    e_max : float
        Upper edge of the search window for the seed, the top of the barrier;
        must be positive.

    Returns
    -------
    energies : numpy.ndarray
        Complex array of shape (n_eta,) holding E(eta) along the followed branch.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have n
        rows, if etas is not a strictly increasing array of at least 3 non-negative
        points, if e_max is not positive, or if no eigenvalue at the seed lies
        strictly between zero and e_max.
    '''
    return energies
```

### Step 6

branch_velocity

Goal
----
The logarithmic energy velocity along the trajectory, evaluated analytically rather than by differencing.

```python
import numpy as np

def branch_velocity(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                    eta_seed: float, e_max: float) -> "np.ndarray":
    '''Logarithmic energy velocity of the followed resonance branch.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    etas : numpy.ndarray
        Strictly increasing grid of non-negative CAP strengths, at least 3 points.
    eta_seed : float
        CAP strength at which the resonance is identified.
    e_max : float
        Top of the barrier, the upper edge of the seed search window.

    Returns
    -------
    velocity : numpy.ndarray
        Real array of shape (n_eta,) holding |eta dE/deta| along the branch, with
        dE/deta obtained analytically from the Hellmann-Feynman theorem in the
        c-product rather than by differencing E(eta).

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have n
        rows, if etas is not a strictly increasing array of at least 3 non-negative
        points, if e_max is not positive, or if no eigenvalue at the seed lies
        strictly between zero and e_max.
    '''
    return velocity
```

### Step 7

solution_etas

Goal
----
Every CAP solution on the trajectory, located as an interior local minimum of the velocity and sharpened by parabolic interpolation.

```python
import numpy as np

def solution_etas(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                  eta_seed: float, e_max: float) -> "np.ndarray":
    '''Refined CAP strengths of every solution on the trajectory.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    etas : numpy.ndarray
        Strictly increasing grid of non-negative CAP strengths, at least 3 points.
    eta_seed : float
        CAP strength at which the resonance is identified.
    e_max : float
        Top of the barrier, the upper edge of the seed search window.

    Returns
    -------
    eta_opt : numpy.ndarray
        Real array of shape (n_solutions,) holding the parabolically refined CAP
        strength of each velocity local minimum, in increasing order. Empty if the
        velocity has no interior local minimum on the grid.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have n
        rows, if etas is not a strictly increasing array of at least 3 non-negative
        points, if e_max is not positive, or if no eigenvalue at the seed lies
        strictly between zero and e_max.
    '''
    return eta_opt
```

### Step 8

deperturbed_widths

Goal
----
The first-order corrected resonance width at every CAP solution.

```python
import numpy as np

def deperturbed_widths(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                       eta_seed: float, e_max: float) -> "np.ndarray":
    '''First-order corrected resonance widths at every CAP solution.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    etas : numpy.ndarray
        Strictly increasing grid of non-negative CAP strengths, at least 3 points.
    eta_seed : float
        CAP strength at which the resonance is identified.
    e_max : float
        Top of the barrier, the upper edge of the seed search window.

    Returns
    -------
    widths : numpy.ndarray
        Real array of shape (n_solutions,) holding Gamma = -2 Im(E - eta dE/deta)
        in hartree at each CAP solution, ordered by increasing CAP strength. Empty
        if the velocity has no interior local minimum on the grid.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have n
        rows, if etas is not a strictly increasing array of at least 3 non-negative
        points, if e_max is not positive, or if no eigenvalue at the seed lies
        strictly between zero and e_max.
    '''
    return widths
```

### Step 9

branch_velocity_slope

Goal
----
The slope of the logarithmic energy velocity with respect to ln(eta) along the followed branch, from analytic first and second derivatives of the resonance energy.

```python
import numpy as np

def branch_velocity_slope(integrals: "np.ndarray", transform: "np.ndarray", etas: "np.ndarray",
                          eta_seed: float, e_max: float) -> "np.ndarray":
    '''Slope dv/d ln(eta) of the logarithmic energy velocity along the followed branch.

    Parameters
    ----------
    integrals : numpy.ndarray
        Array of shape (4, n_basis, n_basis) holding S, H0, W and X2; the second
        block is used as the Hamiltonian exactly as supplied.
    transform : numpy.ndarray
        Canonical orthogonalisation matrix of shape (n_basis, n_kept).
    etas : numpy.ndarray
        Strictly increasing grid of strictly positive CAP strengths, at least 3
        points.
    eta_seed : float
        CAP strength at which the resonance is identified; snapped to the nearest
        grid point.
    e_max : float
        Top of the barrier, the upper edge of the seed search window; must be
        positive.

    Returns
    -------
    slope : numpy.ndarray
        Real array of shape (n_eta,) holding d|eta dE/deta| / d ln(eta) along the
        branch, with dE/deta from the c-product Hellmann-Feynman theorem and
        d2E/deta2 from c-product second-order perturbation theory over all
        retained eigenstates, neither obtained by differencing.

    Raises
    ------
    ValueError
        If integrals does not have shape (4, n, n), if transform does not have n
        rows, if etas is not a strictly increasing array of at least 3 strictly
        positive points, if e_max is not positive, or if no eigenvalue at the seed
        lies strictly between zero and e_max.
    '''
    return slope
```

### Step 10

solution_fold_strength

Goal
----
The continuum-remover strength at which one chosen CAP solution ceases to exist.

```python
import numpy as np

def solution_fold_strength(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                           potential_width: float, cap_onset: float, threshold: float,
                           eta_min: float, eta_max: float, n_eta: int, eta_seed: float,
                           eta_ref: float, kappa_step: float, kappa_max: float) -> float:
    '''Continuum-remover strength at which one chosen CAP solution vanishes.

    Parameters
    ----------
    alpha0 : float
        Exponent of the most diffuse Gaussian.
    beta : float
        Even-tempered ratio.
    n_basis : int
        Number of Gaussians.
    barrier_strength : float
        Coefficient A of V(x) = A x^2 exp(-lambda_pot x^2).
    potential_width : float
        Exponent lambda_pot of the potential.
    cap_onset : float
        Onset of the box CAP.
    threshold : float
        Overlap-eigenvalue cut for canonical orthogonalisation.
    eta_min : float
        Smallest CAP strength on the grid; must be positive.
    eta_max : float
        Largest CAP strength on the grid; must exceed eta_min.
    n_eta : int
        Number of geometrically spaced grid points; must be at least 3.
    eta_seed : float
        CAP strength at which the resonance is identified at kappa = 0.
    eta_ref : float
        Reference CAP strength; the solution followed is the velocity minimum of the
        unshifted problem nearest to it in ln(eta). Must be positive.
    kappa_step : float
        Positive ladder spacing in kappa used to bracket the coalescence.
    kappa_max : float
        Largest remover strength examined; must exceed kappa_step.

    Returns
    -------
    kappa_c : float
        The strength of the real term kappa in H0 + (kappa - i eta) W at which the chosen
        CAP solution coalesces with the velocity maximum immediately below it in eta and
        vanishes, to an absolute accuracy of 1e-10.

    Raises
    ------
    ValueError
        If eta_ref is not positive, if kappa_step is not positive, if kappa_max does not
        exceed kappa_step, if the unshifted problem has no CAP solution, if the followed
        branch is lost before it coalesces, if the solution still exists at kappa_max, or
        if any other argument violates the constraints of the steps it feeds.
    '''
    return kappa_c
```

### Step 11

remover_survival_strength

Goal
----
The continuum-remover strength beyond which no CAP solution of the model survives.

```python
import numpy as np

def remover_survival_strength(alpha0: float, beta: float, n_basis: int, barrier_strength: float,
                              potential_width: float, cap_onset: float, threshold: float,
                              eta_min: float, eta_max: float, n_eta: int, eta_seed: float,
                              kappa_step: float, kappa_max: float) -> float:
    '''Remover strength at which the last surviving CAP solution of the model vanishes.

    Parameters
    ----------
    alpha0 : float
        Exponent of the most diffuse Gaussian.
    beta : float
        Even-tempered ratio.
    n_basis : int
        Number of Gaussians.
    barrier_strength : float
        Coefficient A of V(x) = A x^2 exp(-lambda_pot x^2).
    potential_width : float
        Exponent lambda_pot of the potential.
    cap_onset : float
        Onset of the box CAP.
    threshold : float
        Overlap-eigenvalue cut for canonical orthogonalisation.
    eta_min : float
        Smallest CAP strength on the grid; must be positive.
    eta_max : float
        Largest CAP strength on the grid; must exceed eta_min.
    n_eta : int
        Number of geometrically spaced grid points; must be at least 3.
    eta_seed : float
        CAP strength at which the resonance is identified at kappa = 0.
    kappa_step : float
        Positive ladder spacing in kappa used to bracket each coalescence.
    kappa_max : float
        Largest remover strength examined; must exceed kappa_step.

    Returns
    -------
    kappa_last : float
        The largest of the coalescence strengths of the CAP solutions of the unshifted
        problem, that is the smallest remover strength at which none of them is left,
        to an absolute accuracy of 1e-10. Solutions are followed in order of decreasing
        first-order corrected width, and coalescence strengths that agree to within
        1e-12 are resolved in favour of the wider solution.

    Raises
    ------
    ValueError
        If the unshifted problem has no CAP solution, if any solution still exists at
        kappa_max or is lost before it coalesces, or if any other argument violates the
        constraints of the steps it feeds.
    '''
    return kappa_last
```

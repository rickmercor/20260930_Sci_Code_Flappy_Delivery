# Chemistry-Quantum_Chemistry-32

## Background

The local one-particle spectral function of a correlated lattice model encodes the energies and
residues of its single-particle excitations, and its power moments are the expectation values of
nested commutators of the Hamiltonian with the creation and annihilation operators. The classical
moment problem turns a finite sequence of such moments into a discrete measure: a Gauss quadrature
rule whose nodes are the eigenvalues of a symmetric tridiagonal (Jacobi) matrix built from the
moments and whose weights are the squared first components of the corresponding eigenvectors. Every
retained moment is then reproduced exactly by the discrete measure, which is what makes finite-pole
representations attractive as impurity solvers: they carry exact sum rules, produce a causal Green
function with real poles and positive residues, and can be embedded in a dynamical mean-field loop.

Two difficulties limit the naive use of moment data. First, the moments of an interacting model are
not fixed numbers: beyond the lowest orders they involve correlation functions (bond amplitudes,
double occupancies and higher correlators) that are themselves functionals of the spectral function
being reconstructed, so the moments and the measure must be determined together, self-consistently, and any practical scheme
replaces the exact moment hierarchy by an approximate closure in which some correlators are updated
from the reconstructed spectral function.
Second, the Hankel matrix of a moment sequence is exponentially ill-conditioned in its size, so an
over-resolved pole set produces spurious pole-zero pairs that carry no physical weight but distort the
spectrum; a principled truncation of the retained rank is required, and the singular values of the
Hankel matrix supply it.

The single-band Hubbard model on the infinite-coordination Bethe lattice is the standard test bed
for such solvers. Its bare density of states is semielliptic, the dynamical mean-field mapping onto a
self-consistent impurity problem is exact, and the physics at half filling is the crossover from a
metal with a central quasiparticle resonance flanked by two Hubbard satellites to a Mott insulator
with a gap at the Fermi level. In a low-rank discrete representation the central pole is the carrier
of the low-energy spectral weight of the reconstruction, and its weight relative to the non-interacting
value is the finite-rank proxy such schemes use for the interaction-induced suppression of the central
resonance; it is a property of the chosen moment closure and rank, not an exact residue. Particle-hole symmetry at half filling pins the
central pole to the Fermi level, makes the odd moments vanish and the occupancy exactly one half per
spin, and must be respected by any approximate closure of the moment hierarchy if the iteration is
to have a genuine fixed point rather than a drifting or cycling sequence of approximants.

## Problem

Finite-pole representations of a local one-particle spectral function are attractive impurity solvers
because they carry exact spectral sum rules and can be embedded directly into a dynamical mean-field
loop, but two things have held them back: the input data (the local spectral moments) contain
many-body expectation values that depend on the spectral function itself, and the moment problem is
notoriously ill-conditioned, so an over-resolved pole set acquires spurious pole-zero pairs. A recent
source addresses both. The reconstructed discrete spectral function is fed back into the expectation
values that enter the moments, and the moments are iterated to a fixed point, so that the spectral
function that closes the moments coincides with the one the quadrature reconstructs; and the number
of poles actually retained is not the nominal order but the numerically resolvable rank read off from
the singular values of the Hankel moment matrix. The primary inputs of the scheme are the interaction,
the bare band, the nominal pole order and the numerical thresholds; its output is a set of real poles
with positive weights.

Consider the single-band Hubbard model at half filling (chemical potential U/2, occupancy 1/2 per
spin) on the infinite-coordination Bethe lattice, whose bare local density of states is the
semielliptic band rho_0(omega) = (2/(pi D^2)) sqrt(D^2 - omega^2) on |omega| < D and zero outside, with
half-bandwidth D. The source works with the scaled Bethe hopping t = D/2 and the coordination-scaled
combination t^2 z_NN = D^2/4 (z_NN the coordination number). Apply the source's inner self-consistency
loop at nominal pole order N = 3 with the bath held fixed at this bare band throughout (the first
iteration of the outer mean-field loop, with no bath update), for U = 1.5 and D = 1.0. The task
reproduces the source's approximate moment closure for this model, not the exact equation-of-motion
spectral moments of the Hubbard Hamiltonian: in the source's closure at half filling, mu_0 = 1, the odd
moments vanish by particle-hole symmetry, the second moment is mu_2 = U^2/4 + t^2 z_NN <c^dag_i c_j>
where <c^dag_i c_j> is the nearest-neighbour bond amplitude updated self-consistently (so that mu_2
itself depends on the reconstructed spectral function, unlike the exact moment), and the fourth moment
is closed at the leading connected correction beyond the square of mu_2, as the source does for this
model; without that correction the 3 x 3 Hankel matrix is degenerate at particle-hole symmetry and the
central pole carries no weight. The bond amplitude is closed on the reconstructed spectral function
A(omega) = sum_i w_i delta(omega - eps_i) through the source's hopping-weighted overlap with the bare
band, <c^dag_i c_j> = integral over the occupied side of t rho_0(omega) A(omega) d omega with t = D/2;
because A is a discrete measure and particle-hole symmetry is enforced exactly at the level of the
moment update, evaluate that occupied-side integral as one half of the same integral over the whole
real line (a pole pinned at the Fermi level contributes half its weight). The central-pole weight
obtained this way is a property of the source's closure at the chosen rank, not the quasiparticle
residue of the Hubbard model.

Seed the bond amplitude with its non-interacting value 4/(3 pi^2) (the same overlap evaluated with
A = rho_0), form the moments mu_0, ..., mu_5, and iterate: build the 3 x 3 Hankel matrix of the
moments, determine the resolvable rank as the largest n whose n-th singular value exceeds tau = 1e-8
times the largest singular value, construct the Gauss quadrature rule of that rank (the nodes are the
eigenvalues of the Jacobi matrix of the moment sequence and the weights are mu_0 times the squared
first components of its normalised eigenvectors), close the bond amplitude on the resulting poles and
weights, recompute the moments, and mix the recomputed moments linearly with the previous ones with
mixing factor 0.5 on the new moments. Stop when the maximum over the nonzero moments of the relative
change |mu_n^(l+1) - mu_n^(l)| / |mu_n^(l)| between successive mixed moment sets falls below
delta = 1e-8, counting one iteration per closure-and-mixing update, and allow at most 500 iterations.
Then report the weight of the pole nearest to omega = 0 in the converged quadrature rule, to at least
five significant figures.

In the reasoning, state the conventions you adopted for the moment closure and justify each from the
source, and give the seed second and fourth moments, the converged bond amplitude, the converged second
and fourth moments, the resolvable rank at the fixed point together with the smallest-to-largest
singular-value ratio of the Hankel matrix that fixes it, the number of iterations needed to reach the
tolerance, the position and weight of the outer poles, and the weight of the central pole. All
energies are in units of D.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> to a few hundred words. Report every scalar the task statement asks for, together with the conventions you adopted, and keep the rest of the working brief.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

semielliptic_dos

Goal
----
Evaluate the non-interacting density of states of the Bethe lattice with half-bandwidth D on a 1-D grid of energies, normalised to unit integral and identically zero outside the band.

```python
def semielliptic_dos(omega: "np.ndarray", half_bandwidth: float) -> "np.ndarray":
    """Evaluate the non-interacting density of states of the Bethe lattice with half-bandwidth D on a 1-D grid of energies, normalised to unit integral and identically zero outside the band.

    Parameters
    ----------
    omega : numpy.ndarray
        Non-empty finite 1-D array of energies.
    half_bandwidth : float
        Positive half-bandwidth D of the semielliptic band.

    Returns
    -------
    rho : numpy.ndarray
        Density of states at each energy, same shape as omega (float64).

    Raises
    ------
    ValueError
        If omega is not a non-empty finite 1-D array, or half_bandwidth is not positive and finite.
    """
    return rho
```

### Step 2

half_filled_moments

Goal
----
Return the six moments mu_0 ... mu_5 of the source's approximate moment closure for the half-filled Hubbard model on the Bethe lattice, given the interaction U, the half-bandwidth D and the nearest-neighbour bond amplitude. This is the source's self-consistent moment map, in which the second and fourth moments depend on the bond amplitude through the scaled Bethe hopping t = D/2 and the coordination-scaled t^2 z_NN; it is not the exact equation-of-motion moment sequence of the Hubbard Hamiltonian. The odd moments vanish at particle-hole symmetry.

```python
def half_filled_moments(U: float, half_bandwidth: float, bond_amplitude: float) -> "np.ndarray":
    """Return the six moments mu_0 ... mu_5 of the source's approximate moment closure for the half-filled Hubbard model on the Bethe lattice, given the interaction U, the half-bandwidth D and the nearest-neighbour bond amplitude. This is the source's self-consistent moment map, in which the second and fourth moments depend on the bond amplitude through the scaled Bethe hopping t = D/2 and the coordination-scaled t^2 z_NN; it is not the exact equation-of-motion moment sequence of the Hubbard Hamiltonian. The odd moments vanish at particle-hole symmetry.

    Parameters
    ----------
    U : float
        Nonnegative on-site interaction.
    half_bandwidth : float
        Positive half-bandwidth D of the semielliptic band.
    bond_amplitude : float
        Finite nearest-neighbour bond amplitude <c+ c> of one spin.

    Returns
    -------
    moments : numpy.ndarray
        Array of shape (6,) holding mu_0 ... mu_5 (float64).

    Raises
    ------
    ValueError
        If U is negative or not finite, half_bandwidth is not positive and finite, or bond_amplitude is not finite.
    """
    return moments
```

### Step 3

resolvable_rank

Goal
----
Determine the numerically resolvable pole rank of the moment sequence at the requested order from the singular-value spectrum of its Hankel moment matrix, using the source's relative criterion with threshold tau.

```python
def resolvable_rank(moments: "np.ndarray", order: int, tau: float) -> int:
    """Determine the numerically resolvable pole rank of the moment sequence at the requested order from the singular-value spectrum of its Hankel moment matrix, using the source's relative criterion with threshold tau.

    Parameters
    ----------
    moments : numpy.ndarray
        Finite 1-D array holding at least 2*order-1 moments mu_0, mu_1, ...
    order : int
        Positive nominal pole order N.
    tau : float
        Dimensionless threshold in (0, 1).

    Returns
    -------
    n_star : int
        Resolvable rank N* with 1 <= N* <= order.

    Raises
    ------
    ValueError
        If order is not a positive integer, moments is too short or not finite, or tau is outside (0, 1).
    """
    return n_star
```

### Step 4

jacobi_matrix

Goal
----
Build the symmetric tridiagonal Jacobi matrix of the rank-point Gauss-Christoffel quadrature rule from the moment sequence, following the source's factorisation route through the Hankel moment matrix.

```python
def jacobi_matrix(moments: "np.ndarray", rank: int) -> "np.ndarray":
    """Build the symmetric tridiagonal Jacobi matrix of the rank-point Gauss-Christoffel quadrature rule from the moment sequence, following the source's factorisation route through the Hankel moment matrix.

    Parameters
    ----------
    moments : numpy.ndarray
        Finite 1-D array holding at least 2*rank moments.
    rank : int
        Positive number of quadrature points.

    Returns
    -------
    jacobi : numpy.ndarray
        Symmetric array of shape (rank, rank) (float64).

    Raises
    ------
    ValueError
        If rank is not a positive integer, moments is too short or not finite, or the Hankel matrix at this rank is not positive definite.
    """
    return jacobi
```

### Step 5

quadrature_nodes_weights

Goal
----
Return the Gauss-Christoffel nodes and weights encoded by a symmetric Jacobi matrix and the zeroth moment, with the nodes in ascending order and the weights in the matching order.

```python
def quadrature_nodes_weights(jacobi: "np.ndarray", mu0: float) -> "np.ndarray":
    """Return the Gauss-Christoffel nodes and weights encoded by a symmetric Jacobi matrix and the zeroth moment, with the nodes in ascending order and the weights in the matching order.

    Parameters
    ----------
    jacobi : numpy.ndarray
        Finite symmetric square matrix.
    mu0 : float
        Positive zeroth moment.

    Returns
    -------
    nodes_weights : numpy.ndarray
        Array of shape (2, n): row 0 the ascending nodes, row 1 the weights (float64).

    Raises
    ------
    ValueError
        If jacobi is not a finite symmetric square matrix, or mu0 is not positive and finite.
    """
    return nodes_weights
```

### Step 6

bond_amplitude_closure

Goal
----
Evaluate the nearest-neighbour bond amplitude <c+ c> of one spin from a discrete spectral measure given by nodes and weights, using the source's closure kernel built on the bare Bethe density of states, with particle-hole symmetry enforced at the level of the moment update as the source prescribes: the amplitude is one half of the full-line sum over all poles of w_i t rho_0(eps_i) with the scaled Bethe hopping t = D/2, for any particle-hole-symmetric input measure, so a pole at zero energy contributes half its weight and a pole at or beyond the band edge contributes nothing.

```python
def bond_amplitude_closure(nodes: "np.ndarray", weights: "np.ndarray", half_bandwidth: float) -> float:
    """Evaluate the nearest-neighbour bond amplitude <c+ c> of one spin from a discrete spectral measure given by nodes and weights, using the source's closure kernel built on the bare Bethe density of states, with particle-hole symmetry enforced at the level of the moment update as the source prescribes: the amplitude is one half of the full-line sum over all poles of w_i t rho_0(eps_i) with the scaled Bethe hopping t = D/2, for any particle-hole-symmetric input measure, so a pole at zero energy contributes half its weight and a pole at or beyond the band edge contributes nothing.

    Parameters
    ----------
    nodes : numpy.ndarray
        Finite 1-D array of pole positions.
    weights : numpy.ndarray
        Finite 1-D array of nonnegative pole weights, same shape as nodes.
    half_bandwidth : float
        Positive half-bandwidth D of the semielliptic band.

    Returns
    -------
    amplitude : float
        The bond amplitude <c+ c> as a native Python float.

    Raises
    ------
    ValueError
        If nodes and weights are not 1-D arrays of equal nonzero length, are not finite, a weight is negative, or half_bandwidth is not positive and finite.
    """
    return amplitude
```

### Step 7

self_consistent_central_weight

Goal
----
Run the inner moment-quadrature self-consistency loop of the source at the given pole order for the half-filled Hubbard model on the Bethe lattice, seeded from the given bond amplitude, and return the weight of the pole closest to zero energy at convergence. Compose the earlier steps: form the moments, select the resolvable rank, build the Jacobi matrix, extract nodes and weights, close the bond amplitude, mix the new moments linearly with the previous ones using the given mixing factor, and stop when the source's relative moment criterion falls below delta over the nonzero moments.

```python
def self_consistent_central_weight(U: float, half_bandwidth: float, order: int, tau: float,
                                delta: float, mixing: float, seed_amplitude: float,
                                max_iter: int) -> float:
    """Run the inner moment-quadrature self-consistency loop of the source at the given pole order for the half-filled Hubbard model on the Bethe lattice, seeded from the given bond amplitude, and return the weight of the pole closest to zero energy at convergence. Compose the earlier steps: form the moments, select the resolvable rank, build the Jacobi matrix, extract nodes and weights, close the bond amplitude, mix the new moments linearly with the previous ones using the given mixing factor, and stop when the source's relative moment criterion falls below delta over the nonzero moments.

    Parameters
    ----------
    U : float
        Nonnegative on-site interaction.
    half_bandwidth : float
        Positive half-bandwidth D.
    order : int
        Positive nominal pole order N.
    tau : float
        Rank threshold in (0, 1).
    delta : float
        Positive relative convergence tolerance on the moments.
    mixing : float
        Linear mixing factor in (0, 1] applied to the new moments.
    seed_amplitude : float
        Finite bond amplitude used to seed the moments.
    max_iter : int
        Positive maximum number of iterations.

    Returns
    -------
    w_central : float
        Weight of the pole closest to zero energy at the fixed point, as a native Python float; the rule is rebuilt from the final (mixed) moment set after the stop test is met.

    Raises
    ------
    ValueError
        If any argument is outside its stated domain, or the iteration does not converge within max_iter.
    """
    return w_central
```

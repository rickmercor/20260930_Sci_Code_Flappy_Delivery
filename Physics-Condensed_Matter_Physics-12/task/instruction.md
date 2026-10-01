# Physics-Condensed_Matter_Physics-12

## Background

Self-consistent approximations to interacting electrons couple one-particle propagation to two-particle scattering. The parquet framework organizes scattering processes into coupled channels, making it useful for studying competing collective fluctuations. Solving these equations requires an iterative numerical method, whose convergence can change as the interaction grows. Local stability analysis distinguishes a poor iteration from an inconsistent physical approximation and helps determine how strongly successive updates should be mixed. Different choices of iterated variables can share a self-consistent solution while having different convergence properties.

## Problem

A single site carrying two spin species is treated by a self-consistent two-particle scheme in which the scattering vertex is split into channels and the one- and two-particle quantities are solved together on a finite grid of Matsubara frequencies. The way the vertex is split among the channels, the frequency relabelling by which each channel feeds the others, the equation of motion that links the vertex to the self-energy, and the criterion that fixes how strongly the iteration must be damped are the contribution of the reference below and must be taken from it. The fully irreducible vertex is held fixed at the bare on-site interaction for the whole calculation, so the scheme is an approximation and its fixed point is not the exact result for this site.

Configuration, in units where the inverse temperature is 1: the site Hamiltonian is U(n_up - 1/2)(n_dn - 1/2) - dmu(n_up + n_dn) with on-site interaction U = 5 and chemical potential dmu = 1, so the site sits away from half filling and away from particle-hole symmetry. The site is not isolated: it is coupled to a wide, flat, featureless bath whose entire effect on the Matsubara axis is to add i*Gam*sign(nu) to the inverse non-interacting propagator, with hybridisation strength Gam = 0.5, so the non-interacting propagator already carries a finite lifetime and the site has no atomic limit. The frequency box holds 8 fermionic and 7 bosonic Matsubara frequencies; the fermionic box straddles zero symmetrically and the bosonic box is centred on zero, and the rule that an argument outside its box contributes nothing applies to the shifted propagator and vertex arguments in the contractions. The static part of the self-energy is the exception: retain the equal-time contact contribution there and evaluate the finite-box occupation per spin as one half plus the reciprocal of the inverse temperature times the sum over the fermionic box of the propagator with its 1/(i nu) tail subtracted.

The iteration is damped, retaining a fixed fraction of each fresh sweep and carrying the remainder over from the previous state, and is started from a vanishing reducible vertex, which reaches the fixed point continuously connected to the non-interacting limit; that fixed point is not stable under an undamped sweep, so the damping is what makes the iteration converge at all.

The reference also constructs a second iteration that carries the full vertex itself rather than its reducible parts, recovering each channel's irreducible vertex by inverting the same ladder relations instead of summing them and then evaluating the same channel decomposition forward to a fresh full vertex. It states that update in closed form only for a zero-dimensional showcase model, where extra relations among the channels let that decomposition instead be solved for the full vertex; the solved form has no analogue on a frequency box, so the forward evaluation is the one to use here, and the equation of motion takes the vertex this sweep carries. The two schemes share their fixed points but linearise differently, so they do not tolerate the same damping.

Determine the single retained fresh-sweep fraction that the reference's local-stability criterion allows for both iterations at once, using a safety factor of 0.9; equivalently, report the largest retained fraction under that safety-adjusted bound that leaves each of the two schemes locally stable.

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

Implement **all 12 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

matsubara_grids

Goal
----
Build the fermionic and bosonic Matsubara grids of a finite frequency box and return them concatenated, fermionic first.

```python
def matsubara_grids(beta: float, Nf: int, Nb: int) -> 'np.ndarray':
    """Build the fermionic and bosonic Matsubara grids of a finite frequency box and return them concatenated, fermionic first.

    Parameters
    ----------
    beta : float, inverse temperature
    Nf : int, number of fermionic frequencies
    Nb : int, number of bosonic frequencies

    Returns
    -------
    numpy array of shape (Nf+Nb,), the Nf fermionic frequencies followed by the Nb bosonic ones, each in increasing order

    """
    return result
```

### Step 2

bare_propagator

Goal
----
Give the non-interacting Green's function of the hybridised single-site model on the fermionic box, including the interaction-induced shift of the chemical potential and the damping supplied by the bath.

```python
def bare_propagator(beta: float, Nf: int, U: float, dmu: float, hyb: float) -> 'np.ndarray':
    """Give the non-interacting Green's function of the hybridised single-site model on the fermionic box, including the interaction-induced shift of the chemical potential and the damping supplied by the bath.

    Parameters
    ----------
    beta : float, inverse temperature
    Nf : int, number of fermionic frequencies
    U : float, on-site interaction
    dmu : float, chemical potential measured from half filling
    hyb : float, hybridisation strength of the flat bath, non-negative

    Returns
    -------
    numpy array of shape (Nf,), complex, the propagator on the fermionic box

    """
    return result
```

### Step 3

bare_lambda

Goal
----
Give the spin-diagonalised bare interaction in the four channels, derived from the spin structure of the on-site interaction.

```python
def bare_lambda(U: float) -> 'np.ndarray':
    """Give the spin-diagonalised bare interaction in the four channels, derived from the spin structure of the on-site interaction.

    Parameters
    ----------
    U : float, on-site interaction

    Returns
    -------
    numpy array of shape (4,), complex, the channel constants ordered density, magnetic, singlet, triplet

    """
    return result
```

### Step 4

irreducible_vertices

Goal
----
Assemble the channel-irreducible vertices by combining the bare channel constants with the reducible vertices of all four channels under the frequency substitutions and weights given below.

```python
def irreducible_vertices(Phi: 'np.ndarray', Lam: 'np.ndarray') -> 'np.ndarray':
    """Assemble the channel-irreducible vertices by combining the bare channel constants with the reducible vertices of all four channels under the frequency substitutions and weights given below.

    Parameters
    ----------
    Phi : array of shape (4, Nf, Nf, Nb), complex, the reducible vertices
    Lam : array of shape (4,), the bare channel constants

    Returns
    -------
    numpy array of shape (4, Nf, Nf, Nb), complex, the irreducible vertices in the same channel order

    """
    return result
```

### Step 5

pair_bubbles

Goal
----
Build the two bare two-particle bubbles from a propagator: the particle-hole product G(nu)G(nu+omega) and the particle-particle product G(nu)G(-nu-omega).

```python
def pair_bubbles(G: 'np.ndarray', Nb: int) -> 'np.ndarray':
    """Build the two bare two-particle bubbles from a propagator: the particle-hole product G(nu)G(nu+omega) and the particle-particle product G(nu)G(-nu-omega).

    Parameters
    ----------
    G : array of shape (Nf,), complex, the propagator on the fermionic box
    Nb : int, number of bosonic frequencies

    Returns
    -------
    numpy array of shape (2, Nf, Nb), complex, the particle-hole bubble first and the particle-particle bubble second

    """
    return result
```

### Step 6

self_energy

Goal
----
Give the self-energy from the equation of motion: a static Hartree term plus a dynamic term built from whichever combination of the density and magnetic full vertices survives at leading order in the interaction.

```python
def self_energy(Phi: 'np.ndarray', Gam: 'np.ndarray', G: 'np.ndarray', beta: float, U: float) -> 'np.ndarray':
    """Give the self-energy from the equation of motion: a static Hartree term plus a dynamic term built from whichever combination of the density and magnetic full vertices survives at leading order in the interaction.

    Parameters
    ----------
    Phi : array of shape (4, Nf, Nf, Nb), complex, reducible vertices
    Gam : array of shape (4, Nf, Nf, Nb), complex, irreducible vertices
    G : array of shape (Nf,), complex, the propagator
    beta : float, inverse temperature
    U : float, on-site interaction

    Returns
    -------
    numpy array of shape (Nf,), complex, the self-energy on the fermionic box

    """
    return result
```

### Step 7

parquet_map

Goal
----
Apply one full sweep of the self-consistent scheme to a state, returning the next state.

```python
def parquet_map(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float, dmu: float, hyb: float, Nf: int, Nb: int) -> 'np.ndarray':
    """Apply one full sweep of the self-consistent scheme to a state, returning the next state.

    Parameters
    ----------
    state : array of shape (4*Nf*Nf*Nb + Nf,), complex
    Lam : array of shape (4,), bare channel constants
    beta, U, dmu, hyb : float, model parameters
    Nf, Nb : int, box sizes

    Returns
    -------
    numpy array of shape (4*Nf*Nf*Nb + Nf,), complex, the next state in the same layout

    """
    return result
```

### Step 8

relax_parquet

Goal
----
Drive the state to a fixed point of the sweep by damped iteration, mixing a fraction p of the swept state into the old one at every step.

```python
def relax_parquet(state0: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float, dmu: float, hyb: float, Nf: int, Nb: int, p: float) -> 'np.ndarray':
    """Drive the state to a fixed point of the sweep by damped iteration, mixing a fraction p of the swept state into the old one at every step.

    Parameters
    ----------
    state0 : array of shape (4*Nf*Nf*Nb + Nf,), complex, the starting state
    Lam : array of shape (4,), bare channel constants
    beta, U, dmu, hyb : float, model parameters
    Nf, Nb : int, box sizes
    p : float, damping in (0, 1]

    Returns
    -------
    numpy array of shape (4*Nf*Nf*Nb + Nf,), complex, the converged state

    Raises
    ------
    ValueError
        If p lies outside (0, 1], if the iterate becomes non-finite, or if the stopping criterion is not met within the step budget.

    """
    return result
```

### Step 9

gamma_from_full_vertex

Goal
----
Recover the channel-irreducible vertices from the FULL vertices by inverting the ladder relations, rather than by summing them.

```python
def gamma_from_full_vertex(F: 'np.ndarray', G: 'np.ndarray', beta: float, Nb: int) -> 'np.ndarray':
    """Recover the channel-irreducible vertices from the FULL vertices by inverting the ladder relations, rather than by summing them.

    Parameters
    ----------
    F : array of shape (4, Nf, Nf, Nb), complex, the full vertices in channel order
    G : array of shape (Nf,), complex, the propagator the bubbles are built from
    beta : float, inverse temperature
    Nb : int, number of bosonic frequencies

    Returns
    -------
    numpy array of shape (4, Nf, Nf, Nb), complex, the irreducible vertices in channel order

    """
    return result
```

### Step 10

strong_coupling_map

Goal
----
Perform one sweep of the alternative iteration that carries the FULL vertices instead of the reducible ones, in the same flattened layout.

```python
def strong_coupling_map(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float, dmu: float, hyb: float, Nf: int, Nb: int) -> 'np.ndarray':
    """Perform one sweep of the alternative iteration that carries the FULL vertices instead of the reducible ones, in the same flattened layout.

    Parameters
    ----------
    state : array of shape (4*Nf*Nf*Nb + Nf,), complex, full vertices then propagator
    Lam : array of shape (4,), bare channel constants
    beta, U, dmu, hyb : float, model parameters
    Nf, Nb : int, box sizes

    Returns
    -------
    numpy array of shape (4*Nf*Nf*Nb + Nf,), complex, the next state in the same layout

    """
    return result
```

### Step 11

stability_spectrum

Goal
----
Give the eigenvalues of the matrix that controls local convergence of the damped iteration for EITHER sweep, namely the identity minus the derivative of the named sweep with respect to the state.

```python
def stability_spectrum(state: 'np.ndarray', Lam: 'np.ndarray', beta: float, U: float, dmu: float, hyb: float, Nf: int, Nb: int, scheme: str) -> 'np.ndarray':
    """Give the eigenvalues of the matrix that controls local convergence of the damped iteration for EITHER sweep, namely the identity minus the derivative of the named sweep with respect to the state.

    Parameters
    ----------
    state : array of shape (4*Nf*Nf*Nb + Nf,), complex, in the layout the named sweep expects
    Lam : array of shape (4,), bare channel constants
    beta, U, dmu, hyb : float, model parameters
    Nf, Nb : int, box sizes
    scheme : str, either 'reducible' or 'full'

    Returns
    -------
    numpy array of shape (4*Nf*Nf*Nb + Nf,), complex, the eigenvalues of Pi

    Raises
    ------
    ValueError
        If the state length does not match the box, or if scheme is neither 'reducible' nor 'full'.

    """
    return result
```

### Step 12

minimal_damping_audit

Goal
----
Run the benchmark configuration end to end and report the largest retained fresh-sweep fraction that still leaves the fixed point locally stable under the safety-adjusted criterion for BOTH iteration schemes at once.

```python
def minimal_damping_audit(c: float, p_solve: float) -> float:
    """Run the benchmark configuration end to end and report the largest retained fresh-sweep fraction that still leaves the fixed point locally stable under the safety-adjusted criterion for BOTH iteration schemes at once.

    Parameters
    ----------
    c : float, safety factor strictly between zero and one
    p_solve : float, damping used to reach the fixed point, in (0, 1]

    Returns
    -------
    float, the largest retained fresh-sweep fraction allowed by the safety-adjusted criterion

    Raises
    ------
    ValueError
        If c lies outside (0, 1), if p_solve lies outside (0, 1], if the reducible sweep fails to reach its fixed point at the given p_solve, which happens above roughly 0.82, or if either stability spectrum has an eigenvalue with non-positive real part.

    """
    return result
```

# Physics-Computational_Physics-27

## Background

Non-Hermitian Hamiltonians arise whenever a system exchanges energy or particles with its
surroundings — monitored open quantum systems, photonic media with gain and loss, non-reciprocal and
active matter. The generator is then a complex, generally non-normal matrix and the evolution is
non-unitary: probability is not conserved and eigenvectors need not be orthogonal. Such systems host
effects with no Hermitian analogue, notably exceptional points and the non-Hermitian skin effect.
Simulating them means propagating a state under a matrix exponential. For the large sparse
Hamiltonians typical of lattice problems the propagator cannot be formed explicitly, so the
workhorse is a polynomial expansion needing only matrix-vector products. Chebyshev expansions have
filled that role since the 1980s, with closed-form coefficients and a cheap three-term recursion.
Extending them to non-Hermitian generators has been treated as unresolved. The Hermitian recipe
first maps the spectrum onto the real interval where the polynomials stay bounded, and much of the
physics literature reads that interval as a hard limit of validity — motivating substitutes such as
Runge-Kutta integration, Faber polynomials and Hermitisation schemes. Approximation theory instead
holds that the Chebyshev series of an entire function converges over the whole complex plane, so the
two literatures appear to conflict. Non-normality sharpens the stakes, since such matrices lack the
perturbation stability that Weyl's inequality supplies in the Hermitian case.
The work behind this task resolves that conflict: the expansion stays valid for arbitrary complex
spectra, and the real obstacle is accumulated floating-point rounding rather than any failure of
convergence. Accuracy degrades in a structured way with distance from the real interval, and that
structure yields analytic estimates for choosing simulation parameters that keep numerical error
below a target. The demonstration system is a one-dimensional chain with directional hopping — the
minimal model of non-reciprocity, solvable in closed form under both open and periodic boundary
conditions, and readily pushed into the complex regime the conventional treatment excludes.

## Problem

Non-Hermitian generators describe open, dissipative and non-reciprocal systems, and propagating a state under one is a routine but numerically delicate task. Expanding the propagator in Chebyshev polynomials of the first kind is standard practice for Hermitian problems, where the spectrum is first mapped affinely into the interval $[-1, 1]$, and the physics literature has long held that the expansion is unusable outside that interval; a recent analysis established instead that the expansion remains valid across the entire complex plane, and that what limits it in practice is floating-point rounding rather than convergence. That analysis supplies a closed-form upper bound on the rounding error accumulated by one expansion, and reading the bound backwards converts a requested accuracy into a ceiling on the length of a single time step. Your task is to apply this error-controlled scheme to one concrete non-reciprocal chain and report a single number characterising the numerical error budget of the whole simulation.

The physical system is a one-dimensional tight-binding ring of $N = 100$ sites with non-reciprocal nearest-neighbour hoppings, the amplitude for hopping onto the left-hand neighbour being $\gamma(1 + p)$ and onto the right-hand neighbour $\gamma(1 - p)$, with $\gamma = 0.85$, $p = 0.35$ and periodic boundary conditions. The initial state is a Gaussian envelope of width $\sigma = 10$ centred on the middle of the chain and carrying momentum $k = \pi/2$, normalised to unit Euclidean norm and propagated to a total time $T = 20$; because the generator is non-Hermitian the norm is not conserved, so the state is renormalised after every step.

Measure how far that spectrum sits from $[-1, 1]$ using whichever measure governs how fast the expansion polynomials grow away from that interval, convert the result into the longest single time step whose accumulated rounding error stays within $\Delta_{\max} = 10^{-12}$, and take the smallest number of equal steps that tiles $T = 20$ without exceeding that ceiling. Propagate the packet over those steps by applying the three-term Chebyshev recursion directly to the state vector, truncating each expansion adaptively by stopping once the Euclidean norm of the added contribution has stayed below $10^{-14}$ for five consecutive orders, and record $M$, the total number of Chebyshev terms summed across all steps including the two lowest orders that seed the recursion, taking the double-precision unit roundoff throughout to be $\varepsilon = 1.11 \times 10^{-16}$. The scalars that determine the result, and which your reasoning should therefore report, are the enclosing radius, the step ceiling, the step count and the realised step, the per-step and total Chebyshev term counts, the norm of the first term to fall below the stopping tolerance together with that tolerance, and the per-step growth of the state norm; name also the two identities you used, one to select the step and one to bound the error. Quote every one of those scalars to at least five significant figures. Your final answer must be a single number, quoted to at least five significant figures: the base-10 logarithm of the accumulated rounding-error bound for the whole simulation, evaluated with the realised term count $M$ and the time step actually taken.

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

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_build_hatano_nelson

Goal
----
Assemble the Hatano-Nelson Hamiltonian matrix for a one-dimensional tight-binding

chain with non-reciprocal nearest-neighbour hoppings.



The first super-diagonal carries gamma * (1 + p) and the first sub-diagonal carries

gamma * (1 - p). Under periodic boundary conditions the bottom-left corner

H[N-1, 0] carries gamma * (1 + p) and the top-right corner H[0, N-1] carries

gamma * (1 - p); under open boundary conditions both corners stay zero.



Inputs

------

n_sites: int, number of chain sites N (>= 2)

gamma: float, hopping energy scale

p: float, non-reciprocity, |p| < 1

alpha_bc: float, boundary switch, 0.0 (OBC) or 1.0 (PBC)



Returns

-------

H: (n_sites, n_sites) complex ndarray



Raises

------

ValueError: if n_sites is below 2, gamma is not finite, |p| is not below 1, or alpha_bc is neither 0.0 nor 1.0

```python
import numpy as np


def build_hatano_nelson(n_sites: int, gamma: float, p: float, alpha_bc: float) -> np.ndarray:
    '''Assemble the Hatano-Nelson Hamiltonian in the site basis.

    Parameters
    ----------
    n_sites : int
        Number of chain sites N, must be >= 2.
    gamma : float
        Hopping energy scale multiplying every hopping amplitude.
    p : float
        Non-reciprocity of the hoppings, must satisfy |p| < 1.
    alpha_bc : float
        Boundary-condition switch: 0.0 for open, 1.0 for periodic.

    Returns
    -------
    H : np.ndarray
        Complex array of shape (n_sites, n_sites) holding the Hamiltonian.

    Raises
    ------
    ValueError
        Raised if n_sites is below 2, gamma is not finite, |p| is not below 1, or alpha_bc is neither 0.0 nor 1.0.
    '''
    return H
```

### Step 2

02_analytic_spectrum

Goal
----
Return the closed-form eigenvalue spectrum of the Hatano-Nelson chain, selecting

the periodic or the open branch according to the boundary switch.



Inputs

------

n_sites: int, number of chain sites N (>= 2)

gamma: float, hopping energy scale

p: float, non-reciprocity, |p| < 1

alpha_bc: float, boundary switch, 0.0 (OBC) or 1.0 (PBC)



Returns

-------

spectrum: (n_sites,) complex ndarray, ordered by the mode index a = 1 .. N



Raises

------

ValueError: if n_sites is below 2, gamma is not finite, |p| is not below 1, or alpha_bc is neither 0.0 nor 1.0

```python
import numpy as np


def analytic_spectrum(n_sites: int, gamma: float, p: float, alpha_bc: float) -> np.ndarray:
    '''Evaluate the closed-form Hatano-Nelson eigenvalues.

    Parameters
    ----------
    n_sites : int
        Number of chain sites N, must be >= 2.
    gamma : float
        Hopping energy scale.
    p : float
        Non-reciprocity of the hoppings, must satisfy |p| < 1.
    alpha_bc : float
        Boundary-condition switch: 0.0 for open, 1.0 for periodic.

    Returns
    -------
    spectrum : np.ndarray
        Complex array of shape (n_sites,) holding the eigenvalues in order a = 1 .. N.

    Raises
    ------
    ValueError
        Raised if n_sites is below 2, gamma is not finite, |p| is not below 1, or alpha_bc is neither 0.0 nor 1.0.
    '''
    return spectrum
```

### Step 3

03_bernstein_radius

Goal
----
Determine the smallest Bernstein-ellipse radius whose ellipse encloses a supplied

set of complex points, such as a spectrum.



Inputs

------

points: array-like of complex numbers, non-empty and finite



Returns

-------

rho: float, the smallest Bernstein radius rho >= 1 enclosing every supplied point



Raises

------

ValueError: if points is empty or contains a non-finite value

```python
import numpy as np


def bernstein_radius(points: np.ndarray) -> float:
    '''Compute the smallest Bernstein-ellipse radius enclosing a set of points.

    Parameters
    ----------
    points : np.ndarray
        Array-like of complex (or real) points to be enclosed. Must be non-empty
        and contain only finite values.

    Returns
    -------
    rho : float
        The smallest radius rho >= 1 such that every supplied point lies inside or
        on the Bernstein ellipse E_rho.

    Raises
    ------
    ValueError
        Raised if points is empty or contains a non-finite value.
    '''
    return rho
```

### Step 4

04_max_time_step

Goal
----
Convert a target accuracy into the longest single time step a Chebyshev expansion

can take without its accumulated rounding error exceeding that target.



Inputs

------

rho: float, Bernstein radius enclosing the spectrum, >= 1

delta_max: float, target absolute tolerance, > 0

eps: float, machine precision, > 0 (default 1.11e-16)



Returns

-------

t_max: float, the largest single time step meeting the tolerance



Raises

------

ValueError: if rho is below 1, or delta_max or eps is not positive and finite

```python
import numpy as np


def max_time_step(rho: float, delta_max: float, eps: float = 1.11e-16) -> float:
    '''Compute the largest admissible Chebyshev time step for a target tolerance.

    Parameters
    ----------
    rho : float
        Bernstein-ellipse radius enclosing the spectrum, must be >= 1.
    delta_max : float
        Target absolute rounding-error tolerance, must be > 0.
    eps : float
        Machine precision used in the error bound, must be > 0.

    Returns
    -------
    t_max : float
        Largest time step t such that the accumulated rounding error of a single
        Chebyshev expansion stays within delta_max.

    Raises
    ------
    ValueError
        Raised if rho is below 1, or delta_max or eps is not positive and finite.
    '''
    return t_max
```

### Step 5

05_rounding_error_bound

Goal
----
Evaluate the accumulated floating-point rounding-error bound of a truncated

Chebyshev expansion, given the number of terms actually summed.



Inputs

------

n_terms: int, total number of Chebyshev terms summed, >= 1

dt: float, the time step of a single expansion, > 0

rho: float, Bernstein radius enclosing the spectrum, >= 1

eps: float, machine precision, > 0 (default 1.11e-16)



Returns

-------

bound: float, the accumulated rounding-error bound for the stated number of summed terms



Raises

------

ValueError: if n_terms is below 1, dt is not positive, rho is below 1, or eps is not positive

```python
import numpy as np


def rounding_error_bound(n_terms: int, dt: float, rho: float, eps: float = 1.11e-16) -> float:
    '''Evaluate the accumulated Chebyshev rounding-error bound.

    Parameters
    ----------
    n_terms : int
        Total number of Chebyshev terms summed, must be >= 1.
    dt : float
        Length of a single time step, must be > 0.
    rho : float
        Bernstein-ellipse radius enclosing the spectrum, must be >= 1.
    eps : float
        Machine precision used in the bound, must be > 0.

    Returns
    -------
    bound : float
        The accumulated rounding-error bound for the stated number of summed terms.

    Raises
    ------
    ValueError
        Raised if n_terms is below 1, dt is not positive, rho is below 1, or eps is not positive.
    '''
    return bound
```

### Step 6

06_chebyshev_coefficients

Goal
----
Generate the Chebyshev expansion coefficients of the propagator exp(-i t H) up to

a given order, for a bare time step t.



Inputs

------

dt: float, the time step t, > 0

max_order: int, highest expansion order retained, >= 1



Returns

-------

coefficients: (max_order + 1,) complex ndarray, entries c_0 .. c_max_order



Raises

------

ValueError: if dt is not positive and finite, or max_order is below 1

```python
import numpy as np
from scipy.special import jv


def chebyshev_coefficients(dt: float, max_order: int) -> np.ndarray:
    '''Build the Chebyshev expansion coefficients of exp(-i t H).

    Parameters
    ----------
    dt : float
        Time step t entering the Bessel functions, must be > 0.
    max_order : int
        Highest Chebyshev order retained, must be >= 1.

    Returns
    -------
    coefficients : np.ndarray
        Complex array of shape (max_order + 1,) holding c_0 .. c_max_order.

    Raises
    ------
    ValueError
        Raised if dt is not positive and finite, or max_order is below 1.
    '''
    return coefficients
```

### Step 7

07_chebyshev_step

Goal
----
Propagate a state vector through one time step by summing the Chebyshev series

directly onto the vector, truncating it with an adaptive stopping rule, and report

how many terms were summed.



Inputs

------

hamiltonian: (d, d) complex ndarray, may be non-Hermitian

psi: (d,) complex ndarray, the state before the step

dt: float, the time step, > 0

tol: float, term-norm stopping tolerance, > 0 (default 1e-14)

patience: int, consecutive sub-tolerance terms required to stop, >= 1 (default 5)

max_order: int, hard cap on the expansion order, >= 1 (default 4000)



Returns

-------

psi_next: (d,) complex ndarray, the propagated unnormalised state

n_terms: int, the number of Chebyshev terms summed



Raises

------

ValueError: if hamiltonian is not square, psi does not match its dimension, dt or tol is not positive, or patience or max_order is below 1

```python
import numpy as np


def chebyshev_step(
    hamiltonian: np.ndarray,
    psi: np.ndarray,
    dt: float,
    tol: float = 1e-14,
    patience: int = 5,
    max_order: int = 4000,
) -> tuple:
    '''Advance a state by one time step with an adaptively truncated Chebyshev series.

    Parameters
    ----------
    hamiltonian : np.ndarray
        Square array of shape (d, d); may be non-Hermitian.
    psi : np.ndarray
        State vector of shape (d,).
    dt : float
        Length of the time step, must be > 0.
    tol : float
        Euclidean-norm tolerance a term must fall below to count toward stopping.
    patience : int
        Number of consecutive sub-tolerance terms required before stopping.
    max_order : int
        Hard cap on the Chebyshev order, must be >= 1.

    Returns
    -------
    psi_next : np.ndarray
        Propagated, unnormalised state of shape (d,).
    n_terms : int
        Number of Chebyshev terms summed, counting orders 0 and 1.

    Raises
    ------
    ValueError
        Raised if hamiltonian is not square, psi does not match its dimension, dt or tol is not positive, or patience or max_order is below 1.
    '''
    return psi_next, n_terms
```

### Step 8

08_evolve_wavepacket

Goal
----
Evolve a Gaussian wave packet on the Hatano-Nelson chain over a fixed total time

split into equal Chebyshev steps, renormalising after each step, and accumulate the

total number of expansion terms summed.



Inputs

------

n_sites: int, number of chain sites N (>= 2)

gamma: float, hopping energy scale

p: float, non-reciprocity, |p| < 1

alpha_bc: float, boundary switch, 0.0 (OBC) or 1.0 (PBC)

momentum: float, the wave packet momentum k

sigma: float, the wave packet width, > 0

t_max: float, total evolution time, > 0

n_steps: int, number of equal time steps, >= 1

tol: float, term-norm stopping tolerance, > 0 (default 1e-14)

patience: int, consecutive sub-tolerance terms required to stop, >= 1 (default 5)



Returns

-------

psi_final: (n_sites,) complex ndarray, the normalised state at t_max

total_terms: int, Chebyshev terms summed across all steps



Raises

------

ValueError: if n_sites is below 2, |p| is not below 1, alpha_bc is neither 0.0 nor 1.0, sigma or t_max is not positive, or n_steps or patience is below 1

```python
import numpy as np


def evolve_wavepacket(
    n_sites: int,
    gamma: float,
    p: float,
    alpha_bc: float,
    momentum: float,
    sigma: float,
    t_max: float,
    n_steps: int,
    tol: float = 1e-14,
    patience: int = 5,
) -> tuple:
    '''Propagate a Gaussian packet on the Hatano-Nelson chain and count expansion terms.

    Parameters
    ----------
    n_sites : int
        Number of chain sites N, must be >= 2.
    gamma : float
        Hopping energy scale.
    p : float
        Non-reciprocity of the hoppings, must satisfy |p| < 1.
    alpha_bc : float
        Boundary-condition switch: 0.0 for open, 1.0 for periodic.
    momentum : float
        Momentum k imprinted on the initial Gaussian envelope.
    sigma : float
        Width of the initial Gaussian envelope, must be > 0.
    t_max : float
        Total evolution time, must be > 0.
    n_steps : int
        Number of equal time steps, must be >= 1.
    tol : float
        Euclidean-norm tolerance a term must fall below to count toward stopping.
    patience : int
        Number of consecutive sub-tolerance terms required before stopping.

    Returns
    -------
    psi_final : np.ndarray
        Normalised complex state of shape (n_sites,) at time t_max.
    total_terms : int
        Total number of Chebyshev terms summed across every step.

    Raises
    ------
    ValueError
        Raised if n_sites is below 2, |p| is not below 1, alpha_bc is neither 0.0 nor 1.0, sigma or t_max is not positive, or n_steps or patience is below 1.
    '''
    return psi_final, total_terms
```

### Step 9

09_total_error_budget

Goal
----
ORCHESTRATOR - final step. Chain every earlier step into the accumulated

rounding-error budget of a complete non-unitary Chebyshev simulation and return its

base-10 logarithm.



Inputs

------

n_sites: int, number of chain sites N (>= 2)

gamma: float, hopping energy scale

p: float, non-reciprocity, |p| < 1

alpha_bc: float, boundary switch, 0.0 (OBC) or 1.0 (PBC)

momentum: float, the wave packet momentum k

sigma: float, the wave packet width, > 0

t_max: float, total evolution time, > 0

delta_max: float, per-step tolerance driving the step selection, > 0

eps: float, machine precision, > 0 (default 1.11e-16)

tol: float, term-norm stopping tolerance, > 0 (default 1e-14)

patience: int, consecutive sub-tolerance terms required to stop, >= 1 (default 5)



Returns

-------

log10_budget: float, base-10 logarithm of the accumulated rounding-error budget



Composition

-----------

Steps 02, 03, 04, 05 and 08 are called directly from this step and their

results are consumed here. Steps 01, 06 and 07 are composed through step 08,

which builds the Hamiltonian, forms the expansion coefficients and applies the

vector recursion, returning the realised term count this step needs.



Raises

------

ValueError: if n_sites is below 2, |p| is not below 1, alpha_bc is neither 0.0 nor 1.0, or t_max, delta_max or eps is not positive

```python
import numpy as np


def total_error_budget(
    n_sites: int,
    gamma: float,
    p: float,
    alpha_bc: float,
    momentum: float,
    sigma: float,
    t_max: float,
    delta_max: float,
    eps: float = 1.11e-16,
    tol: float = 1e-14,
    patience: int = 5,
) -> float:
    '''Run the full pipeline and return log10 of the accumulated error budget.

    Parameters
    ----------
    n_sites : int
        Number of chain sites N, must be >= 2.
    gamma : float
        Hopping energy scale.
    p : float
        Non-reciprocity of the hoppings, must satisfy |p| < 1.
    alpha_bc : float
        Boundary-condition switch: 0.0 for open, 1.0 for periodic.
    momentum : float
        Momentum k imprinted on the initial Gaussian envelope.
    sigma : float
        Width of the initial Gaussian envelope, must be > 0.
    t_max : float
        Total evolution time, must be > 0.
    delta_max : float
        Per-step rounding-error tolerance driving the step selection, must be > 0.
    eps : float
        Machine precision used in the bounds, must be > 0.
    tol : float
        Euclidean-norm tolerance a term must fall below to count toward stopping.
    patience : int
        Number of consecutive sub-tolerance terms required before stopping.

    Returns
    -------
    log10_budget : float
        Base-10 logarithm of the accumulated rounding-error budget, evaluated with the
        total number of Chebyshev terms summed across the simulation.

    Raises
    ------
    ValueError
        Raised if n_sites is below 2, |p| is not below 1, alpha_bc is neither 0.0 nor 1.0, or t_max, delta_max or eps is not positive.
    '''
    return log10_budget
```

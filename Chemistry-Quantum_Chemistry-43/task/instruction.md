# Chemistry-Quantum_Chemistry-43

## Background

Density-functional theory rests on the map from external potentials to ground-state densities. Kohn-Sham inversion is the reverse problem: given a density, find the potential of a non-interacting system whose ground state reproduces it. Exact inversions are the standard way to obtain reference exchange-correlation potentials, but the problem is ill-posed: not every density is the ground-state density of some potential, the universal functional is not differentiable in the usual sense, and small errors in the density can produce large, oscillatory errors in the potential.
 
Moreau-Yosida regularisation addresses this at the level of the functional. Replacing the universal functional by its infimal convolution with a quadratic penalty on a Hilbert space of densities produces a functional that is differentiable everywhere, with a derivative that can be read off from the proximal density, the minimiser of the regularised problem. As the regularisation parameter is sent to zero, the negative of this derivative converges to a potential of the given density whenever one exists. The choice of density space is not incidental: the Sobolev spaces used for periodic systems penalise rapid oscillations, which is exactly what a potential-targeting inversion needs, and the duality between densities and potentials in that setting is what turns density displacements into potentials.
 
The periodic Gross-Pitaevskii equation is a one-orbital nonlinear eigenvalue problem with the same structure as the Kohn-Sham equations, and for it the potential to be recovered is known in closed form from the ground-state density. That makes it an exact testbed: the proximal density can be computed to machine precision by direct minimisation, and the potential produced by the inversion can be compared with the true one for every value of the regularisation parameter.

## Problem

Kohn-Sham inversion asks for the potential whose non-interacting ground state reproduces a
given density. The map from densities to potentials is ill-posed, and a recent scheme
regularises it by Moreau-Yosida regularisation of the universal functional on a periodic
Sobolev space: the regularised functional has a well-defined derivative at every density, and
the negative of that derivative, obtained from the displacement between the given density and
its proximal density, converges to the sought potential as the regularisation parameter vanishes.
 
Work with the source's own test model, the one-dimensional periodic Gross-Pitaevskii equation
(-1/2 d^2/dx^2 + v_ext + g rho) phi = E phi with rho = |phi|^2 on the cell [0, a) with periodic
boundary conditions, a = 10, unit normalisation (the integral of rho over the cell equals 1),
coupling g = 1, and the optical-lattice potential v_ext(x) = -sin^2(pi x / a). Use a
spectral (plane-wave) discretisation on a uniform grid of M = 256 points, x_j = j a / M, with
Fourier-space derivatives and Fourier-space definitions of every norm, and converge every
minimisation to at least 1e-10 in the density.
 
First obtain the ground-state density rho_gs of this equation. Then carry out the source's
Moreau-Yosida-based inversion for this model at the single regularisation parameter
eps = 0.01: form the proximal density of rho_gs with the guiding functional, the density
space and its norm that the source uses for this example, and from it form the potential that
the source's scheme determines at this finite eps. Report the error of that potential against
the potential the inversion is meant to recover for this model, measured in the norm of the
source's potential space. Alongside that error, report the six scalars that fix the calculation: the
ground-state value of the Gross-Pitaevskii energy functional (kinetic plus external plus interaction
energy, not the eigenvalue E), the maximum of the reference density, the distance between the
proximal and reference densities in the source's density norm, the minimum value of the regularised
objective (the guiding functional at the proximal density plus the Moreau-Yosida penalty), the norm
of the sought potential in the source's potential norm, and the maximum over the cell of the inverted
potential with its constant Fourier mode removed.
 
State the conventions you adopted and justify each from the source.
 
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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
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

external_potential

Goal
----
Return the optical-lattice potential of the source's periodic Gross-Pitaevskii example sampled on the uniform periodic grid x_j = j a / M for j = 0, ..., M-1 of the cell [0, a): v_ext(x) = -sin^2(pi x / a). Raise ValueError if a is not positive or M is below 8.

```python
def external_potential(a: float, M: int) -> "np.ndarray":
    """ndarray of float64 with shape (M,)."""
    return result
```

### Step 2

duality_map

Goal
----
Given the grid values f of a real periodic function on the uniform grid of the cell [0, a), return the grid values of its image under the duality mapping of the source's density space, i.e. the map that sends an element of the density space to the element of the potential space that represents it (the source's Definition 1 applied to its periodic Sobolev setting), using Fourier-space (spectral) arithmetic. Raise ValueError if f is not a one-dimensional array with at least 8 points or a is not positive.

```python
def duality_map(f: "np.ndarray", a: float) -> "np.ndarray":
    """ndarray of float64 with shape (M,)."""
    return result
```

### Step 3

density_norm

Goal
----
Given the grid values f of a real periodic function on the uniform grid of the cell [0, a), return its norm in the source's density space for periodic systems, evaluated with Fourier-space (spectral) arithmetic. Raise ValueError if f is not a one-dimensional array with at least 8 points or a is not positive.

```python
def density_norm(f: "np.ndarray", a: float) -> float:
    """float."""
    return 0.0
```

### Step 4

potential_norm

Goal
----
Given the grid values v of a real periodic potential on the uniform grid of the cell [0, a), return its norm in the source's potential space for periodic systems, evaluated with Fourier-space (spectral) arithmetic. Raise ValueError if v is not a one-dimensional array with at least 8 points or a is not positive.

```python
def potential_norm(v: "np.ndarray", a: float) -> float:
    """float."""
    return 0.0
```

### Step 5

gpe_ground_state

Goal
----
Given the grid values vext of the external potential on the uniform periodic grid of the cell [0, a) and the coupling constant g, return the ground-state density rho = |phi|^2 of the periodic Gross-Pitaevskii equation (-1/2 d^2/dx^2 + vext + g rho) phi = E phi with unit normalisation (integral of rho over the cell equal to 1), computed with spectral derivatives and converged to at least 1e-10 in the density. Raise ValueError if vext is not a one-dimensional array with at least 8 points, if a is not positive, if g is negative, or if the numerical solve does not converge.

```python
def gpe_ground_state(vext: "np.ndarray", a: float, g: float) -> "np.ndarray":
    """ndarray of float64 with shape (M,)."""
    return result
```

### Step 6

proximal_density

Goal
----
Given a reference ground-state density rho_gs and the external potential vext on the uniform periodic grid of the cell [0, a), and a regularisation parameter eps, return the proximal density of the source's Moreau-Yosida-regularised inversion scheme for this Gross-Pitaevskii model: the unit-mass density that minimises the source's guiding functional for this model plus the Moreau-Yosida term with parameter eps, measured in the norm of the source's density space. Converge the minimiser to at least 1e-10 in the density (the source notes that a plain self-consistent-field iteration can fail here). Raise ValueError if rho_gs and vext are not one-dimensional arrays of one length of at least 8 points, if a or eps is not positive, if rho_gs is negative somewhere or does not integrate to 1, or if the numerical solve does not converge.

```python
def proximal_density(
    rho_gs: "np.ndarray",
    vext: "np.ndarray",
    a: float,
    eps: float,
) -> "np.ndarray":
    """ndarray of float64 with shape (M,)."""
    return result
```

### Step 7

inversion_error

Goal
----
Orchestrator. On the uniform periodic grid of M points of the cell [0, a), build the external potential, compute the Gross-Pitaevskii ground-state density for coupling g, compute the proximal density for the regularisation parameter eps, form the potential that the source's inversion scheme determines at this finite eps from the two densities, and return the error of that potential against the potential the inversion is meant to recover for this model, measured in the norm of the source's potential space. Call the earlier step functions rather than reimplementing any of them. Raise ValueError if a, g or eps is out of range, if M is below 8, if an upstream solve does not converge, or if the proximal-density displacement is not finite.

```python
def inversion_error(a: float, M: int, g: float, eps: float) -> float:
    """float, the potential error in the source's norm."""
    return 0.0
```

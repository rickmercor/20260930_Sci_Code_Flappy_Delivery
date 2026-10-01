# Chemistry-Computational_Chemistry-1

## Background

Molecular dynamics is inherently serial: each integrator step needs the forces at the configuration the previous step produced, so adding compute does not make a single trajectory advance faster. For an expensive force field that sequential cost dominates, and it is what keeps accurate machine-learned and ab initio potentials out of the long-timescale regime.

Speculative sampling buys back parallelism without paying for it in accuracy. Borrowed from autoregressive language models, a cheap draft model proposes a run of steps and the expensive target model checks them concurrently, keeping only those distributed as the target itself would have produced. Everything rests on the verification rule being a genuine coupling of the two step distributions rather than a tolerance test: accepting merely similar proposals silently shifts the sampled ensemble, which a free-energy calculation cannot absorb. Carrying this to Langevin dynamics turns on integrator structure, since operator-splitting schemes separate a step into position drifts, force kicks and a thermostat action, and the force field enters only one contiguous block.

Free energy enters by a different route. A free-energy surface is defined on a few collective variables, but those are functions of the full configuration, so the free energy at a given value is an integral over a curved level set rather than a slice through the density. Its gradient is a conditional expectation of a local mean force, which differs from the force simply projected along the collective variable, and reconstructing a profile from that gradient is what makes the estimate usable when sampling is poor.

The two concerns meet because a speculative sampler's rejection rate is not uniform over configuration space: it is large wherever the cheap and expensive force fields disagree, and those regions need not coincide with the barriers and basins that shape the free-energy landscape.

## Problem

Speculative sampling accelerates molecular dynamics by letting a cheap draft force field propose integrator steps that an expensive target force field verifies in parallel, and a recent extension to second-order Langevin dynamics makes that verification exact, because the draft and target momentum updates differ only in their mean and a reflection-maximal coupling therefore accepts or reflects the draft momentum without biasing the sampled distribution. Separately, a recent reduced-space free-energy method works entirely in collective-variable space, obtaining the free-energy gradient as a conditional expectation of a local mean force over level sets of the collective variable. Combining the two answers a question neither addresses alone, namely whether a speculative sampler rejects preferentially in the regions that matter free-energetically: given the two force fields, the thermostat parameters and a collective variable, the output is a single free-energy difference.

The target potential is the four-Gaussian Muller-Brown surface with the standard coefficients, scaled by 0.05 and confined by an added (1/2) * kappa * ||x - x_c||^2 term with kappa = 2.0 and x_c = (-0.5, 0.75); the draft potential keeps only the two Gaussians of that set with the deepest minima, under the identical scaling and confinement, the confinement being applied to both surfaces because the draft's retained Gaussians vanish at infinity and would leave it unconfined on its own. Work in reduced units with k_B * T = 1, unit mass, friction gamma = 1.0 and timestep dt = 0.02, and take the collective variable to be xi(x) = x_0 + c * sin(k * x_1) with c = 0.35 and k = 1.5.

Evaluate the target Boltzmann measure on a cell-centred 400 x 400 tensor grid spanning x_0 in [-1.8, 1.3] and x_1 in [-0.4, 2.1], with cell centres at the midpoint of each cell. For every grid point form the mean momentum each force field would produce after the momentum block of the integrator, obtain the reflection-maximal rejection rate from the Mahalanobis separation of those two means, and independently form the local mean force of the reduced free-energy method from the target gradient. Do not report the value obtained by forming that mean force from the draft gradient instead. Bin the grid into uniform collective-variable bins of width 0.175 spanning [-1.6, 1.2], numbered from 0 at the lowest bin, take the Boltzmann-weighted conditional average of both quantities in each bin, integrate the mean-force profile across bin centres by the trapezoid rule anchored at zero in bin 0, and report the free-energy difference between the bin of highest average rejection and the bin of lowest average rejection. Alongside that difference, state in your reasoning the relations you used (the mean and covariance of the momentum block and how the incoming momentum enters the rejection rate; the closed form of the rejection rate; the local mean force, including any geometric term this collective variable requires; which of the two surfaces actually needs the confinement, and why; and the weighting used for the bin averages), then give a compact audit of the run: the momentum-block variance; the first four and the last four bin-averaged mean forces and rejection rates; the indices of the highest- and lowest-rejection bins with their average rates and each one's gap to its runner-up; the reconstructed free energy in those two bins; the profile minimum and its bin; the range of the per-point rejection rate over the grid; the number of grid points in the least-populated bin; and the value the difference takes when the grid is refined to 800 x 800 with everything else fixed.

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

01_evaluate_reference_surface

Goal
----
Evaluate the reference surface and its gradient on the supplied points.

```python
"""Evaluate the reference surface and its gradient on the supplied points."""

import numpy as np


def evaluate_reference_surface(X: np.ndarray, scale: float = 0.05, kappa: float = 2.0, xc: tuple = (-0.5, 0.75)) -> np.ndarray:
    """Evaluate the reference surface and its gradient on the supplied points.

    Parameters
    ----------
    X
        Finite points with shape ``(n_points, 2)``.
    scale
        Strictly positive amplitude applied to the Gaussian terms.
    kappa
        Strictly positive confinement stiffness.
    xc
        Two-component confinement centre.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, 3)`` holding the surface value and its two
                partial derivatives, in that column order.

    Raises
    ------
    ValueError
        If ``X`` is not a two-dimensional array of shape ``(n_points, 2)``
        holding finite values, if ``scale`` or ``kappa`` is not a finite
        strictly positive number, or if ``xc`` does not hold two finite
        components.
    """
    return np.empty((np.asarray(X).shape[0], 3), dtype=float)  # placeholder
```

### Step 2

02_evaluate_reduced_surface

Goal
----
Evaluate the reduced surface and its gradient on the same points as the reference.

```python
"""Evaluate the reduced surface and its gradient on the same points as the reference."""

import numpy as np


def evaluate_reduced_surface(X: np.ndarray, scale: float = 0.05, kappa: float = 2.0, xc: tuple = (-0.5, 0.75)) -> np.ndarray:
    """Evaluate the reduced surface and its gradient on the same points as the reference.

    Parameters
    ----------
    X
        Finite points with shape ``(n_points, 2)``.
    scale
        Strictly positive amplitude applied to the retained Gaussian
        terms.
    kappa
        Strictly positive confinement stiffness.
    xc
        Two-component confinement centre.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, 3)`` holding the surface value and its two
                partial derivatives, in that column order.

    Raises
    ------
    ValueError
        If ``X`` is not a two-dimensional array of shape ``(n_points, 2)``
        holding finite values, if ``scale`` or ``kappa`` is not a finite
        strictly positive number, or if ``xc`` does not hold two finite
        components.
    """
    return np.empty((np.asarray(X).shape[0], 3), dtype=float)  # placeholder
```

### Step 3

03_compute_propagated_means

Goal
----
Compute the propagated mean of the stochastic update for a supplied force.

```python
"""Compute the propagated mean of the stochastic update for a supplied force."""

import numpy as np
from math import erf


def compute_propagated_means(p_prev: np.ndarray, force: np.ndarray, dt: float, gamma: float) -> np.ndarray:
    """Compute the propagated mean of the stochastic update for a supplied force.

    Parameters
    ----------
    p_prev
        Incoming values with shape ``(n_points, 2)``.
    force
        Applied force aligned with ``p_prev``.
    dt
        Strictly positive step size.
    gamma
        Strictly positive relaxation rate.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, 2)`` holding the propagated mean for each point.

    Raises
    ------
    ValueError
        If ``p_prev`` and ``force`` differ in shape, if either is not of
        shape ``(n_points, 2)``, or if ``dt`` or ``gamma`` is not strictly
        positive.
    """
    return np.empty_like(np.asarray(p_prev, dtype=float))  # placeholder
```

### Step 4

04_compute_separation_fractions

Goal
----
Compute the separation fraction between two propagated mean fields.

```python
"""Compute the separation fraction between two propagated mean fields."""

import numpy as np
from math import erf


def compute_separation_fractions(mean_target: np.ndarray, mean_draft: np.ndarray, dt: float, gamma: float, mass: float, kbt: float) -> np.ndarray:
    """Compute the separation fraction between two propagated mean fields.

    Parameters
    ----------
    mean_target
        Reference means with shape ``(n_points, 2)``.
    mean_draft
        Comparison means aligned with ``mean_target``.
    dt
        Strictly positive step size.
    gamma
        Strictly positive relaxation rate.
    mass
        Strictly positive inertia scale.
    kbt
        Strictly positive thermal energy scale.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points,)`` holding a fraction in ``[0, 1]`` per point.

    Raises
    ------
    ValueError
        If ``mean_target`` and ``mean_draft`` differ in shape, if either is
        not of shape ``(n_points, 2)``, or if any of ``dt``, ``gamma``,
        ``mass`` or ``kbt`` is not strictly positive.
    """
    return np.empty(np.asarray(mean_target).shape[0], dtype=float)  # placeholder
```

### Step 5

05_derive_coordinate_geometry

Goal
----
Derive the reduced coordinate and the geometric factors of its level sets.

```python
"""Derive the reduced coordinate and the geometric factors of its level sets."""

import numpy as np
from math import erf


def derive_coordinate_geometry(X: np.ndarray, c: float, k: float) -> np.ndarray:
    """Derive the reduced coordinate and the geometric factors of its level sets.

    Parameters
    ----------
    X
        Finite points with shape ``(n_points, 2)``.
    c
        Amplitude of the coordinate's oscillatory term.
    k
        Angular frequency of the coordinate's oscillatory term.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points, 4)`` holding the coordinate value, the two
                pseudo-inverse components, and their divergence, in that column order.

    Raises
    ------
    ValueError
        If ``X`` is not a two-dimensional array of shape ``(n_points, 2)``.
    """
    return np.empty((np.asarray(X).shape[0], 4), dtype=float)  # placeholder
```

### Step 6

06_assemble_gradient_terms

Goal
----
Assemble the reduced gradient observable from a gradient and the geometry.

```python
"""Assemble the reduced gradient observable from a gradient and the geometry."""

import numpy as np
from math import erf


def assemble_gradient_terms(grad_U: np.ndarray, cv_geom: np.ndarray) -> np.ndarray:
    """Assemble the reduced gradient observable from a gradient and the geometry.

    Parameters
    ----------
    grad_U
        Surface gradients with shape ``(n_points, 2)``.
    cv_geom
        Geometry table with shape ``(n_points, 4)`` aligned with
        ``grad_U``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_points,)`` holding the assembled observable per point.

    Raises
    ------
    ValueError
        If ``grad_U`` is not of shape ``(n_points, 2)``, or if ``cv_geom`` is
        not of shape ``(n_points, 4)`` with the same number of rows.
    """
    return np.empty(np.asarray(grad_U).shape[0], dtype=float)  # placeholder
```

### Step 7

07_aggregate_binned_averages

Goal
----
Aggregate a weighted conditional average of a quantity over uniform bins.

```python
"""Aggregate a weighted conditional average of a quantity over uniform bins."""

import numpy as np
from math import erf


def aggregate_binned_averages(cv: np.ndarray, values: np.ndarray, logw: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Aggregate a weighted conditional average of a quantity over uniform bins.

    Parameters
    ----------
    cv
        Coordinate values with shape ``(n_points,)``.
    values
        Quantity to average, aligned with ``cv``.
    logw
        Natural-log weights, aligned with ``cv``.
    edges
        Monotone bin edges with shape ``(n_bins + 1,)``.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_bins,)`` holding the weighted average in each bin.

    Raises
    ------
    ValueError
        If ``cv``, ``values`` and ``logw`` do not share one shape, or if
        ``edges`` is not one-dimensional with at least two entries.
    """
    return np.empty(np.asarray(edges).size - 1, dtype=float)  # placeholder
```

### Step 8

08_resolve_profile_scalar

Goal
----
Apply the complete analysis and return the selected scalar.

```python
"""Apply the complete analysis and return the selected scalar."""

import numpy as np
from math import erf


def resolve_profile_scalar(x_lo: float, x_hi: float, y_lo: float, y_hi: float, n_grid: int, n_bins: int, cv_lo: float, cv_hi: float, c: float, k: float, dt: float, gamma: float, mass: float, kbt: float, scale: float = 0.05, kappa: float = 2.0, xc: tuple = (-0.5, 0.75)) -> float:
    """Apply the complete analysis and return the selected scalar.

    Parameters
    ----------
    x_lo, x_hi
        Increasing bounds of the first coordinate.
    y_lo, y_hi
        Increasing bounds of the second coordinate.
    n_grid
        Number of cell-centred nodes per dimension (``n_grid >= 2``).
    n_bins
        Number of uniform bins (``n_bins >= 2``).
    cv_lo, cv_hi
        Increasing bounds of the binned coordinate range.
    c, k
        Amplitude and angular frequency of the reduced coordinate.
    dt, gamma
        Strictly positive step size and relaxation rate.
    mass, kbt
        Strictly positive inertia and thermal energy scales.
    scale, kappa, xc
        Surface amplitude, confinement stiffness and centre.

    Returns
    -------
    float
        The reported scalar, as a native Python float.

    Raises
    ------
    ValueError
        If ``n_grid`` or ``n_bins`` is below 2, or if any collective-variable
        bin receives no grid point, which leaves its conditional average
        undefined. A ``ValueError`` raised by an earlier step for its own
        invalid inputs propagates unchanged.
    """
    return 0.0  # placeholder
```

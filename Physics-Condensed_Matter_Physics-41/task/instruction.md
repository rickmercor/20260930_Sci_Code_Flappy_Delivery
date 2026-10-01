# Physics-Condensed_Matter_Physics-41

## Background

Path integral Monte Carlo (PIMC) represents a quantum particle at inverse temperature $\beta$ by a periodic imaginary-time worldline containing $P$ beads. At low temperature, standard free-particle fixed-endpoint bridge proposals can be inefficient in a strongly confined anharmonic basin because they ignore the local curvature and frequently propose low-probability bead positions. A mixed reference-action approach instead uses the exactly solvable harmonic approximation around a local potential minimum inside a finite domain and retains free-particle bridges outside that domain. Because the proposal family changes with the bead position, moves that cross the domain boundary require a generalized Metropolis–Hastings correction to preserve detailed balance. Under the same harmonic reference splitting, the finite-$P$ total-energy estimator combines analytically known harmonic link contributions with the residual anharmonic potential. This construction improves sampling in locally confined regions while retaining a valid discretized path integral distribution across the full configuration space.

## Problem

Consider a one-dimensional periodic $P$-bead worldline for a quantum particle in the anharmonic well specified below, and perform one sequential sweep of the state-dependent mixed harmonic/free fixed-endpoint update. Process the listed 0-based bead indices in order, using the current path at each move, the supplied standard-normal deviate to instantiate the selected Gaussian proposal, the supplied uniform deviate, and the strict acceptance rule $u<A$. Use harmonic-reference Trotter splitting, the mixed-kernel generalized Metropolis–Hastings correction, $q=x-x_\star$, the closed harmonic domain $|q|\le d$, $V_{\mathrm{ho}}(q)=\tfrac12 kq^2$, and $V(q)=\tfrac12 kq^2(1+\alpha_3q+\alpha_4q^2)$. The numerical inputs are:
```text
m = 1.3
hbar = 0.7
beta = 4.8
P = 8
x_star = -0.15
k = 3.2
alpha_3 = 0.3125
alpha_4 = 12.5
d = 0.44
initial_path = [0.08, 0.24, 0.22, 0.26, 0.03, -0.12, -0.60, -0.05]
bead_indices = [3, 4, 6]
normal_draws = [0.7, 0.5, 1.1]
uniform_draws = [0.8964, 0.8590, 0.9477]
```

After all three accept/reject decisions, evaluate the finite-$P$ harmonic-reference total-energy estimator on the final path with periodic closure, including the last-to-first link. Use IEEE-754 binary64 arithmetic without intermediate rounding and report the single energy value rounded to exactly 12 digits after the decimal point.

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

compute_reference_frequency

Goal
----
Compute the local harmonic reference frequency from a positive particle mass and positive curvature, returning one native Python float.

```python
import math

from numbers import Real

import numpy as np


def compute_reference_frequency(mass: float, curvature: float) -> float:
    """Compute the local harmonic reference frequency.

    Parameters
    ----------
    mass : float
        Positive particle mass.
    curvature : float
        Positive second derivative of the potential at the local minimum.

    Returns
    -------
    omega : float
        Positive native Python float for the harmonic reference frequency.
        The result must remain finite whenever the exact positive
        square-root ratio is representable, even if the direct
        curvature-to-mass quotient is not representable in binary64.

    Raises
    ------
    ValueError
        If ``mass`` or ``curvature`` is not a finite real scalar or is not
        strictly positive, or if the resulting frequency is non-finite or
        non-positive.
    """
    return float("nan")
```

### Step 2

compute_bridge_statistics

Goal
----
Compute the harmonic and free fixed-endpoint Gaussian bridge moments for one bead between two fixed neighboring beads.

```python
import math

from numbers import Real

import numpy as np


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def compute_bridge_statistics(
    left: float,
    right: float,
    minimum: float,
    tau: float,
    mass: float,
    omega: float,
    hbar: float,
) -> np.ndarray:
    """Compute exact harmonic and free fixed-endpoint bridge statistics.

    Parameters
    ----------
    left, right : float
        Fixed neighboring bead positions.
    minimum : float
        Position of the local harmonic minimum.
    tau : float
        Positive imaginary-time step.
    mass : float
        Positive particle mass.
    omega : float
        Positive harmonic reference frequency.
    hbar : float
        Positive reduced Planck constant.

    Returns
    -------
    statistics : numpy.ndarray
        Length-4 float array ``[harmonic_mean, harmonic_variance,
        free_mean, free_variance]``. Both the hyperbolic factors and the
        affine endpoint combinations must be evaluated without overflow
        for finite inputs whose four exact outputs are finite.

    Raises
    ------
    ValueError
        If any scalar input is not a finite real value; if ``tau``,
        ``mass``, ``omega``, or ``hbar`` is not strictly positive; or if
        the resulting bridge statistics are non-finite or either variance
        is not strictly positive.
    """
    return np.full(4, np.nan, dtype=float)
```

### Step 3

select_proposal_parameters

Goal
----
Select the Gaussian proposal mean and variance from the current bead’s membership in the closed harmonic domain.

```python
import math

from numbers import Real

import numpy as np

def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

def select_proposal_parameters(
    current: float,
    minimum: float,
    domain_radius: float,
    harmonic_mean: float,
    harmonic_variance: float,
    free_mean: float,
    free_variance: float,
) -> np.ndarray:
    """Select the state-dependent Gaussian proposal parameters.

    Parameters
    ----------
    current : float
        Current bead position.
    minimum : float
        Center of the harmonic domain.
    domain_radius : float
        Nonnegative radius of the closed harmonic domain.
    harmonic_mean, harmonic_variance : float
        Mean and positive variance of the harmonic bridge.
    free_mean, free_variance : float
        Mean and positive variance of the free bridge.

    Returns
    -------
    selected : numpy.ndarray
        Length-2 float array ``[selected_mean, selected_variance]``.

    Raises
    ------
    ValueError
        If any scalar input is not a finite real value; if
        ``domain_radius`` is negative; or if ``harmonic_variance`` or
        ``free_variance`` is not strictly positive.
    """
    return np.full(2, np.nan, dtype=float)
```

### Step 4

generate_trial_position

Goal
----
Instantiate one deterministic Gaussian trial position from a selected mean, variance, and supplied standard-normal deviate.

```python
import math

from numbers import Real

import numpy as np

def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result

def generate_trial_position(
    mean: float,
    variance: float,
    normal_draw: float,
) -> float:
    """Instantiate a Gaussian bridge proposal from a fixed normal deviate.

    Parameters
    ----------
    mean : float
        Gaussian proposal mean.
    variance : float
        Positive Gaussian proposal variance.
    normal_draw : float
        Supplied finite standard-normal deviate.

    Returns
    -------
    trial : float
        Proposed bead position as a native Python float.

    Raises
    ------
    ValueError
        If ``mean``, ``variance``, or ``normal_draw`` is not a finite
        real value; if ``variance`` is not strictly positive; or if the
        resulting trial position is non-finite.
    """
    return float("nan")
```

### Step 5

compute_hastings_factor

Goal
----
Compute the generalized Hastings correction generated by state-dependent switching between harmonic and free bridge proposals.

```python
import math

from numbers import Real

import numpy as np


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _log_normal_pdf(value: float, mean: float, variance: float) -> float:
    standardized = (value - mean) / math.sqrt(variance)
    return -0.5 * (
        math.log(2.0 * math.pi)
        + math.log(variance)
        + standardized * standardized
    )


def compute_hastings_factor(
    old: float,
    trial: float,
    minimum: float,
    domain_radius: float,
    harmonic_mean: float,
    harmonic_variance: float,
    free_mean: float,
    free_variance: float,
) -> float:
    """Compute the generalized state-dependent Hastings factor.

    Parameters
    ----------
    old, trial : float
        Current and proposed bead positions.
    minimum : float
        Center of the closed harmonic domain.
    domain_radius : float
        Nonnegative radius of the harmonic domain.
    harmonic_mean, harmonic_variance : float
        Mean and positive variance of the harmonic bridge.
    free_mean, free_variance : float
        Mean and positive variance of the free bridge.

    Returns
    -------
    factor : float
        Positive finite generalized Hastings factor as a native Python
        float. Evaluate the required Gaussian likelihood ratios directly
        in a cancellation-safe log form. The result must remain finite
        when individual densities or individual standardized quadratic
        terms are not representable, including the unequal-variance case,
        provided the exact combined log-ratio is finite.

    Raises
    ------
    ValueError
        If any scalar input is not a finite real value; if
        ``domain_radius`` is negative; if either proposal variance is not
        strictly positive; or if the resulting factor is non-finite or
        non-positive.
    """
    return float("nan")
```

### Step 6

compute_mixed_acceptance

Goal
----
Compute the mixed-proposal Metropolis acceptance probability using the proposal correction and the anharmonic residual potential.

```python
import math

from numbers import Real

import numpy as np


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def compute_mixed_acceptance(
    old: float,
    trial: float,
    tau: float,
    minimum: float,
    curvature: float,
    alpha_3: float,
    alpha_4: float,
    hastings_factor: float,
) -> float:
    """Compute the harmonic-splitting mixed-proposal acceptance probability.

    Parameters
    ----------
    old, trial : float
        Current and proposed bead positions.
    tau : float
        Positive imaginary-time step.
    minimum : float
        Position of the local minimum.
    curvature : float
        Positive harmonic curvature.
    alpha_3, alpha_4 : float
        Finite cubic and quartic coefficients of the shifted residual.
    hastings_factor : float
        Positive finite generalized proposal correction.

    Returns
    -------
    acceptance : float
        Acceptance probability in the closed interval ``[0, 1]``. The
        residual-potential change must be evaluated directly in a
        scale-aware paired form whenever separate residuals or separate
        unweighted powers are not representable although the exact
        weighted difference is finite. The untruncated ratio must be
        handled in log space.

    Raises
    ------
    ValueError
        If any scalar input is not a finite real value; if ``tau``,
        ``curvature``, or ``hastings_factor`` is not strictly positive; if
        the finite residual-action difference cannot be represented; or
        if the resulting probability is non-finite.
    """
    return float("nan")
```

### Step 7

compute_harmonic_energy

Goal
----
Evaluate the finite-bead harmonic-reference total-energy estimator on a periodic one-dimensional worldline.

```python
import math

from numbers import Real

import numpy as np


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def compute_harmonic_energy(
    path: np.ndarray,
    beta: float,
    mass: float,
    omega: float,
    hbar: float,
    minimum: float,
    alpha_3: float,
    alpha_4: float,
) -> float:
    """Evaluate the finite-P harmonic-reference total-energy estimator.

    Parameters
    ----------
    path : numpy.ndarray
        One-dimensional periodic path with at least two finite bead
        positions.
    beta : float
        Positive inverse temperature.
    mass : float
        Positive particle mass.
    omega : float
        Positive harmonic reference frequency.
    hbar : float
        Positive reduced Planck constant.
    minimum : float
        Position of the local minimum.
    alpha_3, alpha_4 : float
        Finite coefficients of the shifted anharmonic residual.

    Returns
    -------
    energy : float
        Finite harmonic-reference total-energy estimate as a native
        Python float. Periodic closure, stable small- and large-argument
        hyperbolic limits, and scale-aware evaluation of the mass-weighted
        cubic-quartic residual are required. The result must remain finite
        whenever the exact weighted link and residual terms are finite,
        even if unweighted powers or separate polynomial contributions are
        not representable in binary64.

    Raises
    ------
    ValueError
        If ``path`` cannot be represented as a finite one-dimensional
        float array with at least two beads; if any scalar input is not a
        finite real value; if ``beta``, ``mass``, ``omega``, or ``hbar``
        is not strictly positive; or if the resulting energy is non-finite.
    """
    return float("nan")
```

### Step 8

run_mixed_pimc_sweep

Goal
----
Run the ordered mixed harmonic/free bead updates on a periodic path and return the harmonic-reference energy of the final path.

```python
import math

from numbers import Integral, Real

import numpy as np


def _finite_float(name: str, value: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def run_mixed_pimc_sweep(
    path: np.ndarray,
    bead_indices: np.ndarray,
    normal_draws: np.ndarray,
    uniform_draws: np.ndarray,
    beta: float,
    mass: float,
    curvature: float,
    hbar: float,
    minimum: float,
    domain_radius: float,
    alpha_3: float,
    alpha_4: float,
) -> float:
    """Run a deterministic sequential mixed-kernel PIMC sweep.

    Parameters
    ----------
    path : numpy.ndarray
        One-dimensional periodic path with at least three finite beads.
    bead_indices : numpy.ndarray
        One-dimensional integer indices processed sequentially.
    normal_draws : numpy.ndarray
        Finite standard-normal deviates, one per proposed move.
    uniform_draws : numpy.ndarray
        Values in ``[0, 1)`` used with the strict rule ``u < A``.
    beta : float
        Positive inverse temperature.
    mass : float
        Positive particle mass.
    curvature : float
        Positive local curvature.
    hbar : float
        Positive reduced Planck constant.
    minimum : float
        Position of the local minimum.
    domain_radius : float
        Nonnegative radius of the closed harmonic domain.
    alpha_3, alpha_4 : float
        Finite coefficients of the anharmonic well.

    Returns
    -------
    energy : float
        Final harmonic-reference energy estimate as a native Python
        float. The input ``path`` must not be modified.

    Raises
    ------
    ValueError
        If an array has invalid shape, length, type, index, or non-finite
        content; if a uniform draw lies outside ``[0, 1)``; if ``beta``,
        ``mass``, ``curvature``, or ``hbar`` is not strictly positive; or
        if ``domain_radius`` is negative.
    """
    return float("nan")
```

# Physics-Condensed_Matter_Physics-29

## Background

Fermionic Gaussian phase-space representations replace the exponentially large many-body wavefunction by stochastic trajectories of one-body density-matrix variables. For a number-conserving Hubbard system, hopping generates deterministic drift while the onsite interaction generates complex diffusion linking opposite-spin sectors. Observables are recovered as trajectory averages, so the numerical form of the noise factor directly affects the stability and useful simulation time even when it represents the same formal diffusion tensor.

The central computational improvement is to exploit the low-rank, opposite-spin block structure of the diffusion tensor instead of factorizing the full matrix blindly. A randomized subspace construction compresses the relevant column space before a projected singular-value decomposition, and a structured complex noise factor then reproduces the required Itô covariance. This task exercises that state-dependent contraction, low-rank factorization, deterministic gauge convention, stochastic propagation, and an interaction-energy estimator in one small system that remains executable with NumPy alone.

## Problem

A three-site spinful Fermi–Hubbard Gaussian phase-space trajectory is represented by the complex one-body matrices $n_\uparrow=\begin{bmatrix}0.82&0.12+0.05i&-0.04+0.02i\\0.12-0.05i&0.47&0.09-0.03i\\-0.04-0.02i&0.09+0.03i&0.21\end{bmatrix}$ and $n_\downarrow=\begin{bmatrix}0.18&-0.07+0.02i&0.05-0.01i\\-0.07-0.02i&0.53&-0.11+0.04i\\0.05+0.01i&-0.11-0.04i&0.79\end{bmatrix}$, with hopping matrix $J=\begin{bmatrix}0&1&0.35\\1&0&0.8\\0.35&0.8&0\end{bmatrix}$, onsite interaction $U=1.3$, and $\hbar=1$. Use zero-based site indices, row-major ordering within each spin block, spin-up variables before spin-down variables, and hole matrices $\tilde n_\sigma=I-n_\sigma$. Construct the hopping drift and the onsite opposite-spin diffusion tensor using the explicit indexed equations in Appendix A, Eqs. (A4) and (A5), respectively, then obtain the low-rank numerical diffusion gauge using the source-specific opposite-spin block reduction and a randomized range finder with target rank $r=2n_s=6$. Generate the real standard-normal sketch with `numpy.random.default_rng(314159)`, use economy QR followed by a projected SVD, discard singular values not exceeding $64\,\epsilon_{\mathrm{mach}}\max(m,n)s_{\max}$, and remove phase ambiguity by choosing, for each retained left singular vector, the smallest index whose magnitude is within $10^{-10}$ relative of the column maximum and rotating that entry to be real and nonnegative. Verify the complex Itô covariance factorization, then perform one Euler–Maruyama step of size $dt=0.015$ for $M=7$ trajectories using real standard-normal Wiener draws from `numpy.random.default_rng(271828)` in an array whose first axis indexes noise channels and whose second axis indexes trajectories; do not round intermediate values. For each updated trajectory, evaluate the onsite interaction-energy density estimator $U\,n_s^{-1}\sum_i n_{ii\uparrow}n_{ii\downarrow}$, average across trajectories, take the real part, and round once to 12 decimal places. In `<reasoning>`, state the source-specific opposite-spin contraction, reduced singular-value factorization, and structured noise-factor identities, then report the stacked drift norm, opposite-spin-block norm and its zero-based row-major entries $(D_q)_{1,3}$ and $(D_q)_{2,7}$, retained singular values and effective rank, phase-pivot indices, noise-factor dimensions and norm, relative covariance residual, first trajectory’s spin-resolved diagonal occupations and interaction estimator, and the final complex ensemble mean before real-part rounding.

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

compute_hubbard_drift

Goal
----
Compute the hopping drift for the two spin-resolved phase-space matrices.

```python
def compute_hubbard_drift(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    hopping: "np.ndarray",
    hbar: float = 1.0,
) -> "np.ndarray":
    """Compute the spin-resolved hopping drift.

    Parameters
    ----------
    n_up : np.ndarray
        Complex square spin-up phase-space matrix with shape ``(n_sites, n_sites)``.
    n_down : np.ndarray
        Complex square spin-down phase-space matrix with the same shape as ``n_up``.
    hopping : np.ndarray
        Finite square hopping matrix with shape ``(n_sites, n_sites)``.
    hbar : float, default=1.0
        Positive finite reduced Planck constant in the chosen units.

    Returns
    -------
    drift : np.ndarray
        Complex array with shape ``(2, n_sites, n_sites)``; index 0 is spin-up and
        index 1 is spin-down.

    Raises
    ------
    ValueError
        If the matrices are not finite compatible square arrays or ``hbar`` is not
        a positive finite real scalar.
    """
    return drift
```

### Step 2

compute_opposite_spin_diffusion

Goal
----
Compute the opposite-spin diffusion block generated by onsite interactions.

```python
def compute_opposite_spin_diffusion(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    interaction: float,
) -> "np.ndarray":
    """Compute the spin-up to spin-down diffusion block.

    Parameters
    ----------
    n_up : np.ndarray
        Finite complex square spin-up phase-space matrix.
    n_down : np.ndarray
        Finite complex square spin-down phase-space matrix with the same shape.
    interaction : float
        Finite real onsite interaction strength.

    Returns
    -------
    diffusion_block : np.ndarray
        Complex array with shape ``(n_sites**2, n_sites**2)``. Row and column
        pair indices use row-major ordering ``i * n_sites + j``.

    Raises
    ------
    ValueError
        If the spin matrices are incompatible or non-finite, or if ``interaction``
        is not a finite real scalar.
    """
    return diffusion_block
```

### Step 3

compute_randomized_svd_factors

Goal
----
Compute deterministic low-rank singular factors of the opposite-spin diffusion block.

```python
def compute_randomized_svd_factors(
    diffusion_block: "np.ndarray",
    rank: int,
    seed: int,
    tolerance_factor: float = 64.0,
) -> "np.ndarray":
    """Compute packed low-rank factors for the numerical diffusion gauge.

    Parameters
    ----------
    diffusion_block : np.ndarray
        Finite complex square opposite-spin block with shape ``(m, m)``.
    rank : int
        Sketch width satisfying ``1 <= rank <= m``.
    seed : int
        Seed used by ``numpy.random.default_rng`` for a real standard-normal
        sketch of shape ``(m, rank)``.
    tolerance_factor : float, default=64.0
        Positive multiplier in the singular-value cutoff
        ``tolerance_factor * eps * m * s_max``.

    Returns
    -------
    factors : np.ndarray
        Complex array with shape ``(2*m + 1, k)``. The first ``m`` rows are
        left singular vectors, row ``m`` stores singular values, and the final
        ``m`` rows are right singular vectors. The phase pivot is the smallest
        index within ``1e-10`` relative of each left-vector magnitude maximum.

    Raises
    ------
    ValueError
        If the block, rank, seed, or tolerance is outside the documented domain.
    """
    return factors
```

### Step 4

construct_diffusion_gauge

Goal
----
Construct the structured complex noise factor from packed singular vectors.

```python
def construct_diffusion_gauge(factors: "np.ndarray") -> "np.ndarray":
    """Construct the structured numerical diffusion gauge.

    Parameters
    ----------
    factors : np.ndarray
        Packed complex factor array with shape ``(2*m + 1, k)``: left singular
        vectors, one singular-value row, then right singular vectors.

    Returns
    -------
    gauge : np.ndarray
        Complex noise matrix with shape ``(2*m, 2*k)``.

    Raises
    ------
    ValueError
        If the packed array has an invalid shape, non-finite values, or singular
        values that are not finite nonnegative reals.
    """
    return gauge
```

### Step 5

compute_factorization_residual

Goal
----
Measure how accurately the structured noise factor reproduces the diffusion tensor.

```python
def compute_factorization_residual(
    diffusion_block: "np.ndarray",
    gauge: "np.ndarray",
) -> float:
    """Compute the relative residual of the complex diffusion factorization.

    Parameters
    ----------
    diffusion_block : np.ndarray
        Finite complex square opposite-spin block with shape ``(m, m)``.
    gauge : np.ndarray
        Finite complex noise matrix with ``2*m`` rows.

    Returns
    -------
    residual : float
        Nonnegative native Python float. For a zero target diffusion tensor, the
        absolute Frobenius residual is returned.

    Raises
    ------
    ValueError
        If input shapes or values are incompatible.
    """
    return residual
```

### Step 6

propagate_phase_space_ensemble

Goal
----
Advance a deterministic ensemble of complex phase-space trajectories by one stochastic step.

```python
def propagate_phase_space_ensemble(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    drift: "np.ndarray",
    gauge: "np.ndarray",
    dt: float,
    n_trajectories: int,
    seed: int,
) -> "np.ndarray":
    """Advance the spin-resolved phase-space variables by one Euler–Maruyama step.

    Parameters
    ----------
    n_up : np.ndarray
        Finite complex square spin-up matrix.
    n_down : np.ndarray
        Finite complex square spin-down matrix with the same shape.
    drift : np.ndarray
        Finite complex drift with shape ``(2, n_sites, n_sites)``.
    gauge : np.ndarray
        Finite complex noise matrix with ``2*n_sites**2`` rows.
    dt : float
        Positive finite step size.
    n_trajectories : int
        Positive number of independent trajectory columns.
    seed : int
        Seed for channel-major real standard-normal draws from
        ``numpy.random.default_rng``.

    Returns
    -------
    trajectories : np.ndarray
        Complex array with shape ``(2*n_sites**2, n_trajectories)`` using
        row-major spin-up variables followed by row-major spin-down variables.

    Raises
    ------
    ValueError
        If shapes, values, or scalar controls are outside the documented domain.
    """
    return trajectories
```

### Step 7

compute_interaction_energy_density

Goal
----
Compute the sampled onsite interaction-energy density from trajectory matrices.

```python
def compute_interaction_energy_density(
    trajectories: "np.ndarray",
    interaction: float,
    n_sites: int,
) -> float:
    """Compute the real ensemble-averaged onsite interaction-energy density.

    Parameters
    ----------
    trajectories : np.ndarray
        Finite complex array with shape ``(2*n_sites**2, n_trajectories)``.
    interaction : float
        Finite real onsite interaction strength.
    n_sites : int
        Positive number of lattice sites.

    Returns
    -------
    energy_density : float
        Native Python float equal to the real part of the trajectory-averaged
        onsite interaction energy divided by ``n_sites``.

    Raises
    ------
    ValueError
        If dimensions or scalar inputs are outside the documented domain.
    """
    return energy_density
```

### Step 8

run_diffusion_gauge_benchmark

Goal
----
Run the complete low-rank diffusion-gauge trajectory benchmark and return one scalar.

```python
def run_diffusion_gauge_benchmark(
    n_up: "np.ndarray",
    n_down: "np.ndarray",
    hopping: "np.ndarray",
    interaction: float,
    hbar: float,
    rank: int,
    sketch_seed: int,
    tolerance_factor: float,
    dt: float,
    n_trajectories: int,
    noise_seed: int,
    residual_tolerance: float = 1.0e-10,
    decimals: int = 12,
) -> float:
    """Run the complete deterministic diffusion-gauge benchmark.

    Parameters
    ----------
    n_up, n_down : np.ndarray
        Compatible finite complex spin-resolved phase-space matrices.
    hopping : np.ndarray
        Compatible finite hopping matrix.
    interaction : float
        Finite real onsite interaction strength.
    hbar : float
        Positive finite reduced Planck constant.
    rank : int
        Randomized sketch width.
    sketch_seed : int
        Seed for the range-finder sketch.
    tolerance_factor : float
        Positive singular-value cutoff multiplier.
    dt : float
        Positive Euler–Maruyama step size.
    n_trajectories : int
        Positive number of trajectory columns.
    noise_seed : int
        Seed for channel-major real Wiener draws.
    residual_tolerance : float, default=1e-10
        Positive maximum permitted relative diffusion-factor residual.
    decimals : int, default=12
        Number of decimal places for the single final rounding operation.

    Returns
    -------
    result : float
        Rounded real interaction-energy density as a native Python float.

    Raises
    ------
    ValueError
        If any upstream contract fails, the residual exceeds its tolerance, or
        final rounding controls are invalid.
    """
    return result
```

# Chemistry-Quantum_Chemistry-1

## Background

Auxiliary-field quantum Monte Carlo (AFQMC) estimates many-electron ground-state properties by projecting an ensemble of Slater determinants in imaginary time. The two-electron interaction is represented through auxiliary fields, while low-rank factorizations reduce storage and the cost of applying the corresponding propagators.

Importance sampling uses a trial determinant to guide the stochastic evolution and control variance. Because the propagated overlaps and weights can be complex, practical phaseless algorithms impose an overlap-based constraint and combine the surviving walkers through a weighted mixed estimator.

This task isolates one deterministic propagation block so that its factorization, basis conventions, importance shift, walker update, phase constraint, and energy reduction can be checked independently. The precise ITHC conventions needed for the scientific interpretation must be recovered from the pinned paper and implementation sources.

## Problem

Audit one deterministic phaseless auxiliary-field projection block for a single-spin Slater ensemble whose two-electron interaction is supplied in an enlarged isometric factorization; execute the following NumPy fixture exactly once, because its sole PCG64 stream fixes every numerical input.

```python
import numpy as np

rng = np.random.default_rng(26091500)
n_orb, n_aux, n_e, n_w = 7, 12, 3, 11
dt = 0.08

q, r = np.linalg.qr(rng.normal(size=(n_aux, n_orb)))
q = q * np.where(np.diag(r) >= 0.0, 1.0, -1.0)
u = q.T

channels = rng.normal(scale=0.48, size=(n_aux, n_aux)) / np.sqrt(n_aux)
W = channels.T @ channels

raw_h = rng.normal(scale=0.18, size=(n_orb, n_orb))
h = 0.5 * (raw_h + raw_h.T) - 0.55 * np.eye(n_orb)
eps, C = np.linalg.eigh(h)
half = (C * np.exp(-0.5 * dt * eps)) @ C.T

trial, rt = np.linalg.qr(rng.normal(size=(n_orb, n_e)))
trial = trial * np.where(np.diag(rt) >= 0.0, 1.0, -1.0)

walkers = []
for _ in range(n_w):
    z = trial + 0.11 * rng.normal(size=trial.shape)
    z = z + 0.07j * rng.normal(size=trial.shape)
    walker, rw = np.linalg.qr(z)
    d = np.diag(rw)
    phase = np.where(np.abs(d) > 0.0, d / np.abs(d), 1.0)
    walkers.append(walker * phase)
walkers = np.asarray(walkers)

walker_weights = 0.4 + rng.random(n_w)
auxiliary_fields = rng.normal(scale=0.7, size=(n_w, n_aux))
eri_reference = np.einsum(
    "pa,qa,ab,rb,sb->pqrs", u, u, W, u, u, optimize=True
)
eri_tolerance = 1.0e-11
```

Use the primary article, its enlarged-basis construction, and the pinned ITHC implementation listed under Browsing Sources to evaluate the source-defined projected phaseless mixed estimator for the supplied ensemble. Here `channels` is the real factor stored through `W = channels.T @ channels`; recover its associated auxiliary-field representation and the consequential Green-function, projection, importance and local-energy conventions from those sources. Treat the supplied `half`, walkers, weights and auxiliary fields as the complete state of a single-spin, one-block specialization, and use the article's stateless overlap-ratio importance update for that block. Reject the fixture if its isometry or ERI certificate fails.

Return the normalized post-propagation mixed energy and, in `<reasoning>`, cite the source evidence that fixes the extended-space projection, Green/overlap orientation, force-bias contraction, phaseless cosine projection, and direct-minus-exchange local-energy contraction. Report only the following compact end-to-end audit quantities: the maximum row-isometry residual, the maximum ERI-certificate residual, the first walker's pre-two-body extended-Green trace, the updated-weight denominator

\[
D=\sum_i w'_i,
\]

the unnormalized real-energy numerator

\[
N=\sum_i w'_i\,\operatorname{Re}(E_i),
\]

the effective sample size

\[
N_{\mathrm{eff}}=D^2/\sum_i (w'_i)^2,
\]

and the eleven leave-one-walker-out estimates, in supplied walker order,

\[
E_{(-i)}=\frac{N-w'_i\operatorname{Re}(E_i)}{D-w'_i}.
\]

Finally identify the one-based walker index with the largest absolute influence \(|E_{(-i)}-N/D|\) and report that magnitude. Report all real scalars to at least twelve decimal places and the trace as a `(real, imag)` pair. Do not report the occupation, force-bias, multiplier, local-energy, or normalized-weight vectors: they are internal working quantities, not requested outputs.

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

reconstruct_ithc_eri

Goal
----
Recover a physical ERI tensor from an isometric THC representation.

```python
def reconstruct_ithc_eri(
    isometry: 'np.ndarray',
    kernel: 'np.ndarray',
) -> 'np.ndarray':
    """Reconstruct the physical four-index ERI represented by ITHC.

    Parameters
    ----------
    isometry
        Real row-orthonormal array with shape
        ``(n_orbitals, n_auxiliary)``.
    kernel
        Real symmetric positive-semidefinite array with shape
        ``(n_auxiliary, n_auxiliary)``.

    Notes
    -----
    Return the tensor in ``(p, q, r, s)`` order using
    ``V[p,q,r,s] = sum_ab u[p,a] u[q,a] W[a,b] u[r,b] u[s,b]``.

    Returns
    -------
    np.ndarray
        Reconstructed ERI tensor with shape ``(n, n, n, n)``.

    Raises
    ------
    ValueError
        If the inputs have incompatible shapes, are nonfinite, or violate the
        stated isometry and kernel conditions.
    """
    return result
```

### Step 2

extended_mixed_green

Goal
----
Build the mixed one-body Green matrix in the ITHC enlarged basis.

```python
def extended_mixed_green(
    trial: 'np.ndarray',
    walker: 'np.ndarray',
    isometry: 'np.ndarray',
) -> 'np.ndarray':
    """Evaluate the trial-walker mixed Green matrix in the ITHC space.

    Parameters
    ----------
    trial, walker
        Matching nonempty Slater matrices with shape
        ``(n_orbitals, n_electrons)``.
    isometry
        Real row-orthonormal array with shape
        ``(n_orbitals, n_auxiliary)``.

    Notes
    -----
    Set ``A_t = u.T @ trial`` and ``B_t = u.T @ walker``. With
    ``O = B_t.T @ A_t.conj()``, return
    ``A_t.conj() @ solve(O, B_t.T)``. The walker uses an ordinary transpose,
    not a Hermitian transpose.

    Returns
    -------
    np.ndarray
        Complex mixed Green matrix with shape
        ``(n_auxiliary, n_auxiliary)``.

    Raises
    ------
    ValueError
        If shapes, finiteness, row orthonormality, or overlap conditioning
        are invalid.
    """
    return result
```

### Step 3

ithc_force_bias

Goal
----
Evaluate the enlarged-basis importance-sampling force bias.

```python
def ithc_force_bias(
    extended_green: 'np.ndarray',
    channels: 'np.ndarray',
    time_step: float,
) -> 'np.ndarray':
    """Form the importance-sampling force bias in the extended basis.

    Parameters
    ----------
    extended_green
        Square mixed Green matrix in the auxiliary basis.
    channels
        Channel factor with shape ``(n_fields, n_auxiliary)``.
    time_step
        Finite nonnegative propagation interval.

    Notes
    -----
    The supplied real channels factor the positive interaction kernel. Define
    the imaginary auxiliary-field operators as ``v = 1j * channels`` and
    return ``-sqrt(time_step) * v @ diag(extended_green)``. Only diagonal
    occupations of ``extended_green`` enter the contraction.

    Returns
    -------
    np.ndarray
        Complex force-bias vector with one entry per auxiliary field.

    Raises
    ------
    ValueError
        If shapes, finiteness, or the time-step domain are invalid.
    """
    return result
```

### Step 4

propagate_ithc_walker

Goal
----
Apply the diagonal ITHC field and finish a projected walker step.

```python
def propagate_ithc_walker(
    half_propagated_walker: 'np.ndarray',
    half_one_body: 'np.ndarray',
    isometry: 'np.ndarray',
    channels: 'np.ndarray',
    auxiliary_field: 'np.ndarray',
    force_bias: 'np.ndarray',
    time_step: float,
) -> 'np.ndarray':
    """Complete one ITHC walker step after its first one-body half-step.

    Parameters
    ----------
    half_propagated_walker
        Walker after the first one-body half-step.
    half_one_body
        Physical-basis one-body half-step propagator.
    isometry
        Real row-orthonormal physical-to-auxiliary map.
    channels
        Channel factor with shape ``(n_fields, n_auxiliary)``.
    auxiliary_field, force_bias
        Matching sampled-field and importance-shift vectors.
    time_step
        Finite nonnegative propagation interval.

    Notes
    -----
    Form ``B_t = u.T @ half_propagated_walker`` and
    ``z = sqrt(complex(-time_step)) * ((auxiliary_field-force_bias) @ channels)``
    using the principal complex square root. Return
    ``half_one_body @ (u @ (exp(z)[:,None] * B_t))``.

    Returns
    -------
    np.ndarray
        Propagated physical-basis Slater matrix.

    Raises
    ------
    ValueError
        If shapes, finiteness, row orthonormality, or the time-step domain
        are invalid, or if propagation exceeds the floating-point range.
    """
    return result
```

### Step 5

phaseless_importance

Goal
----
Convert a complex walker overlap change into a phaseless multiplier.

```python
def phaseless_importance(
    trial: 'np.ndarray',
    old_walker: 'np.ndarray',
    new_walker: 'np.ndarray',
    auxiliary_field: 'np.ndarray',
    force_bias: 'np.ndarray',
) -> float:
    """Evaluate the phaseless importance multiplier for one walker step.

    Parameters
    ----------
    trial
        Trial Slater determinant.
    old_walker, new_walker
        Physical walker before and after the step.
    auxiliary_field, force_bias
        Matching sampled-field and importance-shift vectors.

    Notes
    -----
    Let
    ``S = det(trial.conj().T @ new_walker) /
    det(trial.conj().T @ old_walker)``
    and
    ``F = dot(x, xbar) - 0.5*dot(xbar, xbar)``,
    where both dots are unconjugated. Return
    ``abs(S*exp(F)) * max(0, cos(angle(S)))``.

    Returns
    -------
    float
        Nonnegative phaseless importance multiplier.

    Raises
    ------
    ValueError
        If shapes or finiteness are invalid, the old overlap is zero, or
        the importance magnitude exceeds the floating-point range.
    """
    return result
```

### Step 6

ithc_local_energy

Goal
----
Contract physical and enlarged Green matrices into an ITHC local energy.

```python
def ithc_local_energy(
    one_body: 'np.ndarray',
    kernel: 'np.ndarray',
    physical_green: 'np.ndarray',
    extended_green: 'np.ndarray',
) -> complex:
    """Evaluate the mixed local energy with the ITHC interaction.

    Parameters
    ----------
    one_body
        Hermitian physical-basis one-body matrix.
    kernel
        Real symmetric auxiliary-basis interaction kernel.
    physical_green
        Physical-basis mixed Green matrix.
    extended_green
        Auxiliary-basis mixed Green matrix.

    Notes
    -----
    Use
    ``E1 = sum_pq one_body[p,q]*physical_green[p,q]``.
    For ``n = diag(extended_green)``, use
    ``E2 = 0.5*sum_ab kernel[a,b]*(n[a]*n[b]
    - extended_green[a,b]*extended_green[b,a])``
    and return ``E1 + E2``.

    Returns
    -------
    complex
        Mixed local energy.

    Raises
    ------
    ValueError
        If shapes, finiteness, Hermiticity, or kernel symmetry are invalid.
    """
    return result
```

### Step 7

phaseless_mixed_energy

Goal
----
Reduce weighted phaseless walker energies to a real mixed estimate.

```python
def phaseless_mixed_energy(
    local_energies: 'np.ndarray',
    walker_weights: 'np.ndarray',
) -> float:
    """Reduce phaseless walker data to the normalized mixed energy.

    Parameters
    ----------
    local_energies
        Complex local energies, one per walker.
    walker_weights
        Matching nonnegative post-step weights with positive total weight.

    Notes
    -----
    Normalize the weights by their sum and contract them with the real parts
    of the local energies.

    Returns
    -------
    float
        Normalized real mixed energy.

    Raises
    ------
    ValueError
        If the vectors are incompatible or nonfinite, or the weights are
        invalid.
    """
    return result
```

### Step 8

ithc_afqmc_projected_energy

Goal
----
Orchestrate one certified deterministic ITHC-AFQMC projection block.

```python
def ithc_afqmc_projected_energy(
    one_body: 'np.ndarray',
    half_one_body: 'np.ndarray',
    isometry: 'np.ndarray',
    kernel: 'np.ndarray',
    channels: 'np.ndarray',
    eri_reference: 'np.ndarray',
    trial: 'np.ndarray',
    walkers: 'np.ndarray',
    walker_weights: 'np.ndarray',
    auxiliary_fields: 'np.ndarray',
    time_step: float,
    eri_tolerance: float,
) -> float:
    """Run one deterministic phaseless ITHC-AFQMC block.

    Parameters
    ----------
    one_body, half_one_body
        Physical-basis one-body Hamiltonian and its half-step propagator.
    isometry, kernel, channels
        ITHC isometry, interaction kernel, and channel factor.
    eri_reference
        Physical four-index ERI used to certify the supplied ITHC factors.
    trial
        Trial Slater determinant.
    walkers
        Batch of physical-basis walker determinants.
    walker_weights
        Nonnegative initial walker weights.
    auxiliary_fields
        One sampled auxiliary-field vector per walker.
    time_step
        Finite nonnegative propagation interval.
    eri_tolerance
        Finite nonnegative relative ERI-certificate tolerance.

    Notes
    -----
    Certify the ITHC factors first. Then call all seven preceding public
    functions to propagate every supplied walker, update its weight, evaluate
    its local energy, and form the normalized mixed estimate. All inputs are
    required; the function has no hidden built-in fixture mode.

    Returns
    -------
    float
        Normalized post-propagation mixed energy.

    Raises
    ------
    ValueError
        If inputs are missing, incompatible, nonfinite, or fail the ITHC
        certificate.
    """
    return result
```

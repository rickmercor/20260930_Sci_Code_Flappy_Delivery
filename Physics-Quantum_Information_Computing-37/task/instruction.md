# Physics-Quantum_Information_Computing-37

## Background

An adaptive device can receive classical stimuli and emit classical actions while storing its predictive memory in nonorthogonal quantum states. Low memory entropy does not by itself reduce the Hilbert-space dimension needed to implement that device. A reference input process organizes the device's typical temporal histories into a stationary compression target, but a correlated reference carries its own memory that must remain external to the device. Direct truncation can destroy probability conservation, so the reduced dynamics must be repaired into a valid instrument for each possible stimulus. A comparison of purified histories quantifies reference-driven distortion, while a forced observation record probes how the repaired device actually predicts and updates its memory away from an average-case description.

## Problem

Quantify how a finite quantum clock's conditional prediction changes when its memory is compressed while remaining a valid controlled quantum instrument. Use $N=6$ zero-based ages with nonorthogonal memory states $|s_j\rangle=(N-j)^{-1/2}\sum_{k=j}^{N-1}|k\rangle$ in a fixed orthonormal computational basis. With $y=1$ denoting a tick, the controlled isometries are $V_0|s_j\rangle=\sqrt{(N-j-1)/(N-j)}|s_{j+1}\rangle|0\rangle_Y|0\rangle_E+(N-j)^{-1/2}|s_0\rangle|1\rangle_Y|0\rangle_E$ and $V_1|s_j\rangle=|s_0\rangle|0\rangle_Y|s_j\rangle_E$, where the first term of $V_0$ is absent at $j=N-1$, there is no periodic wrap, and $\eta=0,\ldots,5$ labels the computational environment basis.

The passive reference generator has $R_{cx c^{\prime}}=p(x,c^{\prime}\mid c)$ given by $R=[[[0.86,0.08],[0.01,0.05]],[[0.10,0.51],[0.19,0.20]]]$, in old-source, stimulus, new-source order, and the original joint source–memory state is its unique stationary state under this same-timestep routing. Choose the smallest dimension $r\in\{2,3,4,5,6\}$ whose agent-local maximal-retained-weight truncation, followed by a separate nearest-isometry polar repair for each stimulus, has common-label purified-history fidelity-divergence certificate at most $\delta=0.007$ bits per timestep under $R$, rather than using discarded weight as the certificate. For each projected update use squared singular values strictly greater than $10^{-12}$ as the polar support, complete any missing retained support with its orthogonal projector as one extra Kraus operator at $(y,\eta)=(0,6)$, zero-pad the original on that label, retain the fixed environment gauge, and treat full retention as exactly zero certificate; every visited interior memory-eigenvalue cut is required to have gap greater than $10^{-10}$.

The forced record is $x=(0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0)$ and $y=(0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1)$. Start the original filter in its stationary memory marginal $\rho_M$ and the reduced filter in $P\rho_M P/\operatorname{Tr}(P\rho_M)$ for the selected projector $P$, condition on actions sequentially with environment outcomes unobserved, and do not attach reference-source probabilities to these externally forced stimuli. Return $\ell=\ln p_{\mathrm{reduced}}(y\mid x)-\ln p_{\mathrm{original}}(y\mid x)$ in nats rounded to ten decimal places, using binary64 or higher precision with no intermediate rounding or random draws; in the reasoning justify the compression and certificate identities and the certificate's scope, and report the leading original memory eigenvalue, selected retained weight, smallest supported evolve-Gram eigenvalue, decisive adjacent-dimension certificate values, selected dimension, and the two conditional log probabilities.

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

build_clock_instrument

Goal
----
Construct the controlled instrument of a finite quantum renewal clock in the specified tail-state gauge.

```python
from numbers import Integral

import numpy as np

def build_clock_instrument(n: int) -> np.ndarray:
    """Construct the controlled renewal-clock Kraus operators.

    Parameters
    ----------
    n : int
        Number of ages and memory dimension, at least 2. Boolean values are
        not integers for this contract. All indices start at zero.

    Returns
    -------
    kraus : np.ndarray
        Complex array of shape (2, 2, n, n, n), indexed by stimulus, action,
        environment, outgoing memory, incoming memory. Action 1 is a tick.
        The memory and environment gauges are the tail states described above.

    Raises
    ------
    ValueError
        If n is not an integer at least 2.
    """
    return np.zeros((2, 2, n, n, n), dtype=complex)
```

### Step 2

solve_routed_stationary

Goal
----
Find the stationary quantum-memory blocks conditioned on the state of a correlated classical input generator.

```python
import numpy as np

def solve_routed_stationary(kraus: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Solve the unique stationary state of the classically routed channel.

    Parameters
    ----------
    kraus : np.ndarray
        Finite complex array (X, Y, E, d, d) with nonempty axes. For each
        stimulus, the sum of K.conj().T @ K over actions and environment
        outcomes equals the identity within absolute tolerance 1e-10.
    reference : np.ndarray
        Finite real array (C, X, C), C >= 1, indexed by old source state,
        stimulus, new source state. Entries are nonnegative; each old-state
        slice sums to one within absolute tolerance 1e-12.

    Returns
    -------
    blocks : np.ndarray
        Complex array (C, d, d) of positive semidefinite stationary blocks,
        jointly normalized to total trace one. Inputs are not mutated.

    Raises
    ------
    ValueError
        If shapes, finiteness, stochastic normalization or Kraus completeness
        fail, or the trace-one stationary solution is not unique. The augmented
        stationary linear system must have full column rank with relative singular-value cutoff 1e-12.
    """
    return np.zeros((reference.shape[0], kraus.shape[-1], kraus.shape[-1]), dtype=complex)
```

### Step 3

select_memory_subspace

Goal
----
Return the gauge-independent projector that maximizes the retained stationary agent-memory weight at a fixed rank.

```python
from numbers import Integral

import numpy as np

def select_memory_subspace(blocks: np.ndarray, rank: int) -> np.ndarray:
    """Select a unique maximal-weight memory subspace.

    Parameters
    ----------
    blocks : np.ndarray
        Finite complex array (C, d, d), C,d >= 1, of Hermitian positive
        semidefinite source-conditioned blocks with total trace one.
        Hermiticity, positivity and normalization use absolute tolerance 1e-10.
    rank : int
        Retained memory dimension in [1,d], excluding booleans. At an interior
        spectral cut the retained/discarded eigenvalue gap must exceed 1e-10.
        Full retention is defined even for degenerate spectra.

    Returns
    -------
    projector : np.ndarray
        Complex (d,d) orthogonal projector. Inputs are not mutated.

    Raises
    ------
    ValueError
        If a density-block condition fails, rank is invalid, or an interior
        eigenvalue cut has gap at most 1e-10.
    """
    return np.zeros((blocks.shape[-1], blocks.shape[-1]), dtype=complex)
```

### Step 4

repair_controlled_instrument

Goal
----
Convert the projected controlled dynamics into a valid reduced quantum instrument by stimulus-wise polar completion.

```python
from numbers import Real

import numpy as np

def repair_controlled_instrument(kraus: np.ndarray, projector: np.ndarray, support_tol: float = 1e-12) -> np.ndarray:
    """Repair the projected instrument in the original ambient memory basis.

    Parameters
    ----------
    kraus : np.ndarray
        Finite complex array (X,Y,E,d,d) defining a trace-preserving instrument
        for each stimulus, within absolute tolerance 1e-10.
    projector : np.ndarray
        Nonzero Hermitian orthogonal projector (d,d), checked at absolute
        tolerance 1e-10. Retained and discarded directions may be non-coordinate.
    support_tol : float
        Finite real number in (0,1), excluding booleans. Gram eigenvalues
        strictly greater than this absolute cutoff define the polar support.

    Returns
    -------
    repaired : np.ndarray
        Complex array (X,Y,E+1,d,d). Existing environment labels keep their
        order; the extra label E, action 0, contains the projector onto the
        missing retained support. Each stimulus is complete on projector.
        No input is mutated. Full retention returns the original operators
        with one zero-padded environment label.

    Raises
    ------
    ValueError
        If shapes or finiteness fail, the instrument is not trace preserving,
        projector is not a nonzero orthogonal projector, or support_tol is invalid.
    """
    return np.zeros((kraus.shape[0], kraus.shape[1], kraus.shape[2]+1, kraus.shape[-1], kraus.shape[-1]), dtype=complex)
```

### Step 5

compute_history_rate

Goal
----
Evaluate the purified-history divergence rate from the common-label mixed transfer map under the correlated reference.

```python
import numpy as np

def compute_history_rate(kraus: np.ndarray, repaired: np.ndarray, reference: np.ndarray) -> float:
    """Compute the common-dilation purified-history divergence certificate.

    Parameters
    ----------
    kraus : np.ndarray
        Finite complex original instrument (X,Y,E,d,d), complete on identity.
    repaired : np.ndarray
        Finite complex instrument (X,Y,E+1,d,d), complete on one common
        nonzero orthogonal support P and satisfying B=P@B@P. Label E is
        compared with a zero operator in the original dilation. Completeness
        and support are checked at absolute tolerance 1e-10.
    reference : np.ndarray
        Real nonnegative array (C,X,C) of old-source/stimulus/new-source
        probabilities. Each old-state slice sums to one within 1e-12.

    Returns
    -------
    rate : float
        Native finite Python float, in bits per timestep, obtained from the
        common-label purified-history mixed transfer spectral radius. The
        original and reduced histories use identical source labels. A value
        of the spectral radius above one by at most 1e-9 is clipped to one.
        Identical zero-padded instruments have rate exactly zero.

    Raises
    ------
    ValueError
        If shapes, finiteness, normalization, completeness or support fail,
        or the mixed spectral radius is zero, nonfinite, or greater than
        1+1e-9. The supplied histories must have nonzero asymptotic overlap.
    """
    return 0.0
```

### Step 6

compute_record_loglikelihood

Goal
----
Condition sequentially on a supplied stimulus/action record and accumulate its finite log probability.

```python
import math

import numpy as np

def compute_record_loglikelihood(kraus: np.ndarray, initial: np.ndarray, stimuli: np.ndarray, actions: np.ndarray) -> float:
    """Evaluate a forced-stimulus record by sequential quantum filtering.

    Parameters
    ----------
    kraus : np.ndarray
        Finite complex array (X,Y,E,d,d), complete on one common nonzero
        orthogonal support P. Each operator has B=P@B@P. Identity support is
        allowed. Completeness and support tolerance is 1e-10.
    initial : np.ndarray
        Finite Hermitian positive semidefinite (d,d) density matrix of trace
        one, supported on P. These checks use absolute tolerance 1e-10.
    stimuli : np.ndarray
        One-dimensional integer array, length T, with entries in [0,X).
        Booleans are excluded. Empty records are allowed.
    actions : np.ndarray
        One-dimensional integer array, length T, with entries in [0,Y).
        Booleans are excluded. Environment outcomes are unobserved.

    Returns
    -------
    log_probability : float
        Native finite Python float in nats. The empty-record value is zero.
        Posteriors are normalized sequentially; no input array is mutated.
        The result must remain finite for long records with positive
        one-step probabilities even when their product underflows.

    Raises
    ------
    ValueError
        If shapes, finiteness, instrument support, density-matrix conditions
        or integer record ranges fail, lengths differ, or any observed action
        has zero conditional probability.
    """
    return 0.0
```

### Step 7

evaluate_compression_logratio

Goal
----
Use the six preceding public operations to select a certified retained dimension and evaluate the reduced-to-original log-likelihood ratio.

```python
from numbers import Real

import numpy as np

def evaluate_compression_logratio(
    n: int = 6,
    reference: np.ndarray = (((.86,.08),(.01,.05)),((.10,.51),(.19,.20))),
    rate_limit: float = .007,
    stimuli: np.ndarray = (0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0),
    actions: np.ndarray = (0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1)
) -> float:
    """Select a certified clock-memory rank and compare record probabilities.

    Parameters
    ----------
    n : int
        Clock size, at least 2; the default is the six-age prompt instance.
    reference : np.ndarray
        Stochastic array (C,2,C) in old-source/stimulus/new-source order.
        The default is the two-state correlated source in the problem.
    rate_limit : float
        Nonnegative finite real certificate budget in bits per timestep,
        excluding booleans. The default is 0.007. Accept equality.
    stimuli : np.ndarray
        One-dimensional integer stimulus record in {0,1}; the default is
        the 24-observation prompt record. Inputs are externally forced.
    actions : np.ndarray
        Matching integer action record in {0,1}; the default is the prompt
        record. Empty paired records are allowed.

    Returns
    -------
    log_ratio : float
        Unrounded finite native Python float in nats: reduced log likelihood
        minus original log likelihood. Select the first rank in 2,...,n whose
        purified-history rate is at most rate_limit. The initial original
        memory is its reference-stationary marginal; the initial reduced
        state is its normalized projection, not the repaired stationary state.
        Use support cutoff 1e-12 and spectral-cut gap 1e-10. Do not mutate inputs.

    Raises
    ------
    ValueError
        If n or rate_limit is invalid; the reference/instrument lacks a unique
        stationary state; a visited interior spectral cut is ambiguous; record
        shapes, ranges or density conditions fail; a conditional probability
        vanishes; or no rank has a defined admissible certificate.
    """
    return 0.0
```

# Biology-Biochemistry-22

## Background

Protein directed evolution aims to identify amino-acid substitutions that improve properties such as catalytic activity, binding affinity, stability, or expression. Exhaustively testing all variants is generally impossible because the sequence space grows exponentially with protein length, while experimental fitness measurements are expensive and available for only a small fraction of possible sequences. The problem is further complicated by epistasis, in which the effect of one mutation depends on the identities of other residues, producing a rugged fitness landscape containing multiple local optima.

Machine-learning-guided directed evolution addresses this limitation by fitting a surrogate model to the currently measured sequence–fitness pairs and using an acquisition strategy to select promising variants for the next experimental round. Sequence-only models may produce noisy or biologically implausible fitness landscapes because they do not explicitly account for the relationship between amino-acid changes and protein conformation. The source study therefore uses predicted structural perturbations between mutants and the wild-type protein as prior information during surrogate learning, encouraging the learned representation to capture smoother and more biologically meaningful sequence–structure–function relationships.

The study’s acquisition strategy represents discrete amino-acid sequences within a continuous state space so that gradient-based Hamiltonian dynamics can explore the surrogate fitness landscape. Momentum enables proposals to travel beyond nearby variants, while reflective virtual barriers constrain the relaxed coordinates to their valid range. Continuous states are subsequently converted back into discrete amino-acid identities, and a Metropolis correction reduces errors introduced by this discretization. Predictions from an ensemble of surrogate models additionally provide an uncertainty estimate, allowing an upper-confidence-bound criterion to balance exploitation of high predicted fitness with exploration of uncertain regions.

Together, these components form a structure-informed Bayesian optimization framework for proposing diverse, high-fitness protein variants under a limited experimental-query budget. The computational example isolates the acquisition component of this framework and treats the differentiable surrogate ensemble as already trained, allowing the sequence-relaxation, Hamiltonian sampling, discretization, acceptance, and uncertainty-aware selection principles to be evaluated without requiring neural-network training or external protein-structure prediction.

## Problem

Structure-aware protein optimization can be framed as acquisition over discrete protein sequences by relaxing one-hot sequence coordinates and evolving them with Hamiltonian dynamics. This task uses a deterministic, reduced five-position and four-class benchmark derived from the cited source. The surrogate models and all numerical inputs are supplied; no model training, structure prediction, wet-lab evaluation, or external dataset is required.

Let the initial zero-based residue-class sequence be

```python
initial_sequence = np.array([0, 1, 2, 3, 0], dtype=int)
alphabet_size = 4
```

Encode it as the one-hot reference state $q^{\mathrm{init}} \in \mathbb{R}^{L\times A}$. For ensemble member $j$, use the supplied structure-aware surrogate

$$
f_j(q)=b_j
+\sum_{r=0}^{L-1}\sum_{c=0}^{A-1}(W_j)_{r,c}q_{r,c}
+\lambda_j\,q_{0,:}^{\mathsf T}Cq_{L-1,:}
-\frac{\rho_j}{2}\sum_{r=0}^{L-1}\sum_{c=0}^{A-1}
\left(q_{r,c}-q^{\mathrm{init}}_{r,c}\right)^2.
$$

Here, the double sum is the Frobenius inner product between $W_j$ and $q$; $q_{0,:}^{\mathsf T}Cq_{L-1,:}$ is the interaction between the first and last sequence positions; $b_j$ is the bias; $\lambda_j$ is the contact scale; and $\rho_j$ is the penalty scale.

Generate the complete IEEE-754 float64 benchmark state by executing these statements once, in the displayed order:

```python
parameter_rng = np.random.default_rng(7049)

weights_ensemble = parameter_rng.normal(
    0.0,
    0.85,
    size=(5, 5, 4),
).astype(np.float64)

raw_contact = parameter_rng.normal(
    0.0,
    0.35,
    size=(4, 4),
).astype(np.float64)
contact_matrix = (raw_contact + raw_contact.T) / 2.0

biases = parameter_rng.normal(
    -0.15,
    0.30,
    size=5,
).astype(np.float64)
contact_scales = parameter_rng.uniform(
    0.35,
    0.95,
    size=5,
).astype(np.float64)
penalty_scales = parameter_rng.uniform(
    0.12,
    0.42,
    size=5,
).astype(np.float64)
seeds = parameter_rng.integers(
    1000,
    100000,
    size=5,
    dtype=np.int64,
)
epsilon = float(parameter_rng.uniform(0.48, 0.82))
trajectory_steps = 12
```

The `parameter_rng` is used only to construct the benchmark arrays and scalars. For each ensemble member $j$, initialize a separate `np.random.default_rng(seeds[j])`. Use IEEE-754 float64 arithmetic without intermediate rounding, preserve accepted candidates in first-occurrence order, and calculate ensemble UCB values using the population standard deviation. Determine from the cited source how each member's stochastic stream is consumed within the acquisition loop, including the relationship between the initial momentum draw and the update-level acceptance draws.

Apply the cited source's potential, reflective Hamiltonian update, position-discretization, Metropolis proposal-collection procedure, and ensemble UCB ranking for exactly $T$ Hamiltonian updates per ensemble member. Determine from the cited source where discretization, acceptance testing, and candidate collection occur relative to each update, and how the underlying continuous state proceeds after each acceptance decision.

As a scientific control, also evaluate an endpoint-only variant: use the same initialized momentum for each surrogate, complete all $T$ continuous Hamiltonian updates before discretizing once, and use the first uniform variate generated after the momentum draw for that single Metropolis comparison. Do not collect intermediate proposals in this control.

In `<reasoning>`, identify the source-derived potential, boundary rule, proposal-collection location, random-stream order, and continuous-state behavior after acceptance and rejection. As compact numerical evidence, report the number of distinct accepted source-style candidates; the three source-style updates with the smallest absolute difference between the Metropolis probability and its uniform draw, giving the one-based surrogate number, one-based update number, discrete proposal, probability, draw, and decision for each; the three largest accepted-candidate UCB values with their candidates in descending order and the gap between the two largest values; the accepted endpoint-only candidates with their UCB values and maximum; and the increase of the source-style maximum over the endpoint-only maximum. Do not reproduce continuous-state matrices, momentum matrices, the complete accepted-candidate list, or the complete update path. The required final scalar is the maximum accepted-candidate UCB from the source-style per-update acquisition.

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

01_encode_sequence.py

Goal
----
Encode a reduced-alphabet protein sequence as a continuous one-hot state. Each residue-class index becomes one row of a numerical matrix that can subsequently be relaxed and evolved through Hamiltonian dynamics.

```python
import numpy as np


def encode_sequence(
    sequence: np.ndarray,
    alphabet_size: int,
) -> np.ndarray:
    """Encode integer residue indices as a one-hot matrix.

    Parameters
    ----------
    sequence : np.ndarray
        One-dimensional array of zero-based residue-class indices.
    alphabet_size : int
        Number of allowed residue classes; must be positive.
    Raises
    ------
    ValueError
        If ``sequence`` is empty, not one-dimensional, contains
        non-integer or out-of-range indices, or ``alphabet_size``
        is not a positive integer.

    Returns
    -------
    one_hot : np.ndarray
        Float64 array of shape ``(len(sequence), alphabet_size)``.
    """
    return one_hot
```

### Step 2

02_surrogate_score_gradient.py

Goal
----
Evaluate one fixed structure-aware surrogate model and its analytic gradient. The surrogate combines position-specific residue scores, an interaction between the first and last sequence positions, and a quadratic penalty for movement away from the initial protein sequence.

```python
import numpy as np


def surrogate_score_gradient(
    q: np.ndarray,
    q_init: np.ndarray,
    weights: np.ndarray,
    contact_matrix: np.ndarray,
    bias: float,
    contact_scale: float,
    penalty_scale: float,
) -> tuple[float, np.ndarray]:
    """Evaluate one surrogate score and its gradient.

    Parameters
    ----------
    q : np.ndarray
        Continuous sequence state of shape ``(L, A)``, with ``L >= 2``.
    q_init : np.ndarray
        Initial one-hot state with the same shape as ``q``.
    weights : np.ndarray
        Position-specific weights with the same shape as ``q``.
    contact_matrix : np.ndarray
        Residue-contact matrix of shape ``(A, A)``.
    bias : float
        Scalar surrogate intercept.
    contact_scale : float
        Multiplier for the first-to-last position interaction.
    penalty_scale : float
        Non-negative quadratic penalty multiplier.

    Raises
    ------
    ValueError
        If the array shapes are incompatible, an input is non-finite,
        or ``penalty_scale`` is negative.

    Returns
    -------
    result : tuple[float, np.ndarray]
        Surrogate score and float64 gradient of shape ``(L, A)``.
    """
    return result
```

### Step 3

03_potential_energy_gradient.py

Goal
----
Convert a surrogate protein-fitness score into the probability-derived potential energy used by Hamiltonian dynamics, and calculate the corresponding analytic potential gradient.

```python
import numpy as np


def potential_energy_gradient(
    score: float,
    score_gradient: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Calculate the probability-derived potential and gradient.
 
    Parameters
    ----------
    score : float
        Finite scalar surrogate fitness score.
    score_gradient : np.ndarray
        Non-empty finite numerical gradient of the score with respect to
        position.
 
    Returns
    -------
    result : tuple[float, np.ndarray]
        Scalar potential energy and float64 potential-gradient array.
 
    Raises
    ------
    ValueError
        If ``score`` is not a finite scalar, or if ``score_gradient`` is
        empty or contains a non-finite value.
    """
    return result
```

### Step 4

04_reflective_leapfrog.py

Goal
----
Evolve one continuously relaxed protein sequence through leapfrog Hamiltonian dynamics. Whenever a coordinate leaves the permitted interval, apply repeated virtual-barrier reflections and reverse the corresponding momentum component.

```python
from importlib import import_module

import numpy as np


def reflective_leapfrog(
    q_init: np.ndarray,
    momentum: np.ndarray,
    weights: np.ndarray,
    contact_matrix: np.ndarray,
    bias: float,
    contact_scale: float,
    penalty_scale: float,
    epsilon: float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a reflective leapfrog trajectory for one surrogate member.
 
    Parameters
    ----------
    q_init : np.ndarray
        Initial continuous state of shape ``(L, A)`` with ``L >= 2`` and
        all entries in ``[0, 1]``.
    momentum : np.ndarray
        Finite initial momentum with the same shape as ``q_init``.
    weights : np.ndarray
        Finite surrogate weight matrix with the same shape as ``q_init``.
    contact_matrix : np.ndarray
        Finite residue-contact matrix of shape ``(A, A)``.
    bias : float
        Finite surrogate intercept.
    contact_scale : float
        Finite contact-interaction multiplier.
    penalty_scale : float
        Finite non-negative penalty relative to ``q_init``.
    epsilon : float
        Positive finite leapfrog step size.
    n_steps : int
        Positive integer number of leapfrog steps.
 
    Returns
    -------
    result : tuple[np.ndarray, np.ndarray]
        Final float64 position and momentum matrices.
 
    Raises
    ------
    ValueError
        If an array has an invalid or incompatible shape; an input contains
        a non-finite value; ``q_init`` lies outside ``[0, 1]``;
        ``penalty_scale`` is negative; ``epsilon`` is not positive and
        finite; ``n_steps`` is not a positive integer; or reflective
        correction cannot return a coordinate to the unit interval.
    """
    return result
```

### Step 5

05_discretize_positions.py

Goal
----
Convert a continuously relaxed protein-sequence state back into discrete residue-class indices. Select the largest coordinate at each sequence position and reconstruct the corresponding numerical one-hot candidate.

```python
import numpy as np


def discretize_positions(
    q: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Discretize each sequence position using sitewise argmax.

    Parameters
    ----------
    q : np.ndarray
        Finite continuous position matrix of shape ``(L, A)``.

    Raises
    ------
    ValueError
        If ``q`` is empty, not two-dimensional, or contains
        non-finite values.

    Returns
    -------
    result : tuple[np.ndarray, np.ndarray]
        Integer residue-index array of shape ``(L,)`` and float64
        one-hot matrix of shape ``(L, A)``.
    """
    return result
```

### Step 6

06_metropolis_acceptance.py

Goal
----
Calculate the discretization-aware Metropolis acceptance probability for a proposed protein sequence and return a numerical accept-or-reject decision.

```python
import numpy as np


def metropolis_acceptance(
    initial_potential: float,
    proposed_potential: float,
    initial_momentum: np.ndarray,
    final_momentum: np.ndarray,
    uniform_draw: float,
) -> tuple[float, int]:
    """Calculate one update's Metropolis probability and decision.

    Parameters
    ----------
    initial_potential : float
        Potential energy of the continuous state before the current update.
    proposed_potential : float
        Potential energy of the discretized post-update proposal.
    initial_momentum : np.ndarray
        Momentum array before the current Hamiltonian update.
    final_momentum : np.ndarray
        Momentum array after the current Hamiltonian update, with matching shape.
    uniform_draw : float
        Uniform variate for this update in the half-open interval ``[0, 1)``.

    Returns
    -------
    result : tuple[float, int]
        Acceptance probability and binary decision, where 1 means accepted.

    Raises
    ------
    ValueError
        If either potential or ``uniform_draw`` is not a finite scalar; if the
        momentum arrays have different shapes; if either momentum array is
        empty or contains non-finite values; or if ``uniform_draw`` is outside
        the interval ``[0, 1)``.
    """
    return result
```

### Step 7

07_ensemble_ucb.py

Goal
----
Evaluate every accepted discrete protein candidate using the complete surrogate ensemble and calculate its upper-confidence-bound score from the ensemble mean and population standard deviation.

```python
from importlib import import_module

import numpy as np


def ensemble_ucb(
    candidates: np.ndarray,
    q_init: np.ndarray,
    weights_ensemble: np.ndarray,
    contact_matrix: np.ndarray,
    biases: np.ndarray,
    contact_scales: np.ndarray,
    penalty_scales: np.ndarray,
) -> np.ndarray:
    """Calculate one UCB score per accepted candidate.

    Parameters
    ----------
    candidates : np.ndarray
        Accepted one-hot candidates of shape ``(K, L, A)``.
    q_init : np.ndarray
        Initial one-hot state of shape ``(L, A)``.
    weights_ensemble : np.ndarray
        Surrogate weight matrices of shape ``(M, L, A)``.
    contact_matrix : np.ndarray
        Residue-contact matrix of shape ``(A, A)``.
    biases : np.ndarray
        Surrogate biases of shape ``(M,)``.
    contact_scales : np.ndarray
        Surrogate contact multipliers of shape ``(M,)``.
    penalty_scales : np.ndarray
        Surrogate penalty multipliers of shape ``(M,)``.

    Raises
    ------
    ValueError
        If the candidate or ensemble shapes are invalid, the ensemble
        is empty, parameter-vector lengths do not match, or a supplied
        surrogate parameter is invalid.

    Returns
    -------
    ucb_scores : np.ndarray
        Float64 vector of shape ``(K,)`` containing the UCB
        score of each accepted candidate.
    """
    return ucb_scores
```

### Step 8

08_optimize_protein_candidate.py

Goal
----
Orchestrate the complete deterministic protein-acquisition pipeline. Call the preceding sequence-encoding, surrogate, potential, reflective-leapfrog, discretization, Metropolis, and ensemble-UCB steps in order, then return the maximum UCB among the accepted candidates.

```python
from importlib import import_module

import numpy as np


def optimize_protein_candidate(
    initial_sequence: np.ndarray,
    alphabet_size: int,
    weights_ensemble: np.ndarray,
    contact_matrix: np.ndarray,
    biases: np.ndarray,
    contact_scales: np.ndarray,
    penalty_scales: np.ndarray,
    seeds: np.ndarray,
    epsilon: float,
    n_steps: int,
) -> float:
    """Run all acquisition stages and return the maximum UCB.

    Parameters
    ----------
    initial_sequence : np.ndarray
        One-dimensional array of zero-based residue-class indices.
    alphabet_size : int
        Number of permitted residue classes.
    weights_ensemble : np.ndarray
        Surrogate weight matrices of shape ``(M, L, A)``.
    contact_matrix : np.ndarray
        Residue-contact matrix of shape ``(A, A)``.
    biases : np.ndarray
        Surrogate biases of shape ``(M,)``.
    contact_scales : np.ndarray
        Surrogate contact multipliers of shape ``(M,)``.
    penalty_scales : np.ndarray
        Surrogate penalty multipliers of shape ``(M,)``.
    seeds : np.ndarray
        One non-negative integer seed per ensemble member.
    epsilon : float
        Positive leapfrog step size.
    n_steps : int
        Positive number of leapfrog steps per trajectory.

    Raises
    ------
    ValueError
        If ensemble dimensions or seeds are invalid, a downstream
        input fails validation, or no candidate is accepted.

    Returns
    -------
    maximum_ucb : float
        Maximum UCB across the accepted discrete candidates.
    """
    return maximum_ucb
```

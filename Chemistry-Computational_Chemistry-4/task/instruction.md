# Chemistry-Computational_Chemistry-4

## Background

Free energy differences ΔF between equilibrium states are hard to get from equilibrium simulation (full equilibration needed at every intermediate state), which motivated non-equilibrium methods based on the Jarzynski relation ⟨e^(−W)⟩ = e^(−ΔF): drive the system from A to B along a finite-time, time-dependent Hamiltonian H(t) and average the exponentiated work W over sampled trajectories. The resulting finite-sample estimator ΔF^(n) = −ln[(1/n)Σe^(−Wᵢ)] converges slowly because it's dominated by rare low-work trajectories. The relation holds for any choice of intermediate H(t), so the natural question is which choice minimizes the estimator's error — unlike prior work, which optimized dissipation (not error) over small parametrized protocol families.

This is addressed by working with discrete-time Markov chains (states x=1..m on a line, MC time τ=0..N+1) with Glauber transition rates π(x→y;τ) = g(x→y)f(H(y,τ+1)−H(x,τ+1)), f(ΔE)=1/(1+e^ΔE), which satisfy detailed balance and reduce to overdamped 1D diffusion in the continuum limit. Work accumulates as δW(x,τ)=H(x,τ+1)−H(x,τ). Rather than dissipation, the target is the large-n asymptotic MSE of the Jarzynski estimator, MSE ∝ ⟨e^(−2(W−ΔF))⟩, which — since ΔF is fixed by the boundary Hamiltonians — reduces to minimizing the proxy MSE′ ≡ ⟨e^(−2W)⟩ over all intermediate energies.

The key enabling trick is that for discrete-time chains, ⟨e^(−αW)⟩ is computable exactly via a "tilted" master equation: reweight the initial distribution and each transition matrix by e^(−αδW), then take the matrix-vector product p̃(α)ᵀπ̃(0,α)⋯π̃(N,α)𝟙. This costs only O(m²N), making gradient-based optimization (autodiff + Adam, multiple restarts) over all free intermediate energies tractable — a Monte Carlo gradient estimate would be far too noisy. A gauge freedom, H(x,τ)→H(x,τ)+c(τ), leaves both the dynamics and MSE′ unchanged and is used to fix H(1,τ)≡0, leaving one free energy trajectory H(2,τ) in the two-state case.

Across all systems studied, optimal intermediates jump discontinuously at the first and last steps rather than interpolating smoothly, with jump size growing as the energy gap widens, and can cut MSE by over an order of magnitude versus linear/logarithmic interpolation — without necessarily dissipating less work, contradicting the usual low-dissipation-implies-low-error intuition. The target case (m=2, N=10, Eᵢ=−8, E_f=8, β=1) sits in this strongly-driven regime shown qualitatively in Fig. 2b of the source, which only plots the shape of the optimal H(2,τ) curve there, not the resulting scalar MSE′ — so the value must actually be computed via the optimization, not read off a figure.

## Problem

Non-equilibrium free energy estimators based on the Jarzynski relation converge poorly because the exponential work average ⟨e⁻ᵂ⟩ is dominated by rare low-work trajectories, and this poor convergence has historically been addressed only by restricting the time-dependent switching protocol to a low-dimensional parametrized family. A recent method instead determines, without any restriction on functional form, the sequence of intermediate Hamiltonians connecting an initial and final Hamiltonian that minimizes the large-sample-limit mean squared error (MSE) of the Jarzynski estimator for discrete-time Markov chains. The primary input is a discrete-time Markov model with specified boundary Hamiltonians, a fixed number of intermediate time steps, and inverse temperature β=1; the primary output is the minimal achievable value of the error proxy MSE′ = ⟨e⁻²ᵂ⟩ over all choices of the intermediate energies.

Consider the two-state toy model (states x∈{1,2}) evolving under discrete-time Glauber dynamics, whose transition rates satisfy detailed balance with respect to the time-dependent Hamiltonian H(x,τ) and depend only on nearest-neighbor energy differences. The trajectory work is accumulated as the Hamiltonian is updated at each time step before the system stochastically transitions, and total work is additive over the full sequence of updates. Because the gauge transformation H(x,τ) → H(x,τ)+c(τ) leaves both the Markov dynamics and MSE′ invariant, fix H(1,τ)=0 for all τ, leaving only H(2,τ) as a free energy level at each intermediate time; the initial condition is drawn from the equilibrium distribution of the initial Hamiltonian. Using this gauge, set the boundary values H(2,τ=0) = Eᵢ = −8 and H(2,τ=N+1) = E_f = 8 with N=10 intermediate steps (so 10 free energy parameters λ₁,…,λ₁₀ ≡ H(2,1),…,H(2,10)), and take the tilting exponent α=2 in MSE′.

Using the source method's own approach for evaluating generalized exponential work averages exactly (without resorting to trajectory sampling) for discrete-time Markov chains, construct the objective MSE′(λ₁,…,λ₁₀) for the two-state chain described above, then numerically minimize it via gradient-based optimization with multiple random restarts to guard against convergence to a poor local minimum, and report the smallest value of MSE′ found at convergence.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:

The tags are required. Do not omit them or leave them empty.
The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> concise but complete. Briefly state the key method steps and conventions needed to justify the result, together with the few scalars that determine the final number.
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

glauber_transition_matrix

Goal
----
Builds the single-time-step Glauber transition matrix for a linear chain of m states, as defined by the source method's Eqs. 6-8. Given the Hamiltonian values H(x, tau+1) governing the jump out of time step tau, this step computes the row-stochastic, tridiagonal transition matrix pi(tau) with nearest-neighbor proposal probability g(x->y)=1/2 and Glauber acceptance probability f(dE)=1/(1+exp(dE)). This matrix is the core primitive reused, unmodified, by every later step in the pipeline: the equilibrium-distribution propagator, the exponentially tilted matrices used to evaluate the exact work average <exp(-alpha*W)>, and ultimately the MSE' objective that is minimized over the intermediate Hamiltonians.

```python
def glauber_transition_matrix(energies: np.ndarray) -> np.ndarray:
    '''Build the Glauber transition matrix for a linear chain of states.

    Parameters
    ----------
    energies : np.ndarray
        1D array of length m >= 2 giving the Hamiltonian value H(x, tau+1)
        for each state x = 1, ..., m (in order along the chain), i.e. the
        Hamiltonian that governs the transitions out of time step tau.

    Returns
    -------
    pi : np.ndarray
        m x m array of dtype float64. pi[i, j] is the Glauber transition
        probability from state (i+1) to state (j+1). Transitions are only
        nonzero between nearest neighbors on the chain (|i - j| == 1); each
        row sums to 1, with the diagonal entry pi[i, i] absorbing the
        probability of remaining in state (i+1).

    Raises
    ------
    ValueError
        If `energies` is not convertible to a one-dimensional array of real
        numbers, if it has fewer than 2 elements, or if it contains any NaN
        or infinite value.
    '''
    return pi  # placeholder
```

### Step 2

work_increments

Goal
----
Computes delta_W(x, tau) = H(x, tau+1) - H(x, tau) for every state x andevery time step tau = 0, ..., N (source method's Eq. 11), the elementary work performed on the system when the Hamiltonian is updated immediately before each stochastic transition. This is the quantity summed over tau to obtain the total trajectory work (Eq. 12), and its per-step values are what Step 4 uses to tilt the initial equilibrium vector.

```python
def work_increments(H: np.ndarray) -> np.ndarray:
    '''Compute per-state work increments along a Hamiltonian trajectory.

    Parameters
    ----------
    H : np.ndarray
        2D array of shape (m, T), giving the Hamiltonian trajectory
        H(x, tau) for each state x = 1, ..., m (rows) and each time step
        tau = 0, ..., T-1 (columns, increasing in time).

    Returns
    -------
    dW : np.ndarray
        Array of shape (m, T-1) and dtype float64, where dW[:, tau] =
        H(:, tau+1) - H(:, tau) for tau = 0, ..., T-2.

    Raises
    ------
    ValueError
        If H is not convertible to a two-dimensional array of real numbers,
        if H has fewer than 2 rows (states) or fewer than 2 columns (time
        steps), or if H contains any NaN or infinite value.
    '''
    return dW  # placeholder
```

### Step 3

tilted_transition_matrix

Goal
----
Builds the exponentially "tilted" version of a single-time-step Glauber transition matrix, as used in the tilted master-equation representation of the source method's Eqs. 16-17. Given the Hamiltonian values at two consecutive time steps, this step reuses the untilted transition matrix from Step 1 (evaluated at the earlier time step) andreweights each column by exp(-alpha * delta_W(y, s)), where delta_W is the per-state work increment between the two time steps (Step 2). Chaining these tilted matrices together (one per time step) is what allows the exact, noise-free evaluation of the generalized work average <exp(-alpha*W)> that Step 4 builds the MSE' optimization objective from.

```python
def tilted_transition_matrix(
    energies_here: np.ndarray, energies_ahead: np.ndarray, alpha: float
) -> np.ndarray:
    '''Build the alpha-tilted Glauber transition matrix for one time step.

    Parameters
    ----------
    energies_here : np.ndarray
        1D array of length m giving H(x, s), the Hamiltonian values at time
        step s. Used both to build the untilted Glauber transition matrix
        (as in Step 1) and as the earlier reference point of the
        destination-state work increment.
    energies_ahead : np.ndarray
        1D array of length m giving H(x, s+1), the Hamiltonian one time step
        later than energies_here. Used together with energies_here to
        compute delta_W(y, s) = H(y, s+1) - H(y, s) for each destination
        state y. Must have the same length as energies_here.
    alpha : float
        Tilting exponent used to reweight each transition by
        exp(-alpha * delta_W(y, s)).

    Returns
    -------
    pi_tilde : np.ndarray
        m x m array of dtype float64. pi_tilde[i, j] equals the untilted
        Glauber transition probability pi(i+1 -> j+1) built from
        energies_here (Step 1), multiplied by
        exp(-alpha * (energies_ahead[j] - energies_here[j])).

    Raises
    ------
    ValueError
        If energies_here or energies_ahead is not convertible to a
        one-dimensional array of real numbers, if they do not have the same
        length, if either has fewer than 2 elements, if either contains any
        NaN or infinite value, or if alpha is not convertible to a finite
        real scalar.
    '''
    return pi_tilde  # placeholder
```

### Step 4

exponential_work_average

Goal
----
Chains the per-time-step tilted Glauber transition matrices from Step 3 into the full tilted master-equation product (source method's Eq. 17) to compute the exact generalized work average <exp(-alpha*W)> for a complete Hamiltonian trajectory H(x, tau), tau = 0, ..., N+1. Combines the equilibrium distribution of the initial Hamiltonian (beta = 1) with the initial work increment from Step 2 and the product of N tilted transition matrices from Step 3, contracted against the all-ones vector. This is the core quantity minimized (with alpha = 2) over the free intermediate Hamiltonians in Steps 6-7 to obtain the MSE'-optimal switching protocol.

```python
def exponential_work_average(H: np.ndarray, alpha: float) -> float:
    '''Compute the exact generalized work average <exp(-alpha*W)>.

    Parameters
    ----------
    H : np.ndarray
        2D array of shape (m, N+2), giving the full Hamiltonian trajectory
        H(x, tau) for each state x = 1, ..., m (rows) and each time step
        tau = 0, ..., N+1 (columns, in increasing time order). Column 0 is
        the initial Hamiltonian and column N+1 is the final Hamiltonian;
        any columns strictly in between are the intermediate Hamiltonians.
        Reduced units with inverse temperature beta = 1 are used throughout,
        so the initial equilibrium distribution is p_eq(x) proportional to
        exp(-H(x, 0)).
    alpha : float
        Tilting exponent in the generalized work average <exp(-alpha*W)>.
        alpha = 1 recovers the plain Jarzynski average <exp(-W)>; alpha = 2
        gives the MSE' error proxy used for optimization.

    Returns
    -------
    result : float
        Native Python float, the exact value of <exp(-alpha*W)> for the
        given Hamiltonian trajectory, computed via the tilted master
        equation (Eq. 16-17).

    Raises
    ------
    ValueError
        If H is not convertible to a two-dimensional array of real numbers,
        if H has fewer than 2 rows (states) or fewer than 2 columns (time
        steps), if H contains any NaN or infinite value, or if alpha is not
        convertible to a finite real scalar.
    '''
    return result  # placeholder
```

### Step 5

linear_interpolation_mse

Goal
----
Evaluates the MSE' error proxy (Step 4, alpha=2) for the naive linear interpolation of the intermediate Hamiltonians between the boundary energies, H(2,tau) = Ei + (Ef-Ei)*tau/(N+1) (the source method's Eq. 18). This is the reference baseline against which the source method's optimal intermediates are compared (Fig. 3a): it reports that optimal intermediates reduce the MSE by up to an order of magnitude relative to this baseline for large changes in the energy landscape.

```python
def linear_interpolation_mse(Ei: float, Ef: float, N: int, alpha: float) -> float:
    '''Evaluate MSE' for linearly interpolated intermediate Hamiltonians.

    Parameters
    ----------
    Ei : float
        Initial energy of state 2, H(2, tau=0).
    Ef : float
        Final energy of state 2, H(2, tau=N+1).
    N : int
        Number of intermediate time steps. Must be a positive integer.
    alpha : float
        Tilting exponent of the objective <exp(-alpha*W)> (Step 4).
        alpha = 2 gives the MSE' error proxy used by the source method.

    Returns
    -------
    result : float
        Native Python float, the value of <exp(-alpha*W)> (Step 4)
        evaluated on the linearly interpolated trajectory
        H(2, tau) = Ei + (Ef - Ei) * tau / (N + 1) for tau = 0, ..., N+1.

    Raises
    ------
    ValueError
        If Ei, Ef, or alpha is not convertible to a finite real scalar, or
        if N is not a positive integer.
    '''
    return result  # placeholder
```

### Step 6

optimal_intermediate_energies

Goal
----
Runs a deterministic multi-start gradient-based search (following the source method's Sec. II C, minimized ten times from different random initial guesses to guard against spurious local minima) over the free intermediate energies H(2,1), ..., H(2,N) of a two-state Glauber chain, and returns the best (argmin) vector found. This exposes the recovered optimal protocol itself, whose boundary behavior -- a finite jump at the initial step and a much larger jump at the final step -- is the qualitative signature reported for large changes in the energy landscape (Fig. 2).

```python
def optimal_intermediate_energies(
    Ei: float, Ef: float, N: int, alpha: float, n_restarts: int, seed: int
) -> np.ndarray:
    '''Find the MSE-minimizing intermediate energies for a two-state
    Glauber chain via deterministic multi-start optimization.

    Parameters
    ----------
    Ei : float
        Initial energy of state 2, H(2, tau=0).
    Ef : float
        Final energy of state 2, H(2, tau=N+1).
    N : int
        Number of free intermediate energies to optimize. Must be a
        positive integer.
    alpha : float
        Tilting exponent of the objective <exp(-alpha*W)> (Step 4) being
        minimized. alpha = 2 gives the MSE' error proxy used by the source method.
    n_restarts : int
        Number of independent optimization runs from different starting
        points. Must be a positive integer.
    seed : int
        Seed passed to np.random.default_rng to deterministically generate
        the random restart starting points. Must be a non-negative integer.

    Returns
    -------
    lam_star : np.ndarray
        Array of shape (N,) and dtype float64: the intermediate energies
        H(2,1), ..., H(2,N) achieving the smallest value of the Step 4
        objective found across all restarts.

    Raises
    ------
    ValueError
        If Ei, Ef, or alpha is not convertible to a finite real scalar, if
        N is not a positive integer, if n_restarts is not a positive
        integer, or if seed is not a non-negative integer.
    '''
    return lam_star  # placeholder
```

### Step 7

orchestrator

Goal
----
Evaluates the MSE' objective (Step 4) at the MSE'-optimal intermediate energies recovered by Step 6, and cross-checks the result against the linear-interpolation baseline (Step 5), returning the single deterministic scalar that is the source method's central numerical result for a given two-state model instance (Eq. 15). This step exercises the full pipeline: Step 1 (transition matrix) through Step 6 (optimal intermediates) and Step 5 (baseline), matching every stage of the reference derivation. For the source method's own large-DeltaE example (N=10, Ei=-8, Ef=8, alpha=2), the optimized value is approximately an order of magnitude smaller than the linear-interpolation baseline.

```python
def orchestrator(
    Ei: float, Ef: float, N: int, alpha: float, n_restarts: int, seed: int
) -> float:
    '''Compute the minimized MSE' achieved by the optimal intermediate
    energies for a two-state Glauber chain, cross-checked against the
    linear-interpolation baseline.

    Parameters
    ----------
    Ei : float
        Initial energy of state 2, H(2, tau=0).
    Ef : float
        Final energy of state 2, H(2, tau=N+1).
    N : int
        Number of free intermediate energies. Must be a positive integer.
    alpha : float
        Tilting exponent of the objective <exp(-alpha*W)> (Step 4).
        alpha = 2 gives the MSE' error proxy used by the source method.
    n_restarts : int
        Number of independent optimization runs used by Step 6 to recover
        the optimal intermediate energies. Must be a positive integer.
    seed : int
        Seed passed to Step 6 to deterministically generate the restart
        starting points. Must be a non-negative integer.

    Returns
    -------
    result : float
        Native Python float: the Step 4 objective evaluated at the optimal
        intermediate energies returned by Step 6, i.e. the globally
        MSE'-minimizing value for this model instance. Internally this
        value is verified to be no larger than the Step 5 linear-
        interpolation baseline before being returned.

    Raises
    ------
    ValueError
        If Ei, Ef, or alpha is not convertible to a finite real scalar, if
        N is not a positive integer, if n_restarts is not a positive
        integer, or if seed is not a non-negative integer.
    '''
    return result  # placeholder
```

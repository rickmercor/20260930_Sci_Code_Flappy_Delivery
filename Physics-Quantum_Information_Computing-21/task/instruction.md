# Physics-Quantum_Information_Computing-21

## Background

Open quantum systems interact with their surroundings, so a real device rarely evolves under a closed unitary alone. When a system is monitored continuously, for example through homodyne or heterodyne detection, the observer's knowledge of the state is updated in real time, and the conditional state follows a stochastic equation rather than the ordinary Lindblad master equation. Reconstructing the underlying Hamiltonian and dissipative couplings from such continuous measurement records, rather than from repeated discrete measurements, is a standard problem in quantum metrology and characterization, since these parameters set the behavior of the device but are rarely known with certainty in advance.

Once a physical process is characterized as a completely positive trace preserving map, it becomes a channel that can be composed with other channels. A long standing question in quantum information is whether the order in which two noisy channels act matters, and whether that order can itself be treated as a quantum resource. The quantum switch is a device that places two operations in a coherent superposition of orders, controlled by an auxiliary system, rather than a fixed sequential application. This indefinite causal order has been studied for its effect on communication capacity, and more recently on the survival of entanglement passed through noisy channels, since noise can otherwise degrade or entirely destroy the correlations that a fixed order of channels would preserve.

Separately, deciding whether a given quantum state is entangled, and by how much, is computationally hard in general. The positive partial transpose criterion gives an efficiently checkable necessary condition for separability, but it is not sufficient once the local dimension is large enough, since there exist entangled states that nonetheless pass this test. Hierarchies of successively stronger semidefinite tests, based on requiring a state to extend consistently across multiple copies of one subsystem while remaining positive under partial transpose, refine this criterion and can, in principle, detect any entangled state at a high enough level. Turning such a hierarchy into a genuine quantitative measure of entanglement, rather than a pass or fail test, connects computational tractability with an operationally meaningful resource quantifier.

## Problem

A qutrit dissipative process is estimated from noisy continuous monitoring, turned into a completely positive channel, placed in coherent superposition with a second given channel through a quantum switch, and the resulting output state is scored by a resource-theoretic entanglement measure. The task combines three independent, recently developed techniques into a single reported number.

Stage 1. A three level system with true Hamiltonian $H_{\alpha_0}=\alpha_0 H_0$ and a single jump operator $L_{\beta_0}=\beta_0 L_0$ is continuously monitored. What is physically recorded is the continuous measurement current, whose increments are set by the system's conditional state, and that conditional state follows a nonlinear stochastic master equation driven by measurement back-action, not a deterministic drift with independent additive noise on top of it. Build the fixed generators with `rng = np.random.default_rng(101)`, drawing `A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))` and setting $H_0=(A+A^\dagger)/2$, then drawing `L0 = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))` from the same generator, in that order. The true parameter values are $\alpha_0=1.3$, $\beta_0=0.8$. The initial state is $\rho_0=|0\rangle\langle0|$ (the first computational basis state of the qutrit).

Simulate $N=30$ independent noisy observation trajectories on $[0,1]$, reported at $n=40$ equal coarse steps. Each trajectory is built by Euler-Maruyama stepping of the true nonlinear stochastic master equation on a finer sub-grid of $n_{\rm sub}=25$ steps per coarse step, drawing one noise increment per fine sub-step with `rng = np.random.default_rng(202)` calling `rng.standard_normal(1)` once per trajectory per fine sub-step, in trajectory-major, then coarse-step-major, then sub-step-minor order; each fine observation increment is formed from the state at the start of its sub-step. After every fine sub-step, project the resulting matrix back onto the physical set before continuing, by taking its Hermitian part, clipping any negative eigenvalues to zero, and rescaling to unit trace, since a finite-step update of this nonlinear equation can otherwise leave that set. Aggregate the fine observation increments onto each of the $n$ coarse steps, then average the $N$ trajectories' coarse increments to obtain the data actually available for estimation.

A recently developed statistical method reconstructs the unknown parameter pair $\theta=(\alpha,\beta)$ from this averaged data by maximizing a discrete contrast function built from the deterministic averaged dynamics (the ordinary, noise-free Lindblad master equation) evaluated exactly, not by a finite-step discretization, at each trial parameter. Both $\alpha,\beta$ are known a priori to lie in $[0,2]$. Because this contrast surface is not globally well behaved, maximize it the way the source method does: evaluate the contrast on a uniform grid over $[0,2]\times[0,2]$ with spacing $0.1$, keep every grid point whose contrast is at least as large as all of its up-to-eight grid neighbors as a local-maximum candidate, greedily select the highest-scoring candidates enforcing a minimum Euclidean separation of $0.15$ between selected points (up to 8 candidates), refine each selected candidate by a bounded Gauss-Newton-type nonlinear least-squares fit constrained to $[0,2]\times[0,2]$, and report the refined candidate with the largest contrast value as $\hat\theta=(\hat\alpha,\hat\beta)$.

From $\hat\theta$, build the estimated Lindblad generator on the qutrit using $H_{\hat\alpha}=\hat\alpha H_0$ and $L_{\hat\beta}=\hat\beta L_0$ in the standard Lindblad form, exponentiate this generator over a fixed time $\tau=0.3$ to obtain a completely positive trace preserving map on the qutrit, and extract its Kraus operators from the resulting Choi matrix, keeping only eigenvalue components above $10^{-8}$. Confirm that the resulting set of Kraus operators is complete. Treat this as a channel $\mathcal E_1$ acting as the identity on the first, reference qutrit and as this estimated map on the second, i.e. with Kraus operators $I_3\otimes K_i$.

Stage 2. A second channel $\mathcal E_2$ acts jointly on both qutrits: a two-qutrit depolarizing channel acting as $\rho\mapsto\lambda_2\rho+(1-\lambda_2)\,I_9/9$ with depolarizing parameter $\lambda_2=0.35$ (so $\lambda_2=1$ leaves the state unchanged), realized with Kraus operators proportional to the 81 products $D_{mn}\otimes D_{m'n'}$ of the qutrit Weyl (clock and shift) operators $X|k\rangle=|(k+1)\bmod 3\rangle$ and $Z|k\rangle=\omega^k|k\rangle$ with $\omega=e^{2\pi i/3}$. A recently developed protocol places $\mathcal E_1$ and $\mathcal E_2$ in a quantum switch, coherently controlling which channel acts first, with a fixed local unitary $U=D_{2,0}\otimes D_{2,0}$ (using $D_{mn}=X^mZ^n$) inserted between the two channel applications, and a control qubit measured in the phase basis at phase $\phi=0$. Feed in the maximally entangled two-qutrit input state $|\psi\rangle=\frac1{\sqrt3}(|00\rangle+|11\rangle+|22\rangle)$. Apply the protocol's own construction to obtain the two postselected conditional output states (for the two control outcomes at $\phi=0$), their postselection probabilities, and, for comparison, the two possible fixed-order outputs (channel $\mathcal E_1$ then $U$ then $\mathcal E_2$, and the reverse). Compute the entanglement negativity of all four candidate output states across the two-qutrit bipartition, take the larger of the two fixed-order negativities as the best achievable classical-mixture benchmark, and report the resulting gain from using the coherent switch over that benchmark. Identify the postselected output with the larger negativity as the branch of interest, and normalize it to a valid density matrix $\hat\rho_f$.

Stage 3. A recently developed hierarchy of entanglement measures $E_k$, built from states admitting a $k$-symmetric extension that stays positive under partial transpose, quantifies how far a state is from this hierarchy's free set at level $k$, and is computable as a semidefinite program. This program has a bilinear domination constraint in its natural (normalized) form; writing it instead in an unnormalized extension variable removes that bilinearity, turning the whole computation into a single semidefinite program with no outer search needed. Using $k=2$, compute $E_2(\hat\rho_f)$ for the state obtained at the end of Stage 2, using this homogeneous formulation solved to convergence.

Report, as the final numerical answer, the value $E_2(\hat\rho_f)$.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

averaged_lindblad_evolution

Goal
----
Evolve the deterministic averaged state of an open quantum system forward in time by exact propagation of its (time-independent) Lindblad generator, via the matrix exponential of the generator's vectorized superoperator.



This averaged, noise-free evolution is the building block that a maximum-contrast parameter estimator for continuously monitored open quantum systems evaluates repeatedly at trial parameter values. Using the exact propagator, rather than a finite-step discretization, keeps every evaluated state exactly trace preserving and positive semidefinite, and matches how the deterministic averaged dynamics entering the contrast function are evaluated.

```python
def averaged_lindblad_evolution(H: "np.ndarray", Ls: "np.ndarray", rho0: "np.ndarray", T: float, n: int) -> "np.ndarray":
    """Evolve the averaged Lindblad master equation forward by exact propagation.

    Parameters
    ----------
    H : np.ndarray
        (d, d) complex Hermitian Hamiltonian.
    Ls : np.ndarray
        (r, d, d) complex array of r jump operators.
    rho0 : np.ndarray
        (d, d) complex Hermitian, trace-1 initial state.
    T : float
        Total evolution time, T > 0.
    n : int
        Number of equally spaced time points at which to report the state, n >= 1.

    Returns
    -------
    traj : np.ndarray
        (n+1, d, d) complex array, the averaged state at t_0, ..., t_n, obtained by
        repeatedly applying the exact propagator exp((T/n) * L) to rho0.

    Raises
    ------
    ValueError
        If T <= 0 or n < 1.
    """
    return traj
```

### Step 2

simulate_averaged_observation_data

Goal
----
Simulate the observation record produced by continuously monitoring a qutrit whose true Hamiltonian and jump operator are fixed multiples of two seeded generators, and average the resulting noisy increments across many independent trajectories.

What is physically recorded is the continuous measurement current, whose increments are set by the conditional state: that conditional state follows a nonlinear stochastic master equation driven by measurement back-action, not a deterministic drift with additive noise on top. Each trajectory is simulated by Euler-Maruyama stepping of this nonlinear equation on a fine sub-grid, with the resulting state projected back onto the physical set after every fine step (Hermitian part taken, negative eigenvalues clipped to zero, rescaled to unit trace) to prevent the finite-step update from leaving that set, each fine observation increment formed from the state at the start of its sub-step, and the fine increments aggregated onto the coarser observation grid. This averaged data is what a maximum-contrast estimator for continuously monitored open quantum systems uses to reconstruct the unknown Hamiltonian and jump-operator scales.

```python
def simulate_averaged_observation_data(seed_gen: int, seed_noise: int, alpha0: float, beta0: float, T: float, n: int, N: int, n_sub: int) -> "np.ndarray":
    """Simulate N averaged noisy observation increments for a monitored qutrit.

    Parameters
    ----------
    seed_gen : int
        RNG seed used to build the fixed Hamiltonian and jump-operator generators:
        with ``rng = np.random.default_rng(seed_gen)``, draw
        ``A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` and set
        ``H0 = (A + A.conj().T) / 2``, then draw
        ``L0 = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` from the
        same generator, in that order. The true generators are ``H = alpha0 * H0``
        and ``L = beta0 * L0``. The monitored qutrit starts in the fixed initial
        state ``rho0 = |0><0|`` (the first computational basis state), independent
        of ``seed_gen``.
    seed_noise : int
        RNG seed used to draw the observation noise: with
        ``rng_noise = np.random.default_rng(seed_noise)``, one
        ``rng_noise.standard_normal(1)`` value is drawn per fine Euler-Maruyama
        sub-step, in trajectory-major, then coarse-step-major, then sub-step-minor
        order (all draws for trajectory 0 before any for trajectory 1, and so on).
    alpha0 : float
        True Hamiltonian scale, so that the true Hamiltonian is alpha0 * H0.
    beta0 : float
        True jump-operator scale, so that the true jump operator is beta0 * L0.
    T : float
        Total observation time, T > 0.
    n : int
        Number of equal coarse observation steps, n >= 1.
    N : int
        Number of independent observation trajectories to average, N >= 1.
    n_sub : int
        Number of Euler-Maruyama sub-steps simulated within each coarse observation
        step, n_sub >= 1. The true conditional-state trajectory and its observation
        increments are generated on this finer grid and then aggregated onto the n
        coarse steps.

    Returns
    -------
    Y_avg : np.ndarray
        (n,) real array, the trajectory-averaged observation increment at each coarse
        step.

    Raises
    ------
    ValueError
        If T <= 0, n < 1, N < 1, or n_sub < 1.
    """
    return Y_avg
```

### Step 3

discrete_contrast_estimator

Goal
----
Reconstruct the unknown Hamiltonian and jump-operator scale factors of a continuously monitored qutrit from averaged observation data, by maximizing a discrete contrast function built from the deterministic averaged dynamics evaluated at a trial parameter.



The maximization follows the source paper's own prescribed numerical procedure: the contrast is first evaluated on a uniform grid over the whole bounded parameter space, grid points that are local-maximum candidates (no neighbor scores higher) are selected as well-separated starting points, and each selected candidate is then refined by a bounded Gauss-Newton-type local optimization (equivalently, a bounded nonlinear least-squares fit, since maximizing this contrast is exactly the same as minimizing a weighted sum of squared residuals between the trial drift and the data), with the best-scoring refined candidate reported as the estimate. This combination is required because the contrast surface is not globally well-behaved: a single local run from one starting point can converge to a poor local optimum.

```python
def discrete_contrast_estimator(seed_gen: int, Y_avg: "np.ndarray", T: float, n: int) -> "np.ndarray":
    """Maximize the discrete contrast function to estimate (alpha, beta).

    Parameters
    ----------
    seed_gen : int
        RNG seed used to build the fixed Hamiltonian and jump-operator generators:
        with ``rng = np.random.default_rng(seed_gen)``, draw
        ``A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` and set
        ``H0 = (A + A.conj().T) / 2``, then draw
        ``L0 = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` from the
        same generator, in that order. The trial Hamiltonian and jump operator at a
        candidate ``(alpha, beta)`` are ``alpha * H0`` and ``beta * L0``, and the
        deterministic dynamics used to build the contrast start from the fixed
        initial state ``rho0 = |0><0|`` (the first computational basis state),
        independent of ``seed_gen``.
    Y_avg : np.ndarray
        (n,) real array, the trajectory-averaged observation increment at each step.
    T : float
        Total observation time, T > 0.
    n : int
        Number of equal discrete time steps, n >= 1, matching Y_avg's length.

    Returns
    -------
    theta_hat : np.ndarray
        (2,) real array, the fitted (alpha_hat, beta_hat) maximizing the contrast,
        obtained by a uniform grid search over [0, 2] x [0, 2] with spacing 0.1,
        selecting up to 8 well-separated local-maximum candidates (minimum Euclidean
        separation 0.15), refining each by a bounded nonlinear least-squares fit
        constrained to [0, 2] x [0, 2], and returning the refined candidate with the
        largest contrast value.

    Raises
    ------
    ValueError
        If T <= 0, n < 1, or Y_avg does not have length n.
    """
    return theta_hat
```

### Step 4

channel_from_lindbladian

Goal
----
Build a completely positive trace preserving quantum channel from a fitted Hamiltonian and jump-operator scale pair, by exponentiating the resulting Lindblad generator over a fixed time and extracting Kraus operators from the resulting Choi matrix.



This turns a statistically fitted open-system generator into an object, a quantum channel, that can be composed with other channels, for example placed in a coherent switch with a second, independently specified channel.

```python
def channel_from_lindbladian(seed_gen: int, alpha_hat: float, beta_hat: float, tau: float) -> "np.ndarray":
    """Build channel Kraus operators from a fitted Lindbladian by exponentiation.

    Parameters
    ----------
    seed_gen : int
        RNG seed used to build the fixed Hamiltonian and jump-operator generators:
        with ``rng = np.random.default_rng(seed_gen)``, draw
        ``A = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` and set
        ``H0 = (A + A.conj().T) / 2``, then draw
        ``L0 = rng.standard_normal((3,3)) + 1j*rng.standard_normal((3,3))`` from the
        same generator, in that order. The estimated Hamiltonian and jump operator
        exponentiated here are ``H_hat = alpha_hat * H0`` and
        ``L_hat = beta_hat * L0``.
    alpha_hat : float
        Fitted Hamiltonian scale.
    beta_hat : float
        Fitted jump-operator scale.
    tau : float
        Fixed exponentiation time, tau > 0.

    Returns
    -------
    Ks : np.ndarray
        (m, d, d) complex array of the resulting channel's Kraus operators, one per
        Choi eigenvalue above 1e-8, ordered by increasing eigenvalue index.

    Raises
    ------
    ValueError
        If tau <= 0.
    """
    return Ks
```

### Step 5

entanglement_negativity

Goal
----
Compute the entanglement negativity of a bipartite density matrix across a given bipartition, from the trace norm of its partial transpose.



Negativity vanishes on every state with a positive partial transpose, so a nonzero value certifies entanglement, and it is used here to compare the outputs of different causal-order configurations of two channels.

```python
def entanglement_negativity(rho: "np.ndarray", dA: int, dB: int) -> float:
    """Compute the negativity of a bipartite state across the A|B partition.

    Parameters
    ----------
    rho : np.ndarray
        (dA*dB, dA*dB) complex Hermitian, trace-1 density matrix.
    dA : int
        Dimension of subsystem A, dA >= 1.
    dB : int
        Dimension of subsystem B, dB >= 1.

    Returns
    -------
    result : float
        The negativity of rho across A|B, as a native Python float.

    Raises
    ------
    ValueError
        If rho does not have shape (dA*dB, dA*dB).
    """
    return result
```

### Step 6

quantum_switch_output

Goal
----
Place an estimated single-qutrit channel and a given two-qutrit Weyl-noise depolarizing channel in a quantum switch with a fixed inserted local unitary, and return the postselected output branch with the larger entanglement negativity, normalized to a valid density matrix.



A quantum switch coherently controls the order in which two channels act, rather than applying them in a fixed sequence, and postselecting on the control outcome can leave more entanglement in the output than either fixed order or any classical mixture of the two fixed orders.

```python
def quantum_switch_output(Ks_est: "np.ndarray", lam2: float, dloc: int) -> "np.ndarray":
    """Run the two channels through a quantum switch and return the best branch.

    Parameters
    ----------
    Ks_est : np.ndarray
        (m, dloc, dloc) complex array, the Kraus operators of the estimated channel
        acting on a single qutrit, to be applied locally to the second of two qutrits
        (the first qutrit carries the identity): the embedded Kraus operators are
        ``np.kron(np.eye(dloc), K)`` for each ``K`` in ``Ks_est``, and the returned
        state's tensor factors follow this same first-qutrit-then-second-qutrit
        ordering throughout.
    lam2 : float
        Depolarizing parameter of the given two-qutrit Weyl-noise channel, 0 <= lam2 <= 1,
        in the convention where the channel acts as rho -> lam2*rho + (1-lam2)*I/dloc**2
        (lam2 = 1 is the identity channel), realized with Kraus operators proportional to
        the dloc**4 products of the local Weyl operators.
    dloc : int
        Local dimension, dloc >= 3. The inserted local unitary D_{2,0} (x) D_{2,0}
        is fixed in terms of the Weyl index 2, which requires a local dimension of
        at least 3; this function is defined for qutrits and qudits of dimension 3
        or higher, not for qubits (dloc = 2) or a trivial single level (dloc = 1).

    Returns
    -------
    rho_f : np.ndarray
        (dloc*dloc, dloc*dloc) complex array, the normalized output state of the
        postselected switch branch with the larger negativity, for the maximally
        entangled two-qutrit input, the inserted local unitary D_{2,0} (x) D_{2,0}, and
        control phase 0. If one control outcome has exactly zero postselection
        probability, the other (valid) branch is returned directly without a
        negativity comparison.

    Raises
    ------
    ValueError
        If lam2 is not in [0, 1], if dloc < 3, or if both control outcomes have
        zero postselection probability.
    """
    return rho_f
```

### Step 7

symmetric_extension_sdp_solve

Goal
----
Solve, for a bipartite state, the homogeneous semidefinite program that directly certifies the level-k symmetric-extension entanglement measure, in an unnormalized extension variable, by consensus ADMM.



States admitting a symmetric, partial-transpose-positive extension form a hierarchy of outer approximations to the separable states, refined by the number of extension copies. Writing the measure's own defining domination constraint rho <= (1+t)*sigma in terms of the unnormalized variable omega' := (1+t)*omega removes the bilinear coupling between the scalar t and the extension variable, so the whole problem becomes a single linear-objective semidefinite program in omega' alone, with no outer search over t needed: minimizing Tr(omega') directly returns 1+t at the optimum. Consensus ADMM solves this by alternately projecting a shared candidate onto every constraint set in turn (positive semidefiniteness, every partial-transpose positivity, permutation symmetry, and the domination constraint against the given state) and tracking the constraint-wise disagreement at every cycle, giving a real, per-constraint convergence record rather than a single residual.

```python
def symmetric_extension_sdp_solve(rho: "np.ndarray", dA: int, dB: int, k: int, iters: int) -> "np.ndarray":
    """Solve the homogeneous k-symmetric-extension SDP for a state by consensus ADMM.

    Parameters
    ----------
    rho : np.ndarray
        (dA*dB, dA*dB) complex Hermitian, trace-1 density matrix.
    dA : int
        Dimension of the first subsystem, dA >= 1.
    dB : int
        Dimension of each of the k extended copies of the second subsystem, dB >= 1.
    k : int
        Number of symmetric extension copies, k >= 1.
    iters : int
        Number of ADMM cycles to run, iters >= 1.

    Returns
    -------
    omega : np.ndarray
        (dA*dB**k, dA*dB**k) complex array, the ADMM-converged unnormalized extension
        operator omega'; Tr(omega') - 1 equals the level-k symmetric-extension measure.

    Raises
    ------
    ValueError
        If k < 1 or iters < 1.
    """
    return omega
```

### Step 8

compute_Ek

Goal
----
Compute the level-k symmetric-extension entanglement measure of a bipartite state from the converged solution of its homogeneous semidefinite program.



Because the domination constraint of the measure's defining program becomes linear once written in the unnormalized extension variable omega' := (1+t)*omega, the trace of the converged solution directly recovers 1+t, so the measure itself is simply Tr(omega') - 1; no outer search over the threshold t is needed.

```python
def compute_Ek(rho: "np.ndarray", dA: int, dB: int, k: int, iters: int) -> float:
    """Compute E_k(rho) from the converged homogeneous-SDP extension operator.

    Parameters
    ----------
    rho : np.ndarray
        (dA*dB, dA*dB) complex Hermitian, trace-1 density matrix.
    dA : int
        Dimension of the first subsystem, dA >= 1.
    dB : int
        Dimension of each of the k extended copies of the second subsystem, dB >= 1.
    k : int
        Number of symmetric extension copies, k >= 1.
    iters : int
        Number of ADMM cycles used to solve the homogeneous SDP.

    Returns
    -------
    result : float
        E_k(rho) = Tr(omega') - 1, where omega' is the converged unnormalized
        extension operator.

    Raises
    ------
    ValueError
        If k < 1 or iters < 1.
    """
    return result
```

### Step 9

full_certified_switch_pipeline

Goal
----
Orchestrator: run the full three-stage pipeline end to end. Estimate a qutrit's Hamiltonian and jump-operator scale from simulated continuous monitoring, build the resulting channel, run it against a given channel through a quantum switch, and certify the entanglement of the best postselected output with a symmetric-extension measure.



This combines three independent techniques, a continuous-monitoring parameter estimator, a causal-order quantum switch, and a symmetric-extension entanglement hierarchy, into a single deterministic numerical pipeline.

```python
def full_certified_switch_pipeline(
    seed_gen: int,
    seed_noise: int,
    alpha0: float,
    beta0: float,
    T: float,
    n: int,
    N: int,
    n_sub: int,
    tau: float,
    lam2: float,
    dloc: int,
    k: int,
    admm_iters: int,
) -> float:
    """Run the full estimation -> channel -> switch -> certification pipeline.

    Parameters
    ----------
    seed_gen : int
        RNG seed for the fixed Hamiltonian and jump-operator generators.
    seed_noise : int
        RNG seed for the observation noise.
    alpha0 : float
        True Hamiltonian scale used to generate the observation data.
    beta0 : float
        True jump-operator scale used to generate the observation data.
    T : float
        Total observation time, T > 0.
    n : int
        Number of equal coarse observation steps, n >= 1.
    N : int
        Number of independent observation trajectories to average, N >= 1.
    n_sub : int
        Number of Euler-Maruyama sub-steps simulated within each coarse observation
        step, n_sub >= 1.
    tau : float
        Fixed exponentiation time for the estimated channel, tau > 0.
    lam2 : float
        Depolarizing strength of the given second channel, 0 <= lam2 <= 1.
    dloc : int
        Local qutrit dimension, dloc >= 3.
    k : int
        Number of symmetric extension copies used by the entanglement measure, k >= 1.
    admm_iters : int
        Number of ADMM cycles used to solve the homogeneous symmetric-extension SDP.

    Returns
    -------
    result : float
        The E_k value of the switch's best-branch output state, as a native Python
        float.
    """
    return result
```

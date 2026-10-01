# Mathematics-Computational_Mechanics-14

## Background

Flow cytometry measures the fluorescence of thousands of individual cells at a single instant, so a time course consists of independent population snapshots rather than tracked trajectories. Recent work in quantitative cell biology interprets such snapshots with ordinary differential equation models in which every cell carries its own parameters, drawn from a population distribution; inference then targets the hyperparameters of that distribution rather than the cells themselves. Two applications motivate this line of work: the association of engineered nano-particles with cells, where each cell has an association rate and a carrying capacity and all cells in a well deplete the same pool of free particles, and receptor-mediated internalisation of antibodies.

Nominally identical experimental replicates are, in practice, not identical. Differences in sample preparation, instrument settings and operator handling shift the parameter distribution from one replicate to the next. The standard approach pools all replicates into one population, which folds this inter-experimental heterogeneity into the estimated intra-experimental (biological) heterogeneity and can produce parameter distributions that describe none of the replicates.

The source study proposes a Bayesian multilevel hierarchical framework that separates the two. Cell parameters are independent within a replicate and follow replicate-specific distributions; the replicate-level parameters are themselves drawn from a population-level law with its own hyperparameters. Because the cell-level parameters must be integrated out and each replicate's cells interact through a shared environment, the likelihood is intractable, and inference is likelihood-free: simulated snapshots are compared with observed ones through a distributional distance, and a sequential Monte Carlo sampler adapts its acceptance threshold across populations. Simulation studies with deliberately corrupted replicates show that the hierarchical analysis recovers each replicate's parameters and the population-level spread, while the pooled analysis is biased in every hyperparameter.

The setting here is the particle-cell association example of that framework: three replicates, one of which has a larger mean carrying capacity, observed at a few incubation times, with all sizes reduced so that the complete inference is a deterministic and inexpensive computation.

## Problem

Flow-cytometry snapshots from nominally repeated cell experiments confound biological heterogeneity within a replicate with experimental heterogeneity between replicates, so pooled inference can misrepresent every constituent population. Determine the particle mean of the second replicate carrying-capacity mean, $\mu_{K,2}$, after two populations of the multilevel hierarchical affinity benchmark defined below.

The benchmark has $M=3$ replicates, $N=60$ cells per replicate and time, observation times $t=3600(1,2,4)$ s, fractional coverage $c=1$, cell surface area $s=5.31\times10^{-5}$, medium volume $V=1.01\times10^{-6}$, initial free-particle density $u_0=9.95\times10^7$, $N_{\mathrm{prep}}=30$, fine-grid spacing $1800$ s, and a 1000-point empirical-CDF grid. Generate the observations from $\theta_{\mathrm{true}}=(9.0\times10^{-7},3.86125\times10^{-7},3.86125\times10^{-7},10,20,10,4.56962\times10^{-7},2,3.86125\times10^{-7},10,0,0)$, ordered as $(\mu_{r,1:3},\mu_{K,1:3},\sigma_r,\sigma_K,m_r,m_K,s_r,s_K)$, with data seed 1701.

For $i=0,\ldots,63$, use $q_i=(i+0.5)/64$, $z_i=\Phi^{-1}(q_i)$, $D_{\mathrm{cell},i}=\exp(5.3+0.35z_i)$, and $D_{\mathrm{particle},i}=\exp(3.0+0.25z_i)$ as empirical calibration distributions sampled with replacement. Condition the replicate means to be positive and use $\sigma_r\sim U(0,2\times10^{-6})$, $\sigma_K\sim U(0,5)$, $m_r\sim U(0,10^{-6})$, $m_K\sim U(0,50)$, $s_r\sim\mathrm{HalfCauchy}(0,3\times10^{-7})$, and $s_K\sim\mathrm{HalfCauchy}(0,5)$.

Use arithmetic-moment log-normal laws, replicate-wise shared environments, empirical fluorescence, the hierarchical Anderson-Darling discrepancy, and adaptive ABC-SMC with 120 particles, $R_{\mathrm{trial}}=20$, $c_{\mathrm{SMC}}=0.01$, resampled fraction $a=0.5$, $R_{\max}=5000$, and inference seed 424242; all seeded draws use the NumPy PCG64 generator, with one serial stream for the complete inference.

To make the seeded trajectory unique, draw uniforms, Normals, log-normals and standard Cauchy variates directly from that stream; draw a half-Cauchy scale $\gamma$ as $\gamma|C|$ for one standard Cauchy variate $C$; draw each positive Normal replicate mean by inverse-CDF transformation of one uniform variate on $[\Phi(-m/s),1]$; and draw calibration indices as integers in $\{0,\ldots,63\}$. Redraw pilot pairs at every fine-grid time, including $t=0$, replicate by replicate, with all cell draws for a replicate following its pilot draws and a fresh set of cells drawn at each observation time in the order rates, capacities, autofluorescence indices, particle indices; use $\theta+Lz$ with $L$ the lower Cholesky factor of $(2.38^2/d)$ times the sample covariance, with divisor $n-1$, of all 120 particles after resampling, use ordinary Normal target factors after enforcing positive replicate means, draw one Metropolis uniform for every proposal that passes the prior-support check and for no other, and simulate only a proposal that passes the Metropolis test.

Build each observed empirical CDF as the strict count $\#\{X_i<x\}/N$ on a 1000-point grid from $\min X$ to $\max X+10^{-3}$, interpolate it linearly, extrapolate it flat, and clip it to $[10^{-9},1-10^{-9}]$ before taking logarithms; sum the nine replicate-time terms without normalisation.

Initialise the population by drawing each prior particle, its six hyperparameters and then its replicate means, each group in the order of $\theta$, and immediately simulating and scoring it before drawing the next; integrate the mean free-particle density on the fine grid by the trapezoidal rule; draw the $N_{\mathrm{prep}}$ pilot association rates as one vector followed by the $N_{\mathrm{prep}}$ pilot carrying capacities as one vector; take each threshold as the largest retained discrepancy and resample the other half uniformly with replacement from the retained half before any move; and treat $R$ as the total move count, so that $R-R_{\mathrm{trial}}$ further moves follow the trial block, each block running particle by particle in resampled order with one particle completing its moves before the next begins.

In the reasoning, report the mean and standard deviation of the logarithm of the first-replicate association rate, the first-replicate integrated environments $I(3600)$, $I(7200)$ and $I(14400)$, the two sequential thresholds $\epsilon_1$ and $\epsilon_2$, the two adaptive move counts $R_1$ and $R_2$, and the source you took the particle-depletion solution from.

The answer is graded to an absolute tolerance of $0.01$.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules: - The tags are required. Do not omit them or leave them empty. - The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose. - Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_parameterise_hierarchical_lognormals

Goal
----
The output contains the Gaussian parameters of their logarithms in the column order $(m_{r,j}, s_{r,j}, m_{K,j}, s_{K,j})$.

```python
def parameterise_hierarchical_lognormals(
    mu_r: np.ndarray,
    mu_k: np.ndarray,
    sigma_r: float,
    sigma_k: float,
) -> np.ndarray:
    """Convert hierarchical arithmetic moments to log-normal parameters.

    Parameters
    ----------
    mu_r : np.ndarray
        Positive replicate-specific arithmetic means of association rate.
    mu_k : np.ndarray
        Positive replicate-specific arithmetic means of carrying capacity.
    sigma_r : float
        Shared non-negative arithmetic standard deviation of association rate.
    sigma_k : float
        Shared non-negative arithmetic standard deviation of carrying capacity.

    Returns
    -------
    result : np.ndarray
        Array of shape (M, 4) with columns $(m_r, s_r, m_K, s_K)$.

    Raises
    ------
    ValueError
        If the mean arrays are not non-empty one-dimensional arrays of equal
        length, or any mean is non-positive, or any standard deviation is
        negative or non-finite.
    """
    return
```

### Step 2

02_compute_shared_environment_integral

Goal
----
Each cell obeys $\dot P_i=r_ics(1-P_i/K_i)u(t)$ while the shared environment obeys $\dot u=-\frac{1}{N}\sum_i\frac{r_ics}{V}(1-P_i/K_i)u(t)$, so summing the two gives $u(t)=u_0-\frac{1}{N}\sum_iP_i(t)/V$: the mean free-particle density at a node is the pilot average of $u_0-P_q(t)/V$. Evaluate each pilot on its own, as a population in which every cell shares that pilot's rate and capacity, so that $P_q(t)$ is the exact trajectory of that homogeneous population. At each fine-grid node, including $t=0$, draw the $N_{\mathrm{prep}}$ pilot association rates as one vector and then the $N_{\mathrm{prep}}$ pilot carrying capacities as one vector, in that order, without reseeding. Integrate the resulting $\bar u$ by the trapezoidal rule and return $I(t)$ at the requested times.

```python
def compute_shared_environment_integral(
    log_params: np.ndarray,
    times: np.ndarray,
    n_prep: int,
    fine_step: float,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Compute $I(t)=\int_0^t \bar u(\tau)d\tau$ for one replicate.

    Parameters
    ----------
    log_params : np.ndarray
        Length-4 vector $(m_r,s_r,m_K,s_K)$ for one replicate.
    times : np.ndarray
        Observation times in seconds, all lying on the fine grid.
    n_prep : int
        Number of pilot cells sampled at each fine-grid time.
    fine_step : float
        Fine-grid spacing in seconds.
    rng : np.random.Generator
        Shared PCG64 random-number stream. The function draws pilot pairs at
        every fine-grid time, including zero, without reseeding.

    Returns
    -------
    result : np.ndarray
        Integrated mean free-particle density at each requested time.

    Raises
    ------
    ValueError
        If log_params does not have shape (4,), any log-normal scale is
        negative, times is empty, negative or off the fine grid, or n_prep or
        fine_step is not positive.
    """
    return
```

### Step 3

03_simulate_hierarchical_fluorescence

Goal
----
The shared environmental integral is supplied by the preceding stage. Individual cells draw association rates and carrying capacities from replicate-specific log-normal laws, then convert continuous particle counts to empirical fluorescence.

```python
def simulate_hierarchical_fluorescence(
    log_params: np.ndarray,
    environment_integral: np.ndarray,
    n_cells: int,
    cell_calibration: np.ndarray,
    particle_calibration: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate fluorescence snapshots for one replicate.

    Parameters
    ----------
    log_params : np.ndarray
        Length-4 vector $(m_r,s_r,m_K,s_K)$ for one replicate.
    environment_integral : np.ndarray
        Shared integrated free-particle density at each observation time.
    n_cells : int
        Number of observed cells at each time.
    cell_calibration : np.ndarray
        Empirical cell-autofluorescence sample.
    particle_calibration : np.ndarray
        Empirical single-particle fluorescence sample.
    rng : np.random.Generator
        Shared PCG64 random-number stream, already advanced through this
        replicate's pilot-cell draws.

    Notes
    -----
    At each observation time, consume the stream by drawing direct log-normal
    rates, direct log-normal capacities, integer autofluorescence indices and one
    concatenated vector of integer particle-fluorescence indices, in that order.

    Returns
    -------
    result : np.ndarray
        Array of shape (time, cell).

    Raises
    ------
    ValueError
        If log_params does not have shape (4,), any log-normal scale is
        negative, the environment integral is empty or negative, n_cells is not
        positive, or either calibration array is empty.
    """
    return
```

### Step 4

04_compute_snapshot_discrepancy

Goal
----
The total discrepancy is the unnormalised sum of Anderson-Darling distances over all matching replicate-time pairs.

```python
def compute_snapshot_discrepancy(
    observed: np.ndarray,
    simulated: np.ndarray,
    n_grid: int,
) -> float:
    """Compute the unnormalised replicate-time Anderson-Darling sum.

    Parameters
    ----------
    observed : np.ndarray
        Observed snapshots with shape (time, replicate, cell).
    simulated : np.ndarray
        Simulated snapshots with matching time and replicate dimensions.
    n_grid : int
        Number of points in each interpolated empirical-CDF grid.

    Returns
    -------
    result : float
        Sum of all replicate-time Anderson-Darling distances.

    Raises
    ------
    ValueError
        If either array is not three-dimensional with matching time and
        replicate dimensions, an observed snapshot has fewer than two finite
        values, a simulated snapshot has no finite values, or n_grid is below 2.
    """
    return
```

### Step 5

05_evaluate_hierarchical_prior_and_proposal

Goal
----
The adaptive Gaussian random walk is based on $(2.38^2/d)$ times the empirical particle covariance and is factorised through a dimensionless correlation matrix with relative diagonal jitter $10^{-12}$, including for zero-variance coordinates.

```python
import numpy as np

def evaluate_hierarchical_prior_and_proposal(
    theta: np.ndarray,
    particles: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Return one prior draw, one log-prior value, and the proposal factor.

    Parameters
    ----------
    theta : np.ndarray
        Parameter vector whose hierarchical log prior is evaluated.
    particles : np.ndarray
        Current particle matrix with shape (dimension, n_particles).
    rng : np.random.Generator
        Shared PCG64 random-number stream.

    Notes
    -----
    Draw the two within-replicate spreads and two population means directly, then
    draw each half-Cauchy scale as its scale times the absolute value of one
    standard Cauchy variate. Draw each positive replicate mean by inverse CDF from
    one uniform on $[\Phi(-m/s),1]$. Form the lower proposal factor from
    $(2.38^2/d)$ times the sample covariance with divisor $n-1$, factorised
    through the correlation matrix with relative diagonal jitter $10^{-12}$; a
    coordinate with zero sample variance receives the diagonal factor entry
    $\sqrt{\text{tiny}}$, the square root of the smallest positive float, and
    zero off-diagonal entries.

    Returns
    -------
    result : np.ndarray
        Numeric vector containing the prior draw, log prior, and flattened
        scale-aware Gaussian proposal factor.

    Raises
    ------
    ValueError
        If theta does not have dimension 2M+6 matching the particle matrix, or
        the particle matrix has fewer than two particles.
    """
    return
```

### Step 6

06_advance_abc_smc_population

Goal
----
The update sorts particles by discrepancy, retains the best fraction, resamples the remainder, constructs the scale-aware Gaussian random-walk covariance, estimates a trial acceptance rate, and applies the adaptive number of ABC-MCMC moves.

```python
def advance_abc_smc_population(
    theta: np.ndarray,
    rho: np.ndarray,
    observed: np.ndarray,
    times: np.ndarray,
    n_cells: int,
    n_prep: int,
    fine_step: float,
    cell_calibration: np.ndarray,
    particle_calibration: np.ndarray,
    n_grid: int,
    r_trial: int,
    c_smc: float,
    split_fraction: float,
    max_moves: int,
    rng: np.random.Generator,
) -> np.ndarray:
    r"""Advance one adaptive ABC-SMC population and pack the numeric state.

    Parameters
    ----------
    theta : np.ndarray
        Current particle matrix with shape (dimension, n_particles).
    rho : np.ndarray
        Current discrepancy for each particle.
    observed : np.ndarray
        Observed fluorescence array with shape (time, replicate, cell).
    times : np.ndarray
        Observation times in seconds.
    n_cells : int
        Simulated cells per replicate and time.
    n_prep : int
        Pilot cells per fine-grid time.
    fine_step : float
        Fine-grid spacing in seconds.
    cell_calibration : np.ndarray
        Empirical cell-autofluorescence sample.
    particle_calibration : np.ndarray
        Empirical single-particle fluorescence sample.
    n_grid : int
        Number of empirical-CDF grid points.
    r_trial : int
        Trial ABC-MCMC moves per resampled particle.
    c_smc : float
        Target probability bound for a particle never moving.
    split_fraction : float
        Fraction of particles resampled and moved.
    max_moves : int
        Maximum total ABC-MCMC moves per resampled particle.
    rng : np.random.Generator
        Shared PCG64 stream for resampling and move-stage randomness.

    Notes
    -----
    Resample the rejected particle indices first. Process those particles in
    ascending position order, drawing each Gaussian proposal as $\theta+Lz$ with
    the lower Cholesky factor of the divisor-$n-1$ scaled covariance before its support
    check and drawing a Metropolis uniform only for a finite-prior proposal.
    Simulate the proposed data only after that Metropolis screen passes.

    Returns
    -------
    result : np.ndarray
        Packed vector containing four update diagnostics, the updated particle
        matrix, and the updated discrepancies. The first four entries are
        $(\epsilon,\widehat p_{\mathrm{trial}},R,\widehat p_{\mathrm{extra}})$.

    Raises
    ------
    ValueError
        If theta and rho have inconsistent shapes, there are fewer than two
        particles, split_fraction is outside (0, 1) or empties a subset,
        r_trial is below 1, c_smc is outside (0, 1), or max_moves is below
        r_trial.
    """
    return
```

### Step 7

07_run_hierarchical_affinity_benchmark

Goal
----
Run the complete two-population hierarchical affinity ABC-SMC benchmark.

```python
def run_hierarchical_affinity_benchmark(
    theta_true: np.ndarray,
    times: np.ndarray,
    n_cells: int,
    n_prep: int,
    fine_step: float,
    n_grid: int,
    n_particles: int,
    r_trial: int,
    c_smc: float,
    split_fraction: float,
    max_moves: int,
    data_seed: int,
    inference_seed: int,
) -> float:
    """Run two hierarchical ABC-SMC populations from the raw benchmark inputs.

    Parameters
    ----------
    theta_true : np.ndarray
        Generating vector ordered as replicate means followed by six hyperparameters.
    times : np.ndarray
        Observation times in seconds.
    n_cells : int
        Observed and simulated cells per replicate and time.
    n_prep : int
        Pilot cells per fine-grid time.
    fine_step : float
        Fine-grid spacing in seconds.
    n_grid : int
        Empirical-CDF grid size.
    n_particles : int
        Number of ABC-SMC particles.
    r_trial : int
        Trial ABC-MCMC moves per resampled particle.
    c_smc : float
        Target bound used to adapt the move count.
    split_fraction : float
        Fraction of particles resampled and moved.
    max_moves : int
        Maximum total ABC-MCMC moves per resampled particle.
    data_seed : int
        Random seed for generating the observed synthetic dataset.
    inference_seed : int
        Seed for one shared inference random-number stream.

    Notes
    -----
    Generate observations from the independent data stream. On the inference
    stream, initialise particles one at a time by drawing the prior and then its
    simulated data. Each simulation consumes replicate-major pilot and observation
    draws. Carry that stream through both updates, resampling rejected indices first
    and processing moved particles in ascending position order; draw each proposal
    before its support check and a Metropolis uniform only for a finite-prior
    proposal, then simulate only after the screen passes.

    Returns
    -------
    result : float
        Particle mean of the second replicate carrying-capacity mean after two
        sequential ABC-SMC updates.

    Raises
    ------
    ValueError
        If theta_true does not have dimension 2M+6 with at least two
        replicates, or n_particles is below 4.
    """
    return
```

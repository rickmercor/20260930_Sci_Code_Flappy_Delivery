# Chemistry-Computational_Chemistry-70

## Background

RBMD 2.0 addresses the scalability bottleneck of exact pairwise nonbonded force evaluation in large-scale molecular dynamics by replacing exhaustive neighbor summation with the random batch list (RBL) method, a stochastic, momentum-conserving subsampling of each particle's interaction shell that reduces per-step cost while remaining statistically unbiased. DINaMo instead removes explicit step-by-step integration altogether, representing an entire short-time trajectory as a single differentiable function trained purely by minimizing residuals of Newton's equation and momentum/energy conservation on one equilibrated initial state, with no simulator-generated trajectory ever used as a label. Both papers therefore answer the same underlying question for a classical Lennard-Jones system — how closely can an approximate scheme reproduce the exact deterministic dynamics of a given initial condition — but from opposite directions: RBL keeps the integrator exact and randomizes the force, while DINaMo keeps the force exact and replaces the integrator with a learned function.

The shared force law is the finite-range, shifted-force Lennard-Jones potential with σ=ε=m=1 and cutoff r_c=2.937σ, whose energy and force vanish smoothly at the cutoff. The RBL estimator partitions each particle's neighborhood into an exactly-treated core (r ≤ r_c,core) and a stochastically sampled shell (r_c,core < r ≤ r_s), reweights the sampled shell contribution by its cardinality-to-sample-size ratio, and applies a mean-force momentum-conserving correction so the batch-averaged force stays unbiased. DINaMo instead imposes the exact initial position, velocity, and acceleration through a hard analytic ansatz built from this same Lennard-Jones force, and learns only the higher-order nonlinear departure by minimizing a causally weighted objective built entirely from Newton, momentum, and energy residuals evaluated on its own predicted path.

## Sources

The random batch list (RBL) method, its core/shell decomposition, and its momentum-conserving correction are RBMD 2.0's contribution. The differentiable Newtonian trajectory representation, hard initial-condition ansatz, physics-only training objective, and the N=50/ρ=0.15 initial-condition and reference protocol this problem's fixed parameters are built around all originate from DINaMo.

## Problem

The primary input is a fully self-contained N=50, ρ=0.15 reduced-unit Lennard-Jones argon initial condition (box length L=(N/ρ)^{1/3}): positions are placed by overlap-guarded random insertion (seed 12345, minimum separation 0.8σ) and energy-relaxed, velocities are drawn at reduced temperature Θ=2.50487911765 (seed 987654) and carried through a short thermostatted burn-in so they become dynamically consistent with the relaxed configuration; the required output is a single scalar quantifying how the two paradigms' trajectory errors compare on that shared state. Generate the exact velocity-Verlet trajectory at Δt=5×10⁻⁴τLJ over the 241-frame window T=0.12τLJ using the unapproximated pairwise force as ground truth. Independently re-integrate the same initial state on the same integrator and time grid, but evaluate the force at every step with the RBL estimator using a core cutoff of 1.5σ, a shell cutoff equal to the full interaction cutoff, and a fixed shell-sample size of 10 particles drawn each step from a single pseudo-random stream seeded with 20260911; separately, train a DINaMo-style solver (hard initial-condition ansatz, an ISRU-activated MLP correction initialized near zero, Adam optimizer, fixed epoch budget) on physics residuals alone against the identical initial state and time grid. Compute the whole-trajectory position RMSE of each solver against the exact reference using the same per-atom, minimum-image normalization convention, and report the base-10 logarithm of the ratio RMSE_RBL / RMSE_DINaMo as the final answer.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:

The tags are required. Do not omit them or leave them empty.
The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Fixed Protocol Parameters

This problem uses one specific, fully self-contained instantiation of each method's protocol below. Do not substitute a value from RBMD 2.0's or DINaMo's own published benchmark runs for any of the quantities listed here — those papers evaluate different initial conditions, system sizes, and (for DINaMo's Table A.1) architectures than the ones fixed for this problem. Every number needed to reproduce this problem's RMSE values is stated in this section; none of it needs to be recovered from either paper.

**Position relaxation ("energy-relaxed").** After overlap-guarded insertion, relax all N atoms together with the following loop, run for up to 100 iterations, each iteration proceeding in this exact order:
1. Evaluate the current net force on every atom, F (shape (N,3)), using the shared shifted-force Lennard-Jones law.
2. Let f_max be the single largest per-atom force-vector norm across all N atoms (one scalar for the whole system, not one value per atom). If f_max < 10⁻⁶ε/σ, stop; the current positions are the relaxed configuration.
3. Compute one scalar step-length s = 0.005/max(f_max, 10⁻¹²). This same scalar multiplies every atom's own force vector uniformly -- it is derived once from the single worst-offending atom's force and then applied identically to all N atoms, not recomputed per atom from that atom's own force. Consequently the atom carrying f_max moves by exactly 0.005σ and every other atom moves proportionally less, in the same step.
4. Move to new positions = current positions + s*F (wrapped into the periodic box). This move is always taken unconditionally -- there is no energy comparison, backtracking, or rejection step, and no adaptive or state-dependent step size. Each iteration is a fixed, memoryless function of the current positions alone. (This is deliberate: energy minimization of a many-body Lennard-Jones system is a non-convex, chaotically-sensitive process, and removing every discrete accept/reject decision and adaptive step-size state removes the mechanisms most likely to amplify tiny floating-point differences between execution environments into a materially different relaxed configuration.)

**Velocity burn-in ("short thermostatted burn-in").** After the Gaussian velocities (Θ=2.50487911765, seed 987654) have their center-of-mass component removed and are rescaled to match Θ exactly, integrate the relaxed configuration and these velocities forward with velocity-Verlet at the production timestep Δt=5×10⁻⁴τ_LJ for exactly 1000 steps (a burn-in window of 0.5τ_LJ), rescaling all velocities uniformly so the instantaneous kinetic temperature again equals Θ every 20 steps. After the last step, remove the center-of-mass velocity component once more and rescale a final time to Θ; the resulting positions and velocities are the shared initial condition passed to every solver.

**A note on reproducibility.** This is a chaotic many-body dynamical system (positive Lyapunov exponents), so even fully deterministic, identically-seeded code can produce measurably different numeric results across different numpy/BLAS builds or hardware, because floating-point operations are not exactly associative and this pipeline compounds many iterative steps. The relaxation and burn-in windows above are deliberately short and (for relaxation) free of any branching specifically to limit how much such differences can compound before the actual evaluation window begins; residual environment-to-environment variation in the reported RMSE values and final answer is expected and is why the tolerances on those quantities are set as wide as they are, rather than reflecting genuine uncertainty about the correct method.

**DINaMo correction network and training.** The correction network g_θ is a 2-hidden-layer MLP: Linear(1 → 64) → ISRU → Linear(64 → 64) → ISRU → Linear(64 → 3N), where ISRU(x) = x/√(1+x²) and N=50 (output width 150). Every weight and bias is drawn once from a single pseudo-random stream seeded with 0, in the order (layer-1 weight, layer-1 bias, layer-2 weight, layer-2 bias, layer-3 weight, layer-3 bias), each entry uniform on [-1/√fan_in, 1/√fan_in] for that layer's fan-in; the layer-3 (output) weight and bias are then both multiplied by 10⁻³ so the network's contribution is negligible at initialization relative to the analytic ballistic term. T in the ansatz T²s³g_θ(s) is the total prediction window n_steps·Δt (or Δt itself when n_steps=0), with s=t/T. Train with Adam (learning rate 10⁻³, β1=0.9, β2=0.999, ε=10⁻⁸) for exactly 3000 full-batch epochs, minimizing 1.0×(Newton residual) + 0.02×(momentum residual) + 0.02×(energy residual), where the energy residual is normalized by a per-atom scale of 0.1ε before being weighted and squared.

**RBL sampling and momentum correction.** For each particle, at each force evaluation: sum the exact shifted-force Lennard-Jones force from every neighbor within the core cutoff (1.5σ); if the shell (neighbors with core cutoff < distance ≤ full cutoff 2.937σ) contains 10 or fewer particles, sum all of them exactly; otherwise draw exactly 10 of them without replacement, uniformly, from the single continuing pseudo-random stream seeded with 20260911, sum their forces, and multiply that sum by (shell size)/10 before adding it to the core sum. This gives every particle a raw, per-draw force estimate. The momentum-conserving correction then subtracts the mean of these N raw per-particle force vectors (averaged over all particles in the system) from every particle's own estimate, so the corrected forces used for integration always sum to exactly zero net force on the system.

**Initial-state random streams, force law and RBL evaluation order (fully pinned).** Every random stream in this problem is `numpy.random.default_rng(seed)` (PCG64), used only with the calls stated here. *Positions:* with `rng = default_rng(12345)`, repeatedly draw a candidate `rng.random(3)*L` (uniform on [0,L)³); accept it only if its minimum-image distance to every already-accepted atom is at least 0.8σ, otherwise discard it and draw the next candidate; stop when N=50 atoms are accepted (the relaxation above then acts on these positions). *Velocities:* with `rng = default_rng(987654)`, draw once `rng.normal(0, sqrt(Θ), size=(N,3))`, subtract the mean over atoms, and rescale by sqrt(Θ/T_inst) with T_inst = 2·KE/(3N−3) and KE = ½Σ|v|² (m=1); every later rescale in the burn-in uses this same formula with 3N−3 degrees of freedom. *Force law:* for r ≤ r_c=2.937σ, F_sf(r) = F_LJ(r) − F_LJ(r_c) and U_sf(r) = U_LJ(r) − U_LJ(r_c) + (r − r_c)F_LJ(r_c), with F_LJ(r) = (24ε/r)[2(σ/r)¹² − (σ/r)⁶] and U_LJ(r) = 4ε[(σ/r)¹² − (σ/r)⁶]; both are zero for r > r_c, and all pair distances use the minimum-image convention. *RBL trajectory:* velocity-Verlet in which every force evaluation is the RBL estimate: one evaluation at the initial positions, then one at the new positions after each position update. Within an evaluation, particles are processed in index order 0,…,N−1, and a shell sample is drawn only for a particle with more than 10 shell neighbors, via `rng.choice(shell_indices, size=10, replace=False)` on the single stream `default_rng(20260911)`.

**DINaMo collocation and residuals (fully pinned).** Training uses all 241 frame times t_b = b·Δt (b = 0,…,240) at every epoch (full batch). The predicted position is r(t) = r₀ + t v₀ + ½ a₀ t² + T² s³ g_θ(s) with s = t/T, T = 240·Δt and a₀ = F_sf(r₀) (m=1). At each t_b the velocity and acceleration of this function are obtained by central finite differences in t with step h = 10⁻⁴: V = [r(t+h) − r(t−h)]/(2h) and A = [r(t+h) − 2r(t) + r(t−h)]/h². Let F and U be the shifted-force Lennard-Jones force and potential energy evaluated at r(t_b), and E = ½Σ|V|² + U. The Newton residual at t_b is (1/N)Σ_i |A_i − F_i|²; the momentum residual is |Σ_i V_i(t_b) − Σ_i V_i(t_0)|²; the energy residual is [(E(t_b) − E(t_0))/(N·0.1ε)]². The loss is the mean over the 241 frames of 1.0×(Newton) + 0.02×(momentum) + 0.02×(energy), with gradients taken through V, A and through the force and energy evaluated at r(t_b) (no stop-gradient). The predicted trajectory is r(t_b) wrapped into the periodic box.

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

lj_force_energy

Goal
----
Implement a function that computes the net force on every particle and the total potential energy of an N-particle configuration, using a finite-range, force-shifted Lennard-Jones potential evaluated under the minimum-image convention in a cubic periodic box. Pairs separated (by minimum image) by more than the cutoff contribute zero force and zero energy; pairs within the cutoff use the shifted-force form so that both energy and force go to zero continuously at r = r_c.

```python
import numpy as np
def lj_force_energy(positions: np.ndarray, box_length: float, cutoff: float,
                     epsilon: float = 1.0, sigma: float = 1.0) -> tuple:
    """
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) with N >= 2, giving the Cartesian coordinates of
        each particle. Values need not already be wrapped into the primary box.
    box_length : float
        Side length of the cubic periodic box.
    cutoff : float
        Interaction cutoff r_c for the shifted-force potential.
    epsilon : float, optional
        Lennard-Jones well depth. Default 1.0.
    sigma : float, optional
        Lennard-Jones length scale. Default 1.0.

    Returns
    -------
    forces : np.ndarray
        Array of shape (N, 3), the net shifted-force Lennard-Jones force on
        each particle.
    potential_energy : float
        Total shifted-force Lennard-Jones potential energy of the
        configuration, as a native Python float.


    Raises
    ------
    ValueError
        If `positions` does not have shape (N, 3) with N >= 2, or contains
        non-finite values; if `box_length`, `epsilon`, or `sigma` is not
        finite and > 0; or if `cutoff` is not finite, > 0, and <
        `box_length` / 2.
    """
    return np.zeros_like(positions), 0.0  # placeholder
```

### Step 2

generate_initial_condition

Goal
----
Implement a function that builds the single shared Lennard-Jones initial

condition (particle positions, velocities, and box length) used by every

downstream solver in this problem. Positions must be produced by

overlap-guarded random insertion into a cubic periodic box and then

energy-relaxed; velocities must be drawn at a target reduced temperature,

have their center-of-mass component removed, and then be carried through a

short thermostatted dynamics burn-in so they become dynamically consistent

with the relaxed configuration.

```python
import numpy as np
def generate_initial_condition(n_atoms: int, density: float, temperature: float,
                                pos_seed: int, vel_seed: int,
                                min_separation: float = 0.8, cutoff: float = 2.937,
                                epsilon: float = 1.0, sigma: float = 1.0,
                                minimize_steps: int = 100, burn_in_steps: int = 1000,
                                dt: float = 5e-4, rescale_interval: int = 20,
                                max_attempts: int = 200000) -> tuple:
    """

    Parameters
    ----------
    n_atoms : int
        Number of atoms, must be >= 2.
    density : float
        Reduced number density rho = n_atoms / box_length**3.
    temperature : float
        Target reduced temperature Theta for the Maxwell-Boltzmann velocities.
    pos_seed : int
        Seed for the position-insertion random number generator.
    vel_seed : int
        Seed for the velocity-sampling random number generator.
    min_separation : float, optional
        Minimum allowed minimum-image separation between inserted atoms, in
        units of sigma. Default 0.8.
    cutoff : float, optional
        Lennard-Jones interaction cutoff used for minimization, burn-in, and
        energy evaluation. Default 2.937.
    epsilon : float, optional
        Lennard-Jones well depth. Default 1.0.
    sigma : float, optional
        Lennard-Jones length scale. Default 1.0.
    minimize_steps : int, optional
        Number of fixed-step, always-accepted steepest-descent relaxation
        iterations (see notes below). Default 100. A value of 0 skips
        minimization entirely.
    burn_in_steps : int, optional
        Number of thermostatted velocity-Verlet burn-in steps. Default 1000.
        A value of 0 skips the burn-in entirely.
    dt : float, optional
        Burn-in integration timestep. Default 5e-4.
    rescale_interval : int, optional
        Number of burn-in steps between temperature rescalings. Default 20.
    max_attempts : int, optional
        Maximum number of candidate positions tried during insertion before
        giving up. Default 200000.

    Returns
    -------
    positions : np.ndarray
        Array of shape (n_atoms, 3), the relaxed, wrapped particle positions.
    velocities : np.ndarray
        Array of shape (n_atoms, 3), the burned-in particle velocities.
    box_length : float
        Side length of the cubic periodic box, (n_atoms / density) ** (1/3),
        as a native Python float.

    Raises
    ------
    ValueError
        If `n_atoms` is not an integer >= 2; if `density`, `temperature`,
        or `dt` is not finite and > 0; if `pos_seed` or `vel_seed` is not
        a non-negative integer; if `minimize_steps` or `burn_in_steps` is
        not a non-negative integer; if `rescale_interval` or
        `max_attempts` is not a positive integer; if `min_separation` or
        `cutoff` is not finite, > 0, and < `box_length` / 2; if `epsilon`
        or `sigma` is not finite and > 0; or if no valid configuration of
        `n_atoms` particles satisfying `min_separation` can be placed
        within `max_attempts` insertion attempts.
    """
    positions = np.zeros((n_atoms, 3))  # placeholder
    velocities = np.zeros((n_atoms, 3))  # placeholder
    box_length = (n_atoms / density) ** (1.0 / 3.0)  # placeholder
    return positions, velocities, box_length
```

### Step 3

generate_reference_trajectory

Goal
----
Implement a function that integrates the exact, unapproximated Lennard-Jones

equations of motion forward from a given initial state using velocity-Verlet,

producing the deterministic reference trajectory that every approximate

solver in this problem (the RBL stochastic-force solver and the DINaMo-style

physics-informed solver) is judged against.

```python
import numpy as np
def generate_reference_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
                                   box_length: float, n_steps: int, dt: float = 5e-4,
                                   cutoff: float = 2.937, epsilon: float = 1.0,
                                   sigma: float = 1.0) -> np.ndarray:
    """
    Parameters
    ----------
    positions0 : np.ndarray
        Array of shape (N, 3) with N >= 2, the initial particle positions.
    velocities0 : np.ndarray
        Array of shape (N, 3), matching `positions0`, the initial particle
        velocities.
    box_length : float
        Side length of the cubic periodic box.
    n_steps : int
        Number of integration steps to take, >= 0.
    dt : float, optional
        Integration timestep. Default 5e-4.
    cutoff : float, optional
        Lennard-Jones interaction cutoff. Default 2.937.
    epsilon : float, optional
        Lennard-Jones well depth. Default 1.0.
    sigma : float, optional
        Lennard-Jones length scale. Default 1.0.

    Returns
    -------
    trajectory : np.ndarray
        Array of shape (n_steps + 1, N, 3): the wrapped particle positions at
        t = 0, dt, 2*dt, ..., n_steps*dt, with `trajectory[0]` equal to
        `positions0` wrapped into `[0, box_length)`.

    Raises
    ------
    ValueError
        If `positions0` does not have shape (N, 3) with N >= 2, or
        contains non-finite values; if `velocities0` does not match
        `positions0`'s shape or contains non-finite values; if
        `box_length`, `dt`, `epsilon`, or `sigma` is not finite and > 0;
        if `n_steps` is not a non-negative integer; or if `cutoff` is not
        finite, > 0, and < `box_length` / 2.
    """
    n = positions0.shape[0]
    trajectory = np.zeros((n_steps + 1, n, 3))  # placeholder
    return trajectory
```

### Step 4

rbl_force

Goal
----
Implement a single-evaluation force estimator that reproduces RBMD 2.0's

random batch list (RBL) method: for each particle, sum the exact

Lennard-Jones force from all neighbors inside a core cutoff, add a

reweighted stochastic estimate of the force from a randomly sampled subset of

the neighbors in an outer shell, and apply the momentum-conserving

correction that removes the net spurious force the sampling would otherwise

introduce.

```python
import numpy as np

def rbl_force(positions: np.ndarray, box_length: float, core_cutoff: float,
               cutoff: float, batch_size: int, rng: np.random.Generator,
               epsilon: float = 1.0, sigma: float = 1.0) -> np.ndarray:
    """
    Parameters
    ----------
    positions : np.ndarray
        Array of shape (N, 3) with N >= 2, particle positions.
    box_length : float
        Side length of the cubic periodic box.
    core_cutoff : float
        Core cutoff r_c; neighbors within this distance are always included
        exactly.
    cutoff : float
        Full interaction cutoff r_s at which the underlying shifted-force
        Lennard-Jones law vanishes; must exceed `core_cutoff`.
    batch_size : int
        Number of shell neighbors to sample per particle when the shell is
        larger than `batch_size`.
    rng : np.random.Generator
        Random number generator (e.g. from `np.random.default_rng(seed)`)
        used for shell sampling. Its internal state is advanced by this call.
    epsilon : float, optional
        Lennard-Jones well depth. Default 1.0.
    sigma : float, optional
        Lennard-Jones length scale. Default 1.0.

    Returns
    -------
    forces : np.ndarray
        Array of shape (N, 3), the momentum-conserving RBL force estimate on
        each particle.


    Raises
    ------
    ValueError
        If `positions` does not have shape (N, 3) with N >= 2, or contains
        non-finite values; if `box_length`, `core_cutoff`, `epsilon`, or
        `sigma` is not finite and > 0; if `cutoff` is not finite, >
        `core_cutoff`, and < `box_length` / 2; if `batch_size` is not a
        positive integer; or if `rng` is not an instance of
        `np.random.Generator`.
    """
    forces = np.zeros_like(positions)  # placeholder
    return forces
```

### Step 5

generate_rbl_trajectory

Goal
----
Implement a function that integrates the Lennard-Jones equations of motion

forward from a given initial state using velocity-Verlet, exactly as in

Step 3, but replacing the exact pairwise force evaluation at every step with

RBMD 2.0's random batch list (RBL) estimator from Step 4, threading a single

continuing pseudo-random stream through the whole trajectory rather than

reseeding at each step.

```python
import numpy as np
def generate_rbl_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
                             box_length: float, n_steps: int, core_cutoff: float,
                             cutoff: float, batch_size: int, seed: int,
                             dt: float = 5e-4, epsilon: float = 1.0,
                             sigma: float = 1.0) -> np.ndarray:
    """
    Parameters
    ----------
    positions0 : np.ndarray
        Array of shape (N, 3) with N >= 2, the initial particle positions.
    velocities0 : np.ndarray
        Array of shape (N, 3), matching `positions0`, the initial particle
        velocities.
    box_length : float
        Side length of the cubic periodic box.
    n_steps : int
        Number of integration steps to take, >= 0.
    core_cutoff : float
        RBL core cutoff; neighbors within this distance are always included
        exactly at every step.
    cutoff : float
        Full interaction cutoff at which the underlying shifted-force
        Lennard-Jones law vanishes; must exceed `core_cutoff`.
    batch_size : int
        Number of shell neighbors sampled per particle per step when the
        shell is larger than `batch_size`.
    seed : int
        Seed for the single `np.random.default_rng` stream used across all
        steps.
    dt : float, optional
        Integration timestep. Default 5e-4.
    epsilon : float, optional
        Lennard-Jones well depth. Default 1.0.
    sigma : float, optional
        Lennard-Jones length scale. Default 1.0.

    Returns
    -------
    trajectory : np.ndarray
        Array of shape (n_steps + 1, N, 3): the wrapped particle positions at
        t = 0, dt, 2*dt, ..., n_steps*dt, with `trajectory[0]` equal to
        `positions0` wrapped into `[0, box_length)`.

    Raises
    ------
    ValueError
        If `positions0` does not have shape (N, 3) with N >= 2, or
        contains non-finite values; if `velocities0` does not match
        `positions0`'s shape or contains non-finite values; if
        `box_length`, `core_cutoff`, `dt`, `epsilon`, or `sigma` is not
        finite and > 0; if `n_steps` is not a non-negative integer; if
        `cutoff` is not finite, > `core_cutoff`, and < `box_length` / 2;
        if `batch_size` is not a positive integer; or if `seed` is not a
        non-negative integer.
    """
    n = positions0.shape[0]
    trajectory = np.zeros((n_steps + 1, n, 3))  # placeholder
    return trajectory
```

### Step 6

train_dinamo_trajectory

Goal
----
Implement a function that trains DINaMo's trajectory-unsupervised,

physics-informed neural solver on one initial-value problem and returns its

predicted position trajectory at the same collocation frames used by Steps 3

and 5. The predicted trajectory must be built from a hard initial-condition

ansatz (so the initial position, velocity, and acceleration are exact by

construction) plus a small neural correction, trained by minimizing only

residuals of Newton's equation, momentum conservation, and energy

conservation evaluated on the network's own predicted path -- never against

any simulator-generated position, velocity, force, or energy.

```python
import numpy as np
def train_dinamo_trajectory(positions0: np.ndarray, velocities0: np.ndarray,
                             box_length: float, n_steps: int, dt: float = 5e-4,
                             cutoff: float = 2.937, epsilon: float = 1.0,
                             sigma: float = 1.0, hidden_width: int = 64,
                             n_epochs: int = 3000, lr: float = 1e-3, seed: int = 0,
                             w_newton: float = 1.0, w_momentum: float = 0.02,
                             w_energy: float = 0.02, energy_scale: float = 0.1,
                             final_layer_scale: float = 1e-3) -> np.ndarray:
    """
    Parameters
    ----------
    positions0 : np.ndarray
        Array of shape (N, 3) with N >= 2, the initial particle positions.
    velocities0 : np.ndarray
        Array of shape (N, 3), matching `positions0`, the initial particle
        velocities.
    box_length : float
        Side length of the cubic periodic box.
    n_steps : int
        Number of collocation intervals; the trajectory is returned at the
        n_steps + 1 frames t_b = b*dt for b = 0, ..., n_steps. Must be >= 0.
    dt : float, optional
        Spacing between collocation frames. Default 5e-4.
    cutoff : float, optional
        Lennard-Jones interaction cutoff. Default 2.937.
    epsilon : float, optional
        Lennard-Jones well depth. Default 1.0.
    sigma : float, optional
        Lennard-Jones length scale. Default 1.0.
    hidden_width : int, optional
        Width of each hidden layer of the correction network. Default 64.
    n_epochs : int, optional
        Number of full-batch Adam training epochs, >= 0. A value of 0 returns
        the untrained ansatz. Default 3000.
    lr : float, optional
        Adam learning rate. Default 1e-3.
    seed : int, optional
        Seed for `np.random.default_rng`, controlling the network's initial
        weights. Default 0.
    w_newton, w_momentum, w_energy : float, optional
        Non-negative weights on the three residual terms. Defaults 1.0, 0.02,
        0.02.
    energy_scale : float, optional
        Per-atom energy normalization scale for the energy residual. Default
        0.1.
    final_layer_scale : float, optional
        Factor multiplying the correction network's final layer weights and
        bias at initialization, >= 0. Default 1e-3.

    Returns
    -------
    trajectory : np.ndarray
        Array of shape (n_steps + 1, N, 3): the predicted positions at t_b =
        b*dt for b = 0, ..., n_steps, wrapped into the primary periodic box
        `[0, box_length)`.

    Raises
    ------
    ValueError
        If `positions0` does not have shape (N, 3) with N >= 2, or
        contains non-finite values; if `velocities0` does not match
        `positions0`'s shape or contains non-finite values; if
        `box_length`, `dt`, `lr`, or `energy_scale` is not finite and > 0;
        if `n_steps` or `n_epochs` is not a non-negative integer; if
        `cutoff` is not finite, > 0, and < `box_length` / 2; if `epsilon`
        or `sigma` is not finite and > 0; if `hidden_width` is not a
        positive integer; if `seed` is not a non-negative integer; if
        `w_newton`, `w_momentum`, or `w_energy` is not finite and >= 0; or
        if `final_layer_scale` is not finite and >= 0.
"""
    n = positions0.shape[0]
    trajectory = np.zeros((n_steps + 1, n, 3))  # placeholder
    return trajectory
```

### Step 7

trajectory_rmse

Goal
----
Implement a function that computes the whole-trajectory position root-mean-

square error between a predicted trajectory and the exact reference

trajectory, using the minimum-image displacement convention and per-atom,

per-frame normalization. This single metric function is applied to both

approximate solvers in this problem: the RBL trajectory of Step 5 and the

DINaMo trajectory of Step 6, each compared against the exact reference

trajectory of Step 3.

```python
import numpy as np
def trajectory_rmse(predicted_trajectory: np.ndarray, reference_trajectory: np.ndarray,
                     box_length: float) -> float:
    """
    Parameters
    ----------
    predicted_trajectory : np.ndarray
        Array of shape (B, N, 3) with B >= 1 and N >= 2, the predicted
        particle positions at each stored frame.
    reference_trajectory : np.ndarray
        Array of the same shape as `predicted_trajectory`, the reference
        particle positions at each stored frame.
    box_length : float
        Side length of the cubic periodic box.

    Returns
    -------
    rmse : float
        The whole-trajectory position RMSE, as a native Python float.

    Raises
    ------
    ValueError
        If `predicted_trajectory` does not have shape (B, N, 3) with
        B >= 1 and N >= 2, or contains non-finite values; if
        `reference_trajectory` does not match `predicted_trajectory`'s
        shape or contains non-finite values; or if `box_length` is not
        finite and > 0.
    """
    return 0.0  # placeholder
```

### Step 8

orchestrator

Goal
----
Implement the function that ties every earlier step together into the single

scalar this problem asks for: build the shared Lennard-Jones initial

condition, generate the exact reference trajectory, independently generate

the RBL-approximated trajectory and train the DINaMo-style physics-only

trajectory, compute each approximate solver's whole-trajectory position RMSE

against the exact reference, and return the base-10 logarithm of the ratio

RMSE_RBL / RMSE_DINaMo.

```python
import numpy as np
def orchestrator(n_atoms: int = 50, density: float = 0.15,
                  temperature: float = 2.50487911765, pos_seed: int = 12345,
                  vel_seed: int = 987654, n_steps: int = 240, dt: float = 5e-4,
                  cutoff: float = 2.937, core_cutoff: float = 1.5,
                  batch_size: int = 10, rbl_seed: int = 20260911,
                  hidden_width: int = 64, n_epochs: int = 3000, lr: float = 1e-3,
                  dinamo_seed: int = 0, epsilon: float = 1.0, sigma: float = 1.0,
                  min_separation: float = 0.8, minimize_steps: int = 100,
                  burn_in_steps: int = 1000, rescale_interval: int = 20) -> float:
    """Orchestrate the full RBL-vs-DINaMo comparison and return log10 of the
    RMSE ratio.
    ...

    Raises
    ------
    ValueError
        If any parameter is invalid for the step it configures (see
        `generate_initial_condition`, `generate_reference_trajectory`,
        `generate_rbl_trajectory`, and `train_dinamo_trajectory` for the
        exact conditions on each shared parameter), or if the computed
        RMSE_DINaMo is 0 (e.g. because `n_steps` is 0), making
        log10(RMSE_RBL / RMSE_DINaMo) undefined.
    """
    return 0.0  # placeholder
```

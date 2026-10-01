# Chemistry-Computational_Chemistry-5

## Background

Heterogeneous catalysts work at temperatures and coverages where the surface itself is in motion and where many adsorbates share the same facet, so the picture of one reactant crossing one static barrier on an otherwise bare surface breaks down. Molecular dynamics driven by machine-learned interatomic potentials can now follow such systems directly, which shifts the problem from enumerating elementary steps to extracting kinetics from unbiased trajectory data. Markov state models and the transition-path machinery built on them answer that need: they coarse-grain trajectories into discrete states, identify which collective motions are slow, and turn the sampled dynamics into committors and rate constants without the reactive event having to be specified beforehand. The rate constants obtained this way are the quantities that can be compared against transition state theory estimates and against measured catalytic activity, and the differences between them are where the cooperative and structural effects that static models miss show up.

## Problem

I have a 2000 ps coarse-grained trajectory of one tagged hydrogen atom adsorbed on a rhodium row, and I want the rate constant in inverse picoseconds for that atom leaving the separated-atom state and arriving in the state where it is bound into an adsorbed hydrogen molecule. Its configuration is `(r1, r2, s)`: the distances in angstrom to its two hydrogen neighbours, and its lateral coordinate in angstrom along a row of period `L = 8.10` angstrom whose site at `s = 0` is an under-coordinated edge and whose sites at `L/3` and `2L/3` are terrace sites. Writing `b(r) = 0.040 * exp(-(r - 1.22)^2 / (2 * 0.30^2))` and `e(s) = 0.5 * (1 + cos(2*pi*s/L))`, the free energy in eV is

```
U(r1, r2, s) = sum over i in {1, 2} of [ 0.20 * (0.55 / r_i)^6
                                         + 0.5 * 0.12 * (r_i - 2.60)^2
                                         - 0.25 * exp(-(r_i - 0.80)^2 / (2 * 0.25^2))
                                         + b(r_i) * (1 + 3.0 * e(s)) ]
               + 0.020 * (1 - cos(6*pi*s/L))
               + 0.045 * (1 - cos(2*pi*s/L))
               + 0.06 * exp(-((r1 - 0.80)^2 + (r2 - 0.80)^2) / (2 * 0.25^2))
```

These are the settings my trajectory was produced with, and the analysis conventions I have already fixed. For the remaining endpoint and estimator rules, use the published atom-centred local-environment Markov-state protocol for hydrogen association and dissociation on rhodium nanoparticles together with the exact variational transition-rate formulation for practical finite lags.

- Overdamped Langevin at `kT = 0.0387780` eV, one Euler-Maruyama step per recorded frame, frame spacing `dt = 0.01` ps, 200000 frames, each coordinate moving with a mobility equal to its diffusion constant divided by `kT`, which is 0.5 angstrom^2/ps for `r1` and `r2` and 2.0 angstrom^2/ps for `s`.
- Start at `(2.60, 2.60, 0.0)`, recorded as frame 0; increments drawn once as `rng.standard_normal((200000, 3))` from `np.random.default_rng(20260212)`, row `t` advancing frame `t` to frame `t + 1`; `s` reduced modulo `L` after every step and `r1`, `r2` left unwrapped.
- Per-frame local environment: Rh atoms sit at lateral positions `m*L + {0, L/3, 2L/3}` for cell indices `m = -2 .. 2`, the atom at offset `0` lying 0.30 angstrom above the plane of the other two and the hydrogen 1.00 angstrom above that plane. Every neighbour distance `d` enters through `f(d) = 0.5 * (1 + cos(pi*d/7.0))` for `d < 7.0` and `0` beyond, and through seven Gaussian bins centred at `0.5, 1.5, ..., 6.5` angstrom with Gaussian standard deviation `sigma = 0.5` angstrom. The 21 descriptors are, per bin: that cutoff-weighted Gaussian summed over the Rh images; the same summed over `r1` and `r2`; and `f(r1) * f(r2)` times the Gaussian evaluated at `(r1 + r2)/2`.
- Slow-coordinate projection, fitted on and applied to every frame of the trajectory: mean-free descriptors, instantaneous covariance `Y^T Y / T`, symmetrised lagged correlation at 20 frames, whitening on the covariance eigenvectors keeping eigenvalues above `1e-6` times the largest, retaining the five components with largest lagged eigenvalues, ordered by decreasing lagged eigenvalue and each signed so the largest-magnitude entry of its descriptor-space loading is positive.
- Clustering of the frames belonging to neither endpoint state: k-means++ with `rng = np.random.default_rng(0)`, first centre at index `rng.integers(n)`, each later centre at index `rng.choice(n, p=w)` with `w` the running squared distance to the nearest chosen centre normalized to sum to one, then Lloyd iterations with lowest-index tie-breaking, empty centres left in place, stopping when an assignment repeats or after 100 iterations.
- A lag of 20 frames wherever an estimator takes one, time averages over all 200000 frames, and any time-window average taken centred over an odd number of frames with the window truncated symmetrically at the two ends of the trajectory.

Report one final rate constant in inverse picoseconds. In the supporting calculation, state: the fraction of frames assigned to each of the three states; the complete endpoint-state test applied to each frame, including any time smoothing; the number of retained slow coordinates and the number of microstates in the transition region; the estimators used for the committor and reactive flux, and how the two committor directions are related; the range of the committor over the transition-region frames; the trajectory average of the backward committor; the net reactive flux in inverse picoseconds; how many completed passages the trajectory makes from the atomic endpoint to the molecular endpoint, counted between successive visits to either endpoint while ignoring intervening transition-region frames, and what that implies for a rate obtained by counting those passages; and what the raised saddle at the edge site implies for the reaction there. As a finite-lag estimator check, hold the 20-frame slow-coordinate projection and its 64-microstate partition fixed, repeat only the stopped committor and reactive-current estimators at lags of 10 and 40 frames, and report each signed percentage change in the final rate relative to the 20-frame result.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 10 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_simulate_surface_trajectory

Goal
----
Step name: 01_simulate_surface_trajectory
Step description: Integrate overdamped Langevin dynamics of one tagged adsorbed hydrogen atom on the reduced free-energy surface and return its frame-by-frame coordinates.

```python
import numpy as np
def simulate_surface_trajectory(n_frames: int, seed: int, dt_ps: float = 0.01,
                                d_bond: float = 0.5, d_lateral: float = 2.0,
                                kt_ev: float = 0.0387780) -> np.ndarray:
    """Propagate the tagged hydrogen atom and record one row per frame.

    The state is ``(r1, r2, s)``: the distances in angstrom from the tagged
    hydrogen to its two hydrogen neighbours, and its lateral coordinate in
    angstrom along a Rh row of period ``L = 8.10``. With
    ``b(r) = 0.040 * exp(-(r - 1.22) ** 2 / (2 * 0.30 ** 2))`` and
    ``e(s) = 0.5 * (1 + cos(2 * pi * s / L))`` the free energy in eV is

        U = sum_i [ 0.20 * (0.55 / r_i) ** 6
                    + 0.5 * 0.12 * (r_i - 2.60) ** 2
                    - 0.25 * exp(-(r_i - 0.80) ** 2 / (2 * 0.25 ** 2))
                    + b(r_i) * (1 + 3.0 * e(s)) ]
            + 0.020 * (1 - cos(6 * pi * s / L))
            + 0.045 * (1 - cos(2 * pi * s / L))
            + 0.06 * exp(-((r1 - 0.80) ** 2 + (r2 - 0.80) ** 2)
                         / (2 * 0.25 ** 2))

    so that ``s = 0`` is the under-coordinated edge site, where the saddle of
    the separation coordinate is raised fourfold.

    Conventions fixed by this step. The trajectory starts from
    ``(2.60, 2.60, 0.0)``, which is recorded as frame 0. The Gaussian
    increments are drawn once as ``rng.standard_normal((n_frames, 3))`` from
    ``np.random.default_rng(seed)``, and row ``t`` of that array advances the
    state from frame ``t`` to frame ``t + 1``. Each coordinate takes one
    Euler-Maruyama step of duration ``dt_ps``, with a mobility equal to that
    coordinate's diffusion constant divided by ``kt_ev``; ``r1`` and ``r2`` use
    ``d_bond`` and ``s`` uses ``d_lateral``. After every step ``s`` is reduced
    modulo ``L``, while ``r1`` and ``r2`` are left unwrapped.

    Parameters
    ----------
    n_frames : int
        Number of recorded frames, at least 2.
    seed : int
        Seed of the increment generator.
    dt_ps : float
        Frame spacing in picoseconds, positive.
    d_bond : float
        Diffusion constant of each hydrogen-hydrogen distance in angstrom^2
        per picosecond, positive.
    d_lateral : float
        Diffusion constant of the lateral coordinate in angstrom^2 per
        picosecond, positive.
    kt_ev : float
        Thermal energy in eV, positive.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_frames, 3)`` whose rows are ``(r1, r2, s)``.

    Raises
    ------
    ValueError
        If ``n_frames`` is not an integer of at least 2, if ``seed`` is not an
        integer, if any of ``dt_ps``, ``d_bond``, ``d_lateral`` or ``kt_ev`` is
        not a positive finite float, or if the integration drives a
        hydrogen-hydrogen distance to a non-positive value.
    """
    return trajectory

# EXPECTED RETURN
#
```

### Step 2

02_build_local_descriptors

Goal
----
Step name: 02_build_local_descriptors
Step description: Convert every trajectory frame into an adsorbate-centred descriptor vector built from smoothly truncated radial basis functions over the tagged hydrogen's neighbours.

```python
import numpy as np
def build_local_descriptors(trajectory: np.ndarray, n_radial: int = 7,
                            r_cut: float = 7.0, n_images: int = 2) -> np.ndarray:
    """Return the adsorbate-centred descriptor matrix of a trajectory.

    The rhodium row has period ``L = 8.10`` with three surface atoms per
    period at lateral offsets ``0``, ``L / 3`` and ``2 * L / 3``. The atom at
    offset ``0`` is the under-coordinated edge atom and sits ``0.30``
    angstrom above the plane of the other two; the tagged hydrogen sits
    ``1.00`` angstrom above that plane. Rh images are taken for cell indices
    ``-n_images`` through ``+n_images``, so the distance from the hydrogen at
    lateral coordinate ``s`` to an image at lateral position ``p`` and height
    ``z`` is ``sqrt((1.00 - z) ** 2 + (s - p) ** 2)``.

    Every neighbour distance ``d`` enters through the cosine cutoff
    ``f(d) = 0.5 * (1 + cos(pi * d / r_cut))`` for ``d < r_cut`` and ``0``
    otherwise, and through ``n_radial`` Gaussian bins centred at
    ``mu_k = (k + 0.5) * r_cut / n_radial`` with common width
    ``sigma = r_cut / (2 * n_radial)``. Bin ``k`` of the three channels is

        Rh channel : sum over Rh images of f(d) * exp(-(d - mu_k) ** 2
                     / (2 * sigma ** 2))
        H channel  : sum over r1 and r2 of f(r) * exp(-(r - mu_k) ** 2
                     / (2 * sigma ** 2))
        HH channel : f(r1) * f(r2) * exp(-(0.5 * (r1 + r2) - mu_k) ** 2
                     / (2 * sigma ** 2))

    The output columns are ordered Rh channel first, then H channel, then HH
    channel, each in increasing bin index.

    Parameters
    ----------
    trajectory : np.ndarray
        Float array of shape ``(T, 3)`` with rows ``(r1, r2, s)`` and ``T``
        at least 1.
    n_radial : int
        Number of radial bins per channel, at least 1.
    r_cut : float
        Interaction cutoff in angstrom, positive.
    n_images : int
        Number of periodic cell images taken on each side, at least 0.

    Returns
    -------
    np.ndarray
        Float array of shape ``(T, 3 * n_radial)``.

    Raises
    ------
    ValueError
        If ``trajectory`` is not a non-empty real array of shape ``(T, 3)``,
        if either hydrogen distance is not positive, if ``n_radial`` is not an
        integer of at least 1, if ``n_images`` is not an integer of at least
        0, or if ``r_cut`` is not a positive finite float.
    """
    return descriptors
```

### Step 3

03_label_reaction_endpoints

Goal
----
Step name: 03_label_reaction_endpoints
Step description: Assign every frame to the atomic reactant state, the molecular product state or the intermediate region using time-smoothed hydrogen-hydrogen distances.

```python
import numpy as np
def label_reaction_endpoints(trajectory: np.ndarray, window_frames: int = 5,
                             r_atomic: float = 1.5, r_molecular: float = 1.0) -> np.ndarray:
    """Return the endpoint label of every frame of a trajectory.

    Each of the two hydrogen-hydrogen distances is first replaced by a centred
    moving average over ``window_frames`` frames; near the two ends of the
    trajectory the window is truncated symmetrically, so frame ``t`` averages
    the frames from ``max(t - h, 0)`` to ``min(t + h, T - 1)`` inclusive with
    ``h = window_frames // 2``. Labels are then assigned from the smoothed
    distances alone.

    Labels are ``0`` for the atomic reactant state ``A``, ``1`` for the
    molecular product state ``B``, and ``2`` for the intermediate region that
    belongs to neither. A frame whose smoothed distances to both neighbours
    are below ``r_atomic`` is intermediate whatever else holds; otherwise the
    frame is molecular when its smallest smoothed distance is below
    ``r_molecular``, atomic when that distance exceeds ``r_atomic``, and
    intermediate in between.

    Parameters
    ----------
    trajectory : np.ndarray
        Float array of shape ``(T, 3)`` with rows ``(r1, r2, s)`` and ``T``
        at least 1.
    window_frames : int
        Odd positive number of frames in the centred smoothing window.
    r_atomic : float
        Smoothed separation in angstrom above which a frame is atomic.
    r_molecular : float
        Smoothed separation in angstrom below which a frame is molecular.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}``.

    Raises
    ------
    ValueError
        If ``trajectory`` is not a finite non-empty array of shape ``(T, 3)``,
        if ``window_frames`` is not an odd integer of at least 1, if either
        radius is not a positive finite float, or if ``r_molecular`` is not
        strictly smaller than ``r_atomic``.
    """
    return labels
```

### Step 4

04_project_dynamical_components

Goal
----
Step name: 04_project_dynamical_components
Step description: Reduce the descriptor matrix to its slowest linear components by solving the time-lagged generalized eigenvalue problem.

```python
import numpy as np
def project_dynamical_components(descriptors: np.ndarray, lag_frames: int = 20,
                                 n_components: int = 5,
                                 rank_cutoff: float = 1e-6) -> np.ndarray:
    """Return the slowest linear components of a descriptor trajectory.

    Conventions fixed by this step. The descriptors are first made mean free
    over the whole trajectory. The instantaneous covariance is
    ``C0 = Y.T @ Y / T`` and the lagged correlation is the symmetrised
    ``Ct = (Yt.T @ Yl + Yl.T @ Yt) / (2 * (T - lag_frames))`` where ``Yt`` and
    ``Yl`` are the first and last ``T - lag_frames`` rows of ``Y``.
    Whitening is done on the eigendecomposition of ``C0``, keeping only the
    eigenvalues larger than ``rank_cutoff`` times the largest one and scaling
    each retained eigenvector by the inverse square root of its eigenvalue.
    The requested number of components is taken in order of decreasing
    eigenvalue of the whitened lagged matrix. Each returned component is
    signed so that the entry of largest absolute value in its descriptor-space
    loading vector is positive; on a tie the earliest such entry decides.

    Parameters
    ----------
    descriptors : np.ndarray
        Float array of shape ``(T, d)``.
    lag_frames : int
        Correlation lag in frames, at least 1 and smaller than ``T``.
    n_components : int
        Number of components returned, at least 1.
    rank_cutoff : float
        Relative eigenvalue floor for the whitening, strictly between 0 and 1.

    Returns
    -------
    np.ndarray
        Float array of shape ``(T, n_components)`` holding the projected
        trajectory, each column with unit variance up to the whitening.

    Raises
    ------
    ValueError
        If ``descriptors`` is not a finite two-dimensional array, if
        ``lag_frames`` is not an integer in ``1 .. T - 1``, if
        ``n_components`` is not an integer of at least 1 or exceeds the
        retained rank, or if ``rank_cutoff`` is not a float strictly between
        0 and 1.
    """
    return components
```

### Step 5

05_assign_intermediate_microstates

Goal
----
Step name: 05_assign_intermediate_microstates
Step description: Cluster the projected coordinates of the frames outside both endpoint states into discrete microstates that will serve as the Galerkin basis.

```python
import numpy as np
def assign_intermediate_microstates(components: np.ndarray, labels: np.ndarray,
                                    n_microstates: int = 64, seed: int = 0,
                                    max_iter: int = 100) -> np.ndarray:
    """Return the microstate index of every intermediate frame.

    Only the rows of ``components`` whose label is ``2`` are clustered; every
    other frame receives ``-1``.

    Conventions fixed by this step, applied to the intermediate rows in
    trajectory order. A generator ``np.random.default_rng(seed)`` is created
    once. The first centre is the row at index ``rng.integers(n)``. Each later
    centre is the row at index ``rng.choice(n, p=w)`` where ``w`` is the
    running squared distance to the nearest chosen centre divided by its own
    sum. Lloyd iterations then alternate assignment and recentring: a row is
    assigned to the centre of smallest squared Euclidean distance with the
    lowest index winning a tie, a centre with no rows keeps its previous
    position, and the loop stops as soon as an assignment repeats the previous
    one or after ``max_iter`` iterations.

    Parameters
    ----------
    components : np.ndarray
        Float array of shape ``(T, d)`` of projected coordinates.
    labels : np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}``.
    n_microstates : int
        Number of microstates, at least 1 and at most the number of
        intermediate frames.
    seed : int
        Seed of the seeding generator.
    max_iter : int
        Maximum number of Lloyd iterations, at least 1.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(T,)`` holding a microstate index in
        ``0 .. n_microstates - 1`` for intermediate frames and ``-1``
        elsewhere.

    Raises
    ------
    ValueError
        If ``components`` is not a finite two-dimensional array, if ``labels``
        is not an integer array of matching length with values in
        ``{0, 1, 2}``, if ``n_microstates`` is not an integer of at least 1 or
        exceeds the number of intermediate frames, if ``seed`` is not an
        integer, or if ``max_iter`` is not an integer of at least 1.
    """
    return microstates
```

### Step 6

06_compute_stopping_times

Goal
----
Step name: 06_compute_stopping_times
Step description: Tabulate, for every frame, the first later frame and the last earlier frame at which the trajectory occupies one of the two endpoint states.

```python
import numpy as np
def compute_stopping_times(labels: np.ndarray) -> np.ndarray:
    """Return the forward and backward endpoint entry times of every frame.

    A frame is an endpoint frame when its label is ``0`` or ``1``. For frame
    ``t`` of a trajectory of ``T`` frames, column 0 of the output holds the
    smallest index ``t' >= t`` that is an endpoint frame and column 1 holds
    the largest index ``t' <= t`` that is an endpoint frame. An endpoint frame
    is therefore its own entry time in both columns.

    Conventions fixed by this step. When no endpoint frame exists at or after
    ``t`` the forward entry is ``T``; when none exists at or before ``t`` the
    backward entry is ``-1``. Both sentinels are chosen so that clamping a
    forward entry from above, or a backward entry from below, by any index
    inside the trajectory leaves that index unchanged.

    Parameters
    ----------
    labels : np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}`` and ``T``
        at least 1.

    Returns
    -------
    np.ndarray
        Integer array of shape ``(T, 2)`` whose columns are the forward and
        backward endpoint entry times.

    Raises
    ------
    ValueError
        If ``labels`` is not a non-empty one-dimensional integer array whose
        values all lie in ``{0, 1, 2}``.
    """
    return stopping
```

### Step 7

07_build_galerkin_system

Goal
----
Step name: 07_build_galerkin_system
Step description: Accumulate the linear system whose solution gives the expansion coefficients of the committor in the basis of intermediate-microstate indicator functions.

```python
import numpy as np
def build_galerkin_system(labels: np.ndarray, microstates: np.ndarray,
                          stopping: np.ndarray, lag_frames: int = 20,
                          n_microstates: int = 64) -> np.ndarray:
    """Return the augmented Galerkin system of the halted committor problem.

    The trial basis is the set of indicator functions of the intermediate
    microstates, and the reference function is the indicator of the molecular
    product state. Entry ``(i, j)`` of the returned matrix, for ``j`` below
    ``n_microstates``, is the trajectory estimate of the correlation of
    indicator ``i`` with the change of indicator ``j`` produced by the halted
    dynamics over ``lag_frames``; the final column is the same estimate taken
    against the product-state indicator instead of a basis indicator.

    Conventions fixed by this step. Windows are indexed by their first frame
    ``t`` running from ``0`` to ``T - lag_frames - 1`` inclusive, so there are
    ``T - lag_frames`` of them. Every window contributes twice, once read
    forwards from ``t`` and once read backwards from ``t + lag_frames``, and
    the accumulated total is divided by twice the number of windows. A frame
    that is not intermediate carries no basis indicator and so contributes no
    row. The halting instants must be taken from ``stopping``, whose two
    columns are the forward and backward endpoint entry times, and must be
    clamped to the window being read.

    Parameters
    ----------
    labels : np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}``.
    microstates : np.ndarray
        Integer array of shape ``(T,)`` holding a microstate index in
        ``0 .. n_microstates - 1`` on intermediate frames and ``-1``
        elsewhere.
    stopping : np.ndarray
        Integer array of shape ``(T, 2)`` of forward and backward endpoint
        entry times.
    lag_frames : int
        Lag of the halted dynamics in frames, at least 1 and smaller than
        ``T``.
    n_microstates : int
        Size of the basis, at least 1.

    Returns
    -------
    np.ndarray
        Float array of shape ``(n_microstates, n_microstates + 1)`` holding
        the Galerkin matrix in its leading columns and the reference vector
        in its final column.

    Raises
    ------
    ValueError
        If ``labels``, ``microstates`` and ``stopping`` are not integer arrays
        of matching length and the stated shapes, if a label lies outside
        ``{0, 1, 2}``, if an intermediate frame carries a microstate index
        outside ``0 .. n_microstates - 1`` or a non-intermediate frame carries
        one that is not ``-1``, if ``lag_frames`` is not an integer in
        ``1 .. T - 1``, or if ``n_microstates`` is not an integer of at least
        1.
    """
    return system
```

### Step 8

08_solve_committor

Goal
----
Step name: 08_solve_committor
Step description: Solve the Galerkin system for the basis coefficients and expand them into a committor value on every frame of the trajectory.

```python
import numpy as np
def solve_committor(labels: np.ndarray, microstates: np.ndarray,
                    system: np.ndarray) -> np.ndarray:
    """Return the forward committor of every frame of the trajectory.

    The leading square block of ``system`` is the Galerkin matrix and its
    final column is the reference vector of the same projected problem, as
    produced by the previous step. The coefficient vector solves that system
    so that the expansion reproduces the projected boundary value problem;
    the reference column enters as an inhomogeneity, not as an extra unknown.

    Conventions fixed by this step. The committor is exactly ``0.0`` on every
    atomic frame and exactly ``1.0`` on every molecular frame, and an
    intermediate frame takes the coefficient of its own microstate, so the
    returned values are not clipped to the unit interval.

    Parameters
    ----------
    labels : np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}``.
    microstates : np.ndarray
        Integer array of shape ``(T,)`` holding a microstate index on
        intermediate frames and ``-1`` elsewhere.
    system : np.ndarray
        Float array of shape ``(n, n + 1)``.

    Returns
    -------
    np.ndarray
        Float array of shape ``(T,)`` of committor values.

    Raises
    ------
    ValueError
        If ``labels`` and ``microstates`` are not integer arrays of equal
        length with the stated value ranges, if ``system`` is not a finite
        float array of shape ``(n, n + 1)`` with ``n`` at least 1, if an
        intermediate frame carries a microstate index outside ``0 .. n - 1``,
        or if the Galerkin matrix is singular.
    """
    return committor
```

### Step 9

09_estimate_reactive_flux

Goal
----
Step name: 09_estimate_reactive_flux
Step description: Accumulate the net reactive flux per frame from the committor and the endpoint entry times over lagged trajectory windows.

```python
import numpy as np
def estimate_reactive_flux(labels: np.ndarray, committor: np.ndarray,
                           stopping: np.ndarray, lag_frames: int = 20,
                           block_frames: int = 20000) -> float:
    """Return the net reactive flux from the atomic to the molecular state.

    The estimate is the average, over all lagged windows of the trajectory, of
    the flux carried by the single-frame increments inside a window, with the
    committor of each increment's endpoints evaluated where the halted
    dynamics stop. The backward committor is obtained from the forward one
    under detailed balance rather than estimated separately.

    Conventions fixed by this step. Windows are indexed by their first frame
    ``t`` running from ``0`` to ``T - lag_frames - 1`` inclusive, and every
    window contributes the increments from its own frames ``t`` to
    ``t + lag_frames - 1``, each increment running to the next frame. Halting
    instants come from the two columns of ``stopping`` and are clamped to the
    window that contains them, so no increment ever refers to a frame outside
    its own window. The two time directions are averaged with equal weight,
    the sum over increments inside a window is divided by ``lag_frames``, and
    the sum over windows is divided by the number of windows, so the result is
    a flux per frame and not per window. ``block_frames`` only sets how many
    windows are accumulated at a time and must not change the result beyond
    floating-point summation order.

    Parameters
    ----------
    labels : np.ndarray
        Integer array of shape ``(T,)`` with values in ``{0, 1, 2}``.
    committor : np.ndarray
        Float array of shape ``(T,)`` of forward committor values.
    stopping : np.ndarray
        Integer array of shape ``(T, 2)`` of forward and backward endpoint
        entry times.
    lag_frames : int
        Lag of the halted dynamics in frames, at least 1 and smaller than
        ``T``.
    block_frames : int
        Number of windows accumulated per block, at least 1.

    Returns
    -------
    float
        Net reactive flux per frame.

    Raises
    ------
    ValueError
        If the three arrays do not have matching length and the stated shapes
        and dtypes, if a label lies outside ``{0, 1, 2}``, if ``committor`` is
        not finite, if ``lag_frames`` is not an integer in ``1 .. T - 1``, or
        if ``block_frames`` is not an integer of at least 1.
    """
    return flux
```

### Step 10

10_compute_association_rate

Goal
----
Step name: 10_compute_association_rate
Step description: Compose every earlier step to obtain the association rate constant of the tagged adsorbed hydrogen atom in inverse picoseconds.

```python
import numpy as np
def compute_association_rate(n_frames: int = 200000, seed: int = 20260212,
                             dt_ps: float = 0.01, lag_frames: int = 20,
                             n_components: int = 5, n_microstates: int = 64,
                             cluster_seed: int = 0) -> float:
    """Return the rate constant for atomic hydrogen combining into a molecule.

    Simulate the tagged hydrogen atom for ``n_frames`` frames from ``seed``,
    build its adsorbate-centred descriptors with the default radial basis,
    label its endpoint states with the default smoothing window and radii,
    project the descriptors onto ``n_components`` slow coordinates at a lag of
    ``lag_frames``, cluster the intermediate frames into ``n_microstates``
    microstates from ``cluster_seed``, tabulate the endpoint entry times,
    solve the projected committor problem at a lag of ``lag_frames`` and
    integrate the reactive current at the same lag.

    Conventions fixed by this step. The rate is the net reactive flux per
    frame divided by the trajectory average of the backward committor over
    every frame of the trajectory and by ``dt_ps``, so it is reported in
    inverse picoseconds. The defaults reproduce the problem statement.

    Parameters
    ----------
    n_frames : int
        Number of recorded frames, at least 2.
    seed : int
        Seed of the trajectory increments.
    dt_ps : float
        Frame spacing in picoseconds, positive.
    lag_frames : int
        Lag in frames used for the projection, the committor problem and the
        reactive current, at least 1 and smaller than ``n_frames``.
    n_components : int
        Number of slow coordinates retained, at least 1.
    n_microstates : int
        Number of intermediate microstates, at least 1.
    cluster_seed : int
        Seed of the microstate seeding generator.

    Returns
    -------
    float
        Association rate constant in inverse picoseconds.

    Raises
    ------
    ValueError
        If any argument violates the contract of the step that consumes it,
        or if the trajectory average of the backward committor is not
        positive.
    """
    return rate
```

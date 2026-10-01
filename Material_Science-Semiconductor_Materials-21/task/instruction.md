# Material_Science-Semiconductor_Materials-21

## Background

Thermal management is now one of the binding constraints on semiconductor device design. In most semiconductors and insulators heat is carried predominantly by phonons, but in nanoscale transistors, power electronics and thermoelectric elements the electron population is driven far from the lattice temperature, and the energy that flows from electrons into phonons through electron–phonon scattering sets both the peak junction temperature and the size of the hot spot. The characteristic electron–phonon energy-relaxation length and time are frequently comparable to the mean free paths of the carriers themselves, so the two subsystems neither equilibrate locally nor transport energy diffusively. Macroscopic two-temperature models, which assume local equilibrium within each population, therefore break down, while atomistic methods such as molecular dynamics or Green–Kubo evaluation are confined to domains far smaller than a device. The mesoscopic middle ground is a pair of coupled Boltzmann transport equations for the electron and phonon distribution functions, closed with relaxation-time approximations whose rates can be supplied by first-principles calculations, and coupled to one another by energy-conserving electron–phonon scattering terms.

Solving that coupled kinetic system is expensive. The unknowns live in a six-dimensional phase space, and the scattering terms are stiff: the equilibrium distributions depend on pseudo-temperatures that are themselves angular integrals of the unknowns. Deterministic solvers therefore decouple the integrals with an inner iteration solve the transport equations for the distribution functions with the temperatures frozen, integrate the result to get new temperatures, repeat. This source-iteration loop, familiar from neutron and radiation transport, has a well-known pathology: information travels only one mean free path per iteration, so as any relaxation process becomes diffusive the number of inner iterations explodes, and the accompanying numerical dissipation degrades accuracy at the same time. Because a device simulation must resolve regimes from ballistic to diffusive simultaneously, and often within a single geometry, a solver that is only efficient at large Knudsen number is of limited practical use.

A synthetic iterative scheme addresses this by solving, at every inner step, a small set of macroscopic moment equations alongside the kinetic equations. The moment equations for the electron and phonon energy densities and heat fluxes are obtained exactly by taking the zeroth and first angular moments of the kinetic equations, but they are not closed: the flux equations involve a higher-order stress-like moment of the unknown distributions, and the treatment of that term is what decides whether the resulting solver stays accurate across the whole range of Knudsen numbers or degenerates into a diffusion approximation. The macroscopic solution in turn supplies the equilibrium pseudo-temperatures and a correction to the distribution functions for the next kinetic sweep, so information passes between the kinetic and macroscopic levels in both directions at every inner step. Whether such a construction actually converges quickly, and whether it does so uniformly across regimes rather than only where the conventional loop was already adequate, is a question about the linearized inner iteration rather than about any particular simulation: it can be settled analytically, without a mesh and without running a solver, by asking how a single plane-wave error mode is damped in one iteration. Because a solver is only as good as its slowest regime, the figure of merit for a multiscale scheme is not its rate at any one operating point but the worst rate it attains anywhere in the parameter range it claims to cover. The practical stakes are large: reported speed-ups reach three orders of magnitude in production simulations, precisely in the multiscale regimes where the conventional loop stalls.

## Problem

In a semiconductor or metallic film driven out of equilibrium, heat is carried by electrons and by phonons, and the two populations exchange energy through electron–phonon scattering, so their Boltzmann transport equations must be solved together. In the linearised gray relaxation-time form the coupled system is stiff, and the usual inner loop — sweep the two kinetic equations, then rebuild the equilibrium terms from the angular moments of the swept distributions — stalls whenever any of the three scattering Knudsen numbers, electron–phonon $\mathtt{Kn}_{\mathrm{e\text{-}p}}$, phonon–electron $\mathtt{Kn}_{\mathrm{p\text{-}e}}$ or phonon–phonon $\mathtt{Kn}_{\mathrm{p\text{-}p}}$, becomes small, because the scattering source then returns almost the whole of the previous iterate. A synthetically accelerated inner loop breaks that stall by solving, after each kinetic sweep, a closed set of macroscopic energy and heat-flux moment equations whose flux terms carry the exact first-order Chapman–Enskog diffusion limit of the coupled system plus higher-order closure terms evaluated from the freshly swept distributions, and by taking the equilibrium terms of the next sweep from the macroscopic energies so obtained rather than from the kinetic angular integrals.

Characterise the convergence of one such accelerated inner iteration by Fourier stability analysis. Treat the difference between two consecutive inner iterates of the electron and phonon intensity perturbation functions as the error, expand it in a single spatial Fourier mode of unit wave-vector magnitude in two dimensions, and reduce one iteration to a linear map acting on the paired electron and phonon energy-error amplitudes; its convergence rate is the spectral radius of that map. Work throughout in dimensionless variables with both heat capacities and both carrier group speeds equal to unity, and evaluate every solid-angle integral exactly over the full unit sphere rather than by numerical quadrature, so that the reported value is deterministic. The wave vector lies in the plane of the two-dimensional spatial expansion while the carrier direction vector ranges over the full unit sphere; the two are independent, and the direction enters only through its projection onto the wave vector.

Sweep the equal-Knudsen line $\mathtt{Kn}_{\mathrm{e\text{-}p}} = \mathtt{Kn}_{\mathrm{p\text{-}e}} = \mathtt{Kn}_{\mathrm{p\text{-}p}} = \mathtt{Kn}$ over the 41 logarithmically spaced values running from $10^{-2}$ to $10^{2}$ inclusive, and report the largest convergence rate the accelerated scheme reaches anywhere on that sweep. Your reasoning should also report, as evidence that the analysis ran: the Knudsen number at which that largest rate occurs, and, evaluated there, the effective phonon relaxation scale, the energy-exchange parameters, the flux coefficients of the macroscopic balances, the solid-angle integrals each carrier contributes, the coefficients through which the two energy errors drive the scattering sources, the entries of the unaccelerated amplification matrix, the entries of the closed macroscopic operator, the full eigenvalue pair of the accelerated map, and the unaccelerated inner loop's rate; then both schemes' rates at the two ends of the sweep, and what the comparison at the ballistic end says about where the acceleration does and does not pay for itself. Those quantities are the few derived scalars that determine the final number, so quoting them is what the output requirements below call for; what those requirements exclude is restating supplied input data and bulk per-iteration or per-candidate tables, of which this problem has none.

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

01_compute_system_coefficients

Goal
----
Derive every scalar coefficient of the coupled electron-phonon system that the rest of the stability analysis consumes.

```python
import numpy as np
def compute_system_coefficients(kn_ep: float, kn_pe: float, kn_pp: float,
                                C_e: float = 1.0, C_p: float = 1.0,
                                v_e: float = 1.0, v_p: float = 1.0) -> np.ndarray:
    """Derive the scalar coefficients of the coupled electron-phonon system.

    Parameters
    ----------
    kn_ep, kn_pe, kn_pp : float
        The three scattering Knudsen numbers, finite and strictly positive.
    C_e, C_p, v_e, v_p : float
        Electron and phonon heat capacities and group speeds, positive.

    Returns
    -------
    coefficients : np.ndarray
        Shape (11,), in this order: the effective phonon transport Knudsen
        number; the two energy-exchange rates of the macroscopic energy
        balances, electron row then phonon row; the four flux coefficients of
        those balances written in the energy densities, ordered ee, ep, pe, pp;
        and the four scattering-source coefficients in the same order.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return np.zeros(11, dtype=float)  # placeholder
```

### Step 2

02_compute_angular_moments

Goal
----
Evaluate the two solid-angle integrals of the swept-error factor that both iteration operators are assembled from.

```python
import numpy as np
def compute_angular_moments(u: float) -> np.ndarray:
    """Evaluate the two solid-angle integrals of the swept-error factor.

    Parameters
    ----------
    u : float
        Product of the relevant relaxation Knudsen number and group speed,
        finite and non-negative. Zero is admissible.

    Returns
    -------
    moments : np.ndarray
        Shape (2,): the plain solid-angle integral of the swept-error factor
        over the full unit sphere, then the integral of the same factor
        weighted by the squared projection of the direction vector onto the
        unit wave vector. Both are real.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return np.zeros(2, dtype=float)  # placeholder
```

### Step 3

03_build_unaccelerated_matrix

Goal
----
Assemble the two-by-two error amplification matrix of the unaccelerated inner loop.

```python
import numpy as np
def build_unaccelerated_matrix(coefficients: np.ndarray, moments_e: np.ndarray,
                               moments_p: np.ndarray) -> np.ndarray:
    """Assemble the amplification matrix of the unaccelerated inner loop.

    Parameters
    ----------
    coefficients : np.ndarray
        Shape (11,) as returned by ``compute_system_coefficients``.
    moments_e, moments_p : np.ndarray
        Shape (2,) each, as returned by ``compute_angular_moments`` at the
        electron and at the phonon argument respectively.

    Returns
    -------
    unaccelerated : np.ndarray
        Real array of shape (2, 2) mapping the electron and phonon
        energy-density error amplitudes of one iterate onto those of the next,
        electron row first.

    Raises
    ------
    ValueError
        If ``coefficients`` is not a finite array of shape (11,), or if
        ``moments_e`` or ``moments_p`` is not a finite array of shape (2,).

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return np.zeros((2, 2), dtype=float)  # placeholder
```

### Step 4

04_build_accelerated_operators

Goal
----
Assemble the pair of two-by-two operators the accelerated inner iteration acts with, one on the new energy amplitudes and one on the old.

```python
import numpy as np
def build_accelerated_operators(coefficients: np.ndarray, moments_e: np.ndarray,
                                moments_p: np.ndarray, unaccelerated: np.ndarray,
                                kn_ep: float, v_e: float = 1.0,
                                v_p: float = 1.0) -> np.ndarray:
    """Assemble the two operators of the accelerated inner iteration.

    Parameters
    ----------
    coefficients : np.ndarray
        Shape (11,) as returned by ``compute_system_coefficients``.
    moments_e, moments_p : np.ndarray
        Shape (2,) each, from ``compute_angular_moments``, electron then phonon.
    unaccelerated : np.ndarray
        Shape (2, 2) from ``build_unaccelerated_matrix``.
    kn_ep, v_e, v_p : float
        Electron-phonon Knudsen number and the two group speeds, all positive.

    Returns
    -------
    operators : np.ndarray
        Real shape (2, 2, 2): element 0 acts on the new energy amplitudes and
        element 1 on the old ones, at unit wave-vector magnitude.

    Raises
    ------
    ValueError
        If an array input has the wrong shape or contains a non-finite value,
        or if ``kn_ep``, ``v_e`` or ``v_p`` is non-finite or not strictly
        positive.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return np.zeros((2, 2, 2), dtype=float)  # placeholder
```

### Step 5

05_compute_spectral_radii

Goal
----
Read the convergence rate of each scheme off the spectral radius of its error amplification map.

```python
import numpy as np
def compute_spectral_radii(unaccelerated: np.ndarray,
                           operators: np.ndarray) -> np.ndarray:
    """Return the convergence rate of each scheme as a spectral radius.

    Parameters
    ----------
    unaccelerated : np.ndarray
        Shape (2, 2) as returned by ``build_unaccelerated_matrix``.
    operators : np.ndarray
        Shape (2, 2, 2) as returned by ``build_accelerated_operators``, the
        operator on the new amplitudes first and non-singular.

    Returns
    -------
    rates : np.ndarray
        Shape (2,): the unaccelerated convergence rate followed by the
        accelerated one, each the largest eigenvalue modulus of its map.

    Notes
    -----
    Include every import your implementation needs inside the function body.
    """
    return np.zeros(2, dtype=float)  # placeholder
```

### Step 6

06_compute_convergence_rates

Goal
----
Chain the sub-problem functions 01-05 to return both schemes' convergence rates at one Knudsen-number triple.

```python
import numpy as np
def compute_convergence_rates(kn_ep: float, kn_pe: float, kn_pp: float,
                              C_e: float = 1.0, C_p: float = 1.0,
                              v_e: float = 1.0, v_p: float = 1.0) -> np.ndarray:
    """Return both schemes' convergence rates at one Knudsen-number triple.

    Parameters
    ----------
    kn_ep, kn_pe, kn_pp : float
        The three scattering Knudsen numbers, finite and strictly positive.
    C_e, C_p, v_e, v_p : float
        Electron and phonon heat capacities and group speeds, positive.

    Returns
    -------
    rates : np.ndarray
        Shape (2,): the unaccelerated convergence rate followed by the
        accelerated one, at unit perturbation wave-vector magnitude.

    Notes
    -----
    This is an orchestrating step: call the public functions of sub-problems
    01-05, feeding each returned value into the next. Import inside the body.
    """
    return np.zeros(2, dtype=float)  # placeholder
```

### Step 7

07_run_stability_sweep

Goal
----
Sweep the equal-Knudsen line through every transport regime and return the worst convergence rate the accelerated scheme attains on it.

```python
import numpy as np
def run_stability_sweep(kn_min: float = 1e-2, kn_max: float = 1e2,
                        n_points: int = 41, C_e: float = 1.0, C_p: float = 1.0,
                        v_e: float = 1.0, v_p: float = 1.0) -> float:
    """Return the worst accelerated convergence rate on the equal-Knudsen line.

    Parameters
    ----------
    kn_min, kn_max : float
        Inclusive ends of the logarithmic sweep, finite and strictly positive
        with kn_min <= kn_max.
    n_points : int
        Number of logarithmically spaced sweep points, at least 2.
    C_e, C_p, v_e, v_p : float
        Electron and phonon heat capacities and group speeds, positive.

    Returns
    -------
    worst_rate : float
        The largest accelerated-scheme convergence rate over the sweep, as a
        native Python float.

    Notes
    -----
    This is the final, orchestrating step: call the public function of sub-problem
    06 (``compute_convergence_rates``) at each sweep point. Import inside the body.
    """
    return 0.0  # placeholder
```

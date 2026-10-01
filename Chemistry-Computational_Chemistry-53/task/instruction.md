# Chemistry-Computational_Chemistry-53

## Background

Many molecules possess two nearly equivalent configurations separated by a low barrier: proton-transfer tautomers, internal rotors, inverting groups and conformers related by a hydroxy or methyl torsion. Quantum tunnelling through the barrier couples the vibrational ground states of the two configurations and splits or shifts their energy levels. These splittings are resolved by high-resolution microwave, infrared and Raman spectroscopy and are among the most sensitive probes of barrier shapes and tunnelling pathways in molecules.

When the two configurations are related by symmetry, the two wells are identical and the lowest pair of levels is split by tunnelling alone. When they are not, for example when one configuration is stabilized by a weak intramolecular interaction or by an asymmetric environment, the wells differ both in depth and in the curvature that sets their zero-point energies. The two lowest states are then only partly delocalized, and the measured splitting reflects the competition between this asymmetry and the tunnelling coupling.

Exact quantum-mechanical calculations of tunnelling splittings become unaffordable beyond a few degrees of freedom. Semiclassical instanton theory, formulated in terms of imaginary-time path integrals discretized as ring polymers, describes the dominant tunnelling pathway directly in full dimensionality at the cost of an optimization of the discretized path, and it becomes accurate as the barrier grows high compared with the vibrational quanta of the wells.

## Problem

When a molecule can tunnel between two nearly equivalent configurations whose potential minima have different depths, its two lowest vibrational levels behave like a two-state system: one state localized in each well, offset in energy by the asymmetry of the wells and coupled by a tunnelling matrix element $\hbar\Omega$, called the tunnelling frequency. The observable level splitting combines the asymmetry with $\hbar\Omega$, so predicting it requires the tunnelling frequency itself. Ring-polymer instanton theory obtains tunnelling splittings of symmetric double wells from a single optimized discretized imaginary-time path at modest cost. A 2026 study extended ring-polymer instanton theory so that it applies to such asymmetric double wells. Given a potential energy surface, the particle mass and a ring-polymer discretization of imaginary time, it returns the tunnelling frequency $\hbar\Omega$.

Consider one particle of mass $m = 1$ moving in one dimension, in reduced units with $\hbar = 1$, on the lower adiabatic potential of a two-state diabatic model,

$$V(x) = \tfrac{1}{2}\left[V_{00}(x) + V_{11}(x)\right] - \tfrac{1}{2}\sqrt{\left[V_{00}(x) - V_{11}(x)\right]^2 + 4V_{01}^2}, \qquad V_{00}(x) = \tfrac{1}{2}\omega_\ell^2 (x + x_0)^2 - \varepsilon, \qquad V_{11}(x) = \tfrac{1}{2}\omega_r^2 (x - x_0)^2,$$

with $\omega_\ell = 1.0$, $\omega_r = 0.3$, $x_0 = 4.6$, $\varepsilon = -0.19$ and $V_{01} = 1.5$.

Compute $\hbar\Omega$ for this particle with that extended ring-polymer instanton formulation, applied exactly as its authors formulate and evaluate it, representing imaginary time by a ring polymer of $N = 4096$ beads at inverse temperature $\beta = 300$ (reduced units).

In your reasoning, report the values your result rests on, in any order: the positions and potential energies of the stationary points of the potential and the harmonic frequencies of its wells; the end points of the optimized ring polymer; and the value of every term and factor of the final expression you evaluate for $\hbar\Omega$.

Give the final answer as $\hbar\Omega$ in reduced units. Between the <final_answer> tags place only that value, written as one finite decimal number with at least four significant figures (for example 0.001234 or 1.234e-3), with no units, symbols, words, equations or any other text.

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

two_diabat_potential

Goal
----
Implement two_diabat_potential, which evaluates the lower adiabatic potential energy surface of a
two-state diabatic model, together with its first and second derivatives, at a set of positions.

```python
def two_diabat_potential(x: "np.ndarray", params: "np.ndarray") -> "np.ndarray":
    '''Lower adiabatic potential of a two-state diabatic model and its x-derivatives.

    Parameters
    ----------
    x : np.ndarray
        Positions, a scalar or a 1-D array of n finite values.
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01], five finite values. The
        diabatic potentials are V00(x) = omega_l**2 (x + x0)**2 / 2 - eps and
        V11(x) = omega_r**2 (x - x0)**2 / 2, they are coupled by the constant V01, and the
        lower adiabatic surface is
        V(x) = (V00 + V11) / 2 - sqrt((V00 - V11)**2 + 4 V01**2) / 2.
        omega_l, omega_r and V01 must be positive and x0 must be nonzero.

    Returns
    -------
    values : np.ndarray
        Array of shape (3, n): row 0 holds V(x), row 1 dV/dx and row 2 d2V/dx2, one column
        per position in the order of x (n = 1 for a scalar x).

    Raises
    ------
    ValueError
        If params does not hold exactly five finite values, omega_l, omega_r or V01 is not
        positive, x0 is zero, or x contains a non-finite value.
    '''
    return values
```

### Step 2

locate_stationary_points

Goal
----
Implement locate_stationary_points, which finds the two minima and the barrier top of the lower
adiabatic surface of the two-state diabatic model, together with the harmonic frequencies of the
two wells.

```python
def locate_stationary_points(params: "np.ndarray") -> "np.ndarray":
    '''Minima, barrier top and well frequencies of the lower adiabatic surface.

    Parameters
    ----------
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.

    Returns
    -------
    points : np.ndarray
        Array of 8 floats [x_low, x_top, x_high, V_low, V_top, V_high, omega_low,
        omega_high]. x_low and x_high are the positions of the lower (deeper) and the higher
        minimum, and x_top is the position of the barrier maximum between them, each accurate
        to 1e-10. V_low, V_top and V_high are the potential energies at these three points,
        and omega_low and omega_high are the harmonic frequencies sqrt(V''/m) of the two wells
        at their minima (m = 1). When the surface is mirror-symmetric (omega_l == omega_r and
        eps == 0), the two minima are degenerate, and the minimum at negative x is labelled
        low.

    Raises
    ------
    ValueError
        If params is invalid (as in two_diabat_potential) or the surface does not have
        exactly two minima.
    '''
    return points
```

### Step 3

half_ring_action

Goal
----
Implement half_ring_action, which evaluates the discretized imaginary-time action of a folded
ring polymer on the two-state surface, together with its gradient and Hessian with respect to
the bead positions.

```python
def half_ring_action(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int) -> tuple:
    '''Action of the folded half ring polymer and its first and second bead derivatives.

    Parameters
    ----------
    beads : np.ndarray
        Positions y_0, ..., y_{N/2} of the N/2 + 1 beads of the folded half ring. The closed
        N-bead ring polymer is x_j = y_j for 0 <= j <= N/2 and x_j = y_{N-j} for
        N/2 < j < N, so the end beads y_0 and y_{N/2} (the turning points) appear once on the
        ring and every other bead appears twice.
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.
    beta : float
        Inverse temperature in reduced units (hbar = 1); neighbouring ring beads are
        separated by the imaginary-time step beta*hbar/N.
    n_beads : int
        Number of beads N of the full ring polymer, an even integer >= 4.

    Returns
    -------
    result : tuple
        (S_half, gradient, hessian_diag, hessian_offdiag):
        S_half : float, half of the standard discretized Euclidean action S_N of the closed
            N-bead ring polymer (harmonic springs between neighbouring beads, potential energy
            at every bead, particle mass m = 1, hbar = 1), as a function of the half-ring beads.
        gradient : np.ndarray, shape (N/2 + 1,), dS_half/dy_j.
        hessian_diag : np.ndarray, shape (N/2 + 1,), d2S_half/dy_j^2.
        hessian_offdiag : np.ndarray, shape (N/2,), d2S_half/(dy_j dy_{j+1}); all other
            Hessian elements vanish.

    Raises
    ------
    ValueError
        If n_beads is not an even integer >= 4, beta is not positive, beads does not hold
        N/2 + 1 finite values, or params is invalid.
    '''
    return (S_half, gradient, hessian_diag, hessian_offdiag)
```

### Step 4

optimize_instanton

Goal
----
Implement optimize_instanton, which returns the ring-polymer instanton of the two-state surface:
the folded half ring polymer at which the discretized imaginary-time action is stationary, with
exactly one unstable direction.

```python
def optimize_instanton(params: "np.ndarray", beta: float, n_beads: int) -> "np.ndarray":
    '''Folded half-ring instanton of the lower adiabatic surface.

    Parameters
    ----------
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.
    beta : float
        Inverse temperature in reduced units (hbar = 1).
    n_beads : int
        Number of beads N of the full ring polymer, an even integer >= 4.

    Returns
    -------
    beads : np.ndarray
        Half-ring bead positions y_0, ..., y_{N/2}, shape (N/2 + 1,), in the folded convention
        of half_ring_action. They form a stationary point of S_half with respect to all
        N/2 + 1 beads (the end beads are free), converged to max_j |dS_half/dy_j| < 1e-10, at
        which the Hessian of S_half has exactly one negative eigenvalue and the path crosses
        the barrier: y_0 is the end on the side of the lower minimum of
        locate_stationary_points and y_{N/2} the end on the side of the higher minimum. When
        the surface is mirror-symmetric (omega_l == omega_r and eps == 0), the returned
        stationary point is the mirror-symmetric one, y_{N/2-j} = -y_j for every j.

    Raises
    ------
    ValueError
        If params is invalid (as in two_diabat_potential), n_beads is not an even integer
        >= 4, beta is not positive, the surface does not have exactly two minima, or no such
        stationary point is found.
    '''
    return beads
```

### Step 5

select_dividing_surface

Goal
----
Implement select_dividing_surface, which places the dividing surface on a folded ring-polymer
path and returns how the closed ring polymer, and with it the imaginary time, is shared between
the two wells, together with the imaginary-time speed of the path at the dividing surface.

```python
def select_dividing_surface(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int) -> "np.ndarray":
    '''Dividing-surface bead, partition of the ring between the wells and path speed there.

    Parameters
    ----------
    beads : np.ndarray
        Half-ring bead positions y_0, ..., y_{N/2} in the folded convention of
        half_ring_action, with y_0 on the lower-well side (normally the instanton of
        optimize_instanton). All quantities are evaluated for the beads as supplied.
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.
    beta : float
        Inverse temperature in reduced units (hbar = 1).
    n_beads : int
        Number of beads N of the full ring polymer, an even integer >= 4.

    Returns
    -------
    surface : np.ndarray
        Array of 7 floats [k, x_sigma, n_low, n_high, tau_low, tau_high, qdot_sigma].
        k is the index of the dividing-surface bead: the interior half-ring bead
        (1 <= k <= N/2 - 1) with the largest potential energy V(y_k), the smallest such
        index in case of a tie. x_sigma = y_k. On the closed ring the dividing surface is
        formed by the two copies x_k and x_{N-k} of this bead. n_low is the number of ring
        segments (springs) on the arc between them that contains y_0, and n_high = N - n_low is
        the number on the other arc. tau_low and tau_high are the imaginary times spanned by
        the two arcs (hbar = 1). qdot_sigma is the magnitude of the imaginary-time velocity of
        the path at the dividing surface, evaluated by the central difference across bead k.

    Raises
    ------
    ValueError
        If n_beads is not an even integer >= 4, beta is not positive, beads does not hold
        N/2 + 1 finite values, or params is invalid.
    '''
    return surface
```

### Step 6

fluctuation_factor

Goal
----
Implement fluctuation_factor, which evaluates the pinned fluctuation factor of a ring-polymer
instanton: the ratio of the fluctuation determinant of the ring polymer pinned at its dividing
surface to those of the two wells, each well represented by the part of the ring assigned to it.

```python
def fluctuation_factor(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int, k: int) -> "np.ndarray":
    '''Pinned fluctuation factor Phi_pin of a ring-polymer path and its log-determinants.

    Parameters
    ----------
    beads : np.ndarray
        Half-ring bead positions y_0, ..., y_{N/2} in the folded convention of
        half_ring_action, with y_0 on the lower-well side (normally the instanton of
        optimize_instanton).
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.
    beta : float
        Inverse temperature in reduced units (hbar = 1).
    n_beads : int
        Number of beads N of the full ring polymer, an even integer >= 4.
    k : int
        Index of the dividing-surface bead of the half ring, 1 <= k <= N/2 - 1 (normally the
        first entry of select_dividing_surface).

    Returns
    -------
    factors : np.ndarray
        Array of 4 floats [logdet_J_pin, logdet_J_low, logdet_J_high, Phi_pin]; the first
        three are natural logarithms of determinants. With dtau = beta*hbar/N and m = 1, J
        is (dtau/m) times the Hessian of the full-ring action S_N with respect to the N ring bead
        positions x_j, evaluated at the ring built from beads. J_pin is J with the rows and
        columns of the two dividing-surface beads x_k and x_{N-k} removed. J_low (J_high) is
        the same matrix for a ring of n_low (n_high) beads with the same dtau and every bead at
        the lower (higher) minimum of locate_stationary_points, where n_low and n_high are the
        arc lengths of select_dividing_surface for this k. Finally,
        Phi_pin = sqrt(dtau/m) * [det J_pin / (det J_low * det J_high)]**(1/4).

    Raises
    ------
    ValueError
        If k is not an integer in 1..N/2 - 1, n_beads is not an even integer >= 4, beta is
        not positive, beads does not hold N/2 + 1 finite values, params is invalid, or the
        surface does not have exactly two minima.
    '''
    return factors
```

### Step 7

instanton_exponent

Goal
----
Implement instanton_exponent, which evaluates the exponent of the instanton tunnelling frequency:
the action of the folded ring-polymer path measured relative to the two wells over the imaginary
times assigned to them.

```python
def instanton_exponent(beads: "np.ndarray", params: "np.ndarray", beta: float, n_beads: int, k: int) -> float:
    '''Exponent S_inst - tau_low V_low / 2 - tau_high V_high / 2 of the tunnelling frequency.

    Parameters
    ----------
    beads : np.ndarray
        Half-ring bead positions y_0, ..., y_{N/2} in the folded convention of
        half_ring_action, with y_0 on the lower-well side (normally the instanton of
        optimize_instanton).
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.
    beta : float
        Inverse temperature in reduced units (hbar = 1).
    n_beads : int
        Number of beads N of the full ring polymer, an even integer >= 4.
    k : int
        Index of the dividing-surface bead of the half ring, 1 <= k <= N/2 - 1 (normally the
        first entry of select_dividing_surface).

    Returns
    -------
    exponent : float
        S_inst - tau_low * V_low / 2 - tau_high * V_high / 2 in units of hbar (hbar = 1).
        S_inst is S_half of half_ring_action at the given beads, V_low and V_high are the
        energies of the lower and the higher minimum of locate_stationary_points, and tau_low
        and tau_high are the imaginary times of the two arcs of select_dividing_surface for
        this k.

    Raises
    ------
    ValueError
        If k is not an integer in 1..N/2 - 1, n_beads is not an even integer >= 4, beta is
        not positive, beads does not hold N/2 + 1 finite values, params is invalid, or the
        surface does not have exactly two minima.
    '''
    return exponent
```

### Step 8

tunnelling_frequency

Goal
----
Implement tunnelling_frequency, the end-to-end pipeline, which returns the tunnelling frequency
hbar*Omega between the two wells of the lower adiabatic surface of the two-state model from
ring-polymer instanton theory for asymmetric wells, at a given inverse temperature and
ring-polymer size.

```python
def tunnelling_frequency(params: "np.ndarray", beta: float, n_beads: int) -> float:
    '''Ring-polymer instanton tunnelling frequency hbar*Omega for asymmetric wells.

    Parameters
    ----------
    params : np.ndarray
        Model parameters [omega_l, omega_r, x0, eps, V01] of the surface V(x) of
        two_diabat_potential.
    beta : float
        Inverse temperature in reduced units (hbar = 1).
    n_beads : int
        Number of beads N of the full ring polymer, an even integer >= 4.

    Returns
    -------
    hbar_omega : float
        hbar*Omega = hbar * qdot_sigma / (Phi_pin * sqrt(2*pi*hbar)) * exp(-exponent/hbar)
        with hbar = 1, where the instanton comes from optimize_instanton, qdot_sigma and its
        dividing-surface bead k from select_dividing_surface, Phi_pin from fluctuation_factor
        and the exponent from instanton_exponent, all at this k.

    Raises
    ------
    ValueError
        If params is invalid (as in two_diabat_potential), n_beads is not an even integer
        >= 4, beta is not positive, the surface does not have exactly two minima, or the
        instanton is not found.
    '''
    return hbar_omega
```

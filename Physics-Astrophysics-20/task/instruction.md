# Physics-Astrophysics-20

## Background

Relativistically hot electron-positron plasmas, as found in pulsar wind nebulae, relativistic
jets and radio lobes, lose energy by synchrotron emission. In a high-beta plasma this cooling
drives the pressure anisotropic, and the anisotropy soon reaches the kinetic firehose
threshold, beyond which small-scale fluctuations grow and scatter particles in pitch angle.

The task concerns a one-dimensional periodic slab, with the magnetic field perpendicular to
the direction of variation, that starts in total-pressure balance with a modulated density and
field. All quantities are dimensionless in the units of the reference state defined in the
prompt. The reported quantity is one dimensionless number measuring, at a single late time,
how far the pressure anisotropy lies below the firehose threshold across the slab.

## Problem

A collisionless, ultra-relativistically hot electron-positron plasma with equal species densities and temperatures cools by synchrotron emission in a strong magnetic field, and the pressure anisotropy this produces excites kinetic firehose fluctuations wherever it exceeds the firehose threshold. Consider a periodic one-dimensional slab with field B(x,t) along z, bulk flow along x, gyrotropic pressures P_perp and P_par summed over both species and no heat flux, in the limit in which compressive signals cross the slab in a vanishing fraction of the cooling time, and report the volume-averaged squared shortfall of the anisotropy below the firehose threshold at a late time.

Let T0 = theta0 m_e c^2 with theta0 = 1000 be the reference temperature, beta0 = 8 pi n0 T0/B0^2 = 80 the reference total-pair beta, Omega0 = e B0 c/(3 T0) and tau0 = m_e^2 c^3/(2 r_e^2 B0^2 T0), with tau0 Omega0 = 2000. At t = 0 the slab is at rest and isotropic, with n/n0 = B/B0 = b = 1 - 0.28 cos(2 pi s) + 0.05 cos(4 pi s + pi/3) at fractional position s = x/L along the slab period L, and P_perp = P_par = n0 T0 [1 + (1 - b^2)/beta0]. Without radiation or scattering, a fluid element compressed perpendicular to the field keeps P_perp proportional to n^(8/5) and P_par proportional to n^(4/5). The radiation-reaction losses of both pressures are closed by two constants, alpha_perp = n integral p_perp^4 gamma^-2 f d^3p/(6 m_e^2 P_perp^2) and alpha_par = n integral p_perp^2 p_par^2 gamma^-2 f d^3p/(3 m_e^2 P_perp P_par). They equal 16/15 and 8/15, respectively, before t_d = 0.17 tau0 and 0.9 and 0.48, respectively, everywhere afterwards.

From t_d on, wherever P_par - P_perp exceeds 1.4 B^2/(8 pi), the firehose fluctuations relax the two pressures toward each other at fixed 2 P_perp + P_par, with P_par - P_perp decaying at rate 3 nubar, where nubar = tau_e^(-1/3) Omega_e^(2/3) and tau_e and Omega_e are the formulas for tau0 and Omega0 evaluated at the local B and the local T = (2 P_perp + P_par)/(3 n). There is no scattering before t_d or below the threshold, and the on-off switch is taken in its zero-width limit.

At t = 8 tau0, compute Q = integral from 0 to 1 of [max(0, 1 - (P_par - P_perp)/(1.4 B^2/(8 pi)))]^2 ds, converged in resolution.

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

radiative_pressure_sinks

Goal
----
Synchrotron radiation-reaction sinks of the two pressures.

```python
def radiative_pressure_sinks(
    n: ArrayLike,
    B: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    alpha_perp: float,
    alpha_par: float,
    beta0: float,
    theta0: float,
    chi: float,
) -> np.ndarray:
    """Return the synchrotron radiation-reaction sinks of the two pressures.

    Parameters
    ----------
    n : float or array_like
        Total (electron plus positron) number density in units of n0.
        Strictly positive.
    B : float or array_like
        Magnetic field strength in units of sqrt(4 pi n0 m_e c^2). Strictly
        positive.
    p_perp : float or array_like
        Total perpendicular pressure P_perp in units of n0 m_e c^2. Strictly
        positive.
    p_par : float or array_like
        Total parallel pressure P_par in units of n0 m_e c^2. Strictly
        positive.
    alpha_perp : float
        Closure constant
        alpha_perp = n int p_perp^4 gamma^-2 f d^3p / (6 m_e^2 P_perp^2),
        where inside the integral p_perp is the particle momentum component
        perpendicular to B, gamma the particle Lorentz factor and f the
        momentum distribution; it equals 16/15 for an isotropic
        ultra-relativistic thermal plasma. Non-negative.
    alpha_par : float
        Closure constant
        alpha_par = n int p_perp^2 p_par^2 gamma^-2 f d^3p / (3 m_e^2 P_perp P_par),
        with p_par the particle momentum component along B; it equals 8/15 for
        an isotropic ultra-relativistic thermal plasma. Non-negative.
    beta0 : float
        Reference plasma beta 8 pi P0 / B0^2. Strictly positive.
    theta0 : float
        Reference temperature T0 / (m_e c^2). Strictly positive.
    chi : float
        Reference product tau0 Omega0 of the synchrotron cooling time and the
        relativistic gyrofrequency. Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2,) + S``, with ``S`` the broadcast shape of ``n``,
        ``B``, ``p_perp`` and ``p_par``. Row 0 is dP_perp/dt and row 1 is
        dP_par/dt caused by radiation reaction alone, in units of
        n0 m_e c^2 Omega0.

    Raises
    ------
    ValueError
        If ``n``, ``B``, ``p_perp`` or ``p_par`` has a non-finite or
        non-positive entry, if these four do not broadcast together, if
        ``alpha_perp`` or ``alpha_par`` is negative, non-finite or not a
        scalar, or if ``beta0``, ``theta0`` or ``chi`` is not a finite,
        strictly positive scalar.
    """
    return sinks  # placeholder
```

### Step 2

regulated_scattering_rate

Goal
----
Firehose-regulated pitch-angle scattering.

```python
def regulated_scattering_rate(
    n: ArrayLike,
    B: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    du_dx: ArrayLike,
    t: float,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
) -> np.ndarray:
    """Return the effective pitch-angle scattering rate nu of every cell.

    Parameters
    ----------
    n : float or array_like
        Number density in units of n0. Strictly positive.
    B : float or array_like
        Magnetic field strength in units of sqrt(4 pi n0 m_e c^2). Strictly
        positive.
    p_perp : float or array_like
        Perpendicular pressure in units of n0 m_e c^2. Strictly positive.
    p_par : float or array_like
        Parallel pressure in units of n0 m_e c^2. Strictly positive.
    du_dx : float or array_like
        Local gradient of the bulk velocity along x (units Omega0), i.e. the
        rate of perpendicular expansion felt by the fluid element (negative in
        compression). Finite.
    t : float
        Time (units 1/Omega0) at which the schedule is evaluated.
    beta0 : float
        Reference plasma beta (step 01). Strictly positive.
    theta0 : float
        Reference temperature T0 / (m_e c^2) (step 01). Strictly positive.
    chi : float
        Reference product tau0 Omega0 (step 01). Strictly positive.
    t_d : float
        Onset time of scattering (units 1/Omega0).
    c_th : float, optional
        Firehose threshold constant. Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``S`` (broadcast shape of ``n``, ``B``, ``p_perp``,
        ``p_par`` and ``du_dx``) holding nu in units of Omega0. For t < t_d
        every entry is 0. For t >= t_d, with D = P_par - P_perp and
        D_th = c_th B^2/2: nu = nubar where D > D_th (1 + 1e-10); nu = 0 where
        D < D_th (1 - 1e-10); otherwise (on the threshold) nu is the rate at
        which D - D_th of the fluid element, whose velocity gradient is
        ``du_dx``, stays constant in time under the full local model of the
        task, clipped to [0, nubar].

    Raises
    ------
    ValueError
        If ``n``, ``B``, ``p_perp`` or ``p_par`` has a non-finite or
        non-positive entry, if ``du_dx`` has a non-finite entry, if the five
        arrays do not broadcast together, if ``t`` or ``t_d`` is not a finite
        scalar, or if ``beta0``, ``theta0``, ``chi`` or ``c_th`` is not a
        finite, strictly positive scalar.
    """
    return rate  # placeholder
```

### Step 3

advance_uniform_cells

Goal
----
Exact local evolution of the two pressures in uniform cells.

```python
def advance_uniform_cells(
    n: ArrayLike,
    B: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    t_start: float,
    t_end: float,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
) -> np.ndarray:
    """Return the two pressures of uniform cells at rest after local evolution.

    Each cell keeps its density ``n`` and field ``B``; only its pressures
    evolve, by the radiative sinks of step 01 plus the regulated scattering of
    step 02 evaluated with ``du_dx = 0``, under the schedule of step 02. The
    coefficient switch happens exactly at ``t_d``: the part of
    [t_start, t_end] before t_d follows the pre-onset rules, the part at or
    after t_d the post-onset rules. The zero-width switch of step 02 is
    followed exactly: a cell on the threshold stays on it while the step-02
    on-threshold rate lies strictly between 0 and nubar; it leaves the
    threshold without scattering once that rate falls to 0, and with the full
    rate nubar once the rate would exceed nubar. A cell whose on-threshold rate
    is exactly 0 evolves without scattering; one whose rate equals nubar
    evolves with the full rate. A cell that reaches the threshold from either
    side continues by the same rule. Input states within
    1e-12 (2 P_perp + P_par + c_th B^2/2) of the threshold count as on it.

    Parameters
    ----------
    n : float or array_like
        Number density of each cell (units n0), constant in time. Strictly
        positive.
    B : float or array_like
        Field strength of each cell (units sqrt(4 pi n0 m_e c^2)), constant in
        time. Strictly positive.
    p_perp : float or array_like
        Perpendicular pressure at ``t_start`` (units n0 m_e c^2). Strictly
        positive.
    p_par : float or array_like
        Parallel pressure at ``t_start`` (units n0 m_e c^2). Strictly
        positive.
    t_start : float
        Start time (units 1/Omega0).
    t_end : float
        End time (units 1/Omega0); not earlier than ``t_start``.
    beta0 : float
        Reference plasma beta (step 01). Strictly positive.
    theta0 : float
        Reference temperature T0 / (m_e c^2) (step 01). Strictly positive.
    chi : float
        Reference product tau0 Omega0 (step 01). Strictly positive.
    t_d : float
        Onset time of step 02 (units 1/Omega0).
    c_th : float, optional
        Firehose threshold constant of step 02. Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(2,) + S``, with ``S`` the broadcast shape of ``n``,
        ``B``, ``p_perp`` and ``p_par``. Row 0 is P_perp(t_end) and row 1 is
        P_par(t_end), the exact solution of the local initial-value problem to
        a relative accuracy of 1e-11 or better.

    Raises
    ------
    ValueError
        If ``n``, ``B``, ``p_perp`` or ``p_par`` has a non-finite or
        non-positive entry, if these four do not broadcast together, if
        ``t_start``, ``t_end`` or ``t_d`` is not a finite scalar, if
        ``t_end < t_start``, or if ``beta0``, ``theta0``, ``chi`` or ``c_th``
        is not a finite, strictly positive scalar.
    """
    return pressures  # placeholder
```

### Step 4

pressure_balanced_state

Goal
----
The pressure-balanced state of a slab of fluid elements.

```python
def pressure_balanced_state(
    mass: ArrayLike,
    n: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    L: float,
    beta0: float,
    theta0: float,
) -> np.ndarray:
    """Return the pressure-balanced state of a slab of fluid elements.

    Parameters
    ----------
    mass : array_like
        Element masses (units n0 times the length unit of ``L``), shape (N,),
        N >= 1. Strictly positive.
    n : array_like
        Element densities (units n0), shape (N,). Strictly positive.
    p_perp : array_like
        Element perpendicular pressures (units n0 m_e c^2), shape (N,).
        Strictly positive.
    p_par : array_like
        Element parallel pressures (units n0 m_e c^2), shape (N,). Strictly
        positive.
    L : float
        Slab length, in the same length unit as ``mass / n``. Strictly
        positive.
    beta0 : float
        Reference plasma beta (step 01). Strictly positive.
    theta0 : float
        Reference temperature T0 / (m_e c^2) (step 01). Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape (4, N). Rows 0, 1 and 2 hold n, P_perp and P_par of
        every element in the pressure-balanced state; row 3 holds the common
        total perpendicular pressure P_perp + B^2/2 in every entry. The field
        of an element is B = B0 n with B0 the reference field of step 01. The
        balance and the length condition hold to a relative accuracy of 1e-12
        or better.

    Raises
    ------
    ValueError
        If ``mass``, ``n``, ``p_perp`` or ``p_par`` is not a one-dimensional
        array with at least one entry, if their lengths differ, if any entry
        is non-finite or non-positive, or if ``L``, ``beta0`` or ``theta0`` is
        not a finite, strictly positive scalar.
    """
    return state  # placeholder
```

### Step 5

pressure_balanced_step

Goal
----
One time step of the pressure-balanced slab.

```python
def pressure_balanced_step(
    mass: ArrayLike,
    n: ArrayLike,
    p_perp: ArrayLike,
    p_par: ArrayLike,
    t: float,
    dt: float,
    L: float,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
) -> np.ndarray:
    """Return the slab elements after one splitting time step.

    Parameters
    ----------
    mass, n, p_perp, p_par : array_like
        Element masses, densities and pressures at time ``t``, as in step 04.
    t : float
        Time at the start of the step (units 1/Omega0). Finite.
    dt : float
        Step size (units 1/Omega0). Finite and non-negative.
    L : float
        Slab length (step 04). Strictly positive.
    beta0 : float
        Reference plasma beta (step 01). Strictly positive.
    theta0 : float
        Reference temperature T0 / (m_e c^2) (step 01). Strictly positive.
    chi : float
        Reference product tau0 Omega0 (step 01). Strictly positive.
    t_d : float
        Onset time of the schedule of step 02 (units 1/Omega0). Finite.
    c_th : float, optional
        Firehose threshold constant of step 02. Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape (4, N). Rows 0, 1 and 2 hold n, P_perp and P_par of
        every element at ``t + dt``; row 3 holds, in every entry, the common
        total perpendicular pressure of the step-04 map taken inside the
        step. The field of an element is B = B0 n. For ``dt = 0`` the rows
        are the pressure-balanced state of the input.

    Raises
    ------
    ValueError
        For any input that step 04 rejects, if ``t`` or ``t_d`` is not a finite
        scalar, if ``dt`` is negative or not a finite scalar, or if ``chi`` or
        ``c_th`` is not a finite, strictly positive scalar.
    """
    return state  # placeholder
```

### Step 6

evolve_pressure_balanced_slab

Goal
----
Evolution of the pressure-balanced radiating slab.

```python
def evolve_pressure_balanced_slab(
    N: int,
    t_end: float,
    L: float,
    seed: ArrayLike,
    beta0: float,
    theta0: float,
    chi: float,
    t_d: float,
    c_th: float = 1.4,
    dt_max: float = 0.5,
) -> np.ndarray:
    """Evolve the pressure-balanced slab and return its elements at ``t_end``.

    Initial elements: element i = 0, ..., N-1 has initial width L/N and centre
    x_i = (i + 1/2) L/N; b_i = 1 + sum_j amp_j cos(2 pi k_j x_i / L + phase_j)
    over the rows (amp_j, k_j, phase_j) of ``seed``; n_i = b_i,
    mass_i = n_i L/N and P_perp,i = P_par,i = P0 [1 + (1 - b_i^2)/beta0] with
    P0 = theta0. This state is first mapped through step 04.

    Time grid: the marks are t_d (only if 0 < t_d < t_end) and t_end, taken in
    increasing order from time 0. The interval [a, b] between consecutive marks
    is divided into k = ceil((b - a)/dt_max - 1e-9) equal steps; step j of the
    interval starts at a + j (b - a)/k and is taken with step 05.

    Parameters
    ----------
    N : int
        Number of elements; at least 4.
    t_end : float
        Final time (units 1/Omega0); non-negative. ``t_end = 0`` returns the
        mapped initial state.
    L : float
        Slab length (any length unit). Strictly positive.
    seed : array_like
        Array of shape (M, 3), M >= 1, with rows (amplitude, integer wavenumber
        index k, phase in radians).
    beta0 : float
        Reference plasma beta (step 01). Strictly positive.
    theta0 : float
        Reference temperature T0 / (m_e c^2) (step 01). Strictly positive.
    chi : float
        Reference product tau0 Omega0 (step 01). Strictly positive.
    t_d : float
        Onset time of the schedule of step 02 (units 1/Omega0). Finite.
    c_th : float, optional
        Firehose threshold constant of step 02. Strictly positive.
    dt_max : float, optional
        Largest allowed step size (units 1/Omega0). Strictly positive.

    Returns
    -------
    numpy.ndarray
        Array of shape (4, N): rows n, P_perp, P_par and width mass/n of every
        element at ``t_end``, elements in their initial order (units of steps
        01 and 04).

    Raises
    ------
    ValueError
        If ``N`` is not an integer >= 4; if ``t_end`` is negative or not
        finite; if ``L``, ``beta0``, ``theta0``, ``chi``, ``c_th`` or ``dt_max``
        is not a finite, strictly positive scalar; if ``t_d`` is not finite; if
        ``seed`` is not an (M, 3) array of finite numbers with M >= 1 and
        integer wavenumber indices; or if the initial b or pressures are not
        strictly positive for every element.
    """
    return slab  # placeholder
```

### Step 7

anisotropy_shortfall_measure

Goal
----
End-to-end benchmark value (final orchestrator step).

```python
def anisotropy_shortfall_measure(
    N: int = 512,
    t_end: float = 16000.0,
    L: float = 1.0,
    beta0: float = 80.0,
    theta0: float = 1000.0,
    chi: float = 2000.0,
    t_d: float = 340.0,
    c_th: float = 1.4,
    dt_max: float = 0.5,
) -> float:
    """Return the benchmark anisotropy-shortfall measure Q of the slab.

    The slab of step 06 is evolved from the fixed seed with rows
    (amplitude, k, phase) = (-0.28, 1, 0) and (0.05, 2, pi/3) to ``t_end``,
    and

        Q = (1/L) sum_i w_i [max(0, 1 - (P_par,i - P_perp,i) / (c_th B_i^2 / 2))]^2

    is returned, with w_i the element widths and B_i = B0 n_i the element
    fields of the final state.

    Parameters
    ----------
    N : int, optional
        Number of elements of step 06; at least 4.
    t_end : float, optional
        Final time (units 1/Omega0); non-negative.
    L : float, optional
        Slab length (any length unit). Strictly positive.
    beta0 : float, optional
        Reference plasma beta (step 01). Strictly positive.
    theta0 : float, optional
        Reference temperature T0 / (m_e c^2) (step 01). Strictly positive.
    chi : float, optional
        Reference product tau0 Omega0 (step 01). Strictly positive.
    t_d : float, optional
        Onset time of the schedule of step 02 (units 1/Omega0). Finite.
    c_th : float, optional
        Firehose threshold constant of step 02. Strictly positive.
    dt_max : float, optional
        Largest step size of step 06 (units 1/Omega0). Strictly positive.

    Returns
    -------
    float
        The anisotropy-shortfall measure Q (dimensionless, between 0 and 1).

    Raises
    ------
    ValueError
        For any argument that step 06 rejects, or if the final state fails one
        of these consistency checks against the earlier steps: the element
        widths add up to ``L`` within a relative 1e-9; mapping the state through
        step 04 changes no density by more than a relative 1e-2; a zero-length
        step of step 05 reproduces that mapped state within a relative 1e-12;
        the radiative sinks of step 01 remove energy (2 dP_perp + dP_par < 0) in
        every element; step 02 gives no scattering (rate exactly 0) in any
        element lying below the threshold by more than a relative 1e-10 at
        du_dx = 0; and a unit-time advance by step 03 does not increase
        2 P_perp + P_par in any element (relative slack 1e-12).
    """
    return 0.0
```

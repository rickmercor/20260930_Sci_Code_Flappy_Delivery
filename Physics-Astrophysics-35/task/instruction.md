# Timestep-induced apsidal precession of an eccentric companion in a symplectic simulation

## Background

Long-term simulations of planetary and stellar systems rely on symplectic integrators of the Wisdom–Holman family. They split the Hamiltonian into a dominant Keplerian part, advanced exactly, and small perturbations, advanced as kicks and drifts. Many widely used codes apply this splitting in democratic heliocentric coordinates, which measure positions from the star and momenta from the barycentre. A symplectic map of this kind exactly follows a nearby modified Hamiltonian, so its errors do not grow secularly in energy. They can, however, appear as slow, systematic changes in the orientation of an orbit. Such spurious apsidal motion competes with genuine effects, such as the relativistic precession of close-in orbits, and has been discussed as a possible numerical influence on the long-term stability of the inner solar system. Quantifying it means relating the map's modified Hamiltonian to the motion of the orbit's line of apsides.

## Problem

A companion of mass $m_1$ orbits a star of mass $m_0$ on an eccentric orbit. The isolated pair is simulated with the second-order Wisdom–Holman map in democratic heliocentric coordinates, in which $\mathbf{Q}$ is the companion's position relative to the star and $\mathbf{P}$ is its barycentric momentum. The map splits the two-body Hamiltonian into two parts. The first is $A = T_1 + V_\odot$, with $T_1 = |\mathbf{P}|^2/(2m_1)$ and $V_\odot = -Gm_0m_1/|\mathbf{Q}|$, advanced as an exact Kepler step. The second is $B = T_0 = |\mathbf{P}|^2/(2m_0)$, the star's kinetic energy, advanced as a drift. Each step of size $h$ applies $B$ for $h/2$, $A$ for $h$ and $B$ for $h/2$ (the BAB ordering). General relativity is included through the potential $V_{\rm GR} = -3G^2m_0^2m_1/(c^2|\mathbf{Q}|^2)$, applied as a kick within $B$. The benchmark is:

| Quantity | Symbol | Value |
|---|---|---|
| Star mass | $m_0$ | $1.0~M_\odot$ |
| Companion mass | $m_1$ | $0.07~M_\odot$ |
| Semimajor axis of the relative orbit | $a$ | $0.06~\mathrm{au}$ |
| Eccentricity | $e$ | $0.5$ |
| Map step | $h$ | $14~\mathrm{minutes}$ |
| Gravitational constant | $G$ | $4\pi^2~\mathrm{au^3}\,M_\odot^{-1}\,\mathrm{yr^{-2}}$ |
| Year | yr | $365.25~\mathrm{days}$ |
| Speed of light | $c$ | $299\,792\,458~\mathrm{m\,s^{-1}}$ |
| Astronomical unit | au | $1.495\,978\,707\times10^{11}~\mathrm{m}$ |

Report, as a single number in arcseconds per century, the orbit-averaged apsidal precession rate of the companion's orbit that this simulation exhibits, counted positive for prograde precession (in the sense of the orbital angular momentum). Evaluate it to leading order in the step size (the $O(h^2)$ term, keeping the exact dependence on both masses) and to first order in $V_{\rm GR}$, treating the physical and map-induced contributions as additive. Alongside it, report the mass ratio $m_1/m_0$, the orbital period, the precession rate produced by $V_{\rm GR}$ alone, the map-induced (artificial) precession rate alone, and the map step at which the simulated net precession would vanish. Include a concise derivation of the map-induced rate: identify the secular $O(h^2)$ term of the map's modified Hamiltonian that produces the artificial precession, and state the exact leading-order expression for the artificial precession rate in terms of $h$, $G$, $m_0$, $m_1$, $a$ and $e$ (any algebraically equivalent form is acceptable) before evaluating it for the benchmark. Also state, as internal checks:
- whether reversing the map to the ABA ordering ($A$ for $h/2$, $B$ for $h$, $A$ for $h/2$) changes the leading-order artificial precession;
- what the artificial precession of this two-body system would be if it were integrated with the Wisdom–Holman map in Jacobi coordinates instead.

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

compute_two_body_constants

Goal
----
Compute the basic constants of an isolated star-companion two-body system: the mass ratio, the total and reduced masses, the coefficient of the Runge-Lenz vector used by the later steps, and the period of the relative orbit.

```python
def compute_two_body_constants(m0: float, m1: float, a: float, e: float) -> np.ndarray:
    """Basic constants of the star-companion two-body problem.

    Parameters
    ----------
    m0 : float
        Mass of the star (Msun), a finite number > 0.
    m1 : float
        Mass of the companion (Msun), a finite number > 0.
    a : float
        Semimajor axis of the relative orbit (au), a finite number > 0.
    e : float
        Eccentricity of the relative orbit, a finite number with 0 < e < 1.

    Returns
    -------
    consts : np.ndarray
        Array of shape (5,): [m1/m0, total mass M, reduced mass mu,
        Runge-Lenz coefficient c0, orbital period in years].

    Raises
    ------
    ValueError
        If m0, m1 or a is not a finite number > 0, or if e is not a finite
        number with 0 < e < 1.
    """
    return consts  # placeholder
```

### Step 2

democratic_heliocentric_coordinates

Goal
----
Transform the inertial positions and momenta of an N-body system, star first, into democratic heliocentric coordinates.

```python
def democratic_heliocentric_coordinates(masses: np.ndarray, positions: np.ndarray,
                                        momenta: np.ndarray) -> np.ndarray:
    """Inertial positions and momenta -> democratic heliocentric coordinates.

    Parameters
    ----------
    masses : np.ndarray
        Shape (N,), N >= 2, the masses (Msun), star first; every entry finite and > 0.
    positions : np.ndarray
        Shape (N, 3), inertial positions (au), row i for body i; finite.
    momenta : np.ndarray
        Shape (N, 3), inertial momenta (Msun au / yr), row i for body i; finite.

    Returns
    -------
    QP : np.ndarray
        Shape (2, N, 3): QP[0] holds Q (row 0 the centre-of-mass position,
        row i the position of body i relative to the star) and QP[1] holds P
        (row 0 the total momentum, row i the barycentric momentum of body i).

    Raises
    ------
    ValueError
        If masses is not one-dimensional with at least 2 entries; if positions
        or momenta does not have shape (N, 3); if any input contains a
        non-finite value; or if any mass is <= 0.
    """
    return QP  # placeholder
```

### Step 3

kepler_drift

Goal
----
Advance a relative position and velocity along their exact Keplerian orbit for a given time.

```python
def kepler_drift(q: np.ndarray, v: np.ndarray, gm: float, dt: float) -> np.ndarray:
    """Exact Keplerian propagation of a relative position and velocity.

    Parameters
    ----------
    q : np.ndarray
        Shape (3,), relative position (au); finite and not the zero vector.
    v : np.ndarray
        Shape (3,), relative velocity (au / yr); finite.
    gm : float
        Gravitational parameter of the central attraction (au^3 / yr^2), a
        finite number > 0.
    dt : float
        Time to advance (yr), any finite number (negative values propagate
        backwards).

    Returns
    -------
    qv : np.ndarray
        Shape (2, 3): qv[0] the position and qv[1] the velocity after dt.

    Raises
    ------
    ValueError
        If q or v does not have shape (3,) or contains a non-finite value; if
        q is the zero vector; if gm is not a finite number > 0; or if dt is
        not finite.
    """
    return qv  # placeholder
```

### Step 4

runge_lenz_vector

Goal
----
Compute the Runge-Lenz vector of the relative two-body orbit from its democratic heliocentric coordinates.

```python
def runge_lenz_vector(Q: np.ndarray, P: np.ndarray, m0: float, m1: float) -> np.ndarray:
    """Runge-Lenz vector of the two-body relative orbit.

    Parameters
    ----------
    Q : np.ndarray
        Shape (3,), companion position relative to the star (au); finite, nonzero.
    P : np.ndarray
        Shape (3,), barycentric momentum of the companion (Msun au / yr); finite.
    m0 : float
        Mass of the star (Msun), a finite number > 0.
    m1 : float
        Mass of the companion (Msun), a finite number > 0.

    Returns
    -------
    R : np.ndarray
        Shape (3,), the Runge-Lenz vector P x L - c0 Q/|Q| with L = Q x P.

    Raises
    ------
    ValueError
        If Q or P does not have shape (3,) or contains a non-finite value; if
        Q is the zero vector; or if m0 or m1 is not a finite number > 0.
    """
    return R  # placeholder
```

### Step 5

error_hamiltonian_rl_rate

Goal
----
Compute the rate of change of the Runge-Lenz vector generated by the secular part of the leading-order error Hamiltonian of a BAB Wisdom-Holman map, at a given state.

```python
def error_hamiltonian_rl_rate(Q: np.ndarray, P: np.ndarray, m0: float, m1: float,
                              h: float) -> np.ndarray:
    """Rate of change of the Runge-Lenz vector under the secular error Hamiltonian H_B.

    Parameters
    ----------
    Q : np.ndarray
        Shape (3,), companion position relative to the star (au); finite, nonzero.
    P : np.ndarray
        Shape (3,), barycentric momentum of the companion (Msun au / yr); finite.
    m0 : float
        Mass of the star (Msun), a finite number > 0.
    m1 : float
        Mass of the companion (Msun), a finite number > 0.
    h : float
        Map step size (yr), a finite number > 0.

    Returns
    -------
    Rdot : np.ndarray
        Shape (3,), dR/dt = {R, H_B} (units of R per year).

    Raises
    ------
    ValueError
        If Q or P does not have shape (3,) or contains a non-finite value; if
        Q is the zero vector; or if m0, m1 or h is not a finite number > 0.
    """
    return Rdot  # placeholder
```

### Step 6

orbit_averaged_artificial_precession

Goal
----
Average the instantaneous apsidal precession rate produced by the error Hamiltonian over one unperturbed Keplerian orbit, giving the secular (artificial) precession rate of the BAB map.

```python
def orbit_averaged_artificial_precession(m0: float, m1: float, a: float, e: float, h: float,
                                          n_samples: int = 512) -> float:
    """Orbit-averaged artificial apsidal precession rate of the BAB map (rad / yr).

    Parameters
    ----------
    m0 : float
        Mass of the star (Msun), a finite number > 0.
    m1 : float
        Mass of the companion (Msun), a finite number > 0.
    a : float
        Semimajor axis of the relative orbit (au), a finite number > 0.
    e : float
        Eccentricity, a finite number with 0 < e < 1.
    h : float
        Map step size (yr), a finite number > 0.
    n_samples : int
        Number of equally spaced sample times over one period, an integer >= 16.

    Returns
    -------
    rate : float
        The orbit-averaged precession rate of the Runge-Lenz vector about the
        orbital angular momentum, in rad / yr (positive = prograde).

    Raises
    ------
    ValueError
        If m0, m1 or a is not a finite number > 0; if e is not a finite number
        with 0 < e < 1; if h is not a finite number > 0; or if n_samples is not
        an integer >= 16.
    """
    return rate  # placeholder
```

### Step 7

gr_precession_rate

Goal
----
Compute the orbit-averaged apsidal precession rate produced by the simulation's general-relativity potential.

```python
def gr_precession_rate(m0: float, m1: float, a: float, e: float) -> float:
    """Orbit-averaged apsidal precession rate from V_GR = -3 G^2 m0^2 m1 / (c^2 |Q|^2) (rad / yr).

    Parameters
    ----------
    m0 : float
        Mass of the star (Msun), a finite number > 0.
    m1 : float
        Mass of the companion (Msun), a finite number > 0.
    a : float
        Semimajor axis of the relative orbit (au), a finite number > 0.
    e : float
        Eccentricity, a finite number with 0 < e < 1.

    Returns
    -------
    rate : float
        The orbit-averaged apsidal precession rate, in rad / yr (positive = prograde).

    Raises
    ------
    ValueError
        If m0, m1 or a is not a finite number > 0, or if e is not a finite
        number with 0 < e < 1.
    """
    return rate  # placeholder
```

### Step 8

compute_net_apsidal_precession

Goal
----
Orchestrator: compute the net orbit-averaged apsidal precession rate, in arcseconds per century, that a BAB Wisdom-Holman simulation in democratic heliocentric coordinates with the GR potential shows for the star-companion orbit.

```python
def compute_net_apsidal_precession(m0: float = 1.0, m1: float = 0.07, a: float = 0.06,
                                   e: float = 0.5, timestep_minutes: float = 14.0,
                                   n_samples: int = 512) -> float:
    """Net apsidal precession rate (GR plus artificial) of the simulated orbit, in arcsec / century.

    Parameters
    ----------
    m0 : float
        Mass of the star (Msun), a finite number > 0.
    m1 : float
        Mass of the companion (Msun), a finite number > 0.
    a : float
        Semimajor axis (au), a finite number > 0.
    e : float
        Eccentricity, a finite number with 0 < e < 1.
    timestep_minutes : float
        Map step size in minutes, a finite number > 0.
    n_samples : int
        Number of sample times used for the orbit average, an integer >= 16.

    Returns
    -------
    rate : float
        Net orbit-averaged apsidal precession rate in arcsec / century
        (positive = prograde).

    Raises
    ------
    ValueError
        If timestep_minutes is not a finite number > 0, or if any underlying
        step raises ValueError on its own inputs.
    """
    return rate  # placeholder
```

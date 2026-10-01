# Physics-Particle_Physics-29

## Background

Coupled pseudo-scalar fields with temperature-dependent masses are a generic feature of multi-axion extensions of the Standard Model, motivated by string theory constructions that typically predict many axion-like fields rather than a single one. When two such fields mix, their combined cosmological evolution can differ sharply from the single-field case: rather than each field's abundance being fixed independently once it begins to oscillate, a resonant level crossing between the two mass eigenstates can transfer population from one eigenstate to the other, in close analogy to the Landau-Zener effect studied originally in atomic and molecular physics, and to the Mikheyev-Smirnov-Wolfenstein effect governing flavor conversion of solar neutrinos as they pass through the Sun's varying electron density. Existing treatments of multi-axion cosmology have generally been restricted to the adiabatic limit, in which the mixing angle changes slowly enough that no such conversion occurs and each eigenstate's abundance can be computed independently using the standard single-field misalignment mechanism. Determining the dark matter abundance in the opposite, non-adiabatic regime, where the crossing is traversed quickly enough that population transfer between eigenstates is significant, requires tracking how the pre-crossing conserved quantities of each eigenstate combine into new post-crossing conserved quantities, and then correctly identifying which physical eigenstate carries which of these post-crossing quantities all the way to the present day, since the mapping between the mass eigenstates and the underlying flavor fields is itself temperature-dependent and need not be the same before and after the crossing.

## Problem

A cosmological model contains two coupled pseudo-scalar fields, $a$ and $a_S$, both initially frozen by Hubble friction in the early universe. The field $a$ couples to gluons through the usual QCD axion mechanism, so its mass grows from a strongly suppressed value at high temperature toward a fixed zero-temperature value as the universe cools through the QCD confinement scale. The field $a_S$ has a fixed mass, independent of temperature. The two fields mix through an interaction term in their combined potential, so the true mass eigenstates are temperature-dependent linear combinations of $a$ and $a_S$, and the heavier and lighter eigenstates are labeled $a_H$ and $a_L$ respectively at any given time.

Because the QCD-coupled field's mass grows relative to the fixed mass of the other field as the universe cools, there is a temperature at which the splitting between the two mass eigenvalues is smallest: a level crossing. Depending on how rapidly the mixing angle changes relative to the splitting between the two mass eigenstates at that moment, the crossing can be traversed adiabatically (each eigenstate's identity is preserved) or non-adiabatically (the eigenstates partially or fully exchange populations, a Landau-Zener-type transition). Well before the crossing, each field begins independently oscillating around the minimum of its own effective potential once the Hubble rate drops enough, at a temperature that depends on how strongly that field's own mass varies with temperature at that time; once oscillating, each mass eigenstate's comoving number of quanta is separately conserved until the crossing is reached.

The system is specified by the axion decay constant $f_a$, the dimensionless ratios $R_f \equiv f_S/f_a$ and $R_m \equiv m_S/m_{a,0}$ (with $m_{a,0}$ the zero-temperature mass of the QCD-coupled field and $m_S$ the fixed mass of the other field), the QCD confinement scale $T_{\rm QCD}$, the exponent $n$ such that the QCD-coupled field's mass falls off as $T^{-n}$ above $T_{\rm QCD}$ (equivalently, the topological susceptibility, which is proportional to the mass squared, falls off as $T^{-2n}$), the effective relativistic degrees of freedom $g_*$, and the initial misalignment angles $\theta_a \equiv a/f_a$ and $\theta_s \equiv a_S/f_S$ at the onset of the fields' evolution. The combined potential is $V = m_a(T)^2 f_a^2[1-\cos(a/f_a)] + m_S^2 f_S^2[1-\cos(a_S/f_S + a/f_a)]$. The zero-temperature mass $m_{a,0}$ is not given directly: it follows from $f_a$ through the standard chiral-perturbation-theory relation for the QCD axion mass, evaluated with the hadronic inputs in the table below. The Hubble rate uses the reduced Planck mass given in the table. The benchmark values are:

| Quantity | Symbol | Value |
|---|---|---|
| Axion decay constant | $f_a$ | $1.0\times10^{12}~\mathrm{GeV}$ |
| Mixing ratio | $R_f$ | $0.02$ |
| Mass ratio | $R_m$ | $0.10$ |
| Initial misalignment angle ('a') | $\theta_a$ | $0.02$ |
| Initial misalignment angle ('$a_S$') | $\theta_s$ | $0.8$ |
| QCD confinement scale | $T_{\rm QCD}$ | $0.100~\mathrm{GeV}$ |
| Exponent of $m_a(T)$ above $T_{\rm QCD}$ ($m_a \propto T^{-n}$) | $n$ | $3.34$ |
| Effective degrees of freedom | $g_*$ | $61.75$ |
| Present-day temperature | $T_0$ | $2.348\times10^{-13}~\mathrm{GeV}$ |
| Neutral pion mass | $m_\pi$ | $0.135~\mathrm{GeV}$ |
| Pion decay constant | $f_\pi$ | $0.092~\mathrm{GeV}$ |
| Up-quark mass | $m_u$ | $2.16\times10^{-3}~\mathrm{GeV}$ |
| Down-quark mass | $m_d$ | $4.67\times10^{-3}~\mathrm{GeV}$ |
| Reduced Planck mass | $M_{\rm Pl}$ | $2.435\times10^{18}~\mathrm{GeV}$ |

Report, as a single number, the fraction of the total present-day dark matter energy density carried by the light mass eigenstate, $f_L = \rho_L(T_0)/[\rho_H(T_0)+\rho_L(T_0)]$. Evaluate $f_L$ as the small-amplitude WKB and Landau-Zener estimate: each mass eigenstate's pre-crossing comoving number density is the WKB invariant of a harmonic (small-amplitude) oscillation fixed at that state's oscillation onset, the crossing mixes the two densities through the Landau-Zener probability, and each resulting density is then conserved to the present day. Do not integrate the full nonlinear equations of motion of the cosine potential; the requested number is this small-amplitude estimate, not the result of the full anharmonic dynamics. Alongside $f_L$, report the zero-temperature masses $m_{a,0}$ and $m_S$, the level-crossing temperature, the adiabaticity parameter (normalized so that $P_{LZ} = \exp(-\pi\gamma/2)$) and the Landau-Zener conversion probability at the crossing, the two oscillation temperatures, and the ratio of the light to the heavy eigenstate's comoving number density just after the crossing. Also state, as internal checks: what the abundance fraction would reduce to if the level crossing were instead treated as fully adiabatic (no conversion between mass eigenstates at all), and the flavor content of the heavier mass eigenstate well before the crossing, exactly at the crossing, and today (in particular, whether its flavor content well before the crossing is the same as today).

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

compute_axion_mass_scales

Goal
----
Compute the zero-temperature (present-day) masses of the two axion flavor fields from the axion decay constant and the model's dimensionless mixing ratios.

```python
import numpy as np


def compute_axion_mass_scales(fa: float, Rm: float) -> tuple:
    """Zero-temperature axion mass scales m_{a,0} and mS.

    Parameters
    ----------
    fa : float
        Axion decay constant (GeV), a finite number > 0.
    Rm : float
        Mass ratio mS / m_{a,0}, a finite number > 0.

    Returns
    -------
    masses : tuple
        (m_a0, mS): two floats, both finite and > 0, in GeV.

    Raises
    ------
    ValueError
        If fa or Rm is not a finite number > 0.
    """
    return masses  # placeholder
```

### Step 2

compute_axion_mass_squared_at_temperature

Goal
----
Compute the squared mass of the QCD-coupled axion flavor field at a given temperature.

```python
import numpy as np


def compute_axion_mass_squared_at_temperature(T: float, m_a0: float, T_QCD: float, n: float) -> float:
    """Temperature-dependent squared mass of the QCD-coupled axion field.

    Parameters
    ----------
    T : float
        Temperature (GeV), a finite number > 0.
    m_a0 : float
        Zero-temperature axion mass (GeV), a finite number > 0.
    T_QCD : float
        QCD confinement scale (GeV), a finite number > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n, so m_a^2 and the topological
        susceptibility fall off as T^-2n), a finite number > 0.

    Returns
    -------
    m_a_sq : float
        The squared mass m_a(T)^2 in GeV^2, finite and > 0.

    Raises
    ------
    ValueError
        If T, m_a0, T_QCD, or n is not a finite number > 0.
    """
    return m_a_sq  # placeholder
```

### Step 3

diagonalize_axion_mass_matrix

Goal
----
Construct the coupled two-field mass matrix implied by the axion-mixing potential, and diagonalize it to obtain the physical mass eigenvalues and mixing angle at a given temperature.

```python
import numpy as np


def diagonalize_axion_mass_matrix(m_a_sq: float, mS: float, Rf: float) -> tuple:
    """Construct and diagonalize the 2x2 axion mass matrix at a given temperature.

    Parameters
    ----------
    m_a_sq : float
        The QCD-coupled field's squared mass m_a(T)^2 at the temperature of
        interest (GeV^2), a finite number > 0.
    mS : float
        The second flavor field's mass (GeV), a finite number > 0.
    Rf : float
        Dimensionless ratio fS/fa, a finite number > 0.

    Returns
    -------
    result : tuple
        (m_H_sq, m_L_sq, xi): the heavy and light squared mass eigenvalues
        (GeV^2, m_H_sq >= m_L_sq > 0) and the mixing angle xi (radians),
        defined so that the heavy eigenvector in the (aS, a) basis is
        (cos xi, sin xi).

    Raises
    ------
    ValueError
        If m_a_sq, mS, or Rf is not a finite number > 0.
    """
    return result  # placeholder
```

### Step 4

find_crossing_temperature

Goal
----
Find the temperature at which the two diagonal entries of the axion mass matrix become equal, defining the level-crossing (resonance) temperature.

```python
import numpy as np


def find_crossing_temperature(m_a0: float, mS: float, Rf: float, T_QCD: float, n: float) -> float:
    """Root-find the level-crossing temperature T_x.

    Parameters
    ----------
    m_a0 : float
        Zero-temperature axion mass (GeV), a finite number > 0.
    mS : float
        Second flavor field's mass (GeV), a finite number > 0.
    Rf : float
        Dimensionless ratio fS/fa, a finite number > 0 and < 1 (Rf < 1 is
        required for a crossing to exist).
    T_QCD : float
        QCD confinement scale (GeV), a finite number > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n, so m_a^2 and the topological
        susceptibility fall off as T^-2n), a finite number > 0.

    Returns
    -------
    T_x : float
        The crossing temperature in GeV, finite and > 0.

    Raises
    ------
    ValueError
        If m_a0, mS, T_QCD, or n is not a finite number > 0; if Rf is not
        a finite number with 0 < Rf < 1; or if no crossing exists for the
        given parameters (the value of m_a(T)^2 that the crossing
        condition requires is at least m_a0^2, so it can never be reached).
    """
    return T_x  # placeholder
```

### Step 5

compute_landau_zener_probability

Goal
----
Compute the adiabaticity parameter governing the level crossing, and the resulting Landau-Zener probability of conversion between the two mass eigenstates.

```python
import numpy as np


def compute_landau_zener_probability(T_x: float, m_a0: float, mS: float, Rf: float,
                                      T_QCD: float, n: float, g_star: float) -> tuple:
    """Adiabaticity parameter and Landau-Zener conversion probability at the crossing.

    Parameters
    ----------
    T_x : float
        Crossing temperature (GeV), a finite number > 0.
    m_a0 : float
        Zero-temperature axion mass (GeV), a finite number > 0.
    mS : float
        Second flavor field's mass (GeV), a finite number > 0.
    Rf : float
        Dimensionless ratio fS/fa, a finite number > 0.
    T_QCD : float
        QCD confinement scale (GeV), a finite number > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n, so m_a^2 and the topological
        susceptibility fall off as T^-2n), a finite number > 0.
    g_star : float
        Effective relativistic degrees of freedom at T_x, a finite number > 0.

    Returns
    -------
    result : tuple
        (gamma, P_LZ): the adiabaticity parameter and the Landau-Zener
        conversion probability, both finite, gamma >= 0 and 0 <= P_LZ <= 1
        (P_LZ may underflow to exactly 0 when gamma is large).

    Raises
    ------
    ValueError
        If any of T_x, m_a0, mS, Rf, T_QCD, n, or g_star is not a finite
        number > 0.
    """
    return result  # placeholder
```

### Step 6

compute_oscillation_temperatures_and_initial_fields

Goal
----
Determine the temperatures at which each mass eigenstate begins to oscillate, and the initial field values each eigenstate carries once oscillations begin.

```python
import numpy as np


def compute_oscillation_temperatures_and_initial_fields(
    m_a0: float, mS: float, Rf: float, fa: float, T_QCD: float, n: float,
    g_star: float, theta_a: float, theta_s: float,
) -> tuple:
    """Oscillation temperatures and initial mass-eigenstate field values.

    Parameters
    ----------
    m_a0 : float
        Zero-temperature axion mass (GeV), a finite number > 0.
    mS : float
        Second flavor field's mass (GeV), a finite number > 0.
    Rf : float
        Dimensionless ratio fS/fa, a finite number > 0.
    fa : float
        Axion decay constant (GeV), a finite number > 0.
    T_QCD : float
        QCD confinement scale (GeV), a finite number > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n, so m_a^2 and the topological
        susceptibility fall off as T^-2n), a finite number > 0.
    g_star : float
        Effective relativistic degrees of freedom, a finite number > 0.
    theta_a : float
        Initial misalignment angle of the 'a' flavor field, finite.
    theta_s : float
        Initial misalignment angle of the 'aS' flavor field, finite.

    Returns
    -------
    result : tuple
        (T_osc_H, T_osc_L, a_H, a_L): the two oscillation temperatures
        (GeV, > 0) and the two initial mass-eigenstate field values (GeV).

    Raises
    ------
    ValueError
        If m_a0, mS, Rf, fa, T_QCD, n, or g_star is not a finite number
        > 0; or if theta_a or theta_s is not finite.
    """
    return result  # placeholder
```

### Step 7

assemble_relic_abundance_fraction

Goal
----
Assemble the present-day dark matter abundance carried by each mass eigenstate, and report the fraction carried by the light eigenstate.

```python
import numpy as np


def assemble_relic_abundance_fraction(
    m_a0: float, mS: float, Rf: float, T_osc_H: float, T_osc_L: float,
    a_H: float, a_L: float, P_LZ: float, T0: float, T_QCD: float, n: float,
) -> float:
    """Present-day relic abundance fraction carried by the light eigenstate.

    Parameters
    ----------
    m_a0 : float
        Zero-temperature axion mass (GeV), a finite number > 0.
    mS : float
        Second flavor field's mass (GeV), a finite number > 0.
    Rf : float
        Dimensionless ratio fS/fa, a finite number > 0.
    T_osc_H : float
        Oscillation temperature of the heavy eigenstate (GeV), > 0.
    T_osc_L : float
        Oscillation temperature of the light eigenstate (GeV), > 0.
    a_H : float
        Initial field value of the heavy eigenstate (GeV), finite.
    a_L : float
        Initial field value of the light eigenstate (GeV), finite.
    P_LZ : float
        Landau-Zener conversion probability, 0 <= P_LZ <= 1 (P_LZ = 0 is the
        fully adiabatic limit, with no conversion).
    T0 : float
        Present-day temperature (GeV), a finite number > 0, with T0 < T_QCD.
    T_QCD : float
        QCD confinement scale (GeV), a finite number > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n, so m_a^2 and the topological
        susceptibility fall off as T^-2n), a finite number > 0.

    Returns
    -------
    f_L : float
        The present-day abundance fraction carried by the light eigenstate,
        a finite number in (0, 1).

    Raises
    ------
    ValueError
        If m_a0, mS, Rf, T_osc_H, T_osc_L, T0, or T_QCD is not a finite
        number > 0; if a_H or a_L is not finite; if P_LZ is not in [0, 1];
        or if T0 is not strictly less than T_QCD.
    """
    return f_L  # placeholder
```

### Step 8

compute_relic_abundance_fraction

Goal
----
Orchestrator: compute the present-day dark matter abundance fraction carried by the light mass eigenstate of a two-axion system, from the model's fundamental parameters.

```python
import numpy as np


def compute_relic_abundance_fraction(
    fa: float = 1.0e12,
    Rf: float = 0.02,
    Rm: float = 0.10,
    theta_a: float = 0.02,
    theta_s: float = 0.8,
    T_QCD: float = 0.100,
    n: float = 3.34,
    g_star: float = 61.75,
    T0: float = 2.348e-13,
) -> float:
    """Present-day relic abundance fraction of a two-axion Landau-Zener system.

    Parameters
    ----------
    fa : float
        Axion decay constant (GeV), > 0.
    Rf : float
        Dimensionless ratio fS/fa, with 0 < Rf < 1.
    Rm : float
        Dimensionless ratio mS/m_{a,0}, > 0.
    theta_a : float
        Initial misalignment angle of the 'a' flavor field, finite.
    theta_s : float
        Initial misalignment angle of the 'aS' flavor field, finite.
    T_QCD : float
        QCD confinement scale (GeV), > 0.
    n : float
        Exponent of the QCD-coupled field's mass above T_QCD
        (m_a proportional to T^-n; m_a^2 falls off as T^-2n), > 0.
    g_star : float
        Effective relativistic degrees of freedom, > 0.
    T0 : float
        Present-day temperature (GeV), > 0, with T0 < T_QCD.

    Returns
    -------
    f_L : float
        The present-day abundance fraction carried by the light mass
        eigenstate, a finite number in (0, 1).

    Raises
    ------
    ValueError
        If any underlying step raises ValueError on its own inputs, or if
        the top-level parameter constraints described above are violated.
    """
    return f_L  # placeholder
```

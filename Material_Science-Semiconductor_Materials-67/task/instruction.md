# Material_Science-Semiconductor_Materials-67

## Background

### Lateral plasmonic crystals

A two-dimensional electron system supports plasma waves whose frequency is set by the electron density, the effective mass and the dielectric surroundings. Placing a metal gate close to the electrons screens the Coulomb interaction and turns the square-root dispersion of an open layer into a linear, slower one. When the gate is a periodic grating, the plasma velocity alternates along the channel and the plasma waves form bands and gaps, like electrons in a periodic potential. Because the density under each strip follows the voltage on that strip, the band structure of such a lateral plasmonic crystal can be tuned electrically, which is why grating-gate transistors are studied as tunable terahertz detectors, emitters, phase shifters and ratchet rectifiers.

### Kronig-Penney description, bright and dark modes

The simplest quantitative model divides each period into two strips, treats the wave in each strip as a plasma wave of an infinite layer of that kind, and matches the ac current and the ac potential at the strip boundaries. Imposing Bloch periodicity then gives a transcendental dispersion relation for the crystal. At the zone centre the modes split into two families. Bright modes carry a net oscillating dipole in each period and absorb normally incident radiation, dark modes carry none and are invisible in transmission. The two families interleave, and their order depends on the filling factor of the grating and on the densities under the strips.

### Massive and linear plasmons

Near the zone centre the bands of the lowest bright and dark modes are parabolic, which makes it natural to describe them by effective plasmon masses, positive or negative according to the curvature. The masses are small, because the bands are shallow on the scale of the electron dispersion, and they are very sensitive to the geometry. At particular filling factors the fundamental bright and dark modes become degenerate; there the masses vanish and change sign, and the two branches leave the zone centre as a cone with a finite speed. How fast the mass grows as the filling factor is moved away from such a point measures how strongly the grating can tune the plasmon inertia.

### Why the screening must be kept exactly

Most analytical treatments use the two limits of the gate screening: a fully screened linear plasmon under a gate that is much closer to the electrons than the plasma wavelength, and an unscreened plasmon where there is no gate. For micrometre periods and gates a few hundred nanometres to a micrometre away, possibly on different dielectrics, the product of wave number and gate distance at terahertz frequencies is of order a few tenths, which is neither limit. The nonlocal effective permittivity of a layer under an ideal metal plane then has to be kept in full in every strip, so the local wave numbers can only be found numerically and the positions of the modes, the masses and the degeneracy all move away from their limit-formula values. The model also neglects damping, which is only justified when the momentum relaxation time of the electrons is long compared with the plasma period.

## Problem

A lateral plasmonic crystal is made from a GaAs two-dimensional electron layer (electron effective mass m* = 0.067 m_0) with two interleaved metal gratings of period L = 6.0 um: in each period a strip of width fL lies under grating 1, which sits h_1 = 300 nm above the electrons on a spacer of permittivity kappa_1 = 12.8, and holds N_1 = 3.0 × 10^11 cm^-2; the remaining strip of width (1 - f)L lies under grating 2, which sits h_2 = 1500 nm above the electrons on a spacer of permittivity kappa_2 = 7.0, and holds N_2 = 3.0 × 10^12 cm^-2. The semiconductor below the layer has permittivity 12.8, each grating acts on its own strip as an ideal metal plane, and damping is neglected.\n\nTreat the crystal in the Kronig-Penney spirit with the full gate screening kept in both strips: inside strip n a plasma wave of angular frequency omega has the local wave number q fixed by omega^2 = 2 pi N_n e^2 q / (m* eps_n(q)) with eps_n(q) = [12.8 + kappa_n coth(q h_n)]/2, its ac potential equals 2 pi e^2 / (q eps_n(q)) times its density perturbation as in an infinite layer of that kind, the ac particle current N v and the ac potential are continuous at every strip boundary, and the fields are Bloch periodic with quasimomentum k. A zone-centre (k = 0) mode is bright when its ac current averaged over one period is nonzero and dark when that average vanishes, the lowest-frequency mode of each kind is its fundamental mode, and the effective plasmon mass M of a fundamental mode is defined by hbar omega(k) = hbar omega(0) + hbar^2 k^2/(2M) + O(k^4). Work in Gaussian units with e = 4.80320471 × 10^-10 statC, m_0 = 9.1093837015 × 10^-28 g and hbar = 1.054571817 × 10^-27 erg s.\n\nLet f* be the largest filling factor in 0 < f < 1 at which the fundamental bright and dark zone-centre frequencies coincide, at a common frequency f_0 = omega_0/(2 pi), where both branches leave the zone centre linearly with a common speed s_0. Report as your final answer the derivative dM_b/df of the fundamental bright plasmon mass with respect to the filling factor at f = f*, in units of 10^-4 m*. The scalars that determine this number, and that your reasoning should set out alongside it, are, at f = 0.20, the fundamental bright and dark zone-centre frequencies in GHz and M_b in units of 10^-4 m*; and f*, f_0 in GHz, s_0 in cm/s and the rate d(f_b - f_d)/df at f* at which the difference between the fundamental bright and dark frequencies changes with filling factor, in GHz, each to at least three significant figures.\n\nKronig-Penney descriptions of this kind ignore damping. Judge that against the 2025 terahertz transmission study of grating-gated plasmonic crystals made from a high-mobility AlGaAs/GaAs quantum well, in which the fundamental plasma frequency was measured against the gate filling factor at grating periods of 8 and 12 um, by setting out in your reasoning three further scalars next to those above, also to at least three significant figures: the Drude momentum relaxation time implied by the electron mobility reported for that heterostructure (with m* as above), the product omega_0 tau for your frequency f_0, and the relaxation time that study inferred from the linewidth of the fundamental resonance at small filling factor.
 

Output Format Requirements:
Emit `<final_answer>` immediately, then `<reasoning>`. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside `<final_answer>...</final_answer>`, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep `<reasoning>` short (a few hundred words). Show only the few scalars that determine the final number.
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

local_plasmon_wavenumber

Goal
----
Local wave number of a two-dimensional plasma wave under an ideal metal plane, from its nonlocal screened dispersion.

```python
import numpy as np

def local_plasmon_wavenumber(frequency_ghz: float, density: float, gate_nm: float, eps_substrate: float,
                             eps_spacer: float, mass_ratio: float) -> float:
    '''Wave number of the screened two-dimensional plasma wave at a given frequency.

    Parameters
    ----------
    frequency_ghz : float
        Wave frequency omega/(2 pi) in GHz.
    density : float
        Equilibrium electron density N in cm^-2.
    gate_nm : float
        Distance h between the electron layer and the ideal metal plane, in nm.
    eps_substrate : float
        Permittivity eps_s of the semiconductor below the layer.
    eps_spacer : float
        Permittivity eps of the spacer between the layer and the metal plane.
    mass_ratio : float
        Electron effective mass in units of the free-electron mass.

    Returns
    -------
    wavenumber : float
        The unique positive q, in cm^-1, satisfying the screened dispersion.

    Raises
    ------
    ValueError
        If any argument is not a positive finite number.
    '''
    return wavenumber
```

### Step 2

bloch_phase_cosine

Goal
----
Cosine of the Bloch phase of a plasma wave in a two-strip lateral plasmonic crystal at a given frequency.

```python
import numpy as np

def bloch_phase_cosine(frequency_ghz: float, period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float) -> float:
    '''Half-trace of the one-period transfer matrix of the two-strip plasmonic crystal.

    Parameters
    ----------
    frequency_ghz : float
        Wave frequency omega/(2 pi) in GHz.
    period_um : float
        Period L of the crystal in micrometres.
    fill : float
        Filling factor f, the fraction of each period occupied by strip 1; 0 < f < 1.
    density1 : float
        Electron density N_1 of strip 1 in cm^-2.
    density2 : float
        Electron density N_2 of strip 2 in cm^-2.
    gate1_nm : float
        Distance h_1 between the electron layer and the metal plane above strip 1, in nm.
    gate2_nm : float
        Distance h_2 between the electron layer and the metal plane above strip 2, in nm.
    eps_substrate : float
        Permittivity eps_s of the semiconductor below the electron layer.
    eps_spacer1 : float
        Permittivity kappa_1 of the spacer between the electron layer and the metal plane above strip 1.
    eps_spacer2 : float
        Permittivity kappa_2 of the spacer between the electron layer and the metal plane above strip 2.
    mass_ratio : float
        Electron effective mass in units of the free-electron mass.

    Returns
    -------
    cos_kl : float
        cos(k L) of the Bloch wave at this frequency; values outside [-1, 1] mark a band gap.

    Raises
    ------
    ValueError
        If any physical argument is not a positive finite number or if fill is not
        strictly between 0 and 1.
    '''
    return cos_kl
```

### Step 3

zone_center_modes

Goal
----
Frequencies of the lowest bright and dark zone-centre plasma modes of the two-strip plasmonic crystal.

```python
import numpy as np

def zone_center_modes(period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float, n_modes: int) -> "np.ndarray":
    '''Lowest bright and dark zone-centre plasma frequencies of the crystal.

    Parameters
    ----------
    period_um : float
        Period L of the crystal in micrometres.
    fill : float
        Filling factor f, the fraction of each period occupied by strip 1; 0 < f < 1.
    density1 : float
        Electron density N_1 of strip 1 in cm^-2.
    density2 : float
        Electron density N_2 of strip 2 in cm^-2.
    gate1_nm : float
        Distance h_1 between the electron layer and the metal plane above strip 1, in nm.
    gate2_nm : float
        Distance h_2 between the electron layer and the metal plane above strip 2, in nm.
    eps_substrate : float
        Permittivity eps_s of the semiconductor below the electron layer.
    eps_spacer1 : float
        Permittivity kappa_1 of the spacer between the electron layer and the metal plane above strip 1.
    eps_spacer2 : float
        Permittivity kappa_2 of the spacer between the electron layer and the metal plane above strip 2.
    mass_ratio : float
        Electron effective mass in units of the free-electron mass.
    n_modes : int
        Number of modes of each family to return; at least 1.

    Returns
    -------
    frequencies : numpy.ndarray
        Array of shape (2, n_modes); row 0 bright, row 1 dark, in GHz, each row increasing.

    Raises
    ------
    ValueError
        If any physical argument is not a positive finite number, if fill is not
        strictly between 0 and 1, or if n_modes is not a positive integer.
    '''
    return frequencies
```

### Step 4

plasmon_effective_mass

Goal
----
Effective mass of the fundamental bright or dark plasmon of the two-strip plasmonic crystal.

```python
import numpy as np

def plasmon_effective_mass(period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float, kind: str) -> float:
    '''Zone-centre effective mass of the fundamental bright or dark plasmon.

    Parameters
    ----------
    period_um : float
        Period L of the crystal in micrometres.
    fill : float
        Filling factor f, the fraction of each period occupied by strip 1; 0 < f < 1.
    density1 : float
        Electron density N_1 of strip 1 in cm^-2.
    density2 : float
        Electron density N_2 of strip 2 in cm^-2.
    gate1_nm : float
        Distance h_1 between the electron layer and the metal plane above strip 1, in nm.
    gate2_nm : float
        Distance h_2 between the electron layer and the metal plane above strip 2, in nm.
    eps_substrate : float
        Permittivity eps_s of the semiconductor below the electron layer.
    eps_spacer1 : float
        Permittivity kappa_1 of the spacer between the electron layer and the metal plane above strip 1.
    eps_spacer2 : float
        Permittivity kappa_2 of the spacer between the electron layer and the metal plane above strip 2.
    mass_ratio : float
        Electron effective mass in units of the free-electron mass.
    kind : str
        Either "bright" or "dark".

    Returns
    -------
    mass : float
        Signed effective plasmon mass M in units of 1e-4 m*.

    Raises
    ------
    ValueError
        If kind is not "bright" or "dark", if any physical argument is not a positive
        finite number, or if fill is not strictly between 0 and 1.
    '''
    return mass
```

### Step 5

degeneracy_point

Goal
----
Largest filling factor, and the common frequency there, at which the fundamental bright and dark zone-centre plasmons of the crystal coincide.

```python
import numpy as np

def degeneracy_point(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
                     eps_substrate: float, eps_spacer1: float, eps_spacer2: float, mass_ratio: float) -> "np.ndarray":
    '''Largest filling factor at which the fundamental bright and dark modes coincide.

    Parameters
    ----------
    period_um : float
        Period L of the crystal in micrometres.
    density1 : float
        Electron density N_1 of strip 1 in cm^-2.
    density2 : float
        Electron density N_2 of strip 2 in cm^-2.
    gate1_nm : float
        Distance h_1 between the electron layer and the metal plane above strip 1, in nm.
    gate2_nm : float
        Distance h_2 between the electron layer and the metal plane above strip 2, in nm.
    eps_substrate : float
        Permittivity eps_s of the semiconductor below the electron layer.
    eps_spacer1 : float
        Permittivity kappa_1 of the spacer between the electron layer and the metal plane above strip 1.
    eps_spacer2 : float
        Permittivity kappa_2 of the spacer between the electron layer and the metal plane above strip 2.
    mass_ratio : float
        Electron effective mass in units of the free-electron mass.

    Returns
    -------
    point : numpy.ndarray
        Array [f_star, f0_ghz] with 0 < f_star < 1 and f0_ghz in GHz.

    Raises
    ------
    ValueError
        If any argument is not a positive finite number, or if the fundamental bright
        and dark frequencies do not coincide anywhere in 0 < f < 1.
    '''
    return point
```

### Step 6

degeneracy_linear_velocity

Goal
----
Speed of the linearly dispersing plasmon branches at the largest bright-dark coincidence of the crystal.

```python
import numpy as np

def degeneracy_linear_velocity(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
                     eps_substrate: float, eps_spacer1: float, eps_spacer2: float, mass_ratio: float) -> float:
    '''Speed of the linear plasmon branches at the largest bright-dark coincidence.

    Parameters
    ----------
    period_um : float
        Period L of the crystal in micrometres.
    density1 : float
        Electron density N_1 of strip 1 in cm^-2.
    density2 : float
        Electron density N_2 of strip 2 in cm^-2.
    gate1_nm : float
        Distance h_1 between the electron layer and the metal plane above strip 1, in nm.
    gate2_nm : float
        Distance h_2 between the electron layer and the metal plane above strip 2, in nm.
    eps_substrate : float
        Permittivity eps_s of the semiconductor below the electron layer.
    eps_spacer1 : float
        Permittivity kappa_1 of the spacer between the electron layer and the metal plane above strip 1.
    eps_spacer2 : float
        Permittivity kappa_2 of the spacer between the electron layer and the metal plane above strip 2.
    mass_ratio : float
        Electron effective mass in units of the free-electron mass.

    Returns
    -------
    speed : float
        s_0 in cm/s.

    Raises
    ------
    ValueError
        If any argument is not a positive finite number, or if no coincidence exists.
    '''
    return speed
```

### Step 7

bright_mass_slope_at_degeneracy

Goal
----
Rate at which the effective mass of the fundamental bright plasmon changes with filling factor at the largest bright-dark coincidence.

```python
import numpy as np

def bright_mass_slope_at_degeneracy(period_um: float, density1: float, density2: float, gate1_nm: float, gate2_nm: float,
                     eps_substrate: float, eps_spacer1: float, eps_spacer2: float, mass_ratio: float) -> float:
    '''Derivative of the fundamental bright plasmon mass with filling factor at the largest coincidence.

    Parameters
    ----------
    period_um : float
        Period L of the crystal in micrometres.
    density1 : float
        Electron density N_1 of strip 1 in cm^-2.
    density2 : float
        Electron density N_2 of strip 2 in cm^-2.
    gate1_nm : float
        Distance h_1 between the electron layer and the metal plane above strip 1, in nm.
    gate2_nm : float
        Distance h_2 between the electron layer and the metal plane above strip 2, in nm.
    eps_substrate : float
        Permittivity eps_s of the semiconductor below the electron layer.
    eps_spacer1 : float
        Permittivity kappa_1 of the spacer between the electron layer and the metal plane above strip 1.
    eps_spacer2 : float
        Permittivity kappa_2 of the spacer between the electron layer and the metal plane above strip 2.
    mass_ratio : float
        Electron effective mass in units of the free-electron mass.

    Returns
    -------
    slope : float
        dM_b/df at f = f*, in units of 1e-4 m*.

    Raises
    ------
    ValueError
        If any argument is not a positive finite number, if no coincidence exists, or
        if the bright mass does not cross zero linearly at f*.
    '''
    return slope
```

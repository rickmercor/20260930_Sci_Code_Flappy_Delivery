# Material_Science-Semiconductor_Materials-71

## Background

Ultra-wide-gap oxides sit near the historical boundary between semiconductors and insulators. Gap size alone is a weak classifier: a material remains device-relevant only if dopants can ionize and carriers stay mobile. That requirement matters for compact power electronics and deep-ultraviolet optoelectronics, where larger gaps raise the dielectric breakdown field but often deepen impurities and strengthen lattice localization.

Prior materials searches frequently emphasized band alignment and native-defect compensation under equilibrium growth. Those checks are important, yet they do not by themselves answer whether an intentionally introduced impurity remains shallow or whether the freed carrier stays delocalized once ionic polarization is included. Continuum descriptors built from near-edge band masses and dielectric screening close that gap: ionization depends on how strongly the static lattice screens a light or heavy mass, while polaron binding depends on the residual ionic response after electronic screening is removed. The same mass also sets the scattering-limited mobility that enters unipolar power ranking, and the gap enters through a mapped critical field.

The computational role of the recent extreme-gap oxide screening framework is therefore to convert sparse near-edge band samples and measured or computed dielectrics into a gated materials prioritization score. Candidates that fail the framework ionization ceiling or whose continuum polaron binding exceeds the thermal-energy acceptance rule are discarded; survivors are ranked by the unipolar epsilon-mu-E_c^3 product with thermal-ionization attenuation. This yields a deterministic, physically interpretable filter before heavier first-principles defect and transport calculations.

## Problem

Ultra-wide-gap oxides challenge the usual gap-based split between semiconductors and insulators: useful devices still require ionizable dopants and delocalized carriers when the gap exceeds that of AlN. Continuum screening frameworks for extreme-gap oxides convert near-edge band samples and dielectric responses into a materials prioritization score for power-electronics relevance.

Compute the deterministic prioritization score P for one n-type candidate. Configuration inputs: nonparabolicity alpha = 0.38 eV^-1; crystal momenta k = (0.048, 0.052, 0.045) A^-1 with energies above the conduction-band edge E = (0.0393135912340984, 0.024891847042767198, 0.02465676303235815) eV; static and high-frequency relative permittivities epsilon_s = 12.4 and epsilon_inf = 5.1; gap E_g = 8.40 eV; scattering time tau = 1.4e-14 s; thermal energy kT = 0.025 eV; critical-field map E_c = 1.55*(E_g/6.2)^2.5 MV/cm. Physical constants: hbar^2/m_e = 7.6199642 eV*A^2, e = 1.602176634e-19 C, and m_e = 9.1093837015e-31 kg.

Using the extreme-gap continuum screening framework, recover the carrier-mass and dielectric descriptors required by that framework, evaluate its continuum ionization and polaron quantities, form the drift mobility in cm^2/V*s, evaluate E_c from the supplied map, and return the gated prioritization score P. Retrieve the framework ionization and polaron acceptance criteria. For survivors, assemble P as epsilon_s * mu * E_c^3 attenuated by exp(-E_ion/kT).

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

01_directional_band_masses

Goal
----
Recover directional carrier masses from near-edge nonparabolic band samples.

```python
def directional_band_masses(
    k_values: "np.ndarray",
    energy_values: "np.ndarray",
    alpha: float,
    hbar2_over_me: float = 7.6199642,
) -> "np.ndarray":
    """Return directional m*/m_e for each near-edge sample.

    Raises
    ------
    ValueError
        If inputs are mismatched or empty, alpha or hbar2_over_me is non-positive,
        any k or energy is non-positive, or the inversion denominator is non-positive.
    """
    return []
```

### Step 2

02_dos_geometric_mass

Goal
----
Reduce directional masses to one density-of-states mass.

```python
def dos_geometric_mass(directional_masses: "np.ndarray") -> float:
    """Return the geometric-mean density-of-states mass in units of m_e.

    Raises
    ------
    ValueError
        If directional_masses is not a non-empty 1D array of positive values.
    """
    return 0.0
```

### Step 3

03_continuum_ionization_energy

Goal
----
Evaluate the continuum dopant ionization energy.

```python
def continuum_ionization_energy(
    m_star_over_me: float,
    epsilon_s: float,
    rydberg_eV: float = 13.6,
) -> float:
    """Return continuum ionization energy in eV.

    Raises
    ------
    ValueError
        If m_star_over_me, epsilon_s, or rydberg_eV is non-positive.
    """
    return 0.0
```

### Step 4

04_ionic_electronic_dielectric

Goal
----
Build the effective dielectric from ionic and electronic responses.

```python
def ionic_electronic_dielectric(epsilon_s: float, epsilon_inf: float) -> float:
    """Return the effective dielectric constant from epsilon_s and epsilon_inf.

    Raises
    ------
    ValueError
        If permittivities are non-positive or epsilon_s does not exceed epsilon_inf.
    """
    return 0.0
```

### Step 5

05_continuum_polaron_energy

Goal
----
Evaluate continuum polaron formation energy.

```python
def continuum_polaron_energy(
    m_star_over_me: float,
    epsilon_eff: float,
    e_ha: float = 27.2,
) -> float:
    """Return continuum polaron formation energy in eV.

    Raises
    ------
    ValueError
        If m_star_over_me, epsilon_eff, or e_ha is non-positive.
    """
    return 0.0
```

### Step 6

06_scattering_time_mobility

Goal
----
Convert scattering time and mass into drift mobility.

```python
def scattering_time_mobility(
    m_star_over_me: float,
    tau: float,
    elementary_charge: float = 1.602176634e-19,
    electron_mass_kg: float = 9.1093837015e-31,
) -> float:
    """Return drift mobility in cm^2/V*s.

    Raises
    ------
    ValueError
        If mass, tau, or physical constants are non-positive.
    """
    return 0.0
```

### Step 7

07_gap_scaled_critical_field

Goal
----
Map band gap onto a critical breakdown field.

```python
def gap_scaled_critical_field(
    band_gap_eV: float,
    e_c_ref: float = 1.55,
    e_g_ref: float = 6.2,
    exponent: float = 2.5,
) -> float:
    """Return gap-scaled critical field in MV/cm.

    Raises
    ------
    ValueError
        If band_gap_eV or reference parameters are non-positive.
    """
    return 0.0
```

### Step 8

08_prioritization_score

Goal
----
Assemble the gated, thermally attenuated prioritization score P.

```python
def prioritization_score(
    epsilon_s: float,
    mobility: float,
    e_c: float,
    e_ion: float,
    e_pol: float,
    kT: float = 0.025,
    ion_ceiling_eV: float = 0.21,
    pol_ceiling_eV: float = 0.025,
) -> float:
    """Return prioritization score P.

    Raises
    ------
    ValueError
        If epsilon_s, e_c, kT, or ceilings are non-positive, or mobility is negative.
    """
    return 0.0
```

### Step 9

09 - Orchestrate extreme-gap prioritization

Goal
----
Run the full continuum prioritization pipeline end to end.

```python
def orchestrate_extreme_gap_prioritization(
    k_values: "np.ndarray",
    energy_values: "np.ndarray",
    alpha: float,
    epsilon_s: float,
    epsilon_inf: float,
    band_gap_eV: float,
    tau: float,
    kT: float = 0.025,
    hbar2_over_me: float = 7.6199642,
    rydberg_eV: float = 13.6,
    e_ha: float = 27.2,
    elementary_charge: float = 1.602176634e-19,
    electron_mass_kg: float = 9.1093837015e-31,
    e_c_ref: float = 1.55,
    e_g_ref: float = 6.2,
    exponent: float = 2.5,
    ion_ceiling_eV: float = 0.21,
    pol_ceiling_eV: float = 0.025,
) -> float:
    """Return P assembled by calling the earlier sub-problem functions.

    Raises
    ------
    ValueError
        Propagated from invalid upstream inputs.
    """
    return 0.0
```

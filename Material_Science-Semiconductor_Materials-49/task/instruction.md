# Material_Science-Semiconductor_Materials-49

## Background

Avalanche breakdown in a one sided abrupt junction is governed by the ionisation integral taken
across the depleted drift region. Measuring $x$ from the blocking junction of an n-type drift layer,
across which electrons drift away from the junction towards the stop layer and holes drift back
towards it, the condition for the multiplication factor to diverge with both carrier species active is

$$\int_0^{W}\alpha_n\,\exp\!\left(-\int_0^{x}(\alpha_n-\alpha_p)\,dx'\right)dx = 1,$$

with Chynoweth coefficients $\alpha_n = a_n\exp(-b_n/E)$ and $\alpha_p = a_p\exp(-b_p/E)$. The nested
exponential prevents a closed form in this two carrier form, which is why the source replaces the
pair by a single effective coefficient and reduces the condition, for a triangular field profile, to
$e^{-\zeta}/\zeta - E_1(\zeta) = 1/\phi$, where $\zeta$ is the effective field scale divided by the
peak field and $\phi$ is a dimensionless group of the effective coefficients, the permittivity and the
doping, and then fits that relation rather than solving it. Device design has otherwise leaned on
material specific power law fits whose parameters carry no physical meaning and have to be
recalibrated for every new semiconductor.

For a one sided abrupt junction the depleted field falls linearly,
$E(x) = qN_D(W-x)/\varepsilon$, so the peak field at the blocking junction is
$E_{cr} = qN_DW/\varepsilon$. In a non punch-through layer the field reaches zero at the far edge and
the profile is triangular, so the blocking voltage is $\tfrac12 E_{cr}W$. A punch-through layer
terminates on a heavily doped stop layer before the field has decayed, leaving a trapezoidal profile
running from $E_{cr}$ down to a finite terminal field $E_1$, so the blocking voltage becomes
$\tfrac12 (E_{cr}+E_1)W$. Punch-through therefore stores the same voltage in a thinner, more lightly
doped layer, which is why it is the preferred geometry when conduction loss matters.

The drift contribution to specific on-resistance under complete ionisation is
$R_{sp} = W/(q\mu_n N_D)$. Because mobility itself degrades as ionised impurity scattering grows,
holding it at its lightly doped plateau flatters the design at low voltage ratings, where the
optimum doping is highest. A Caughey-Thomas form captures the degradation for 4H-SiC at room
temperature.

Donor ionisation is not complete at room temperature. Nitrogen substitutes on two inequivalent
lattice sites in 4H-SiC in equal numbers, one shallow and one roughly twice as deep, and the
occupied fraction of each follows the usual donor statistics, so the free electron density is
set by charge neutrality across both sites and must be solved for rather than assumed. The gap between $n$ and $N_D$ widens rapidly with doping, so it is
mild at multi-kilovolt ratings and severe at the low ratings where the optimum doping is highest.

## Problem

A unipolar 4H-SiC power device blocks voltage across a lightly doped n-type drift layer, and that
layer sets both the blocking capability and the conduction loss. Raising the doping thins the layer
and lowers its resistance but brings on avalanche breakdown sooner, so a rating fixes a trade-off
rather than a single free choice. A punch-through layer sits on a heavily doped stop layer, so at
breakdown the field has not decayed to zero at the far edge and the profile is trapezoidal rather
than triangular. That leaves the designer two coupled degrees of freedom, the doping and the
thickness, and for a given rating one pairing minimises the drift specific on-resistance. The layer that carries current in the on state is only
partly ionised at room temperature, and carries correspondingly fewer electrons than there are
donors.

Work at room temperature with kT = 0.025852 eV. Avalanche breakdown is set by the two carrier
ionisation integral over the drift field, with electron and hole ionisation coefficients of
Chynoweth form, prefactors a_n = 8.2e9 cm^-1 and a_p = 4.5e6 cm^-1 and field scales b_n = 39.4 MV/cm
and b_p = 12.8 MV/cm. Take the relative permittivity as 9.7, the elementary charge as 1.602e-19 C
and the vacuum permittivity as 8.854e-14 F/cm. The nitrogen donors occupy two inequivalent lattice
sites in equal numbers with activation energies of 60 meV and 120 meV, the donor degeneracy factor
is 2, and the conduction band effective density of states is 1.7e19 cm^-3. The room temperature
electron mobility falls as 40 + 910/(1 + (N_D/2e17)^0.61) cm^2 V^-1 s^-1 with N_D in cm^-3.

Determine the minimum drift specific on-resistance of a punch-through 4H-SiC drift layer rated to
block 650 V, in milliohm centimetre squared. The doping and the thickness are yours to choose:
minimise the drift specific on-resistance subject to blocking that rating, with the doping-dependent
mobility and the incomplete donor ionisation inside the quantity being minimised rather than applied to
a design fixed beforehand, and with the depleted layer carrying the full chemical doping as space
charge. Keep the electron and hole coefficients separate in the full two carrier ionisation integral
and solve that breakdown condition to full double precision at every doping you evaluate, rather than
through an effective coefficient or an empirical fit. Report three comparisons, each with the
percentage by which it differs from that minimum: the minimum found instead by following the source in
collapsing the two carrier coefficients onto a single effective Chynoweth pair, a_eff = sqrt(a_n a_p)
and b_eff = (b_n + b_p)/2, with its transcendental condition solved exactly; and the optimised design
re-evaluated with every donor fully ionised and, separately, with the mobility held at its lightly
doped plateau. In your reasoning report the critical field at breakdown, the selected doping,
the drift thickness, the free electron density and its ionised fraction, and the mobility. Also record
from the earlier literature the minimum specific on-resistance that the updated room-temperature
trade-off study of 4H-SiC unipolar devices quotes at a 1 kV rating, how much lower the optimum doping
and thickness of a punch-through layer came out than those of the non-punch-through optimum in the
4H-SiC drift-design study built on a fitted Konstantinov critical-field relation, and how the
ionisation coefficients used here were measured, naming the device arrangement and the temperature
range. All of these belong in the reasoning even though it is kept short.

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

Implement **all 12 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_effective_ionization_parameters

Goal
----
Collapse the separate electron and hole ionization coefficients into one effective pair.

```python
def effective_ionization_parameters(a_n: float, b_n: float, a_p: float, b_p: float) -> np.ndarray:
    '''Effective Chynoweth prefactor and field scale for the two carrier species.

    Parameters
    ----------
    a_n, a_p : float
        Electron and hole ionization prefactors in cm^-1.
    b_n, b_p : float
        Electron and hole ionization field scales in V/cm.

    Returns
    -------
    numpy.ndarray
        Two element array [a_eff, b_eff]; a_eff in cm^-1, b_eff in V/cm.

    Raises
    ------
    ValueError
        If any of a_n, b_n, a_p, b_p is not finite or is not strictly positive.
    '''
    return [0.0, 0.0]  # placeholder
```

### Step 2

02_ionization_dimensionless_group

Goal
----
Form the dimensionless group that controls the breakdown condition.

```python
def ionization_dimensionless_group(a_eff: float, b_eff: float, eps_r: float, doping: float) -> float:
    '''Dimensionless group controlling the avalanche condition.

    Parameters
    ----------
    a_eff : float
        Effective prefactor in cm^-1.
    b_eff : float
        Effective field scale in V/cm.
    eps_r : float
        Relative permittivity of the semiconductor.
    doping : float
        Drift region doping in cm^-3.

    Returns
    -------
    float
        The dimensionless group.

    Raises
    ------
    ValueError
        If any of a_eff, b_eff, eps_r, doping is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder
```

### Step 3

03_avalanche_zeta

Goal
----
Solve the breakdown condition exactly for its dimensionless field variable.

```python
def avalanche_zeta(phi: float) -> float:
    '''Dimensionless field variable satisfying the exact avalanche condition.

    Parameters
    ----------
    phi : float
        The dimensionless group from the previous step.

    Returns
    -------
    float
        The dimensionless field variable (no units).

    Raises
    ------
    ValueError
        If phi is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder
```

### Step 4

04_critical_field

Goal
----
Convert the dimensionless field variable into the peak field at breakdown.

```python
def critical_field(b_eff: float, zeta: float) -> float:
    '''Peak electric field at avalanche breakdown.

    Parameters
    ----------
    b_eff : float
        Effective field scale in V/cm.
    zeta : float
        Dimensionless field variable.

    Returns
    -------
    float
        Critical field in V/cm.

    Raises
    ------
    ValueError
        If any of b_eff, zeta is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder
```

### Step 5

05_punchthrough_field_ratio

Goal
----
Field ratio at the punch-through design that makes the constant-mobility, complete-ionisation drift resistance stationary.

```python
def punchthrough_field_ratio(zeta: float) -> float:
    '''Terminal to peak field ratio that makes the constant-mobility, complete-ionisation drift resistance stationary.

    Parameters
    ----------
    zeta : float
        Dimensionless field variable.

    Returns
    -------
    float
        The field ratio (no units).

    Raises
    ------
    ValueError
        If zeta is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder
```

### Step 6

06_punchthrough_depletion_width

Goal
----
Drift thickness of the trapezoidal punch-through design.

```python
def punchthrough_depletion_width(eps_r: float, doping: float, e_crit: float, eta: float) -> float:
    '''Drift region thickness of the punch-through design.

    Parameters
    ----------
    eps_r : float
        Relative permittivity.
    doping : float
        Drift doping in cm^-3.
    e_crit : float
        Critical field in V/cm.
    eta : float
        Terminal to peak field ratio.

    Returns
    -------
    float
        Drift thickness in cm.

    Raises
    ------
    ValueError
        If any of eps_r, doping, e_crit is not finite or is not strictly positive.
        ``eta`` outside ``[0, 1)`` is rejected: a terminal field at or above the
        peak field is not a punch-through profile.
    '''
    return 0.0  # placeholder
```

### Step 7

07_punchthrough_breakdown_voltage

Goal
----
Blocking voltage supported by the trapezoidal profile.

```python
def punchthrough_breakdown_voltage(eps_r: float, doping: float, e_crit: float, eta: float) -> float:
    '''Blocking voltage of the punch-through design.

    Parameters
    ----------
    eps_r : float
        Relative permittivity.
    doping : float
        Drift doping in cm^-3.
    e_crit : float
        Critical field in V/cm.
    eta : float
        Terminal to peak field ratio.

    Returns
    -------
    float
        Blocking voltage in V.

    Raises
    ------
    ValueError
        If any of eps_r, doping, e_crit is not finite or is not strictly positive.
        ``eta`` outside ``[0, 1)`` is rejected: a terminal field at or above the
        peak field is not a punch-through profile.
    '''
    return 0.0  # placeholder
```

### Step 8

08_field_ratio_at_rating

Goal
----
Terminal to peak field ratio forced on the punch-through drift layer by its voltage rating.

```python
def field_ratio_at_rating(eps_r: float, doping: float, e_crit: float, bv_target: float) -> float:
    '''Terminal-to-peak field ratio that makes the trapezoidal profile block bv_target.

    Parameters
    ----------
    eps_r : float
        Relative permittivity of the semiconductor.
    doping : float
        Drift doping in cm^-3.
    e_crit : float
        Peak field at breakdown in V/cm.
    bv_target : float
        Blocking voltage the design must support in V.

    Returns
    -------
    float
        Field ratio in [0, 1).

    Raises
    ------
    ValueError
        If any of eps_r, doping, e_crit, bv_target is not finite or is not strictly positive.
        A doping that cannot reach bv_target before the field ratio falls to zero is rejected.
    '''
    return 0.0  # placeholder
```

### Step 9

09_free_electron_density

Goal
----
Free electron density in the neutral on-state drift layer under incomplete donor ionisation.

```python
def free_electron_density(doping: float, n_c: float, degeneracy: float, ea_shallow: float, ea_deep: float, kt: float) -> float:
    '''Free electron density under incomplete ionisation of two donor sites.

    Parameters
    ----------
    doping : float
        Total chemical donor density in cm^-3.
    n_c : float
        Conduction band effective density of states in cm^-3.
    degeneracy : float
        Donor degeneracy factor.
    ea_shallow, ea_deep : float
        Activation energies of the two donor sites in eV.
    kt : float
        Thermal energy in eV.

    Returns
    -------
    float
        Free electron density in cm^-3.

    Raises
    ------
    ValueError
        If any of doping, n_c, degeneracy, kt is not finite or is not strictly positive.
        Activation energies must be finite and non-negative.
    '''
    return 0.0  # placeholder
```

### Step 10

10_drift_electron_mobility

Goal
----
Doping dependent electron mobility of the 4H-SiC drift layer at room temperature.

```python
def drift_electron_mobility(doping: float) -> float:
    '''Room temperature electron mobility of the 4H-SiC drift layer.

    Parameters
    ----------
    doping : float
        Drift doping in cm^-3.

    Returns
    -------
    float
        Electron mobility in cm^2 V^-1 s^-1.

    Raises
    ------
    ValueError
        If doping is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder
```

### Step 11

11_two_carrier_critical_field

Goal
----
Critical field at avalanche breakdown with the electron and hole ionisation coefficients kept separate.

```python
def two_carrier_critical_field(a_n: float, b_n: float, a_p: float, b_p: float, eps_r: float, doping: float) -> float:
    '''Peak field at which the electron-initiated two-carrier ionisation integral reaches one.

    Parameters
    ----------
    a_n, a_p : float
        Electron and hole ionisation prefactors in cm^-1.
    b_n, b_p : float
        Electron and hole ionisation field scales in V/cm.
    eps_r : float
        Relative permittivity.
    doping : float
        Drift layer doping in cm^-3.

    Returns
    -------
    float
        Critical field in V/cm, converged to a relative precision of at least 1e-11.

    Raises
    ------
    ValueError
        If any of a_n, b_n, a_p, b_p, eps_r, doping is not finite or is not strictly positive.
    '''
    return 0.0  # placeholder
```

### Step 12

12_minimum_specific_on_resistance

Goal
----
Minimum drift specific on-resistance of the optimised punch-through design with the two carrier coefficients kept separate.

```python
def minimum_specific_on_resistance(a_n: float, b_n: float, a_p: float, b_p: float, eps_r: float, bv_target: float,
                                   n_c: float, degeneracy: float, ea_shallow: float, ea_deep: float, kt: float) -> float:
    '''Minimum drift specific on-resistance of the optimised punch-through design.

    Parameters
    ----------
    a_n, a_p : float
        Ionization prefactors in cm^-1.
    b_n, b_p : float
        Ionization field scales in V/cm.
    eps_r : float
        Relative permittivity.
    bv_target : float
        Target blocking voltage in V.
    n_c : float
        Conduction band effective density of states in cm^-3.
    degeneracy : float
        Donor degeneracy factor.
    ea_shallow, ea_deep : float
        Activation energies of the two donor sites in eV.
    kt : float
        Thermal energy in eV.

    Returns
    -------
    float
        Specific on-resistance in mOhm cm^2.

    Raises
    ------
    ValueError
        If any of a_n, b_n, a_p, b_p, eps_r, bv_target, n_c, degeneracy, kt is not finite or is not strictly positive.
        Activation energies must be finite and non-negative.
    '''
    return 0.0  # placeholder
```

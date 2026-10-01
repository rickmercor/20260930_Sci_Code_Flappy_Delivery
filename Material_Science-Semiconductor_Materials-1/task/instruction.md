# Material_Science-Semiconductor_Materials-1

## Background

Wurtzite group-III nitrides lack a centre of symmetry, so each unit cell carries an electric dipole along the c axis even with every atom at its equilibrium position. The macroscopic consequence is a spontaneous polarization, and straining the crystal adds a piezoelectric contribution on top of it. When a thin AlGaN layer is grown coherently on a relaxed GaN buffer the two materials have different total polarization, and the divergence of the polarization field at the interface appears as a fixed areal charge. That charge is what draws a dense electron gas to the interface in an undoped heterostructure, which is the mechanism behind the high electron mobility transistors used in power switching and in radio-frequency amplifiers.

Because the electron gas is induced electrostatically rather than by dopants, its density is set by a chain of quantities that are not measured directly: the polarization of each layer, the conduction band discontinuity, the thickness of the barrier, and the height of the metal barrier formed on the free surface. Groups working on these devices routinely invert that chain, extracting a surface barrier height from a measured sheet density or threshold voltage. The extracted numbers have long been uncomfortable, coming out well above the barrier heights reported for the same metals by direct measurement, and the discrepancy has usually been absorbed by postulating a large density of deep levels at the surface.

An alternative reading is that the electrostatic model itself is too coarse. Treating the polarization charge as a single sheet located exactly on the interface throws away the fact that polarization is a property of dipoles distributed through each unit cell, and the two facing bound sheets are physically displaced from one another by roughly one lattice constant along c. That displacement is small compared with a barrier thickness but not negligible compared with the distances that set the band profile near the interface, and carrying it through changes what a given measured density implies about the surface barrier. Distinguishing a genuine electrical defect population from an artefact of an over-idealised electrostatic model matters for how these devices are designed and how their surfaces are passivated.

The channel itself is a quantum well: the interface field confines the electrons within a few nanometres of the heterointerface, so the Fermi level sits a substantial fraction of an electronvolt above the conduction band edge at the interface and rises with the sheet density. The standard way to obtain it is a self-consistent Schrodinger-Poisson calculation, in which the envelope functions of the subbands are found in the band profile that the electron charge itself produces, the subbands are filled with the two-dimensional density of states, and the two parts are iterated to consistency. Variational triangular-well formulas approximate this calculation; the calculation itself has no closed form and is defined by its discretisation.

## Problem

A Ga-polar AlGaN/GaN heterostructure grown on a thick relaxed GaN buffer confines a two-dimensional electron gas at the interface with no intentional doping anywhere in the structure. The confinement comes entirely from the step in polarization between the strained barrier and the relaxed buffer, which leaves a net bound sheet charge at the interface, and the density of the electron gas is fixed by the electrostatics of that bound charge together with the metal barrier on the free surface. Extracting a surface barrier height from a measured density is how the electrostatic picture of these devices is tested, and the number extracted depends on how the bound charge and the channel are described.

Use the displaced-sheet description of the interface. Polarization originates in dipoles inside every unit cell, so the positive sheet that terminates the barrier and the negative sheet that begins the buffer are not coincident: place them a separation delta apart, symmetrically about the interface. At pinch-off this has two consequences. A thin charged slab appears between the two sheets, and the potential step across it is the field of the buffer's spontaneous polarization sheet integrated over each half-gap with the permittivity of the layer that half-gap lies in. The same displacement moves each bound sheet inward from the surface it terminates, so the net bound charge sustains its field over the barrier thickness reduced by the full separation. Away from pinch-off, describe the channel self-consistently rather than with a fixed setback: at zero gate bias the barrier acts as a capacitor across its whole thickness, because the free charges on the metal and in the channel do not move with the bound sheets, and the density it supports is set by the gate overdrive, the negative of the threshold voltage, minus the Fermi level the electrons themselves occupy. Measure that Fermi level from the conduction band edge at the interface and obtain it from a self-consistent Schrodinger-Poisson solution of the GaN side alone: electrons of effective mass 0.22 electron masses in a channel closed by an impenetrable wall at the interface, whose band edge is bent only by their own charge with the permittivity of GaN, the field at the wall fixed by Gauss's law to the electron sheet and vanishing deep in the buffer, and every subband below the Fermi level filled at zero temperature with the two-dimensional density of states. Define that solution on a uniform grid of 401 points spanning 40 nm from the interface, with the envelope functions vanishing at both ends of the box, the kinetic operator as the second-order central difference, each envelope normalised by its grid sum times the spacing, the field at grid point j as (q/eps_GaN) times the spacing times the sum of the electron density over the points k >= j, and the band edge at point j, measured from its value at the interface, as q times the spacing times the sum of the field over the points k < j, iterated until the Fermi level is converged to a nanoelectronvolt; the empty channel places the Fermi level at the ground level of the empty box.

Obtain every alloy quantity by linear interpolation in Al fraction from these binary values, and take the in-plane strain of a layer as (a_GaN - a(x))/a(x) with a(x) its own interpolated lattice constant: a-axis lattice constants 3.189 Angstrom for GaN and 3.112 Angstrom for AlN; spontaneous polarizations -0.034 and -0.090 C m^-2; piezoelectric constants e31 of -0.34 and -0.53 C m^-2 and e33 of 0.67 and 1.50 C m^-2; elastic constants C13 of 106 and 108 GPa and C33 of 398 and 373 GPa; relative permittivities 8.9 and 8.5. The barrier has a free top surface and the conduction band discontinuity is 0.301 eV; the sheet separation is not known in advance. Fix the metal barrier height at the literature value of 1.32 V for a nickel contact and fit the single separation, searched between 0 and 1.5 nm, that minimises the sum over the following as-grown structures of the squared relative difference between the model's zero-bias density and the measured one:

| Al fraction | barrier thickness (nm) | sheet density (10^12 cm^-2) |
|---|---|---|
| 0.22 | 26 | 8.28 |
| 0.26 | 21 | 9.55 |
| 0.30 | 17 | 10.6 |
| 0.33 | 14 | 11.1 |
| 0.36 | 12 | 11.5 |

Then turn to a separate as-grown wafer, Al(0.31)Ga(0.69)N of thickness 9.5 nm, whose non-contact eddy-current characterisation returns a sheet resistance of 690 ohm per square with an electron mobility of 1500 cm^2 V^-1 s^-1 at room temperature. Report the effective Schottky barrier height, in volts, at which the fitted displaced-sheet model reproduces that wafer's measured density. Alongside that number the reasoning should record the fitted sheet separation and the residuals of that fit, the wafer's measured density and the net bound charge at its interface, the barrier heights that a 5 Angstrom separation and a coincident-sheet treatment would each have required instead, the channel Fermi level with the subband levels beneath it and the threshold voltage of the wafer under the fitted model, the measured difference between the Schottky barrier heights of the two GaN polarities in the earlier polarity study from which the published value of the separation was taken, how the surface barrier height varied with barrier thickness in the earlier barrier-thickness series that the displaced-sheet model was first compared against, and where in energy the surface donor states identified in that same series lie. All of these must appear in the reasoning even though it is kept short.

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

Sheet carrier density from room-temperature transport

Goal
----
Recover the areal density of a two-dimensional electron gas from a non-contact transport measurement.

An as-grown heterostructure is characterised without any processing by an eddy-current measurement that
returns the sheet resistance together with the room-temperature Hall mobility. For a single carrier
species the sheet conductance is the product of the carrier charge, the areal density and the mobility.

```python
def sheet_carrier_density(sheet_resistance_ohm_per_sq: float, hall_mobility_cm2_per_Vs: float) -> float:
    """Sheet resistance in ohm/sq and Hall mobility in cm^2/(V s) -> 2DEG sheet carrier density in m^-2.
 
    Raises ValueError if either input is not strictly positive.
    """
    return 0.0  # placeholder
```

### Step 2

Total polarization of a layer grown coherently on relaxed GaN

Goal
----
Total polarization of a group-III nitride layer grown coherently on a thick relaxed GaN buffer.

The layer is forced to the in-plane lattice constant of the buffer while its own free-standing constant
follows the composition, so its in-plane strain is (a_GaN - a(x)) / a(x), measured against the layer's own
lattice constant a(x). Its polarization along the c axis has two parts: the spontaneous polarization the
material carries unstrained, and the part the strain induces. The layer has a free top surface, so it
carries no stress along the c axis. Every alloy constant is linear in Al fraction between the binary
endpoints. The buffer itself is the relaxed binary, so passing it in at zero Al fraction returns its
spontaneous value alone.

```python
def pseudomorphic_layer_polarization(al_fraction: float, gan: dict, aln: dict) -> float:
    """Al mole fraction and the GaN and AlN parameter dictionaries -> magnitude of the total
    polarization of the coherently strained layer in C/m^2.
 
    Raises ValueError if al_fraction is outside [0, 1], a key is missing, or a lattice constant or c33
    is not positive.
    """
    return 0.0  # placeholder
```

### Step 3

Pinch-off threshold voltage with the bound sheets displaced

Goal
----
Pinch-off threshold voltage of a gated AlGaN/GaN barrier when the two bound polarization sheets at the
heterointerface are displaced by a separation delta, symmetrically about the interface.

With the channel empty, walking the conduction band from the gate metal to the neutral buffer gives the
threshold voltage of the displaced-sheet model,

    V_th = phi_B - dE_C + |P_sp,GaN| (delta / 2) (1 / eps_b + 1 / eps_GaN) - sigma (d_B - delta) / eps_b,

with all terms in volts. phi_B is the metal barrier height, dE_C the conduction band offset in eV (it
enters directly in volts), |P_sp,GaN| the spontaneous polarization magnitude of the relaxed GaN buffer,
sigma the net bound interface charge (total polarization of the strained barrier minus |P_sp,GaN|), d_B the
barrier thickness, eps_b the absolute permittivity of the barrier (relative permittivity linear in Al
fraction) and eps_GaN that of the buffer. The third term is the potential step across the thin slab
between the two displaced sheets; the fourth is the field of the net bound charge over the reduced span
d_B - delta. Setting delta to zero recovers the coincident-sheet result.

```python
def displaced_sheet_threshold_voltage(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                                      barrier_height: float, conduction_band_offset_eV: float,
                                      gan: dict, aln: dict) -> float:
    """Displaced-sheet pinch-off threshold voltage in V for the given barrier and geometry.
 
    Raises ValueError if barrier_thickness <= 0, sheet_separation < 0 or >= barrier_thickness, or al_fraction
    is outside [0, 1].
    """
    return 0.0  # placeholder
```

### Step 4

Self-consistent Fermi level of the channel

Goal
----
Fermi level of the two-dimensional electron gas relative to the conduction band edge at the interface,
for a given sheet density, from a self-consistent Schrodinger-Poisson solution of the GaN channel.

The channel is the GaN side of the heterointerface, z >= 0, with the interface treated as an impenetrable
wall. Electrons of effective mass m* move in a conduction band that their own charge bends; the field at
the interface is tied to the electron sheet by Gauss's law and vanishes deep in the buffer, so no barrier
quantity enters. At zero temperature every subband below the Fermi level is occupied with the
two-dimensional density of states m* / (pi hbar^2), and the total density fixes E_F.

The object is defined on a uniform grid z_j = j h, j = 0 .. M-1, spanning a box of length L, h = L/(M-1).
The envelope functions vanish at both ends of the box, the kinetic operator is the second-order central
difference, and each envelope is normalised so that h * sum_j |psi_i(z_j)|^2 = 1. The electron density is
n(z_j) = sum_i N_i |psi_i(z_j)|^2 with N_i = (m* / (pi hbar^2)) (E_F - E_i) over the occupied subbands, and
the field and band edge follow from plain sums, F_j = (q / eps_GaN) h sum_(k >= j) n(z_k) and
E_C(z_j) - E_C(0) = q h sum_(k < j) F_k. The lowest four levels are computed, which covers every density
in this task. Schrodinger and Poisson are iterated to self-consistency; the converged Fermi level does not
depend on the mixing scheme and is reproducible to better than 1e-8 eV. At zero density the Fermi level
is placed at the ground level of the empty box.

```python
def channel_fermi_level(sheet_density: float, effective_mass_ratio: float, gan: dict,
                        box_length: float = 40e-9, grid_points: int = 401) -> float:
    """Self-consistent Schrodinger-Poisson Fermi level of the 2DEG above the interface band edge, in eV.
 
    Zero-temperature filling of the subbands of the hard-wall GaN channel on the uniform grid described in
    the step background. Raises ValueError if sheet_density < 0, effective_mass_ratio <= 0, box_length <= 0
    or grid_points < 5.
    """
    return 0.0  # placeholder
```

### Step 5

Self-consistent zero-bias sheet density

Goal
----
Sheet density of the two-dimensional electron gas at zero gate bias, solved self-consistently for the
displaced-sheet barrier.

At zero bias the channel holds the charge that the barrier, acting as a capacitor over its whole thickness
d_B, supports against the gate overdrive -V_th, reduced by the Fermi level the electrons themselves occupy.
The free charges, on the metal at the surface and in the channel at the interface, do not move with the
bound sheets, so the displacement enters only through V_th:

    n_s = eps_b ( -V_th - E_F(n_s) ) / ( q d_B ),

with V_th the displaced-sheet threshold voltage and E_F(n_s) the self-consistent Schrodinger-Poisson
Fermi level of the channel in volts, from the Fermi-level step at its default grid. Because E_F grows with
n_s, this is a fixed-point equation with a single root, to be solved numerically. When -V_th does not
exceed the Fermi level of the empty channel, E_F(0), the channel is empty and the density is zero.

```python
def self_consistent_sheet_density(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                                  barrier_height: float, conduction_band_offset_eV: float,
                                  effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    """Zero-bias 2DEG sheet density in m^-2 solved self-consistently with the Schrodinger-Poisson Fermi level.
 
    Returns 0.0 when -V_th does not exceed the empty-channel Fermi level. Raises ValueError for a non-positive
    thickness, an invalid sheet separation or a non-positive effective mass ratio.
    """
    return 0.0  # placeholder
```

### Step 6

Fit of the sheet separation to a set of as-grown structures

Goal
----
Fit the sheet separation of the displaced-sheet model to a set of as-grown structures.

Given several heterostructures of known Al fraction and barrier thickness with measured zero-bias sheet
densities, and a common metal barrier height, find the single sheet separation delta that minimises the
sum over structures of the squared relative residual

    sum_i ( n_model,i(delta) / n_meas,i - 1 )^2,

where n_model,i is the self-consistent sheet density of structure i. The search runs over
0 <= delta <= 1.5 nm, where the cost has one minimum; return that minimiser converged to at least
1e-15 m.

```python
def fit_sheet_separation(al_fractions: "np.ndarray", barrier_thicknesses: "np.ndarray", sheet_densities: "np.ndarray",
                         barrier_height: float, conduction_band_offset_eV: float, effective_mass_ratio: float,
                         gan: dict, aln: dict) -> float:
    """Least-squares (relative residual) sheet separation in m over 0 <= delta <= 1.5 nm.
 
    Raises ValueError if the arrays differ in length, are empty, or a measured density is not positive.
    """
    return 0.0  # placeholder
```

### Step 7

Barrier height required to reproduce a measured density

Goal
----
Metal barrier height that the displaced-sheet model, with a given sheet separation, requires in order to
reproduce a measured zero-bias sheet density.

The self-consistent sheet density falls monotonically as the barrier height rises, so the barrier height
that reproduces a measured density is the single root of n_model(phi_B) = n_meas. Search over
0.1 V <= phi_B <= 5.0 V and return the root converged to at least 1e-12 V.

```python
def required_barrier_height(al_fraction: float, barrier_thickness: float, sheet_separation: float,
                            target_sheet_density: float, conduction_band_offset_eV: float,
                            effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    """Barrier height in V at which the self-consistent displaced-sheet model reproduces the target density.
 
    Raises ValueError if target_sheet_density <= 0 or no root lies in [0.1, 5.0] V.
    """
    return 0.0  # placeholder
```

### Step 8

End-to-end barrier height of the as-grown wafer after fitting

Goal
----
End-to-end pipeline: fit the displaced-sheet model to a set of as-grown structures and report the barrier
height it then requires for a separately characterised as-grown sample.

Chain the earlier steps. Recover the sample's sheet density from its transport pair; fit the common sheet
separation to the structure set at the literature barrier height; then find the barrier height at which
the fitted, self-consistent displaced-sheet model reproduces the sample's measured density.

```python
def fitted_model_barrier_height(sheet_resistance_ohm_per_sq: float, hall_mobility_cm2_per_Vs: float,
                              al_fraction: float, barrier_thickness: float,
                              set_al_fractions: "np.ndarray", set_thicknesses: "np.ndarray", set_sheet_densities: "np.ndarray",
                              literature_barrier_height: float, conduction_band_offset_eV: float,
                              effective_mass_ratio: float, gan: dict, aln: dict) -> float:
    """Barrier height in V the fitted displaced-sheet model requires for the as-grown sample.
 
    Raises ValueError under the conditions of the chained steps: a non-positive sheet resistance or mobility,
    a non-positive barrier thickness, an invalid sheet separation, a non-positive effective mass ratio, or
    structure-set arrays that are empty, of unequal length or contain a non-positive density.
    """
    return 0.0  # placeholder
```

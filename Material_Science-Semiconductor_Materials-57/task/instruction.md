# Material_Science-Semiconductor_Materials-57

## Background

Two-dimensional magnetic semiconductors with opposite spin character on the valence and conduction edges make electrically reversible spin-polarized currents possible in a FET geometry. Contact work function and interface energetics decide whether each spin edge forms an Ohmic or Schottky alignment, while the applied biases set the chemical-potential window that enters the Landauer–Büttiker current. Device figures of merit such as spin filtering efficiency and subthreshold swing are then obtained from those spin-resolved currents without requiring a full first-principles transport rerun for a compact deterministic instance.

## Problem

Nanoscale spin transistors that use a two-dimensional magnetic semiconductor channel can deliver a gate-controlled, fully spin-polarized current when metal contacts set the band-edge alignment. Follow the paper’s interface and transport analysis: evaluate lattice mismatch and interfacial binding energy, obtain Schottky barriers from the zero-gate edge–Fermi alignment, map the source-referenced bias and gate into chemical potentials, build spin-resolved transmission from the gated opposite-spin band edges, compute Landauer–Büttiker spin currents and spin filtering efficiency, and report the subthreshold swing on the spin channel selected by the compact model’s admissibility conditions and the paper’s contact-type criteria.

Deterministic instance: a_electrode = 4.127 Å, n_electrode = 1, a_channel = 4.19 Å, n_channel = 1; E_het = -180.243 eV, E_electrode = -100.0 eV, E_channel = -80.0 eV, E_cp = 0.02 eV, A = 15.21 Å²; zero-bias, zero-gate edges E_vbm_up = 0.05 eV, E_cbm_down = 0.58 eV (EF = 0); T = 300 K; V_b = -0.2 V; uniform energy grid E ∈ [-1.5, 1.5] eV with 4001 samples including both endpoints; current scale I0 = 4300 µA/µm per eV of integrated transmission × occupation window; V_g1 = 0.20 V, V_g2 = 0.32 V. For this compact model, admit an interface when its binding energy is negative and its lattice mismatch, referenced to the channel matching-cell length, is at most 2.1%; these are task-defined admissibility conditions. The valence and conduction edges have opposite spin character and are shifted by the gate; the compact transmission is one at and below the spin-up valence edge and one at and above the spin-down conduction edge, and zero on the respective other sides. Evaluate the sampled Landauer integrals with the composite trapezoidal rule on the stated grid, without interpolating the edge positions. Reference the chemical potentials and the gate to the device Fermi level exactly as the paper does, and let the gate shift both edges rigidly with unit coupling.

In the supporting scientific reasoning, state the interfacial binding-energy and Schottky-barrier definitions, the source/drain chemical-potential and effective-gate assignments, the gated band-edge relation, the spin-resolved threshold spectra, the Landauer current expression, and the spin filtering efficiency definition. Evaluate the lattice mismatch, binding energy and barriers, and explain the contact-based selection of the scored spin channel. Include both selected-spin currents in µA/µm to at least two decimal places as intermediate calculations supporting the swing calculation.

The sole final answer is the selected-spin subthreshold swing between V_g1 and V_g2, expressed in mV/dec to at least two decimal places. The intermediate currents and supporting equations belong in the reasoning, not in the final answer.

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

01_lattice_mismatch

Goal
----
Compute percent lattice mismatch for electrode/channel matching cells.

```python
def lattice_mismatch(
    a_electrode: float,
    n_electrode: int,
    a_channel: float,
    n_channel: int,
) -> float:
    """Return lattice mismatch δ in percent.

    δ = 100*|n_electrode*a_electrode - n_channel*a_channel|/(n_channel*a_channel).

    Raises
    ------
    ValueError
        If any lattice constant is not positive, either repeat count is
        less than 1, or any input is non-finite.
    """
    return 0.0
```

### Step 2

02_binding_energy

Goal
----
Evaluate the BSSE-corrected interfacial binding energy per area.

```python
def binding_energy(
    E_het: float,
    E_electrode: float,
    E_channel: float,
    E_cp: float,
    area_A2: float,
) -> float:
    """Return Eb in meV/Å².

    Energies are in eV and area in Å². The returned value is in meV/Å².

    Raises
    ------
    ValueError
        If area_A2 is not positive or any input is non-finite.
    """
    return 0.0
```

### Step 3

03_schottky_barriers

Goal
----
Convert zero-gate opposite-spin edges into hole and electron Schottky barrier heights.

```python
def schottky_barriers(E_vbm_up: float, E_cbm_down: float) -> "np.ndarray":
    """Return [phi_p, phi_n], the hole and electron barriers in eV.

    The spin-up valence and spin-down conduction edge energies are in eV
    relative to EF = 0. An Ohmic alignment has zero injection barrier.

    Raises
    ------
    ValueError
        If either edge energy is non-finite.
    """
    return [0.0, 0.0]
```

### Step 4

04_terminal_potentials

Goal
----
Map source-referenced drain and gate voltages onto chemical potentials and an effective gate.

```python
def terminal_potentials(V_b: float, V_g: float) -> "np.ndarray":
    """Return [mu_s, mu_d, V_g_eff] with mu in eV and V_g_eff in V.

    Raises
    ------
    ValueError
        If V_b or V_g is non-finite.
    """
    return [0.0, 0.0, 0.0]
```

### Step 5

05_gated_band_edges

Goal
----
Rigidly shift zero-gate opposite-spin edges by the effective gate voltage.

```python
def gated_band_edges(E_vbm0: float, E_cbm0: float, V_g_eff: float) -> "np.ndarray":
    """Return [E_vbm0 - V_g_eff, E_cbm0 - V_g_eff] in eV.

    Raises
    ------
    ValueError
        If any input is non-finite.
    """
    return [0.0, 0.0]
```

### Step 6

06_threshold_spectra

Goal
----
Build opposite-spin edge-threshold transmission spectra on a fixed energy grid.

```python
def threshold_spectra(
    energies_eV: "np.ndarray",
    E_vbm: float,
    E_cbm: float,
) -> "np.ndarray":
    """Return stacked spectra T_up=1[E<=E_vbm], T_down=1[E>=E_cbm].

    Raises
    ------
    ValueError
        If the energy grid is not 1-D with at least two finite strictly
        increasing points, or either edge is non-finite.
    """
    return [[0.0], [0.0]]
```

### Step 7

07_landauer_spin_currents

Goal
----
Integrate spin-resolved Landauer–Büttiker currents from spectra and terminal chemical potentials.

```python
def landauer_spin_currents(
    energies_eV: "np.ndarray",
    T_spin: "np.ndarray",
    mu_s: float,
    mu_d: float,
    temperature_K: float,
    I0: float,
) -> "np.ndarray":
    """Return signed [I_up, I_down] in µA/µm.

    Raises
    ------
    ValueError
        If the energy grid is invalid, T_spin shape mismatches, temperature
        is not positive, or inputs are non-finite.
    """
    return [0.0, 0.0]
```

### Step 8

08_spin_filtering_efficiency

Goal
----
Convert spin-resolved currents into signed spin filtering efficiency.

```python
def spin_filtering_efficiency(I_up: float, I_down: float) -> float:
    """Return the spin filtering efficiency in percent.

    Raises
    ------
    ValueError
        If either current is non-finite or I_up + I_down is zero.
    """
    return 0.0
```

### Step 9

09_decade_swing

Goal
----
Evaluate the positive subthreshold-swing magnitude in mV/dec between two transfer points.

```python
def decade_swing(V_g1: float, V_g2: float, I1: float, I2: float) -> float:
    """Return the positive subthreshold-swing magnitude in mV/dec.

    The result is unchanged when the two transfer points are exchanged and
    is positive for either increasing or decreasing current magnitudes.

    Signed currents enter through their magnitudes |I1| and |I2|.

    Raises
    ------
    ValueError
        If inputs are non-finite, gate voltages are equal, currents are zero,
        or log10 magnitudes are equal.
    """
    return 0.0
```

### Step 10

10_orchestrate_device_swing

Goal
----
Assemble the full paper-equation pipeline and return the selected-spin subthreshold swing between two gate points for the deterministic instance.

```python
def orchestrate_device_swing(
    a_electrode: float = 4.127,
    n_electrode: int = 1,
    a_channel: float = 4.19,
    n_channel: int = 1,
    E_het: float = -180.243,
    E_electrode: float = -100.0,
    E_channel: float = -80.0,
    E_cp: float = 0.02,
    area_A2: float = 15.21,
    E_vbm_up: float = 0.05,
    E_cbm_down: float = 0.58,
    V_b: float = -0.2,
    temperature_K: float = 300.0,
    I0: float = 4300.0,
    E_min: float = -1.5,
    E_max: float = 1.5,
    n_energy: int = 4001,
    V_g1: float = 0.20,
    V_g2: float = 0.32,
) -> float:
    """Return the positive selected-spin subthreshold-swing magnitude in mV/dec.

    Assembled by calling the earlier sub-problem functions:
    lattice_mismatch, binding_energy, schottky_barriers, terminal_potentials,
    gated_band_edges, threshold_spectra, landauer_spin_currents,
    spin_filtering_efficiency, and decade_swing.

    Raises
    ------
    ValueError
        If any intermediate step raises, the interface/contact selection
        rejects the instance, or SFE does not match the scored branch.
    """
    return 0.0
```

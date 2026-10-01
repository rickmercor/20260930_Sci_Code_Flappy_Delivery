# Material_Science-Semiconductor_Materials-36

## Background

Through-silicon vias carry signals and power vertically through the thinned dies of three-dimensional integrated circuits. A typical via is a copper cylinder separated from the silicon by a thin dielectric liner, and because copper expands several times more than silicon on heating, every temperature excursion, whether from processing, power cycling or a local thermal event, loads the via and the surrounding substrate with thermomechanical stress. That stress concentrates around the liner, where it can drive interfacial delamination, cracking and copper protrusion at the via ends, and it reaches into the silicon where the active devices sit.

The analytical descriptions of via stress that are most widely used in design are steady-state models. The two-dimensional Lamé solution treats a long via in plane strain under a uniform temperature change, and three-dimensional Kane-Mindlin-type models capture free-surface effects but depend on empirical correction factors. Finite-element simulation resolves transient temperature and stress fields but is expensive for design sweeps, which is why accurate closed-form treatments of layered vias, validated against simulation, are of practical interest.

The stress matters electrically through the piezoresistive effect: lattice strain changes the band structure and carrier scattering in silicon, so the carrier mobility of transistors near a via shifts, and layouts reserve a keep-out zone around each via inside which that shift would exceed what the circuit tolerates.

## Problem

A copper-filled through-silicon via in a three-dimensional integrated circuit is quenched: it starts at a uniform 400 K and, from t = 0, a heat sink clamps its top face to the 300 K ambient, so the transient thermal stress of the cooling via shifts the carrier mobility of transistors in the surrounding silicon. The structure is axisymmetric, h = 200 μm tall, and consists of three concentric layers whose outer radii are r1 = 15 μm (copper), r2 = 16 μm (silicon dioxide) and r3 = 30 μm (silicon), with z measured from the base of the via. Thermally, the base z = 0 and the outer surface r = r3 are adiabatic, the top face z = h stays at the ambient temperature for all t > 0, and neighbouring layers are in ideal thermal contact. Mechanically, the layers are perfectly bonded, isotropic and linear thermoelastic and respond quasi-statically, the outer surface r = r3 is traction free, u_z = 0 and σ_rz = 0 on the base z = 0, and u_r = 0 and σ_zz = 0 on the top face z = h. The structure is stress free at the 300 K ambient temperature. The material properties are temperature independent:

- Copper: k = 400 W/(m·K), c_p = 385 J/(kg·K), ρ = 8960 kg/m^3, α = 17 × 10^-6 K^-1, E = 110 GPa, ν = 0.35
- Silicon dioxide: k = 1.4 W/(m·K), c_p = 730 J/(kg·K), ρ = 2200 kg/m^3, α = 0.5 × 10^-6 K^-1, E = 70 GPa, ν = 0.17
- Silicon: k = 130 W/(m·K), c_p = 700 J/(kg·K), ρ = 2329 kg/m^3, α = 2.6 × 10^-6 K^-1, E = 170 GPa, ν = 0.28

A PMOS transistor in the silicon has its channel aligned with the stress. Compute the carrier mobility change rate Δμ/μ of that transistor at r = 20 μm on the plane z = h/3, 0.05 ms after the quench, and give it in percent, with its sign, as the final answer. Take Δμ/μ positive when the hole mobility increases, so that Δμ/μ = −Δρ/ρ with Δρ/ρ the piezoresistive change of the channel resistivity; this is stated as a correction, because the relation is sometimes printed without the minus sign. In your reasoning, state as short assertions the radial stress at the copper-oxide interface (r = 15 μm) and the hoop stress in the silicon immediately outside the oxide-silicon interface, both on z = h/3 at 0.05 ms, and the mobility change at the same point at 0.01 ms and at 0.3 ms, with how it evolves over that interval. As validation checks, also state the temperature on the via axis at z = 0 at 0.05 ms and how much the radial stress varies across the copper core, from the axis to r = 15 μm, on z = h/3 at 0.05 ms.

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

01_compute_axial_eigenmodes

Goal
----
Build the axial eigenvalues and their squared norms for a through-silicon via that is adiabatic on its bottom face and held at the ambient temperature by a heat sink on its top face.

```python
import numpy as np

def compute_axial_eigenmodes(height: float, n_modes: int) -> np.ndarray:
    """Axial eigenvalues and squared norms of the via temperature expansion.

    Parameters
    ----------
    height : float
        Total via height h in metres (h > 0).
    n_modes : int
        Number of axial modes to return (n_modes >= 1).

    Returns
    -------
    modes : np.ndarray
        Array of shape (n_modes, 2). Column 0 holds the axial eigenvalues
        eta_m in inverse metres, ordered from the smallest upward. Column 1
        holds the squared norms N_zm in metres.

    Raises
    ------
    ValueError
        If height is not a finite number greater than 0, or if n_modes
        is not an integer greater than or equal to 1.
    """
    return modes  # placeholder
```

### Step 2

02_propagate_radial_eigenfunction

Goal
----
Propagate one radial eigenfunction of the layered via outward from the axis, returning the two Bessel amplitudes and the separation constant of every material layer for a trial thermal decay rate.

```python
import numpy as np

def propagate_radial_eigenfunction(radii: np.ndarray, conductivity: np.ndarray,
                                   diffusivity: np.ndarray, axial_eigenvalue: float,
                                   decay_rate: float) -> np.ndarray:
    """Amplitudes and separation constants of one radial eigenfunction.

    Parameters
    ----------
    radii : np.ndarray
        Shape (l,). Outer radius of each layer in metres, strictly increasing
        and strictly positive, with l >= 1.
    conductivity : np.ndarray
        Shape (l,). Thermal conductivity of each layer in W/(m K), all > 0.
    diffusivity : np.ndarray
        Shape (l,). Thermal diffusivity of each layer in m^2/s, all > 0.
    axial_eigenvalue : float
        Axial eigenvalue eta in inverse metres (eta > 0).
    decay_rate : float
        Trial thermal decay rate mu in inverse seconds (mu > 0).

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (l, 3). Row i holds the first amplitude A_i, the second
        amplitude B_i and the separation constant beta_i^2 of layer i. The
        innermost layer is normalised to A_1 = 1, B_1 = 0.

    Raises
    ------
    ValueError
        If radii, conductivity and diffusivity do not share a single
        length l >= 1; if radii is not finite, positive and strictly
        increasing; if any conductivity or diffusivity entry is not
        finite and greater than 0; or if axial_eigenvalue or decay_rate
        is not a finite number greater than 0.
    """
    return coefficients  # placeholder
```

### Step 3

03_compute_radial_eigenvalues

Goal
----
Locate the leading thermal decay rates of the layered via for one axial mode by finding the zeros of the outward radial slope at the adiabatic outer surface.

```python
import numpy as np

def compute_radial_eigenvalues(radii: np.ndarray, conductivity: np.ndarray,
                               diffusivity: np.ndarray, axial_eigenvalue: float,
                               n_modes: int,
                               samples_per_half_wave: int = 24) -> np.ndarray:
    """Leading thermal decay rates of the layered via for one axial mode.

    Parameters
    ----------
    radii : np.ndarray
        Shape (l,). Outer radius of each layer in metres, strictly increasing
        and strictly positive.
    conductivity : np.ndarray
        Shape (l,). Thermal conductivity of each layer in W/(m K), all > 0.
    diffusivity : np.ndarray
        Shape (l,). Thermal diffusivity of each layer in m^2/s, all > 0.
    axial_eigenvalue : float
        Axial eigenvalue eta in inverse metres (eta > 0).
    n_modes : int
        Number of radial modes to return (n_modes >= 1).
    samples_per_half_wave : int
        Number of scan samples per expected root spacing (>= 4).

    Returns
    -------
    decay_rates : np.ndarray
        Array of shape (n_modes,) holding the decay rates mu in inverse
        seconds, ascending.

    Raises
    ------
    ValueError
        If radii, conductivity and diffusivity do not share a single
        length l >= 1; if radii is not finite, positive and strictly
        increasing; if any conductivity or diffusivity entry is not
        finite and greater than 0; if axial_eigenvalue is not a finite
        number greater than 0; if n_modes is not an integer greater than
        or equal to 1; if samples_per_half_wave is not an integer
        greater than or equal to 4; or if the scan fails to bracket the
        requested number of modes.
    """
    return decay_rates  # placeholder
```

### Step 4

04_compute_modal_coefficients

Goal
----
Project a uniform initial temperature rise onto the layered eigenfunctions of one axial mode and return the expansion coefficient of every radial mode.

```python
import numpy as np

def compute_modal_coefficients(radii: np.ndarray, conductivity: np.ndarray,
                               diffusivity: np.ndarray, height: float,
                               axial_eigenvalue: float, axial_norm: float,
                               eigen_coefficients: np.ndarray,
                               initial_rise: float) -> np.ndarray:
    """Expansion coefficients of a uniform initial temperature rise.

    Parameters
    ----------
    radii : np.ndarray
        Shape (l,). Outer radius of each layer in metres, strictly increasing.
    conductivity : np.ndarray
        Shape (l,). Thermal conductivity of each layer in W/(m K), all > 0.
    diffusivity : np.ndarray
        Shape (l,). Thermal diffusivity of each layer in m^2/s, all > 0.
    height : float
        Total via height h in metres (h > 0).
    axial_eigenvalue : float
        Axial eigenvalue eta of this mode in inverse metres (eta > 0).
    axial_norm : float
        Squared norm N_z of the axial eigenfunction in metres (> 0).
    eigen_coefficients : np.ndarray
        Shape (n_radial, l, 3). For each radial mode, the two amplitudes and
        the separation constant of every layer.
    initial_rise : float
        Uniform initial temperature rise above ambient in kelvin.

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (n_radial,) holding the expansion coefficients in
        kelvin.

    Raises
    ------
    ValueError
        If radii, conductivity and diffusivity do not share a single
        length l >= 1; if radii is not finite, positive and strictly
        increasing; if any conductivity or diffusivity entry is not
        finite and greater than 0; if eigen_coefficients is not finite
        or does not have shape (n_radial, l, 3) with n_radial >= 1; if
        height, axial_eigenvalue or axial_norm is not a finite number
        greater than 0; or if initial_rise is not a finite number.
    """
    return coefficients  # placeholder
```

### Step 5

05_compute_temperature_rise

Goal
----
Evaluate the transient excess temperature of the layered via at one point and one instant by summing the assembled double eigenfunction series.

```python
import numpy as np

def compute_temperature_rise(radii: np.ndarray, axial_eigenvalues: np.ndarray,
                             decay_rates: np.ndarray,
                             modal_coefficients: np.ndarray,
                             eigen_coefficients: np.ndarray, radius: float,
                             depth: float, time: float) -> float:
    """Excess temperature of the layered via at one point and one instant.

    Parameters
    ----------
    radii : np.ndarray
        Shape (l,). Outer radius of each layer in metres, strictly increasing.
    axial_eigenvalues : np.ndarray
        Shape (n_axial,). Axial eigenvalues in inverse metres.
    decay_rates : np.ndarray
        Shape (n_axial, n_radial). Thermal decay rates in inverse seconds.
    modal_coefficients : np.ndarray
        Shape (n_axial, n_radial). Expansion coefficients in kelvin.
    eigen_coefficients : np.ndarray
        Shape (n_axial, n_radial, l, 3). Per mode and layer, the two radial
        amplitudes and the separation constant.
    radius : float
        Radial coordinate in metres, 0 < radius <= radii[-1].
    depth : float
        Axial coordinate in metres, measured from the insulated face (>= 0).
    time : float
        Elapsed time since the quench in seconds (>= 0).

    Returns
    -------
    rise : float
        Excess temperature above ambient in kelvin, as a native Python float.

    Raises
    ------
    ValueError
        If radii or axial_eigenvalues is empty; if radii is not finite,
        positive and strictly increasing; if decay_rates does not have
        shape (n_axial, n_radial); if modal_coefficients does not have
        the same shape as decay_rates; if eigen_coefficients does not
        have shape (n_axial, n_radial, l, 3); if radius, depth or time
        is not a finite number; if radius lies outside the half-open
        interval (0, radii[-1]]; or if depth or time is negative.
    """
    return rise  # placeholder
```

### Step 6

06_compute_potential_stress_terms

Goal
----
Evaluate the displacement and stress contributions of the thermoelastic displacement potential in one layer, for one axial mode, at one radius and instant.

```python
import numpy as np


def compute_potential_stress_terms(radius: float, axial_eigenvalue: float,
                                   decay_rates: np.ndarray,
                                   modal_coefficients: np.ndarray,
                                   layer_eigen_coefficients: np.ndarray,
                                   expansion: float, poisson: float, shear: float,
                                   time: float) -> np.ndarray:
    """Thermoelastic-potential contribution of one layer for one axial mode.

    Parameters
    ----------
    radius : float
        Radial coordinate in metres (> 0).
    axial_eigenvalue : float
        Axial eigenvalue eta of this mode in inverse metres (> 0).
    decay_rates : np.ndarray
        Shape (n_radial,). Thermal decay rates of this axial mode, in inverse
        seconds.
    modal_coefficients : np.ndarray
        Shape (n_radial,). Expansion coefficients of this axial mode, in kelvin.
    layer_eigen_coefficients : np.ndarray
        Shape (n_radial, 3). For each radial mode, the two amplitudes and the
        separation constant of the layer of interest.
    expansion : float
        Coefficient of thermal expansion of the layer in 1/K (> 0).
    poisson : float
        Poisson ratio of the layer, 0 < poisson < 0.5.
    shear : float
        Shear modulus of the layer in Pa (> 0).
    time : float
        Elapsed time since the quench in seconds (>= 0).

    Returns
    -------
    terms : np.ndarray
        Array of shape (6,) holding the amplitudes of the radial displacement,
        the axial displacement, the radial stress, the hoop stress, the axial
        stress and the shear stress, in metres for the displacements and Pa for
        the stresses.

    Raises
    ------
    ValueError
        If decay_rates and modal_coefficients do not share a single
        length greater than or equal to 1; if layer_eigen_coefficients
        does not have shape (n_radial, 3); if any decay_rates entry is
        not finite and greater than 0; if modal_coefficients or
        layer_eigen_coefficients is not finite; if radius,
        axial_eigenvalue, expansion or shear is not a finite number
        greater than 0; if poisson is not a finite number in the open
        interval (0, 0.5); or if time is not a finite number greater
        than or equal to 0.
    """
    return terms  # placeholder
```

### Step 7

07_compute_love_mode_coefficients

Goal
----
Solve the interface and traction-free conditions of one axial mode for the Love displacement function coefficients of every layer.

```python
import numpy as np

def compute_love_mode_coefficients(radii: np.ndarray, poisson: np.ndarray,
                                   shear: np.ndarray, axial_eigenvalue: float,
                                   potential_terms: np.ndarray) -> np.ndarray:
    """Love displacement function coefficients of one axial mode.

    Parameters
    ----------
    radii : np.ndarray
        Shape (l,). Outer radius of each layer in metres, strictly increasing,
        with l >= 2.
    poisson : np.ndarray
        Shape (l,). Poisson ratio of each layer, each in (0, 0.5).
    shear : np.ndarray
        Shape (l,). Shear modulus of each layer in Pa, all > 0.
    axial_eigenvalue : float
        Axial eigenvalue eta of this mode in inverse metres (> 0).
    potential_terms : np.ndarray
        Shape (l, 2, 6). Entry [i, 0] holds the six potential amplitudes of
        layer i evaluated at its outer radius radii[i], and entry [i, 1] holds
        them at its inner radius radii[i-1]. Entry [0, 1] is never used.

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (4 * l - 2,). The first two entries are the bounded
        coefficients E and P of the innermost layer; each following block of
        four holds E, F, P and Q of the next layer outward.

    Raises
    ------
    ValueError
        If radii, poisson and shear do not share a single length l >= 2;
        if radii is not finite, positive and strictly increasing; if any
        poisson entry is not finite and inside the open interval (0,
        0.5); if any shear entry is not finite and greater than 0; if
        potential_terms is not a finite array of shape (l, 2, 6); if
        axial_eigenvalue is not a finite number greater than 0; or if
        the assembled interface system has an identically zero row or
        column.
    """
    return coefficients  # placeholder
```

### Step 8

08_assemble_mode_stress

Goal
----
Combine the Love displacement function of a layer with the thermoelastic potential of the same layer to obtain the four stress amplitudes of one axial mode at one radius.

```python
import numpy as np

def assemble_mode_stress(radius: float, radii: np.ndarray, poisson: np.ndarray,
                         shear: np.ndarray, axial_eigenvalue: float,
                         love_coefficients: np.ndarray,
                         potential_terms: np.ndarray) -> np.ndarray:
    """Total stress amplitudes of one axial mode at one radius.

    Parameters
    ----------
    radius : float
        Radial coordinate in metres, 0 < radius <= radii[-1].
    radii : np.ndarray
        Shape (l,). Outer radius of each layer in metres, strictly increasing,
        with l >= 2.
    poisson : np.ndarray
        Shape (l,). Poisson ratio of each layer, each in (0, 0.5).
    shear : np.ndarray
        Shape (l,). Shear modulus of each layer in Pa, all > 0.
    axial_eigenvalue : float
        Axial eigenvalue eta of this mode in inverse metres (> 0).
    love_coefficients : np.ndarray
        Shape (4 * l - 2,). Love coefficients ordered from the innermost layer
        outward, the innermost layer carrying only its two bounded ones.
    potential_terms : np.ndarray
        Shape (6,). Potential amplitudes of the layer containing radius,
        evaluated at that radius.

    Returns
    -------
    stress : np.ndarray
        Array of shape (4,) holding the radial, hoop, axial and shear stress
        amplitudes in Pa.

    Raises
    ------
    ValueError
        If radii, poisson and shear do not share a single length l >= 2;
        if radii is not finite, positive and strictly increasing; if any
        poisson entry is not finite and inside the open interval (0,
        0.5); if any shear entry is not finite and greater than 0; if
        love_coefficients is not a finite array of length 4 * l - 2; if
        potential_terms is not a finite array of length 6; if
        axial_eigenvalue is not a finite number greater than 0; if
        radius is not a finite number; or if radius lies outside the
        half-open interval (0, radii[-1]].
    """
    return stress  # placeholder
```

### Step 9

09_run_mobility_shift_pipeline

Goal
----
Chain the sub-problem functions 01-08 end to end on the quenched through-silicon via and return the piezoresistive carrier mobility change rate at the requested point and instant.

Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (compute_axial_eigenmodes, propagate_radial_eigenfunction, compute_radial_eigenvalues, compute_modal_coefficients, compute_temperature_rise, compute_potential_stress_terms, compute_love_mode_coefficients, assemble_mode_stress) rather than reimplementing them.

```python
import numpy as np


def run_mobility_shift_pipeline(radius: float = 20.0e-6,
                                depth: float = 200.0e-6 / 3.0,
                                time: float = 5.0e-5,
                                n_axial: int = 8, n_radial: int = 8,
                                radii=(15.0e-6, 16.0e-6, 30.0e-6),
                                height: float = 200.0e-6,
                                conductivity=(400.0, 1.4, 130.0),
                                density=(8960.0, 2200.0, 2329.0),
                                heat_capacity=(385.0, 730.0, 700.0),
                                expansion=(17.0e-6, 0.5e-6, 2.6e-6),
                                young=(110.0e9, 70.0e9, 170.0e9),
                                poisson=(0.35, 0.17, 0.28),
                                initial_rise: float = 100.0,
                                piezo_coefficient: float = 71.8e-11,
                                orientation_factor: float = 1.0) -> float:
    """Run the full transient thermal stress and mobility measurement.

    Parameters
    ----------
    radius : float
        Radial coordinate of the evaluation point in metres, 0 < radius <= radii[-1].
    depth : float
        Axial coordinate of the evaluation point in metres, measured from the
        insulated base (>= 0).
    time : float
        Elapsed time since the quench in seconds (>= 0).
    n_axial : int
        Number of axial modes retained (n_axial >= 1).
    n_radial : int
        Number of radial modes retained per axial mode (n_radial >= 1).
    radii : sequence of float
        Outer radius of each layer in metres, strictly increasing, l >= 2.
    height : float
        Total via height in metres (> 0).
    conductivity : sequence of float
        Thermal conductivity of each layer in W/(m K), all > 0.
    density : sequence of float
        Density of each layer in kg/m^3, all > 0.
    heat_capacity : sequence of float
        Specific heat capacity of each layer in J/(kg K), all > 0.
    expansion : sequence of float
        Coefficient of thermal expansion of each layer in 1/K, all > 0.
    young : sequence of float
        Young modulus of each layer in Pa, all > 0.
    poisson : sequence of float
        Poisson ratio of each layer, each in (0, 0.5).
    initial_rise : float
        Uniform initial temperature rise above ambient in kelvin.
    piezo_coefficient : float
        Piezoresistive coefficient in 1/Pa.
    orientation_factor : float
        Orientation factor between the stress and the transistor channel.

    Returns
    -------
    shift : float
        Piezoresistive carrier mobility change rate in percent, positive when
        the hole mobility increases, as a native Python float.

    Raises
    ------
    ValueError
        If radii holds fewer than two layers; if conductivity, density,
        heat_capacity, expansion, young or poisson does not have one
        entry per layer or carries an entry that is not finite and
        greater than 0; if radii is not finite, positive and strictly
        increasing; if any poisson entry is not below 0.5; if n_axial or
        n_radial is not an integer greater than or equal to 1; if
        height, initial_rise or piezo_coefficient is not a finite number
        greater than 0; if radius, depth, time or orientation_factor is
        not a finite number; if radius lies outside the half-open
        interval (0, radii[-1]]; or if depth or time is negative.
    """
    return shift  # placeholder
```

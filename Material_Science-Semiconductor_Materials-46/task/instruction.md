# Peak transient thermal stress in a through-silicon via

## Background

A through-silicon via is a bonded, concentric multilayer cylinder whose copper conductor, silicon-dioxide liner, and silicon substrate have strongly different thermal conductivities and coefficients of thermal expansion. When cooling begins, the layers do not remain at one uniform temperature. Their constrained thermal strains therefore produce a spatially varying, multiaxial stress field. At the early time and elevated cross-section specified in this task, a steady uniform-temperature approximation would not represent the actual loading.

The temperature field is described by axisymmetric transient heat conduction. The end conditions determine the axial modes, while continuity of temperature and radial heat flux couples the layers into a composite radial eigenvalue problem. Each coupled thermal mode has one decay rate shared by all layers. Projecting the initial uniform temperature excess onto these composite modes gives the transient temperature distribution at the requested time and position.

The mechanical response is treated as quasi-static, isotropic linear thermoelasticity. A thermally driven particular displacement field is combined with a complementary elastic field so that perfect bonding, free-surface tractions, and the prescribed end conditions are all satisfied. Retaining the axial normal stress and radial-axial shear stress is essential: a reduced two-dimensional cross-sectional model omits part of the stress state and can bias the von Mises stress. The four nonzero axisymmetric stress components are therefore evaluated across the requested plane, including both sides of material interfaces, to determine the peak equivalent stress.

## Problem

A through-silicon via bonds materials whose coefficients of thermal expansion differ by more than an order of magnitude, so any change in die temperature loads the metal–liner and liner–substrate interfaces, and that load is transient because the via carries its own evolving internal temperature field rather than a single uniform excursion. For the via specified below, report the largest von Mises stress present anywhere on the cross-section at nine tenths of the via height, 0.05 ms after the via begins to cool.

The via is axisymmetric and 200 µm tall, with a copper conductor out to a radius of 15 µm, a silicon dioxide liner out to 16 µm and silicon out to 25 µm. Every layer is isotropic, linear elastic and has temperature-independent properties, given here as thermal conductivity in W m⁻¹ K⁻¹, specific heat capacity in J kg⁻¹ K⁻¹, density in kg m⁻³, coefficient of thermal expansion in ppm K⁻¹, Young's modulus in GPa and Poisson's ratio: copper 400, 385, 8960, 17, 110, 0.35; silicon dioxide 1.4, 730, 2200, 0.5, 70, 0.17; silicon 130, 700, 2329, 2.6, 170, 0.28. Both interfaces are perfectly bonded, thermally and mechanically.

The via is uniformly at 400 K at the initial instant, and the ambient temperature is 300 K. The outer cylindrical surface and the face at $z = 0$ pass no heat, the face at $z = 200$ µm is held at the ambient temperature, and the temperature remains finite on the axis. The mechanical problem is quasi-static, axisymmetric and free of body forces: the outer cylindrical surface carries no traction, the face at $z = 0$ neither displaces axially nor carries shear traction, and the face at $z = 200$ µm neither displaces radially nor carries axial normal stress.

Your final answer must be a single number: the peak von Mises stress on that cross-section, in MPa. Report the converged value, refining whatever discretisation your method introduces until further refinement no longer changes the first six significant figures. The few scalars that determine that number are these, and they are the ones to show: the three layer thermal diffusivities; the fundamental axial eigenvalue, the slowest decay rate belonging to it, its squared radial parameter in the metal, and the relative weight of the next two radial modes at the requested instant; the temperature on the axis of the examined plane; the four stress components and the equivalent stress on the axis, on both sides of each interface and at the outer silicon surface of that plane, as one compact table; and the value the peak equivalent stress takes when either the axial normal stress or the radial-axial shear stress is dropped from the equivalent-stress formula. Preface those numbers with a brief account of the modelling decisions they rest on and of what a reduced two-dimensional treatment of the same cross-section would lose; a few sentences is enough.

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

01_compute_axial_modes

Goal
----
Return the axial separation eigenvalues of the via, together with the norm of each axial eigenfunction and its integral over the via height, for a via that is insulated on its bottom face and held at the ambient temperature on its top face.

```python
import numpy as np

def compute_axial_modes(height: float, n_modes: int) -> np.ndarray:
    """Return the axial separation eigenvalues and their two integrals.

    Parameters
    ----------
    height : float
        Height of the via in m, measured from the insulated face at z = 0 to
        the face held at the ambient temperature (height > 0).
    n_modes : int
        Number of axial modes to return, ordered by increasing eigenvalue
        (n_modes >= 1).

    Returns
    -------
    axial : np.ndarray
        Array of shape (n_modes, 3) whose columns are the axial eigenvalue in
        1/m, the square norm of the corresponding axial eigenfunction over the
        height in m, and the integral of that eigenfunction over the height
        in m.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value.
    """
    return axial  # placeholder
```

### Step 2

02_solve_radial_eigenvalues

Goal
----
Return the slowest decay rates of the layered via for one axial mode, obtained as the eigenvalues of the radial problem that couples all material layers through the interface conditions.

```python
import numpy as np


def solve_radial_eigenvalues(radii: np.ndarray, conductivities: np.ndarray,
                             diffusivities: np.ndarray, eta: float,
                             n_modes: int) -> np.ndarray:
    """Return the slowest decay rates of the layered via for one axial mode.

    Parameters
    ----------
    radii : np.ndarray
        Outer radius of each layer in m, shape (n_layers,), strictly
        increasing and positive; the innermost layer starts on the axis.
    conductivities : np.ndarray
        Thermal conductivity of each layer in W/(m K), shape (n_layers,),
        all positive.
    diffusivities : np.ndarray
        Thermal diffusivity of each layer in m^2/s, shape (n_layers,),
        all positive.
    eta : float
        Axial eigenvalue of the mode in 1/m (eta > 0).
    n_modes : int
        Number of decay rates to return, ordered by increasing value
        (n_modes >= 1).

    Returns
    -------
    decay_rates : np.ndarray
        Array of shape (n_modes,) holding the decay rates in 1/s, ascending.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, or if the
        search cannot locate n_modes decay rates; invalid input must raise
        ValueError rather than return a sentinel value.
    """
    return decay_rates  # placeholder
```

### Step 3

03_build_radial_eigenfunction

Goal
----
Return, for one decay rate of one axial mode, each layer's radial parameter and the two amplitudes that define its radial eigenfunction, normalised so that the amplitude of the regular solution on the axis is unity.

```python
import numpy as np


def build_radial_eigenfunction(radii: np.ndarray, conductivities: np.ndarray,
                               diffusivities: np.ndarray, eta: float,
                               decay_rate: float) -> np.ndarray:
    """Return the radial parameter and the two amplitudes of every layer.

    Parameters
    ----------
    radii : np.ndarray
        Outer radius of each layer in m, shape (n_layers,), strictly
        increasing and positive.
    conductivities : np.ndarray
        Thermal conductivity of each layer in W/(m K), shape (n_layers,),
        all positive.
    diffusivities : np.ndarray
        Thermal diffusivity of each layer in m^2/s, shape (n_layers,),
        all positive.
    eta : float
        Axial eigenvalue of the mode in 1/m (eta > 0).
    decay_rate : float
        Decay rate of the mode in 1/s (decay_rate > 0).

    Returns
    -------
    eigenfunction : np.ndarray
        Array of shape (n_layers, 3) whose columns are the squared radial
        parameter of the layer in 1/m^2, the amplitude of the regular radial
        solution and the amplitude of the singular radial solution.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value.
    """
    return eigenfunction  # placeholder
```

### Step 4

4_compute_expansion_coefficient

Goal
----
Return the amplitude with which one product eigenmode enters the expansion of a via that starts at a spatially uniform temperature above ambient.

```python
import numpy as np



def compute_expansion_coefficient(radii: np.ndarray, capacities: np.ndarray,
                                  eigenfunction: np.ndarray,
                                  axial_mode: np.ndarray,
                                  initial_rise: float) -> float:
    """Return the expansion amplitude of one product eigenmode.

    Parameters
    ----------
    radii : np.ndarray
        Outer radius of each layer in m, shape (n_layers,), strictly
        increasing and positive.
    capacities : np.ndarray
        Volumetric heat capacity of each layer in J/(m^3 K), shape
        (n_layers,), all positive.
    eigenfunction : np.ndarray
        Array of shape (n_layers, 3) holding the squared radial parameter and
        the two radial amplitudes of every layer.
    axial_mode : np.ndarray
        Array of shape (3,) holding the axial eigenvalue in 1/m, the axial
        square norm in m and the axial eigenfunction integral in m.
    initial_rise : float
        Spatially uniform initial temperature of the via above ambient in K.

    Returns
    -------
    coefficient : float
        Expansion amplitude of the mode in K, as a native Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, or if the
        heat-capacity-weighted square norm of the eigenfunction vanishes;
        invalid input must raise ValueError rather than return a sentinel
        value.
    """
    return coefficient  # placeholder
```

### Step 5

05_evaluate_temperature_rise

Goal
----
Sum the double eigenmode series to return the transient temperature of the via above ambient at one position and one instant.

```python
import numpy as np


def evaluate_temperature_rise(axial_modes: np.ndarray, decay_rates: np.ndarray,
                              eigenfunctions: np.ndarray,
                              coefficients: np.ndarray, layer: int,
                              radius: float, axial_position: float,
                              time: float) -> float:
    """Return the temperature above ambient at one point and one instant.

    Parameters
    ----------
    axial_modes : np.ndarray
        Array of shape (n_axial, 3) holding the axial eigenvalue, square norm
        and eigenfunction integral of every axial mode.
    decay_rates : np.ndarray
        Array of shape (n_axial, n_radial) holding the decay rate in 1/s of
        every mode.
    eigenfunctions : np.ndarray
        Array of shape (n_axial, n_radial, n_layers, 3) holding the squared
        radial parameter and the two radial amplitudes of every layer for
        every mode.
    coefficients : np.ndarray
        Array of shape (n_axial, n_radial) holding the expansion amplitude in
        K of every mode.
    layer : int
        Index of the layer containing the evaluation point, 0-based.
    radius : float
        Radial coordinate of the evaluation point in m (radius >= 0).
    axial_position : float
        Axial coordinate of the evaluation point in m (axial_position >= 0).
    time : float
        Time since the start of the transient in s (time >= 0).

    Returns
    -------
    temperature_rise : float
        Temperature above ambient in K, as a native Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value.
    """
    return temperature_rise  # placeholder
```

### Step 6

06_compute_thermal_source_terms

Goal
----
Return the displacement and stress contributions that the temperature field alone produces in one layer for one axial mode, before any boundary or interface condition is enforced.

```python
import numpy as np


def compute_thermal_source_terms(decay_rates: np.ndarray,
                                 eigenfunctions: np.ndarray,
                                 coefficients: np.ndarray, eta: float,
                                 temperature_rise: float,
                                 thermal_expansion: float, poisson: float,
                                 shear_modulus: float, layer: int,
                                 radius: float, time: float) -> np.ndarray:
    """Return the thermally generated displacements and stresses in one layer.

    Parameters
    ----------
    decay_rates : np.ndarray
        Array of shape (n_radial,) holding the decay rate in 1/s of every
        radial mode belonging to this axial mode.
    eigenfunctions : np.ndarray
        Array of shape (n_radial, n_layers, 3) holding the squared radial
        parameter and the two radial amplitudes of every layer.
    coefficients : np.ndarray
        Array of shape (n_radial,) holding the expansion amplitude in K of
        every radial mode.
    eta : float
        Axial eigenvalue of the mode in 1/m (eta > 0).
    temperature_rise : float
        Temperature of this layer above ambient in K at this radius and
        instant, contributed by this axial mode alone and stripped of its
        axial cosine factor, as returned by the temperature evaluation step.
    thermal_expansion : float
        Coefficient of thermal expansion of the layer in 1/K.
    poisson : float
        Poisson's ratio of the layer, -1 < poisson < 0.5.
    shear_modulus : float
        Shear modulus of the layer in Pa (shear_modulus > 0).
    layer : int
        Index of the layer, 0-based.
    radius : float
        Radial coordinate in m (radius >= 0).
    time : float
        Time since the start of the transient in s (time >= 0).

    Returns
    -------
    terms : np.ndarray
        Array of shape (6,) holding the radial displacement in m, the axial
        displacement in m, and the radial, hoop, axial and shear stresses in
        Pa that the temperature field alone produces, each stripped of its
        axial cosine or sine factor.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value.
    """
    return terms  # placeholder
```

### Step 7

07_assemble_love_system

Goal
----
Assemble the linear system whose solution is the set of complementary-solution amplitudes that restore interface continuity and a traction-free outer surface for one axial mode.

```python
import numpy as np


def assemble_love_system(radii: np.ndarray, poisson: np.ndarray,
                         shear_moduli: np.ndarray, eta: float,
                         thermal_terms: np.ndarray) -> np.ndarray:
    """Assemble the augmented linear system for the complementary amplitudes.

    Parameters
    ----------
    radii : np.ndarray
        Outer radius of each layer in m, shape (n_layers,), strictly
        increasing and positive.
    poisson : np.ndarray
        Poisson's ratio of each layer, shape (n_layers,), each in (-1, 0.5).
    shear_moduli : np.ndarray
        Shear modulus of each layer in Pa, shape (n_layers,), all positive.
    eta : float
        Axial eigenvalue of the mode in 1/m (eta > 0).
    thermal_terms : np.ndarray
        Array of shape (n_layers, 2, 6) holding, for every layer, the six
        thermally generated displacement and stress contributions evaluated
        at that layer's inner radius (index 0) and outer radius (index 1).

    Returns
    -------
    system : np.ndarray
        Array of shape (4 * n_layers - 2, 4 * n_layers - 1) holding the
        coefficient matrix with the right-hand side appended as its last
        column.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value.
    """
    return system  # placeholder
```

### Step 8

08_solve_love_coefficients

Goal
----
Solve the assembled interface system and return, for every layer, the four amplitudes of the biharmonic complementary solution in a fixed order.

```python
import numpy as np


def solve_love_coefficients(system: np.ndarray, n_layers: int) -> np.ndarray:
    """Solve the assembled system and return the amplitudes of every layer.

    Parameters
    ----------
    system : np.ndarray
        Array of shape (4 * n_layers - 2, 4 * n_layers - 1) holding the
        coefficient matrix with the right-hand side as its final column.
    n_layers : int
        Number of material layers in the stack (n_layers >= 1).

    Returns
    -------
    amplitudes : np.ndarray
        Array of shape (n_layers, 4) holding, for every layer, the amplitudes
        of the regular harmonic shape, the singular harmonic shape, the
        regular biharmonic shape and the singular biharmonic shape, in that
        order; the two singular amplitudes of the innermost layer are zero.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above, or if the
        coefficient matrix is singular; invalid input must raise ValueError
        rather than return a sentinel value.
    """
    return amplitudes  # placeholder
```

### Step 9

09_evaluate_stress_components

Goal
----
Combine the complementary amplitudes of one layer with the thermally generated contributions at the same point to return the four axisymmetric stress components of one axial mode.

```python
import numpy as np


def evaluate_stress_components(love_coefficients: np.ndarray,
                               thermal_terms: np.ndarray, eta: float,
                               poisson: float, shear_modulus: float,
                               radius: float,
                               axial_position: float) -> np.ndarray:
    """Return the four stress components contributed by one axial mode.

    Parameters
    ----------
    love_coefficients : np.ndarray
        Array of shape (4,) holding the amplitudes of the regular harmonic,
        singular harmonic, regular biharmonic and singular biharmonic shapes
        of the layer.
    thermal_terms : np.ndarray
        Array of shape (6,) holding the thermally generated displacement and
        stress contributions of the layer at this radius, stripped of their
        axial factor.
    eta : float
        Axial eigenvalue of the mode in 1/m (eta > 0).
    poisson : float
        Poisson's ratio of the layer, -1 < poisson < 0.5.
    shear_modulus : float
        Shear modulus of the layer in Pa (shear_modulus > 0).
    radius : float
        Radial coordinate in m (radius >= 0).
    axial_position : float
        Axial coordinate in m (axial_position >= 0).

    Returns
    -------
    components : np.ndarray
        Array of shape (4,) holding the radial normal, hoop, axial normal and
        shear stresses in Pa contributed by this axial mode.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value. The two
        singular amplitudes, the second and fourth entries of
        love_coefficients, must be zero when radius is 0, since the singular
        shapes are unbounded on the axis; a non-zero value there is invalid.
    """
    return components  # placeholder
```

### Step 10

10_run_tsv_thermal_stress_pipeline

Goal
----
Chain the sub-problem functions 01-09 end to end on the layered via and return the largest von Mises stress found on the requested cross-section at the requested instant, expressed in megapascals.

```python
import numpy as np


def run_tsv_thermal_stress_pipeline(radii: tuple = (15.0e-6, 16.0e-6, 25.0e-6),
                                    conductivities: tuple = (400.0, 1.4, 130.0),
                                    densities: tuple = (8960.0, 2200.0, 2329.0),
                                    specific_heats: tuple = (385.0, 730.0, 700.0),
                                    expansions: tuple = (17.0e-6, 0.5e-6, 2.6e-6),
                                    youngs_moduli: tuple = (110.0e9, 70.0e9, 170.0e9),
                                    poisson: tuple = (0.35, 0.17, 0.28),
                                    height: float = 200.0e-6,
                                    initial_rise: float = 100.0,
                                    plane_fraction: float = 0.9,
                                    time: float = 5.0e-5,
                                    n_axial: int = 12, n_radial: int = 6,
                                    n_samples: int = 401) -> float:
    """Return the peak von Mises stress on a cross-section of the layered via.

    Parameters
    ----------
    radii : tuple
        Outer radius of each layer in m, strictly increasing and positive.
    conductivities : tuple
        Thermal conductivity of each layer in W/(m K), all positive.
    densities : tuple
        Density of each layer in kg/m^3, all positive.
    specific_heats : tuple
        Specific heat capacity of each layer in J/(kg K), all positive.
    expansions : tuple
        Coefficient of thermal expansion of each layer in 1/K.
    youngs_moduli : tuple
        Young's modulus of each layer in Pa, all positive.
    poisson : tuple
        Poisson's ratio of each layer, each in (-1, 0.5).
    height : float
        Height of the via in m (height > 0).
    initial_rise : float
        Uniform initial temperature of the via above ambient in K.
    plane_fraction : float
        Position of the examined cross-section as a fraction of the height,
        0 <= plane_fraction <= 1.
    time : float
        Instant at which the stress is evaluated in s (time >= 0).
    n_axial : int
        Number of axial modes retained (n_axial >= 1).
    n_radial : int
        Number of radial modes retained per axial mode (n_radial >= 1).
    n_samples : int
        Number of radii sampled within each layer, endpoints included
        (n_samples >= 2).

    Returns
    -------
    peak_stress : float
        Largest von Mises stress on the cross-section in MPa, as a native
        Python float.

    Raises
    ------
    ValueError
        If any argument violates the constraints stated above; invalid input
        must raise ValueError rather than return a sentinel value.
    """
    return peak_stress  # placeholder
```

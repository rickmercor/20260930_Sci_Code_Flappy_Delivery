# Critical activation energy for heterogeneous bubble nucleation in nanoscale flow boiling on zirconium

## Background

# Scientific background

When a loss-of-coolant accident forces a water-cooled reactor into the reflood phase, the fuel rods are quenched by a dispersed flow of droplets carried downstream by fast steam. Whether the cladding cools or stays hot is decided at the surface of the zirconium alloy itself, over a liquid layer a few nanometres thick, on timescales of hundreds of picoseconds. Peak cladding temperature and quench-front progression, the two numbers reactor safety analysis actually reports, are set there. Continuum boiling correlations and macroscopic two-phase models cannot resolve that layer, because the mechanisms that matter over it, namely the finite thermal resistance of the solid-liquid contact, the layering of water into a solid-like film against the metal, and the competition between the thermal motion of individual molecules and the cohesive forces holding them in the liquid, are all molecular. Non-equilibrium molecular dynamics is the tool that reaches them, and the last few years have seen it applied systematically to boiling on metal substrates.

Almost all of that work has been pool boiling: a quiescent film on a heated plate, where the questions are how surface texture, nanoparticles or wettability change the nucleation rate and the critical heat flux. Reflood is not pool boiling. The liquid is sheared past the wall by steam moving at tens of metres per second, and imposing that shear inside a molecular dynamics cell is delicate, because an external body force applied to the fluid injects energy that would otherwise be mistaken for heating from the substrate. The device that solves this is a partitioned cell: a short upstream region where the body force is applied, a thermostatted region immediately downstream that resets the temperature the force disturbed, and a long collection region where the boiling is observed with no thermostat acting on it at all. Only with that separation can the driving force and the substrate temperature be varied independently, and only then can one ask whether flow changes nucleation or merely convects the bubble away.

The picture that emerges is that a nucleus appears when the average kinetic energy of the water molecules in a near-wall cell exceeds the magnitude of their average potential energy, and that the fate of the nucleus after it appears, growth, collapse or repeated metastable reappearance, is decided by the same balance at the bubble interface. Wettability enters twice: a strongly interacting surface transfers heat into the liquid faster, and it also builds a thicker solid-like layer that feeds microlayer evaporation. The driving force enters through a channel of its own, continually replacing the liquid standing over the nucleation site, and whether that advances or retards the onset of boiling is something the simulations have to be asked rather than assumed.

The interpretation of that molecular picture is where a second, independent line of reasoning is needed. Classical heterogeneous nucleation theory prices a nucleus by its surface energy against an isothermal, uniformly superheated liquid, which is exactly what a nanoscale wall boundary layer is not: the temperature falls steeply across the height of the nucleus, and the liquid touching the metal is already tens or hundreds of kelvin below the metal itself because the interface has a finite thermal resistance. A thermodynamic framework built for that situation has to price the nucleus some other way, and the closures that let it do so, namely how the temperature field inside the nucleus is anchored, what increment of free energy the nucleus is charged for displacing that liquid, and which vapour properties enter that increment, are particular to the framework rather than consequences of classical theory, so they have to be taken from it rather than reconstructed from first principles. What the framework returns is a barrier: an activation energy, and the critical nucleus size at which it is reached. Comparing how that barrier moves with wettability and with the imposed flow, against how the simulated waiting times move, is the check that decides whether the molecular observations have a thermodynamic explanation or are being read into one.

## Problem

Non-equilibrium molecular dynamics of water flowing over a heated zirconium cladding surface resolves the onset of boiling directly and returns nucleation waiting times, bubble volumes and quasi-steady contact angles, but it cannot say by itself whether the wettability and driving-force trends it produces are the ones a continuum thermodynamic description of the same interface implies. That description grows a vapour nucleus inside a liquid whose temperature falls linearly away from the wall and whose wall-adjacent value is held below the substrate by an interfacial temperature jump set by the solid-liquid thermal resistance and therefore by the wettability, and it measures the difficulty of nucleation by the maximum over nucleus size of an available-energy increment $\Delta\Psi$ of its own. Apply the microscale heterogeneous-nucleation framework these simulations are analysed with, exactly as that framework defines it, to the eight measured quasi-steady states tabulated below, taking from it the cylinder depth, the liquid-vapour surface tension, the reduced-unit temperature scale, the wall-normal temperature gradient and the definition of $\Delta\Psi$ itself — all fixed by that framework rather than free to choose — and saying what you took.

- Nucleus: a cylinder along the flow-normal direction, of the depth that framework tabulates for it, whose cross-section is the circular segment of radius $r$ cut off by the substrate plane, with liquid-side contact angle $\theta = \pi - \theta_d$, where $\theta_d$ is the measured quasi-steady contact angle of the bubble
- Temperature: one field shared by the nucleus and the liquid it displaces, linear in the wall-normal coordinate $Z$, equal to $T_S - \Delta T_j$ at the wall and falling with the gradient that framework prescribes, every state run at the same substrate temperature $T_S^* = 9.95$ in the reduced Lennard-Jones units of the TIP4P/2005 water model, each carrying its own measured $\Delta T_j^*$ in those same units
- Vapour pressure: the value the Young-Laplace relation gives at the curvature of the segment
- Liquid state: $P_l = 101325$ Pa and $T_{sat} = 373.15$ K, with $\rho_v$ the ideal-gas value at $(P_l, T_{sat})$
- Latent heat: the Watson correlation with exponent $0.38$, referred to $2441.7$ kJ/kg at $298.15$ K with $T_c = 647.096$ K, reconciled against the thermophysical properties the framework tabulates for itself
- Surfaces: hydrophilic at $\varepsilon^*_{\mathrm{Zr\text{-}O}} = 7.78$ with an equilibrium water contact angle of $62.5^\circ$, hydrophobic at $\varepsilon^*_{\mathrm{Zr\text{-}O}} = 5.91$ with $118.6^\circ$, four imposed driving forces each

| $\varepsilon^*_{\mathrm{Zr\text{-}O}}$ | $F^*$ | $\theta_d$ | $\Delta T_j^*$ |
|---|---|---|---|
| 7.78 | 0.00 | $117.5^\circ$ | 1.45 |
| 7.78 | 0.47 | $112.4^\circ$ | 1.70 |
| 7.78 | 0.94 | $106.8^\circ$ | 1.95 |
| 7.78 | 1.41 | $101.3^\circ$ | 2.10 |
| 5.91 | 0.00 | $61.4^\circ$ | 2.05 |
| 5.91 | 0.47 | $57.9^\circ$ | 2.30 |
| 5.91 | 0.94 | $53.6^\circ$ | 2.55 |
| 5.91 | 1.41 | $49.2^\circ$ | 2.70 |

Identify the state this model makes hardest to nucleate, and say what its barrier and its critical size imply about the reach of the model against the nucleation behaviour the simulations themselves report; report compactly, but do report the closures you took, the intermediate values the number rests on and the checks you made on it. Your final answer must be a single number: the largest of the eight critical activation energies, divided by the product of the Boltzmann constant and the wall-adjacent liquid temperature of that same state.

Output Format Requirements: Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags. You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure. Rules: - The tags are required. Do not omit them or leave them empty. - The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose. - Put only that one number between the tags. No units, no words, no extra lines. Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_convert_reduced_temperatures

Goal
----
Convert the substrate temperature and the interfacial temperature jump from Lennard-Jones reduced units to kelvin and return the wall-adjacent liquid temperature that drives the nucleation model.

```python
import numpy as np

def convert_reduced_temperatures(t_substrate_red: float, t_jump_red: float,
                                 epsilon_ev: float = 0.008031) -> np.ndarray:
    """Convert reduced substrate and jump temperatures to kelvin.

    Parameters
    ----------
    t_substrate_red : float
        Substrate temperature in reduced Lennard-Jones units (> 0).
    t_jump_red : float
        Interfacial temperature jump in reduced Lennard-Jones units (>= 0).
    epsilon_ev : float
        Lennard-Jones well depth of the reference pair in electronvolts (> 0).

    Returns
    -------
    temperatures : np.ndarray
        Array of shape (3,) holding the substrate temperature, the interfacial
        temperature jump and the wall-adjacent liquid temperature, all in
        kelvin.
    """
    return temperatures  # placeholder
```

### Step 2

02_compute_vapour_state

Goal
----
Compute the specific gas constant of water vapour, the saturated vapour density from the ideal-gas law and the latent heat of vaporisation from the Watson correlation.

```python
import numpy as np

def compute_vapour_state(t_sat: float, p_liquid: float,
                         h_reference: float = 2441.7e3,
                         t_reference: float = 298.15,
                         t_critical: float = 647.096,
                         watson_exponent: float = 0.38) -> np.ndarray:
    """Compute the mass-based vapour properties at the saturation state.

    Parameters
    ----------
    t_sat : float
        Saturation temperature of the liquid in kelvin (0 < t_sat < t_critical).
    p_liquid : float
        Liquid pressure in pascal (> 0).
    h_reference : float
        Reference latent heat of vaporisation in J/kg (> 0).
    t_reference : float
        Temperature in kelvin at which the reference latent heat applies
        (0 < t_reference < t_critical).
    t_critical : float
        Critical temperature of water in kelvin (> 0).
    watson_exponent : float
        Exponent of the Watson correlation (> 0).

    Returns
    -------
    vapour_state : np.ndarray
        Array of shape (3,) holding the specific gas constant of water vapour
        in J/(kg K), the saturated vapour density in kg/m^3 and the latent heat
        of vaporisation in J/kg.
    """
    return vapour_state  # placeholder
```

### Step 3

03_compute_segment_volume_factor

Goal
----
Compute the dimensionless factor that turns the squared nucleus radius and the cylinder depth into the volume of the truncated cylindrical nucleus.

```python
import numpy as np
def compute_segment_volume_factor(theta: float) -> float:
    """Compute the dimensionless volume factor of the truncated cylindrical nucleus.

    Parameters
    ----------
    theta : float
        Liquid-side contact angle in radians, strictly between 0 and pi.

    Returns
    -------
    volume_factor : float
        Dimensionless factor such that the nucleus volume equals the cylinder
        depth multiplied by the squared radius multiplied by this factor.
    """
    return volume_factor  # placeholder
```

### Step 4

04_compute_thermal_volume_moment

Goal
----
Compute the dimensionless first moment of the nucleus volume against the linear wall-normal temperature profile.

```python
import numpy as np

def compute_thermal_volume_moment(theta: float, volume_factor: float) -> float:
    """Compute the dimensionless temperature-weighted volume factor.

    Parameters
    ----------
    theta : float
        Liquid-side contact angle in radians, strictly between 0 and pi.
    volume_factor : float
        Dimensionless volume factor of the same segment, as returned by the
        previous step (> 0).

    Returns
    -------
    moment_factor : float
        Dimensionless factor such that the integral of the temperature over the
        nucleus equals the wall-adjacent liquid temperature multiplied by the
        cylinder depth, the squared radius and this factor.
    """
    return moment_factor  # placeholder
```

### Step 5

05_compute_free_energy_coefficients

Goal
----
Reduce the available-energy increment of the nucleus to the three parameters that define its dependence on the nucleus radius.

```python
import numpy as np

def compute_free_energy_coefficients(volume_factor: float, moment_factor: float,
                                     t_liquid: float, vapour_state: np.ndarray,
                                     t_sat: float = 373.15,
                                     p_liquid: float = 101325.0,
                                     gamma_lv: float = 67.70e-3,
                                     depth: float = 31.30e-10) -> np.ndarray:
    """Reduce the available-energy increment to its three defining parameters.

    The increment at nucleus radius r is the squared radius multiplied by the
    balance coefficient plus the Laplace coefficient multiplied by the natural
    logarithm of one plus the capillary length divided by r.

    Parameters
    ----------
    volume_factor : float
        Dimensionless volume factor of the segment (> 0).
    moment_factor : float
        Dimensionless temperature-weighted volume factor of the segment (> 0).
    t_liquid : float
        Wall-adjacent liquid temperature in kelvin (> 0).
    vapour_state : np.ndarray
        Array of shape (3,) holding the specific gas constant in J/(kg K), the
        saturated vapour density in kg/m^3 and the latent heat in J/kg.
    t_sat : float
        Saturation temperature of the liquid in kelvin (> 0).
    p_liquid : float
        Liquid pressure in pascal (> 0).
    gamma_lv : float
        Liquid-vapour surface tension in N/m (> 0).
    depth : float
        Depth of the cylindrical nucleus in metres (> 0).

    Returns
    -------
    coefficients : np.ndarray
        Array of shape (3,) holding the balance coefficient in J/m^2, the
        Laplace coefficient in J/m^2 and the capillary length in metres.
    """
    return coefficients  # placeholder
```

### Step 6

06_evaluate_available_energy_profile

Goal
----
Evaluate the available-energy increment of the nucleus over a grid of nucleus radii.

```python
import numpy as np

def evaluate_available_energy_profile(coefficients: np.ndarray,
                                      radii: np.ndarray) -> np.ndarray:
    """Evaluate the available-energy increment over a grid of nucleus radii.

    Parameters
    ----------
    coefficients : np.ndarray
        Array of shape (3,) holding the balance coefficient in J/m^2, the
        Laplace coefficient in J/m^2 and the capillary length in metres.
    radii : np.ndarray
        One-dimensional array of strictly positive nucleus radii in metres.

    Returns
    -------
    profile : np.ndarray
        Array with the same shape as radii holding the available-energy
        increment in joules.
    """
    return profile  # placeholder
```

### Step 7

07_locate_critical_nucleus

Goal
----
Locate the maximum of the available-energy increment and return the critical nucleus radius and the critical activation energy.

```python
import numpy as np

def locate_critical_nucleus(coefficients: np.ndarray) -> np.ndarray:
    """Locate the maximum of the available-energy increment.

    Parameters
    ----------
    coefficients : np.ndarray
        Array of shape (3,) holding the balance coefficient in J/m^2, the
        Laplace coefficient in J/m^2 and the capillary length in metres.

    Returns
    -------
    critical_state : np.ndarray
        Array of shape (2,) holding the critical nucleus radius in metres and
        the critical activation energy in joules.
    """
    return critical_state  # placeholder
```

### Step 8

08_compute_thermal_barrier_ratio

Goal
----
Express the critical activation energy in units of the thermal energy of the wall-adjacent liquid.

```python
import numpy as np

def compute_thermal_barrier_ratio(energy_critical: float, t_liquid: float) -> float:
    """Express the critical activation energy in units of the thermal energy.

    Parameters
    ----------
    energy_critical : float
        Critical activation energy in joules (> 0).
    t_liquid : float
        Wall-adjacent liquid temperature in kelvin (> 0).

    Returns
    -------
    barrier_ratio : float
        Critical activation energy divided by the product of the Boltzmann
        constant and the wall-adjacent liquid temperature.
    """
    return barrier_ratio  # placeholder
```

### Step 9

09_sweep_measured_contact_angles

Goal
----
Turn a table of measured quasi-steady bubble contact angles and their wall-adjacent liquid temperatures into the dimensionless nucleation barrier of each state.

```python
import numpy as np

def sweep_measured_contact_angles(theta_d_deg: np.ndarray, t_liquid: np.ndarray,
                                  vapour_state: np.ndarray,
                                  t_sat: float = 373.15,
                                  p_liquid: float = 101325.0,
                                  gamma_lv: float = 67.70e-3,
                                  depth: float = 31.30e-10) -> np.ndarray:
    """Compute the dimensionless nucleation barrier of every measured state.

    Parameters
    ----------
    theta_d_deg : np.ndarray
        One-dimensional array of measured quasi-steady bubble contact angles in
        degrees, each strictly between 0 and 180.
    t_liquid : np.ndarray
        One-dimensional array of wall-adjacent liquid temperatures in kelvin,
        one per measured state and of the same shape as theta_d_deg.
    vapour_state : np.ndarray
        Array of shape (3,) holding the specific gas constant in J/(kg K), the
        saturated vapour density in kg/m^3 and the latent heat in J/kg.
    t_sat : float
        Saturation temperature of the liquid in kelvin (> 0).
    p_liquid : float
        Liquid pressure in pascal (> 0).
    gamma_lv : float
        Liquid-vapour surface tension in N/m (> 0).
    depth : float
        Depth of the cylindrical nucleus in metres (> 0).

    Returns
    -------
    barrier_ratios : np.ndarray
        Array with the same shape as theta_d_deg holding the critical
        activation energy of each state in units of the thermal energy of its
        own wall-adjacent liquid.
    """
    return barrier_ratios  # placeholder
```

### Step 10

10_run_nucleation_barrier_pipeline

Goal
----
Chain the sub-problem functions 01-09 end-to-end on the measured flow-boiling states and return the largest dimensionless nucleation barrier.

Orchestrator: yes - this is the final step; it integrates the earlier sub-problems (convert_reduced_temperatures, compute_vapour_state, compute_segment_volume_factor, compute_thermal_volume_moment, compute_free_energy_coefficients, evaluate_available_energy_profile, locate_critical_nucleus, compute_thermal_barrier_ratio, sweep_measured_contact_angles) rather than reimplementing them.

```python
import numpy as np

def run_nucleation_barrier_pipeline(t_substrate_red: float = 9.95,
                                    t_jump_red: tuple = (1.45, 1.70, 1.95, 2.10,
                                                         2.05, 2.30, 2.55, 2.70),
                                    theta_d_deg: tuple = (117.5, 112.4, 106.8, 101.3,
                                                          61.4, 57.9, 53.6, 49.2),
                                    epsilon_ev: float = 0.008031,
                                    t_sat: float = 373.15,
                                    p_liquid: float = 101325.0,
                                    gamma_lv: float = 67.70e-3,
                                    depth: float = 31.30e-10) -> float:
    """Run the full nucleation-barrier measurement on the flow-boiling testbed.

    Parameters
    ----------
    t_substrate_red : float
        Substrate temperature in reduced Lennard-Jones units (> 0), shared by
        every measured state.
    t_jump_red : tuple
        Interfacial temperature jump of each measured state in reduced
        Lennard-Jones units (>= 0), of the same length as theta_d_deg.
    theta_d_deg : tuple
        Measured quasi-steady bubble contact angles in degrees, each strictly
        between 0 and 180.
    epsilon_ev : float
        Lennard-Jones well depth of the reference pair in electronvolts (> 0).
    t_sat : float
        Saturation temperature of the liquid in kelvin (> 0).
    p_liquid : float
        Liquid pressure in pascal (> 0).
    gamma_lv : float
        Liquid-vapour surface tension in N/m (> 0).
    depth : float
        Depth of the cylindrical nucleus in metres (> 0).

    Returns
    -------
    barrier_max : float
        Largest critical activation energy across the measured states, in units
        of the thermal energy of its own wall-adjacent liquid, as a native
        Python float.
    """
    return barrier_max  # placeholder
```

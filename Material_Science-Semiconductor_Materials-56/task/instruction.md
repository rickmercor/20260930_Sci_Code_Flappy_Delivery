# Material_Science-Semiconductor_Materials-56

## Background

Every logic transistor manufactured today dissipates its power in a volume a few tens of nanometres across. In a planar device that heat had a wide silicon substrate to spread into; in a fin or a stack of gate-all-around nanosheets it does not, because the channel is a thin body surrounded by oxide, the drain-side dissipation is concentrated into a hot spot smaller than the channel itself, and the only escape routes are through narrow silicon necks and across interfaces. The consequence is a self-heating problem that now sets circuit-level limits: the local temperature rise degrades carrier mobility and drive current, accelerates bias-temperature instability, hot-carrier injection and electromigration, and couples thermally to neighbouring devices. Predicting that rise is therefore part of device design rather than a downstream reliability check, and the tool the industry reaches for first Fourier's law with a bulk or a thin-film thermal conductivity is exactly the tool that stops being trustworthy at these dimensions.

It stops being trustworthy because heat in a semiconductor is carried by phonons whose mean free paths, in silicon at room temperature, are distributed over four orders of magnitude, with half of the conductivity coming from modes that travel a few hundred nanometres between collisions. When the device, or merely the heat source inside it, is smaller than those distances, the local-equilibrium assumption that Fourier's law rests on fails: phonons stream from the hot spot to a boundary without scattering, diffusive, quasi-ballistic and ballistic modes coexist in the same volume, and the temperature that appears in the heat equation is no longer defined by a local balance. This is the Casimir–Knudsen regime, and the two aspects of its name matter separately: boundary scattering truncates mean free paths at the device size, and a heat source smaller than the mean free paths produces the same non-local behaviour even where boundaries are far away. The observable symptoms are a temperature jump at every thermalising contact, an interior temperature profile flatter than the diffusive one, and an effective thermal resistance well above what any conductivity value inserted into Fourier's law would give.

The kinetic description that does hold is the phonon Boltzmann transport equation in the relaxation-time approximation, which tracks the phonon energy distribution over position, propagation direction and frequency band and relaxes it towards a local equilibrium set by the lattice temperature. Its cost is that the unknown lives in a high-dimensional phase space: a deterministic finite-volume discrete-ordinates solve must carry one transport problem per band and per direction over the whole mesh, and non-gray physics real dispersion, real polarisation-resolved lifetimes — is what separates a quantitative prediction from a qualitative one. Two design decisions dominate the cost. The first is how the continuous spectrum is compressed into a handful of representative bands, which is not a matter of slicing the frequency axis evenly but of respecting how conductivity accumulates with mean free path, since it is the long-mean-free-path tail that produces the size effect. The second is how the coupled system is iterated to steady state, because the obvious scheme — solve the transport equations, update the temperature from the local collision balance, repeat converges quickly where transport is ballistic and stalls where it is nearly diffusive, which has motivated synthetic schemes that carry a macroscopic diffusion equation alongside the kinetic one and feed the non-Fourier part of the flux back into it.

Once such a solver exists, the question it is built to answer is quantitative: by how much does the real, kinetic hot-spot temperature exceed the Fourier prediction, and how does that gap grow as the device shrinks? The comparison is only meaningful if the dissipation is scaled so that the diffusive prediction stays fixed while the geometry changes, since otherwise the trend merely reflects the change in heating power. What is left after that normalisation is a purely kinetic amplification, and it is the number a thermal engineer needs in order to know whether a compact model built on an effective conductivity is safe, or whether it is understating the junction temperature of the device it is supposed to protect.

## Problem

Dissipation in an aggressively scaled transistor is deposited in a volume whose dimensions lie well below the phonon mean free paths that have to carry the heat away, so the Fourier estimate of the hot-spot temperature rise stops being conservative and becomes an underestimate of unknown size. For the film specified below, quantify that failure: compute the peak steady-state lattice-temperature rise given by the non-gray phonon Boltzmann transport equation under the relaxation-time approximation, divided by the peak rise Fourier's law gives for the same geometry, the same dissipation, the same isothermal walls and the bulk conductivity of the same material.

The phonon spectrum is an isotropic Debye construction: a sphere of radius $k_D = (6\pi^2 n)^{1/3}$ with $n = 5.00\times10^{28}$ m$^{-3}$, carrying three degenerate acoustic polarisations with $\omega(k) = \omega_{\max}\sin\!\left(\pi k / 2k_D\right)$, $\omega_{\max}/2\pi = 9.0$ THz, and group velocities given by the analytic derivative. Sample it on $N_k = 400$ shells of equal width in $k$, each represented by its midpoint, with Bose–Einstein heat capacities at $T_0 = 300$ K. Mode lifetimes are $\tau(\omega)^{-1} = A\omega^4 + B\omega^2$ with $A = 1.32\times10^{-45}$ s$^3$ held fixed and $B$ fixed by requiring the bulk relaxation-time-approximation conductivity of this model at $T_0$ to equal $\kappa = 148$ W m$^{-1}$ K$^{-1}$. Reduce the spectrum to $N_b = 12$ bands of equal contribution to $\kappa$, obtained by cutting the mean-free-path-ordered spectrum at equally spaced levels of accumulated conductivity, each sampled shell belonging to the band that contains the accumulated fraction reached once that shell is included. Represent each band by a single heat capacity, a single group velocity and a single relaxation time that reproduce that band's heat capacity, its contribution to $\kappa$, and its heat-capacity-weighted group velocity.

The device is a film of thickness $L = 20$ nm with transport across the thickness, both faces being thermalising reservoirs held at $T_0$. Discretise the film into $N_c = 200$ uniform finite volumes with first-order upwinding, and the angular space into $N_d = 16$ Gauss–Legendre ordinates in the direction cosine, weighted so that the ordinate set spans the full solid angle. A volumetric dissipation of $1.5\times10^{19}$ W m$^{-3}$ is deposited uniformly over the central fifth of the film and enters the transport equation as an equilibrium phonon source. All phonon properties are evaluated at $T_0$ and the equilibrium distribution is linearised about it, so the transport problem is linear in the temperature rise; drive the steady state until the relative max-norm change of the temperature field between successive updates falls below $10^{-12}$. Your final answer must be a single number: the self-heating amplification factor for this configuration.

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

01_build_spectral_model

Goal
----
Sample the isotropic phonon spectrum of the testbed material and return, for every sampled shell, its angular frequency, its group velocity and the volumetric heat capacity it contributes at the reference temperature.

```python
import numpy as np
def build_spectral_model(n_shells: int, omega_max: float, number_density: float,
                         temperature: float) -> np.ndarray:
    """Sample the isotropic phonon spectrum of the testbed material.

    Parameters
    ----------
    n_shells : int
        Number of equal-width shells in wave-vector magnitude (n_shells >= 1).
    omega_max : float
        Maximum angular frequency of the dispersion in rad/s (omega_max > 0).
    number_density : float
        Atomic number density in m^-3 (number_density > 0).
    temperature : float
        Reference temperature in K (temperature > 0).

    Returns
    -------
    spectral : np.ndarray
        Array of shape (n_shells, 3) whose columns are the angular frequency
        in rad/s, the group velocity in m/s and the volumetric heat capacity
        in J/(m^3 K) of each sampled shell.

    Raises
    ------
    ValueError
        If ``n_shells`` is not an integer at least 1, or if ``omega_max``,
        ``number_density`` or ``temperature`` is not finite and strictly
        positive.
    """
    return spectral  # placeholder
```

### Step 2

02_calibrate_umklapp_coefficient

Goal
----
Fix the strength of the anharmonic scattering term of the lifetime model by requiring the bulk relaxation-time-approximation conductivity of the sampled spectrum to reproduce a prescribed value.

```python
import numpy as np

def calibrate_umklapp_coefficient(spectral: np.ndarray, impurity: float,
                                  kappa_bulk: float) -> float:
    """Fix the anharmonic scattering coefficient from the bulk conductivity.

    Parameters
    ----------
    spectral : np.ndarray
        Array of shape (n_shells, 3) holding the angular frequency, group
        velocity and volumetric heat capacity of every sampled shell.
    impurity : float
        Coefficient of the fourth-power scattering channel in s^3
        (impurity >= 0).
    kappa_bulk : float
        Target bulk thermal conductivity in W/(m K) (kappa_bulk > 0).

    Returns
    -------
    umklapp : float
        Coefficient of the second-power scattering channel in s, as a native
        Python float, for which the relaxation-time-approximation
        conductivity of the sampled spectrum equals kappa_bulk.

    Raises
    ------
    ValueError
        If ``spectral`` is not a finite ``(n_shells, 3)`` array with positive
        frequencies and non-negative heat capacities, if ``impurity`` is not
        finite and non-negative, if ``kappa_bulk`` is not finite and strictly
        positive, or if no admissible coefficient brackets the target.
    """
    return umklapp  # placeholder
```

### Step 3

03_discretize_phonon_bands

Goal
----
Compress the sampled spectrum into a small set of representative phonon bands of equal contribution to the bulk conductivity, each carrying one heat capacity, one group velocity and one relaxation time.

```python
import numpy as np

def discretize_phonon_bands(spectral: np.ndarray, impurity: float,
                            umklapp: float, n_bands: int) -> np.ndarray:
    """Reduce the sampled spectrum to representative phonon bands.

    Parameters
    ----------
    spectral : np.ndarray
        Array of shape (n_shells, 3) holding the angular frequency, group
        velocity and volumetric heat capacity of every sampled shell.
    impurity : float
        Coefficient of the fourth-power scattering channel in s^3
        (impurity >= 0).
    umklapp : float
        Coefficient of the second-power scattering channel in s
        (umklapp > 0).
    n_bands : int
        Number of representative bands (1 <= n_bands <= n_shells).

    Returns
    -------
    bands : np.ndarray
        Array of shape (n_bands, 3) whose columns are the volumetric heat
        capacity in J/(m^3 K), the group velocity in m/s and the relaxation
        time in s of each representative band, ordered by increasing mean
        free path.

    Raises
    ------
    ValueError
        If ``spectral`` has an invalid shape or contains non-finite or
        non-physical values, if either scattering coefficient is outside its
        stated domain, if ``n_bands`` is not an integer in
        ``[1, n_shells]``, or if the requested partition contains an empty or
        zero-heat-capacity band.
    """
    return bands  # placeholder
```

### Step 4

04_build_angular_quadrature

Goal
----
Build the discrete-ordinates set for one-dimensional cross-plane transport, returning the direction cosines and the solid-angle weights that integrate a distribution over the full sphere.

```python
import numpy as np

def build_angular_quadrature(n_dirs: int) -> np.ndarray:
    """Build the one-dimensional discrete-ordinates set.

    Parameters
    ----------
    n_dirs : int
        Number of ordinates; must be even and at least 2.

    Returns
    -------
    quadrature : np.ndarray
        Array of shape (n_dirs, 2) whose first column holds the direction
        cosines and whose second column holds the solid-angle weights, the
        weights summing to the full solid angle.

    Raises
    ------
    ValueError
        If ``n_dirs`` is not an even integer at least 2.
    """
    return quadrature  # placeholder
```

### Step 5

05_assemble_transport_operator

Goal
----
Assemble the reusable finite-volume transport matrices for every band-direction pair on the one-dimensional mesh, combining the collision term with the upwinded advection term and the thermalising wall condition.

```python
import numpy as np

def assemble_transport_operator(bands: np.ndarray, quadrature: np.ndarray,
                                n_cells: int,
                                cell_size: float) -> np.ndarray:
    """Assemble the reusable transport matrices for all band-direction pairs.

    Parameters
    ----------
    bands : np.ndarray
        Array of shape (n_bands, 3) holding the heat capacity, group velocity
        and relaxation time of each representative band.
    quadrature : np.ndarray
        Array of shape (n_dirs, 2) holding the direction cosines and the
        solid-angle weights of the ordinate set.
    n_cells : int
        Number of finite volumes across the film (n_cells >= 1).
    cell_size : float
        Width of a finite volume in m (cell_size > 0).

    Returns
    -------
    operators : np.ndarray
        Array of shape (n_bands, n_dirs, n_cells, n_cells) holding every
        temperature-independent transport matrix in units of s^-1.

    Raises
    ------
    ValueError
        If ``bands`` or ``quadrature`` has an invalid shape or contains
        non-finite values, if a band velocity or relaxation time is not
        strictly positive, if a direction cosine lies outside ``[-1, 1]``,
        if ``n_cells`` is not an integer at least 1, or if ``cell_size`` is
        not finite and strictly positive.
    """
    return operator  # placeholder
```

### Step 6

06_distribute_mode_source

Goal
----
Convert a volumetric dissipation profile into the per-band, per-direction source density that enters each band-direction transport equation.

```python
import numpy as np

def distribute_mode_source(bands: np.ndarray, heat_source: np.ndarray,
                           solid_angle: float) -> np.ndarray:
    """Split a volumetric dissipation profile among bands and directions.

    Parameters
    ----------
    bands : np.ndarray
        Array of shape (n_bands, 3) holding the heat capacity, group velocity
        and relaxation time of each representative band.
    heat_source : np.ndarray
        Array of shape (n_cells,) holding the volumetric dissipation in each
        finite volume, in W/m^3.
    solid_angle : float
        Total angular quadrature weight in steradians (solid_angle > 0).

    Returns
    -------
    mode_source : np.ndarray
        Array of shape (n_bands, n_cells) holding the source density per unit
        solid angle of every band in every cell, in W/(m^3 sr).

    Raises
    ------
    ValueError
        If ``bands`` or ``heat_source`` has an invalid shape or contains
        non-finite values, if any band heat capacity is not strictly positive,
        or if ``solid_angle`` is not finite and strictly positive.
    """
    return mode_source  # placeholder
```

### Step 7

07_solve_transport_sweep

Goal
----
Reuse the preassembled band-direction transport operators to solve every transport equation once for a frozen lattice-temperature field and return the resulting deviational energy-density tensor.

```python
import numpy as np

def solve_transport_sweep(bands: np.ndarray, quadrature: np.ndarray,
                          transport_operators: np.ndarray,
                          mode_source: np.ndarray,
                          lattice_temperature: np.ndarray) -> np.ndarray:
    """Solve every band-direction transport equation for a frozen temperature.

    Parameters
    ----------
    bands : np.ndarray
        Array of shape (n_bands, 3) holding the heat capacity, group velocity
        and relaxation time of each representative band.
    quadrature : np.ndarray
        Array of shape (n_dirs, 2) holding the direction cosines and the
        solid-angle weights of the ordinate set.
    transport_operators : np.ndarray
        Array of shape (n_bands, n_dirs, n_cells, n_cells) holding the
        temperature-independent matrices assembled by
        ``assemble_transport_operator`` before the outer iteration.
    mode_source : np.ndarray
        Array of shape (n_bands, n_cells) holding the isotropic per-band
        source density in W/(m^3 sr).
    lattice_temperature : np.ndarray
        Array of shape (n_cells,) holding the current temperature rise above
        the reference temperature, in K.
    Returns
    -------
    energy : np.ndarray
        Array of shape (n_bands, n_dirs, n_cells) holding the deviational
        energy density per unit solid angle, in J/(m^3 sr).

    Raises
    ------
    ValueError
        If the arrays have incompatible shapes or contain non-finite values,
        if any band velocity or relaxation time is not strictly positive, if
        a direction cosine lies outside ``[-1, 1]``, if the quadrature weights
        do not sum to a positive solid angle, or if a supplied transport
        operator has a non-positive diagonal.
    """
    return energy  # placeholder
```

### Step 8

08_compute_macroscopic_fields

Goal
----
Reduce the band-direction energy tensor to the three macroscopic fields the outer iteration needs: the lattice temperature implied by the collision balance, the heat flux, and the divergence of that flux.

```python
import numpy as np

def compute_macroscopic_fields(bands: np.ndarray, quadrature: np.ndarray,
                               energy: np.ndarray,
                               cell_size: float) -> np.ndarray:
    """Reduce the energy tensor to the macroscopic fields of the iteration.

    Parameters
    ----------
    bands : np.ndarray
        Array of shape (n_bands, 3) holding the heat capacity, group velocity
        and relaxation time of each representative band.
    quadrature : np.ndarray
        Array of shape (n_dirs, 2) holding the direction cosines and the
        solid-angle weights of the ordinate set.
    energy : np.ndarray
        Array of shape (n_bands, n_dirs, n_cells) holding the deviational
        energy density per unit solid angle.
    cell_size : float
        Width of a finite volume in m (cell_size > 0).

    Returns
    -------
    fields : np.ndarray
        Array of shape (3, n_cells) whose rows are the lattice temperature
        rise in K, the heat flux along the transport coordinate in W/m^2 and
        the divergence of that flux in W/m^3.

    Raises
    ------
    ValueError
        If the band, quadrature or energy arrays have incompatible shapes or
        contain non-finite values, if any band heat capacity or relaxation
        time is not strictly positive, if ``cell_size`` is not finite and
        strictly positive, or if the collision closure denominator is not
        positive.
    """
    return fields  # placeholder
```

### Step 9

9_update_macroscopic_temperature

Goal
----
Solve the macroscopic diffusion equation for the temperature field driven by the dissipation and by the divergence of the non-Fourier part of the heat flux, between isothermal walls.

```python
import numpy as np

def update_macroscopic_temperature(kappa_bulk: float, heat_source: np.ndarray,
                                   flux_divergence: np.ndarray,
                                   previous_temperature: np.ndarray,
                                   cell_size: float) -> np.ndarray:
    """Solve the macroscopic diffusion equation for the temperature rise.

    Parameters
    ----------
    kappa_bulk : float
        Bulk thermal conductivity in W/(m K) (kappa_bulk > 0).
    heat_source : np.ndarray
        Array of shape (n_cells,) holding the volumetric dissipation in W/m^3.
    flux_divergence : np.ndarray
        Array of shape (n_cells,) holding the divergence of the kinetic heat
        flux in W/m^3.
    previous_temperature : np.ndarray
        Array of shape (n_cells,) holding the temperature rise of the previous
        iterate in K.
    cell_size : float
        Width of a finite volume in m (cell_size > 0).

    Returns
    -------
    temperature : np.ndarray
        Array of shape (n_cells,) holding the updated temperature rise above
        the wall temperature, in K.

    Raises
    ------
    ValueError
        If the field arrays are empty, non-finite or do not have identical
        one-dimensional shapes, or if ``kappa_bulk`` or ``cell_size`` is not
        finite and strictly positive.
    """
    return temperature  # placeholder
```

### Step 10

10_run_self_heating_pipeline

Goal
----
Chain the sub-problem functions 01-09 end to end on the film testbed and return the self-heating amplification factor, the peak kinetic temperature rise divided by the peak Fourier temperature rise.

```python
import numpy as np

def run_self_heating_pipeline(length: float = 2.0e-8, n_cells: int = 200,
                              n_bands: int = 12, n_dirs: int = 16,
                              n_shells: int = 400,
                              omega_max: float = 56548667764616.27,
                              number_density: float = 5.0e28,
                              temperature: float = 300.0,
                              impurity: float = 1.32e-45,
                              kappa_bulk: float = 148.0,
                              power_density: float = 1.5e19,
                              hotspot_fraction: float = 0.2,
                              tol: float = 1.0e-12, max_iter: int = 5000,
                              scheme: str = "sequential") -> float:
    """Run the full self-heating amplification measurement on the film testbed.

    Parameters
    ----------
    length : float
        Film thickness in m (length > 0).
    n_cells : int
        Number of finite volumes across the film (n_cells >= 1).
    n_bands : int
        Number of representative phonon bands
        (1 <= n_bands <= n_shells).
    n_dirs : int
        Number of ordinates; must be even and at least 2.
    n_shells : int
        Number of spectral shells (n_shells >= 1).
    omega_max : float
        Maximum angular frequency of the dispersion in rad/s (omega_max > 0).
    number_density : float
        Atomic number density in m^-3 (number_density > 0).
    temperature : float
        Reference temperature in K (temperature > 0).
    impurity : float
        Coefficient of the fourth-power scattering channel in s^3
        (impurity >= 0).
    kappa_bulk : float
        Bulk thermal conductivity in W/(m K) (kappa_bulk > 0).
    power_density : float
        Volumetric dissipation inside the hot spot in W/m^3
        (power_density > 0).
    hotspot_fraction : float
        Fraction of the film thickness occupied by the centred hot spot,
        0 < hotspot_fraction <= 1.
    tol : float
        Relative max-norm convergence tolerance on the temperature field
        (tol > 0).
    max_iter : int
        Maximum number of outer iterations (max_iter >= 1).
    scheme : str
        Temperature update driving the iteration, either "sequential" or
        "synthetic".

    Returns
    -------
    amplification : float
        Peak kinetic temperature rise divided by the peak Fourier temperature
        rise, as a native Python float.

    Raises
    ------
    ValueError
        If an integer resolution or iteration count is outside its stated
        domain, if ``n_dirs`` is odd, if a physical scalar or tolerance is
        non-finite or outside its stated domain, if ``n_bands`` exceeds
        ``n_shells``, if ``scheme`` is not ``"sequential"`` or
        ``"synthetic"``, or if the discretised source yields no positive
        Fourier temperature rise.
    """
    return amplification  # placeholder
```

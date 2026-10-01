# osc

## Problem

An organic solar cell has an undoped active layer between two ohmic contacts, so its
photocurrent is limited not by how many free charges are generated but by how many of them
reach an electrode before recombining with one another. The source examines what the
photocurrent of such a device actually reports, using a one-dimensional drift-diffusion
description in which free charges are generated uniformly and the only loss channel available
to them is bimolecular recombination between the two carrier species.

Implement that description and report the fill factor of the illuminated device, converged with
respect to the spatial discretisation.

The following are declared:

```
LAYER       thickness 100 nm, relative permittivity 3.5, temperature 300 K. Undoped: there is
            no fixed space charge anywhere in the layer.

BANDS       transport gap 1.40 eV. Effective density of states 1e20 cm^-3 for each carrier.

CONTACTS    both ohmic, with an injection barrier of 0.25 eV at each. Each contact pins both
            carrier densities at their equilibrium values there, so extraction is not itself
            rate limiting. The potential is pinned at both contacts, the applied bias entering
            as the difference between them.

TRANSPORT   balanced mobilities, 2e-4 cm^2 V^-1 s^-1 for each carrier, independent of field
            and of density. Einstein's relation holds.

GENERATION  free charges are generated uniformly at 1e22 cm^-3 s^-1, independent of field, with
            a generation efficiency of one everywhere. The recombination reduction factor is
            one, so recombination runs at the full strength the source's coefficient sets.

QUANTITY    fill factor, meaning the largest output power density the illuminated device can
            deliver divided by the product of its short-circuit current density and its
            open-circuit voltage.
```

Everything else is stated uniquely by the source and is yours to recover: the equations the
model couples, how the carrier flux between neighbouring points must be written so the
discretisation survives the potential drop across this layer, what fixes the recombination
coefficient, and what the built-in voltage and the contact densities follow from.

State the conventions you adopted and justify each from the source, and report the four
scalars your fill factor is built from: the short-circuit current density, the open-circuit
voltage, the bias at the maximum power point, and the power density there. Give the numerical
evidence that your fill factor is converged in the spatial grid. Also state what the collection
efficiency would become if recombination were switched off, what the source concludes about
what the photocurrent versus effective voltage construction actually reports, and where that
source finds the normalised photocurrent to saturate.

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
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.
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

derived_constants

Goal
----
Return the derived constants the rest of the calculation needs, as a dictionary with keys 'VT', 'Vbi', 'beta_L', 'ni2' and 'eps': the thermal voltage, the built-in voltage, the bimolecular recombination coefficient, the square of the intrinsic density, and the absolute permittivity. Raise ValueError for a non-positive temperature.

```python
def derived_constants(spec: dict) -> dict:
    """Return the derived constants the rest of the calculation needs, as a dictionary with keys 'VT', 'Vbi', 'beta_L', 'ni2' and 'eps': the thermal voltage, the built-in voltage, the bimolecular recombination coefficient, the square of the intrinsic density, and the absolute permittivity. Raise ValueError for a non-positive temperature.

    Parameters
    ----------
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).

    Returns
    -------
    dict with keys 'VT', 'Vbi', 'beta_L', 'ni2', 'eps', all SI floats
    """
    return {}
```

### Step 2

bernoulli

Goal
----
Evaluate the Bernoulli function that the source's flux discretisation is built on, elementwise and without loss of accuracy near zero or at large magnitude. Raise ValueError for a non-finite argument.

```python
def bernoulli(x: np.ndarray) -> np.ndarray:
    """Evaluate the Bernoulli function that the source's flux discretisation is built on, elementwise and without loss of accuracy near zero or at large magnitude. Raise ValueError for a non-finite argument.

    Parameters
    ----------
    x : np.ndarray
        Dimensionless argument, elementwise.

    Returns
    -------
    np.ndarray of the same shape as the input, dimensionless
    """
    return x
```

### Step 3

contact_densities

Goal
----
Return the equilibrium carrier densities held at the two ohmic contacts, as the four numbers [electrons at the anode, holes at the anode, electrons at the cathode, holes at the cathode]. Raise ValueError if the injection barrier does not lie inside the transport gap.

```python
def contact_densities(spec: dict) -> np.ndarray:
    """Return the equilibrium carrier densities held at the two ohmic contacts, as the four numbers [electrons at the anode, holes at the anode, electrons at the cathode, holes at the cathode]. Raise ValueError if the injection barrier does not lie inside the transport gap.

    Parameters
    ----------
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).

    Returns
    -------
    np.ndarray of shape (4,), in inverse cubic metres
    """
    return None
```

### Step 4

poisson_step

Goal
----
Advance the electrostatic potential by one linearised Poisson solve at fixed carrier densities, on a uniform grid spanning the active layer. Raise ValueError if the three profiles do not share one grid.

```python
def poisson_step(psi: np.ndarray, n: np.ndarray, p: np.ndarray, spec: dict, V: float) -> np.ndarray:
    """Advance the electrostatic potential by one linearised Poisson solve at fixed carrier densities, on a uniform grid spanning the active layer. Raise ValueError if the three profiles do not share one grid.

    Parameters
    ----------
    psi : np.ndarray
        Current electrostatic potential on a uniform grid, in volts.
    n : np.ndarray
        Electron density on the same grid.
    p : np.ndarray
        Hole density on the same grid.
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).
    V : float
        Applied bias, in volts.

    Returns
    -------
    np.ndarray of the same shape as psi, in volts
    """
    return psi
```

### Step 5

electron_density

Goal
----
Solve the electron continuity equation on the given potential for the electron density, using the source's flux discretisation, with the recombination term linearised in the electron density at the given hole density. The contact values are held fixed. Raise ValueError if the potential and the hole density do not share one grid, or if the generation rate is negative.

```python
def electron_density(psi: np.ndarray, p: np.ndarray, spec: dict, G: float) -> np.ndarray:
    """Solve the electron continuity equation on the given potential for the electron density, using the source's flux discretisation, with the recombination term linearised in the electron density at the given hole density. The contact values are held fixed. Raise ValueError if the potential and the hole density do not share one grid, or if the generation rate is negative.

    Parameters
    ----------
    psi : np.ndarray
        Electrostatic potential on a uniform grid, in volts.
    p : np.ndarray
        Hole density on the same grid.
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).
    G : float
        Uniform free-charge generation rate.

    Returns
    -------
    np.ndarray of the same shape as psi, in inverse cubic metres
    """
    return psi
```

### Step 6

hole_density

Goal
----
Solve the hole continuity equation on the given potential for the hole density, using the same flux discretisation as for electrons, with the recombination term linearised in the hole density at the given electron density. The contact values are held fixed. Raise ValueError if the potential and the electron density do not share one grid, or if the generation rate is negative.

```python
def hole_density(psi: np.ndarray, n: np.ndarray, spec: dict, G: float) -> np.ndarray:
    """Solve the hole continuity equation on the given potential for the hole density, using the same flux discretisation as for electrons, with the recombination term linearised in the hole density at the given electron density. The contact values are held fixed. Raise ValueError if the potential and the electron density do not share one grid, or if the generation rate is negative.

    Parameters
    ----------
    psi : np.ndarray
        Electrostatic potential on a uniform grid, in volts.
    n : np.ndarray
        Electron density on the same grid.
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).
    G : float
        Uniform free-charge generation rate.

    Returns
    -------
    np.ndarray of the same shape as psi, in inverse cubic metres
    """
    return psi
```

### Step 7

self_consistent_state

Goal
----
Iterate the potential and the two carrier densities to mutual self-consistency at the given bias and generation rate, and return them as the tuple (psi, n, p). Stop when the largest absolute change in the potential and the largest change in each density relative to that density's own maximum have all fallen below 1e-6, and raise ValueError if that has not happened within 2000 sweeps. Raise ValueError for a grid of fewer than five nodes.

```python
def self_consistent_state(V: float, G: float, spec: dict, n_grid: int) -> tuple:
    """Iterate the potential and the two carrier densities to mutual self-consistency at the given bias and generation rate, and return them as the tuple (psi, n, p). Stop when the largest absolute change in the potential and the largest change in each density relative to that density's own maximum have all fallen below 1e-6, and raise ValueError if that has not happened within 2000 sweeps. Raise ValueError for a grid of fewer than five nodes.

    Parameters
    ----------
    V : float
        Applied bias, in volts.
    G : float
        Uniform free-charge generation rate.
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).
    n_grid : int
        Number of uniformly spaced nodes, both contacts included.

    Returns
    -------
    tuple (psi, n, p) of np.ndarray, each of shape (n_grid,)
    """
    return (psi, n, p)
```

### Step 8

terminal_current

Goal
----
Return the terminal current density of the device in amperes per square metre, taking the sign convention in which a current driven out of the device by the built-in field is negative. Raise ValueError if the total current is not independent of position to within one part in a thousand.

```python
def terminal_current(psi: np.ndarray, n: np.ndarray, p: np.ndarray, spec: dict) -> float:
    """Return the terminal current density of the device in amperes per square metre, taking the sign convention in which a current driven out of the device by the built-in field is negative. Raise ValueError if the total current is not independent of position to within one part in a thousand.

    Parameters
    ----------
    psi : np.ndarray
        Self-consistent potential on a uniform grid, in volts.
    n : np.ndarray
        Self-consistent electron density on the same grid.
    p : np.ndarray
        Self-consistent hole density on the same grid.
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).

    Returns
    -------
    float, in amperes per square metre
    """
    return 0.0
```

### Step 9

open_circuit_voltage

Goal
----
Return the bias at which the illuminated device carries no terminal current, in volts. Raise ValueError if no such bias exists below flat band.

```python
def open_circuit_voltage(spec: dict, n_grid: int) -> float:
    """Return the bias at which the illuminated device carries no terminal current, in volts. Raise ValueError if no such bias exists below flat band.

    Parameters
    ----------
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).
    n_grid : int
        Number of uniformly spaced nodes, both contacts included.

    Returns
    -------
    float, in volts
    """
    return 0.0
```

### Step 10

fill_factor

Goal
----
Return the fill factor of the illuminated device, extrapolated to the continuum from solutions on the two given grids. Raise ValueError if the fine grid is not finer than the coarse one, or if the recombination reduction factor is negative.

```python
def fill_factor(spec: dict, n_coarse: int, n_fine: int) -> float:
    """Return the fill factor of the illuminated device, extrapolated to the continuum from solutions on the two given grids. Raise ValueError if the fine grid is not finer than the coarse one, or if the recombination reduction factor is negative.

    Parameters
    ----------
    spec : dict
        The declared device configuration, in SI units, with keys 'T' (temperature, K), 'd' (layer thickness, m), 'Eg' (transport gap, eV), 'Phi' (injection barrier at each contact, eV), 'mu_n' and 'mu_p' (mobilities, m^2 V^-1 s^-1), 'eps_r' (relative permittivity), 'Nc' and 'Nv' (effective densities of states, m^-3), 'Gex' (generation rate, m^-3 s^-1) and 'gamma' (recombination reduction factor).
    n_coarse : int
        Node count of the coarser of the two grids.
    n_fine : int
        Node count of the finer of the two grids.

    Returns
    -------
    float, dimensionless
    """
    return 0.0
```

# Chemistry-Computational_Chemistry-62

## Background

## Polyelectrolyte complexes and salt doping

When a polycation and a polyanion are mixed in water they associate into a dense, water-swollen complex held together by intrinsic ion pairs between oppositely charged repeat units, releasing their small counterions into solution. Added salt reverses part of this: small ions enter the complex, break intrinsic pairs and pair with the freed charges as extrinsic ion pairs. The fraction of repeat units compensated this way, the doping level, controls how soft the complex is, how fast its chains relax and where its glass transition lies, which is why salt can be used to process these materials, for example as extrudable saloplastics, or to trigger shape changes.

## Chemistry-based and physics-based descriptions

Two traditions describe this equilibrium. Chemistry-based treatments write the doping reaction with an equilibrium constant, which captures the specific affinity of each ion for the polymer charges and follows the Hofmeister ordering of ions. Physics-based treatments start from mean-field or field theories of charged polymers, such as the Voorn-Overbeek picture and the random-phase approximation, and describe how long-range electrostatic correlations of the ions change the free energy. Combined approaches correct the equilibrium constants with correlation chemical potentials, so that ion specificity and electrostatic screening enter together, and couple the doping equilibrium to the uptake of water by the complex through polymer solution and rubber elasticity theory.

## Multivalent counterions

Monovalent salts pair one small ion with one polymer charge. Multivalent cations can instead coordinate two or more polyanion charges at once, forming physical crosslinks that stiffen the complex, or bind a single polymer charge while keeping some of their own counterions. Which of these arrangements prevails depends on the cation, its hydration and the salt concentration, and it matters for the mechanical, thermal and transport properties of complexes and multilayers used in coatings, membranes, drug delivery and responsive materials.

## Problem

Polyelectrolyte complexes of poly(styrene sulfonate) and poly(diallyldimethylammonium) take up salt by doping, in which ions from the surrounding solution break intrinsic polycation-polyanion pairs, and a divalent cation such as Ca2+ can then either bridge two polyanion repeat units or occupy a single one while keeping a chloride counterion. A recent phenomenological model extends the salt-doping equilibrium of these complexes to multivalent cations by giving each binding mode its own mass-action law and doping constant, correcting every doping constant with the random-phase electrostatic correlation free energy of the free, Gaussian-smeared salt ions, and solving these laws self-consistently with the swelling equilibrium of the initially dry, salt-free complex (Flory-Huggins mixing, tube-model entanglement and a phantom network crosslinked by intrinsic pairs and bridging cations). Given the salt, its concentration and the ion and polymer parameters, the model returns the fraction of polyion repeat units bound in each mode and the polymer volume fraction of the swollen complex.

Treat the complex as symmetric (identical polycation and polyanion) at 298.15 K, with a repeat-unit size of 6.5 water volumes, 2500 repeat units per chain, a polyelectrolyte concentration of 0.05 M before phase separation, a polymer-water Flory-Huggins parameter of 0.3, and the entanglement group v_P N_e alpha_tube (b/a_0)^2 and the crosslink group v_P N_+- both equal to 1. Describe Ca2+ by a bare radius of 1.00 Angstrom, a hydration number of 7.2 and standard free energies of -4.2 k_BT for the mixed mode and -0.1 k_BT for the bridging mode, Mg2+ by 0.72 Angstrom, 10.0, -2.6 k_BT and -0.7 k_BT, and Cl- by 1.81 Angstrom and 2.0; take the water molecular volume as 29.7 Angstrom^3, the molarity of water as 55.56 M and a relative permittivity of 79 in the Bjerrum length, give each ion the hydration-corrected effective volume of the model and a Gaussian smearing size equal to the cube root of that effective volume, measure every length and wavenumber in the correlation integral in units of the cube root of the water molecular volume, and set the water volume fraction of the complex to one minus its polymer volume fraction. Take the salt left free in the bath as the added salt concentration minus the 0.05 M polyelectrolyte concentration times the total fraction of repeat units bound in all modes, with each free salt unit supplying one cation and two chloride ions, so that the free cation and chloride concentrations are that free salt concentration and twice it.

Find the CaCl2 concentration at which the mixed mode and the bridging mode bind equal numbers of calcium ions. In the reasoning, also give for 0.10 M CaCl2 the mixed-mode and bridging-mode site fractions, the polymer volume fraction and the doping constant of the mixed mode; the mixed-mode and bridging-mode site fractions for 0.10 M MgCl2; the MgCl2 concentration between 0.2 and 3 mol/L at which its two modes bind equal numbers of magnesium ions; the mixed-mode site fraction of CaCl2 at the concentration where the two modes bind equal numbers of calcium ions; the CaCl2 concentration at which the bridging site fraction is largest and that largest value; and, from the literature, the glass transition temperature that the calorimetric study of these complexes assembled in divalent chlorides reported for 0.10 M CaCl2 at 20 wt% hydration, the hydration at which the calorimetric study of counterion type in the same complexes found the glass transition temperature of its 0.1 M NaBr complexes down to 315 K, and the glass transition temperature that the earlier calorimetric study of the role of water in these assemblies measured at 16 wt% water for complexes prepared in 0.5 M NaCl. These quantities are among the scalars that determine and check the final number, so set them out in the reasoning, and report that CaCl2 concentration in mol/L to six decimal places as the final answer.

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

01_free_ion_volume_fractions

Goal
----
Effective volumes of the salt ions relative to water and the volume fractions of the ions that remain free in the bath once the complex has taken up salt.

```python
def free_ion_volume_fractions(salt_conc: float, xi_total: float, polymer_conc: float, valence: int,
                              ions: "np.ndarray") -> "np.ndarray":
    '''Effective ion volumes and volume fractions of the free salt ions.

    Parameters
    ----------
    salt_conc : float
        Salt concentration C_S of the bathing solution in mol/L.
    xi_total : float
        Total fraction xi_PS of polyion repeat units in salt-bound pairs.
    polymer_conc : float
        Repeat-unit concentration C_P0 of the polyelectrolyte solution before phase separation, mol/L.
    valence : int
        Charge number z of the cation; each salt unit carries one cation and z anions.
    ions : np.ndarray
        Shape (2, 2): row 0 cation, row 1 anion; column 0 bare radius (Angstrom), column 1
        hydration number.

    Returns
    -------
    result : np.ndarray
        Float array [omega_cation, omega_anion, phi_cation_free, phi_anion_free].

    Raises
    ------
    ValueError
        If valence is not an integer >= 1, ions is not a (2, 2) array, a radius is not positive, a
        hydration number is negative, salt_conc or polymer_conc is not positive, xi_total is outside
        [0, 1), or the free salt concentration salt_conc - polymer_conc * xi_total is not positive.
    '''
    return result
```

### Step 2

02_screening_wavenumber_squared

Goal
----
Wavenumber-dependent squared inverse screening length of the free, Gaussian-smeared salt ions in units of the water size.

```python
def screening_wavenumber_squared(q: "np.ndarray", phi_free: "np.ndarray", omega: "np.ndarray", valence: int,
                                 temperature: float, dielectric: float) -> "np.ndarray":
    '''Squared inverse screening length k2(q) of Gaussian-smeared free ions, in units of 1/l^2.

    Parameters
    ----------
    q : np.ndarray
        Dimensionless wavenumbers (in units of 1/l).
    phi_free : np.ndarray
        Shape (2,): free volume fractions of cation and anion.
    omega : np.ndarray
        Shape (2,): effective volumes of cation and anion relative to water.
    valence : int
        Cation charge number z.
    temperature : float
        Absolute temperature in K.
    dielectric : float
        Relative permittivity used in the Bjerrum length.

    Returns
    -------
    result : np.ndarray
        k2(q), same shape as q.

    Raises
    ------
    ValueError
        If phi_free or omega does not have shape (2,), an entry of phi_free is negative, an entry of
        omega is not positive, valence is not an integer >= 1, or temperature or dielectric is not
        positive.
    '''
    return result
```

### Step 3

03_log_doping_constants

Goal
----
Logarithms of the doping constants of the z binding modes of a z-valent cation, each corrected by the random-phase correlation integral of the free ions.

```python
def log_doping_constants(delta_g: "np.ndarray", phi_free: "np.ndarray", omega: "np.ndarray", valence: int,
                         temperature: float, dielectric: float) -> "np.ndarray":
    '''Natural logarithms of the correlation-corrected doping constants of the z binding modes.

    Parameters
    ----------
    delta_g : np.ndarray
        Shape (z,): standard free energies of the binding modes k = 1..z in units of k_B T.
    phi_free : np.ndarray
        Shape (2,): free volume fractions of cation and anion.
    omega : np.ndarray
        Shape (2,): effective volumes of cation and anion relative to water.
    valence : int
        Cation charge number z.
    temperature : float
        Absolute temperature in K.
    dielectric : float
        Relative permittivity used in the Bjerrum length.

    Returns
    -------
    result : np.ndarray
        Shape (z,): ln K_k for k = 1..z.

    Raises
    ------
    ValueError
        If delta_g does not have shape (valence,), or on any condition that makes the screening
        wavenumber of the free ions invalid (phi_free or omega not of shape (2,), a negative phi_free
        entry, a non-positive omega entry, valence not an integer >= 1, non-positive temperature or
        dielectric).
    '''
    return result
```

### Step 4

04_doping_at_fixed_swelling

Goal
----
Site fractions of all binding modes from their coupled mass-action laws at a prescribed polymer volume fraction, including free-salt depletion.

```python
def doping_at_fixed_swelling(salt_conc: float, polymer_fraction: float, polymer_conc: float, valence: int,
                             ions: "np.ndarray", delta_g: "np.ndarray", temperature: float,
                             dielectric: float) -> "np.ndarray":
    '''Site fractions of the z binding modes at a prescribed polymer volume fraction.

    Parameters
    ----------
    salt_conc : float
        Salt concentration C_S in mol/L.
    polymer_fraction : float
        Polymer volume fraction Phi_P of the complex.
    polymer_conc : float
        Repeat-unit concentration C_P0 before phase separation, mol/L.
    valence : int
        Cation charge number z.
    ions : np.ndarray
        Shape (2, 2): [[cation radius, cation hydration number], [anion radius, anion hydration number]].
    delta_g : np.ndarray
        Shape (z,): standard free energies of modes k = 1..z in units of k_B T.
    temperature : float
        Absolute temperature in K.
    dielectric : float
        Relative permittivity used in the Bjerrum length.

    Returns
    -------
    result : np.ndarray
        Shape (z,): the site fractions xi_k.

    Raises
    ------
    ValueError
        If polymer_fraction is not strictly between 0 and 1, delta_g does not have shape (valence,),
        valence is not an integer >= 1, ions is not a (2, 2) array with positive radii and non-negative
        hydration numbers, salt_conc, polymer_conc, temperature or dielectric is not positive.
    '''
    return result
```

### Step 5

05_swelling_polymer_fraction

Goal
----
Polymer volume fraction at which the free energy of the doped, crosslinked complex is stationary with respect to its water content.

```python
def swelling_polymer_fraction(xi_modes: "np.ndarray", chi: float, omega_p: float, n_p: float, entanglement: float,
                              crosslink: float) -> float:
    '''Polymer volume fraction of the complex at swelling equilibrium for given mode site fractions.

    Parameters
    ----------
    xi_modes : np.ndarray
        Shape (z,): site fractions of the binding modes k = 1..z.
    chi : float
        Polymer-water Flory-Huggins parameter.
    omega_p : float
        Size of a polyion repeat unit relative to a water molecule.
    n_p : float
        Number of repeat units per chain.
    entanglement : float
        Entanglement group v_P N_e alpha_tube (b/a_0)^2.
    crosslink : float
        Crosslink group v_P N_+-.

    Returns
    -------
    result : float
        Phi_P, strictly between 0 and 1.

    Raises
    ------
    ValueError
        If xi_modes is empty or not one dimensional, has a negative entry or entries summing to 1 or
        more, omega_p or n_p is not positive, entanglement or crosslink is negative, or the network term
        E + (1/2) X xi_PP + sum_{k=2..z} (1 - 1/k) xi_k / 2 is not positive.
    '''
    return result
```

### Step 6

06_equilibrium_doping

Goal
----
Self-consistent mode site fractions and polymer volume fraction of the complex swollen to equilibrium in the salt solution.

```python
def equilibrium_doping(salt_conc: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                       polymer_conc: float, omega_p: float, n_p: float, entanglement: float, crosslink: float,
                       temperature: float, dielectric: float) -> "np.ndarray":
    '''Equilibrium mode site fractions and polymer volume fraction of the swollen, doped complex.

    Parameters
    ----------
    salt_conc : float
        Salt concentration C_S in mol/L.
    valence : int
        Cation charge number z.
    ions : np.ndarray
        Shape (2, 2): [[cation radius, cation hydration number], [anion radius, anion hydration number]].
    delta_g : np.ndarray
        Shape (z,): standard free energies of modes k = 1..z in units of k_B T.
    chi : float
        Polymer-water Flory-Huggins parameter.
    polymer_conc : float
        Repeat-unit concentration C_P0 before phase separation, mol/L.
    omega_p : float
        Size of a repeat unit relative to water.
    n_p : float
        Number of repeat units per chain.
    entanglement : float
        Entanglement group v_P N_e alpha_tube (b/a_0)^2.
    crosslink : float
        Crosslink group v_P N_+-.
    temperature : float
        Absolute temperature in K.
    dielectric : float
        Relative permittivity used in the Bjerrum length.

    Returns
    -------
    result : np.ndarray
        Shape (z + 1,): [xi_1, ..., xi_z, Phi_P].

    Raises
    ------
    ValueError
        If any input violates the conditions of the mode doping at fixed swelling (salt_conc,
        polymer_conc, temperature or dielectric not positive, valence not an integer >= 1, ions not a
        (2, 2) array with positive radii and non-negative hydration numbers, delta_g not of shape
        (valence,)), or omega_p or n_p is not positive, or entanglement or crosslink is negative, or
        entanglement and crosslink are both zero.
    '''
    return result
```

### Step 7

07_dominant_mode_transitions

Goal
----
Salt concentrations, below a given upper limit, at which the binding mode holding the most cations changes as salt is added.

```python
def dominant_mode_transitions(c_max: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                              polymer_conc: float, omega_p: float, n_p: float, entanglement: float,
                              crosslink: float, temperature: float, dielectric: float) -> "np.ndarray":
    '''Salt concentrations below c_max at which the binding mode holding the most cations changes.

    Parameters
    ----------
    c_max : float
        Largest salt concentration considered, mol/L.
    valence : int
        Cation charge number z (at least 2).
    ions : np.ndarray
        Shape (2, 2): [[cation radius, cation hydration number], [anion radius, anion hydration number]].
    delta_g : np.ndarray
        Shape (z,): standard free energies of modes k = 1..z in units of k_B T.
    chi : float
        Polymer-water Flory-Huggins parameter.
    polymer_conc : float
        Repeat-unit concentration C_P0 before phase separation, mol/L.
    omega_p : float
        Size of a repeat unit relative to water.
    n_p : float
        Number of repeat units per chain.
    entanglement : float
        Entanglement group v_P N_e alpha_tube (b/a_0)^2.
    crosslink : float
        Crosslink group v_P N_+-.
    temperature : float
        Absolute temperature in K.
    dielectric : float
        Relative permittivity used in the Bjerrum length.

    Returns
    -------
    result : np.ndarray
        Shape (n,), n >= 0: the salt concentrations in mol/L at which the predominant mode changes, in
        increasing order.

    Raises
    ------
    ValueError
        If valence is not an integer >= 2, c_max is not positive, delta_g does not have shape (valence,),
        or any input is invalid for the equilibrium doping of the complex.
    '''
    return result
```

### Step 8

08_mixed_mode_takeover_concentration

Goal
----
Salt concentration above which the mixed binding mode holds more cations than every other mode; for a divalent cation, where the mixed and bridged modes hold equal numbers of cations.

```python
def mixed_mode_takeover_concentration(c_max: float, valence: int, ions: "np.ndarray", delta_g: "np.ndarray", chi: float,
                                      polymer_conc: float, omega_p: float, n_p: float, entanglement: float,
                                      crosslink: float, temperature: float, dielectric: float) -> float:
    '''Salt concentration above which the mixed mode holds more cations than every other binding mode.

    Parameters
    ----------
    c_max : float
        Largest salt concentration considered, mol/L.
    valence : int
        Cation charge number z (at least 2).
    ions : np.ndarray
        Shape (2, 2): [[cation radius, cation hydration number], [anion radius, anion hydration number]].
    delta_g : np.ndarray
        Shape (z,): standard free energies of modes k = 1..z in units of k_B T.
    chi : float
        Polymer-water Flory-Huggins parameter.
    polymer_conc : float
        Repeat-unit concentration C_P0 before phase separation, mol/L.
    omega_p : float
        Size of a repeat unit relative to water.
    n_p : float
        Number of repeat units per chain.
    entanglement : float
        Entanglement group v_P N_e alpha_tube (b/a_0)^2.
    crosslink : float
        Crosslink group v_P N_+-.
    temperature : float
        Absolute temperature in K.
    dielectric : float
        Relative permittivity used in the Bjerrum length.

    Returns
    -------
    result : float
        The takeover salt concentration C_mix in mol/L.

    Raises
    ------
    ValueError
        If valence is not an integer >= 2, c_max is not positive, the mixed mode is not the predominant
        mode at c_max (no takeover up to c_max), or any input is invalid for the equilibrium doping of the
        complex.
    '''
    return result
```

# Chemistry-Computational_Chemistry-66

## Background

Electrostatics at charged interfaces organises a large part of soft matter and biology: the stability of colloidal suspensions, the interactions of charged membranes, the condensation of counterions onto DNA and other polyelectrolytes, and the structure of the electrode-electrolyte interface all turn on how mobile ions arrange themselves near a charged surface. For dilute solutions the mean-field Poisson-Boltzmann theory describes that arrangement well, and its success rests on treating ions as point charges whose only interaction is the mean electrostatic field.

That approximation fails where the interface is most interesting. Above roughly one molar for a monovalent salt the mean spacing between ions falls to the scale of the ions themselves, and near a strongly charged surface the local concentration can be far higher than in the bulk regardless of how dilute the reservoir is. Point-charge theory responds by predicting densities that exceed close packing, which is not merely quantitatively wrong but qualitatively so: it misses saturation, the plateau in density that forms when the layer next to the wall is physically full. A family of corrections addresses this, some going beyond mean field to capture ion-ion correlations, others adding non-electrostatic interactions, and one line of work replacing the ideal mixing entropy by a lattice-gas entropy that knows the ions have volume. The sterically modified Poisson-Boltzmann theory of that last line reproduces saturation and remains analytically tractable, and it has become the standard mean-field account of a crowded double layer.

Its usual form carries a hidden restriction: ion and solvent are assumed to occupy the same lattice cell, so the theory has no way to express the fact that a hydrated sodium ion, a water molecule and a multivalent organic counterion are objects of quite different size. Replacing the symmetric lattice-gas entropy by a Flory-Huggins entropy, in which a species occupies a number of cells set by its own molecular volume, produces a mean-field theory of the double layer in which valency and size enter as independent parameters. The relation between surface charge and contact density then acquires a size-dependent form that reduces to the familiar one only when ion and solvent match, and the boundary between the regime where point-charge theory is adequate and the regime where saturation dominates moves with both the surface charge and the size ratio.

The richest consequence appears in mixtures. Experiments and simulations on surfaces exposed to several counterion species have reported stratification, with different species peaking at different distances from the wall and, in some cases, a lower-valency ion sitting closer to the surface than a higher-valency one, contrary to the naive expectation that a charged surface simply prefers the highest available charge. For a saturated layer a size-asymmetric theory explains this, because what such a layer rations is volume rather than charge. Far from saturation that argument has nothing to act on, so how the arrangement of the layers changes as a surface is charged more strongly is a question the theory has to answer by calculation.

The computational content is a boundary-value problem solved repeatedly. Given the permittivity and temperature of the solvent, the lattice cell size, the surface charge density, and the valency, relative molecular volume and reservoir composition of every ionic species, the theory determines the electrostatic potential and the volume-fraction profile of each species as a function of distance from the wall, and from those profiles the adsorbed amount and the mean distance from the surface at which each species sits. Following those observables as the surface charge changes shows how the layers rearrange.

## Problem

Near a strongly charged surface a mixture of counterions does not simply sort itself by valency. In a lattice description where every species occupies its own number of solvent-sized cells, a saturated adsorbed layer rations volume rather than charge, and the counterions are expected to stack outward in order of decreasing charge per unit volume, measured by alpha = |z|/v, the ratio of valency magnitude to relative molecular volume. That argument needs a saturated layer. At weak charge the same ions sit in a different order, so as the surface charge density is raised some counterions have to exchange places, and the task is to find the charge at which that rearrangement is complete for one electrolyte.

Model the solvent as a cubic lattice with cell edge a. Species i has valency z_i and occupies v_i cells, so its volume fraction is phi_i = a**3 v_i c_i for number density c_i, and the solvent fills the rest, eta = 1 - sum_i phi_i. The free energy per unit volume is the self-energy of the electric field plus the electrostatic energy of the ions, together with the Flory-Huggins mixing term (k_B T / a**3) [sum_i (phi_i / v_i) ln(phi_i) + eta ln(eta)], so that each ion carries the entropy of one particle while the solvent carries it per cell. Minimising with respect to the potential gives Poisson's equation, and minimising with respect to each ion density at fixed potential gives that species' equilibrium condition. The dielectric constant is uniform and equal to the solvent's, the surface charge is smeared uniformly over the plane, and every quantity depends only on the distance x from it.

A planar wall of uniform positive charge density sigma faces an aqueous electrolyte at T = 298.15 K with relative permittivity 78.5 and a = 5.00 angstrom. Far from the wall is a reservoir that carries no net charge, where the potential is taken as zero and the composition is

    species    valency z    relative volume v    reservoir volume fraction
       A          -1              0.5                   2.0e-3
       B          -2              2.0                   1.0e-3
       C          -3              8.0                   5.0e-4
       D          +1              1.0                   fixed by neutrality of the reservoir

At the wall the electric field is set by sigma in the usual way. For each species the adsorbed amount per unit area is the integral over x, from the wall to infinity, of the local number density minus its reservoir value, and the mean adsorption distance is the first moment of that same integrand divided by the adsorbed amount.

Consider positive surface charge densities up to 10 elementary charges per square nanometre. Find the threshold sigma*, defined as the infimum above which the mean adsorption distances of the three counterions increase strictly as alpha decreases for every sigma satisfying sigma* < sigma <= 10 e/nm**2, and report sigma* in elementary charges per square nanometre. In the reasoning, give the reservoir volume fraction of D, every charge density in this range at which two counterions have equal mean adsorption distances together with that common distance, the reduced wall potential e psi / (k_B T) at sigma* together with the relation that fixes it, and the ratio of the valency-weighted sum of the four adsorbed amounts to -sigma at sigma*, naming in one line each the relations used to obtain them. Also retrieve from the source its two boundary conditions for the saturated regime, apply them at sigma*, and state whether A, B and C lie in that regime. These are the scalars that determine the final number.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 7 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

Bjerrum length, cell volume and the reduced surface parameter

Goal
----
Return the Bjerrum length, the lattice cell volume and the reduced surface parameter of a uniformly charged plane facing a solvent of given permittivity and temperature.

```python
def electrostatic_scales(eps_r: float, T_K: float, a_ang: float, sigma_e_per_A2: float) -> "np.ndarray":
    '''Bjerrum length, lattice cell volume and reduced surface parameter of a charged plane.

    Parameters
    ----------
    eps_r : float
        Relative permittivity of the solvent; strictly positive.
    T_K : float
        Absolute temperature in kelvin; strictly positive.
    a_ang : float
        Lattice cell edge in angstrom; strictly positive.
    sigma_e_per_A2 : float
        Surface charge density in elementary charges per square angstrom; any sign, zero allowed.

    Returns
    -------
    result : np.ndarray
        Real array of length 3 holding, in order, the Bjerrum length in angstrom, the cell volume
        in cubic angstrom and the dimensionless reduced surface parameter zeta.

    Raises
    ------
    ValueError
        If eps_r, T_K or a_ang is not strictly positive.
    '''
    return result  # placeholder
```

### Step 2

Reservoir composition closed by electroneutrality

Goal
----
Return the reservoir volume fraction of every ionic species, with the last one fixed by electroneutrality, followed by the reservoir solvent fraction.

```python
def reservoir_composition(z: "np.ndarray", v: "np.ndarray", phi_free: "np.ndarray") -> "np.ndarray":
    '''Reservoir volume fractions of an electroneutral size-asymmetric electrolyte.

    Parameters
    ----------
    z : np.ndarray
        Real array of length N >= 2 holding the signed valency of each species.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the lattice
        cell volume; every entry strictly positive.
    phi_free : np.ndarray
        Real array of length N - 1 holding the reservoir volume fractions of the first N - 1
        species; every entry strictly positive.

    Returns
    -------
    result : np.ndarray
        Real array of length N + 1 holding the N reservoir volume fractions in the input species
        order, the last one fixed by electroneutrality, followed by the reservoir solvent fraction.

    Raises
    ------
    ValueError
        If the lengths are inconsistent, if any relative volume or specified fraction is not
        strictly positive, if the last species has zero valency, if no strictly positive fraction
        of the last species neutralises the reservoir, or if the ion fractions sum to one or more.
    '''
    return result  # placeholder
```

### Step 3

Local composition of a size-asymmetric electrolyte at a given potential

Goal
----
Return the local volume fraction of every ionic species, and the logarithm of the local solvent fraction relative to its reservoir value, at a point of given reduced electrostatic potential.

```python
def local_composition(z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", Psi: float) -> "np.ndarray":
    '''Local composition of a size-asymmetric lattice electrolyte at a given reduced potential.

    Parameters
    ----------
    z : np.ndarray
        Real array of length N holding the signed valency of each species.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the cell
        volume; every entry strictly positive.
    phi_bulk : np.ndarray
        Real array of length N holding the reservoir volume fraction of each species; every entry
        strictly positive, with a sum strictly less than one.
    Psi : float
        Reduced electrostatic potential e psi / (k_B T) at the point, measured from the reservoir;
        any sign, with magnitude up to 60.

    Returns
    -------
    result : np.ndarray
        Real array of length N + 1 holding the N local volume fractions in the input species order,
        followed by ln(eta / eta^b), the natural logarithm of the local solvent fraction divided by
        the reservoir solvent fraction.

    Raises
    ------
    ValueError
        If the three arrays differ in length, if any relative volume or reservoir fraction is not
        strictly positive, or if the reservoir fractions sum to one or more.
    '''
    return result  # placeholder
```

### Step 4

Potential and composition in contact with a charged wall

Goal
----
Return the reduced electrostatic potential at a positively charged planar wall, together with the volume fractions of every ionic species and of the solvent in contact with it.

```python
def wall_state(z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", zeta: float) -> "np.ndarray":
    '''Reduced wall potential and contact composition of a size-asymmetric electrolyte.

    Parameters
    ----------
    z : np.ndarray
        Real array of length N holding the signed valency of each species.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the cell
        volume; every entry strictly positive.
    phi_bulk : np.ndarray
        Real array of length N holding the reservoir volume fractions of an electroneutral
        reservoir; every entry strictly positive, with a sum strictly less than one, and at least
        one species of negative valency.
    zeta : float
        Reduced surface parameter of the positively charged wall; non-negative.

    Returns
    -------
    result : np.ndarray
        Real array of length N + 2 holding the reduced wall potential Psi_s = e psi(0) / (k_B T),
        which is non-negative, then the N volume fractions in contact with the wall in the input
        species order, then the solvent fraction in contact with the wall.

    Raises
    ------
    ValueError
        If zeta is negative.
    '''
    return result  # placeholder
```

### Step 5

Adsorbed amounts and mean adsorption distances

Goal
----
Return the adsorbed amount per unit area and the mean adsorption distance of every ionic species next to a positively charged planar wall, together with the charge-balance ratio of the layer and the reduced wall potential.

```python
def adsorption_moments(eps_r: float, T_K: float, a_ang: float, sigma_e_per_A2: float, z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray") -> "np.ndarray":
    '''Adsorbed amounts and mean adsorption distances of a size-asymmetric electrolyte at a charged wall.

    Parameters
    ----------
    eps_r : float
        Relative permittivity of the solvent; strictly positive.
    T_K : float
        Absolute temperature in kelvin; strictly positive.
    a_ang : float
        Lattice cell edge in angstrom; strictly positive.
    sigma_e_per_A2 : float
        Surface charge density of the wall in elementary charges per square angstrom; strictly
        positive.
    z : np.ndarray
        Real array of length N holding the signed valency of each species.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the cell volume;
        every entry strictly positive.
    phi_bulk : np.ndarray
        Real array of length N holding the reservoir volume fractions of an electroneutral
        reservoir; every entry strictly positive, with a sum strictly less than one, and at least
        one species of negative valency.

    Returns
    -------
    result : np.ndarray
        Real array of length 2 N + 2 holding the N adsorbed amounts in inverse square angstrom,
        then the N mean adsorption distances in angstrom, both in the input species order, then the
        ratio of sum_i z_i Gamma_i to -sigma / e, then the reduced wall potential e psi(0) / (k_B T).

    Raises
    ------
    ValueError
        If sigma_e_per_A2 is not strictly positive, or if eps_r, T_K or a_ang is not strictly
        positive.
    '''
    return result  # placeholder
```

### Step 6

Surface charge at which two counterions share a mean adsorption distance

Goal
----
Return the surface charge density, inside a given bracket, at which two ionic species have equal mean adsorption distances at a positively charged wall.

```python
def exchange_charge(eps_r: float, T_K: float, a_ang: float, z: "np.ndarray", v: "np.ndarray", phi_bulk: "np.ndarray", i: int, j: int, sigma_lo: float, sigma_hi: float) -> float:
    '''Surface charge density at which two species have equal mean adsorption distances.

    Parameters
    ----------
    eps_r : float
        Relative permittivity of the solvent; strictly positive.
    T_K : float
        Absolute temperature in kelvin; strictly positive.
    a_ang : float
        Lattice cell edge in angstrom; strictly positive.
    z : np.ndarray
        Real array of length N holding the signed valency of each species.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the cell volume;
        every entry strictly positive.
    phi_bulk : np.ndarray
        Real array of length N holding the reservoir volume fractions of an electroneutral
        reservoir; every entry strictly positive, with a sum strictly less than one.
    i : int
        Zero-based index of the first species.
    j : int
        Zero-based index of the second species, different from i.
    sigma_lo : float
        Lower end of the bracket in elementary charges per square angstrom; strictly positive.
    sigma_hi : float
        Upper end of the bracket in elementary charges per square angstrom; larger than sigma_lo.

    Returns
    -------
    sigma_x : float
        Surface charge density in elementary charges per square angstrom, between sigma_lo and
        sigma_hi, at which the mean adsorption distances of species i and j are equal.

    Raises
    ------
    ValueError
        If i and j are equal or out of range, if the bracket is not 0 < sigma_lo < sigma_hi, or if
        the difference of the two mean adsorption distances has the same sign at both ends of the
        bracket.
    '''
    return sigma_x  # placeholder
```

### Step 7

Onset of stratification ordered by charge per volume

Goal
----
Return the threshold surface charge density, defined as the infimum above which the counterions' mean adsorption distances stay strictly ordered by charge per unit volume up to the top of a given charge window.

```python
def stratification_onset(eps_r: float, T_K: float, a_ang: float, z: "np.ndarray", v: "np.ndarray", phi_free: "np.ndarray", sigma_lo_e_per_nm2: float, sigma_hi_e_per_nm2: float) -> float:
    '''Threshold surface charge density for counterion stratification ordered by charge per volume.

    Parameters
    ----------
    eps_r : float
        Relative permittivity of the solvent; strictly positive.
    T_K : float
        Absolute temperature in kelvin; strictly positive.
    a_ang : float
        Lattice cell edge in angstrom; strictly positive.
    z : np.ndarray
        Real array of length N holding the signed valency of each species; at least two species
        have negative valency, and the last species closes electroneutrality.
    v : np.ndarray
        Real array of length N holding each species' molecular volume divided by the cell volume;
        every entry strictly positive.
    phi_free : np.ndarray
        Real array of length N - 1 holding the reservoir volume fractions of the first N - 1
        species; every entry strictly positive.
    sigma_lo_e_per_nm2 : float
        Lower end of the charge window in elementary charges per square nanometre; strictly
        positive.
    sigma_hi_e_per_nm2 : float
        Upper end of the charge window in elementary charges per square nanometre; larger than
        sigma_lo_e_per_nm2.

    Returns
    -------
    sigma_star : float
        The threshold (infimum) in elementary charges per square nanometre such that the strict
        order holds for every sigma satisfying sigma_star < sigma <= sigma_hi_e_per_nm2. If the
        strict order holds across the whole closed window, returns sigma_lo_e_per_nm2.

    Raises
    ------
    ValueError
        If fewer than two species have negative valency, if two counterions share the same alpha,
        if the window is not 0 < sigma_lo < sigma_hi, if the counterions are not ordered by
        decreasing alpha at sigma_hi, or if the reservoir cannot be closed as in step 02.
    '''
    return sigma_star  # placeholder
```

# Material_Science-Molecular_Modeling-18

## Background

## Associating fluids in nanoconfinement

A fluid held between two solid walls separated by a few molecular diameters is not a small piece
of bulk fluid. The walls exclude molecular centres from a layer of their own thickness, impose a
dispersion field that is strongly attractive a little way out and steeply repulsive closer in,
and break the symmetry of the environment around every molecule. The density profile that results
oscillates with a period close to one molecular diameter and can reach several times the
reservoir density in the contact layer. Predicting sorption, selectivity and transport in porous
carbons, zeolites and metal organic frameworks all rest on getting that profile and the
thermodynamics that goes with it right.

Classical density functional theory treats the equilibrium profile as the minimiser of a grand
potential functional. The ideal part is known exactly; everything difficult sits in the residual
intrinsic free energy, which for a molecular fluid is customarily decoupled into a hard sphere
reference, dispersion between segments, chain connectivity, and association between specific
sites. For the hard sphere reference the accepted route replaces the local density by a small
family of weighted densities, each a convolution of the profile with a weight function carrying
one geometric measure of the sphere, so that the free energy density becomes an algebraic
function of those measures. Two of the measures are vectors that vanish identically in a uniform
fluid and grow wherever the profile is strongly layered, and they are what let the theory
reproduce packing against a wall rather than merely averaging over it.

## Hydrogen bonding as a thermodynamic perturbation

Association is handled by treating a hydrogen bond as a short ranged, strongly attractive, highly
directional interaction between designated sites on the molecular surface. In a first order
perturbation treatment the free energy of association depends on the fraction of each type of
site that remains unbonded, and that fraction obeys a mass action condition in which the
partners available to a given site are counted by the density of surrounding molecules weighted
by an association strength. The strength itself separates into a Boltzmann factor for the well
depth, a purely geometric bonding volume set by where the sites sit and how far the well reaches,
and the contact value of the pair correlation function of the reference fluid, which measures how
often two molecules are actually adjacent.

The step that matters in confinement is what plays the role of "the density of surrounding
molecules". Carrying the uniform fluid expression across unchanged puts the local density there,
which is wrong for two reasons at once: it counts neighbours that the wall has removed, and it
ignores that the neighbours a molecule does have are all on one side. A treatment that is to be
trusted in a pore has to repair both, and the contact value of the pair correlation inherits the
same difficulty, since two molecules on the same side of a wall are less likely to be in contact
than a bulk estimate at the same density suggests. How that repair is actually carried out, in the
weighted density formalism that the source adopts for it and couples to its own equation of
state, is what this task is built on.

## Why the size of the sphere is not a free choice

None of this is meaningful until the reference sphere has a size. For a soft repulsive potential
the accepted choice is temperature dependent and follows from an integral of the Mayer factor of
the potential across the repulsive core, so a steeply repulsive segment behaves almost like a
hard sphere of its nominal diameter while a softer one shrinks noticeably as temperature rises.
That single length then propagates everywhere: it sets the ranges of all the weight functions,
the packing measures, the contact value of the pair correlation, and the bonding volume. Because
the bonding volume depends on it through a logarithm and a cubic polynomial, an error in the
sphere size does not stay proportionate, it changes the predicted degree of hydrogen bonding out
of all recognition.

## Problem

Water confined in a graphitic slit only a few molecular diameters across hydrogen bonds far less freely than bulk water, because the walls both layer the fluid and strip neighbours from one side of every molecule; classical density functional treatments of associating fluids capture the packing of such a fluid well, but the association term is customarily carried over from the uniform fluid unchanged, so the predicted bonding state still rests on a local density that no longer describes the local environment a molecule actually sees. Recasting that term so it reads the environment from the same geometry the packing theory already resolves is what makes the hydrogen bonding state of a confined fluid predictable, and it shifts the answer by far more than the tolerance to which the result is reported. Given the molecular parameters, the pore geometry and the reservoir state, the calculation returns the fraction of association sites that remain unbonded at every height in the pore.

Represent the water molecule as one spherical segment interacting through a Mie potential of diameter 3.161 angstrom and well depth 488.75 kelvin with attractive exponent 6 and repulsive exponent 52.367, and fix its effective hard sphere size from the repulsive branch of that potential at the working temperature. Give the segment four association sites, two donors and two acceptors, each placed 0.4 segment diameters from the centre and interacting through a square well of depth 1210 kelvin and range 0.5834 segment diameters, with bonding allowed only between a donor and an acceptor and each site able to carry at most one bond. Model each graphite wall as a stack of structureless planes of solid density 0.114 per cubic angstrom, interlayer spacing 3.35 angstrom, site diameter 3.4 angstrom and well depth 28 kelvin, summed analytically over the planes with a coefficient of 0.61 in the term that accounts for the planes beyond the first, take the solid fluid size and energy parameters from the arithmetic and geometric combining rules, and let the fields of the two walls add. Work at 425 kelvin in a pore 14 angstrom from wall to wall, in contact with a reservoir of number density 0.010 per cubic angstrom.

Relax the hard sphere reference of the confined fluid to the density profile that satisfies equilibrium with that reservoir, taking the reservoir residual chemical potential from the same free energy expression so that pore and reservoir sit on a common thermodynamic footing, and using a uniform grid of 1401 points spanning the pore with both walls included; the fixed point needs damping to stay below close packing on the approach, and should be iterated until the largest change in the undamped update falls below 1e-12 per cubic angstrom. Evaluate the association closure on that converged structure and report the mean fraction of association sites left unbonded, averaged over the pore with the density profile as the weight and the trapezoidal rule on the same grid, as a dimensionless number between zero and one.

Report alongside that number every quantity needed to audit the result: the effective hard sphere radius and diameter; the minimum of the reduced wall field and its position; the reservoir residual chemical potential and packing fraction; the peak density of the converged profile and both peak positions; the largest packing measure; the integrated pore uptake; the minimum and maximum of the anisotropy factor, contact pair correlation and association strength; the bonding volume and square-well Mayer factor; and the minimum free-site fraction, its position and the midplane free-site fraction. For every range or minimum described as being across the fluid, use the occupied-fluid grid points satisfying rho(z) > 1e-8 per cubic angstrom. Also state the relations used for the inhomogeneous contact pair correlation, the bonding volume and the four-site mass-action condition.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 11 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_barker_henderson_radius

Goal
----
Effective hard-sphere radius of a Mie segment at a given temperature.

```python
def barker_henderson_radius(sigma: float, epsilon: float, lambda_a: float, lambda_r: float, temperature: float) -> float:
    '''Effective hard-sphere radius of a Mie segment.

    Parameters
    ----------
    sigma : float
        Mie segment diameter in angstrom.
    epsilon : float
        Mie potential well depth divided by the Boltzmann constant, in kelvin.
    lambda_a : float
        Attractive exponent of the Mie potential.
    lambda_r : float
        Repulsive exponent of the Mie potential.
    temperature : float
        Absolute temperature in kelvin.

    Returns
    -------
    float
        Effective hard-sphere radius in angstrom.

    Raises
    ------
    ValueError
        If sigma, epsilon or temperature is not positive, or if lambda_r <= lambda_a.
    '''
    return None  # placeholder
```

### Step 2

02_steele_slit_potential

Goal
----
Reduced external field of a structureless carbon slit pore.

```python
def steele_slit_potential(z: "np.ndarray", pore_width: float, sigma_f: float, epsilon_f: float, sigma_s: float, epsilon_s: float, rho_s: float, delta_s: float, alpha: float, temperature: float) -> "np.ndarray":
    '''Reduced external potential of a carbon slit pore on its grid.

    Parameters
    ----------
    z : numpy.ndarray
        Distances from the first wall in angstrom, spanning the pore.
    pore_width : float
        Wall to wall separation in angstrom.
    sigma_f, epsilon_f : float
        Fluid segment diameter in angstrom and well depth in kelvin.
    sigma_s, epsilon_s : float
        Solid site diameter in angstrom and well depth in kelvin.
    rho_s : float
        Solid number density in angstrom^-3.
    delta_s : float
        Interlayer spacing of the solid in angstrom.
    alpha : float
        Adjustable coefficient of the third term.
    temperature : float
        Absolute temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        Reduced external potential on the grid, positive infinity where a segment centre is
        excluded.

    Raises
    ------
    ValueError
        If pore_width or temperature is not positive.
    '''
    return None  # placeholder
```

### Step 3

03_fmt_weighted_densities

Goal
----
Geometric weighted densities of the confined fluid on the pore grid.

```python
def fmt_weighted_densities(rho: "np.ndarray", radius: float, dz: float) -> "np.ndarray":
    '''Six fundamental measure weighted densities of a planar profile.

    Parameters
    ----------
    rho : numpy.ndarray
        Segment number density on the grid in angstrom^-3.
    radius : float
        Effective hard-sphere radius in angstrom.
    dz : float
        Uniform grid spacing in angstrom.

    Returns
    -------
    numpy.ndarray
        Array of shape (6, rho.size) holding n0, n1, n2, n3, nV1 and nV2 in that order.

    Raises
    ------
    ValueError
        If radius or dz is not positive.
    '''
    return None  # placeholder
```

### Step 4

04_white_bear_partials

Goal
----
Sensitivity of the hard-sphere free-energy density to each weighted density.

```python
def white_bear_partials(weighted: "np.ndarray") -> "np.ndarray":
    '''Partial derivatives of the hard-sphere free-energy density.

    Parameters
    ----------
    weighted : numpy.ndarray
        Array of shape (6, N) holding n0, n1, n2, n3, nV1 and nV2 on the grid.

    Returns
    -------
    numpy.ndarray
        Array of shape (6, N) holding the partial derivative of the reduced free-energy density
        with respect to each of the six weighted densities, in the same order.

    Raises
    ------
    ValueError
        If weighted does not have six rows.
    '''
    return None  # placeholder
```

### Step 5

05_hard_sphere_functional_derivative

Goal
----
One-body direct correlation of the hard-sphere reference along the pore axis.

```python
def hard_sphere_functional_derivative(rho: "np.ndarray", radius: float, dz: float) -> "np.ndarray":
    '''Functional derivative of the reduced hard-sphere free energy.

    Parameters
    ----------
    rho : numpy.ndarray
        Segment number density on the grid in angstrom^-3.
    radius : float
        Effective hard-sphere radius in angstrom.
    dz : float
        Uniform grid spacing in angstrom.

    Returns
    -------
    numpy.ndarray
        Dimensionless functional derivative on the grid, same length as rho.

    Raises
    ------
    ValueError
        If radius or dz is not positive.
    '''
    return None  # placeholder
```

### Step 6

06_bulk_residual_chemical_potential

Goal
----
Residual chemical potential of the reservoir the pore is in contact with.

```python
def bulk_residual_chemical_potential(rho_bulk: float, radius: float) -> float:
    '''Reduced residual chemical potential of the uniform hard-sphere reference.

    Parameters
    ----------
    rho_bulk : float
        Reservoir number density in angstrom^-3.
    radius : float
        Effective hard-sphere radius in angstrom.

    Returns
    -------
    float
        Dimensionless residual chemical potential.

    Raises
    ------
    ValueError
        If rho_bulk is negative or radius is not positive.
    '''
    return None  # placeholder
```

### Step 7

07_equilibrium_density_profile

Goal
----
Equilibrium density profile of the reference fluid in the slit.

```python
def equilibrium_density_profile(beta_v_ext: "np.ndarray", rho_bulk: float, radius: float, dz: float) -> "np.ndarray":
    '''Converged density profile of the hard-sphere reference in an external field.

    Parameters
    ----------
    beta_v_ext : numpy.ndarray
        Reduced external potential on the grid, positive infinity where excluded.
    rho_bulk : float
        Reservoir number density in angstrom^-3.
    radius : float
        Effective hard-sphere radius in angstrom.
    dz : float
        Uniform grid spacing in angstrom.

    Returns
    -------
    numpy.ndarray
        Converged density profile on the grid in angstrom^-3.

    Raises
    ------
    ValueError
        If rho_bulk is not positive, or radius or dz is not positive.
    '''
    return None  # placeholder
```

### Step 8

08_contact_pair_correlation

Goal
----
Contact pair correlation of the reference fluid, corrected for local anisotropy.

```python
def contact_pair_correlation(weighted: "np.ndarray", hs_diameter: float) -> "np.ndarray":
    '''Local anisotropy factor and contact pair correlation of the hard-sphere reference.

    Parameters
    ----------
    weighted : numpy.ndarray
        Array of shape (6, N) holding n0, n1, n2, n3, nV1 and nV2 on the grid.
    hs_diameter : float
        Effective hard-sphere diameter in angstrom.

    Returns
    -------
    numpy.ndarray
        Array of shape (2, N); first row the dimensionless anisotropy factor, second row the
        contact pair correlation function.

    Raises
    ------
    ValueError
        If weighted does not have six rows or hs_diameter is not positive.
    '''
    return None  # placeholder
```

### Step 9

09_association_strength

Goal
----
Strength of an association interaction between two sites on neighbouring segments.

```python
def association_strength(g_contact: "np.ndarray", hs_diameter: float, sigma: float, epsilon_hb: float, r_site: float, r_cut: float, temperature: float) -> "np.ndarray":
    '''Association strength between two complementary sites.

    Parameters
    ----------
    g_contact : numpy.ndarray
        Contact pair correlation of the reference fluid on the grid.
    hs_diameter : float
        Effective hard-sphere diameter in angstrom.
    sigma : float
        Mie segment diameter in angstrom.
    epsilon_hb : float
        Association well depth divided by the Boltzmann constant, in kelvin.
    r_site : float
        Distance from the segment centre to an association site in angstrom.
    r_cut : float
        Range of the square-well site-site interaction in angstrom.
    temperature : float
        Absolute temperature in kelvin.

    Returns
    -------
    numpy.ndarray
        Association strength on the grid in angstrom^3.

    Raises
    ------
    ValueError
        If hs_diameter, sigma, r_site, r_cut or temperature is not positive.
    '''
    return None  # placeholder
```

### Step 10

10_nonbonded_site_fraction

Goal
----
Fraction of association sites left unbonded at each position in the pore.

```python
def nonbonded_site_fraction(n0: "np.ndarray", zeta: "np.ndarray", delta: "np.ndarray") -> "np.ndarray":
    '''Fraction of association sites that are not hydrogen bonded.

    Parameters
    ----------
    n0 : numpy.ndarray
        Zeroth geometric weighted density on the grid in angstrom^-3.
    zeta : numpy.ndarray
        Dimensionless local anisotropy factor on the grid.
    delta : numpy.ndarray
        Association strength on the grid in angstrom^3.

    Returns
    -------
    numpy.ndarray
        Fraction of non-bonded sites on the grid, between zero and one.

    Raises
    ------
    ValueError
        If the three inputs do not have the same shape.
    '''
    return None  # placeholder
```

### Step 11

11_confined_nonbonded_fraction

Goal
----
Average hydrogen-bonding state of water held in a carbon slit pore.

```python
def confined_nonbonded_fraction(pore_width: float, rho_bulk: float, temperature: float, n_grid: int) -> float:
    '''Profile-weighted mean fraction of non-bonded association sites in the pore.

    Parameters
    ----------
    pore_width : float
        Wall to wall separation in angstrom.
    rho_bulk : float
        Reservoir number density in angstrom^-3.
    temperature : float
        Absolute temperature in kelvin.
    n_grid : int
        Number of equally spaced grid points spanning the pore, endpoints included.

    Returns
    -------
    float
        Profile-weighted mean fraction of non-bonded sites, between zero and one.

    Raises
    ------
    ValueError
        If pore_width, rho_bulk or temperature is not positive, or n_grid is below two.
    '''
    return None  # placeholder
```

# Chemistry-Computational_Chemistry-69

## Background

## Why the air-water interface is a special place for redox chemistry

Aqueous microdroplets accelerate an unusually broad set of oxidations and reductions,
often by many orders of magnitude relative to the same solution in bulk, with no light,
no catalyst and no applied field. The breadth is what makes the effect hard to explain.
A mechanism tuned to one substrate would not account for a scope that runs from
transition-metal complexes through organic dyes to dissolved oxygen and ozone, so the
explanation has to live in the solvent rather than in the chemistry of any one partner.

The classical account treats hydroxide as the electron source and splits the process in
two: ionise the hydroxide, then let the liberated electron find an acceptor. That route
is energetically hopeless. The vertical ionisation energy of hydroxide in bulk water is
around 215 kcal/mol, and even extrapolating to a completely unsolvated ion leaves a cost
above 40 kcal/mol, far beyond anything thermally accessible. Invoking a free electron
also invites a second problem, because an electron at rest in vacuum is such a high
energy species that it would reduce essentially anything in the solution indiscriminately.

## Concerted transfer and the role of partial solvation

The alternative is the picture Marcus introduced for condensed-phase electron transfer:
the donor and the acceptor exchange the electron directly, at a nuclear configuration
where the two diabatic surfaces cross, so no free electron is ever created. The energetic
bookkeeping then involves only the two redox couples referenced to a common electrode,
and the free-electron term drops out.

What the interface contributes is a donor that is only partly solvated. Water at a free
surface has a density profile that decays smoothly over roughly a nanometre, molecules
there are missing hydrogen-bond partners, and a substantial population of
under-coordinated hydroxide exists that has no counterpart in bulk. This matters because
the two members of the redox couple are stabilised by water to wildly different degrees:
hydroxide is a small, densely charged anion with an absolute solvation free energy near
-106 kcal/mol, while the neutral hydroxyl radical it becomes is barely stabilised at all.
Removing solvent therefore destabilises the reactant far more than the product, and the
reaction free energy falls by an amount set by the difference of the two solvation free
energies. That difference is over 100 kcal/mol, which is why a modest degree of
dehydration can reverse the thermodynamics of a reaction that is strongly uphill in bulk.

Simulation supports the picture quantitatively. Alongside the bulk photoemission band,
interfacial hydroxide shows a second, much lower ionisation band that comes almost
entirely from under-coordinated configurations near the surface. Reading that band as a
partial-solvation effect requires a thermodynamic cycle that connects bulk, interface and
dry limit, and the cycle has to account for the fact that a vertical process pays a
solvent reorganisation penalty that an adiabatic one does not.

## Precursor complexes and screening that varies with depth

Marcus theory is written for a precursor complex, not for reactants at infinite
separation, so a driving force taken from tabulated potentials has to be corrected by
the work of bringing the two reactants together and the work of pulling the two products
apart. In bulk water that correction is often small enough to absorb into a constant,
because the medium screens uniformly and the two work terms partly cancel. Neither
excuse survives at an interface. When one partner is charged and the other is not, only
one work term is left, and it does not cancel against anything.

Screening is the harder half. A continuum treatment needs a permittivity for the
material between the two ions, and at an interface that material is not one substance:
it runs from liquid water to near-vacuum along the line joining them. Thin layers of
dielectric stacked along a field line behave as capacitors in series, so what averages
along the path is the reciprocal permittivity rather than the permittivity itself. The
resulting effective dielectric constant is dominated by whichever stretch of the path is
least polarisable, which means a short passage through low-density water costs far more
screening than its length alone would suggest. Because the donor's position is exactly
what the dehydration coordinate parametrises, this screening constant is a function of
that coordinate, and the work term inherits a dependence on it that is not linear and not
monotonic.

## Reorganisation energy, tunnelling and the shape of the average

Two further ingredients decide the rate. The first is the reorganisation energy. For a
cross reaction between two different couples the Marcus cross-relation takes it as the
mean of the two self-exchange values, which is exact for coordinates whose free-energy
surfaces have the same curvature before and after the transfer. The donor's value is not
a fixed number: a self-exchange reorganisation energy divides into an inner-sphere part
from intramolecular rearrangement, which survives dehydration, and an outer-sphere part
from solvent repolarisation, which does not. Because both the driving force and the
reorganisation energy then depend on how dehydrated the donor is, the resonance between
reactant and product states moves across the interface, and a single reaction can occupy
the normal, activationless and inverted regimes at different points of one interface.

The acceptor adds a structural complication that the cross-relation cannot absorb.
Hexaamminecobalt(III) is low-spin d6 and its cobalt(II) partner is high-spin d7, so the
six cobalt-nitrogen bonds lengthen by nearly two tenths of an angstrom on reduction and
the complex becomes much softer along its breathing coordinate. With different force
constants in the two oxidation states the two free-energy surfaces no longer have the
same curvature, the energy the breathing mode contributes to the donor-acceptor gap is no
longer linear in the displacement, and the vertical energy the reduced complex pays at the
oxidised geometry differs markedly from the one the oxidised complex pays at the reduced
geometry. The textbook treatment replaces the two force constants by a single reduced
value. That is adequate close to zero driving force, but here the whole rate comes from
environments where the reaction is strongly downhill, and there the approximation moves
the resonance by several kcal/mol. The consistent classical treatment keeps both surfaces
and averages the resonance condition over the thermal distribution of the breathing
coordinate in the state the reaction starts from.

The second is the electronic coupling. Non-adiabatic transfer rates go as the square of
the donor-acceptor coupling, which decays exponentially with separation at a rate set by
the intervening medium. Liquid water is a good superexchange bridge and empty space is a
poor one, so a path that begins in bulk liquid and ends in the vapour tail has no single
decay constant. Dehydration therefore cuts both ways: it buys driving force, but it moves
the donor outward, lengthening the tunnelling path and replacing water bridge with vacuum.

Finally, none of these quantities can be evaluated at a single representative hydration
state. The interface is a continuum of environments, the rate varies steeply across it,
and the observable is an average weighted by how much of the population sits at each
level of dehydration. Fixing that weight is a structural question about where the donor
sits, and converting a statement about position into a statement about hydration is a
change of variables that has to be done properly.

## Problem

Redox reactions that are hopelessly uphill in bulk water run readily on the surface of aqueous
microdroplets. One explanation is that hydroxide hands an electron directly to an acceptor in a
single concerted step, and that all the interface has to supply is a hydroxide ion that has lost
part of its hydration shell: water stabilises hydroxide far more than the hydroxyl radical it
becomes, so stripping that shell moves the reaction free energy by an amount comparable to a
chemical bond, and a reaction that is forbidden in bulk can become activationless partway across the
interface. Starting from the hydration thermodynamics of hydroxide, the structure of the air-water
interface and the structural and vibrational data of the acceptor, your task is to work out how fast
one such reaction runs once the interface is treated as the continuum of hydration environments it
really is, and to report a single rate constant.

Characterise the donor's environment by a dehydration level theta, the fraction of its bulk solvent
stabilisation that has been removed (theta = 0 is bulk-like, theta = 1 gas-like), and let every
solvent-dependent free energy vary linearly with theta. The water density across the interface is
rho(Z)/rho_bulk = 0.5*(1 + tanh(-(Z - Z_G)/delta)), with Z along the interface normal increasing
towards the vapour, Z_G = 0.84 angstrom and delta = 1.5 angstrom, and the dehydration level at a
point is the local fractional depletion of water density there. The donor is hydroxide, with
absolute solvation free energies of -106.4 kcal/mol for the anion and -3.9 kcal/mol for the radical,
a bulk vertical ionisation energy of 215.0 kcal/mol, a bulk solvent reorganisation energy of the
radical of 69.0 kcal/mol, and a self-exchange reorganisation energy made of a 2.0 kcal/mol internal
part that survives dehydration and a 67.0 kcal/mol outer-sphere part at full hydration that is
removed in proportion to the solvent removed. The OH/OH- couple has a standard potential of 1.90 V
vs SHE, and the SHE lies 4.44 V below a free electron at rest in vacuum. Take the bulk reaction free
energy from the two standard potentials and let dehydration shift it by the lost difference in
solvation between anion and radical; obtain the rate at which the vertical ionisation energy falls
with dehydration, and its dry-limit value, from the cycle these numbers define rather than from any
tabulated gas-phase value. The acceptor is hexaamminecobalt(III), with a standard potential of
+0.108 V vs SHE, fully solvated at a fixed depth of -6.50 angstrom, with an outer-sphere
self-exchange reorganisation energy of 27.6 kcal/mol. Combine the reorganisations whose curvature is
the same in both states, the donor's whole self-exchange value and the acceptor's outer sphere,
through the Marcus cross-relation. The cobalt centre is different: on reduction its six Co-N bonds
lengthen together from 1.936 to 2.114 angstrom along the totally symmetric breathing mode, in which
the metal does not move, and that mode vibrates at 494 cm^-1 in the cobalt(III) complex and at 357
cm^-1 in the cobalt(II) complex, with an ammine mass of 17.031 u. Treat the complex as harmonic in
that coordinate in each oxidation state, with the force constant its own frequency implies and
without replacing the two by any single effective force constant, and treat the transfer as
non-adiabatic, with Fermi's golden rule and every nuclear coordinate classical.

Because the transfer happens inside the precursor complex rather than between separated reactants,
correct the reaction free energy for the electrostatic work of assembling the reactant pair, a
Coulomb attraction between charges -1 and +3 at their separation (the products' work vanishes
because one product is neutral); the population is fixed by the slab below, so the work enters only
through that free energy. Screen the attraction by the intervening material treated as thin slabs in
series, so that the reciprocal of the effective dielectric constant is the mean of the reciprocal
local permittivity along the line joining the ions, with the local permittivity running linearly
from 1 in vacuum to 78.4 in bulk water in proportion to the local water density. For the electronic
coupling take H_0 = 50 cm^-1, extrapolated to zero separation, and |H_DA|^2 = H_0^2 * exp(-integral
of beta(Z) dZ) along the straight path from acceptor to donor, the local decay constant beta being
quoted for the squared coupling and interpolating between 1.6 angstrom^-1 in bulk water and 2.9
angstrom^-1 in vacuum in proportion to the local water density. Interfacial hydroxide is distributed
uniformly in depth over a slab from -3.00 to +3.00 angstrom, and the observable rate constant is the
average of the fixed-theta rate over that population. Work at 298.15 K with F = 23.060548 kcal
mol^-1 V^-1, R = 1.987204259e-3 kcal mol^-1 K^-1, hbar = 1.054571817e-34 J s, e^2/(4*pi*eps0) =
332.0637 kcal mol^-1 angstrom, 1 cm^-1 = 1.98644586e-23 J, 1 kcal/mol = 6.9476954571e-21 J, 1 u =
1.66053906660e-27 kg and c = 2.99792458e10 cm s^-1. Report the population-averaged interfacial rate
constant in s^-1 as the final answer, converged to at least eight significant figures. In the
reasoning give, each as a single value with its unit: the bulk reaction free energy, kept distinct
from the free energy of reducing the acceptor with an electron from rest in vacuum; the slope and
dry-limit value of the vertical ionisation energy; the effective dielectric constant and the work
term at theta = 0.73; the Co-N force constants of the two oxidation states; the nuclear factor,
meaning the thermally averaged density per unit energy of configurations in which reactant and
product are degenerate, at theta = 0.78 in (kcal/mol)^-1; the squared coupling at theta = 0.73; the
dehydration level at which the fixed-theta rate peaks, and the rate-weighted mean dehydration level;
and, treating the transfer as pseudo-first-order in the acceptor, the half-life and the fraction of
acceptor reduced after contact times of 2.00 and 10.0 microseconds.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 12 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_acceptor_energetics

Goal
----
Step 01 - Vacuum-referenced and hydroxide-referenced acceptor energetics.

```python
import numpy as np
import numpy.typing as npt


def acceptor_energetics(E0_acceptor: float) -> np.ndarray:
    '''Vacuum-referenced and hydroxide-referenced free energies of an acceptor.

    Parameters
    ----------
    E0_acceptor : float
        Standard one-electron reduction potential of the acceptor couple,
        in volts versus the standard hydrogen electrode.

    Returns
    -------
    energetics : numpy.ndarray
        Array of shape (2,) in kcal/mol. Element 0 is the free energy change
        for reducing the acceptor with an electron taken from rest in vacuum.
        Element 1 is the standard free energy change of the bulk-water
        reaction in which hydroxide is the electron donor and the hydroxyl
        radical is the oxidised product.

    Raises
    ------
    ValueError
        If E0_acceptor is not finite.
    '''
    return energetics
```

### Step 2

02_slab_dehydration_limits

Goal
----
Step 02 - Mapping interfacial depth onto dehydration level.

```python
import numpy as np
import numpy.typing as npt


def slab_dehydration_limits(Z_lo: float, Z_hi: float, theta_probe: float) -> np.ndarray:
    '''Depth-to-dehydration map evaluated on a slab and inverted at one point.

    Parameters
    ----------
    Z_lo : float
        Depth of the liquid-side face of the interfacial slab, angstrom.
    Z_hi : float
        Depth of the vapour-side face of the interfacial slab, angstrom,
        strictly greater than Z_lo.
    theta_probe : float
        A dehydration level, strictly inside (0, 1), to be mapped back to a depth.

    Returns
    -------
    limits : numpy.ndarray
        Array of shape (3,). Element 0 is the dehydration level at Z_lo,
        element 1 is the dehydration level at Z_hi, both dimensionless.
        Element 2 is the depth in angstrom at which the dehydration level
        equals theta_probe.

    Raises
    ------
    ValueError
        If any argument is not finite, if Z_hi is not strictly greater than
        Z_lo, or if theta_probe does not lie strictly inside (0, 1).
    '''
    return limits
```

### Step 3

03_effective_dielectric

Goal
----
Step 03 - Effective dielectric constant screening a pair across the interface.

```python
import numpy as np
import numpy.typing as npt


def effective_dielectric(theta: npt.ArrayLike, Z_acceptor: float) -> np.ndarray:
    '''Path-averaged dielectric constant screening the donor-acceptor pair.

    Parameters
    ----------
    theta : array_like
        Dehydration level or levels of the donor, each strictly inside (0, 1).
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal,
        angstrom, strictly below the donor.

    Returns
    -------
    eps_eff : numpy.ndarray
        Dimensionless effective dielectric constant screening the pair at each
        dehydration level, same shape as theta.

    Raises
    ------
    ValueError
        If Z_acceptor is not finite, if theta is not finite, if any element of
        theta does not lie strictly inside (0, 1), or if the donor does not lie
        strictly above the acceptor at every requested dehydration level.
    '''
    return eps_eff
```

### Step 4

04_ion_pair_work

Goal
----
Step 04 - Electrostatic work of assembling the reactant ion pair.

```python
import numpy as np
import numpy.typing as npt


def ion_pair_work(theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    '''Screened Coulomb work of bringing the two reactants into contact.

    Parameters
    ----------
    theta : array_like
        Dehydration level or levels of the donor, each strictly inside (0, 1).
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal,
        angstrom, strictly below the donor.
    charge_product : float
        Product of the formal charges of the two reactants, in units of the
        elementary charge, signed.

    Returns
    -------
    work : numpy.ndarray
        Electrostatic work of assembling the reactant pair at each dehydration
        level, kcal/mol, same shape as theta. Negative for oppositely charged
        reactants.

    Raises
    ------
    ValueError
        If any argument is not finite, if any element of theta does not lie
        strictly inside (0, 1), or if the donor does not lie strictly above the
        acceptor at every requested dehydration level.
    '''
    return work
```

### Step 5

05_interfacial_driving_force

Goal
----
Step 05 - Driving force of the concerted transfer inside the precursor complex.

```python
import numpy as np
import numpy.typing as npt


def interfacial_driving_force(E0_acceptor: float, dG_sol_anion: float, dG_sol_radical: float, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    '''Work-corrected reaction free energy versus dehydration level.

    Parameters
    ----------
    E0_acceptor : float
        Standard one-electron reduction potential of the acceptor, V vs SHE.
    dG_sol_anion : float
        Absolute solvation free energy of the donor anion, kcal/mol, signed.
    dG_sol_radical : float
        Absolute solvation free energy of the neutral radical, kcal/mol, signed.
    theta : array_like
        Dehydration level or levels, each strictly inside (0, 1).
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal, angstrom.
    charge_product : float
        Product of the formal charges of the two reactants, signed.

    Returns
    -------
    dG : numpy.ndarray
        Reaction free energy of the concerted electron transfer inside the
        precursor complex at each dehydration level, kcal/mol, same shape as
        theta.

    Raises
    ------
    ValueError
        If any argument is not finite, if any element of theta does not lie
        strictly inside (0, 1), or if the donor does not lie strictly above the
        acceptor at every requested dehydration level.
    '''
    return dG
```

### Step 6

06_solvent_reorganisation_energy

Goal
----
Step 06 - Symmetric part of the cross-reaction reorganisation energy.

```python
import numpy as np
import numpy.typing as npt


def solvent_reorganisation_energy(lam_acceptor_outer: float, theta: npt.ArrayLike) -> np.ndarray:
    '''Symmetric (equal-curvature) part of the cross-reaction reorganisation energy.

    Parameters
    ----------
    lam_acceptor_outer : float
        Outer-sphere self-exchange reorganisation energy of the acceptor couple,
        kcal/mol, positive and independent of the donor's hydration.
    theta : array_like
        Dehydration level or levels of the donor, each in [0, 1].

    Returns
    -------
    lam_sym : numpy.ndarray
        Cross-relation reorganisation energy of the equal-curvature coordinates
        (donor self-exchange and acceptor outer sphere) in kcal/mol at each
        dehydration level, same shape as theta.

    Raises
    ------
    ValueError
        If lam_acceptor_outer is not finite or not positive, if theta is not
        finite, or if any element of theta lies outside the closed interval
        [0, 1].
    '''
    return lam_sym
```

### Step 7

07_breathing_mode_surfaces

Goal
----
Step 07 - Harmonic metal-ligand breathing surfaces of the two acceptor oxidation states.

```python
import numpy as np
import numpy.typing as npt


def breathing_mode_surfaces(breathing: npt.ArrayLike) -> np.ndarray:
    '''Force constants and vertical energies of a metal-ligand breathing mode.

    Parameters
    ----------
    breathing : array_like
        Length-5 sequence [nu_ox, nu_red, d_ox, d_red, m_ligand]: the totally
        symmetric breathing wavenumbers of the oxidised and reduced complexes in
        cm^-1, the metal-ligand bond lengths of the oxidised and reduced
        complexes in angstrom, and the ligand mass in unified atomic mass units.
        The complex is octahedral with six equivalent metal-ligand bonds.

    Returns
    -------
    surfaces : numpy.ndarray
        Array of shape (4,). Element 0 is the metal-ligand force constant of the
        oxidised complex and element 1 that of the reduced complex, both in N/m.
        Element 2 is the energy of the reduced-state surface at the oxidised
        equilibrium geometry, measured from the reduced-state minimum, and
        element 3 the energy of the oxidised-state surface at the reduced
        equilibrium geometry, measured from the oxidised-state minimum, both in
        kcal/mol.

    Raises
    ------
    ValueError
        If breathing does not have exactly five elements, if any element is not
        finite, or if any wavenumber, bond length or mass is not positive.
    '''
    return surfaces
```

### Step 8

08_nuclear_factor

Goal
----
Step 08 - Exact classical nuclear factor with unequal metal-ligand curvatures.

```python
import numpy as np
import numpy.typing as npt


def nuclear_factor(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    '''Classical Franck-Condon weighted density of states at each dehydration level.

    Parameters
    ----------
    E0_acceptor : float
        Standard one-electron reduction potential of the acceptor, V vs SHE.
    lam_acceptor_outer : float
        Outer-sphere self-exchange reorganisation energy of the acceptor couple,
        kcal/mol, positive.
    breathing : array_like
        Length-5 sequence [nu_ox, nu_red, d_ox, d_red, m_ligand] describing the
        acceptor's octahedral breathing mode, as for breathing_mode_surfaces.
    theta : array_like
        Dehydration level or levels of the donor, each strictly inside (0, 1).
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal, angstrom.
    charge_product : float
        Product of the formal charges of the two reactants, signed.

    Returns
    -------
    fc : numpy.ndarray
        Thermal average, over the reactant state, of the delta function of the
        product-minus-reactant energy gap, in (kcal/mol)^-1, at 298.15 K and
        with all nuclear motion classical, same shape as theta. The value must
        be accurate to a relative 1e-10 or better.

    Raises
    ------
    ValueError
        If any scalar argument is not finite, if lam_acceptor_outer is not
        positive, if breathing is not a valid breathing-mode description, or if
        any element of theta does not lie strictly inside (0, 1).
    '''
    return fc
```

### Step 9

09_tunnelling_coupling_squared

Goal
----
Step 09 - Path-integrated electronic coupling between donor and acceptor.

```python
import numpy as np
import numpy.typing as npt


def tunnelling_coupling_squared(theta: npt.ArrayLike, Z_acceptor: float) -> np.ndarray:
    '''Squared electronic coupling for a donor at a given dehydration level.

    Parameters
    ----------
    theta : array_like
        Dehydration level or levels of the donor, each strictly inside (0, 1).
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal,
        angstrom, on the liquid side of the donor.

    Returns
    -------
    coupling_sq : numpy.ndarray
        Squared donor-acceptor electronic coupling in joules squared, same
        shape as theta.

    Raises
    ------
    ValueError
        If Z_acceptor is not finite, if theta is not finite, or if any element
        of theta does not lie strictly inside (0, 1).
    '''
    return coupling_sq
```

### Step 10

10_dehydration_density

Goal
----
Step 10 - Population density of the dehydration level across the slab.

```python
import numpy as np
import numpy.typing as npt


def dehydration_density(theta: npt.ArrayLike, Z_lo: float, Z_hi: float) -> np.ndarray:
    '''Normalised probability density of the dehydration level across a slab.

    Parameters
    ----------
    theta : array_like
        Dehydration level or levels, each strictly inside (0, 1).
    Z_lo : float
        Depth of the liquid-side face of the interfacial slab, angstrom.
    Z_hi : float
        Depth of the vapour-side face of the interfacial slab, angstrom,
        strictly greater than Z_lo.

    Returns
    -------
    density : numpy.ndarray
        Probability density of the dehydration level, dimensionless, normalised
        to unit integral over the dehydration levels the slab contains, same
        shape as theta.

    Raises
    ------
    ValueError
        If Z_lo or Z_hi is not finite, if Z_hi is not strictly greater than
        Z_lo, if theta is not finite, or if any element of theta does not lie
        strictly inside (0, 1).
    '''
    return density
```

### Step 11

11_rate_density

Goal
----
Step 11 - Golden-rule rate constant at a fixed dehydration level.

```python
import numpy as np
import numpy.typing as npt


def rate_density(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, theta: npt.ArrayLike, Z_acceptor: float, charge_product: float) -> np.ndarray:
    '''Non-adiabatic electron transfer rate constant at fixed dehydration level.

    Parameters
    ----------
    E0_acceptor : float
        Standard one-electron reduction potential of the acceptor, V vs SHE.
    lam_acceptor_outer : float
        Outer-sphere self-exchange reorganisation energy of the acceptor couple,
        kcal/mol, positive.
    breathing : array_like
        Length-5 sequence [nu_ox, nu_red, d_ox, d_red, m_ligand] describing the
        acceptor's octahedral breathing mode, as for breathing_mode_surfaces.
    theta : array_like
        Dehydration level or levels of the donor, each strictly inside (0, 1).
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal,
        angstrom, on the liquid side of the donor.
    charge_product : float
        Product of the formal charges of the two reactants, signed.

    Returns
    -------
    rate : numpy.ndarray
        Electron transfer rate constant in inverse seconds for a donor held at
        each dehydration level, same shape as theta, accurate to a relative
        1e-10 or better.

    Raises
    ------
    ValueError
        If any scalar argument is not finite, if lam_acceptor_outer is not
        positive, if breathing is not a valid breathing-mode description, or if
        any element of theta does not lie strictly inside (0, 1).
    '''
    return rate
```

### Step 12

12_interfacial_rate_constant

Goal
----
Step 12 - Orchestrator: population-averaged interfacial rate constant.

```python
import numpy as np
import numpy.typing as npt


def interfacial_rate_constant(E0_acceptor: float, lam_acceptor_outer: float, breathing: npt.ArrayLike, Z_lo: float, Z_hi: float, Z_acceptor: float, charge_product: float) -> float:
    '''Population-averaged interfacial electron transfer rate constant.

    Parameters
    ----------
    E0_acceptor : float
        Standard one-electron reduction potential of the acceptor, V vs SHE.
    lam_acceptor_outer : float
        Outer-sphere self-exchange reorganisation energy of the acceptor couple,
        kcal/mol, positive.
    breathing : array_like
        Length-5 sequence [nu_ox, nu_red, d_ox, d_red, m_ligand] describing the
        acceptor's octahedral breathing mode, as for breathing_mode_surfaces.
    Z_lo : float
        Depth of the liquid-side face of the interfacial slab, angstrom.
    Z_hi : float
        Depth of the vapour-side face of the interfacial slab, angstrom,
        strictly greater than Z_lo.
    Z_acceptor : float
        Depth of the fully solvated acceptor along the interface normal,
        angstrom, strictly below the liquid-side face of the slab.
    charge_product : float
        Product of the formal charges of the two reactants, signed.

    Returns
    -------
    rate_constant : float
        Population-averaged electron transfer rate constant in inverse seconds,
        as a native Python float, accurate to a relative 1e-10 or better.

    Raises
    ------
    ValueError
        If any scalar argument is not finite, if lam_acceptor_outer is not
        positive, if breathing is not a valid breathing-mode description, if
        Z_hi is not strictly greater than Z_lo, or if Z_acceptor is not strictly
        below Z_lo.
    '''
    return rate_constant
```

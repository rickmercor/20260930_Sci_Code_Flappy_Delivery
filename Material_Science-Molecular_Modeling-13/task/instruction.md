# Material_Science-Molecular_Modeling-13

## Background

A dilute solute in a phase-separating solvent feels the liquid-vapor interface as a one-body field. Within classical density functional theory of an additive square-well mixture, the hard-sphere reference in a fundamental-measure functional and the attraction in mean-field form determine the coexisting phases, the interfacial profile and, through Widom's insertion theorem, the effective potential the interface exerts on the solute and its excess adsorption.

## Problem

A dilute solute dissolved in a phase-separating solvent feels the liquid-vapor interface as a one-body field, and for nanoparticles in a simple solvent that field can be several $k_B T$ deep. A recent source treats this within classical density functional theory of an additive binary square-well mixture: the hard-sphere reference in a fundamental-measure functional, the attraction in a mean-field form written with weighted densities of the same geometric kind, liquid-vapor coexistence of the solvent, the interfacial density profile, and the effective one-body potential acting on the solute, obtained in the dilute limit by Widom's insertion theorem from the solvent profile alone. Recover that construction from the source and evaluate it exactly for the interfacial profile the source uses to describe its interface. The load-bearing choices are the source's and are not derivable from the statement below: which version of the fundamental-measure functional is used and what its uniform limit implies for the mixture; how the well depth of the solute is fixed so that a dilute solute prefers neither coexisting phase; how the effective potential follows from the insertion theorem, with which weight functions and against which reference; and how the excess adsorption of the solute at the interface is defined from that potential.

The model is the source's. Species 1, the solvent, and species 2, the solute, are additive hard spheres of diameters $\sigma_1$ and $\sigma_2$, $\sigma_{ij} = (\sigma_i + \sigma_j)/2$, with square-well attractions of depths $\epsilon_{ij}$ and ranges $\lambda_{ij}\sigma_{ij}$; the solute is at infinite dilution. The hard-sphere reference is treated by fundamental measure theory with Rosenfeld's weight functions, the volume weight $\Theta(R_i - |\mathbf r|)$, the surface weight $\delta(R_i - |\mathbf r|)$, its reductions by $4\pi R_i$ and $4\pi R_i^2$, and the vector surface weight $(\mathbf r/|\mathbf r|)\,\delta(R_i - |\mathbf r|)$ with its reduction by $4\pi R_i$, the weighted densities being $n_\alpha(\mathbf r) = \sum_i \int d\mathbf r'\,\rho_i(\mathbf r')\,w^i_\alpha(\mathbf r - \mathbf r')$. The attraction is treated in the mean-field random-phase form, $\beta F_{sw} = \tfrac{1}{2}\sum_{ij}\int\!\!\int \rho_i(\mathbf r)\rho_j(\mathbf r')\,\beta\phi^{ij}_{sw}(|\mathbf r - \mathbf r'|)$ with the well extended through the hard core, $\phi^{ij}_{sw}(r) = -\epsilon_{ij}$ for $r < \lambda_{ij}\sigma_{ij}$ and zero beyond. The cross energy is the geometric mean $\epsilon_{12} = \sqrt{\epsilon_1\epsilon_2}$; the cross range is $\lambda_{12} = (\lambda_1\sigma_1 + \lambda_2\sigma_2)/(\sigma_1 + \sigma_2)$, as implied by the source's weighted-density form of the attraction; and the solute range is fixed by the source's rule that the solute well has the same width as the solvent well, $\sigma_2(\lambda_2 - 1) = \sigma_1(\lambda_1 - 1)$. Lengths are in units of $\sigma_1$, energies in units of $k_B T$, and the thermal wavelength is set to $\sigma_1$. The one-body direct correlation function of a species is $c^{(1)}_i(\mathbf r) = -\delta\beta F_{ex}/\delta\rho_i(\mathbf r)$; in a uniform fluid it equals minus the excess chemical potential. The planar liquid-vapor interface of the solvent at coexistence, with the vapor at negative z and the liquid at positive z, is represented as the source does, by the sigmoidal profile $\rho_1(z) = \rho_v + (\rho_l - \rho_v)/(1 + e^{-a z})$ with the coexisting densities and steepness $a$, the midpoint at $z = 0$.

Implement eight functions with these conventions: coexistence is located to a relative precision of $10^{-14}$ as the pair of densities at which the chemical potentials and the pressures of the vapor and liquid branches agree, and raises `ValueError` when the equation of state has no such pair; every convolution is an integral of a smooth function over a finite window and is evaluated to better than $10^{-12}$, for instance by composite Gauss-Legendre quadrature, never on a fixed grid; the line integrals over the interface normal are taken over $|z| \le 25$, beyond which their integrands vanish to double precision; the minimum of the effective potential is located on $|z| \le 2$ to $10^{-12}$; and the derivatives of the free energy density are evaluated from its closed form, never by finite differences.

`swi_bulk_thermo(rho, beps, lam)` returns, for the pure solvent, the chemical potential $\beta\mu$, the pressure $\beta P\sigma_1^3$, the hard-sphere excess chemical potential and the square-well part of the chemical potential. `swi_coexistence(beps, lam)` returns the coexisting vapor and liquid densities and the common chemical potential and pressure. `swi_solute_bulk(rho1, sigma2, beps12, lam12)` returns the excess chemical potential of the solute at infinite dilution in the uniform solvent and its hard-sphere and square-well parts. `swi_symmetric_energy(rho_v, rho_l, sigma2, beps1, lam1)` returns the solute range, the contact distance, the cross range, the cross well depth at which the dilute solute has the same excess chemical potential in both coexisting phases, and the solute well depth that produces it. `swi_weighted_densities(z, rho_v, rho_l, a, R)` returns, shape $(6,)$, the planar weighted densities $n_0, n_1, n_2, n_3$ and the normal components of $n_{1v}, n_{2v}$ of the sigmoidal profile at z for the weights of a sphere of radius R. `swi_wb_derivatives(n)` returns, shape $(7,)$, the excess free energy density of the functional at six given weighted densities and its six partial derivatives. `swi_solute_c1(z, rho_v, rho_l, a, sigma2, beps12, lam12)` returns the one-body direct correlation function of the dilute solute at z in the sigmoidal solvent profile and the source's effective one-body potential $\beta V_{eff}(z)$ measured from the bulk liquid. `swi_audit(beps1, lam1, a, sigma2)`, the orchestrator, must call the earlier functions rather than reimplementing them and returns ten values: the coexisting vapor and liquid densities, the chemical potential and the pressure at coexistence; the cross well depth of the indifferent solute and the solute well depth that produces it; the surface excess grand potential per unit area $\beta\gamma\sigma_1^2$ of the sigmoidal profile at coexistence, the line integral of the grand potential density plus the pressure; the effective potential at $z = 0$; its minimum; and the source's reduced excess adsorption of the solute, the line integral of $e^{-\beta V_{eff}(z)} - 1$, in units of $\sigma_1$ per unit of the solute reservoir density.

All outputs are float64, finite and deterministic: two runs on identical inputs must agree exactly. Validate inputs and raise `ValueError` on non-finite values, on arrays of the wrong shape, on non-positive densities, well depths, steepness, radius or solute diameter, on a packing fraction of one or more and, in swi_wb_derivatives, on a packing weighted density n3 of one or more, on a well range below one, on a vapor density not below the liquid density, and, in the orchestrator, on a non-positive surface excess or on a solute that is not indifferent between the phases.

Evaluate the audit at $\beta\epsilon_1 = 1$, $\lambda_1 = 1.5$, $a = 1.79/\sigma_1$, $\sigma_2/\sigma_1 = 2$.

In your reasoning report the conventions you used, and justify each from the source: which version of the fundamental-measure functional the source uses and what equation of state its uniform limit gives for the mixture; how the attraction inside the hard core is treated and why that choice decides whether the solvent phase separates at this temperature; how the well depth at which the dilute solute prefers neither phase is fixed from the bulk excess chemical potentials; how the effective one-body potential follows from Widom's insertion theorem, with which weight functions the solvent's free energy derivatives are convolved and from which bulk it is measured; why the source's sigmoidal profile gives a surface excess above the source's own minimised interfacial tension; and how the excess adsorption of the solute is defined from the effective potential.

Report numerically, as evidence that the chain was executed: the coexisting vapor and liquid densities, the chemical potential and the pressure at coexistence; the cross well depth of the indifferent solute and the solute well depth that produces it; the surface excess of the sigmoidal profile; the effective potential at the midpoint; the position and the value of its minimum; and the reduced excess adsorption. These are the scalars that determine the final number.

As the final answer, report the minimum of the effective one-body potential $\beta V_{eff}$ of the interface on the solute, to six significant figures.

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

swi_bulk_thermo

Goal
----
Evaluates the bulk chemical potential and pressure of the square-well solvent in the source's mean-field model.

```python
def swi_bulk_thermo(rho: float, beps: float, lam: float) -> "np.ndarray":
    r"""rho: positive float, number density of the one-component square-well solvent in units of its
    hard-core diameter, with packing fraction below one. beps: positive float, the well depth in units
    of $k_B T$. lam: float not below one, the well range in units of the diameter.

    Returns a numpy float64 array of shape $(4,)$: the total chemical potential $\beta\mu$ with the
    thermal wavelength set to the diameter, the pressure $\beta P \sigma^3$, the hard-sphere excess
    chemical potential, and the square-well contribution to the chemical potential, for the source's
    model: the hard-sphere reference in the White Bear version of fundamental measure theory (whose
    uniform limit is the Carnahan-Starling equation of state) and the square-well attraction in
    mean-field random-phase form with the well extended through the hard core.

    Raises:
        ValueError: on a non-finite or non-positive rho, on a packing fraction of one or more, on a
            non-finite or non-positive beps, or on lam below one.
    """
    return None
```

### Step 2

swi_solute_bulk

Goal
----
Evaluates the excess chemical potential of a single solute particle at infinite dilution in the uniform solvent.

```python
def swi_solute_bulk(rho1: float, sigma2: float, beps12: float, lam12: float) -> "np.ndarray":
    r"""rho1: positive float, the solvent density with packing fraction below one. sigma2: positive
    float, the solute diameter in units of the solvent diameter. beps12: positive float, the
    solvent-solute well depth in units of $k_B T$. lam12: float not below one, the solvent-solute well
    range in units of the solvent-solute contact distance.

    Returns a numpy float64 array of shape $(3,)$: the excess chemical potential of a single solute
    particle at infinite dilution in the uniform solvent, in units of $k_B T$, and its two parts, the
    hard-sphere part from the uniform limit of the mixture functional and the square-well part from
    the mean-field attraction with the well extended through the core.

    Raises:
        ValueError: on a non-finite or non-positive rho1, sigma2 or beps12, on a solvent packing
            fraction of one or more, or on lam12 below one.
    """
    return None
```

### Step 3

swi_symmetric_energy

Goal
----
Fixes the solvent-solute ranges by the source's rules and the well depth at which a dilute solute prefers neither coexisting phase.

```python
def swi_symmetric_energy(rho_v: float, rho_l: float, sigma2: float, beps1: float, lam1: float) -> "np.ndarray":
    r"""rho_v, rho_l: positive floats, the coexisting vapor and liquid densities of the solvent, with
    rho_v below rho_l and packing fractions below one. sigma2: positive float, the solute diameter in
    units of the solvent diameter. beps1: positive float, the solvent well depth in units of $k_B T$.
    lam1: float not below one, the solvent well range.

    Returns a numpy float64 array of shape $(5,)$: the solute well range fixed by the source's rule
    that the solute well has the same width as the solvent well, $\sigma_2(\lambda_2 - 1) =
    \sigma_1(\lambda_1 - 1)$; the solvent-solute contact distance; the solvent-solute well range
    $\lambda_{12} = (\lambda_1\sigma_1 + \lambda_2\sigma_2)/(\sigma_1 + \sigma_2)$ implied by the
    source's weighted-density form of the attraction; the solvent-solute well depth in units of
    $k_B T$ for which a dilute solute has the same excess chemical potential in both coexisting
    phases, so that it prefers neither; and the solute well depth that produces it under the
    geometric-mean rule $\epsilon_{12} = \sqrt{\epsilon_1\epsilon_2}$.

    Raises:
        ValueError: on invalid densities, sigma2, beps1 or lam1, or on rho_v not below rho_l.
    """
    return None
```

### Step 4

swi_weighted_densities

Goal
----
Computes the planar fundamental-measure weighted densities of the sigmoidal solvent profile for a sphere of given radius.

```python
def swi_weighted_densities(z: float, rho_v: float, rho_l: float, a: float, R: float) -> "np.ndarray":
    r"""z: finite float, position along the interface normal in units of the solvent diameter.
    rho_v, rho_l: positive floats, the coexisting densities. a: positive float, the steepness of the
    sigmoidal profile. R: positive float, the radius of the sphere whose weight functions are applied.

    Returns a numpy float64 array of shape $(6,)$: the six planar weighted densities $n_0, n_1, n_2,
    n_3$ and the normal components $n_{1v}, n_{2v}$ of the vector weighted densities of the
    fundamental measure theory of Rosenfeld at z, for the sigmoidal solvent profile
    $\rho(z) = \rho_v + (\rho_l - \rho_v)/(1 + e^{-a z})$ (vapor at negative z, liquid at positive
    z) and the weight functions of a sphere of radius R: the volume weight, the surface weight, the
    surface weight divided by $4\pi R$ and by $4\pi R^2$, and the vector surface weight
    $(\mathbf r/|\mathbf r|)\,\delta(R - |\mathbf r|)$ with its $4\pi R$ reduction, each
    convolved as $n_\alpha(z) = \int dz'\,\rho(z')\,w_\alpha(z - z')$ with the weights integrated
    over the transverse plane. The convolution is an integral of a smooth function over the window
    $|z - z'| \le R$ and is evaluated to better than 1e-12.

    Raises:
        ValueError: on a non-finite z, on non-finite or non-positive rho_v, rho_l, a or R, or on
            rho_v not below rho_l.
    """
    return None
```

### Step 5

swi_wb_derivatives

Goal
----
Evaluates the White Bear excess free energy density and its exact derivatives with respect to the weighted densities.

```python
def swi_wb_derivatives(n: object) -> "np.ndarray":
    r"""n: array-like of shape $(6,)$ of finite floats: the weighted densities $n_0, n_1, n_2, n_3,
    n_{1v}, n_{2v}$, with $n_3$ below one.

    Returns a numpy float64 array of shape $(7,)$: the excess free energy density $\beta\Phi$ of the
    White Bear version of fundamental measure theory at these weighted densities, followed by its six
    partial derivatives with respect to $n_0, n_1, n_2, n_3, n_{1v}$ and $n_{2v}$, evaluated exactly
    from the closed form of $\Phi$, never by finite differences.

    Raises:
        ValueError: on an input of the wrong shape, on non-finite entries, or on $n_3$ not below one.
    """
    return None
```

### Step 6

swi_solute_c1

Goal
----
Evaluates the one-body direct correlation function of a dilute solute in the interfacial solvent profile and the source's effective one-body potential.

```python
def swi_solute_c1(z: float, rho_v: float, rho_l: float, a: float, sigma2: float, beps12: float, lam12: float) -> "np.ndarray":
    r"""z: finite float. rho_v, rho_l, a: as in the weighted-density step. sigma2: positive float, the
    solute diameter in units of the solvent diameter. beps12: positive float, the solvent-solute well
    depth in units of $k_B T$. lam12: float not below one, the solvent-solute well range.

    Returns a numpy float64 array of shape $(2,)$: the one-body direct correlation function
    $c^{(1)}_2(z)$ of a solute particle at infinite dilution at position z in the sigmoidal solvent
    profile, minus the functional derivative of the excess free energy of the mixture with respect to
    the solute density in units of $k_B T$, taken in the limit of vanishing solute density so that only
    the solvent profile enters; and the source's effective one-body potential
    $\beta V_{eff}(z) = c^{(1)}_2(+\infty) - c^{(1)}_2(z)$ of the interface on the solute, measured
    from the bulk liquid. The hard-sphere part is the convolution of the White Bear derivatives of the
    solvent's weighted densities with the weight functions of the solute sphere, and the attractive
    part is the mean-field convolution of the solvent profile with the solvent-solute well extended
    through the core; both are integrals of smooth functions over finite windows and are evaluated to
    better than 1e-12.

    Raises:
        ValueError: on a non-finite z, on invalid rho_v, rho_l, a, sigma2 or beps12, or on lam12 below
            one.
    """
    return None
```

### Step 7

swi_coexistence

Goal
----
Locates the liquid-vapor coexistence of the square-well solvent from the equality of chemical potential and pressure.

```python
def swi_coexistence(beps: float, lam: float) -> "np.ndarray":
    r"""beps: positive float, the well depth in units of $k_B T$. lam: float not below one, the well
    range in units of the diameter.

    Returns a numpy float64 array of shape $(4,)$: the coexisting vapor and liquid densities of the
    solvent and the common chemical potential $\beta\mu$ and pressure $\beta P\sigma^3$ at
    liquid-vapor coexistence, the two densities at which the chemical potentials and the pressures of
    the two branches agree, located to a relative precision of 1e-14.

    Raises:
        ValueError: on invalid beps or lam, or when the equation of state has no liquid-vapor
            coexistence at this temperature.
    """
    return None
```

### Step 8

swi_audit

Goal
----
Runs the whole chain from the bulk thermodynamics to the interfacial effective potential of the dilute solute and its adsorption.

```python
def swi_audit(beps1: float, lam1: float, a: float, sigma2: float) -> "np.ndarray":
    r"""beps1: positive float, the solvent well depth in units of $k_B T$. lam1: float not below one,
    the solvent well range. a: positive float, the steepness of the sigmoidal interfacial profile in
    inverse solvent diameters. sigma2: positive float, the solute diameter in units of the solvent
    diameter.

    The orchestrator. It must call the earlier functions rather than reimplementing them.
    Returns a numpy float64 array of shape $(10,)$: the coexisting vapor and liquid densities, the
    chemical potential and the pressure at coexistence; the solvent-solute well depth at which the
    dilute solute prefers neither phase and the solute well depth that produces it; the surface excess
    grand potential per unit area $\beta\gamma\sigma_1^2$ of the sigmoidal profile at coexistence,
    the integral over the whole line of the grand potential density plus the pressure, evaluated on
    $|z| \le 25$ beyond which the integrand vanishes to double precision; the effective potential at
    the profile midpoint z = 0; its minimum over $|z| \le 2$, located to 1e-12; and the source's
    reduced excess adsorption of the solute, $\int dz\,[e^{-\beta V_{eff}(z)} - 1]$ over the same
    line, in units of the solvent diameter and per unit of the solute reservoir density.

    Raises:
        ValueError: whenever any of the functions it calls would raise, and when the surface excess
            of the sigmoidal profile is not positive.
    """
    return None
```

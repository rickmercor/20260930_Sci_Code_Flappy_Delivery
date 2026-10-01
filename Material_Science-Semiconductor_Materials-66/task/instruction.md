# Material_Science-Semiconductor_Materials-66

## Background

## Why cryogenic transistor modelling is hard

Quantum computers built on solid-state qubits operate in a dilution refrigerator, and the
classical electronics that read and control those qubits would ideally sit next to them
rather than at room temperature behind thousands of wires. That has made cryogenic CMOS a
serious engineering target, and it has exposed how badly standard compact transistor models
degrade below liquid-nitrogen temperature. The industry cannot design a cryogenic control
chip against a model fitted only at 300 K, so the modelling problem is on the
critical path for the hardware.

Three things go wrong at once as a silicon MOSFET is cooled towards 12 K. The threshold
voltage shifts, because the band gap widens and the bulk Fermi level moves as the acceptor
dopants freeze out and stop supplying holes. The subthreshold region stops behaving: the
classical inverse subthreshold slope is proportional to the thermal voltage and should fall
linearly towards zero as the lattice cools, yet measured devices saturate at a slope far
larger than kT/q predicts, and the turn-on stays visibly gradual. And the above-threshold
current fails to rise in proportion to the low-field mobility, even though phonon scattering
is strongly suppressed at low temperature.

## What the source work contributes

The difficulty is not that these effects are unknown individually but that they are easy to
confuse with one another. A model with enough free parameters can reproduce a measured
transfer curve by letting an interface-trap density stand in for a threshold misalignment,
or by reinterpreting the lattice temperature itself so that one knob quietly
absorbs several mechanisms at once. Such a fit has no predictive value away from the
point where it was fitted.

The source work therefore builds a model whose components are deliberately separable. It
keeps the charge-based electrostatic backbone of an established cryogenic MOSFET model and
extends it in four independent places: the metal-semiconductor work-function difference is
evaluated as an explicitly temperature-dependent quantity so that the threshold alignment is
established before anything else is fitted; the discrete interface-trap levels are replaced
by a continuous energy spectrum so that the electrostatics stop depending on where those
levels were placed; the statistical mapping between local potential and mobile carrier
density is modified so that subthreshold carrier formation broadens without touching any
other temperature-dependent quantity; and the mobility degradation is tied to the
electrostatically computed vertical field rather than to the applied gate bias.

The third of those is the load-bearing one. Interface traps reshape how the gate couples to
the channel, but they leave the Boltzmann relation between local potential and carrier
density untouched, so no amount of trap charge can produce a gradual turn-on once the
thermal voltage is small. Earlier work has introduced band-tail states to fix this in
several different ways, each built to reproduce a particular subthreshold metric rather than a
complete transfer characteristic, and each carrying the risk that the band-tail term quietly
compensates for mechanisms it should be independent of.

## Consequence for the computation

Because the modified carrier density has no elementary antiderivative in the potential, the
mobile charge acquires no closed-form expression and the electrostatics must be integrated
numerically at every bias point. The model is therefore a physics-based numerical charge
model rather than a closed-form compact model, and it sits deliberately between the
circuit-oriented compact models that prioritise simulator speed and the detailed numerical
approaches that resolve microscopic trap and band-tail spectra at much greater cost. The
pipeline is a chain: band parameters and bulk neutrality fix the electrostatic reference, the
trap spectrum and the modified carrier statistics fix the charge at a given surface
potential, the gate boundary condition inverts that relation to find the surface potential at
a given bias, and the two ends of the channel feed the charge-based drift-diffusion current.
An error at any link moves the final current, which is what makes a single reported current a
meaningful test of the whole model.

The model is validated against measured transfer characteristics of a scaled bulk silicon
transistor over a wide cryogenic range at fixed low drain bias.

## Problem

Cryogenic CMOS is the control layer that large-scale quantum computers need, and it requires transistor models that stay predictive from room temperature down to a few kelvin. Existing cryo-FET models cannot reproduce the threshold-voltage shift, the gradual subthreshold turn-on and the above-threshold transport of a scaled device at the same time, because those behaviours originate in physically distinct mechanisms that one refitted correction cannot separate. A recent physics-based model keeps a charge-based electrostatic backbone, separates those mechanisms explicitly, and pays for it by having to evaluate the resulting electrostatics numerically, since the mobile charge no longer admits a closed form; your task is to implement that model for one bias point of a 65 nm bulk silicon n-channel transistor and report the drain current.

The electrostatics rest on the one-dimensional Poisson equation under the gradual-channel approximation, with the ionised acceptor concentration given by the Fermi-Dirac occupation of the acceptor level so that dopant freeze-out is described rather than assumed away; its first integral yields the surface field and hence the total semiconductor charge, from which the mobile inversion charge follows by subtracting the depletion charge, itself the exact analytic integral of that same varying ionised-acceptor profile. The gate boundary condition ties the applied gate voltage to the surface potential through the flat-band voltage, which carries a temperature-dependent metal-semiconductor work-function difference alongside the occupied charge of a continuous interface-trap energy distribution that decays exponentially away from the conduction-band edge. What makes this model post-cutoff is the mobile electron density entering the Poisson integrand: it is not the Maxwell-Boltzmann conduction-band population but a broadened mobile-electron response, whose logarithmic sensitivity to the local electrostatic potential is set by a potential-dependent statistical voltage that broadens carrier formation in the subthreshold regime and recovers conventional conduction-band behaviour in strong inversion, and that replacement is the contribution under test and is deliberately not written out here. Only the mobile-carrier response is modified: the actual lattice temperature continues to govern the thermal voltage, the band gap, the intrinsic density, the dopant ionisation and the trap occupation, and the drain current then follows from the charge-based drift-diffusion expression evaluated between the source- and drain-side mobile charges, with the electrostatic coupling factor and the effective-field-degraded mobility both taken from the source-side solution as is appropriate at low drain bias.

Evaluate the model at T = 12 K for the configuration below and report the magnitude of the drain current in microamperes, fixing the numerics as follows so the result is reproducible: evaluate every energy or potential integral by Gauss-Legendre quadrature with exactly 400 nodes; solve the surface potential by bisection bracketed on [psi_b + 1e-9, Eg(T)/2 + Vch + 0.20] volts, stopping when the bracket is narrower than 1e-14 V and returning its midpoint; take the band gap as Eg(T) = 1.170 - 4.730e-4 * T^2 / (T + 636) eV; scale both band-edge effective densities of states as T^(3/2) from NC(300 K) = 2.86e19 cm^-3 and NV(300 K) = 2.66e19 cm^-3; reference all potentials to the intrinsic level under the midgap approximation; use eps_si = 11.7 * eps_0 and eps_ox = 3.9 * eps_0; and neglect the hole contribution when evaluating the mobile charge.

    T            = 12 K                      acceptor doping NA      = 3.0e17 cm^-3
    gate oxide   = 2.6 nm SiO2               acceptor degeneracy gA  = 4
    W            = 2.0 um                    acceptor level EA - EV  = 0.045 eV
    L            = 65 nm                     trap degeneracy gt      = 2
    VGB          = 0.62 V                    gate work function      = 4.35 V
    VDS          = 0.10 V                    silicon affinity        = 4.05 V
    Dit,C        = 1.833e12 cm^-2 eV^-1      U_BT                    = 0.0291 V
    E_decay      = 0.2943 eV                 U_sp                    = 0.0998 V
    mu0          = 395.655 cm^2 V^-1 s^-1    V_offset                = 4.088 mV
    theta1       = 0.55428                   eta                     = 0.5
    theta2       = 0.10007                   E0 (field normaliser)   = 1e6 V/cm

Alongside the tagged value report the scalars that determine it, namely the electrostatic reference quantities of the body, the potential at which the carrier statistics change regime together with the statistical voltage and the band-tail weighting evaluated at the source-side surface potential, the surface potential at each end of the channel, the mobile inversion charge density at each end together with the depletion and the occupied interface-trap charge densities at the source end, the transport coefficients entering the current, and the relative weight of the drift and diffusion contributions. Give the provenance of the method: its source, the framework it extends, how it differs from earlier treatments of the same effect, how its parameters were constrained, and the device it was validated against. Report as a number the ratio of your answer to what a conventional treatment of the carrier statistics would have given for this same configuration.

Everything named in the previous paragraph is part of what is being asked for, so report those values as a compact list; the brevity instruction below governs narration of the derivation, not that list.

## Output format — dedicated field

Emit <final_answer> immediately, followed by <reasoning>. Put exactly one finite decimal number, the drain-current magnitude in microamperes, inside <final_answer>...</final_answer>; include no units or other text inside those tags. Inside <reasoning>...</reasoning>, give a compact list of every scalar requested in the problem, with units, followed by a short provenance and method-summary paragraph. Keep the narrative to a few hundred words; this brevity instruction does not exclude the requested scalar list or provenance. Do not include per-iteration output or a long derivation

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

01_log_intrinsic_carrier_density

Goal
----
Return the natural logarithm of silicon's intrinsic carrier concentration for the supplied temperature and band-edge density-of-states references.

```python
def log_intrinsic_carrier_density(T: float, NC300: float, NV300: float) -> float:
    '''Natural logarithm of the intrinsic carrier concentration of silicon.

    Parameters
    ----------
    T : float
        Lattice temperature in kelvin, strictly positive.
    NC300 : float
        Conduction-band effective density of states at 300 K, in cm^-3,
        strictly positive.
    NV300 : float
        Valence-band effective density of states at 300 K, in cm^-3,
        strictly positive.

    Returns
    -------
    log_ni : float
        Natural logarithm of the intrinsic carrier concentration in cm^-3, as a
        native Python float. Always finite; large and negative at cryogenic
        temperature.

    Raises
    ------
    ValueError
        If `T` is not a finite strictly positive scalar, or if either of `NC300`
        and `NV300` is not a finite strictly positive scalar.
'''
    return log_ni
```

### Step 2

02_bulk_potential

Goal
----
Return the equilibrium bulk potential of the p-type body including incomplete dopant ionisation.

```python
def bulk_potential(T: float, NA: float, gA: float, EA_above_EV: float) -> float:
    '''Equilibrium bulk potential of the p-type body including incomplete ionisation.

    Parameters
    ----------
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level measured upward from the valence-band edge, in eV,
        strictly positive.

    Returns
    -------
    psi_b : float
        Bulk potential in volts referenced to the intrinsic level, as a native
        Python float. Negative for a p-type body.

    Raises
    ------
    ValueError
        If any of `T`, `NA`, `gA`, `EA_above_EV` is not a finite strictly
        positive scalar.
'''
    return psi_b
```

### Step 3

03_work_function_difference

Goal
----
Return the temperature-dependent metal-semiconductor work-function difference.

```python
def work_function_difference(T: float, NA: float, phi_m: float, chi_si: float,
                             gA: float, EA_above_EV: float) -> float:
    '''Metal-semiconductor work-function difference at temperature T.

    Parameters
    ----------
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    phi_m : float
        Gate work function in volts, treated as temperature independent,
        strictly positive.
    chi_si : float
        Electron affinity of silicon in volts, strictly positive.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level measured upward from the valence-band edge, in eV,
        strictly positive.

    Returns
    -------
    phi_ms : float
        Work-function difference in volts, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is not a finite strictly positive scalar.
'''
    return phi_ms
```

### Step 4

04_interface_trap_charge

Goal
----
Return the occupied interface-trap charge density at the specified surface and channel potentials.

```python
def interface_trap_charge(psi_s: float, Vch: float, T: float, DitC: float,
                          Edecay: float, gt: float, n_nodes: int) -> float:
    '''Occupied interface-trap charge density for the continuous trap spectrum.

    Parameters
    ----------
    psi_s : float
        Surface potential in volts, referenced to the intrinsic level.
    Vch : float
        Channel potential in volts, finite.
    T : float
        Lattice temperature in kelvin, strictly positive.
    DitC : float
        Interface-trap density scale at the conduction-band edge, in
        cm^-2 eV^-1, strictly positive.
    Edecay : float
        Characteristic energy decay scale of the trap spectrum, in eV,
        strictly positive.
    gt : float
        Trap degeneracy factor, strictly positive.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.

    Returns
    -------
    Qit : float
        Occupied interface-trap charge density in C/cm^2, negative or zero,
        as a native Python float.

    Raises
    ------
    ValueError
        If `psi_s` or `Vch` is not a finite scalar, if any of `T`, `DitC`,
        `Edecay`, `gt` is not a finite strictly positive scalar, or if
        `n_nodes` is not an integer greater than or equal to 2.
'''
    return Qit
```

### Step 5

05_band_tail_carrier_density

Goal
----
Return the band-tail-assisted mobile electron density at the supplied local electrostatic potential or potentials.

```python
def band_tail_carrier_density(psi: Union[float, np.ndarray], Vch: float, T: float,
                              UBT: float, Usp: float, Voffset: float,
                              n_nodes: int) -> Union[float, np.ndarray]:
    '''Band-tail-assisted mobile electron density at one or more local potentials.

    Parameters
    ----------
    psi : float or np.ndarray
        Local electrostatic potential in volts, referenced to the intrinsic
        level. Either a finite scalar or a finite 1D array.
    Vch : float
        Channel potential in volts, finite.
    T : float
        Lattice temperature in kelvin, strictly positive.
    UBT : float
        Characteristic band-tail statistical voltage in volts, strictly
        positive.
    Usp : float
        Smoothing voltage of the statistical crossover, in volts, strictly
        positive.
    Voffset : float
        Offset in volts locating the crossover relative to the conduction-band
        reference potential. Finite, may be negative.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.

    Returns
    -------
    nBT : float or np.ndarray
        Mobile electron density in cm^-3. A native Python float when `psi` is a
        scalar, otherwise a 1D array of the same length as `psi`.

    Raises
    ------
    ValueError
        If `psi` is not a finite scalar or finite 1D array, if `Vch` or
        `Voffset` is not a finite scalar, if any of `T`, `UBT`, `Usp` is not a
        finite strictly positive scalar, or if `n_nodes` is not an integer
        greater than or equal to 2.
'''
    return nBT
```

### Step 6

06_depletion_charge_density

Goal
----
Return the depletion charge density for the supplied cryogenic electrostatic state.

```python
def depletion_charge_density(psi_s: float, Vch: float, T: float, NA: float,
                             gA: float, EA_above_EV: float) -> float:
    '''Depletion charge density including incomplete dopant ionisation.

    Parameters
    ----------
    psi_s : float
        Surface potential in volts, referenced to the intrinsic level. Must not
        lie below the bulk potential.
    Vch : float
        Channel potential in volts, finite.
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level above the valence-band edge, in eV, strictly positive.

    Returns
    -------
    Qdep : float
        Depletion charge density in C/cm^2, negative or zero, as a native Python
        float.

    Raises
    ------
    ValueError
        If `psi_s` or `Vch` is not a finite scalar, if any of `T`, `NA`, `gA`,
        `EA_above_EV` is not a finite strictly positive scalar, or if `psi_s`
        lies below the bulk potential.
'''
    return Qdep
```

### Step 7

07_mobile_charge_density

Goal
----
Return the mobile inversion charge density at the supplied surface and channel potentials.

```python
def mobile_charge_density(psi_s: float, Vch: float, T: float, NA: float, UBT: float,
                          Usp: float, Voffset: float, gA: float, EA_above_EV: float,
                          n_nodes: int) -> float:
    '''Mobile inversion charge density at a given surface and channel potential.

    Parameters
    ----------
    psi_s : float
        Surface potential in volts, referenced to the intrinsic level. Must not
        lie below the bulk potential.
    Vch : float
        Channel potential in volts, finite.
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    UBT : float
        Characteristic band-tail statistical voltage in volts, strictly positive.
    Usp : float
        Smoothing voltage of the statistical crossover, in volts, strictly positive.
    Voffset : float
        Offset in volts locating the crossover. Finite, may be negative.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level above the valence-band edge, in eV, strictly positive.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.

    Returns
    -------
    Qm : float
        Mobile inversion charge density in C/cm^2, negative or zero, as a native
        Python float.

    Raises
    ------
    ValueError
        If `psi_s`, `Vch` or `Voffset` is not a finite scalar, if any of `T`,
        `NA`, `UBT`, `Usp`, `gA`, `EA_above_EV` is not a finite strictly positive
        scalar, if `n_nodes` is not an integer greater than or equal to 2, or if
        `psi_s` lies below the bulk potential.
'''
    return Qm
```

### Step 8

08_surface_potential

Goal
----
Return the surface potential corresponding to the supplied gate and channel voltages.

```python
def surface_potential(VGB: float, Vch: float, T: float, NA: float, Cox: float,
                      phi_ms: float, DitC: float, Edecay: float, UBT: float,
                      Usp: float, Voffset: float, gt: float, gA: float,
                      EA_above_EV: float, n_nodes: int, tol: float) -> float:
    '''Surface potential solving the gate boundary condition at one bias point.

    Parameters
    ----------
    VGB : float
        Gate-to-body voltage in volts, finite.
    Vch : float
        Channel potential in volts, finite.
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    Cox : float
        Oxide capacitance per unit area in F/cm^2, strictly positive.
    phi_ms : float
        Metal-semiconductor work-function difference in volts, finite.
    DitC : float
        Interface-trap density scale at the conduction-band edge, in cm^-2 eV^-1,
        strictly positive.
    Edecay : float
        Energy decay scale of the trap spectrum, in eV, strictly positive.
    UBT : float
        Characteristic band-tail statistical voltage in volts, strictly positive.
    Usp : float
        Smoothing voltage of the statistical crossover, in volts, strictly positive.
    Voffset : float
        Offset in volts locating the crossover. Finite, may be negative.
    gt : float
        Trap degeneracy factor, strictly positive.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level above the valence-band edge, in eV, strictly positive.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.
    tol : float
        Absolute bracket width at which the bisection stops, in volts, strictly
        positive.

    Returns
    -------
    psi_s : float
        Surface potential in volts, as a native Python float.

    Raises
    ------
    ValueError
        If `VGB`, `Vch`, `phi_ms` or `Voffset` is not a finite scalar, if any of
        `T`, `NA`, `Cox`, `DitC`, `Edecay`, `UBT`, `Usp`, `gt`, `gA`,
        `EA_above_EV`, `tol` is not a finite strictly positive scalar, if
        `n_nodes` is not an integer greater than or equal to 2, or if the residual
        does not change sign across the initial bracket.
'''
    return psi_s
```

### Step 9

09_effective_mobility

Goal
----
Return the effective electron mobility for the supplied depletion and mobile charges.

```python
def effective_mobility(Qdep: float, Qm: float, mu0: float, theta1: float,
                       theta2: float, eta: float) -> float:
    '''Electron mobility degraded by the effective vertical field.

    Parameters
    ----------
    Qdep : float
        Depletion charge density in C/cm^2, finite.
    Qm : float
        Mobile inversion charge density in C/cm^2, finite.
    mu0 : float
        Low-field electron mobility at the operating temperature, in
        cm^2 V^-1 s^-1, strictly positive.
    theta1 : float
        Dimensionless linear mobility-degradation coefficient, non-negative.
    theta2 : float
        Dimensionless quadratic mobility-degradation coefficient, non-negative.
    eta : float
        Weight of the mobile inversion charge in the effective field,
        non-negative.

    Returns
    -------
    mu_n : float
        Effective electron mobility in cm^2 V^-1 s^-1, as a native Python float.

    Raises
    ------
    ValueError
        If `Qdep` or `Qm` is not a finite scalar, if `mu0` is not a finite
        strictly positive scalar, or if any of `theta1`, `theta2`, `eta` is not
        a finite non-negative scalar.
'''
    return mu_n
```

### Step 10

10_coupling_factor

Goal
----
Return the electrostatic coupling factor for the supplied surface and channel potentials.

```python
def coupling_factor(psi_s: float, Vch: float, T: float, NA: float, Cox: float,
                    DitC: float, Edecay: float, gt: float, gA: float,
                    EA_above_EV: float, n_nodes: int) -> float:
    '''Electrostatic coupling factor at a given surface and channel potential.

    Parameters
    ----------
    psi_s : float
        Surface potential in volts. Must lie strictly above the bulk potential.
    Vch : float
        Channel potential in volts, finite.
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    Cox : float
        Oxide capacitance per unit area in F/cm^2, strictly positive.
    DitC : float
        Interface-trap density scale at the conduction-band edge, in cm^-2 eV^-1,
        strictly positive.
    Edecay : float
        Energy decay scale of the trap spectrum, in eV, strictly positive.
    gt : float
        Trap degeneracy factor, strictly positive.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level above the valence-band edge, in eV, strictly positive.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.

    Returns
    -------
    m : float
        Dimensionless electrostatic coupling factor, greater than one, as a
        native Python float.

    Raises
    ------
    ValueError
        If `psi_s` or `Vch` is not a finite scalar, if any of `T`, `NA`, `Cox`,
        `DitC`, `Edecay`, `gt`, `gA`, `EA_above_EV` is not a finite strictly
        positive scalar, if `n_nodes` is not an integer greater than or equal to
        2, or if `psi_s` does not lie strictly above the bulk potential.
'''
    return m
```

### Step 11

11_drain_current

Goal
----
Return the magnitude of the drain current in microamperes for the supplied device, bias and model parameters.

```python
def drain_current_microamp(T: float, NA: float, tox: float, Wch: float, Lch: float,
                           VGB: float, VDS: float, phi_m: float, chi_si: float,
                           DitC: float, Edecay: float, UBT: float, Usp: float,
                           Voffset: float, mu0: float, theta1: float, theta2: float,
                           eta: float, gt: float, gA: float, EA_above_EV: float,
                           n_nodes: int, tol: float) -> float:
    '''Drain current of the cryogenic FET at one bias point, in microamperes.

    Parameters
    ----------
    T : float
        Lattice temperature in kelvin, strictly positive.
    NA : float
        Implanted acceptor concentration in cm^-3, strictly positive.
    tox : float
        Physical gate-oxide thickness in cm, strictly positive.
    Wch : float
        Channel width in cm, strictly positive.
    Lch : float
        Channel length in cm, strictly positive.
    VGB : float
        Gate-to-body voltage in volts, finite.
    VDS : float
        Drain-to-source voltage in volts, finite and non-negative.
    phi_m : float
        Gate work function in volts, strictly positive.
    chi_si : float
        Electron affinity of silicon in volts, strictly positive.
    DitC : float
        Interface-trap density scale at the conduction-band edge, in cm^-2 eV^-1,
        strictly positive.
    Edecay : float
        Energy decay scale of the trap spectrum, in eV, strictly positive.
    UBT : float
        Characteristic band-tail statistical voltage in volts, strictly positive.
    Usp : float
        Smoothing voltage of the statistical crossover, in volts, strictly positive.
    Voffset : float
        Offset in volts locating the crossover. Finite, may be negative.
    mu0 : float
        Low-field electron mobility at T, in cm^2 V^-1 s^-1, strictly positive.
    theta1 : float
        Linear mobility-degradation coefficient, non-negative.
    theta2 : float
        Quadratic mobility-degradation coefficient, non-negative.
    eta : float
        Weight of the mobile charge in the effective field, non-negative.
    gt : float
        Trap degeneracy factor, strictly positive.
    gA : float
        Acceptor degeneracy factor, strictly positive.
    EA_above_EV : float
        Acceptor level above the valence-band edge, in eV, strictly positive.
    n_nodes : int
        Number of Gauss-Legendre nodes, an integer of at least 2.
    tol : float
        Bracket width at which the surface-potential bisection stops, in volts,
        strictly positive.

    Returns
    -------
    IDS : float
        Magnitude of the drain current in microamperes, as a native Python float.

    Raises
    ------
    ValueError
        If `VGB` or `Voffset` is not a finite scalar, if `VDS` is not a finite
        non-negative scalar, if any of `T`, `NA`, `tox`, `Wch`, `Lch`, `phi_m`,
        `chi_si`, `DitC`, `Edecay`, `UBT`, `Usp`, `mu0`, `gt`, `gA`,
        `EA_above_EV`, `tol` is not a finite strictly positive scalar, if either
        of `theta1`, `theta2`, `eta` is not a finite non-negative scalar, or if
        `n_nodes` is not an integer greater than or equal to 2.
'''
    return IDS
```

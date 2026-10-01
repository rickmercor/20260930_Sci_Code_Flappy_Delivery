# Material_Science-Semiconductor_Materials-72

## Background

## Why a quantum-dot LED is hard to simulate at its interfaces

A quantum-dot light-emitting diode is a stack of thin layers, each doing one job. A hole-injection layer accepts holes from the anode, a hole-transport layer carries them inward, a quantum-dot emissive layer captures electrons and holes and radiates, and an electron-transport layer feeds electrons from the other side. Designing such a device means predicting how the applied voltage divides across the stack, where carriers accumulate and what fraction of them recombine radiatively, and that prediction is normally made with a drift-diffusion model: continuity equations for the two carrier species coupled to Poisson's equation for the electrostatics, integrated in time until the device settles.

The awkwardness is concentrated at the junctions between layers. Adjacent organic materials generally differ in electron affinity, ionisation potential and doping, so at a boundary the flat-band levels step discontinuously and the carrier populations on the two sides can differ by orders of magnitude. These are genuine physical discontinuities rather than modelling artefacts, and the numerical scheme has to survive them. Because the carrier profiles change fastest next to such a junction, the grid is often made finer on that side of it than elsewhere.

## Why the problem is a discretisation problem rather than a physics problem

Finite-difference drift-diffusion codes store carrier densities at the nodes and evaluate currents on the boundaries between cells, so a rule is needed to supply a boundary density from the two nodal values flanking it. The obvious rule is the arithmetic mean, and inside a layer, where densities vary gently from node to node, it is perfectly adequate.

At a junction it is not. If one side holds a majority population and the other a sparse one, the arithmetic mean sits close to the majority value. When the local field drives drift out of the sparse side, that inflated boundary density, multiplied by the mobility and the field, gives a flux that removes more carriers from the sparse node than it contains. The nodal density undershoots, and because nothing in an explicit update prevents it, the undershoot carries the density below zero. A negative carrier density is not merely inaccurate: it inverts the sign of the drift term on the next step, which feeds a growing oscillation until the integration diverges.

Several remedies exist. Refining the mesh delays the problem without removing it. Exponentially fitted fluxes of the Scharfetter-Gummel type work well for homojunctions but assume a smooth potential across each interval, which an abrupt band offset violates. Interface-specific boundary treatments introduce extra parameters. A smaller intervention keeps the flux-conservative discretisation exactly as it is and changes only how the boundary density entering the drift term is chosen at the junction. Because only that density is affected, the current leaving one node is still exactly the current entering its neighbour, so charge conservation is untouched.

## Stability, and why the classical bound is not the operative one

Explicit time integration of a diffusion equation is conditionally stable, and the textbook condition bounds the dimensionless group formed from the diffusion coefficient, the time step and the square of the grid spacing; for pure diffusion in one dimension that bound is one half. A drift-diffusion problem at a doped heterojunction is stiffer than pure diffusion: drift adds its own transport rate, recombination is nonlinear in the carrier densities, and the abrupt interface couples strongly to both. The operative bound is therefore more restrictive than the classical one, and it is set by the finest spacing on the grid, so refining the grid next to a junction also tightens the admissible time step.

## Scope of the model treated here

The region treated is the injection-layer and transport-layer pair, where the interfacial difficulty lives, followed through the transient from thermal equilibrium under an applied bias. The emissive layer and its quantum-dot physics, the field-dependent injection and hopping currents between dots, field-enhanced mobilities and radiative recombination lie outside this region and are not modelled, so the conclusions concern the numerical behaviour of the interfacial discretisation rather than a full device characterisation.

## Problem

A quantum-dot light-emitting diode is a stack of thin transport layers, and its drift-diffusion simulation is most fragile at the junction between the PEDOT:PSS hole-injection layer and the TFB hole-transport layer, where the flat-band valence level drops by 0.43 eV across a single grid interval, the acceptor doping falls by more than two orders of magnitude and the intrinsic carrier density falls by twelve. Interpolating the carrier density onto that cell boundary by the conventional arithmetic mean lets the drift current drain the sparse transport-side node below zero, after which the explicit integration diverges, and the source work cures this by choosing the junction density according to the local band-edge field. Your task is to reproduce one deterministic instance of this interfacial transient with the source work's treatment of the junction, on a grid that resolves the transport layer more finely than the injection layer, and to report the hole density on the transport side of the junction.

Starting from per-layer thermal equilibrium, every explicit step re-solves Poisson's equation from the current space charge, forms the valence- and conduction-band driving fields from that potential and the flat-band levels, evaluates the drift-diffusion current densities between nodes, and advances the nodal densities through the discretised continuity equations with a Shockley-Read-Hall sink, the two contact nodes staying at their equilibrium densities. Discretise the stack as the source work discretises it, including its treatment of the junction interval and of unequal grid spacings; where its discretised expressions and its continuum definitions disagree, the continuum definitions govern. Use exactly this configuration:

- Grid: nodes 0 to 20 in the injection layer at 1 nm spacing starting from z = 0; nodes 21 to 100 in the transport layer at 0.25 nm spacing continuing from node 20, so the junction interval between nodes 20 and 21 is 0.25 nm wide
- Time integration: explicit, `dt = 1` ps, `n_steps = 1000`, `T = 300` K
- Potential 0 V at node 0 and `V = 1.0` V at node 100
- Flat-band levels: `EV = -5.17` eV and `-5.60` eV; `EC = -3.60` eV and `-2.60` eV (injection layer, transport layer)
- Acceptor doping `2.81e19` and `1.00e17` cm^-3; intrinsic densities `1.63e6` and `1.59e-6` cm^-3
- Relative permittivity `3.0` and `9.4`, the transport-layer value being the one the source work adopts for this material in its full-device parameter set
- Hole mobilities `2.0e-4` and `2.0e-3` cm^2 V^-1 s^-1; electron mobilities `2.0e-6` and `2.0e-5` cm^2 V^-1 s^-1
- Shockley-Read-Hall lifetimes `tau_p = tau_n = 1e-7` s; Auger capture probabilities zero
- Constants: `q = 1.602176634e-19` C, `k_B = 1.380649e-23` J K^-1, vacuum permittivity `8.8541878128e-14` F cm^-1

Report as the final answer the hole density at node 21 after the 1000 steps, in units of `1e18` cm^-3. The scalars that determine and check this number, the non-integer ones each to four significant figures, are: the hole density at node 20 at the end; the valence-band field at the midpoint between nodes 20 and 21 in the last of the 1000 updates, and the hole current density at that midpoint in the first and in the last update; the diffusion stability number of this configuration, with the smallest time step at which the source work's own direct stability test failed to converge at this transport-layer spacing; the node-21 density after the first update, and the update at which the densities first become non-finite, when the same integration keeps the arithmetic mean at the junction; and the node-21 result when the transport layer instead carries the TFB dielectric constant of the earlier device model that the source work cites for its injection-current and Poisson equations (give that value), together with the source work's stated reason for adopting 9.4. Set them out in the reasoning alongside the final number, naming in one line each the discretisation relations you used and the printed relation, if any, that the continuum definitions override.

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

Implement **all 9 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_grid_and_layer_profiles.py

Goal
----
Build the non-uniform node grid of the injection-layer/transport-layer stack and the piecewise-constant material profiles carried on it.

```python
import numpy as np


def grid_and_layer_profiles(n_hil: int, n_htl: int, dz_hil: float, dz_htl: float,
                            EV_hil: float, EV_htl: float, EC_hil: float, EC_htl: float,
                            NA_hil: float, NA_htl: float, ni_hil: float, ni_htl: float,
                            er_hil: float, er_htl: float) -> np.ndarray:
    '''Node positions and per-node material profiles of the HIL/HTL stack.

    Parameters
    ----------
    n_hil : int
        Number of HIL intervals; nodes 0..n_hil lie in the HIL. Must be at least 1.
    n_htl : int
        Number of HTL intervals, the junction interval included. Must be at least 1.
    dz_hil : float
        HIL grid spacing in cm, positive.
    dz_htl : float
        HTL grid spacing in cm, positive; also the width of the junction interval.
    EV_hil, EV_htl : float
        Flat-band valence levels of the HIL and HTL in eV.
    EC_hil, EC_htl : float
        Flat-band conduction levels of the HIL and HTL in eV.
    NA_hil, NA_htl : float
        Acceptor doping densities in cm^-3, positive.
    ni_hil, ni_htl : float
        Intrinsic carrier densities in cm^-3, positive.
    er_hil, er_htl : float
        Relative permittivities, positive.

    Returns
    -------
    out : np.ndarray
        Array of shape (6, N) with N = n_hil + n_htl + 1. Row 0 holds the node positions z in cm
        (z[0] = 0); rows 1 to 5 hold EV0, EC0, NA, ni and er at every node.

    Raises
    ------
    ValueError
        If n_hil or n_htl is not an integer of at least 1, if a spacing is not positive, or if any
        doping density, intrinsic density or relative permittivity is not positive.
    '''
    return out
```

### Step 2

02_equilibrium_densities.py

Goal
----
Set the equilibrium starting densities of holes and electrons on every node from the local doping and intrinsic density.

```python
import numpy as np


def equilibrium_densities(NA: np.ndarray, ni: np.ndarray) -> np.ndarray:
    '''Equilibrium hole and electron densities of a p-type stack.

    Parameters
    ----------
    NA : np.ndarray
        Array of shape (N,) of acceptor doping densities in cm^-3, all positive.
    ni : np.ndarray
        Array of shape (N,) of intrinsic carrier densities in cm^-3, all non-negative.

    Returns
    -------
    densities : np.ndarray
        Array of shape (2, N); row 0 the hole densities, row 1 the electron densities, in cm^-3.

    Raises
    ------
    ValueError
        If NA and ni are not one-dimensional arrays of the same length, if any NA is not positive,
        or if any ni is negative.
    '''
    return densities
```

### Step 3

03_poisson_potential.py

Goal
----
Solve Poisson's equation for the electrostatic potential on the non-uniform grid from the current space charge.

```python
import numpy as np


def poisson_potential(p: np.ndarray, n: np.ndarray, NA: np.ndarray, er: np.ndarray,
                      z: np.ndarray, phi_left: float, phi_right: float) -> np.ndarray:
    '''Electrostatic potential on a non-uniform grid with Dirichlet contacts.

    Parameters
    ----------
    p, n : np.ndarray
        Arrays of shape (N,) of hole and electron densities in cm^-3.
    NA : np.ndarray
        Array of shape (N,) of acceptor doping densities in cm^-3.
    er : np.ndarray
        Array of shape (N,) of relative permittivities, all positive.
    z : np.ndarray
        Array of shape (N,) of strictly increasing node positions in cm.
    phi_left, phi_right : float
        Potentials in V imposed at node 0 and node N - 1.

    Returns
    -------
    phi : np.ndarray
        Array of shape (N,) of nodal potentials in V; phi[0] = phi_left and phi[-1] = phi_right.

    Raises
    ------
    ValueError
        If the five arrays are not one-dimensional with a common length of at least 3, if z is not
        strictly increasing, or if any relative permittivity is not positive.
    '''
    return phi
```

### Step 4

04_band_edge_fields.py

Goal
----
Evaluate the valence- and conduction-band driving fields at every interval midpoint of the non-uniform grid.

```python
import numpy as np


def band_edge_fields(EV0: np.ndarray, EC0: np.ndarray, phi: np.ndarray, z: np.ndarray) -> np.ndarray:
    '''Band-edge driving fields at the interval midpoints.

    Parameters
    ----------
    EV0, EC0 : np.ndarray
        Arrays of shape (N,) of flat-band valence and conduction levels in eV.
    phi : np.ndarray
        Array of shape (N,) of electrostatic potentials in V.
    z : np.ndarray
        Array of shape (N,) of strictly increasing node positions in cm.

    Returns
    -------
    F : np.ndarray
        Array of shape (2, N-1); row 0 the valence-band field and row 1 the conduction-band field at
        each midpoint, in V/cm, positive where the corresponding band edge rises with z.

    Raises
    ------
    ValueError
        If the four arrays are not one-dimensional with a common length of at least 2, or if z is not
        strictly increasing.
    '''
    return F
```

### Step 5

05_midpoint_densities.py

Goal
----
Supply the carrier densities that enter the drift currents at the interval midpoints, under either the conventional or the field-selected treatment of the junction.

```python
import numpy as np


def midpoint_densities(p: np.ndarray, n: np.ndarray, FV: np.ndarray, FC: np.ndarray,
                       n_hil: int, scheme: str) -> np.ndarray:
    '''Carrier densities at the interval midpoints.

    Parameters
    ----------
    p, n : np.ndarray
        Arrays of shape (N,) of nodal hole and electron densities in cm^-3.
    FV, FC : np.ndarray
        Arrays of shape (N-1,) of valence- and conduction-band driving fields at the midpoints in V/cm.
    n_hil : int
        Index of the last injection-layer node; the junction midpoint lies between nodes n_hil and
        n_hil + 1 and is midpoint index n_hil. Requires 0 <= n_hil <= N - 2.
    scheme : str
        "mean" for the arithmetic mean at every midpoint, or "field" for the field-selected value at
        the junction midpoint and the arithmetic mean elsewhere.

    Returns
    -------
    mid : np.ndarray
        Array of shape (2, N-1); row 0 the midpoint hole densities, row 1 the midpoint electron
        densities, in cm^-3.

    Raises
    ------
    ValueError
        If p and n are not one-dimensional of a common length of at least 2, if FV or FC does not have
        length N - 1, if n_hil is outside 0..N-2, or if scheme is neither "mean" nor "field".
    '''
    return mid
```

### Step 6

06_drift_diffusion_currents.py

Goal
----
Evaluate the hole and electron drift-diffusion current densities on every interval of the non-uniform grid.

```python
import numpy as np


def drift_diffusion_currents(p: np.ndarray, n: np.ndarray, p_mid: np.ndarray, n_mid: np.ndarray,
                             FV: np.ndarray, FC: np.ndarray, mu_p: np.ndarray, mu_n: np.ndarray,
                             z: np.ndarray, T: float) -> np.ndarray:
    '''Drift-diffusion current densities on the grid intervals.

    Parameters
    ----------
    p, n : np.ndarray
        Arrays of shape (N,) of nodal hole and electron densities in cm^-3.
    p_mid, n_mid : np.ndarray
        Arrays of shape (N-1,) of midpoint hole and electron densities in cm^-3.
    FV, FC : np.ndarray
        Arrays of shape (N-1,) of valence- and conduction-band driving fields in V/cm.
    mu_p, mu_n : np.ndarray
        Arrays of shape (N,) of nodal hole and electron mobilities in cm^2 V^-1 s^-1.
    z : np.ndarray
        Array of shape (N,) of strictly increasing node positions in cm.
    T : float
        Temperature in K, positive.

    Returns
    -------
    J : np.ndarray
        Array of shape (2, N-1); row 0 the hole current density, row 1 the electron current density,
        in A cm^-2.

    Raises
    ------
    ValueError
        If the nodal arrays do not share a one-dimensional shape of length at least 2, if a midpoint
        array does not have length N - 1, if z is not strictly increasing, or if T is not positive.
    '''
    return J
```

### Step 7

07_recombination_rate.py

Goal
----
Evaluate the non-radiative recombination sink on every node from trap-assisted and Auger channels.

```python
import numpy as np


def recombination_rate(p: np.ndarray, n: np.ndarray, ni: np.ndarray, tau_p: float, tau_n: float,
                       Cp: float, Cn: float) -> np.ndarray:
    '''Total Shockley-Read-Hall plus Auger recombination rate at every node.

    Parameters
    ----------
    p, n : np.ndarray
        Arrays of shape (N,) of hole and electron densities in cm^-3.
    ni : np.ndarray
        Array of shape (N,) of intrinsic carrier densities in cm^-3, non-negative.
    tau_p, tau_n : float
        Hole and electron SRH lifetimes in s, positive.
    Cp, Cn : float
        Hole and electron Auger capture probabilities in cm^6 s^-1, non-negative.

    Returns
    -------
    U : np.ndarray
        Array of shape (N,) of recombination rates in cm^-3 s^-1 (negative for net generation).

    Raises
    ------
    ValueError
        If p, n and ni are not one-dimensional arrays of a common length, if a lifetime is not positive,
        if an Auger probability or any ni is negative, or if the SRH denominator vanishes at some node.
    '''
    return U
```

### Step 8

08_continuity_update.py

Goal
----
Advance the nodal hole and electron densities by one explicit time step of the flux-conservative continuity equations.

```python
import numpy as np


def continuity_update(p: np.ndarray, n: np.ndarray, Jp: np.ndarray, Jn: np.ndarray, U: np.ndarray,
                      z: np.ndarray, dt: float) -> np.ndarray:
    '''One explicit continuity step with fixed contact densities.

    Parameters
    ----------
    p, n : np.ndarray
        Arrays of shape (N,) of nodal hole and electron densities in cm^-3.
    Jp, Jn : np.ndarray
        Arrays of shape (N-1,) of hole and electron current densities on the intervals in A cm^-2.
    U : np.ndarray
        Array of shape (N,) of nodal recombination rates in cm^-3 s^-1.
    z : np.ndarray
        Array of shape (N,) of strictly increasing node positions in cm.
    dt : float
        Time step in s, positive.

    Returns
    -------
    new : np.ndarray
        Array of shape (2, N); row 0 the updated hole densities, row 1 the updated electron densities,
        with the entries at node 0 and node N - 1 equal to the supplied values.

    Raises
    ------
    ValueError
        If the nodal arrays do not share a one-dimensional shape of length at least 3, if a current
        array does not have length N - 1, if z is not strictly increasing, or if dt is not positive.
    '''
    return new
```

### Step 9

09_interfacial_hole_density.py

Goal
----
Run the explicit interfacial transient of the injection-layer/transport-layer stack from equilibrium and return the hole density on the transport side of the junction.

```python
import numpy as np


def interfacial_hole_density(n_hil: int, n_htl: int, dz_hil: float, dz_htl: float, dt: float,
                             n_steps: int, T: float, V: float,
                             EV_hil: float, EV_htl: float, EC_hil: float, EC_htl: float,
                             NA_hil: float, NA_htl: float, ni_hil: float, ni_htl: float,
                             er_hil: float, er_htl: float, mup_hil: float, mup_htl: float,
                             mun_hil: float, mun_htl: float, tau_p: float, tau_n: float,
                             Cp: float, Cn: float, scheme: str) -> float:
    '''Hole density at the first transport-layer node after an explicit transient.

    Parameters
    ----------
    n_hil, n_htl : int
        Numbers of injection-layer and transport-layer intervals (the junction interval counts in the
        transport layer); each at least 1.
    dz_hil, dz_htl : float
        Injection-layer and transport-layer grid spacings in cm, positive.
    dt : float
        Time step in s, positive.
    n_steps : int
        Number of explicit steps, at least 1.
    T : float
        Temperature in K, positive.
    V : float
        Potential in V applied at the last node; node 0 is held at 0 V.
    EV_hil, EV_htl, EC_hil, EC_htl : float
        Flat-band valence and conduction levels of the two layers in eV.
    NA_hil, NA_htl, ni_hil, ni_htl : float
        Acceptor doping and intrinsic densities of the two layers in cm^-3, positive.
    er_hil, er_htl : float
        Relative permittivities of the two layers, positive.
    mup_hil, mup_htl, mun_hil, mun_htl : float
        Hole and electron mobilities of the two layers in cm^2 V^-1 s^-1.
    tau_p, tau_n : float
        SRH lifetimes in s, positive.
    Cp, Cn : float
        Auger capture probabilities in cm^6 s^-1, non-negative.
    scheme : str
        "field" for the field-selected junction density, "mean" for the arithmetic mean.

    Returns
    -------
    p_int : float
        Hole density at node n_hil + 1 after n_steps steps, in units of 1e18 cm^-3.

    Raises
    ------
    ValueError
        If n_steps is not an integer of at least 1, if dt is not positive, or if any argument violates
        the requirements of the grid, equilibrium, potential, field, midpoint, current, recombination
        or update calculations it is passed to.
    '''
    return p_int
```

# Up to what forward bias the injected-carrier capacitance of an undoped thin-film diode can be trusted

## Background

Capacitance-voltage measurements are the standard tool for reading the built-in potential of a
semiconductor diode, and Mott-Schottky analysis, which extracts it from the intercept of the
inverse-square capacitance against voltage, is its standard evaluation. The analysis assumes a
depletion region in a uniformly doped layer thick enough to hold it. Thin-film diodes based on
undoped organic or perovskite semiconductors violate both assumptions: the active layer is fully
depleted of doping-induced carriers and behaves as a metal-insulator-metal structure, and the
carriers that determine the capacitance are injected from the contacts, where they pile up in
accumulation regions a few nanometres thick. Such devices show a capacitance above the geometric
value that rises with forward bias and saturates to the geometric value in reverse bias, and a
Mott-Schottky plot with no linear region.

Injected carriers accumulate in thin, highly conductive layers at each contact, and the resulting
energy-level bending is itself modulated by the applied voltage. This ties the charge stored on the
electrodes to the space charge inside the layer, so the low-frequency capacitance of such a device
depends on the contact properties as well as on the layer geometry, and a linear-fit evaluation of
capacitance-voltage data can be constructed on that basis.

The reference description of the same device is the steady-state drift-diffusion model with
Boltzmann contacts, which resolves the carrier profiles without the approximations of a closed-form
treatment and therefore serves as the accuracy standard for one. It also supplies synthetic
capacitance-voltage data for contacts that are nearly but not exactly ohmic.

## Problem

Thin-film diodes made of undoped organic or perovskite semiconductors are metal-insulator-metal
devices: the active layer holds no doping-induced carriers, every carrier is injected from a
contact, and the low-frequency capacitance exceeds the geometric value because injected carriers
pile up in thin accumulation regions at the electrodes. Mott-Schottky analysis, which reads the
built-in potential from the intercept of an inverse-square capacitance plot, does not apply to
such devices. The source of this task derives instead a closed-form capacitance-voltage law from
the injected carriers alone: each accumulation region is described by its majority carrier only,
the bulk field is uniform, the energy-level bending at the contacts lowers the built-in potential
to a voltage-dependent effective value, and the capacitance follows by differentiating the charge
displaced through the external circuit. From the law the source proposes a linear-fit protocol
that extracts the built-in potential from measured capacitance-voltage curves, and it validates
the law against drift-diffusion simulations for biases sufficiently far below the built-in
potential. It does not say where the law stops being accurate for a given device, nor how large
the bias of its extraction protocol is when the contacts are only nearly ohmic. That is the
question of this task.

Your task is to build the source's closed-form capacitance model, build the exact steady-state
drift-diffusion reference of the same diode with an exact derivative of its displaced charge, apply
the source's extraction protocol to the exact model, and report the forward bias at which the
closed-form capacitance of the design-point diode deviates from the exact one by one percent.

## The configuration

The active layer of thickness d and relative permittivity eps lies between a hole-injecting
anode at x = 0 and an electron-injecting cathode at x = d. The semiconductor has effective
densities of states N_c and N_v, energy gap E_g, mobilities mu_n and mu_p, and bimolecular
recombination with the reduced Langevin coefficient gamma = zeta q (mu_n + mu_p) / (eps eps0).
The contacts inject Boltzmann carrier densities set by the injection barriers: at the anode
p_an = N_v exp(-phi_an / kT) and n_an = N_c exp(-(E_g - phi_an) / kT), at the cathode
n_cat = N_c exp(-phi_cat / kT) and p_cat = N_v exp(-(E_g - phi_cat) / kT), so that n p = n_i^2 at
both contacts, with n_i^2 = N_c N_v exp(-E_g / kT). The nominal built-in potential is
V_bi,0 = (E_g - phi_an - phi_cat) / q, the electrostatic potential is psi(0) = 0 and
psi(d) = V_bi,0 - V for the applied voltage V (forward bias positive), and the geometric
capacitance is C_geo = eps eps0 / d. The temperature is T. Four designs are run, as rows (d in
nm, eps, T in K, N_c and N_v in 1/m^3, E_g in eV, phi_an and phi_cat in eV, mu_n and mu_p in
m^2/(V s), zeta):

    design 1 (the design point)   100   3.5   300   1e26   1e26   1.5   0.02   0.02   1e-8   5e-9   0.1
    design 2                      200   3.5   300   1e26   1e26   1.5   0.02   0.02   1e-8   5e-9   0.1
    design 3                      100   3.5   300   1e26   1e26   1.5   0.02   0.50   1e-8   5e-9   0.1
    design 4                      100   3.5   300   1e26   1e26   1.5   0.25   0.25   1e-8   5e-9   0.1

Use q = 1.602176634e-19 C, k = 1.380649e-23 J/K and eps0 = 8.8541878128e-12 F/m; report
capacitances relative to C_geo, C_geo itself in nF/cm^2, lengths in nm, fields in MV/m.

Settings of the calculation, not choices: the drift-diffusion reference is solved on a uniform grid
of N = 800 intervals; the deviation level that defines the validity threshold is 1 percent; the
threshold is bracketed by marching from zero bias in steps of 0.05 V; the extraction protocol fits
the exact inverse excess capacitance at the five voltages -0.2, -0.1, 0, 0.1 and 0.2 V; the probe
bias of each design is V_probe = V_bi(0) / 2, one half of its effective built-in potential at zero
bias; the field of the closed-form model is reported at mid-layer, x = d / 2.

## The closed-form model of the source and the conventions that fix its numbers

The Debye screening lengths of the injected carriers at the contacts are
lambda_an = sqrt(2 eps eps0 kT / (q^2 p_an)) and lambda_cat = sqrt(2 eps eps0 kT / (q^2 n_cat)).
For a voltage below V_bi,0 the layer splits into a hole-dominated region next to the anode, in
which p(x) = p_an exp(-q phi(x) / kT) with phi(0) = 0 and the electron density is neglected, and
an electron-dominated region next to the cathode, in which n(x) = n_cat exp(q [phi(x) - phi(d)] / kT)
and the hole density is neglected; far from both contacts the space charge vanishes and the field
is the uniform bulk field E_bulk. Solve Poisson's equation in each region with the field tending
to E_bulk inside the layer: the field follows a hyperbolic-cotangent law in the distance from the
contact, with an offset fixed by the Debye length of that contact (derive it). Integrate the field
across the layer with V - V_bi,0 = integral of E dx and write E_bulk = (V - V_bi(V)) / d, which
defines the effective built-in potential V_bi(V) through an implicit equation: V_bi(V) equals
V_bi,0 minus, for each contact, the energy-level bending 2 kT/q times the natural logarithm of one
half of [1 + sqrt(1 + (2 kT d / (q [V_bi(V) - V] lambda))^2)] (derive it). Solve the implicit
equation by fixed-point iteration from V_bi,0 to machine precision. The contact factors are
eta_an = 1 - 1 / sqrt(1 + (2 kT d / (q [V_bi(V) - V] lambda_an))^2) and likewise eta_cat, the
effective width of an accumulation region is Delta_w = 2 kT d eta / (q [V_bi(V) - V]), and the
crossover x* is the position at which the two field branches take the same value. The two
branches, with the potential obtained by integrating each from its contact, give the field and the
relative carrier densities p / p_an and n / n_cat at any position.

The charge displaced through the external circuit is Q = eps eps0 E_bulk, and the capacitance per
unit area is C = dQ/dV; obtain it in closed form by implicit differentiation of the equation for
V_bi(V) (derive it), and write the relative excess capacitance Delta C / C_geo = C / C_geo - 1
and its inverse. Also derive the weak-injection limit of the excess capacitance for two weakly
injecting contacts (both Debye lengths much larger than d), in which the carrier profiles are
purely exponential.

## The exact reference and what is measured against it

The exact reference is the steady-state drift-diffusion model of the same diode: Poisson's
equation eps eps0 psi'' = -q (p - n), the steady continuity equations dJ_n/dx = q R and
dJ_p/dx = -q R with R = gamma (n p - n_i^2), and the currents J_n = -q mu_n n psi' + q D_n n' and
J_p = -q mu_p p psi' - q D_p p' with D = mu kT/q, with the contact densities and potentials
above. Discretise on the uniform grid of N intervals (nodes x_i = i d / N, spacing h = d / N):
Poisson with the three-point second difference at every interior node, R evaluated at the nodes,
and the currents on every interval by the Scharfetter-Gummel fluxes
J_n = (q D_n / h) [n_{i+1} B(delta_i) - n_i B(-delta_i)] and
J_p = (q D_p / h) [p_i B(delta_i) - p_{i+1} B(-delta_i)], with delta_i = (psi_{i+1} - psi_i) q / kT,
B(x) = x / (exp(x) - 1), and the divergence at a node equal to the difference of the two interval
fluxes divided by h. Solve the discrete nonlinear system to machine precision (every scaled
residual below 1e-12; Newton's method with a line search in psi, ln n and ln p, started from the
zero-bias Poisson-Boltzmann equilibrium and continued in the voltage in steps of at most 0.05 V);
its converged solution is unique and is the state reported as psi q / kT, ln n and ln p at the
nodes. The charge displaced through the external circuit of the exact model is
Q = C_geo (V - V_bi,0) + q times the integral over the layer of [(x / d) p(x) + (1 - x / d) n(x)],
the Ramo-Shockley weighting of every injected carrier by its distance from the contact that
injected it, which is equivalent to the source's expression in terms of the electrode charges and
the mean carrier densities; evaluate the integral by the trapezoidal rule on the grid. The exact
capacitance is dQ/dV of the steady state, obtained exactly by differentiating the discrete
equations with respect to V (the derivative of the state solves the linear system with the
Jacobian of the discrete equations and the derivative of the boundary condition
psi(d) = V_bi,0 - V) and applying the chain rule to Q; finite differences between separately
converged states are not exact enough and must not be used. The ratio of Q to eps eps0 times the
field at mid-layer, E = -(psi_{N/2+1} - psi_{N/2}) / h, tests the source's relation
Q = eps eps0 E_bulk in the exact model.

The extraction protocol of the source is applied to the exact model: form y(V) = (Delta C / C_geo)^-1
from the exact capacitance at the fit voltages, fit the straight line y = S (V* - V) by ordinary
least squares, and read eta_ext = q C(0) / (2 S kT C_geo) and V_bi,ext = V* + S^-1 [1 + C(0) / C_geo]
with the exact zero-bias capacitance, exactly as the source reads its limiting cases (both
contacts ohmic, or one ohmic and one non-injecting). Then derive the exact first-order expansion
of the closed-form inverse excess capacitance about V = 0 for arbitrary contacts,
y(V) = y(0) - S_lin V + ..., using dV_bi/dV = 1 - C/C_geo and the voltage dependence of both
contact factors, and evaluate the source's slope factor f_S = eta^2 - 6 (eta + 1/eta) + 12 at
eta(0), which the source gives for devices with at least one ohmic contact.

The validity threshold of a design is the forward bias V_thr at which the deviation
delta(V) = 100 (C_closed-form(V) / C_exact(V) - 1) in percent first reaches the level, located by
marching from zero bias in the given step until the level is first reached and then bracketing the
crossing to 1e-12 V inside that interval; report with it the remaining effective built-in potential
V_bi(V_thr) - V_thr and the bulk-field parameter z = q [V_bi(V_thr) - V_thr] / (2 kT).

Run the full chain for the four designs and assemble one audit table with one row per design and
the columns [V_bi,0, V_bi(0), eta(0), C_closed-form(0) / C_geo, C_exact(0) / C_geo, delta(0),
V_probe, C_closed-form(V_probe) / C_geo, C_exact(V_probe) / C_geo, delta(V_probe),
Q / (eps eps0 E_mid) at the probe, the closed-form field at mid-layer at the probe, x* at zero bias,
Delta C_weak / C_geo at the probe, Delta C / Delta C_weak at the probe, S, eta_ext, V_bi,ext,
V_bi,ext - V_bi(0), V_thr, V_bi(V_thr) - V_thr, z at the threshold]. The head of the table is
V_thr of design 1.

## What to report

Report, as the final answer, the validity threshold of the design-point diode, V_thr of design 1 in
volts, to six significant figures.

Your reasoning should also report, as evidence that the chain was executed: for the design point
its effective built-in potential V_bi(0) and contact factor eta(0); the effective width of the
anode accumulation region at zero bias; the closed-form and the exact zero-bias capacitance
relative to C_geo; the probe bias and the closed-form and exact capacitance there, with the
deviation delta(V_probe); the ratio Q / (eps eps0 E_mid) and the closed-form field at mid-layer at the probe;
the ratio Delta C / Delta C_weak at the probe; the fitted slope S, the extracted contact factor
eta_ext, the extracted V_bi,ext and its bias with respect to V_bi(0); the first-order slope S_lin and the source's factor f_S at eta(0);
the remaining effective built-in potential and the bulk-field parameter z at the threshold; the
zero-bias crossover position of the two field branches of designs 1 and 3; for design 2 its
V_bi(0), eta(0), its zero-bias closed-form capacitance with its deviation from the exact value,
and V_thr; for design 3 its eta(0), its fitted slope and extraction bias, and V_thr; for design 4
its V_thr and its ratio Delta C / Delta C_weak at the probe. Say, in a sentence or two, why the closed-form
model fails as the bias approaches the built-in potential and how the failure is governed by the
bulk-field parameter, and compare the threshold you find with the validity bound the source states.
Say why the extraction protocol of the source is biased for the design point although its
contacts are nearly ohmic. Say why the displaced charge must be differentiated exactly rather than
by finite differences of two converged states. Report, with a citation, the built-in potentials
and the contact factor that the source extracts from its two experimental organic solar cells
together with the geometric capacitances it assumes, the source's statement of the voltage range
over which its analytical expressions are valid and the limiting values of eta it gives for two
ohmic contacts and for a hole-only diode, and the power law the source gives for the inverse
excess capacitance of a diode with two weakly injecting contacts, and set each against your
results.

The quantities listed above are the reported result, not intermediate bulk output.

## Output format

```
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Inside <reasoning>, include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
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

diode_parameters

Goal
----
Convert the physical description of an undoped thin-film (metal-insulator-metal) diode into the parameter vector used by the rest of the chain. The active layer of thickness d (nm) and relative permittivity eps is sandwiched between a hole-injecting anode at x = 0 and an electron-injecting cathode at x = d; the temperature is T (K); the effective densities of states are N_c and N_v (1/m^3), the energy gap E_g (eV), the injection barriers phi_an at the anode and phi_cat at the cathode (eV), the mobilities mu_n and mu_p (m^2/(V s)) and the reduced Langevin factor zeta. The carrier densities at the contacts follow Boltzmann statistics with the contact Fermi level: p_an = N_v exp(-phi_an / kT) and the corresponding electron density n_an = N_c exp(-(E_g - phi_an) / kT) at the anode, n_cat = N_c exp(-phi_cat / kT) and p_cat = N_v exp(-(E_g - phi_cat) / kT) at the cathode, so that n p = n_i^2 = N_c N_v exp(-E_g / kT) at both contacts. The nominal built-in potential is V_bi,0 = (E_g - phi_an - phi_cat) / q, the geometric capacitance C_geo = eps eps0 / d, the Debye screening lengths of the injected carriers at the contacts are lambda_an = sqrt(2 eps eps0 kT / (q^2 p_an)) and lambda_cat = sqrt(2 eps eps0 kT / (q^2 n_cat)), and the bimolecular recombination coefficient is the reduced Langevin value gamma = zeta q (mu_n + mu_p) / (eps eps0). Use q = 1.602176634e-19 C, k = 1.380649e-23 J/K, eps0 = 8.8541878128e-12 F/m. Raise ValueError if d, eps, T, N_c or N_v is not positive, if a barrier is negative or the barriers together reach the gap, or if a mobility or zeta is not positive.

```python
def diode_parameters(thickness: float, permittivity: float, temperature: float, nc: float, nv: float, gap: float, barrier_anode: float, barrier_cathode: float, mobility_n: float, mobility_p: float, langevin_factor: float) -> "np.ndarray":
    """Convert the physical description of an undoped thin-film (metal-insulator-metal) diode
    into the parameter vector used by the rest of the chain. A (14,) float64 array [kT/q in
    V, V_bi,0 in V, C_geo in nF/cm^2, lambda_an in nm, lambda_cat in nm, ln p_an, ln n_cat,
    ln n_an, ln p_cat (natural logarithms of the densities in 1/m^3), gamma in units of
    1e-18 m^3/s, d in nm, eps, mu_n, mu_p in m^2/(V s)].
    Parameters
    ----------
    thickness : float
        Active-layer thickness d in nm.
    permittivity : float
        Relative permittivity of the active layer.
    temperature : float
        Temperature in K.
    nc : float
        Conduction-band density of states in 1/m^3.
    nv : float
        Valence-band density of states in 1/m^3.
    gap : float
        Transport gap in eV.
    barrier_anode : float
        Hole injection barrier at the anode in eV.
    barrier_cathode : float
        Electron injection barrier at the cathode in eV.
    mobility_n : float
        Electron mobility in m^2/(V s).
    mobility_p : float
        Hole mobility in m^2/(V s).
    langevin_factor : float
        Reduction factor of the Langevin recombination coefficient.

    Raises
    ------
    ValueError
        If thickness, permittivity, temperature or either density of states is
        not positive; if a barrier is negative or the two barriers together
        reach the gap; or if a mobility or the Langevin factor is not positive.
    """
    return None
```

### Step 2

effective_built_in_potential

Goal
----
Compute the effective built-in potential of the injected-carrier model at an applied voltage V below V_bi,0, together with the quantities derived from it. Return V_bi(V) in V, the contact factors eta_an and eta_cat, the bulk field E_bulk in MV/m (negative for V < V_bi), the effective widths of the accumulation regions at the anode and at the cathode in nm, and the crossover position x* in nm at which the field branches of the two contacts take the same value. The relation that defines V_bi(V) is iterated to machine precision, stopping when the change falls below 1e-15 V. Raise ValueError if par does not have 14 entries, if V is not below V_bi,0, or if the iteration leaves the physical domain V_bi(V) > V.

```python
def effective_built_in_potential(voltage: float, par: "np.ndarray") -> "np.ndarray":
    """Derive the effective built-in potential of the injected-carrier model at an applied
    voltage V below V_bi,0. A (7,) float64 array [V_bi(V) in V, eta_an, eta_cat, E_bulk in
    MV/m, Delta_w_an in nm, Delta_w_cat in nm, x* in nm].
    Parameters
    ----------
    voltage : float
        Applied bias V in V, below the nominal built-in potential V_bi,0.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If par does not have 14 entries; if V is not below V_bi,0; or if the
        iteration leaves the physical domain V_bi(V) > V.
    """
    return None
```

### Step 3

field_and_carrier_profiles

Goal
----
Evaluate the injected-carrier model's field and carrier profiles at the positions x (nm, 0 <= x <= d) for an applied voltage V. Use the hyperbolic-cotangent field branch of the anode region for x up to the crossover x* of the previous step and the branch of the cathode region beyond it, both with the bulk field and the effective built-in potential of that step. Return, per position, the field in MV/m, the hole density relative to its contact value p / p_an in the hole-dominated region (zero elsewhere) and the electron density relative to its contact value n / n_cat in the electron-dominated region (zero elsewhere), the densities following from the Boltzmann relation with the electrostatic potential obtained by integrating the field from the respective contact. Raise ValueError if par does not have 14 entries or if a position lies outside the layer.

```python
def field_and_carrier_profiles(positions: "np.ndarray", voltage: float, par: "np.ndarray") -> "np.ndarray":
    """Evaluate the injected-carrier model's field and carrier profiles at the positions x (nm,
    0 <= x <= d) for an applied voltage V. Use the hyperbolic-cotangent field branch of the
    anode region for x up to the crossover x* of the previous step and the branch of the
    cathode region beyond it, both with the bulk field and the effective built-in potential
    of that step. An (m, 3) float64 array with columns [E in MV/m, p / p_an or 0, n / n_cat
    or 0] for the m positions.
    Parameters
    ----------
    positions : np.ndarray
        Positions x in nm at which to evaluate the profiles, 0 <= x <= d.
    voltage : float
        Applied bias V in V.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If par does not have 14 entries, or if any position lies outside the
        active layer 0 <= x <= d.
    """
    return None
```

### Step 4

analytic_capacitance

Goal
----
Compute the low-frequency capacitance of the injected-carrier model at the voltage V. Return C / C_geo, the relative excess capacitance Delta C / C_geo = C / C_geo - 1, its inverse (Delta C / C_geo)^-1, the total contact factor eta(V) = eta_an + eta_cat, and, for comparison, the relative excess capacitance Delta C_weak / C_geo of the weak-injection limit, the limit in which both Debye lengths are much larger than d. Raise ValueError if par does not have 14 entries, if V is not below V_bi,0, or if the closed form gives a non-positive capacitance.

```python
def analytic_capacitance(voltage: float, par: "np.ndarray") -> "np.ndarray":
    """Derive the low-frequency capacitance of the injected-carrier model at the voltage V, in
    closed form from the effective built-in potential and the contact factors of the previous
    step. A (5,) float64 array [C / C_geo, Delta C / C_geo, (Delta C / C_geo)^-1, eta(V),
    Delta C_weak / C_geo].
    Parameters
    ----------
    voltage : float
        Applied bias V in V, below the nominal built-in potential V_bi,0.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If par does not have 14 entries, if V is not below V_bi,0, or if the
        closed form gives a non-positive capacitance.
    """
    return None
```

### Step 5

drift_diffusion_state

Goal
----
Solve the exact steady-state drift-diffusion model of the same diode at the voltage V on a uniform grid of N intervals (nodes x_i = i d / N, i = 0..N) and return the converged state. The unknowns are the electrostatic potential psi, the electron density n and the hole density p. Boundary values: psi(0) = 0, psi(d) = V_bi,0 - V, and the Boltzmann contact densities n_an, p_an, n_cat and p_cat of the parameter step. The discrete nonlinear system is solved to machine precision, with every scaled residual below 1e-12; the converged discrete solution is unique. Return, per node, psi q / kT, ln n and ln p, with densities in 1/m^3. Raise ValueError if par does not have 14 entries, if N is not an even integer of at least 4, or if V is not below V_bi,0.

```python
def drift_diffusion_state(voltage: float, intervals: int, par: "np.ndarray") -> "np.ndarray":
    """Solve the exact steady-state drift-diffusion model of the same diode at the voltage V on
    a uniform grid of N intervals (nodes x_i = i d / N, i = 0..N) and return the converged
    state. An (N + 1, 3) float64 array with columns [psi q / kT, ln n, ln p] at the nodes
    x_i = i d / N.
    Parameters
    ----------
    voltage : float
        Applied bias V in V, below the nominal built-in potential V_bi,0.
    intervals : int
        Number N of grid intervals, an even integer of at least 4.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If par does not have 14 entries, if N is not an even integer of at
        least 4, or if V is not below V_bi,0.
    """
    return None
```

### Step 6

exact_capacitance

Goal
----
From a converged drift-diffusion state at the voltage V, compute the charge displaced through the external circuit and its exact derivative with respect to V, evaluating any integral over the layer with the trapezoidal rule on the grid. Return Q / C_geo in V, the capacitance C / C_geo, and the ratio of Q to eps eps0 times the field at the middle of the layer, taken as (psi_{N/2+1} - psi_{N/2}) / h with the sign of E = -psi'. The capacitance must be the exact derivative of the discrete steady state, not a finite difference of Q between separately converged states. Raise ValueError if the state is not an (N + 1, 3) array with the contact potentials of this bias or if its discrete residuals exceed 1e-9.

```python
def exact_capacitance(state: "np.ndarray", voltage: float, par: "np.ndarray") -> "np.ndarray":
    """From a converged drift-diffusion state at the voltage V, evaluate the charge displaced
    through the external circuit and its exact derivative with respect to V. The displaced
    charge per unit area is Q = C_geo (V - V_bi,0) + q times the integral over the layer of
    [(x / d) p(x) + (1 - x / d) n(x)] (the Ramo-Shockley weighting of the injected carriers
    by their distance from the contact that injected them, which is equivalent to the
    source's expression in terms of the electrode charges and the mean carrier densities);
    evaluate the integral with the trapezoidal rule on the grid. A (3,) float64 array [Q /
    C_geo in V, C / C_geo, Q / (eps eps0 E_mid)].
    Parameters
    ----------
    state : np.ndarray
        The converged (N + 1, 3) drift-diffusion state at this bias.
    voltage : float
        Applied bias V in V at which the state was converged.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If state is not an (N + 1, 3) array, if it does not carry the contact
        potentials of this bias, or if its discrete residuals exceed 1e-9.
    """
    return None
```

### Step 7

built_in_extraction

Goal
----
Apply the source's built-in potential extraction protocol to the exact model and derive the first-order expansion it rests on. Compute the exact capacitance at each of the fit voltages (strictly increasing, below V_bi,0) from converged drift-diffusion states on the grid of N intervals, form y(V) = (Delta C / C_geo)^-1 with Delta C = C - C_geo, and fit the straight line y = S (V* - V) by ordinary least squares in V. Read the contact factor and the built-in potential the way the source does for its limiting cases (both contacts ohmic or one ohmic and one non-injecting): eta_ext = q C(0) / (2 S kT C_geo) and V_bi,ext = V* + S^-1 [1 + C(0) / C_geo], with C(0) the exact zero-bias capacitance. Then derive the exact first-order expansion of the analytic inverse excess capacitance of the injected-carrier model about V = 0 for arbitrary contacts, y(V) = y(0) - S_lin V + ..., using dV_bi/dV = 1 - C/C_geo and the voltage dependence of both contact factors, and return S_lin and V*_lin = y(0) / S_lin, together with the source's slope factor f_S = eta^2 - 6 (eta + 1/eta) + 12 evaluated at eta(0), which the source gives for devices with at least one ohmic contact. Raise ValueError if fewer than three fit voltages are given, if they are not strictly increasing or not below V_bi,0, if par does not have 14 entries, or if N is not an even integer of at least 4.

```python
def built_in_extraction(fit_voltages: "np.ndarray", intervals: int, par: "np.ndarray") -> "np.ndarray":
    """Apply the source's built-in potential extraction protocol to the exact model and derive
    the first-order expansion it rests on. An (8,) float64 array [S in 1/V, V* in V,
    eta_ext, V_bi,ext in V, V_bi,ext - V_bi(0) in V, S_lin in 1/V, V*_lin in V, f_S].
    Parameters
    ----------
    fit_voltages : np.ndarray
        At least three strictly increasing bias values in V at which the
        inverse excess capacitance is fitted, all below V_bi,0.
    intervals : int
        Number N of grid intervals, an even integer of at least 4.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.

    Raises
    ------
    ValueError
        If fit_voltages is not at least three strictly increasing values, if
        par does not have 14 entries, if N is not an even integer of at least
        4, or if any fit voltage is not below V_bi,0.
    """
    return None
```

### Step 8

validity_threshold

Goal
----
Locate the forward bias at which the injected-carrier model stops being accurate. Define the deviation delta(V) = 100 (C_analytic(V) / C_exact(V) - 1) in percent, with the closed-form capacitance of the injected-carrier model and the exact drift-diffusion capacitance on the grid of N intervals. Starting from V = 0, march in steps of march_step (V) until delta first reaches the level (percent), then locate the crossing delta(V) = level inside that interval to 1e-12 V (bisection or any bracketing root finder; the deviation grows monotonically with the bias in that interval). Return the threshold voltage V_thr, the remaining effective built-in potential V_bi(V_thr) - V_thr, and the bulk-field parameter z = q [V_bi(V_thr) - V_thr] / (2 kT) at the threshold (the ratio of the bulk potential drop to 2 kT/q that controls the overlap of the two accumulation regions). Raise ValueError if level or march_step is not positive, if the deviation is already above the level at zero bias, or if it never reaches the level before V_bi,0 - 4 kT/q.

```python
def validity_threshold(level: float, intervals: int, par: "np.ndarray", march_step: float) -> "np.ndarray":
    """Locate the forward bias at which the injected-carrier model stops being accurate. A (3,)
    float64 array [V_thr in V, V_bi(V_thr) - V_thr in V, z].
    Parameters
    ----------
    level : float
        Deviation level in percent that defines the threshold, positive.
    intervals : int
        Number N of grid intervals, an even integer of at least 4.
    par : np.ndarray
        The (14,) parameter vector returned by the parameter step.
    march_step : float
        Bias step in V used to march up from zero bias, positive.

    Raises
    ------
    ValueError
        If par does not have 14 entries, if level or march_step is not
        positive, if N is not an even integer of at least 4, if the deviation
        already exceeds the level at zero bias, or if it never reaches the
        level below the built-in potential.
    """
    return None
```

### Step 9

cv_audit

Goal
----
Run the complete chain for every diode design in design_table, a list of rows (d in nm, eps, T in K, N_c, N_v in 1/m^3, E_g in eV, phi_an, phi_cat in eV, mu_n, mu_p in m^2/(V s), zeta), on the grid of N intervals, and return one table. For each design convert the parameters, evaluate the effective built-in potential and the closed-form capacitance at zero bias and at the probe bias V_probe = V_bi(0) / 2, solve the exact drift-diffusion states at both biases and evaluate the exact capacitances, evaluate the field profile of the injected-carrier model at mid-layer x = d / 2 at the probe bias, the weak-injection excess capacitance at the probe bias, the built-in potential extraction over the fit voltages and the validity threshold at the given level with the given march step. Each design row holds the 22 columns [V_bi,0, V_bi(0), eta(0), C_analytic(0) / C_geo, C_exact(0) / C_geo, delta(0) in percent, V_probe, C_analytic(V_probe) / C_geo, C_exact(V_probe) / C_geo, delta(V_probe) in percent, Q / (eps eps0 E_mid) at the probe, E at mid-layer at the probe in MV/m, x* at zero bias in nm, Delta C_weak / C_geo at the probe, Delta C / Delta C_weak at the probe, S, eta_ext, V_bi,ext, V_bi,ext - V_bi(0), V_thr, V_bi(V_thr) - V_thr, z at the threshold]. Row 0 is the head row [V_thr of the first design, number of designs, N, level, zeros]. Raise ValueError if design_table is empty or a row does not have 11 entries, or if N is not an even integer of at least 4.

```python
def cv_audit(design_table: list, intervals: int, level: float, fit_voltages: "np.ndarray", march_step: float) -> "np.ndarray":
    """Run the complete chain for every diode design in design_table, a list of rows (d in nm,
    eps, T in K, N_c, N_v in 1/m^3, E_g in eV, phi_an, phi_cat in eV, mu_n, mu_p in m^2/(V
    s), zeta), on the grid of N intervals, and return one table. A (1 + n_designs, 22)
    float64 array; row 0 is the head row and row i the row of design i, with the columns
    listed in the description.
    Parameters
    ----------
    design_table : list
        Rows of 11 diode parameters, each row in the argument order of the
        parameter step.
    intervals : int
        Number N of grid intervals, an even integer of at least 4.
    level : float
        Deviation level in percent that defines the validity threshold.
    fit_voltages : np.ndarray
        Bias values in V at which the extraction protocol fits.
    march_step : float
        Bias step in V used to march up from zero bias.

    Raises
    ------
    ValueError
        If a design_table row does not hold the 11 diode parameters, or if N
        is not an even integer of at least 4.
    """
    return None
```

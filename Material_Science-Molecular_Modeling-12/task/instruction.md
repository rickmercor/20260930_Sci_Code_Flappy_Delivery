# How far the reference particle pressure of a mesoscopic liquid model moves once its structure is taken into account

## Background

Dissipative particle dynamics with energy conservation is the mesoscopic method of choice when a
fluid has to be simulated with thermal fluctuations, heat transport and a realistic equation of
state at once. Its generalised form, GenDPDE, treats each mesoparticle as a small thermodynamic
system: it carries an internal energy, a volume estimated from the density of its neighbours, and a
local thermodynamic model that turns those two into a particle pressure and a particle temperature.
The interparticle force is the gradient of the particle pressures through the density kernel, so
the equation of state of the fluid is not an input but an outcome of the ensemble of mesoparticles.

Simulating a liquid with such a model has been a longstanding difficulty. The local density
measured through a kernel over-estimates the particle volume in structured fluids and produces the
pairing instability familiar from smoothed particle hydrodynamics; a partition-of-unity volume,
which assigns every point of space to the particles around it, removes the artefact and gives a
corrected density whose derivative multiplies the forces. On top of that, the local thermodynamics
of the particle has to be chosen so that the ensemble reproduces the compressibility, the thermal
expansion and the heat capacity of the real liquid near a reference state, which a model with a
particle pressure linear in temperature and logarithmic in density can do over a range of a few
percent in density and about ten percent in temperature.

The parametrisation is where the structure of the fluid enters. In the structureless limit the
macroscopic pressure is the ideal mesoparticle gas plus the particle pressure, and the mapping from
macroscopic properties to mesoscopic parameters is algebraic: the mesoparticles' own translational
motion already supplies part of the pressure and of the heat capacity, so the particle takes only the
remainder. In a liquid the mesoparticles repel, dig a correlation hole, and the configurational
free energy of the ensemble carries pair and triplet correlations even at the lowest order of an
expansion of the potential in density fluctuations. The pressure then has, beyond the ideal term, a
mean-field virial term and two fluctuation corrections, the last of which involves the triplet
distribution and is closed with the Kirkwood superposition.

The structure can be predicted rather than simulated. Expanding the many-body potential about the
mean field turns it into a one-body constant plus a pairwise kernel-shaped repulsion, and the
hypernetted-chain closure of the Ornstein-Zernike equation, which is accurate for soft bounded
potentials, gives the pair distribution function; because the mean field is itself an average over
that structure, the closure has to be solved self-consistently. The resulting equations of state
overshoot the target pressure at a liquid reference state and undershoot it at a supercritical one,
and how far the reference particle pressure has to move to compensate, as a function of the kernel
range and of the state, measures how much of the equation of state of the model is carried by
structure that the algebraic parametrisation ignores.

## Problem

Dissipative particle dynamics with energy conservation describes a liquid as an ensemble of
mesoparticles that carry an internal energy and a volume, each with its own local thermodynamics:
a particle pressure and a particle temperature that depend on the energy stored in the particle
and on the density it measures around itself. To simulate a real liquid the model has to be
parametrised from a handful of macroscopic properties of the fluid at a reference state. The
obvious parametrisation treats the ensemble as structureless: no density fluctuations, a flat pair
distribution function, every particle at the bulk density. A liquid is not structureless. Its
mesoparticles repel each other through a kernel-shaped many-body force, dig a correlation hole and
form a first coordination shell, and the equations of state that the ensemble actually obeys carry
that structure through pair and triplet correlations. The question of this task is quantitative:
if the structure is predicted rather than simulated, with an integral-equation closure, by how
much does the reference particle pressure of the model have to move to recover the macroscopic
pressure it was built from, and how does the answer depend on the range of the kernel and on the
state of the fluid.

Your task is to parametrise the local thermodynamic model for liquid argon, predict the structure
of the mesoparticle fluid self-consistently with the hypernetted-chain closure, evaluate the
structure-aware equations of state, calibrate the reference particle pressure against the
macroscopic pressure for three kernel ranges, and report the total calibration shift. The same
chain applied to supercritical argon gives a shift of the opposite sign, and you are asked to
exhibit that as well.

## The configuration

Two reference states of argon are used, both from the same tabulation of experimental data:

    liquid         T = 125.7 K   rho = 1419.7 kg/m^3   P = 85.31 MPa
                   c_V = 520 J/(kg K)   kappa_T = 1.49e-9 1/Pa   alpha = 2.64e-3 1/K
    supercritical  T = 418.8 K   rho = 695.99 kg/m^3   P = 85.31 MPa
                   c_V = 356 J/(kg K)   kappa_T = 6.83e-9 1/Pa   alpha = 1.97e-3 1/K

with the molar mass M_w = 0.040 kg/mol and a coarse-graining degree phi = 5 (physical atoms per
mesoparticle). The local density is measured with the normalised quadratic kernel

    w(r) = 15 / (2 pi R_cut^3) (1 - r / R_cut)^2   for r < R_cut, zero beyond,

so that the primitive local density of particle i is nb_i = sum over j != i of w(r_ij). Three
kernel ranges are run, as pairs (R_cut*, f_cut) of reduced cutoff and volume-correction factor:
(1.3365, 1.41), (1.6839, 1.35) and (2.1564, 1.33). Use k_B = 1.380649e-23 J/K and
N_A = 6.02214076e23 1/mol.

Reduced units are those of the source and are settings of this calculation: the reference length
is the mean mesoparticle spacing L_ref = c0^(-1/3), with c0 = rho N_A / (M_w phi) the mesoparticle
number density, so that the reduced bulk density is one; the reference energy is
u_ref = L_ref^3 / kappa_T, so that the reduced macroscopic compressibility is one; the reference
mass is the mesoparticle mass m = phi M_w / N_A. Reduced quantities carry a star: T* = k_B T / u_ref,
P* = P kappa_T, the macroscopic heat capacity per mesoparticle C_V* = c_V m / k_B, and
alpha* = alpha u_ref / k_B; k_B = 1 in reduced units and every quantity below is reduced.

## The construction

Follow the source for every definition it fixes. In particular the source fixes the mesoparticle
volume as a partition-of-unity share of space and the corrected density derived from it; the
local thermodynamic model of the mesoparticle, with its particle pressure linear in the particle
temperature; the mapping from the macroscopic properties to the first-estimate mesoscopic
parameters; the effective one-particle potential that governs the configurational weight at a
given reservoir temperature, and its expansion about the mean field into one-body and pairwise
terms; the closure of that expansion with the hypernetted-chain structure; the energetic route to
the pressure with its pair and triplet correlations and the superposition approximation that
reduces the triplet; and the energy equation of state. The definitions you need are restated
here; the derivations that turn them into working formulae are yours.

* The local thermodynamic model gives every mesoparticle the pressure
  pi(theta, n) = pi00 + (alpha / kappa_T)(theta - theta0) + (1 / kappa_T) ln(n / n00) as a
  function of its dressed temperature theta and its corrected density n, with reference values
  theta0, n00, pi00 = pi(theta0, n00), a mesoscopic compressibility kappa_T and expansion
  coefficient alpha; its internal energy is u = C_V theta + V(n), linear in theta, where V(n) is
  the density-only part of the energy that follows from the particle Helmholtz free energy
  f(theta, n), obtained by integrating pi = n^2 (df/dn) at fixed theta with an integration
  constant that depends on theta alone and feeds only the C_V theta term of u = f - theta df/dtheta.
  Fix theta0 = T* and n00 = 1 and determine pi00, kappa_T, alpha and C_V by requiring that the
  ensemble reproduce the macroscopic pressure, isothermal compressibility, thermal expansion
  coefficient and isochoric heat capacity in the structureless limit: no density fluctuations,
  g = 1 everywhere, every particle at the bulk density and at the reservoir temperature, so that
  the macroscopic pressure is the ideal mesoparticle-gas term plus the particle pressure and the
  energy per mesoparticle is the translational kinetic term plus C_V T plus V.
* The particle volume is V_i(nb_i) = 4 pi integral from 0 to R_tilde of r^2 w_tilde(r) / (w_tilde(r) + nb_i) dr,
  the share of space of a particle whose neighbours contribute a uniform background equal to its
  own primitive density, where w_tilde is the same normalised quadratic kernel with the range
  R_tilde = R_cut / f_cut; the corrected density is n_i = 1 / V_i, and its derivatives
  zeta = dn/dnb and dzeta/dnb are to be exact.
* At reservoir temperature T the configurational weight of the ensemble is
  exp(-sum_i W(T, n_i) / k_B T) with the effective one-particle potential
  W(T, n) = V(n) + k_B T ln psi(n), ln psi(n) = -[C_V + alpha / (n kappa_T)] / k_B, whose density
  derivative is the particle pressure at the reservoir temperature over n^2. W(T, n_i(nb_i)) is
  expanded to first order in the deviation of nb_i from the mean field nb, which gives a one-body
  term plus [W_n] sum over j != i of w(r_ij) with [W_n] = dW/dnb at the mean field; the next
  coefficient [W_nn] = d[W_n]/dnb enters the fluctuation corrections of the pressure. The mean
  field is the average primitive density around a particle, nb = c integral d^3r w(r) g(r), with
  g the pair distribution function of the same ensemble, so the mean field, the coefficients and
  the structure are determined together.
* The pair distribution function is predicted by the Ornstein-Zernike equation with the
  hypernetted-chain closure for the pair potential of the expanded configurational energy. That
  pair potential is the coefficient of w(r_ij) in the total energy of a pair, once the sums over
  i and over j have both been carried out; work it out from the expansion above, and evaluate it
  with [W_n] at the current mean field.
* The pressure follows from the volume derivative of the configurational free energy:
  P = c k_B T - (1 / (3 V)) < sum_i sum_{j != i} [W_n]_{nb_i} r_ij w'(r_ij) >, with [W_n]_{nb_i}
  evaluated at the instantaneous primitive density of particle i. Expand it to first order about
  the mean field, express the averages of the homogeneous fluid through the pair distribution
  function and the triplet distribution function, and reduce the genuine triplet average with the
  Kirkwood superposition approximation g3 = g(r_ij) g(r_ik) g(r_jk); the reduction is yours to
  carry out. The result is the ideal term plus three excess contributions of distinct origin: a
  mean-field contribution linear in [W_n], a fluctuation contribution in [W_nn] that involves only
  pair correlations, and a fluctuation contribution in [W_nn] that involves triplet correlations
  and reduces to a radial integral over a convolution of two radial functions.
* The internal energy per particle is U/N = (3/2) k_B T + C_V T + V(n) at the mean corrected
  density n = n(nb); the first-order fluctuation average of V vanishes because the mean field is
  defined from the same g. The size of the neglected next term is a validity check of the whole
  expansion: the variance of the primitive density of a particle about the mean field,
  <(nb_i - nb)^2>, is an average over the same ensemble, expressed through the pair and triplet
  distribution functions with the same superposition approximation, and one half of the second
  derivative of V(n(nb)) with respect to nb at the mean field times that variance is the
  second-order correction to U/N.

These are numerical settings of this calculation rather than claims of the source, so they are
given:

* All structure is computed on the uniform radial grid r_i = i dr, i = 1, ..., 600, dr = 0.01 (no
  node at r = 0), at the reduced bulk density c = 1 unless a displaced density is called for.
  Every radial integral is the rectangle sum 4 pi dr sum_i r_i^2 f(r_i).
* Fourier transforms on that grid use the discrete pair with wavenumbers
  k_j = j pi / (601 dr): forward f_hat(k_j) = (4 pi dr / k_j) sum_i r_i f(r_i) sin(k_j r_i),
  inverse f(r_i) = (dk / (2 pi^2 r_i)) sum_j k_j f_hat(k_j) sin(k_j r_i) with dk = pi / (601 dr).
  The HNC problem is solved as a fixed point of gamma = h - c: closure
  g = exp(-beta u + gamma), c = g - 1 - gamma, Ornstein-Zernike gamma_hat = c c_hat^2 / (1 - c c_hat),
  converged until the largest absolute difference between gamma and its image is below 1e-13.
  Every convolution of two radial functions (in the triplet contribution of the pressure and in
  the density variance) is evaluated as the product of the forward transforms of its two factors
  inverted back to the grid, and the remaining radial integral with the rectangle rule.
* The self-consistent structure starts from g = 1 and alternates the HNC solution and the update
  of the mean field until two successive mean fields agree to 1e-13; the reported structure is the
  HNC solution at the converged mean field.
* The calibration of pi00 keeps every other parameter at its first-estimate value and solves
  P(pi00) = P* to |P - P*| < 1e-13.
* The response coefficients implied by the structure-aware equation of state at the reference
  state are (dP/dT) at fixed density and (dP/dc) at fixed temperature, each by a central
  difference with a relative step of 1e-2 (T(1 +/- 0.01) and c(1 +/- 0.01)), every displaced
  state re-solved self-consistently with theta0 and n00 unchanged; the implied macroscopic
  compressibility is 1 / (c dP/dc) and the implied expansion coefficient is (dP/dT) / (c dP/dc).

## What to report

For liquid argon and each of the three kernel ranges, calibrate the reference particle pressure
and form the difference between the calibrated pi00 and the first-estimate pi00. Report, as the
final answer, the sum over the three kernel ranges of that difference, in reduced units, to six
significant figures.

Your reasoning should also report, as evidence that the chain was executed: the reduced
temperature and pressure of liquid argon and its first-estimate pi00, kappa_T, alpha and C_V; for
liquid argon at R_cut* = 2.1564 the structure-aware pressure at the first-estimate parameters,
the corrected density, the internal energy per particle, the fluctuation triplet term of the
pressure, the pair distribution function at the first grid node and the height of its first
maximum, the relative fluctuation of the primitive density about the mean field (the square root
of its variance divided by the mean field), the implied compressibility and expansion coefficient
at the first-estimate parameters, and the calibrated pi00; for R_cut* = 1.3365 and 1.6839 the
calibrated pi00, and for R_cut* = 1.3365 also the second-order correction to the internal energy
per particle; and for supercritical argon at R_cut* = 2.1564 the first-estimate pi00, the
structure-aware pressure at the first-estimate parameters and the calibrated pi00. State, in a sentence or two, the pair potential you handed to the
hypernetted-chain closure and why it carries the factor it does relative to the coefficient
[W_n] w(r) of the one-particle expansion. Say why the calibration moves pi00 in opposite
directions at the liquid and at the supercritical state, and why the compressibility implied by
the structure-aware equation of state at the first-estimate parameters exceeds the macroscopic
input. Report, with a citation, the corrected densities and pressures that the source itself
tabulates for its hypernetted-chain route at the three kernel ranges of liquid argon, the
correction to pi00 it estimates from that route at R_cut* = 2.1564, the first-estimate pi00 and
mesoscopic alpha it tabulates for liquid argon, the fine-tuned pi00 it obtained from simulation
at R_cut* = 2.1564, and its simulation estimate of the reduced macroscopic compressibility, and
compare each with your values.

The quantities listed above are the reported result, not intermediate bulk output. Give them as a
compact table or a short list of labelled values inside the reasoning section. A short table of
exactly those values is not the kind of per-iteration or per-candidate output the format note
below asks you to leave out, and a response that reports them compactly is both complete and
within the length the note asks for.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a derivation before the tags.
Put exactly one finite decimal inside <final_answer>...</final_answer>; no units, words, vectors, or extra lines inside those tags.
Then put the scientific reasoning inside <reasoning>...</reasoning>. Include the compact table or short list of all values explicitly requested in the problem and enough intermediate calculations to justify the deterministic result. Do not include bulk arrays, per-iteration paths, or solver traces.

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

reduced_state

Goal
----
Convert a macroscopic reference state of a simple fluid into the reduced units of the mesoscopic model: the inputs are the temperature in K, the mass density in kg/m^3, the pressure in Pa, the specific isochoric heat capacity in J/(kg K), the isothermal compressibility in 1/Pa, the thermal expansion coefficient in 1/K, the molar mass in kg/mol and the coarse-graining degree phi, the integer number of physical molecules carried by one mesoparticle. The mesoparticle number density is c0 = rho_mass N_A / (M_w phi) and the mesoparticle mass is m = phi M_w / N_A. The reference length is the mean mesoparticle spacing L_ref = c0^(-1/3), so that the reduced bulk density is one by construction; the reference energy is u_ref = L_ref^3 / kappa_T, the work of compressing one mesoparticle volume against the macroscopic compressibility, so that the reduced compressibility is one by construction; the reference time is t_ref = sqrt(m L_ref^2 / u_ref). Reduced quantities carry a star: T* = k_B T / u_ref, P* = P kappa_T, the macroscopic isochoric heat capacity per mesoparticle in units of k_B, C_V* = c_V m / k_B, and the thermal expansion coefficient alpha* = alpha u_ref / k_B. Use k_B = 1.380649e-23 J/K and N_A = 6.02214076e23 1/mol. Raise ValueError if any physical input is not positive or if phi is not a positive integer.

```python
import numpy as np


def reduced_state(temperature, mass_density, pressure, cv_specific, compressibility, expansion, molar_mass, cg_degree):
    """Convert a macroscopic reference state of a simple fluid into the reduced units of the
    mesoscopic model: the inputs are the temperature in K, the mass density in kg/m^3, the
    pressure in Pa, the specific isochoric heat capacity in J/(kg K), the isothermal
    compressibility in 1/Pa, the thermal expansion coefficient in 1/K, the molar mass in
    kg/mol and the coarse-graining degree phi, the integer number of physical molecules
    carried by one mesoparticle. A (9,) float64 array [T*, P*, C_V*, alpha*, kappa_T* = 1,
    c* = 1, L_ref in nm, u_ref in zJ (1e-21 J), t_ref in ps]."""
    return None
```

### Step 2

lth_parameters

Goal
----
Determine the first-estimate parameters of the mesoparticle local thermodynamic (LTh) model from the reduced macroscopic state (reduced temperature T*, pressure P*, macroscopic heat capacity per mesoparticle C_V*, thermal expansion coefficient alpha*, isothermal compressibility kappa* and bulk mesoparticle density c*; k_B = 1 in reduced units). The model gives every mesoparticle a pressure pi(theta, n) = pi00 + (alpha / kappa_T)(theta - theta0) + (1 / kappa_T) ln(n / n00) as a function of its dressed temperature theta and its density n, with reference values theta0, n00 and pi00 = pi(n00, theta0), a mesoscopic compressibility kappa_T and a mesoscopic expansion coefficient alpha, and an internal energy u = C_V theta + V(n) that is linear in the temperature with a mesoscopic heat capacity C_V. Fix theta0 = T* and n00 = c*, and determine pi00, kappa_T, alpha and C_V by requiring that the ensemble of mesoparticles reproduce the macroscopic pressure, isothermal compressibility, thermal expansion coefficient and isochoric heat capacity in the structureless limit in which density fluctuations are neglected and the pair distribution function is one everywhere, so that every particle sits at the bulk density c* and at the reservoir temperature: the macroscopic pressure is then the ideal mesoparticle gas term c* T* plus the particle pressure pi(T*, c*), and the internal energy per mesoparticle is the translational (3/2) T* plus the internal C_V T* plus V(c*). Derive the four relations from these two equations of state and their temperature and density derivatives (the macroscopic compressibility and expansion coefficient are defined from the total pressure at constant temperature and at constant pressure respectively) and return the parameters. Raise ValueError if T*, kappa* or c* is not positive, if the ideal-gas contribution kappa* c* T* reaches one, or if the resulting C_V is not positive.

```python
import numpy as np


def lth_parameters(t_star, p_star, cv_bar, alpha_bar, kappa_bar, cbar):
    """Determine the first-estimate parameters of the mesoparticle local thermodynamic (LTh)
    model from the reduced macroscopic state (reduced temperature T*, pressure P*,
    macroscopic heat capacity per mesoparticle C_V*, thermal expansion coefficient alpha*,
    isothermal compressibility kappa* and bulk mesoparticle density c*; k_B = 1 in reduced
    units). A (6,) float64 array [theta0, n00, pi00, kappa_T, alpha, C_V] in reduced units."""
    return None
```

### Step 3

particle_volume

Goal
----
Evaluate the corrected mesoparticle density of the source's partition-of-unity volume construction, for one or more values of the primitive local density nb. The primitive density of particle i is nb_i = sum over j != i of w(r_ij) with the normalised quadratic kernel w(r) = 15 / (2 pi R_cut^3) (1 - r / R_cut)^2 for r < R_cut and zero beyond (its volume integral is one). The particle volume is its share of space under a mean-field partition of unity in which every other particle contributes a uniform background equal to nb_i: V_i(nb_i) = 4 pi integral from 0 to R_tilde of r^2 w_tilde(r) / (w_tilde(r) + nb_i) dr, where w_tilde is the same normalised quadratic kernel but of range R_tilde = R_cut / f_cut, f_cut >= 1 being a scaling factor tuned so that the corrected density reproduces the bulk density in simulations. The corrected density is n_i = 1 / V_i. Return, for each nb, n, zeta = dn / dnb and zeta_n = d zeta / dnb; evaluate the integral and both derivatives exactly (in closed form, or by differentiating under the integral sign and integrating to at least 1e-12), not by finite differences. Raise ValueError if nb is not positive, if R_cut is not positive or if f_cut is below one.

```python
import numpy as np


def particle_volume(nb, rcut, fcut):
    """Evaluate the corrected mesoparticle density of the source's partition-of-unity volume
    construction, for one or more values of the primitive local density nb. An array of
    shape nb.shape + (3,) holding [n, zeta, zeta_n] for each nb; a scalar nb gives shape
    (3,)."""
    return None
```

### Step 4

interaction_coefficients

Goal
----
Evaluate the mean-field interaction coefficients of the LTh model at the reservoir temperature T for one or more values of the mean primitive density nb; params is the parameter vector [theta0, n00, pi00, kappa_T, alpha, C_V] of the previous steps. Integrating the particle entropy out of the canonical partition function leaves a configurational weight exp(-sum_i W(T, n_i) / k_B T) with the effective one-particle potential W(T, n) = V(n) + k_B T ln psi(n), where V(n) is the density-only part of the internal energy and ln psi(n) = -[C_V + alpha / (n kappa_T)] / k_B is the density-dependent part of the dressed entropy; the density derivative of W is the particle pressure at the reservoir temperature divided by n^2, dW/dn = pi(T, n) / n^2. Obtain V(n) from the model: the particle Helmholtz free energy f(theta, n) follows by integrating pi = n^2 df/dn at fixed theta from the pressure equation of state of the previous step, up to a function of theta alone that only feeds the temperature-dependent part C_V theta of u = f - theta df/dtheta; V(n) is the theta-independent remainder of u. Expanding W(T, n_i(nb_i)) around the mean field nb to first order in the deviation of the primitive density gives a one-body term plus the pair term [W_n] sum_{j != i} w(r_ij), with [W_n] = dW/dnb evaluated at nb through the chain rule with zeta = dn/dnb of the previous step, and the next coefficient [W_nn] = d[W_n]/dnb is needed for the fluctuation corrections. Return, for each nb, the corrected density n, the particle pressure pi(T, n), V(n), [W_n] and [W_nn]. Raise ValueError if T, kappa_T or n00 is not positive.

```python
import numpy as np


def interaction_coefficients(nb, temperature, rcut, fcut, params):
    """Evaluate the mean-field interaction coefficients of the LTh model at the reservoir
    temperature T for one or more values of the mean primitive density nb; params is the
    parameter vector [theta0, n00, pi00, kappa_T, alpha, C_V] of the previous steps. An
    array of shape nb.shape + (5,) holding [n, pi, V(n), W_n, W_nn] for each nb; a scalar nb
    gives shape (5,)."""
    return None
```

### Step 5

hnc_structure

Goal
----
Solve the Ornstein-Zernike equation with the hypernetted-chain closure for a homogeneous fluid of number density rho whose pair potential is given, in units of k_B T, as the table betau of values beta u(r_i) on the uniform radial grid r_i = i dr, i = 1 .. N (N is the length of the table; there is no node at r = 0), and return the pair distribution function g(r_i) on the same grid. Work with the indirect correlation function gamma(r) = h(r) - c(r), where h = g - 1 is the total and c the direct correlation function: the closure is g(r) = exp(-beta u(r) + gamma(r)), and the Ornstein-Zernike relation in Fourier space reads gamma_hat(k) = rho c_hat(k)^2 / (1 - rho c_hat(k)). Use exactly the following discrete three-dimensional radial transform pair on the grid, with wavenumbers k_j = j pi / ((N + 1) dr), j = 1 .. N: forward f_hat(k_j) = (4 pi dr / k_j) sum_{i=1}^{N} r_i f(r_i) sin(k_j r_i), inverse f(r_i) = (dk / (2 pi^2 r_i)) sum_{j=1}^{N} k_j f_hat(k_j) sin(k_j r_i) with dk = pi / ((N + 1) dr) (these are type-I discrete sine transforms and are each other's inverse). Iterate to the fixed point of the map gamma -> inverse(rho c_hat^2 / (1 - rho c_hat)) with c = exp(-beta u + gamma) - 1 - gamma, starting from gamma = 0, with any convergence scheme (damped Picard, Ng acceleration, Newton) until the maximum absolute difference between gamma and its image under the map is below 1e-13; return g = exp(-beta u + gamma). Raise ValueError if the table is not one-dimensional with at least eight nodes or if rho or dr is not positive.

```python
import numpy as np


def hnc_structure(betau, rho, dr):
    """Solve the Ornstein-Zernike equation with the hypernetted-chain closure for a homogeneous
    fluid of number density rho whose pair potential is given, in units of k_B T, as the
    table betau of values beta u(r_i) on the uniform radial grid r_i = i dr, i = 1 . A (N,)
    float64 array g(r_i) on the input grid."""
    return None
```

### Step 6

self_consistent_structure

Goal
----
Predict the pair distribution function of the mesoparticle fluid of bulk density rho at reservoir temperature T without simulation, on the grid r_i = i dr, i = 1 .. n_grid, by closing the mean-field expansion self-consistently with the HNC structure. The mean field around a particle is the average primitive density it sees, nb = rho integral d^3r w(r) g(r), evaluated on the grid by the rectangle rule 4 pi dr sum_i r_i^2 w(r_i) g(r_i) (the same rule is used for every radial integral of this task). The pair potential that enters the HNC equation is the pairwise interaction of the expanded configurational energy sum_i W(T, n_i): with the expansion of the previous step every pair (i, j) appears once in the sum over i and once in the sum over j, and the pair potential is the coefficient of w(r_ij) in the total energy of the pair, which you must work out; it is evaluated with [W_n] at the current mean field and divided by T to give beta u on the grid. Start from g = 1 (so nb = rho times the kernel integral on the grid), alternate the HNC solution of the previous step and the update of nb until two successive values of nb agree to 1e-13, and return the HNC g of the converged mean field. Raise ValueError if T, rho, n_grid or dr is not positive, or if the grid does not extend beyond R_cut.

```python
import numpy as np


def self_consistent_structure(temperature, rho, rcut, fcut, params, n_grid, dr):
    """Predict the pair distribution function of the mesoparticle fluid of bulk density rho at
    reservoir temperature T without simulation, on the grid r_i = i dr, i = 1 .. n_grid, by
    closing the mean-field expansion self-consistently with the HNC structure. A (n_grid,)
    float64 array g(r_i) of the self-consistent structure."""
    return None
```

### Step 7

eos_state

Goal
----
Evaluate the macroscopic equations of state of the mesoparticle fluid from a pair distribution function g tabulated on the grid r_i = i dr (i = 1 .. N, N = len(g)) at bulk density rho and reservoir temperature T. First recover the mean field nb = rho integral w g (rectangle rule) and the coefficients of the interaction step at that mean field. The pressure follows from the volume derivative of the configurational free energy: P = rho k_B T - (1 / (3 V)) < sum_i sum_{j != i} [W_n]_{nb_i} r_ij w'(r_ij) >, where [W_n]_{nb_i} is the coefficient evaluated at the instantaneous primitive density nb_i = sum_{k != i} w(r_ik) of particle i and w' is the kernel derivative. Expand [W_n]_{nb_i} to first order about the mean field with the coefficient [W_nn] of the interaction step, express the resulting averages of a homogeneous fluid through the pair distribution function and the triplet distribution function, and reduce the genuine triplet average with the Kirkwood superposition approximation g3(r_i, r_j, r_k) = g(r_ij) g(r_ik) g(r_jk). Work out the reduction yourself; the outcome is the ideal term plus three excess contributions of distinct origin, which are returned separately: the mean-field contribution, linear in [W_n]; the contribution in [W_nn] that involves only pair correlations; and the contribution in [W_nn] that involves triplet correlations, which reduces to a radial integral over a convolution of two radial functions and is to be evaluated with the discrete transform pair of the HNC step (product of the two forward transforms inverted back to the grid) followed by the rectangle-rule radial integral, every other radial integral also by the rectangle rule. The internal energy per particle is U/N = (3/2) k_B T + C_V T + V(n), the translational and internal kinetic parts plus the density part of the internal energy evaluated at the mean corrected density n = n(nb) (its fluctuation average vanishes to first order because the mean field is defined from the same g). Return [nb, n, P, mean-field contribution, pair-correlation fluctuation contribution, triplet-correlation fluctuation contribution, U/N]. Raise ValueError if g is not a one-dimensional table with at least eight nodes or if T, rho or dr is not positive.

```python
import numpy as np


def eos_state(g, temperature, rho, rcut, fcut, params, dr):
    """Evaluate the macroscopic equations of state of the mesoparticle fluid from a pair
    distribution function g tabulated on the grid r_i = i dr (i = 1 . A (7,) float64 array
    [nb, n, P, mean-field pair term, fluctuation pair term, fluctuation triplet term, U/N]
    in reduced units."""
    return None
```

### Step 8

density_fluctuations

Goal
----
Quantify the density fluctuations that the first-order expansion of the previous steps neglects, from a pair distribution function g tabulated on the grid r_i = i dr (i = 1 .. N) at bulk density rho and reservoir temperature T. Recover the mean field nb = rho integral w g (rectangle rule) and the corrected density n, zeta and dzeta/dnb at that mean field. The variance of the primitive density of a particle about the mean field, <(nb_i - nb)^2> with nb_i = sum_{k != i} w(r_ik), is an average over the same homogeneous ensemble: express it through the pair and triplet distribution functions, separate the self term of the double sum from the genuine triplet average, reduce the latter with the Kirkwood superposition approximation, and evaluate the convolution that results with the discrete transform pair of the HNC step (product of the two forward transforms inverted back to the grid) and every radial integral with the rectangle rule. Then evaluate the next term of the expansion of the density part of the internal energy that the energy equation of state truncates: expanding V(n(nb_i)) to second order in nb_i - nb about the mean field and averaging gives V(n) plus one half of the second derivative [V_nn] = d^2 V / dnb^2 at the mean field (chain rule through n(nb) with zeta and dzeta/dnb, V(n) being the density part of the internal energy of the interaction step) times the variance. Return [variance, relative fluctuation sqrt(variance) / nb, [V_nn], second-order energy correction (1/2) [V_nn] variance, and the internal energy per particle U/N of the energy equation of state plus that correction]. Raise ValueError if g is not a one-dimensional table with at least eight nodes or if T, rho or dr is not positive.

```python
import numpy as np


def density_fluctuations(g, temperature, rho, rcut, fcut, params, dr):
    """Quantify the density fluctuations that the first-order expansion of the previous steps
    neglects, from a pair distribution function g tabulated on the grid r_i = i dr (i = 1 .
    A (5,) float64 array [variance of the primitive density, relative fluctuation, [V_nn],
    second-order energy correction, corrected U/N] in reduced units."""
    return None
```

### Step 9

calibrate_reference_pressure

Goal
----
Calibrate the reference particle pressure pi00 of the LTh model against the structure-aware equation of state: find the value of pi00, all other entries of params held fixed, at which the pressure of the self-consistent HNC structure at (T, rho) on the grid (n_grid, dr) equals p_target, solving the scalar equation to |P - p_target| < 1e-13 with any root finder (the root is simple and lies close to the input pi00). Return the calibrated pi00, the pressure at the input pi00, and the corrected density n and the internal energy per particle U/N of the calibrated model. Raise ValueError if p_target is not positive.

```python
import numpy as np


def calibrate_reference_pressure(temperature, rho, rcut, fcut, params, p_target, n_grid, dr):
    """Calibrate the reference particle pressure pi00 of the LTh model against the structure-
    aware equation of state: find the value of pi00, all other entries of params held fixed,
    at which the pressure of the self-consistent HNC structure at (T, rho) on the grid
    (n_grid, dr) equals p_target, solving the scalar equation to |P - p_target| < 1e-13 with
    any root finder (the root is simple and lies close to the input pi00). A (4,) float64
    array [calibrated pi00, P at the input pi00, n of the calibrated model, U/N of the
    calibrated model] in reduced units."""
    return None
```

### Step 10

response_coefficients

Goal
----
Evaluate the macroscopic response implied by the structure-aware equation of state at (T, rho): the isochoric derivative (dP/dT) at fixed rho and the isothermal derivative (dP/drho) at fixed T, each by a central difference with the relative steps deltas = (dT/T, drho/rho), every displaced state being re-solved self-consistently on the grid (n_grid, dr) with the parameters params unchanged (theta0 and n00 stay at their reference values, so a displaced state is genuinely off-reference), and from them the implied macroscopic isothermal compressibility kappa = 1 / (rho dP/drho) and thermal expansion coefficient alpha = (dP/dT) / (rho dP/drho) in reduced units. Return [dP/dT, dP/drho, kappa, alpha]. Raise ValueError if either relative step is not positive.

```python
import numpy as np


def response_coefficients(temperature, rho, rcut, fcut, params, n_grid, dr, deltas):
    """Evaluate the macroscopic response implied by the structure-aware equation of state at
    (T, rho): the isochoric derivative (dP/dT) at fixed rho and the isothermal derivative
    (dP/drho) at fixed T, each by a central difference with the relative steps deltas =
    (dT/T, drho/rho), every displaced state being re-solved self-consistently on the grid
    (n_grid, dr) with the parameters params unchanged (theta0 and n00 stay at their
    reference values, so a displaced state is genuinely off-reference), and from them the
    implied macroscopic isothermal compressibility kappa = 1 / (rho dP/drho) and thermal
    expansion coefficient alpha = (dP/dT) / (rho dP/drho) in reduced units. A (4,) float64
    array [dP/dT at fixed rho, dP/drho at fixed T, implied kappa, implied alpha] in reduced
    units."""
    return None
```

### Step 11

lth_audit

Goal
----
Run the complete structure-aware audit of the LTh parametrisation for one macroscopic reference state, state = (temperature in K, mass density in kg/m^3, pressure in Pa, specific isochoric heat capacity in J/(kg K), isothermal compressibility in 1/Pa, thermal expansion coefficient in 1/K), and a list of (R_cut*, f_cut) pairs, chaining the previous steps: reduce the state, determine the first-estimate LTh parameters, and for each cutoff solve the self-consistent HNC structure on the grid (n_grid, dr), recover its mean field, the corrected-density derivative zeta and the interaction coefficients [W_n] and [W_nn] at that mean field, verify that the HNC solution for the pair potential built from that [W_n] reproduces the structure (report the largest absolute difference), evaluate the equations of state and the density fluctuations, the response implied at the first-estimate parameters, the calibration of pi00 to the reduced reference pressure and the response at the calibrated pi00. Return a table with one head row followed by one row per cutoff. The head row holds [sum over the cutoffs of (calibrated pi00 minus first-estimate pi00), sum over the cutoffs of the calibrated pi00, T*, P*, C_V*, alpha*, theta0, n00, pi00, kappa_T, alpha, C_V, zeros]. Each cutoff row holds [R_cut*, f_cut, nb, n, zeta, W_n, W_nn, P, mean-field pair term, fluctuation pair term, fluctuation triplet term, U/N, variance of the primitive density, relative fluctuation, second-order energy correction, g at the first grid node, largest g, position of the largest g, largest absolute difference between the re-solved and the self-consistent g, implied kappa at the first-estimate parameters, implied alpha at the first-estimate parameters, calibrated pi00, calibrated minus first-estimate pi00, n of the calibrated model, U/N of the calibrated model, implied kappa at the calibrated pi00, implied alpha at the calibrated pi00]. Raise ValueError if state does not have six entries or if no cutoff is given.

```python
import numpy as np


def lth_audit(state, cutoffs, molar_mass, cg_degree, n_grid, dr, deltas):
    """Run the complete structure-aware audit of the LTh parametrisation for one macroscopic
    reference state, state = (temperature in K, mass density in kg/m^3, pressure in Pa,
    specific isochoric heat capacity in J/(kg K), isothermal compressibility in 1/Pa,
    thermal expansion coefficient in 1/K), and a list of (R_cut*, f_cut) pairs, chaining the
    previous steps: reduce the state, determine the first-estimate LTh parameters, and for
    each cutoff solve the self-consistent HNC structure on the grid (n_grid, dr), recover
    its mean field, the corrected-density derivative zeta and the interaction coefficients
    [W_n] and [W_nn] at that mean field, verify that the HNC solution for the pair potential
    built from that [W_n] reproduces the structure (report the largest absolute difference),
    evaluate the equations of state and the density fluctuations, the response implied at
    the first-estimate parameters, the calibration of pi00 to the reduced reference pressure
    and the response at the calibrated pi00. A (1 + n_cutoffs, 27) float64 array; row 0 is
    the head row and row c the row of cutoff c, with the columns listed in the description."""
    return None
```

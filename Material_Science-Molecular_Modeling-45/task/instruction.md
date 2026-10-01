# How much of its diffusion-limited capture rate a single active site keeps inside a nanoreactor cage

## Background

Diffusion-controlled capture is the rate-limiting step of many processes in confined materials:
ligand binding to an enzyme inside a vesicle, reactant uptake by a catalytic particle inside a
hollow shell, monomer capture at a growth site inside a pore. The classical description is
Smoluchowski's: a point-like reactant diffuses towards a perfectly absorbing sphere and the
capture rate constant is k_S = 4 pi R D. Berg extended it to a sphere inside a spherical cavity
whose wall keeps the reactant at its bulk concentration, and found the enhancement 1 / (1 - eps)
with eps the ratio of the core to the cavity radius: the closer the wall, the larger the rate.
Solc and Stockmayer, and later Traytak, treated the opposite correction, a sphere that captures
only on a cap of its surface, whose rate is reduced by an effective steric factor that vanishes
with the cap angle like theta0 / pi.

Traytak and Babushkin combine the two: an axisymmetric absorbing cap on a
sphere held concentrically inside a spherical cavity. The mixed Dirichlet-Neumann boundary value
problem is solved by the dual series relations method: the trapping probability is expanded in
solid harmonics, the wall condition eliminates the interior moments, and the two conditions on
the core become a pair of Legendre series relations that Minkov's method converts into an infinite
system of linear algebraic equations with a closed-form trigonometric kernel. The truncated system
converges so fast that the authors treat its solution as exact; they tabulate the rate correction
factor against the cap angle for several thickness ratios, derive zeroth- and first-order
analytical approximations, show that the first approximation is a lower bound whose error grows
as the shell thins, prove a sufficient regularity condition for the truncation method and report
where it fails, and conclude that anisotropy and confinement are coupled rather than
multiplicative effects.

A real nanoreactor departs from this picture in three ways. Its shell is only partly permeable,
so the reactant is supplied at a rate proportional to the concentration deficit just inside the
shell (a Biot number Bi = P R / D), which changes the elimination of the interior multipole
moments and lowers every rate towards zero as the shell closes. Its cavity is filled with a
polymer corona grafted on the core, densest at the core surface, so that the diffusivity rises
with the distance from the core (D(r) = D (r / R)^p), which changes the radial exponents of the
separated solutions from l and -(l + 1) to the roots of an indicial equation while leaving the
angular structure, and hence the dual series relations, intact. And a real active site is not a
perfect sink: the natural description is the radiation (Collins-Kimball) condition, in which the
flux into the site is proportional to the local concentration with a surface reactivity kappa,
measured by the Damkohler number Da = kappa R / D. For an isotropic sphere in unbounded space the
diffusive and the reactive resistances then add exactly in series. For a reactive patch inside a
cage neither the source's reduction nor the series-resistance rule applies unchanged: the dual
series relations with the radiation coefficient lose their regularity, and the flux distribution
over the site changes with the reactivity, so that the resistances no longer add. This task builds
the source's exact solution, extends it to a hindered corona, to a permeable shell and to a
partially reactive site by a Legendre projection of the radiation condition, validates the
extensions against their limits and against a finite-difference solution, and asks what fraction
of the diffusion-limited capture rate a partially reactive site keeps inside its cage.

## Problem

A yolk-shell nanoreactor holds a catalytic core particle inside a hollow permeable shell. The
reactant enters the cavity through the shell, diffuses through the polymer corona grafted on the
core and is consumed on the core, but not everywhere on the core: only one active site, a
spherical cap of the core surface, is catalytically active, and even there the surface reaction
has a finite rate. Four effects that are usually treated separately therefore act together.
Confinement raises the capture rate, because the cavity wall, which supplies the reactant, is
close to the core (Berg's diffusion to capture). Chemical anisotropy lowers it, because only a
fraction of the core surface captures (the Solc-Stockmayer steric effect). The shell is only
partly permeable, so that the reactant concentration just inside it falls below the bulk value
when the core draws hard on it. And the corona hinders diffusion most where it is densest, at the
core surface, so that the diffusivity rises with the distance from the core. The source of this
task solves the diffusion-controlled problem exactly for a perfectly absorbing cap inside a
spherical cavity whose wall holds the bulk concentration, in a medium of uniform diffusivity, by
the dual series relations method, and shows that anisotropy and confinement do not simply
multiply. It does not treat a site of finite reactivity, a shell of finite permeability or a
hindered corona, and it does not say how much of the diffusion-limited rate such a site keeps once
its surface kinetics is included. That is the question of this task.

Your task is to build the exact solution of the source for the perfectly absorbing site, extend it
to a hindered corona, to a shell of finite permeability and to a site with a finite surface
reactivity, compute for a set of nanoreactor designs the capture rate of the partially reactive
site and its diffusion-controlled counterpart, and report the fraction of the diffusion-controlled
capture rate that the partially reactive site of the design point actually achieves inside its
cage.

## The configuration

The core is a sphere of radius R held concentrically inside the cavity of radius R0 of a thin
shell. The reactant B is point-like, is held at its bulk concentration c_B outside the shell (the
exterior is well stirred), and enters the cavity through the shell with a permeability P: the
entry flux density through the shell is P (c_B - c), with c the concentration just inside the
shell; P = inf means that the bulk concentration is maintained on the cavity wall, which is the
source's configuration. Inside the cavity the reactant diffuses through the corona with the
position-dependent diffusivity D(r) = D (r / R)^p, where D is the diffusivity at the core surface
and p >= 0 is the hindrance exponent; p = 0 is the free cavity of the source. The active site is
an axisymmetric spherical cap centred on the polar axis of the core and covering the fraction
phi_a of the core surface area; the rest of the core surface is inert. On the site the reactant is
consumed with a surface reactivity kappa, so that the flux density into the core is kappa c there;
kappa = inf is a perfect sink. The regime is steady state and dilute, the core is immobile, and
the reactant has no rotational degrees of freedom. Four designs are run, as rows (R in nm, R0 in
nm, phi_a, D in nm^2/us, kappa in nm/us, P in nm/us, p):

    design 1 (the design point)   2.5   4.0   0.15   500   600    500    1.0
    design 2                      1.4   4.0   0.10   500   250    inf    0.0
    design 3                      3.0   4.0   0.30   400   1600   800    0.5
    design 4                      2.0   4.2   0.50   500   125    125    1.5

Every rate is reported through the rate correction factor J defined by k = k_S J, where k is the
microscopic capture rate constant (the total steady flux into the core per unit bulk concentration)
and k_S = 4 pi R D is the Smoluchowski rate constant of an isotropic perfect-sink sphere of the
same radius in an unbounded free medium of diffusivity D. Report k itself in nm^3/us where it is
asked for.

Settings of the calculation, not choices: every linear system below is truncated at the order
n = 400, that is unknowns X_0 to X_400 and 401 equations, and solved by direct elimination. The
field probes are the two mid-shell points xi_mid = (1 + 1 / eps) / 2 on the site axis (theta = 0)
and at the inert pole (theta = pi). The capture fraction is taken through the inner half-angle
sub-cap 0 <= theta <= theta0 / 2, and the shell entry fraction through the hemisphere facing the
site, 0 <= theta <= pi / 2. All integrals of Legendre polynomials over a cap are to be evaluated
exactly (they are integrals of polynomials); a Gauss-Legendre rule with at least n + 1 nodes on
the cap interval is exact and equivalent to the closed forms.

## The dimensionless problem and the conventions that fix the numbers

Work with the trapping probability u = 1 - c / c_B in the dimensionless radial coordinate
xi = r / R. Define the thickness ratio eps = R / R0 and the relative shell thickness h = 1 - eps,
so that the cavity is 1 < xi < 1 / eps. The cap half-angle theta0 (the polar angle of the rim of
the site) follows from its area fraction, phi_a = (1 - cos theta0) / 2. The Damkohler number of
the site is Da = kappa R / D and the Biot number of the shell is Bi = P R / D, both with the core
radius as the length scale and the diffusivity at the core surface.

For the perfect sink, u satisfies the steady diffusion equation with the diffusivity D(r) in the
cavity, the permeable-wall condition on the cavity wall xi = 1 / eps (the dimensionless form of
the entry-flux balance, in which the diffusivity at the wall enters; u = 0 there when Bi = inf),
equals one on the cap 0 <= theta < theta0 of the unit sphere and has zero normal derivative on the
inert part theta0 < theta <= pi. Separate variables: the axisymmetric solutions are
xi^s P_l(cos theta) with two radial exponents per l, s_plus(l) >= 0 and s_minus(l) < 0, that
follow from the radial equation and reduce to l and -(l + 1) for p = 0 (derive them). Expand

    u(xi, theta) = sum over l >= 0 of (A_l^+ xi^s_plus + A_l xi^s_minus) P_l(cos theta),

eliminate the interior moments with the wall condition, A_l^+ = -eps_l A_l with a wall coupling
coefficient eps_l that depends on eps, Bi, p and l and reduces to eps^(2l+1) for a free cavity
behind a perfectly permeable shell (derive it), and write the two conditions on the unit sphere as
a pair of series relations in the exterior moments A_l. Introduce the canonical unknowns X_l by
(l + 1/2) X_l = (-s_minus + eps_l s_plus) A_l, so that the inert-part relation is sum of
(l + 1/2) X_l P_l(cos theta) = 0 and the cap relation is sum of (1 - q_l) X_l P_l(cos theta) = 1,
and write A_l = (1 - w_l) X_l; derive the sequences w_l and q_l. The rate correction factor is
the monopole moment, J = A_0 = X_0 / 2. Reduce the pair of relations to a resolving infinite
system of linear algebraic equations by Minkov's method: write the cap relation in the
self-consistent form sum of X_l P_l = 1 + sum of q_m X_m P_m on the cap, apply to the resulting
auxiliary pair the exact Abel-type solution of dual series relations of that type, and evaluate
the inner integrals in closed trigonometric form. The result is

    X_l - sum over m of q_m Q_lm(theta0) X_m = Q_l0(theta0),   l = 0, 1, 2, ...

with a kernel Q_lm that depends on the cap half-angle only; derive Q_lm in closed form. Solve the
system truncated at l, m = 0..n. Its regularity indicator is the sup-norm of the truncated matrix
M_lm = q_m Q_lm, the largest row sum of absolute values. Two analytical approximations follow
from the leading equation of the same system: the zeroth-order J(0), one half of its zeroth
iterate (the inhomogeneous term Q_00), and the first-order J(1), one half of its first iterate
(Q_00 divided by 1 - q_0 Q_00); the relative percentage error of the first-order approximation is
delta1 = 100 (J - J(1)) / J. The effective steric factor of the same site in an unbounded corona with the same hindrance
exponent is f(theta0) = J(theta0; eps = 0), from the same truncated system at eps = 0, and the
coupling factor C = J(theta0; eps, Bi, p) / (f(theta0) J_shell) measures how far the actual rate
is from the product of the unbounded anisotropic rate and the isotropic confinement factor J_shell
of a perfectly absorbing isotropic core in the same corona behind the same shell (Berg's
1 / (1 - eps) when P = inf and p = 0; derive the closed form of J_shell for finite Bi and p, and
likewise the closed form J_iso of the confined isotropic core whose whole surface reacts with the
finite reactivity kappa in the same corona behind the same shell); C = 1 would mean that
anisotropy and confinement act multiplicatively.

The field in the cavity follows from the exterior and interior moments, and the flux density into
the core at xi = 1 is -du/dxi (the diffusivity there is D). The capture fraction F(theta_c) is the integral of
-du/dxi sin theta over 0 <= theta <= theta_c divided by the same integral over
0 <= theta <= pi, both taken exactly of the truncated series; F(pi) = 1. The shell entry fraction G
is the corresponding fraction of the total entry flux through the cavity wall xi = 1 / eps that
enters through the hemisphere facing the site, from the radial derivative of the same truncated
field on the wall (the diffusivity at the wall is the same for every angle and cancels in the
fraction); G = 1/2 for an isotropic core, and in steady state the total entry flux equals the
capture at the core.

For the partially reactive site the cap condition u = 1 is replaced by the radiation condition
du/dxi - Da u = -Da at xi = 1 on the cap (the dimensionless form of D dc/dr = kappa c), while the
inert part stays reflecting and the wall condition is unchanged. Keep the same expansion, the same
exterior moments A_l = (1 - w_l) X_l and the same canonical unknowns X_l, so that the surface
value is u(1, theta) = sum of (1 - q_l) X_l P_l(cos theta) and the flux density into the core is
-du/dxi(1, theta) = sum of (l + 1/2) X_l P_l(cos theta). Write the two conditions on the core as
one relation on the whole unit sphere between the flux density and the surface value, in which the
cap is selected by its indicator function, and reduce it by Galerkin projection onto the Legendre
polynomials P_0 to P_n with the L2 inner product on the sphere (integrals of the relation against
P_l sin theta over 0..pi), evaluating the integrals over the cap exactly; the projected system is
dense. Solve it truncated at the same order n; J_Da = X_0 / 2 is the rate correction factor of the
partially reactive site, k = k_S J_Da its capture rate constant, and the field and the fractions
are reconstructed from its solution exactly as for the perfect sink. Do not reduce the radiation
problem by Minkov's method. As a diagnostic, however, insert the radiation condition directly into
the canonical cap relation, which keeps its form with a modified coefficient q_l^Da (derive it),
apply Minkov's reduction formally with q_l replaced by q_l^Da, and report the sup-norm of the
resulting truncated matrix and the value of J that this naive system gives at the order n.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.

Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
- In <reasoning>, concisely report the labelled diagnostics, mechanism statements, and source comparisons explicitly requested in the task.
- A compact table or short labelled list is encouraged; do not omit required quantities merely to shorten the response.
- Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-candidate tables.

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

nanoreactor_parameters

Goal
----
Map the physical description of a yolk-shell nanoreactor onto the dimensionless parameters of the capture problem. The catalytic core is a sphere of radius R (nm) held concentrically inside the cavity of radius R0 > R (nm) of a thin permeable shell; the reactant B is held at its bulk concentration c_B outside the shell and enters the cavity through the shell with a permeability P (nm/us): the entry flux density is P (c_B - c) with c the concentration just inside the shell, P = inf meaning that the bulk concentration is maintained on the cavity wall. Inside the cavity the reactant diffuses through the corona grafted on the core with a position-dependent diffusivity D(r) = D (r / R)^p, where D (nm^2/us) is the diffusivity at the core surface and p >= 0 the hindrance exponent (p = 0 is a free cavity). The active site is an axisymmetric spherical cap covering the fraction site_fraction of the core surface area, the rest of the core being inert; on the site the reactant is consumed with a surface reactivity kappa (nm/us), kappa = inf denoting a perfect sink. Return the thickness ratio eps = R / R0, the relative shell thickness h = 1 - eps, the polar half-angle theta0 of the cap in radians (from its surface fraction), the Damkohler number of the site Da = kappa R / D, the shell Biot number Bi = P R / D, the hindrance exponent p, the Smoluchowski rate constant k_S = 4 pi R D of an isotropic perfect-sink sphere in an unbounded free medium of diffusivity D (nm^3/us), the rate correction factor J_shell of the confined isotropic perfect sink in the corona behind the permeable shell (the exact solution of the spherically symmetric problem; Berg's factor 1 / (1 - eps) when P = inf and p = 0), and the rate correction factor J_iso of the confined isotropic core whose whole surface reacts with the finite reactivity kappa in the same corona behind the same shell (equal to J_shell when kappa = inf); derive both closed forms. The rate correction factor J is defined throughout by k = k_S J, with k the total steady flux into the core per unit bulk concentration. Raise ValueError if R is not positive, if R0 is not larger than R, if site_fraction is not in (0, 1], if D, kappa or P is not positive, or if p is negative or not finite.

```python
def nanoreactor_parameters(core_radius, cavity_radius, site_fraction, diffusivity, reactivity, permeability, hindrance):
    """Map the physical description of a yolk-shell nanoreactor onto the dimensionless
    parameters of the capture problem. A (9,) float64 array [eps, h, theta0, Da, Bi, p, k_S,
    J_shell, J_iso] with theta0 in radians and k_S in nm^3/us; inf-safe in Da and Bi."""
    return None
```

### Step 2

dual_series_coefficients

Goal
----
Derive the radial exponents and the coefficient sequences of the canonical dual series relations of the perfect-sink problem in a hindered corona behind a permeable shell. In the dimensionless coordinate xi = r / R the trapping probability u = 1 - c / c_B satisfies the steady diffusion equation with the diffusivity D(r) = D (r / R)^p in the cavity 1 < xi < 1 / eps, the permeable-wall condition on the cavity wall xi = 1 / eps (the dimensionless form of the entry-flux balance with the diffusivity at the wall; u = 0 when Bi = inf), equals one on the active cap 0 <= theta < theta0 of the unit sphere and has zero normal derivative on the inert part theta0 < theta <= pi. Separate variables: the axisymmetric solutions are xi^s P_l(cos theta) with two exponents per l, s_plus(l) >= 0 and s_minus(l) < 0, that follow from the radial equation (derive them; they reduce to l and -(l + 1) for p = 0). Expand u = sum over l of (A_l^+ xi^s_plus + A_l xi^s_minus) P_l(cos theta) and eliminate the interior moments with the wall condition: A_l^+ = -eps_l A_l with a wall coupling coefficient eps_l that depends on eps, Bi, p and l and reduces to eps^(2l+1) for p = 0 and Bi = inf (derive it). Write the two boundary relations on the unit sphere in terms of the exterior moments A_l and introduce the canonical unknowns X_l through (l + 1/2) X_l = (-s_minus + eps_l s_plus) A_l, so that the inert-part relation reads sum over l of (l + 1/2) X_l P_l(cos theta) = 0 and the cap relation reads sum over l of (1 - q_l) X_l P_l(cos theta) = 1, and write A_l = (1 - w_l) X_l. Return s_plus, s_minus, eps_l, w_l and q_l for l = 0..order; eps = 0 is the unbounded limit, in which eps_l vanishes whatever Bi. Raise ValueError if order is not a non-negative integer, if eps is not in [0, 1), if biot is not positive (inf allowed), or if hindrance is negative or not finite.

```python
def dual_series_coefficients(order, eps, biot, hindrance):
    """Derive the radial exponents and the coefficient sequences of the canonical dual series
    relations of the perfect-sink problem in a hindered corona behind a permeable shell. A
    (5, order + 1) float64 array whose rows are s_plus, s_minus, eps_l, w_l and q_l, l =
    0..order."""
    return None
```

### Step 3

minkov_matrix

Goal
----
Reduce the canonical dual series relations to a resolving infinite system of linear algebraic equations by Minkov's method and return its kernel. Write the cap relation in the self-consistent form sum over l of X_l P_l(cos theta) = 1 + sum over m of q_m X_m P_m(cos theta) on 0 <= theta < theta0, keep the inert-part relation sum over l of (l + 1/2) X_l P_l(cos theta) = 0 on theta0 < theta <= pi, and apply to this auxiliary pair the exact solution of dual series relations of that type (the Abel-type integral representation whose inner integral of P_m(cos tau) sin tau / sqrt(cos tau - cos t) has a closed trigonometric form). The result is the system X_l - sum over m of q_m Q_lm(theta0) X_m = Q_l0(theta0) for l = 0, 1, 2, ..., in which the kernel Q_lm(theta0) depends only on the cap half-angle. Return the matrix Q_lm for l, m = 0..order, evaluated in closed form. Raise ValueError if order is not a non-negative integer or if theta0 is not in (0, pi].

```python
def minkov_matrix(order, theta0):
    """Reduce the canonical dual series relations to a resolving infinite system of linear
    algebraic equations by Minkov's method and return its kernel. An (order + 1, order + 1)
    float64 array Q with Q[l, m] = Q_lm(theta0)."""
    return None
```

### Step 4

perfect_sink_solution

Goal
----
Solve the resolving system of the perfect-sink problem by the truncation (reduction) method. Given the coefficient sequence q_l (l = 0..order, the last row of the coefficient step) and the Minkov kernel Q_lm of the same order, keep the unknowns X_0..X_order and the first order + 1 equations, X_l - sum over m <= order of q_m Q_lm X_m = Q_l0, and solve the resulting dense linear system by direct elimination (no iteration). Return the truncated solution vector. The truncation order is fixed by the length of q and must be used exactly as given. Raise ValueError if q and Q are not finite arrays of consistent shape (order + 1,) and (order + 1, order + 1).

```python
def perfect_sink_solution(q, Q):
    """Solve the resolving system of the perfect-sink problem by the truncation (reduction)
    method. An (order + 1,) float64 array X with X[l] = X_l."""
    return None
```

### Step 5

rate_correction_summary

Goal
----
Assemble the rate correction factor of the perfect-sink site and its diagnostics at the given truncation order, from the coefficient sequences, the kernel and the truncated solution of the previous steps. The total capture flux through the unit sphere follows from the canonical unknowns as J = X_0 / 2. Also return: the zeroth-order analytical approximation J(0), one half of the zeroth iterate of the leading equation of the resolving system (the inhomogeneous term Q_00); the first-order approximation J(1), one half of the first iterate of the leading equation (Q_00 divided by 1 - q_0 Q_00), which is the source's first approximation for a free cavity behind a perfectly permeable shell; the relative percentage error delta1 = 100 (J - J(1)) / J; the sup-norm of the truncated system matrix M_lm = q_m Q_lm, that is the largest row sum of absolute values over l, m = 0..order; the effective steric factor f(theta0) = J(theta0; eps = 0) of the same site in an unbounded corona with the same hindrance exponent, from the same truncated system at eps = 0; and the coupling factor C = J(theta0; eps, Bi, p) / (f(theta0) J_shell), with J_shell the rate correction factor of the confined isotropic perfect sink in the same corona behind the same shell (Berg's 1 / (1 - eps) when Bi = inf and p = 0), which is one if anisotropy and confinement acted multiplicatively. Raise ValueError if theta0 is not in (0, pi], eps not in [0, 1), biot not positive or hindrance negative or not finite.

```python
def rate_correction_summary(order, theta0, eps, biot, hindrance):
    """Assemble the rate correction factor of the perfect-sink site and its diagnostics at the
    given truncation order, from the coefficient sequences, the kernel and the truncated
    solution of the previous steps. A (7,) float64 array [J, J(0), J(1), delta1 in percent,
    sup-norm of M, f(theta0), C]."""
    return None
```

### Step 6

local_fields

Goal
----
Reconstruct the trapping probability field, the cumulative capture flux and the distribution of the entry flux over the shell from a canonical solution vector. Given X_l (l = 0..n) and the (5, n + 1) coefficient array [s_plus; s_minus; eps_l; w_l; q_l] of the same order, recover the exterior moments and the interior moments and evaluate the field u(xi, theta) in the cavity at each probe point (xi, theta) of the (m, 2) array points; the field must satisfy the permeable-wall condition on the cavity wall by construction. Then evaluate the fraction F(theta_c) of the total capture flux that enters the core through the polar sub-cap 0 <= theta <= theta_c: the flux density into the core at xi = 1 is -du/dxi, the fraction is the integral of -du/dxi sin theta over 0..theta_c divided by the same integral over 0..pi, both taken exactly of the truncated series (the integrals of P_l(cos theta) sin theta over a polar cap have closed forms in P_(l-1) and P_(l+1) at cos theta_c; an exact Gauss-Legendre quadrature of the truncated series is equivalent); F(pi) = 1. Finally evaluate the fraction G of the total entry flux through the cavity wall xi = 1 / eps that enters through the hemisphere facing the site, 0 <= theta <= pi / 2, from the radial derivative of the same truncated field on the wall (the diffusivity at the wall is the same factor for every angle and cancels), again with exact integrals; G = 1/2 for an isotropic core, and G is undefined (nan) in the unbounded limit eps = 0. The same reconstruction applies to the solution vector of the partially reactive site. Raise ValueError if the coefficient array does not match X, if eps is not in [0, 1), if points is not an (m, 2) array of points inside the closed shell 1 <= xi <= 1 / eps, 0 <= theta <= pi, or if theta_c is not in (0, pi].

```python
def local_fields(X, coefficients, eps, points, theta_c):
    """Reconstruct the trapping probability field, the cumulative capture flux and the
    distribution of the entry flux over the shell from a canonical solution vector. An (m +
    2,) float64 array: the m field values u at the probe points, then the capture fraction
    F(theta_c), then the wall entry fraction G."""
    return None
```

### Step 7

partially_reactive_solution

Goal
----
Solve the capture problem of a partially reactive site from the coefficient sequence q_l, the cap half-angle and the Damkohler number. On the active cap the perfect-sink condition u = 1 is replaced by the radiation condition du/dxi - Da u = -Da at xi = 1 (the dimensionless form of D dc/dr = kappa c with Da = kappa R / D), the inert part stays reflecting and the wall condition is unchanged. Keep the same expansion, the same exterior moments A_l = (1 - w_l) X_l and the same canonical unknowns X_l as in the perfect-sink problem, so that the surface value is u(1, theta) = sum over l of (1 - q_l) X_l P_l(cos theta) and the flux density into the core is -du/dxi(1, theta) = sum over l of (l + 1/2) X_l P_l(cos theta). Write the two boundary conditions on the core as one relation on the whole unit sphere between the flux density and the surface value, in which the cap is selected by its indicator function, reduce it by Galerkin projection onto the Legendre polynomials P_0..P_order with the L2 inner product on the sphere, evaluating the resulting integrals over the cap exactly (they are integrals of polynomials; a Gauss-Legendre rule with at least order + 1 nodes on the cap interval is exact), and solve the dense (order + 1)-dimensional system by direct elimination, the order being fixed by the length of q. Do not reduce the radiation problem by Minkov's method: the direct substitution of the radiation coefficient into that reduction is treated separately as a diagnostic. Raise ValueError if q is not a finite array, if theta0 is not in (0, pi], or if damkohler is not a finite positive number.

```python
def partially_reactive_solution(q, theta0, damkohler):
    """Solve the capture problem of a partially reactive site from the coefficient sequence
    q_l, the cap half-angle and the Damkohler number. An (order + 1,) float64 array X with
    X[l] = X_l of the partially reactive site."""
    return None
```

### Step 8

radiation_islae_regularity

Goal
----
Diagnose why the radiation problem is not reduced by Minkov's method, from the coefficient sequence q_l, the Minkov kernel Q_lm of the same order and the Damkohler number. Insert the radiation condition directly into the canonical dual series relations: the cap relation then keeps the canonical form sum over l of (1 - q_l^Da) X_l P_l(cos theta) = 1 with a modified coefficient q_l^Da that follows from the surface value and the flux density of the previous step (derive it), while the inert-part relation is unchanged, so that Minkov's reduction applies formally with q_l replaced by q_l^Da. Return the sup-norm (largest row sum of absolute values) of the truncated matrix M_lm = q_m^Da Q_lm, l, m = 0..order, and the value of J = X_0 / 2 that the truncated system X_l - sum over m of M_lm X_m = Q_l0 gives when solved by direct elimination at this order (the order is fixed by the length of q). Raise ValueError if q and Q are not of consistent shape or if damkohler is not a finite positive number.

```python
def radiation_islae_regularity(q, Q, damkohler):
    """Diagnose why the radiation problem is not reduced by Minkov's method, from the
    coefficient sequence q_l, the Minkov kernel Q_lm of the same order and the Damkohler
    number. A (2,) float64 array [sup-norm of M at this order, J from the truncated naive
    system]."""
    return None
```

### Step 9

series_resistance_deviation

Goal
----
Quantify how far the textbook series-resistance rule is from the exact rate of a partially reactive site. The reaction-controlled rate of the site is the flux kappa c_B through the area of the cap, which in units of k_S is J_react = Da (1 - cos theta0) / 2. The series-resistance (Collins-Kimball) estimate combines the diffusion-controlled rate of the same site behind the same shell, J_sink, with J_react as resistances in series, 1 / J_CK = 1 / J_sink + 1 / J_react; for an isotropic sphere in unbounded space this rule is exact. Return J_react, J_CK and the signed percentage deviation delta_CK = 100 (J_CK - J_exact) / J_exact of the estimate from the exact rate J_exact of the projected solution. Raise ValueError if j_sink or j_exact is not positive, if theta0 is not in (0, pi], or if damkohler is not a finite positive number.

```python
def series_resistance_deviation(j_sink, theta0, damkohler, j_exact):
    """Quantify how far the textbook series-resistance rule is from the exact rate of a
    partially reactive site. A (3,) float64 array [J_react, J_CK, delta_CK in percent]."""
    return None
```

### Step 10

nanoreactor_audit

Goal
----
Run the complete chain for every nanoreactor design in design_table, a list of rows (R, R0, site_fraction, D, kappa, P, p) in nm, nm, -, nm^2/us, nm/us, nm/us and - with finite kappa (P may be inf), at the given truncation order, and return one table. For each design map the parameters, derive the coefficient sequences, build the Minkov kernel, solve the truncated perfect-sink system from q and the kernel and assemble its summary, reconstruct the perfect-sink field at the two mid-shell probe points xi_mid = (1 + 1 / eps) / 2 on the site axis (theta = 0) and at the inert pole (theta = pi) together with the fraction of the capture entering through the inner half-angle sub-cap 0 <= theta <= theta0 / 2 and the fraction of the shell entry flux through the hemisphere facing the site, solve the projected radiation system from q, theta0 and Da, evaluate the naive-reduction diagnostic from q, the kernel and Da, the series-resistance deviation and, from the radiation solution, the same half-angle capture fraction. Each design row holds the 29 columns [eps, h, theta0, Da, Bi, p, k_S, J_shell, J_iso, J_sink, J(0), J(1), delta1, sup-norm of M, f(theta0), C, F_half of the sink, u at the axis probe, u at the pole probe, G of the sink, J_Da of the projected system, sup-norm of the naive matrix, J of the naive system, J_react, J_CK, delta_CK, k = k_S J_Da in nm^3/us, F_half of the radiation solution, J_Da / J_sink]. Row 0 is the head row [J_Da / J_sink of the first design, number of designs, order, sum of delta_CK over the designs, zeros]. Raise ValueError if design_table is empty or a row does not have seven entries, or if order is not a positive integer.

```python
def nanoreactor_audit(design_table, order):
    """Run the complete chain for every nanoreactor design in design_table, a list of rows (R,
    R0, site_fraction, D, kappa, P, p) in nm, nm, -, nm^2/us, nm/us, nm/us and - with finite
    kappa (P may be inf), at the given truncation order, and return one table. A (1 +
    n_designs, 29) float64 array; row 0 is the head row and row i the row of design i, with
    the columns listed in the description."""
    return None
```

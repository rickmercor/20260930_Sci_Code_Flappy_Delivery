# Chemistry-Quantum_Chemistry-23

## Background

The adiabatic connection links the non-interacting Kohn-Sham system to the
physical system at a fixed density. Its integrand $W_\alpha[n]$ gives the
exchange-correlation energy when it is integrated over the coupling strength
$\alpha$ from 0 to 1. Interaction-strength interpolation functionals build
$W_\alpha$ from its two limits. At weak coupling the limit comes from
exchange and second-order perturbation theory. At strong coupling the
electrons become strictly correlated: they minimize their repulsion at the
given density, and $W_\alpha\to W_\infty[n]+W'_\infty[n]\,\alpha^{-1/2}$. In
the strictly-correlated-electron (SCE) picture, $W_\infty$ is the SCE
interaction energy minus the Hartree energy, and $W'_\infty$ is the energy of
the zero-point oscillations of the electrons around their strictly
correlated positions. $W'_\infty$ is positive for any many-electron density
and zero for a one-electron density.

Exact SCE calculations are costly and exist only for spherical densities and
a few small systems, so practical functionals use semilocal models. The
point-charge-plus-continuum (PC) model of Seidl, Perdew and Kurth treats each
electron as a point charge in a cell of neutralizing background. It gives
local terms proportional to $n^{4/3}$ for $W_\infty$ and $n^{3/2}$ for
$W'_\infty$, with gradient corrections in the reduced gradient $s$. Later
revisions changed the gradient coefficients and bounded the enhancement
factors. These GGA-type models depend on $s$ alone. A meta-GGA also uses the
iso-orbital indicator $z=\tau_W/\tau$, which is 1 wherever one orbital shape
dominates and tends to 0 for slowly varying densities, so it can treat
one- and two-electron regions differently from the slowly varying limit.

Filled hydrogenic shells are simple spherical model densities with closed
forms for the orbitals, the density and the kinetic energy density. A filled
p shell has six electrons and, above the lowest principal quantum number,
radial nodes. Its angular kinetic energy keeps $\tau$ above $\tau_W$, so $z$
lies between 0 and 1 across most of the shell. Under uniform scaling
$n_Z(\mathbf r)=Z^3n_1(Z\mathbf r)$ the quantities $s$ and $z$ do not
change, and $W'_\infty$ scales as $Z^{3/2}$. These densities therefore test
how a semilocal model behaves between its one-orbital and slowly varying
limits.

## Problem

Consider a filled, spin-unpolarized hydrogenic 4p shell with nuclear charge
$Z=10$. It holds six electrons, two of opposite spin in each orbital
$\psi_{41m}(\mathbf r)=R_{41}(r)Y_{1m}(\theta,\phi)$, $m=-1,0,1$, where
$R_{41}$ is the normalized hydrogenic radial function for charge $Z$. Use
this bare model density as it is, with no screening and no self-consistent
field, and use atomic units. The density $n(\mathbf r)$ is the spherical sum
of the six orbital densities, and the relative spin polarization is
$\zeta=0$ everywhere. Take the kinetic energy density as the positive
Kohn-Sham form $\tau=\tfrac12\sum_i|\nabla\phi_i|^2$, summed over all six
occupied spin-orbitals, and take $\tau_W=|\nabla n|^2/(8n)$,
$s=|\nabla n|/\big[2(3\pi^2)^{1/3}n^{4/3}\big]$ and $z=\tau_W/\tau$.

At large coupling strength $\alpha$ the density-fixed adiabatic-connection
integrand behaves as $W_\alpha[n]\to W_\infty[n]+W'_\infty[n]\,\alpha^{-1/2}+\dots$,
and $W'_\infty[n]$ is the zero-point-oscillation energy of the
strictly-correlated-electron limit. Approximate $W'_\infty[n]$ for this shell
with the published enhanced point-and-charge (ePC) meta-GGA model.
This model writes $W'_\infty=\int C\,n^{3/2}F'(s,z,\zeta)\,d^3r$, where
$C=1.535$ (atomic units) is the coefficient of the point-and-charge model and
the enhancement factor $F'$ interpolates between a branch for $z=0$ and a
branch for $z=1$. The model restores the second-order gradient expansion of
the point-and-charge model for slowly varying densities, gives
$W'_\infty\ge0$ for every density, and is exact for one-electron densities.
Report the ePC value of $W'_\infty$, not $W_\infty$ and not an exact
strictly-correlated-electron value, in hartree, as a single number rounded to
three decimal places.

Output Format
Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>.

Rules:
- Put only the requested value rounded to three decimal places between the final-answer tags.
- In <reasoning>, identify the published ePC W'_inf model and show enough intermediate calculations to justify the result, including the shell density/kinetic-energy ingredients, s and z, the ePC W'_inf branch definitions and fitted constants, their z-dependent interpolation, and the spherical radial integral.
- A compact derivation is sufficient; do not paste arrays or per-grid-point tables.

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

Filled hydrogenic p-shell semilocal ingredients.

Goal
----
For a completely filled, spin-unpolarized hydrogenic p shell with principal quantum number 2, 3 or 4 and nuclear charge Z, evaluate the normalized radial function and derivative, the spherical density and derivative, the positive kinetic-energy density, and the semilocal ingredients needed by the later ePC steps.

```python
def p_shell_semilocal_ingredients(n_principal: int, Z: float, r: "np.ndarray") -> "np.ndarray":
    """Return R, dR/dr, n, dn/dr, tau, tau_W, s and z on positive radii.

    Atomic units. ``n_principal`` must be 2, 3 or 4, ``Z`` must be positive,
    and every radius must be strictly positive. Evaluate normalized hydrogenic
    radial functions and their analytic radial derivatives for the requested shell.
    Where n>=1e-30 compute the requested standard semilocal ingredients and cap z at 1; otherwise set
    tau_W=s=z=0. Return rows [R,dR,n,dn,tau,tau_W,s,z].
    """
    return None
```

### Step 2

Point-charge-plus-continuum reference strong-interaction functionals.

Goal
----
Given a spherical, spin-unpolarized density and its radial

derivative on a radial grid, return the two strong-interaction functionals

W_inf[n] and W'_inf[n] in the original gradient-corrected

point-charge-plus-continuum (PC) model.

```python
def pc_strong_interaction(r: "np.ndarray", n: "np.ndarray", dn: "np.ndarray") -> tuple[float, float]:

    """Return (W_inf, W'_inf) in hartree from the gradient-corrected PC model.



    Inputs are arrays on a common, strictly increasing radial grid ``r``

    (bohr): the spherical, spin-summed density ``n`` and its radial

    derivative ``dn``. Use the original gradient-corrected PC model with
    A=-1.451, B=5.317e-3, C=1.535 and D=-0.02558. Evaluate each radial
    integral with the trapezoidal rule on the supplied grid points. Grid

    points where n < 1e-30 contribute zero. Use the published integrands, with no cap on the gradient terms.



    Expected return: a tuple ``(W_inf, W_prime_inf)`` of two floats in

    hartree. Raise ``ValueError`` if the arrays differ in length or if

    ``r`` is not strictly increasing.

    """

    return (0.0, 0.0)
```

### Step 3

ePC W_inf enhancement-factor branches from the primary paper.

Goal
----
For supplied reduced-gradient values s, return the two enhancement-factor branches used by the primary ePC meta-GGA model for W_inf: the slowly-varying z=0 branch and the iso-orbital z=1 branch.

```python
def epc_w_inf_branches(s: "np.ndarray") -> "np.ndarray":
    """Return the published ePC W_inf branches F0(s) and F1(s).

    ``s`` is a finite nonnegative array. Recover the two branch definitions,
    constants and limiting behavior from the primary ePC paper. Return a
    float array of shape (2,len(s)) whose first row is the z=0 branch F0 and
    second row is the z=1 branch F1. Raise ValueError for negative or
    nonfinite s.
    """
    return None
```

### Step 4

ePC W_inf functional from the primary-paper interpolation.

Goal
----
Given a spherical spin-unpolarized density, its reduced gradient and iso-orbital indicator on a radial grid, evaluate W_inf with the full primary-paper ePC meta-GGA interpolation between the two branches from the previous step.

```python
def epc_w_inf(r: "np.ndarray", n: "np.ndarray", s: "np.ndarray", z: "np.ndarray") -> float:
    """Return W_inf[n] in hartree from the published ePC meta-GGA.

    Inputs share a strictly increasing radial grid. ``n`` is nonnegative,
    ``s`` is the reduced gradient, and ``z`` is the iso-orbital indicator in
    [0,1]. Use ``epc_w_inf_branches`` for the two paper-defined branches and
    recover from the primary paper the z-interpolation that forms the full
    enhancement factor. Use A=-1.451 and integrate
    4*pi*r^2*A*n^(4/3)*F by the trapezoidal rule; n<1e-30 contributes zero.
    """
    return 0.0
```

### Step 5

ePC Wprime-infinity enhancement-factor branches from the primary paper.

Goal
----
For supplied reduced-gradient values s and relative spin polarization zeta, return the two enhancement-factor branches used by the primary ePC meta-GGA model for W'_inf.

```python
def epc_wprime_branches(s: "np.ndarray", zeta: float = 0.0) -> "np.ndarray":
    """Return the published ePC W'_inf branches F0'(s) and F1'(s,zeta).

    ``s`` must be finite and nonnegative and ``zeta`` must lie in [-1,1].
    Recover the branch definitions, the restored gradient coefficient, fitted
    constants and spin factor from the primary ePC paper. Return shape
    (2,len(s)), first row F0' and second row F1'.
    """
    return None
```

### Step 6

ePC Wprime-infinity functional from the primary-paper interpolation.

Goal
----
Given a spherical spin-unpolarized density, reduced gradient and iso-orbital indicator on a radial grid, evaluate W'_inf with the full primary-paper ePC interpolation between the branches from the previous step.

```python
def epc_wprime_inf(r: "np.ndarray", n: "np.ndarray", s: "np.ndarray", z: "np.ndarray") -> float:
    """Return W'_inf[n] in hartree from the published ePC meta-GGA.

    Inputs share a strictly increasing radial grid. The density is
    spin-unpolarized (zeta=0). Use ``epc_wprime_branches`` and recover from
    the primary paper the z-dependent interpolation between its two branches.
    Use C=1.535 and integrate 4*pi*r^2*C*n^(3/2)*F' by the trapezoidal rule;
    n<1e-30 contributes zero.
    """
    return 0.0
```

### Step 7

Filled-shell ePC audit and final W'_inf orchestrator.

Goal
----
For a filled, spin-unpolarized hydrogenic p shell, run the complete reference-plus-ePC pipeline on the fixed radial grid and return the primary-paper ePC W'_inf result.

```python
def p_shell_epc_wprime(n_principal: int, Z: float) -> float:
    """Return the primary-paper ePC W'_inf of a filled hydrogenic p shell.

    Use M=20001 radii geomspaced from 1e-6/Z to
    60*n_principal^2/Z bohr. Run ``p_shell_semilocal_ingredients``, the
    companion ``pc_strong_interaction`` reference, ``epc_w_inf`` and
    ``epc_wprime_inf``. The latter two reach their paper-defined branch
    functions in Steps 3 and 5. Require both returned PC reference values to be finite; otherwise raise ValueError.
    Require ePC W_inf<0 and W'_inf>=0 and return
    W'_inf.
    """
    return 0.0
```

# Material_Science-Semiconductor_Materials-50

## Background

Quasi-one-dimensional materials, that is atomic chains, nanowires and narrow ribbons in which electrons and holes move along a single direction, have become an active field since stable single-chain compounds such as S3, Te3, As2S3, Bi2Te3 and MgN2 were identified as exfoliable from van der Waals crystals. Their optical response is dominated by excitons with binding energies that can exceed one electronvolt, which determines how such wires absorb and emit light and whether they can serve as one-dimensional emitters or photodetectors. The quantitative tool for such excitons is many-body perturbation theory (GW quasiparticle bands plus the Bethe-Salpeter equation), but these calculations are expensive and give little insight into what controls the binding. For two-dimensional sheets the gap between ab initio theory and intuition is bridged by the Rytova-Keldysh potential, a screened electron-hole interaction governed by a single sheet polarizability. No comparable standard potential existed for one-dimensional systems.

A collaboration of theoretical groups in Jena and Rome set out to construct one. Starting from the Poisson equation with an induced charge proportional to a static one-dimensional polarizability, a lateral distribution of the induced charge and the second derivative of the potential along the wire, they solve the resulting integral equation for the potential averaged over the lateral distribution and obtain the screened interaction in Fourier space as the averaged bare interaction divided by an effective one-dimensional dielectric function. The construction is not unique: different assumptions about the lateral extent of the induced charge and about the nonlocality of the screening response give a family of model potentials, which the paper works out in closed form and compares in detail, uncovering along the way a wave-vector dependence of the wire dielectric function that sets it apart from its two-dimensional counterpart and an unexpected long-range feature of the screened interaction.

The second part applies these potentials to Wannier-Mott excitons in the effective-mass approximation. A one-parameter variational treatment gives the binding energy by maximising an energy functional in momentum space. With wire radii, reduced masses and polarizabilities computed from density functional theory for the five chains, the variational binding energies range from 0.6 to 4.6 eV, and the paper places them alongside its own ab initio many-body results for the same chains. The framework lets exciton binding energies of new one-dimensional materials be estimated from three material parameters in seconds.

## Problem

Atomically thin semiconductor chains such as S3, Te3 or As2S3 wires bind excitons with energies of the order of an electronvolt, far above bulk values, because the electron-hole attraction is screened by the one-dimensional electron gas of the wire itself. A continuum description treats the chain as a cylinder of radius R along the chain axis z whose induced charge density is the product of a static one-dimensional polarizability alpha_1D, a lateral distribution function of the induced charge normalised per unit area, and the second z-derivative of the electrostatic potential, and defines the one-dimensional electron-hole interaction as the potential of a point charge averaged over that same lateral distribution. The inputs are R, alpha_1D and the interband reduced mass mu; the output is an exciton binding energy.

For an induced charge spread homogeneously over the cross section the source paper works out the averaged bare interaction and the effective one-dimensional dielectric function in closed form, and defines two treatments of the lateral integral equation for the potential, a local one and a partially nonlocal one, that differ only in the nonlocality form factor of the screening response. The exciton is the ground state of the effective-mass two-particle problem with the statically screened interaction, computed with the source paper's one-parameter variational scheme; the binding energy is the maximum of the momentum-space energy functional over the variational parameter.

In the reasoning, report the scalars that determine the result: the bare interaction and the partially nonlocal form factor of the homogeneous distribution at the wave vector q = 1 per A; the optimised variational parameter and binding energy of both treatments; the kernel integral of the variational functional at the partially nonlocal optimum; and the correction itself with its size relative to the binding energy. State the sign of the correction and the mechanism that sets it, and the long-wavelength limit of the wire dielectric function with the physical reason for it. The same construction extends to the source paper's other lateral distributions of the induced charge, and the reasoning must report the computed cross-checks that validate the pipeline: the binding energies, optimised variational parameters and exciton radii of all seven model potentials of the source paper at this wire, with energies and parameters to two decimals and radii to one (the ribbon model has only the local treatment and uses the effective radius sqrt(L^2 + h^2)/2 with L = 3.33 A and h = 3.12 A; the surface-localised and centre-peaked distributions have both treatments, the centre-peaked one at its default cutoff); the nonlocality corrections of the surface-localised and centre-peaked distributions at this wire; the correction obtained as a wiring control when the surface-distribution form factor is used together with the homogeneous bare interaction; the centre-peaked correction, with its sign, at the control parameters R = 4.02 A, mu = 0.11 free-electron masses and alpha_1D = 51.63 A^2 with the cutoff z0 = R; and the surface-localised and centre-peaked corrections at the Te3 parameters R = 2.13 A, mu = 0.30 free-electron masses and alpha_1D = 17.43 A^2. Also report, from the source paper, its ab initio GW plus Bethe-Salpeter binding energies for the S3 chain. The construction has a two-dimensional counterpart in the literature on dielectric screening in two-dimensional insulators; from the first-principles study of graphane that derives the analytic two-dimensional screened potential, report the exciton binding energy of that model, the ab initio GW plus Bethe-Salpeter value it reproduces, and the ground-state binding energies it gives for a donor and for an acceptor impurity. The items just listed are what the format instructions below mean by the few scalars that determine the final number, so include them all in the reasoning; a short line for each fits comfortably inside the few hundred words that the format instructions allow. Use R = 2.28 A, mu = 0.39 free-electron masses, alpha_1D = 12.06 A^2, e^2 = 14.399645 eV A, a_B = 0.529177 A and R_H = 13.605693 eV. Report the nonlocality correction, defined as the binding energy obtained with the partially nonlocal treatment minus the binding energy obtained with the local treatment, in meV. Your final answer must be a single number: the nonlocality correction in meV.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long
derivation before the tags.
You must emit exactly one finite decimal inside
<final_answer>...</final_answer>, even if the value is approximate or you
are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05).
Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra
lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that
determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration
paths, or per-fold candidate tables.

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

exciton_units

Goal
----
Effective-mass description of a Wannier-Mott exciton in a one-dimensional semiconductor.

The relative electron-hole motion along the chain axis has the kinetic energy

-(hbar^2 / 2 mu) d^2/dz^2 with the interband reduced mass 1 / mu = 1 / m_c + 1 / m_v.

With mu in units of the free-electron mass m, the natural excitonic units are the

exciton Rydberg and the exciton Bohr radius,

```python
def exciton_units(mu: float) -> np.ndarray:
    '''Exciton Rydberg, exciton Bohr radius and bare 1D ground-state separation.

    Parameters
    ----------
    mu : float
        Interband reduced mass in units of the free-electron mass, > 0.

    Returns
    -------
    out : np.ndarray of float, shape (3,)
        [R_exc, a_exc, r_B0]: exciton Rydberg in eV, exciton Bohr radius in Angstrom,
        and the average separation 3 a_exc / 2 of the unscreened 1D ground state in Angstrom.
        Raises ValueError if mu is not a positive finite number.
    '''
    return out
```

### Step 2

bare_potential_fourier

Goal
----
Averaged bare Coulomb interaction of a quasi-one-dimensional wire in Fourier space.

The wire is a cylinder of radius R along z; the induced charge is distributed laterally

with a normalised distribution f(rho), and the one-dimensional interaction is the

electrostatic potential of a point charge averaged over f. In terms of the longitudinal

wave number q this average of the Helmholtz Green's function K_0(|q| rho) gives, with

N_f the normalisation of f,

```python
def bare_potential_fourier(q: "float | np.ndarray", R: float, distribution: str, z0: "float | None" = None) -> np.ndarray:
    '''Laterally averaged bare Coulomb interaction V_bare(q) of a quasi-1D wire.

    Parameters
    ----------
    q : float or array_like of float
        Longitudinal wave number(s) in 1/Angstrom; only |q| enters.
    R : float
        Wire radius (or effective ribbon radius) in Angstrom, > 0.
    distribution : str
        One of 'ribbon', 'surface', 'homogeneous', 'centre'.
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).

    Returns
    -------
    v : np.ndarray of float
        V_bare(q) in 1/Angstrom with the shape of q (+inf where q = 0).
        Raises ValueError for an unknown distribution, R <= 0, z0 <= 0 or non-finite q.
    '''
    return v
```

### Step 3

nonlocal_form_factor

Goal
----
Nonlocality form factor of the one-dimensional screening response. Averaging the

lateral integral equation for the potential over the induced-charge distribution f gives

the screened interaction V(q) = V_bare(q) / [1 + alpha_1D q^2 chi(q) V_bare(q)], where the

form factor chi(q) measures how the Green's function K_0(|q| |rho - rho'|) couples

different lateral positions. In the local treatment the kernel is replaced by its value

at the wire axis and chi reduces to the normalisation N_f of the distribution (4 for the

ribbon, whose four edges carry the charge, and 1 otherwise). The partially nonlocal

treatment replaces the unknown lateral potential in the denominator by its lateral

average and evaluates the remaining double integral over f analytically:

```python
def nonlocal_form_factor(q: "float | np.ndarray", R: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    '''Form factor chi(q) of the 1D screening response.

    Parameters
    ----------
    q : float or array_like of float
        Longitudinal wave number(s) in 1/Angstrom; only |q| enters.
    R : float
        Wire radius in Angstrom, > 0.
    distribution : str
        One of 'ribbon', 'surface', 'homogeneous', 'centre'.
    use_nonlocal : bool
        False returns the local value N_f; True returns the partially nonlocal factor.
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).

    Returns
    -------
    chi : np.ndarray of float
        chi(q) with the shape of q.
        Raises ValueError for an unknown distribution, R <= 0, z0 <= 0, non-finite q,
        or use_nonlocal=True with the ribbon distribution.
    '''
    return chi
```

### Step 4

dielectric_function

Goal
----
Effective one-dimensional dielectric function. With the averaged bare interaction

V_bare(q) and the form factor chi(q) the screening of a quasi-1D semiconductor with

static polarizability alpha_1D (units of length squared) is

```python
def dielectric_function(q: "float | np.ndarray", R: float, alpha: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    '''Effective 1D dielectric function eps(q) = 1 + alpha q^2 chi(q) V_bare(q).

    Parameters
    ----------
    q : float or array_like of float
        Longitudinal wave number(s) in 1/Angstrom; only |q| enters.
    R : float
        Wire radius in Angstrom, > 0.
    alpha : float
        Static 1D polarizability alpha_1D in Angstrom^2, >= 0.
    distribution : str
        One of 'ribbon', 'surface', 'homogeneous', 'centre'.
    use_nonlocal : bool
        Local (False) or partially nonlocal (True) form factor.
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).

    Returns
    -------
    eps : np.ndarray of float
        eps(q) with the shape of q (exactly 1 at q = 0).
        Raises ValueError for invalid distribution, R, z0, alpha < 0, non-finite q,
        or the nonlocal ribbon request.
    '''
    return eps
```

### Step 5

screened_potential_fourier

Goal
----
Statically screened one-dimensional electron-hole interaction in Fourier space,

```python
def screened_potential_fourier(q: "float | np.ndarray", R: float, alpha: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> np.ndarray:
    '''Screened 1D interaction V(q) = V_bare(q) / eps(q).

    Parameters
    ----------
    q : float or array_like of float
        Longitudinal wave number(s) in 1/Angstrom; only |q| enters.
    R : float
        Wire radius in Angstrom, > 0.
    alpha : float
        Static 1D polarizability alpha_1D in Angstrom^2, >= 0.
    distribution : str
        One of 'ribbon', 'surface', 'homogeneous', 'centre'.
    use_nonlocal : bool
        Local (False) or partially nonlocal (True) form factor.
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).

    Returns
    -------
    v : np.ndarray of float
        V(q) in 1/Angstrom with the shape of q (+inf at q = 0).
        Raises ValueError for invalid distribution, R, z0, alpha or q.
    '''
    return v
```

### Step 6

variational_kernel_integral

Goal
----
Momentum-space expectation value of the screened interaction in the variational exciton

state. The trial function has the shape of the one-dimensional hydrogen ground state with

the exciton Bohr radius divided by the variational parameter lambda,

```python
def variational_kernel_integral(lam: float, R: float, alpha: float, mu: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> float:
    '''Kernel integral I(lambda) = int_0^inf dt (1 - 3 t^2) / (1 + t^2)^3 V(2 lambda t / a_exc).

    Parameters
    ----------
    lam : float
        Variational parameter lambda, > 0.
    R : float
        Wire radius in Angstrom, > 0.
    alpha : float
        Static 1D polarizability in Angstrom^2, >= 0.
    mu : float
        Interband reduced mass in free-electron masses, > 0.
    distribution : str
        One of 'ribbon', 'surface', 'homogeneous', 'centre'.
    use_nonlocal : bool
        Local (False) or partially nonlocal (True) form factor.
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).

    Returns
    -------
    val : float
        The dimensionless kernel integral I(lambda) (V in 1/Angstrom times the Angstrom
        from the substitution cancels).
        Raises ValueError for lam <= 0 or invalid wire parameters.
    '''
    return val
```

### Step 7

variational_binding_energy

Goal
----
Variational binding energy as a function of the variational parameter. The kinetic

energy of the trial state is R_exc lambda^2 (the 1D hydrogen value scaled by lambda^2)

and the potential energy is -R_exc lambda (4 / pi) I(lambda), so the binding energy,

defined as the gap minus the lowest pair energy, is

```python
def variational_binding_energy(lam: float, R: float, alpha: float, mu: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None) -> float:
    '''E_B(lambda) = R_exc [-lambda^2 + lambda (4 / pi) I(lambda)] in eV.

    Parameters
    ----------
    lam : float
        Variational parameter lambda, > 0.
    R : float
        Wire radius in Angstrom, > 0.
    alpha : float
        Static 1D polarizability in Angstrom^2, >= 0.
    mu : float
        Interband reduced mass in free-electron masses, > 0.
    distribution : str
        One of 'ribbon', 'surface', 'homogeneous', 'centre'.
    use_nonlocal : bool
        Local (False) or partially nonlocal (True) form factor.
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).

    Returns
    -------
    e_b : float
        Variational binding energy in eV at this lambda.
        Raises ValueError for lam <= 0 or invalid wire parameters.
    '''
    return e_b
```

### Step 8

optimise_exciton

Goal
----
Variational optimum. The exciton binding energy is the maximum of E_B(lambda) over the

variational parameter; the optimum lambda_0 gives the average electron-hole separation

r_B = 3 a_exc / (2 lambda_0). E_B(lambda) is smooth and unimodal on the physical range, so

a bounded scalar maximisation (golden-section / Brent) on a fixed interval is sufficient;

the default bounds 0.02 <= lambda <= 3 cover excitons from very extended to strongly

compressed. Because E_B is stationary at the optimum, the binding energy is insensitive to

the last digits of lambda_0, whereas lambda_0 and r_B themselves are determined to about

six significant figures. Invalid bounds (lower >= upper or lower <= 0) are rejected.

```python
def optimise_exciton(R: float, alpha: float, mu: float, distribution: str, use_nonlocal: bool = True, z0: "float | None" = None, lam_bounds: tuple = (0.02, 3.0)) -> np.ndarray:
    '''Maximise E_B(lambda); return the optimum lambda_0, the binding energy and the exciton radius.

    Parameters
    ----------
    R : float
        Wire radius in Angstrom, > 0.
    alpha : float
        Static 1D polarizability in Angstrom^2, >= 0.
    mu : float
        Interband reduced mass in free-electron masses, > 0.
    distribution : str
        One of 'ribbon', 'surface', 'homogeneous', 'centre'.
    use_nonlocal : bool
        Local (False) or partially nonlocal (True) form factor.
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).
    lam_bounds : tuple of float
        Search interval (lower, upper) for lambda with 0 < lower < upper.

    Returns
    -------
    out : np.ndarray of float, shape (3,)
        [lambda_0, E_B in eV, r_B in Angstrom] at the variational optimum.
        Raises ValueError for invalid bounds or wire parameters.
    '''
    return out
```

### Step 9

nonlocality_correction

Goal
----
Final orchestrator. The chain is: excitonic units from the reduced mass (step 1); the

averaged bare interaction of the chosen lateral distribution (step 2); the local form

factor N_f and the partially nonlocal form factor (step 3); the dielectric function and

the screened interaction (steps 4 and 5); the kernel integral and the variational energy

(steps 6 and 7); and the bounded maximisation over lambda (step 8). The orchestrator calls

every earlier step directly. It obtains the excitonic units, evaluates the bare

interaction, the nonlocal form factor, the dielectric function and the screened

interaction at a reference wave vector and verifies the defining relations

eps = 1 + alpha_1D q^2 chi V_bare and V eps = V_bare, runs the optimisation twice for the

same wire, once with the local form factor and once with the partially nonlocal one,

cross-checks each returned radius against the units through r_B lambda_0 = 3 a_exc / 2,

re-evaluates the kernel integral and the variational energy at each returned lambda_0

against the optimiser energy, and returns the nonlocality correction to the exciton

binding energy

```python
def nonlocality_correction(R: float, alpha: float, mu: float, distribution: str = 'homogeneous', z0: "float | None" = None) -> float:
    '''Nonlocality correction Delta E = E_B(nonlocal) - E_B(local) of the exciton binding energy in meV.

    Calls every earlier step directly. Obtains the excitonic units from exciton_units;
    evaluates bare_potential_fourier, nonlocal_form_factor, dielectric_function and
    screened_potential_fourier at the reference wave vector q = 1 per Angstrom and checks
    eps = 1 + alpha q^2 chi V_bare and V eps = V_bare; runs optimise_exciton twice, once
    with the local and once with the partially nonlocal form factor; cross-checks each
    returned radius against the units (r_B lambda_0 = 3 a_exc / 2); and re-evaluates
    variational_binding_energy and variational_kernel_integral at each returned lambda_0,
    requiring both to reproduce the optimiser energy through
    E_B = R_exc (-lambda_0^2 + lambda_0 (4/pi) I). Any failed check raises ValueError.

    Parameters
    ----------
    R : float
        Wire radius in Angstrom, > 0.
    alpha : float
        Static 1D polarizability in Angstrom^2, >= 0.
    mu : float
        Interband reduced mass in free-electron masses, > 0.
    distribution : str
        One of 'surface', 'homogeneous', 'centre' ('ribbon' is rejected).
    z0 : float or None
        Cutoff length of the 'centre' distribution in Angstrom; None selects 2 R / sqrt(5).

    Returns
    -------
    delta_e : float
        E_B(nonlocal) - E_B(local) in meV.
        Raises ValueError for the ribbon distribution or invalid wire parameters.
    '''
    return delta_e
```

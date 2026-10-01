# Chemistry-Computational_Chemistry-27

## Background

Protein solutions are among the few condensed-phase systems where a handful of effective interaction parameters controls a rich and technologically important phase diagram. When water and the small ions are treated as a structureless background, a globular protein behaves as a colloidal particle whose pair interaction is short ranged compared with its diameter, and the whole phase diagram of the solution can then be organised around that single effective potential. Two boundaries matter in practice. The crystal solubility line marks equilibrium between the dilute solution and the ordered lattice and is what a crystallographer manipulates when growing diffraction-quality crystals. The liquid–liquid phase separation line marks demixing of the solution into coexisting protein-poor and protein-rich fluids; it usually sits below the solubility line, so it is metastable with respect to crystallisation, and the dense droplets it produces are known to act as nurseries in which crystal nuclei form far faster than they do in the bulk dilute solution. The vertical distance between the two boundaries, the metastability gap, therefore controls crystallisation yield, and the same physics governs the biomolecular condensates that organise the interior of living cells.

Because the interaction range is short, the location of both boundaries is exquisitely sensitive to the depth and the angular character of protein–protein contacts, and neither is directly measurable. The standard route is thermodynamic perturbation theory: a hard-sphere reference free energy is corrected by an attractive perturbation whose leading term counts neighbours inside an interaction well and whose next term accounts for the density fluctuations around a central particle. That machinery was built for isotropic fluids, whereas real protein contacts are patchy, involving a small number of specific surface regions rather than the whole surface. Patchiness has an entropic price that is paid in the fluid, where the molecules rotate freely, but not in the crystal, where the lattice fixes every orientation. This asymmetry is why an additive can act in opposite directions on the two boundaries at once: by strengthening a specific lattice contact it lowers solubility, while by concentrating the attraction into a smaller surface patch it weakens the orientationally averaged attraction that drives demixing.

Turning that picture into numbers exposes a known weakness of the perturbation expansion. The conventional second-order term is derived by expanding the Boltzmann weight of the well to quadratic order, which is accurate at high temperature but leaves the theory inconsistent with the second virial coefficient, the one interaction measure that can be obtained directly from light scattering or from the concentration dependence of the diffusion coefficient. Since the second virial coefficient is the quantity experimentalists actually use to rank crystallisation conditions, a model of protein phase behaviour that misses it is being calibrated against the wrong yardstick. Repairing that inconsistency shifts computed coexistence curves by several percent, which is the same order as the additive effects the model is meant to explain, so the repair is not cosmetic. The work behind this task assembles the repaired perturbation theory, an orientational average that carries the patchiness, and a cell-theory description of the crystal into a single framework in which solubility and demixing are computed from one consistent set of interaction parameters.

## Problem

An aqueous protein solution can be driven either to liquid–liquid phase separation or to crystallisation, and a buffer additive is sometimes observed to move those two boundaries in opposite directions: the additive lowers the crystal solubility, which says that contacts in the lattice have become stronger, yet it also lowers rather than raises the temperature at which the solution demixes. Resolving that requires a one-component colloid description in which water and every low-molecular-weight cosolute are absorbed into the background, so that the protein osmotic pressure plays the role of a gas pressure and demixing becomes a gas–liquid transition of an effective square-well fluid. Your task is to build that description for the two systems specified below and to return one deterministic number.

Represent the protein as a square-well colloid of reduced range λ = 1.3 whose reduced Helmholtz free energy is a Carnahan–Starling hard-sphere reference plus a residual attraction treated by thermodynamic perturbation theory carried to second order in the local-compressibility approximation. Three modelling decisions control the result. The residual expansion is anchored on the average number of neighbours lying inside the well, which is obtained from the Carnahan–Starling contact value evaluated not at the true packing fraction but at a published range-dependent effective packing fraction written as a Padé form. The second-order contribution must be expressed so that the model reproduces the exact second virial coefficient of the square-well fluid, which the naive truncation of the perturbation series does not do. Finally, protein–protein attraction is directional: a fraction α of each particle surface carries binding sites of depth ε_SW and the remaining fraction carries none, so the isotropic depth must be replaced by an orientationally averaged, temperature-dependent effective depth, with the convention that one contact is shared between two proteins.

The demixing dome closes at a critical point, where the first and second volume-fraction derivatives of the reduced chemical potential vanish together, and that point bounds the temperature range over which a protein-rich phase exists at all. The crystal is taken as incompressible, so its protein chemical potential follows from cell theory as a lattice-contact energy term plus a residual phase-volume entropy term, and the solubility is the fluid volume fraction whose chemical potential matches it. Work with k_B = 1 so that all energies are in kelvin, set the standard-state chemical potential to zero, and use the reduced variables in which composition is the protein volume fraction φ. Use the following configuration:

- λ = 1.3 for both systems
- System A (reference): ε_SW = 1687 K, α = 0.0372
- System B (with additive): ε_SW = 1732 K, α = 0.0293
- crystal parameters for both systems: n_S = 6 contact regions, ε_S = ε_SW, Ω_S = 2.6 × 10⁻⁶
- reporting temperature 298.15 K, working temperature 250.00 K

Report the average number of neighbours lying inside the well at a protein volume fraction of 0.20; the effective well depth in kelvin at 298.15 K for System A and for System B; the critical protein volume fraction and the critical demixing temperature in kelvin for System A and for System B; the two coexisting protein volume fractions of the liquid–liquid boundary at 250.00 K for System A and for System B; and the crystal solubility volume fraction at 298.15 K for System A and for System B. Your final answer must be a single number: for System B at 250.00 K, the reduced chemical-potential driving force for crystallisation out of the protein-rich coexisting phase, that is the reduced protein chemical potential of the protein-rich coexisting phase minus the reduced protein chemical potential of the fluid that is saturated with respect to the crystal at that same temperature. In the reasoning give those scalars, and name in one line each relation and selection rule you used to obtain them, since together they are the scalars that determine the final number.

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

01_pade_effective_volume_fraction

Goal
----
Step 01 - Effective volume fraction of the Pade contact-value map.

```python
def pade_effective_volume_fraction(phi: float, lam: float) -> float:
    '''Effective volume fraction entering the hard-sphere contact value.

    Parameters
    ----------
    phi : float
        Protein volume fraction of the fluid phase, 0 < phi < 1.
    lam : float
        Reduced square-well range lambda, lam > 1.

    Returns
    -------
    phi_eff : float
        Effective volume fraction phi' as a native Python float.

    Raises
    ------
    ValueError
        If phi is not strictly inside (0, 1), or if lam <= 1, or if either
        argument is not finite.
    '''
    return phi_eff  # placeholder
```

### Step 2

02_contact_number_derivatives

Goal
----
Step 02 - Average number of well neighbours and its first four derivatives.

```python
def contact_number_derivatives(phi: float, lam: float) -> "np.ndarray":
    '''Average well-neighbour count nu_HS and its first four phi-derivatives.

    Parameters
    ----------
    phi : float
        Protein volume fraction of the fluid phase, 0 < phi < 1.
    lam : float
        Reduced square-well range lambda, lam > 1.

    Returns
    -------
    out : np.ndarray
        Array of shape (5,), [nu_HS, d1, d2, d3, d4], where dk is the k-th
        derivative of nu_HS with respect to phi.

    Raises
    ------
    ValueError
        If phi is not strictly inside (0, 1), or if lam <= 1, or if either
        argument is not finite, or if the effective volume fraction reaches 1
        so that the contact value diverges.
    '''
    return out  # placeholder
```

### Step 3

03_hard_sphere_reference

Goal
----
Step 03 - Carnahan-Starling hard-sphere reference terms.

```python
def hard_sphere_reference(phi: float) -> "np.ndarray":
    '''Carnahan-Starling reduced free energy, potentials and mu-derivatives.

    Parameters
    ----------
    phi : float
        Protein volume fraction of the fluid phase, 0 < phi < 1.

    Returns
    -------
    out : np.ndarray
        Array of shape (5,), [a_HS, mu_HS, pi_HS, d(mu_HS)/d(phi),
        d^2(mu_HS)/d(phi)^2], with the standard-state constant mu0 set to zero.

    Raises
    ------
    ValueError
        If phi is not strictly inside (0, 1) or is not finite.
    '''
    return out  # placeholder
```

### Step 4

04_effective_well_depth

Goal
----
Step 04 - Anisotropy-corrected effective well depth.

```python
def effective_well_depth(beta_eps_sw: float, alpha: float) -> float:
    '''Dimensionless anisotropy-corrected well depth beta * eps_eff.

    Parameters
    ----------
    beta_eps_sw : float
        Dimensionless bare square-well depth beta * eps_SW, must be >= 0.
    alpha : float
        Degeneracy factor, the fraction of particle surface carrying binding
        sites, 0 < alpha <= 1.

    Returns
    -------
    beta_eps_eff : float
        Dimensionless effective well depth beta * eps_eff as a native Python float.

    Raises
    ------
    ValueError
        If beta_eps_sw is negative or not finite, or if alpha is outside (0, 1],
        or if alpha is not finite.
    '''
    return beta_eps_eff  # placeholder
```

### Step 5

05_mean_field_term

Goal
----
Step 05 - First-order mean-field perturbation term.

```python
def mean_field_term(phi: float, lam: float, beta_eps_eff: float) -> "np.ndarray":
    '''First-order mean-field free-energy term and its first three phi-derivatives.

    Parameters
    ----------
    phi : float
        Protein volume fraction of the fluid phase, 0 < phi < 1.
    lam : float
        Reduced square-well range lambda, lam > 1.
    beta_eps_eff : float
        Dimensionless effective well depth beta * eps_eff, must be >= 0.

    Returns
    -------
    out : np.ndarray
        Array of shape (4,), [a_mf, d1, d2, d3], where dk is the k-th derivative
        of a_mf with respect to phi.

    Raises
    ------
    ValueError
        If phi is not strictly inside (0, 1), or lam <= 1, or beta_eps_eff is
        negative, or any argument is not finite.
    '''
    return out  # placeholder
```

### Step 6

06_second_order_term

Goal
----
Step 06 - Second-order residual term under the local-compressibility approximation.

```python
def second_order_term(phi: float, lam: float, beta_eps_eff: float) -> "np.ndarray":
    '''Second-order residual free-energy term and its first three phi-derivatives.

    Parameters
    ----------
    phi : float
        Protein volume fraction of the fluid phase, 0 < phi < 1.
    lam : float
        Reduced square-well range lambda, lam > 1.
    beta_eps_eff : float
        Dimensionless effective well depth beta * eps_eff, must be >= 0.

    Returns
    -------
    out : np.ndarray
        Array of shape (4,), [a_R2, d1, d2, d3], where dk is the k-th derivative
        of a_R2 with respect to phi.

    Raises
    ------
    ValueError
        If phi is not strictly inside (0, 1), or lam <= 1, or beta_eps_eff is
        negative, or any argument is not finite.
    '''
    return out  # placeholder
```

### Step 7

07_reduced_potentials

Goal
----
Step 07 - Assembled reduced potentials and their volume-fraction derivatives.

```python
def reduced_potentials(phi: float, lam: float, beta_eps_sw: float,
                       alpha: float) -> "np.ndarray":
    '''Reduced free energy, potentials and the two mu-derivatives of the fluid.

    Parameters
    ----------
    phi : float
        Protein volume fraction of the fluid phase, 0 < phi < 1.
    lam : float
        Reduced square-well range lambda, lam > 1.
    beta_eps_sw : float
        Dimensionless bare square-well depth beta * eps_SW, must be >= 0.
    alpha : float
        Degeneracy factor, 0 < alpha <= 1.

    Returns
    -------
    out : np.ndarray
        Array of shape (5,), [a, mu, pi, d(mu)/d(phi), d^2(mu)/d(phi)^2], with
        the standard-state constant mu0 set to zero.

    Raises
    ------
    ValueError
        If phi is not strictly inside (0, 1), or lam <= 1, or beta_eps_sw is
        negative, or alpha is outside (0, 1], or any argument is not finite.
    '''
    return out  # placeholder
```

### Step 8

08_critical_point

Goal
----
Step 08 - Critical point of the liquid-liquid demixing boundary.

```python
def critical_point(lam: float, eps_sw: float, alpha: float) -> "np.ndarray":
    '''Critical volume fraction and critical temperature of the demixing dome.

    Parameters
    ----------
    lam : float
        Reduced square-well range lambda, lam > 1.
    eps_sw : float
        Bare square-well depth in kelvin, must be > 0.
    alpha : float
        Degeneracy factor, 0 < alpha <= 1.

    Returns
    -------
    out : np.ndarray
        Array of shape (2,), [phi_c, T_c], the critical protein volume fraction
        and the critical temperature in kelvin.

    Raises
    ------
    ValueError
        If lam <= 1, or eps_sw <= 0, or alpha is outside (0, 1], or either
        argument is not finite, or no critical point can be bracketed between
        eps_sw / 50 and 5 * eps_sw in temperature.
    '''
    return out  # placeholder
```

### Step 9

09_llps_binodal

Goal
----
Step 09 - Liquid-liquid coexistence (binodal) volume fractions.

```python
def llps_binodal(lam: float, eps_sw: float, alpha: float,
                 temperature: float) -> "np.ndarray":
    '''Coexisting protein volume fractions of the liquid-liquid binodal.

    Parameters
    ----------
    lam : float
        Reduced square-well range lambda, lam > 1.
    eps_sw : float
        Bare square-well depth in kelvin, must be > 0.
    alpha : float
        Degeneracy factor, 0 < alpha <= 1.
    temperature : float
        Absolute temperature in kelvin, must be > 0.

    Returns
    -------
    out : np.ndarray
        Array of shape (2,), [phi_I, phi_II], the protein-poor and protein-rich
        coexisting volume fractions, with phi_I < phi_II.

    Raises
    ------
    ValueError
        If lam <= 1, or eps_sw <= 0, or alpha is outside (0, 1], or temperature
        <= 0, or any argument is not finite, or the state point admits no
        liquid-liquid coexistence that can be resolved on the volume-fraction
        window (0, 0.74). The latter happens at or above the critical
        temperature, where the loop disappears, and also for extremely deep
        quenches, where the protein-poor branch underflows below the window.
    '''
    return out  # placeholder
```

### Step 10

10_crystal_solubility

Goal
----
Step 10 - Crystal solubility from cell theory.

```python
def crystal_solubility(lam: float, eps_sw: float, alpha: float, temperature: float,
                       n_s: float, eps_s: float, ln_omega_s: float) -> float:
    '''Protein volume fraction of the fluid in equilibrium with the crystal.

    Parameters
    ----------
    lam : float
        Reduced square-well range lambda, lam > 1.
    eps_sw : float
        Bare square-well depth of the fluid model in kelvin, must be > 0.
    alpha : float
        Degeneracy factor, 0 < alpha <= 1.
    temperature : float
        Absolute temperature in kelvin, must be > 0.
    n_s : float
        Number of contact regions per protein in the crystal, must be > 0.
    eps_s : float
        Average attraction energy of one crystal contact in kelvin, must be > 0.
    ln_omega_s : float
        Natural logarithm of the residual phase volume Omega_S of the crystal.

    Returns
    -------
    phi_s : float
        Crystal solubility volume fraction, the smallest root on the stable
        dilute branch, as a native Python float.

    Raises
    ------
    ValueError
        If lam <= 1, or eps_sw <= 0, or alpha is outside (0, 1], or temperature
        <= 0, or n_s <= 0, or eps_s <= 0, or any argument is not finite, or no
        locally stable dilute solubility root exists in (0, 0.70).
    '''
    return phi_s  # placeholder
```

### Step 11

11_crystallization_driving_force

Goal
----
Step 11 - ORCHESTRATOR - crystallization driving force out of the dense liquid phase.

```python
def crystallization_driving_force(lam: float, eps_sw: float, alpha: float,
                                  temperature: float, n_s: float, eps_s: float,
                                  ln_omega_s: float) -> float:
    '''Reduced chemical-potential driving force for crystallisation from the dense phase.

    Parameters
    ----------
    lam : float
        Reduced square-well range lambda, lam > 1.
    eps_sw : float
        Bare square-well depth in kelvin, must be > 0.
    alpha : float
        Degeneracy factor, 0 < alpha <= 1.
    temperature : float
        Absolute temperature in kelvin, must be > 0 and below the critical
        temperature of the system.
    n_s : float
        Number of contact regions per protein in the crystal, must be > 0.
    eps_s : float
        Average attraction energy of one crystal contact in kelvin, must be > 0.
    ln_omega_s : float
        Natural logarithm of the residual phase volume Omega_S of the crystal.

    Returns
    -------
    delta_mu : float
        mu(phi_II) - mu(phi_S) at the working temperature, as a native Python float.

    Raises
    ------
    ValueError
        If any argument is outside the domain described above or is not finite,
        if the working temperature is at or above the critical temperature, if
        the state point admits no liquid-liquid coexistence resolvable on the
        volume-fraction window (0, 0.74), or if no crystal solubility root
        exists in (0, 0.70).
    '''
    return delta_mu  # placeholder
```

# Chemistry-Computational_Chemistry-10

## Background

## Setting

Liquid-liquid phase separation is the process by which a homogeneous solution splits into two coexisting liquid phases. In cell biology it produces membraneless organelles, the condensates that concentrate proteins and nucleic acids without a lipid boundary, and its disruption is associated with several pathological states. The same physics is exploited industrially, from encapsulation and drug delivery to food formulation.

When the demixing is driven mainly by electrostatic attraction between oppositely charged macromolecules, it is called complex coacervation: the dense phase is the coacervate, the dilute phase the supernatant. Flory-Huggins theory provides the classical mean-field account of polymer demixing, but it contains no electrostatics at all, which makes it unsuitable for the charged residues that dominate the behaviour of intrinsically disordered proteins and nucleic acids. The minimal theory that does capture the electrostatics dates from the 1950s and remains the reference point for the field: it keeps the Flory-Huggins mixing entropy of the two polymers and the solvent, and adds an attractive term borrowed from Debye-Hückel theory of electrolyte solutions.

## Why this model is analytically awkward

Evaluated in the dilute-electrolyte limit, the Debye-Hückel free energy contributes a term scaling as a non-integer power of the total polymer charge density. That fractional power is the fingerprint of correlated ionic screening, and it is exactly what obstructs analysis. The two conditions that fix coexistence, equality of the exchange chemical potential and equality of the osmotic pressure between the phases, become a pair of transcendental equations mixing logarithms with half-integer powers, and they have no closed-form solution.

The practical consequences are familiar to anyone who has fitted coacervation data. Phase boundaries are computed numerically, by a common-tangent or equal-area construction on the free energy, which is sensitive to the initial bracketing and prone to instability where the dilute branch runs many orders of magnitude below the dense one. Alternatively one expands about the critical point, which is analytic but degrades quickly as the interaction strength grows, and which can return volume fractions that are negative and therefore unphysical.

## The idea that unlocks it

The route pursued here is to stop treating coexistence as a root-finding problem and instead rewrite it as a fixed-point problem. If the two coexistence conditions can be rearranged so that the pair of coexisting volume fractions appears on both sides of the equality, then the pair becomes the fixed point of a map, and Picard iteration from any sufficiently good starting guess converges to it, with local convergence guaranteed when the spectral radius of the map's Jacobian at the fixed point is below one, and global convergence on a region when the map is a contraction there. Truncating the iteration after a small, fixed number of steps then leaves a genuinely analytic expression for the phase boundaries rather than a numerical answer.

Two features make this more than a change of bookkeeping. First, the near-critical expansion, useless on its own away from the critical point, is a perfectly adequate seed for the iteration, so the two approximations are complementary rather than competing. Second, once the coexisting fractions are available in closed form, other observables become analytically accessible too. One of them is the electrostatic potential step that phase separation establishes between the two phases: because the two polymer species partition unequally whenever they differ in chain length or net charge, the coacervate sits at a different potential from the supernatant, and the sign of that difference decides whether cationic or anionic client molecules are drawn into the dense phase. This potential drop is a physical property of the condensate, not an artefact of the theory, and having it in closed form makes its scaling behaviour near the critical point and in the strongly interacting limit available by inspection.

The same strategy has been applied before to the uncharged Flory-Huggins model, which lacks the fractional power and is correspondingly easier; carrying it into the charged case is the step that matters here.

## Problem

Electrostatically driven liquid-liquid phase separation lets a solution of oppositely charged polymers demix into a dense, polymer-rich phase and a dilute supernatant, and it underpins the membraneless condensates that organise material inside living cells. The minimal mean-field treatment of this phenomenon adds, to the translational entropy of two oppositely charged polymer species and a neutral solvent, an attractive electrostatic term taken from Debye-Hückel theory in the dilute-electrolyte limit, so that electrostatics enters as a non-integer power of the total polymer volume fraction. Because of that non-integer power the model admits no closed-form description of phase coexistence, and its phase boundaries have long been obtained either numerically or from expansions that hold only near the critical point. A recent advance removes this restriction by recasting the two coexistence conditions as a fixed-point problem in the pair of coexisting volume fractions, so that analytic phase boundaries follow by iterating from a near-critical starting estimate.

Compute one deterministic observable of that construction. The mixture holds two oppositely charged polymer species in a neutral solvent, is incompressible, and obeys local charge neutrality. Species 1 is the positively charged polymer, with chain length 4.0 and linear charge density 0.80, and species 2 is the negatively charged polymer, with chain length 2.0 and linear charge density 0.85, with chain lengths measured in units of the reference volume used to define volume fractions. Take the dimensionless electrostatic prefactor to be 3.655, so that the electrostatic free energy per reference volume, in units of thermal energy, is minus that prefactor times the three-halves power of the total polymer charge density, the sum over species of linear charge density times volume fraction. Use charge neutrality to reduce the mixture to a single-component free energy in the total polymer volume fraction, locate the critical point of that reduced free energy, and form the starting estimate of the coexisting fractions by expanding the exchange chemical potential about the critical fraction through cubic order and imposing equality of that potential between the two phases.Evaluate all three expansion coefficients, the cubic one included, at interaction parameter of the reduced description, not at the critical interaction parameter.

Advance the coexistence fixed-point map, the form in which the two coexistence conditions are solved for the coexisting fractions themselves so that one iteration carries a trial pair of fractions to a new pair, through exactly two iterations, counting the pair obtained directly from the starting estimate as the first. On the resulting pair, evaluate the electrostatic potential difference that phase separation establishes between the coexisting phases, normalised by the total net charge of the two chains. Report that potential difference in units of thermal energy per elementary charge, taken as the dense phase minus the dilute phase.

In your reasoning, state the map you iterate, how the starting estimate is fed into it, and the expression you use for the potential difference. Give the interaction parameter and effective chain length of the reduced description, the critical fraction and critical interaction parameter, the two fractions of the starting estimate, the coexisting pair at the requested order, and the separate contributions of the charge asymmetry and the length asymmetry of the chains to the potential difference.

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

reduce_polyelectrolyte_pair

Goal
----
Reduce the two-species mixture to its one-component description.

```python
def reduce_polyelectrolyte_pair(r1: float, r2: float, sigma1: float, sigma2: float,
                                alpha: float = 3.655) -> np.ndarray:
    """Collapse the two-species Voorn-Overbeek mixture to one component.

    Parameters
    ----------
    r1, r2 : float
        Chain lengths of species 1 and 2 in units of the reference volume, both > 0.
    sigma1, sigma2 : float
        Linear charge densities of species 1 and 2, both > 0.
    alpha : float
        Dimensionless electrostatic prefactor of the model.

    Returns
    -------
    out : np.ndarray
        Array of five floats, in order: the charge-density ratio of species 1 to
        species 2; the single charge-density parameter of the reduced model; the
        dimensionless interaction parameter; the effective chain length; and the
        length-asymmetry coefficient equal to one minus the reciprocal effective
        chain length.

    Raises
    ------
    ValueError
        If any argument is not a finite, strictly positive real scalar.
    """
    return out  # placeholder
```

### Step 2

free_energy_density

Goal
----
One-component free-energy density.

```python
def free_energy_density(phi: float, r_eff: float, chi: float) -> float:
    """Free-energy density of the reduced one-component model.

    Parameters
    ----------
    phi : float
        Total polymer volume fraction, strictly inside (0, 1).
    r_eff : float
        Effective chain length, > 0.
    chi : float
        Dimensionless interaction parameter, >= 0.

    Returns
    -------
    f : float
        Free-energy density in units of thermal energy per reference volume, with
        phi-constant and phi-linear terms omitted.

    Raises
    ------
    ValueError
        If phi is outside (0, 1), r_eff is not positive, or chi is negative.
    """
    return f  # placeholder
```

### Step 3

coexistence_potentials

Goal
----
Exchange chemical potential and osmotic pressure.

```python
def coexistence_potentials(phi: float, r_eff: float, chi: float) -> np.ndarray:
    """Exchange chemical potential and osmotic pressure of the reduced model.

    Parameters
    ----------
    phi : float
        Total polymer volume fraction, strictly inside (0, 1).
    r_eff : float
        Effective chain length, > 0.
    chi : float
        Dimensionless interaction parameter, >= 0.

    Returns
    -------
    out : np.ndarray
        Two floats: the exchange chemical potential in units of thermal energy, and
        the osmotic pressure in units of thermal energy per reference volume. The
        chemical potential is shifted by +gamma = 1 - 1/r_eff relative to the
        derivative of the free-energy density returned by free_energy_density (its
        own phi-constant term is dropped too); the pressure carries no such shift.

    Raises
    ------
    ValueError
        If phi is outside (0, 1), r_eff is not positive, or chi is negative.
    """
    return out  # placeholder
```

### Step 4

critical_point

Goal
----
Critical fraction and critical interaction parameter.

```python
def critical_point(r_eff: float) -> np.ndarray:
    """Critical volume fraction and critical interaction parameter.

    Parameters
    ----------
    r_eff : float
        Effective chain length, > 0.

    Returns
    -------
    out : np.ndarray
        Two floats: the critical total polymer volume fraction, and the value of the
        dimensionless interaction parameter at which phase separation begins.

    Raises
    ------
    ValueError
        If r_eff is not a finite, strictly positive real scalar.
    """
    return out  # placeholder
```

### Step 5

landau_seed

Goal
----
Near-critical expansion and its estimate of the coexisting fractions.

```python
def landau_seed(phi_c: float, chi: float, r_eff: float) -> np.ndarray:
    """Near-critical expansion coefficients and the seed pair of fractions.

    Parameters
    ----------
    phi_c : float
        Critical total polymer volume fraction, strictly inside (0, 1).
    chi : float
        Dimensionless interaction parameter, >= 0.
    r_eff : float
        Effective chain length, > 0.

    Returns
    -------
    out : np.ndarray
        Six floats, in order: the linear, quadratic and cubic coefficients of the
        expansion of the exchange chemical potential about phi_c; the width of the
        coexistence gap the expansion implies; and the estimated dense and dilute
        volume fractions, dense first. The dilute estimate may be negative.

    Raises
    ------
    ValueError
        If the arguments are out of range, or if the expansion admits no real
        coexistence gap at this interaction parameter.
    """
    return out  # placeholder
```

### Step 6

fixed_point_operators

Goal
----
Scalar combinations that carry the coexistence conditions.

```python
def fixed_point_operators(phi_I: float, phi_II: float, chi: float, gamma: float,
                         r_eff: float) -> np.ndarray:
    """Scalar combinations and derived arguments of the coexistence fixed-point map.

    Parameters
    ----------
    phi_I : float
        Trial dense-phase volume fraction, strictly inside (0, 1).
    phi_II : float
        Trial dilute-phase volume fraction, strictly inside (0, 1) and below phi_I.
    chi : float
        Dimensionless interaction parameter, >= 0.
    gamma : float
        Length-asymmetry coefficient of the reduced model.
    r_eff : float
        Effective chain length, > 0.

    Returns
    -------
    out : np.ndarray
        Five floats, in order: the scalar combination carrying the equality of
        chemical potentials; the scalar combination carrying the equality of osmotic
        pressures; the first derived argument; the second derived argument; and the
        ratio-of-hyperbolic-sines factor through which the map is expressed.

    Raises
    ------
    ValueError
        If either fraction lies outside (0, 1), if phi_I does not exceed phi_II, or
        if the derived arguments are degenerate.
    """
    return out  # placeholder
```

### Step 7

seed_operators

Goal
----
Derived arguments evaluated on the near-critical seed.

```python
def seed_operators(phi_c: float, chi: float, gamma: float, r_eff: float,
                  D: float) -> np.ndarray:
    """Derived arguments of the fixed-point map for the near-critical seed.

    Parameters
    ----------
    phi_c : float
        Critical total polymer volume fraction, strictly inside (0, 1).
    chi : float
        Dimensionless interaction parameter, >= 0.
    gamma : float
        Length-asymmetry coefficient of the reduced model.
    r_eff : float
        Effective chain length, > 0.
    D : float
        Width of the coexistence gap implied by the near-critical expansion, > 0.

    Returns
    -------
    out : np.ndarray
        Three floats: the first derived argument, the second derived argument, and the
        ratio-of-hyperbolic-sines factor built from them.

    Raises
    ------
    ValueError
        If the arguments are out of range, or if the derived arguments are degenerate.
    """
    return out  # placeholder
```

### Step 8

advance_fixed_point

Goal
----
One iteration of the fixed-point map.

```python
def advance_fixed_point(a: float, b: float, chi: float, gamma: float,
                       r_eff: float) -> np.ndarray:
    """Advance the coexistence fixed-point map by one iteration.

    Parameters
    ----------
    a : float
        Current first derived argument.
    b : float
        Current second derived argument.
    chi : float
        Dimensionless interaction parameter, >= 0.
    gamma : float
        Length-asymmetry coefficient of the reduced model.
    r_eff : float
        Effective chain length, > 0.

    Returns
    -------
    out : np.ndarray
        Three floats: the updated first derived argument, the updated second derived
        argument, and the ratio-of-hyperbolic-sines factor built from the updated pair.

    Raises
    ------
    ValueError
       If any argument is not a finite real scalar, if r_eff is not positive or chi is negative, if the incoming arguments are degenerate, if the factor they imply is not finite and strictly positive, or if the update is not finite.
    """
    return out  # placeholder
```

### Step 9

interphase_potential

Goal
----
Electric potential difference between the coexisting phases.

```python
def interphase_potential(phi_I: float, phi_II: float, r1: float, r2: float,
                        sigma1: float, sigma2: float, alpha: float = 3.655) -> float:
    """Electrostatic potential difference between coexisting coacervate phases.

    Parameters
    ----------
    phi_I : float
        Dense-phase total polymer volume fraction, strictly inside (0, 1).
    phi_II : float
        Dilute-phase total polymer volume fraction, strictly inside (0, 1) and below phi_I.
    r1, r2 : float
        Chain lengths of species 1 and 2, both > 0.
    sigma1, sigma2 : float
        Linear charge densities of species 1 and 2, both > 0.
    alpha : float
        Dimensionless electrostatic prefactor of the model.

    Returns
    -------
    dpsi : float
        Potential difference, dense phase minus dilute phase, in units of thermal
        energy per elementary charge. Zero for a symmetric pair.

    Raises
    ------
    ValueError
        If either fraction lies outside (0, 1), if phi_I does not exceed phi_II, or if
        any other argument is not a finite, strictly positive real scalar.
    """
    return dpsi  # placeholder
```

### Step 10

coacervate_potential_drop

Goal
----
Orchestrator: potential drop on the self-consistent binodal.

```python
def coacervate_potential_drop(r1: float, r2: float, sigma1: float, sigma2: float,
                             alpha: float = 3.655, order: int = 2) -> float:
    """End-to-end interphase potential difference for a coacervating pair.

    Parameters
    ----------
    r1, r2 : float
        Chain lengths of species 1 and 2 in units of the reference volume, both > 0.
    sigma1, sigma2 : float
        Linear charge densities of species 1 and 2, both > 0.
    alpha : float
        Dimensionless electrostatic prefactor of the model.
    order : int
        Order of the self-consistent solution, that is the number of iterations of the
        coexistence fixed-point map counted from the near-critical seed. Must be >= 1.

    Returns
    -------
    dpsi : float
        Potential difference between the coexisting phases, dense minus dilute, in
        units of thermal energy per elementary charge.

    Raises
    ------
    ValueError
        If any argument is out of range, if order is not an integer >= 1, if the
        mixture does not phase separate, if the iteration leaves the physical range,
        or if the earlier steps it chains disagree with one another.
    """
    return dpsi  # placeholder
```

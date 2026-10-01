# Chemistry-Quantum_Chemistry-13

## Background

## Where this problem comes from

London dispersion is the part of the intermolecular interaction that survives when neither
molecule carries a permanent moment. Two neutral, non-polar bodies still attract, because the
charge density on each one fluctuates and the fluctuations on the two bodies correlate through the
Coulomb interaction. The standard theory of this effect, which underpins everything from
Casimir forces to the dispersion corrections bolted onto density functionals, rests on one
structural fact: in thermal equilibrium the spectrum of charge fluctuations on a body and its
dissipative response to an external field are not independent quantities. The fluctuation
dissipation theorem ties them together, and once they are tied together the sign of the
interaction energy follows without any reference to what the bodies are made of. Dispersion is
universally attractive, and that universality is a theorem about equilibrium rather than an
empirical observation.

Molecular electronics puts that theorem under strain. A single molecule contacted by two metal
electrodes and held at a bias voltage is an open quantum system in a nonequilibrium steady state:
electrons enter from one electrode and leave into the other, and the occupation of the molecular
orbital is maintained by that flow rather than by contact with a single thermal bath. The same
geometry appears in stacked two-dimensional materials, where individual layers can be contacted
and driven independently while remaining bound to one another by dispersion forces alone. Coulomb
drag experiments already probe the capacitive coupling between two such layers. What has been
missing is a theory of the dispersion binding itself when both partners carry current.

## The physical content

The natural language for a driven open system is the nonequilibrium Green's function formalism,
which tracks separately how much spectral weight at a given energy is occupied and how much is
empty, rather than assuming a single thermal distribution relates them. Applied to two bodies that
exchange no electrons but do interact electrostatically, it produces an interaction energy at
second order in the intermolecular Coulomb coupling. What appears in that expression is not the
static polarisability of the standard London picture but the full frequency-resolved
charge-fluctuation propagator of each body, evaluated in its own driven steady state.

Read physically, the result splits the fluctuation propagator into a charge noise spectrum, which
counts total fluctuation activity, and a dissipative charge response, which counts the imbalance
between absorbing and emitting a fluctuation of a given frequency. The interaction energy couples
the noise on one body to the response of the other, and vice versa. This is the fluctuation
dissipation structure of the equilibrium theory, but with the two ingredients now free to move
independently. In equilibrium they are locked and the interaction is attractive whatever the
system. Under bias the lock is broken, and both the magnitude and the sign of the interaction
become properties of the driven state rather than of the molecules alone.

Two consequences make this more than a formal exercise. Driving a molecular junction through the
voltage at which its orbital enters the conduction window floods the low-frequency fluctuation
spectrum with current-driven noise, and the attraction is enhanced by more than an order of
magnitude over its equilibrium value. Pushing the electrode distribution further, to the point of
population inversion where emission of a charge fluctuation outweighs absorption, reverses the
sign of the energy: the dispersion interaction becomes repulsive, which no equilibrium argument
permits. A dimensionless ratio of the absorbing to the emitting spectral weight parametrises how
far the steady state has departed from detailed balance, takes its equilibrium Boltzmann value
where the drive is ineffective, and crosses through unity exactly where the interaction changes
sign.

## What the calculation involves

Reduced to its computational core, the problem is one of open-system electronic structure on a
frequency grid. Each molecule is described by a small set of frequency-dependent functions built
from its orbital energy, its coupling to the electrodes and the distributions those electrodes
impose. The intermolecular coupling enters twice: once as a static mean-field shift, which makes
the occupation of each level depend on the charge of its partner and therefore turns the
mean-field problem into a nonlinear equation, and once as a dynamical dressing in which an
electron on one molecule borrows and returns a charge fluctuation from the other. Treating the
energy and the electronic structure consistently means iterating that dressing to convergence
rather than evaluating it once on mean-field inputs, and the difference between those two
treatments is not small. The remaining ingredients are standard tools of many-body theory used
carefully: causality relations that reconstruct a retarded function from its occupied and empty
parts, and frequency convolutions that must be arranged so that no interpolation is introduced.

The inputs are an orbital energy, two electrode hybridisations, a bias voltage and its division
across the junction, an electronic temperature, and an intermolecular coupling constant. The
output is a single interaction energy, whose magnitude and sign encode how far the driven steady
state has been pushed away from thermal equilibrium.

## Problem

Dispersion forces between two molecules are usually thought of as a settled subject: correlated quantum fluctuations of the charge density on each body lower the total energy, and the fluctuation-dissipation theorem guarantees that the result is attractive no matter what the two bodies are. That guarantee is a statement about thermal equilibrium. It has nothing to say about two molecules that are each wired into their own pair of electrodes and carrying current, which is the situation in a molecular junction or in a stack of independently contacted two-dimensional layers. Once a bias voltage drives each molecule into a nonequilibrium steady state, the charge noise and the dissipative charge response stop being locked to one another, and the dispersion interaction is no longer bound by the equilibrium argument. Your task is to compute that interaction for one completely specified pair of current-carrying molecules and report a single energy.

The two molecules are identical. Each is modelled as a single spinless electronic level of energy eps = -0.75 eV measured from a common Fermi energy E_F = 0, coupled to its own left and right electrode. There is no electron tunnelling between the two molecules. They interact only through the density-density Coulomb coupling U (n_a - q_a)(n_b - q_b), with U = 0.90 eV and with positive ionic core charges q_a = q_b = +1 representing transport dominated by the highest occupied molecular orbital. Here n_a and n_b are the occupation number operators of the two levels.

Each electrode is treated in the wide-band limit, so its contribution to the retarded self-energy of the level is a real energy-independent hybridisation. The two electrodes attached to one molecule are not equivalent: the left electrode contributes a hybridisation of 0.050 eV and the right electrode 0.030 eV, and the sum of the two is the total width that enters the retarded Green's function through the usual factor of one half. The bias is applied to both molecules alike but is divided unequally across each junction: of the applied bias V = 2.0 V, a fraction 0.70 is dropped on the left side and 0.30 on the right, so the left electrode sits at a chemical potential of +1.40 eV and the right electrode at -0.60 eV. All four electrodes are held at an electronic temperature of 300 K and are described by ordinary Fermi-Dirac distributions. Use a Boltzmann constant of 8.617333262e-5 eV/K.

The interaction energy is to be evaluated to second order in the intermolecular Coulomb coupling U, and the electronic structure of each molecule must be determined self-consistently at that same order rather than being frozen at its mean-field form. The mean-field stage itself is not innocent: because each level feels a static shift proportional to the charge on its partner, the mean-field occupation satisfies a nonlinear equation whose physically relevant solution is the smallest occupation in [0, 1] that satisfies it, located by bracketing that interval rather than assumed. The static mean-field shift that this produces is then held fixed while the correlation dressing is iterated; only the correlation self-energy is updated inside the self-consistency.

Carry out the whole calculation on a uniform frequency grid defined by w_j = (j - n/2) * D for j = 0, 1, ..., n-1, with n = 4096 points and spacing D = 2 * w_max / n for w_max = 10.0 eV. This grid is not an arbitrary choice: it places the difference of any two grid frequencies exactly on another grid point whenever that difference lies inside the window, so every frequency convolution is an exact index shift and no interpolation is needed; differences outside the window contribute nothing.Treat every frequency integral as a sum over this grid multiplied by the spacing, and iterate the self-consistency until the largest change in any Green's function component between successive iterations falls below 1e-11 eV^-1.

In your reasoning, report the total hybridisation width of one junction in eV; the mean-field occupation of a level and the static level shift in eV that it produces; the occupation of a level after the self-consistent solution is reached; the real and imaginary parts, in eV, of the retarded correlation self-energy of one molecule evaluated at zero frequency; the value at zero frequency of the two spectral weights that describe how strongly a molecule absorbs and how strongly it emits a charge fluctuation, in eV^-1, whose ratio reduces to exp(w/kT) in equilibrium; and the ratio of the absorbing to the emitting weight at a frequency of +0.20 eV. Your final answer must be a single number: the dispersion interaction energy of this pair of current-carrying molecules, in eV.

## Output format

Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Show the scalars the task asks for and one line each on the relation behind them.
Do not paste the frequency grid, the Green's functions or the self-energies.

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

01_lead_selfenergies

Goal
----
Electrode self-energies of a biased junction in the wide-band limit

```python
import numpy as np


def lead_selfenergies(w: np.ndarray, bias_v: float, temp_k: float, gamma_left: float,
                      gamma_right: float, frac_left: float, e_fermi: float) -> np.ndarray:
    '''Lesser and greater electrode self-energies of one junction on a frequency grid.

    Parameters
    ----------
    w : np.ndarray
        Real frequency grid in eV, strictly increasing.
    bias_v : float
        Applied bias across the junction in volts; the electron charge is one, so a bias of
        V volts moves a chemical potential by V eV. May be zero or negative.
    temp_k : float
        Electronic temperature of both electrodes in kelvin. May be negative, which describes a
        population-inverted reservoir. Must not be zero.
    gamma_left, gamma_right : float
        Wide-band hybridisation of the left and right electrode in eV. Both must be positive.
    frac_left : float
        Fraction of the applied bias dropped on the left side, in [0, 1]. The left chemical
        potential is e_fermi + frac_left * bias_v and the right one is
        e_fermi - (1 - frac_left) * bias_v.
    e_fermi : float
        Common equilibrium Fermi energy in eV.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (2, len(w)); row 0 is the lesser self-energy and row 1 the
        greater self-energy, both in eV.

    Raises
    ------
    ValueError
        If gamma_left or gamma_right is not positive, if temp_k is zero, or if frac_left lies
        outside [0, 1].
    '''
    return result  # placeholder

# ============================================================================
```

### Step 2

02_mean_field_occupation

Goal
----
Mean-field occupation of a level that feels the charge on its partner

```python
import numpy as np


def mean_field_occupation(w: np.ndarray, dw: float, eps: float, gamma_total: float,
                          sigma_lesser: np.ndarray, u_coupling: float, q_core: float) -> np.ndarray:
    '''Self-consistent mean-field occupation of a molecular level and the shift it produces.

    Parameters
    ----------
    w : np.ndarray
        Real frequency grid in eV, uniformly spaced.
    dw : float
        Grid spacing in eV; positive.
    eps : float
        Bare molecular level energy in eV.
    gamma_total : float
        Total hybridisation width of the junction in eV; positive.
    sigma_lesser : np.ndarray
        Lesser electrode self-energy on the same grid, in eV.
    u_coupling : float
        Intermolecular density-density coupling U in eV; non-negative.
    q_core : float
        Positive ionic core charge q of the partner molecule.

    Returns
    -------
    result : np.ndarray
        Real array of length 2: the mean-field occupation, then the static level shift in eV
        that this occupation produces on the partner.

    Raises
    ------
    ValueError
        If dw or gamma_total is not positive, if u_coupling is negative, or if the mean-field
        equation admits no root in [0, 1].
    '''
    return result  # placeholder

# ============================================================================
```

### Step 3

03_keldysh_greens

Goal
----
Retarded, lesser and greater Green's functions of a driven level

```python
import numpy as np


def keldysh_greens(w: np.ndarray, eps_shifted: float, gamma_total: float, sigma_lesser: np.ndarray,
                   sigma_greater: np.ndarray, sig_c_ret: np.ndarray, sig_c_lesser: np.ndarray,
                   sig_c_greater: np.ndarray) -> np.ndarray:
    '''Retarded, lesser and greater Green's functions of one molecular level.

    Parameters
    ----------
    w : np.ndarray
        Real frequency grid in eV.
    eps_shifted : float
        Molecular level energy in eV, already including the static mean-field shift.
    gamma_total : float
        Total hybridisation width of the junction in eV; positive.
    sigma_lesser, sigma_greater : np.ndarray
        Lesser and greater electrode self-energies on the same grid, in eV.
    sig_c_ret, sig_c_lesser, sig_c_greater : np.ndarray
        Retarded, lesser and greater correlation self-energies on the same grid, in eV. Pass
        arrays of zeros to obtain the electrode-only result.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (3, len(w)): row 0 the retarded Green's function, row 1 the
        lesser and row 2 the greater, all in eV^-1.

    Raises
    ------
    ValueError
        If gamma_total is not positive, or if any input array length differs from len(w).
    '''
    return result  # placeholder

# ============================================================================
```

### Step 4

04_polarisation_bubbles

Goal
----
Charge-fluctuation propagators of one molecule

```python
import numpy as np


def polarisation_bubbles(g_lesser: np.ndarray, g_greater: np.ndarray, dw: float) -> np.ndarray:
    '''Lesser and greater charge-fluctuation propagators of one molecule.

    Parameters
    ----------
    g_lesser, g_greater : np.ndarray
        Lesser and greater Green's functions of the molecule on the task grid, in eV^-1.
        The grid is w_j = (j - n/2) * dw with n = len(g_lesser).
    dw : float
        Grid spacing in eV; positive.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (2, n); row 0 is the lesser propagator and row 1 the greater
        propagator, both in eV^-1. Frequencies outside the grid contribute nothing.

    Raises
    ------
    ValueError
        If dw is not positive, or if the two input arrays have different lengths.
    '''
    return result  # placeholder

# ============================================================================
```

### Step 5

05_correlation_selfenergies

Goal
----
Correlation self-energy generated by the partner's charge fluctuations

```python
import numpy as np


def correlation_selfenergies(g_lesser: np.ndarray, g_greater: np.ndarray, pi_lesser: np.ndarray,
                             pi_greater: np.ndarray, u_coupling: float, dw: float) -> np.ndarray:
    '''Lesser and greater correlation self-energies of one molecule.

    Parameters
    ----------
    g_lesser, g_greater : np.ndarray
        Lesser and greater Green's functions of the molecule being dressed, in eV^-1.
    pi_lesser, pi_greater : np.ndarray
        Lesser and greater charge-fluctuation propagators of the PARTNER molecule, in eV^-1.
    u_coupling : float
        Intermolecular density-density coupling U in eV; non-negative.
    dw : float
        Grid spacing in eV; positive.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (2, n): row 0 the lesser correlation self-energy and row 1 the
        greater, both in eV.

    Raises
    ------
    ValueError
        If dw is not positive, if u_coupling is negative, or if the four input arrays do not all
        have the same length.
    '''
    return result  # placeholder

# ============================================================================
```

### Step 6

06_retarded_from_keldysh

Goal
----
Retarded component of a self-energy from its lesser and greater parts

```python
import numpy as np


def retarded_from_keldysh(sigma_lesser: np.ndarray, sigma_greater: np.ndarray, w: np.ndarray,
                          dw: float) -> np.ndarray:
    '''Retarded self-energy reconstructed from its lesser and greater components.

    Parameters
    ----------
    sigma_lesser, sigma_greater : np.ndarray
        Lesser and greater components of the self-energy on the task grid, in eV.
    w : np.ndarray
        Real frequency grid in eV, of the form w_j = (j - n/2) * dw.
    dw : float
        Grid spacing in eV; positive.

    Returns
    -------
    result : np.ndarray
        Complex array of length n holding the retarded self-energy in eV.

    Raises
    ------
    ValueError
        If dw is not positive, or if the input arrays do not all have the same length.
    '''
    return result  # placeholder

# ============================================================================
```

### Step 7

07_self_consistent_greens

Goal
----
Closing the loop between the Green's functions and the correlation self-energy

```python
import numpy as np


def self_consistent_greens(w: np.ndarray, dw: float, eps_shifted: float, gamma_total: float,
                           sigma_lesser: np.ndarray, sigma_greater: np.ndarray,
                           u_coupling: float, tol: float, max_iter: int) -> np.ndarray:
    '''Green's functions of one molecule with the correlation dressing iterated to convergence.

    Parameters
    ----------
    w : np.ndarray
        Real frequency grid in eV of the form w_j = (j - n/2) * dw.
    dw : float
        Grid spacing in eV; positive.
    eps_shifted : float
        Molecular level energy in eV including the static mean-field shift, held fixed.
    gamma_total : float
        Total hybridisation width in eV; positive.
    sigma_lesser, sigma_greater : np.ndarray
        Lesser and greater electrode self-energies on the same grid, in eV.
    u_coupling : float
        Intermolecular density-density coupling U in eV; non-negative.
    tol : float
        Convergence threshold on the largest absolute change in any Green's function component
        between successive passes, in eV^-1; positive.
    max_iter : int
        Maximum number of passes; positive.

    Returns
    -------
    result : np.ndarray
        Complex array of shape (2, n): row 0 the converged lesser Green's function and row 1 the
        converged greater Green's function, both in eV^-1.

    Raises
    ------
    ValueError
        If dw, gamma_total, tol or max_iter is not positive, or if u_coupling is negative.
    '''
    return result  # placeholder

# ============================================================================
```

### Step 8

08_noise_response_kms

Goal
----
Charge noise, dissipative response, and how far the drive pushes the molecule off balance

```python
import numpy as np


def noise_response_kms(pi_lesser: np.ndarray, pi_greater: np.ndarray) -> np.ndarray:
    '''Absorbing and emitting spectral weights, noise, response, and their ratio.

    Parameters
    ----------
    pi_lesser, pi_greater : np.ndarray
        Lesser and greater charge-fluctuation propagators of one molecule, in eV^-1.

    Returns
    -------
    result : np.ndarray
        Real array of shape (5, n): row 0 the absorbing spectral weight, the greater propagator
        component multiplied by the imaginary unit; row 1 the emitting spectral weight, the lesser
        component multiplied by the same prefactor; row 2 their half-sum, the charge noise
        spectrum; row 3 their half-difference, the dissipative charge response, all in eV^-1; and
        row 4 the dimensionless ratio of the absorbing to the emitting weight. Both weights are
        non-negative by construction. The ratio is reported as 0.0 wherever the emitting weight
        falls below one ten-thousandth of its own peak, where its value would be set by round-off
        rather than by physics.

    Raises
    ------
    ValueError
        If the two input arrays have different lengths.
    '''
    return result  # placeholder

# ============================================================================
```

### Step 9

09_dispersion_energy

Goal
----
The dispersion interaction energy of the driven pair

```python
import numpy as np


def dispersion_energy(pi_lesser_a: np.ndarray, pi_greater_a: np.ndarray, pi_lesser_b: np.ndarray,
                      pi_greater_b: np.ndarray, w: np.ndarray, dw: float,
                      u_coupling: float) -> float:
    '''Dispersion interaction energy of two Coulomb-coupled molecules in a steady state.

    Parameters
    ----------
    pi_lesser_a, pi_greater_a : np.ndarray
        Lesser and greater charge-fluctuation propagators of molecule A, in eV^-1.
    pi_lesser_b, pi_greater_b : np.ndarray
        Lesser and greater charge-fluctuation propagators of molecule B, in eV^-1.
    w : np.ndarray
        Real frequency grid in eV of the form w_j = (j - n/2) * dw.
    dw : float
        Grid spacing in eV; positive.
    u_coupling : float
        Intermolecular density-density coupling U in eV; non-negative.

    Returns
    -------
    result : float
        The dispersion interaction energy in eV. Negative values are attractive.

    Raises
    ------
    ValueError
        If dw is not positive, if u_coupling is negative, or if the four propagator arrays and w
        do not all have the same length.
    '''
    return result  # placeholder

# ============================================================================
```

### Step 10

10_final_answer

Goal
----
Assembling the interaction energy of the specified pair

```python
import numpy as np


def final_answer(eps: float = -0.75, u_coupling: float = 0.90, bias_v: float = 2.0,
                 temp_k: float = 300.0, gamma_left: float = 0.050, gamma_right: float = 0.030,
                 frac_left: float = 0.70, e_fermi: float = 0.0, q_core: float = 1.0,
                 w_max: float = 10.0, n: int = 4096, tol: float = 1e-11,
                 max_iter: int = 4000) -> float:
    '''Dispersion interaction energy of the pair of current-carrying molecules specified in the
    problem statement.

    Every parameter defaults to the value the problem statement fixes, so calling this with no
    arguments returns the answer the task asks for.

    Parameters
    ----------
    eps : float
        Bare energy of the single spinless level of each molecule, in eV.
    u_coupling : float
        Intermolecular density-density Coulomb coupling in eV. Must be non-negative.
    bias_v : float
        Applied bias across each junction in volts. May be zero or negative.
    temp_k : float
        Electronic temperature of all four electrodes in kelvin. Must not be zero; a negative
        value describes population-inverted reservoirs.
    gamma_left, gamma_right : float
        Wide-band hybridisation of the left and right electrode in eV. Both must be positive.
    frac_left : float
        Fraction of the applied bias dropped on the left side, in [0, 1].
    e_fermi : float
        Common equilibrium Fermi energy in eV.
    q_core : float
        Positive ionic core charge on each molecule.
    w_max : float
        Half-width of the frequency window in eV. Must be positive.
    n : int
        Number of grid points. Must be even and at least 2.
    tol : float
        Convergence threshold on the largest change in any Green's function component, in eV^-1.
    max_iter : int
        Maximum number of self-consistency passes.

    Returns
    -------
    result : float
        The dispersion interaction energy in eV.

    Raises
    ------
    ValueError
        If either hybridisation is not positive, the coupling is negative, the temperature is
        zero, or the grid does not have a positive half-width and an even number of points.
    RuntimeError
        If the converged solution produces a non-finite Green's function or spectral weight, or
        if the converged spectral weights violate the contract documented in the noise and
        response step.
    '''
    return result  # placeholder
```

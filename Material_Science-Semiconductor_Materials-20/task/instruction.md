# Material_Science-Semiconductor_Materials-20

## Background

Self-assembled semiconductor quantum dots are nanometre-sized islands of a lower band-gap material, typically InGaAs or GaAs, embedded in a host crystal. Electrons and holes confined in all three dimensions give them discrete, atom-like levels, and a dot can hold one electron-hole pair (an exciton) or two (a biexciton). Because they emit single photons on demand and can be integrated into photonic structures, quantum dots are leading sources of non-classical light for quantum communication, where entangled photon pairs underpin teleportation, entanglement swapping and quantum repeaters.

The standard route to entangled pairs is the biexciton cascade. A biexciton decays to the ground state through one of two intermediate exciton states, emitting two horizontally or two vertically polarised photons, so the pair is ideally in a maximally entangled polarisation state. In real dots an anisotropic exchange interaction splits the two exciton states by the fine-structure splitting, which tags each decay path with a colour and a timing and degrades the entanglement. Remedies include strain and field tuning of the splitting, spectral or temporal filtering, and optical microcavities that reshape the emission channels, for example by favouring simultaneous two-photon emission from the biexciton over the sequential cascade.

A dot is not an isolated atom: it is part of a vibrating crystal. Longitudinal-acoustic phonons couple to confined carriers through the deformation potential, and in these dots the coupling is strong enough that the carrier and the lattice distortion around it form a polaron. This coupling causes pure dephasing, renormalises the strength of optical and cavity couplings, and drives phonon-assisted transitions between dressed states, all with a characteristic dependence on temperature and on the size of the dot. The lattice has a memory of a few picoseconds, comparable with the dot and cavity time scales, so the dynamics is non-Markovian.

Several theoretical tools describe such open quantum systems. Weak-coupling master equations treat the phonon coupling perturbatively; polaron and variational transformations first absorb the displacement of the lattice and then treat the remaining coupling perturbatively, which extends their validity to strong exciton-phonon interaction; and numerically exact approaches based on real-time path integrals and tensor-network representations of the environment's influence give benchmark results at a much higher computational cost. Comparing approximate treatments against exact ones is an active topic in the modelling of solid-state photon sources.

## Problem

A semiconductor quantum dot prepared in its biexciton state can emit a polarisation-entangled photon pair, and a bimodal microcavity tuned to half the biexciton energy lets the pair leave through direct two-photon emission, which removes most of the which-path information carried by the exciton fine-structure splitting. What then limits the entanglement is the coupling of the dot to longitudinal-acoustic phonons, which in self-assembled GaAs dots is strong and non-Markovian, and a recent study compared a polaron-frame master equation for this system with numerically exact simulations. For the dot, cavity and lattice given below, the task is to find the lattice temperature at which the time-integrated concurrence of the emitted pairs falls to 0.95.

The dot has a ground state G, excitons XH and XV split by the fine-structure splitting delta, and a biexciton XX with binding energy E_B. In the frame rotating at half the biexciton frequency, with both cavity modes H and V at that frequency, G and XX lie at zero energy and XH and XV at E_B/2 + delta/2 and E_B/2 - delta/2, the dot-cavity coupling is g (sigma_H a_H^dag + sigma_V a_V^dag) + h.c. with sigma_k = |G><X_k| + |X_k><XX|, and each mode loses photons at the rate gamma. The lattice couples to the dot through sum_k (g_k b_k^dag + g_k^* b_k) O with O = |XH><XH| + |XV><XV| + 2|XX><XX|, by the electron and hole deformation potentials of bulk GaAs, with a linear acoustic dispersion and isotropic Gaussian electron and hole ground states, psi proportional to exp(-r^2/(2a^2)). Use:

- E_B = 1.2 meV, delta = 0.15 meV, bare dot-cavity coupling g = 0.2 meV, gamma = 0.3 ps^-1 for each mode;
- D_e = 7.0 eV, D_h = -3.5 eV, mass density 5370 kg/m^3, sound velocity c_s = 5110 m/s, a_e = 3.5 nm, a_h = a_e/1.15;
- hbar = 0.6582119569 meV ps and k_B = 0.08617333262 meV/K; the exciton and biexciton energies above are already the phonon-renormalised (polaron-frame) values, so no polaron shift is added to them.

Treat the phonons with the time-local second-order (Born-Markov) master equation in the polaron frame, expanded in the residual dot-phonon coupling with the lattice in thermal equilibrium at temperature T, the memory integrals extended to infinity and the dot-cavity operators evolved under the full phonon-dressed Hamiltonian. Make no secular approximation, keep the energy-shift (imaginary) parts of the response functions and add the cavity losses as Lindblad terms. With the dot in XX, both modes empty and the lattice relaxed around XX at t = 0, define G_ij(t) = Tr[rho(t) a_i^dag a_i^dag a_j a_j] for i, j in {H, V} and the single-time-integrated concurrence C = 2 |int_0^inf G_HV dt| / int_0^inf (G_HH + G_VV) dt, which decreases with T here; the final answer is the temperature T* between 4 K and 40 K at which C = 0.95, in kelvin to at least three decimals.

The scalars that determine the answer, which the reasoning should give to at least four significant figures, are the thermal renormalisation factor <B> of the dot-cavity coupling at 20 K, C without phonons, C at 10 K and at 30 K, int_0^inf (G_HH + G_VV) dt and the complex int_0^inf G_HV dt at 10 K in ps, and dC/dT at T*; the reasoning should also name the smallest set of states whose evolution fixes these correlations and why it suffices, and say what an earlier comparison of concurrence measures for a dot in a cavity tuned to the two-photon resonance found about how the single-time-integrated concurrence relates to the time-dependent concurrence and to the double-time-integrated concurrence.

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

01_lattice_spectral_density

Goal
----
Step 01 - Deformation-potential spectral density of the dot-lattice coupling.

```python
import numpy as np


def lattice_spectral_density(omega: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    '''Deformation-potential spectral density J(omega) of the dot-lattice coupling.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of angular frequencies in rad/ps, every entry >= 0.
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    J : np.ndarray
        Array of the same shape as omega, J(omega) in rad/ps.

    Raises
    ------
    ValueError
        If omega is not a one-dimensional array of finite values >= 0, or if lattice is
        not a numpy array of six finite real numbers with the mass density, c_s, a_e and a_h
        positive.
    '''
    return J
```

### Step 2

02_phonon_propagator

Goal
----
Step 02 - Phonon propagator of the polaron transformation.

```python
import numpy as np


def phonon_propagator(t: np.ndarray, T: float, lattice: np.ndarray) -> np.ndarray:
    '''Phonon propagator phi(t) of the polaron transformation at temperature T.

    Parameters
    ----------
    t : np.ndarray
        One-dimensional array of times in ps, every entry >= 0.
    T : float
        Lattice temperature in K, > 0.
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    phi : np.ndarray
        Shape (2, len(t)): row 0 is Re phi(t), row 1 is Im phi(t) (dimensionless).

    Raises
    ------
    ValueError
        If t is not a one-dimensional array of finite values >= 0, if T is not finite
        and positive, or if lattice is invalid (as in step 01).
    '''
    return phi
```

### Step 3

03_polaron_response_functions

Goal
----
Step 03 - Response functions of the residual polaron-frame coupling.

```python
import numpy as np


def polaron_response_functions(omega: np.ndarray, T: float, lattice: np.ndarray) -> np.ndarray:
    '''One-sided Fourier transforms kappa_x(omega) and kappa_y(omega) of the polaron bath.

    Parameters
    ----------
    omega : np.ndarray
        One-dimensional array of real angular frequencies in rad/ps (any sign).
    T : float
        Lattice temperature in K, > 0.
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    kappa : np.ndarray
        Shape (4, len(omega)): rows are Re kappa_x, Im kappa_x, Re kappa_y, Im kappa_y,
        in ps.

    Raises
    ------
    ValueError
        If omega is not a one-dimensional array of finite values, if T is not finite
        and positive, or if lattice is invalid (as in step 01).
    '''
    return kappa
```

### Step 4

04_polaron_master_equation_rhs

Goal
----
Step 04 - Polaron master equation of the dot-cavity system.

```python
import numpy as np


def polaron_master_equation_rhs(rho: np.ndarray, T: float, dot: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    '''Time derivative of the dot-cavity density matrix under the polaron master equation.

    Parameters
    ----------
    rho : np.ndarray
        Complex (or real) array of shape (13, 13), or a stack of shape (m, 13, 13), in the
        basis order given in the step background.
    T : float
        Lattice temperature in K, > 0.
    dot : np.ndarray
        Shape (4,): [E_B (meV), delta (meV), g (meV) >= 0, gamma (1/ps) >= 0].
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    drho_dt : np.ndarray
        For a (13, 13) input, shape (2, 13, 13): real and imaginary parts of d rho / dt
        in 1/ps. For an (m, 13, 13) input, shape (m, 2, 13, 13).

    Raises
    ------
    ValueError
        If rho is not a numeric array of shape (13, 13) or (m, 13, 13) with finite
        entries, if T is not finite and positive, if dot is not a numpy array of four
        finite real numbers with g >= 0 and gamma >= 0, or if lattice is invalid (as in
        step 01).
    '''
    return drho_dt
```

### Step 5

05_integrated_pair_correlations

Goal
----
Step 05 - Time-integrated two-photon correlations after biexciton preparation.

```python
import numpy as np


def integrated_pair_correlations(T: float, dot: np.ndarray, lattice: np.ndarray) -> np.ndarray:
    '''Integrals over all times of G_HH, G_VV and G_HV after biexciton preparation.

    Parameters
    ----------
    T : float
        Lattice temperature in K, > 0.
    dot : np.ndarray
        Shape (4,): [E_B (meV), delta (meV), g (meV) > 0, gamma (1/ps) > 0].
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    integrals : np.ndarray
        Shape (4,): [int G_HH dt, int G_VV dt, Re int G_HV dt, Im int G_HV dt], in ps.

    Raises
    ------
    ValueError
        If T is not finite and positive, if dot is not a numpy array of four finite real
        numbers with g > 0 and gamma > 0, or if lattice is invalid (as in step 01).
    '''
    return integrals
```

### Step 6

06_integrated_concurrence

Goal
----
Step 06 - Single-time-integrated concurrence of the photon pairs.

```python
import numpy as np


def integrated_concurrence(T: float, dot: np.ndarray, lattice: np.ndarray) -> float:
    '''Single-time-integrated concurrence of the photon pairs at lattice temperature T.

    Parameters
    ----------
    T : float
        Lattice temperature in K, > 0.
    dot : np.ndarray
        Shape (4,): [E_B (meV), delta (meV), g (meV) > 0, gamma (1/ps) > 0].
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    c_bar : float
        The single-time-integrated concurrence, dimensionless, in [0, 1].

    Raises
    ------
    ValueError
        If T is not finite and positive, if dot is not a numpy array of four finite real
        numbers with g > 0 and gamma > 0, or if lattice is invalid (as in step 01).
    '''
    return c_bar
```

### Step 7

07_entanglement_threshold_temperature

Goal
----
Step 07 - Threshold temperature for a target concurrence (orchestrator).

```python
import numpy as np
from scipy.optimize import brentq


def entanglement_threshold_temperature(C_target: float, T_lo: float, T_hi: float, dot: np.ndarray,
                                       lattice: np.ndarray) -> float:
    '''Temperature T* at which the single-time-integrated concurrence equals C_target.

    Parameters
    ----------
    C_target : float
        Target concurrence, 0 < C_target < 1.
    T_lo : float
        Lower end of the temperature bracket in K, 0 < T_lo < T_hi.
    T_hi : float
        Upper end of the temperature bracket in K.
    dot : np.ndarray
        Shape (4,): [E_B (meV), delta (meV), g (meV) > 0, gamma (1/ps) > 0].
    lattice : np.ndarray
        Shape (6,): [D_e (eV), D_h (eV), mass density (kg/m^3), c_s (m/s), a_e (nm), a_h (nm)].

    Returns
    -------
    T_star : float
        The temperature in K, T_lo < T_star < T_hi, at which C_bar(T_star) = C_target.

    Raises
    ------
    ValueError
        If C_target is not a finite number strictly between 0 and 1, if the bracket does
        not satisfy 0 < T_lo < T_hi, if C_bar(T_lo) is not above C_target or C_bar(T_hi)
        is not below it, or if dot or lattice is invalid (as in step 05).
    '''
    return T_star
```

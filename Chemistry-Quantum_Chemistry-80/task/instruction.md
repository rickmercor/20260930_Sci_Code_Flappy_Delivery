# Chemistry-Quantum_Chemistry-80

## Background

## Radiationless decay of excited molecules

After a molecule absorbs light it can return to the ground electronic state without emitting a photon. The spin-allowed version of this process, internal conversion, competes directly with fluorescence from the lowest excited singlet state, so its rate constant decides how bright a dye, an emitter for organic electronics or a fluorescent sensor can be. Internal conversion becomes faster as the energy gap between the two electronic states shrinks, which is one reason why near-infrared emitters are usually weak.

## Where the electronic energy goes

The energy released in internal conversion has to be taken up by the vibrations of the ground electronic state. For typical gaps of ten to twenty-five thousand wavenumbers and molecular vibrations of at most about three thousand, the final vibrational states are highly excited, with several quanta in the stiffest modes. In organic molecules those stiffest modes are the stretches of bonds to hydrogen, and replacing hydrogen by deuterium slows internal conversion markedly, which points to X-H stretches as the main energy acceptors. Large-amplitude motion of this kind is poorly described by harmonic oscillators, so bond anharmonicity, usually modelled with Morse potentials, strongly affects calculated rates, often by orders of magnitude when the gap is large.

## Golden-rule rates and vibrational relaxation

Rate expressions of the Fermi golden-rule type sum the squared nonadiabatic coupling over the final vibronic states that are nearly degenerate with the initial one. Each final state has a finite lifetime, because its vibrational energy flows on into the dense manifold of lower-frequency modes of the molecule and its surroundings, and this broadening makes the transition irreversible. In practical calculations the broadening is often represented by a single Lorentzian width of the order of a femtosecond inverse lifetime. How the energy spreads out in reality is a question of intramolecular vibrational energy redistribution, which is governed by the cubic and quartic couplings among the vibrations and can range from localised, mode-specific behaviour to fast statistical flow.

## Problem

Internal conversion of large organic chromophores is controlled mainly by their high-frequency X-H stretches (X = C, N, O), which take up most of the electronic energy as overtone excitation, and a recent treatment gives every X-H bond its own local Morse oscillator, couples the bonds to one another and to the skeletal vibrations through the cubic and quartic force field, and replaces the single phenomenological line width by a spreading width computed for each final vibrational state. Your task is to compute the S1 -> S0 internal-conversion rate constant of a small model chromophore with this state-correlated local-Morse treatment.

All vibrational coordinates are dimensionless: a harmonic mode of frequency omega has the Hamiltonian (omega / 2)(-d^2/dq^2 + q^2), and bond a has omega_a [-(1/2) d^2/dxi_a^2 + (1 / (4 x_a))(1 - exp(-sqrt(2 x_a) xi_a))^2] in its local coordinate xi_a, with an S1 potential of the same shape whose minimum is moved to xi_a = delta_a. The three bonds are C-H, N-H and O-H with omega_a = 3050, 3420 and 3660 cm^-1, x_a = 0.0205, 0.0215 and 0.0225, and delta_a = +0.12, -0.08 and +0.05; they are related to the three X-H normal coordinates q_1, q_2, q_3 by xi_a = sum_i C_ai q_i with C = [[0.93, 0.25, 0.12], [-0.21, 0.95, 0.20], [0.10, -0.24, 0.97]] (rows C-H, N-H, O-H), and the nonadiabatic coupling coefficients of those three normal modes are P = (20, -12, 8) cm^-1, from which the local coupling coefficients d_a of the bonds follow by requiring the bonds to reproduce that vector, P_i = sum_a C_ai d_a. Four harmonic skeletal modes q_4 to q_7 have frequencies 1605, 1460, 1180 and 760 cm^-1, S1 minima displaced to q = +0.62, +0.35, +0.48 and +0.85, and coupling coefficients 45, 20, -15 and 10 cm^-1, and every coupling coefficient multiplies the derivative integral <chi_0(S1)| d/dq |chi_n(S0)> taken with respect to the dimensionless coordinate, so that all amplitudes are in cm^-1. The S0 anharmonic force field in the dimensionless normal coordinates is given as Taylor derivatives of the potential, one per canonical index tuple, in cm^-1: cubic F112 = -180, F123 = 95, F223 = -120, F133 = 60, F155 = 195, F255 = -140, F345 = 105, F146 = 68, F267 = 45, F377 = -38, F166 = 60, F457 = 35, F567 = 18, F677 = -22, F477 = 5.5, F577 = 5.5, F456 = 2.5 and F667 = 5.0; quartic F1122 = 40, F1233 = -25, F1155 = 27, F2244 = -18, F1257 = 14, F3366 = 10, F4455 = 36, F5555 = 60, F6677 = 48, F4567 = 14 and F5557 = 66.

Take the 0-0 gap E_if = 12800 cm^-1 at zero temperature, keep Morse levels 0 to 12 for every bond throughout (bath levels are unrestricted), and use the production settings of the method: complete product states with |E_if - E_n| <= 200 cm^-1, edges included; an intensity correction to first order in all pure X-H monomials except the single-bond powers xi_a^3 and xi_a^4, which the Morse potentials already contain, kept linear in the amplitude correction even where the intensity turns negative, with pairs closer than 10 cm^-1 left out and not treated any further; spreading widths from the mixed and pure-bath terms with a Lorentzian regulator of half width at half maximum 20 cm^-1, counting only relaxation steps that lie no more than 200 cm^-1 off resonance and in which at least one coordinate loses quanta, at least one gains, and every gaining coordinate has a strictly lower harmonic frequency than the highest-frequency losing one; Hamiltonian coefficients in the local frame screened at 1 cm^-1 for pure X-H terms, 3 cm^-1 for mixed cubic and quartic terms, and 3 and 10 cm^-1 for pure-bath cubic and quartic terms; and, for fixed-width rates, the reference width 10^14 s^-1 (angular frequency) with every window state taken at the peak of its Lorentzian. A state whose spreading width is zero contributes nothing, and wavenumbers are converted with c = 2.99792458 x 10^10 cm s^-1.

Report the state-correlated rate constant k_IC, in which every product state carries its own first-order intensity and its own spreading width, in s^-1 as the final answer, written as a plain decimal number without an exponent and correct to at least four significant figures. In the reasoning give, to four significant figures, the number of product states in the window, the local coupling coefficients of the C-H, N-H and O-H bonds, the fixed-width rate with zero-order intensities, the same rate from the X-H-active amplitudes alone, the fixed-width rate with first-order intensities, the state-resolved rate with zero-order intensities, the mean spreading width over all window states with zero widths included, the zero-temperature relaxation width that the earlier version of this methodology obtained from the Lax-Pekar model, evaluated here with the four skeletal modes as the accepting modes and Huang-Rhys factors equal to half their squared dimensionless displacements, and the fixed-width zero-order rate obtained with that width in place of 10^14 s^-1.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Give the values requested above and only the few other scalars that determine the final number. Do not paste the input matrices, per-state lists or per-iteration paths.

## Output format

```
## Output format

Scientific reasoning wrapped in <reasoning>...</reasoning> tags. Include enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary. A single final numeric answer wrapped in <final_answer>...</final_answer> tags.
```

## Your task

Implement **all 8 functions** below into a single file at `/app/solution.py`. Requirements:

- Use the EXACT function name and signature shown for each step.
- Define every function at module scope (importable).
- Do NOT include tests, example usage, prints, or a `__main__` block.
- Available libraries: numpy, scipy, sympy, pandas, scikit-learn, networkx.

Start your file with these imports:

```python
import numpy as np
```

### Step 1

01_morse_moment_matrices

Goal
----
Matrix elements of the first four powers of the dimensionless bond coordinate between the lowest bound states of a local X-H Morse oscillator.

```python
import math

import numpy as np
from scipy.special import eval_genlaguerre, gammaln


def morse_moment_matrices(x: float, n_max: int) -> np.ndarray:
    '''Moment matrices <m|xi^p|n> of a Morse oscillator, p = 1..4.

    Parameters
    ----------
    x : float
        Anharmonicity of the bond (dimensionless, positive). The potential in units
        of the harmonic frequency is (1 / (4x)) (1 - exp(-sqrt(2x) xi))^2.
    n_max : int
        Highest vibrational level kept; levels 0..n_max are used.

    Returns
    -------
    moments : numpy.ndarray
        Real array of shape (4, n_max + 1, n_max + 1); moments[p - 1, m, n] is
        <m|xi^p|n> for the bound eigenfunctions m and n, each real, normalised
        and positive for large positive xi.

    Raises
    ------
    ValueError
        If x is not positive, if n_max is not a non-negative integer, or if
        n_max + 1/2 >= 1 / (2x) - 2 (the highest kept level must stay at least two
        levels below the dissociation limit).
    '''
    return moments
```

### Step 2

02_promoting_integrals

Goal
----
Franck-Condon overlaps and first-derivative integrals between the vibrational ground state of a mode in the excited electronic state and its ground-state levels, for a Morse bond or, when the anharmonicity is zero, a harmonic mode.

```python
import math

import numpy as np
from scipy.special import eval_genlaguerre, gammaln


def promoting_integrals(x: float, shift: float, n_max: int) -> np.ndarray:
    '''Overlaps g(0, n) and derivative integrals t(0, n) for one mode.

    Parameters
    ----------
    x : float
        Anharmonicity of the mode; x > 0 means a Morse bond as in
        morse_moment_matrices and x = 0 a harmonic mode.
    shift : float
        Position of the S1 potential minimum on the S0 coordinate (dimensionless);
        the S1 ground state is the S0 ground state translated by shift.
    n_max : int
        Highest S0 level kept; levels 0..n_max are returned.

    Returns
    -------
    integrals : numpy.ndarray
        Real array of shape (2, n_max + 1). Row 0 holds g(0, n) = <chi_0^S1|chi_n^S0>,
        row 1 holds t(0, n) = <chi_0^S1| d/dq chi_n^S0>, the derivative taken with
        respect to the dimensionless coordinate and acting on the S0 level.

    Raises
    ------
    ValueError
        If x is negative, if n_max is not a non-negative integer, or, for x > 0,
        if n_max + 1/2 >= 1 / (2x) - 2.
    '''
    return integrals
```

### Step 3

03_local_force_field

Goal
----
Anharmonic force field rewritten in local X-H bond coordinates: the cubic and quartic Hamiltonian coefficients of every canonical monomial, split into pure X-H, mixed and pure-bath classes, with the diagonal bond terms removed and small coefficients screened out.

```python
import itertools
import math

import numpy as np


def local_force_field(C: np.ndarray, cubic: np.ndarray, quartic: np.ndarray, n_bath: int, thresholds: np.ndarray) -> np.ndarray:
    '''Local-coordinate cubic and quartic Hamiltonian coefficients.

    Parameters
    ----------
    C : numpy.ndarray
        Matrix of shape (N_x, M_x) with xi_a = sum_i C[a, i] q_i, a over the N_x
        local bonds and i over the M_x X-H normal coordinates.
    cubic : numpy.ndarray
        Shape (n3, 4), rows (i, j, k, F) with 0-based normal-coordinate indices
        i <= j <= k (0..M_x-1 the X-H normal modes, M_x..M_x+n_bath-1 the bath
        modes) and F = d^3V / dq_i dq_j dq_k in cm^-1. Each tuple appears once.
    quartic : numpy.ndarray
        Shape (n4, 5), rows (i, j, k, l, F) with i <= j <= k <= l and
        F = d^4V / dq_i dq_j dq_k dq_l in cm^-1. Each tuple appears once.
    n_bath : int
        Number of bath modes.
    thresholds : numpy.ndarray
        Six positive screening thresholds in cm^-1, in the order (pure X-H cubic,
        pure X-H quartic, mixed cubic, mixed quartic, pure bath cubic, pure bath
        quartic). A coefficient G is kept when |G| >= threshold.

    Returns
    -------
    terms : numpy.ndarray
        Shape (L, 7). Row (r, a1, a2, a3, a4, G, cls): order r (3 or 4), local-frame
        coordinate indices a1 <= ... <= ar (0..N_x-1 the bonds, N_x..N_x+n_bath-1
        the bath modes in their input order), a4 = -1 for r = 3, G the coefficient
        in cm^-1 of the monomial xi_a1 ... xi_ar in the potential, and cls = 0 for
        pure X-H, 1 for mixed, 2 for pure bath. All cubic rows come first, then
        the quartic rows, each block in lexicographic order of its indices. An
        empty result has shape (0, 7).

    Raises
    ------
    ValueError
        If C is not two-dimensional, n_bath is negative, thresholds does not hold
        six positive numbers, or an index tuple is not integer, not canonical
        (non-decreasing), out of range or repeated.
    '''
    return terms
```

### Step 4

04_window_states

Goal
----
Enumeration of the complete zero-order product states of the local Morse bonds and the harmonic bath whose vibrational excitation energy lies within a resonance window around the electronic energy gap.

```python
import itertools

import numpy as np


def window_states(omega_loc: np.ndarray, x_loc: np.ndarray, omega_bath: np.ndarray, e_gap: float, half_width: float, n_max: int) -> np.ndarray:
    '''Product states within half_width of the energy gap.

    Parameters
    ----------
    omega_loc : numpy.ndarray
        Harmonic frequencies of the N_x local bonds in cm^-1, positive.
    x_loc : numpy.ndarray
        Their anharmonicities, positive, same length as omega_loc.
    omega_bath : numpy.ndarray
        Frequencies of the N_b harmonic bath modes in cm^-1, positive.
    e_gap : float
        0-0 electronic energy gap E_if in cm^-1.
    half_width : float
        Half width of the resonance window in cm^-1, non-negative.
    n_max : int
        Highest Morse level kept for every bond.

    Returns
    -------
    states : numpy.ndarray
        Shape (K, N_x + N_b + 1). Each row holds the local quantum numbers (order of
        omega_loc), the bath quantum numbers (order of omega_bath) and the
        zero-order excitation energy E_n in cm^-1, for every state with
        |e_gap - E_n| <= half_width, sorted as described. An empty window gives
        shape (0, N_x + N_b + 1).

    Raises
    ------
    ValueError
        If omega_loc and x_loc differ in length, a frequency or anharmonicity is
        not positive, half_width is negative, or n_max is not a non-negative
        integer.
    '''
    return states
```

### Step 5

05_zero_order_amplitudes

Goal
----
Zero-order internal-conversion amplitudes of product final states, split into the X-H-active channel, where the promoting derivative acts on a local bond coordinate, and the bath-active channel, where it acts on a bath mode.

```python
import numpy as np


def zero_order_amplitudes(states: np.ndarray, C: np.ndarray, p_xh: np.ndarray, p_bath: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, delta_bath: np.ndarray, n_max: int) -> np.ndarray:
    '''X-H-active and bath-active zero-order amplitudes of product states.

    Parameters
    ----------
    states : numpy.ndarray
        Product states in the layout returned by window_states: shape (K, N + 1),
        columns 0..N_x-1 the local Morse quantum numbers (bond order of omega_loc),
        columns N_x..N-1 the bath quantum numbers (order of omega_bath), last column
        the zero-order excitation energy in cm^-1 (not used here).
    C : numpy.ndarray
        Local-bond transformation, shape (N_x, M_x), xi_a = sum_i C[a, i] q_i.
    p_xh : numpy.ndarray
        Electronic coupling coefficients P_i of the M_x X-H normal modes, cm^-1.
    p_bath : numpy.ndarray
        Coupling coefficients of the N_b bath modes, cm^-1.
    x_loc : numpy.ndarray
        Anharmonicities of the N_x bonds.
    delta_loc : numpy.ndarray
        Positions of the S1 minima of the bonds on their S0 coordinates.
    delta_bath : numpy.ndarray
        Positions of the S1 minima of the bath modes on their S0 coordinates.
    n_max : int
        Highest Morse level kept for every bond.

    Returns
    -------
    amplitudes : numpy.ndarray
        Shape (K, 2); column 0 is the X-H-active amplitude M_XH and column 1 the
        bath-active amplitude M_B of each state, in cm^-1. An empty states array
        (K = 0) gives shape (0, 2).

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, a quantum number is negative, a local
        quantum number exceeds n_max, or n_max lies too close to the dissociation
        limit of a bond (the condition of promoting_integrals).
    '''
    return amplitudes
```

### Step 6

06_first_order_intensities

Goal
----
Transition intensities of product states corrected to first order in the off-diagonal pure X-H anharmonic coupling, which borrows amplitude from other local-bond configurations that share the same bath quantum numbers.

```python
import itertools

import numpy as np


def first_order_intensities(states: np.ndarray, force_terms: np.ndarray, C: np.ndarray, p_xh: np.ndarray, p_bath: np.ndarray, omega_loc: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, delta_bath: np.ndarray, n_max: int, resonance_cutoff: float) -> np.ndarray:
    '''First-order corrected intensities of product states.

    Parameters
    ----------
    states : numpy.ndarray
        Product states in the layout returned by window_states: shape (K, N + 1),
        columns 0..N_x-1 the local Morse quantum numbers (bond order of omega_loc),
        columns N_x..N-1 the bath quantum numbers (order of omega_bath), last column
        the zero-order excitation energy in cm^-1 (not used here).
    force_terms : numpy.ndarray
        Local-coordinate anharmonic terms in the layout returned by
        local_force_field: shape (L, 7), rows (r, a1, a2, a3, a4, G, cls) with
        r = 3 or 4, coordinate indices a1 <= ... <= ar (0..N_x-1 local bonds,
        N_x..N-1 bath modes, a4 = -1 when r = 3), G the coefficient in cm^-1 of
        the monomial prod xi_a in the potential, and cls = 0 (pure X-H),
        1 (mixed X-H and bath) or 2 (pure bath).
    C, p_xh, p_bath, x_loc, delta_loc, delta_bath :
        As in zero_order_amplitudes.
    omega_loc : numpy.ndarray
        Harmonic frequencies of the N_x bonds in cm^-1.
    n_max : int
        Highest Morse level kept for every bond, in the states and in the sum.
    resonance_cutoff : float
        Pairs with |E_v - E_w| < resonance_cutoff (cm^-1) are left out of the
        correction; non-negative.

    Returns
    -------
    intensities : numpy.ndarray
        Shape (K,), the first-order intensities in cm^-2 (they may be negative for
        weak states). An empty states array gives shape (0,).

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, resonance_cutoff is negative, or
        n_max lies too close to the dissociation limit of a bond (the condition of
        morse_moment_matrices).
    '''
    return intensities
```

### Step 7

07_spreading_widths

Goal
----
State-resolved spreading widths of product final states from their golden-rule decay into other product states through the mixed and pure-bath anharmonic terms, with the energy-conserving delta function replaced by a narrow Lorentzian.

```python
import itertools
import math

import numpy as np


def spreading_widths(states: np.ndarray, force_terms: np.ndarray, omega_loc: np.ndarray, x_loc: np.ndarray, omega_bath: np.ndarray, n_max: int, eta: float, half_width: float) -> np.ndarray:
    '''Spreading widths (full width, cm^-1) of product states.

    Parameters
    ----------
    states : numpy.ndarray
        Product states in the layout returned by window_states: shape (K, N + 1),
        columns 0..N_x-1 the local Morse quantum numbers (bond order of omega_loc),
        columns N_x..N-1 the bath quantum numbers (order of omega_bath), last column
        the zero-order excitation energy in cm^-1 (not used here).
    force_terms : numpy.ndarray
        Local-coordinate anharmonic terms in the layout returned by
        local_force_field: shape (L, 7), rows (r, a1, a2, a3, a4, G, cls) with
        r = 3 or 4, coordinate indices a1 <= ... <= ar (0..N_x-1 local bonds,
        N_x..N-1 bath modes, a4 = -1 when r = 3), G the coefficient in cm^-1 of
        the monomial prod xi_a in the potential, and cls = 0 (pure X-H),
        1 (mixed X-H and bath) or 2 (pure bath).
    omega_loc : numpy.ndarray
        Harmonic frequencies of the bonds in cm^-1.
    x_loc : numpy.ndarray
        Anharmonicities of the bonds.
    omega_bath : numpy.ndarray
        Frequencies of the bath modes in cm^-1.
    n_max : int
        Highest Morse level kept for every bond.
    eta : float
        Half width at half maximum of the Lorentzian regulator in cm^-1, positive.
    half_width : float
        Largest allowed energy mismatch of a relaxation step in cm^-1.

    Returns
    -------
    widths : numpy.ndarray
        Shape (K,), the spreading width of each state in cm^-1 (zero when no
        relaxation step is allowed). An empty states array gives shape (0,).

    Raises
    ------
    ValueError
        If the array shapes are inconsistent, eta is not positive, half_width is
        negative, or n_max lies too close to the dissociation limit of a bond (the
        condition of morse_moment_matrices).
    '''
    return widths
```

### Step 8

08_internal_conversion_rate

Goal
----
State-correlated internal-conversion rate constant of the local-Morse doorway model: every product state in the window contributes its first-order intensity with its own spreading width and its own energy mismatch.

```python
import math

import numpy as np


def internal_conversion_rate(omega_loc: np.ndarray, x_loc: np.ndarray, delta_loc: np.ndarray, C: np.ndarray, p_xh: np.ndarray, omega_bath: np.ndarray, delta_bath: np.ndarray, p_bath: np.ndarray, cubic: np.ndarray, quartic: np.ndarray, thresholds: np.ndarray, e_gap: float, n_max: int, half_width: float, resonance_cutoff: float, eta: float) -> float:
    '''State-correlated internal-conversion rate constant in s^-1.

    Parameters
    ----------
    omega_loc, x_loc, delta_loc : numpy.ndarray
        Harmonic frequencies (cm^-1), anharmonicities and S1 minimum positions of
        the N_x local bonds.
    C : numpy.ndarray
        Local-bond transformation, shape (N_x, M_x).
    p_xh : numpy.ndarray
        Coupling coefficients of the M_x X-H normal modes, cm^-1.
    omega_bath, delta_bath, p_bath : numpy.ndarray
        Frequencies (cm^-1), S1 minimum positions and coupling coefficients
        (cm^-1) of the N_b bath modes.
    cubic, quartic : numpy.ndarray
        Normal-coordinate force field as in local_force_field.
    thresholds : numpy.ndarray
        Screening thresholds as in local_force_field.
    e_gap : float
        0-0 energy gap E_if in cm^-1.
    n_max : int
        Highest Morse level kept for every bond.
    half_width : float
        Half width of the resonance window and largest relaxation-step mismatch,
        cm^-1.
    resonance_cutoff : float
        Resonance cutoff of the first-order correction, cm^-1.
    eta : float
        Half width of the Lorentzian regulator in the spreading widths, cm^-1.

    Returns
    -------
    rate : float
        The internal-conversion rate constant k_IC in s^-1 (0.0 for an empty
        window).

    Raises
    ------
    ValueError
        If resonance_cutoff or half_width is negative or eta is not positive, and
        under the conditions listed for window_states, local_force_field,
        zero_order_amplitudes, first_order_intensities and spreading_widths.
    '''
    return rate
```

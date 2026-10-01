# Physics-Astrophysics-23

## Background

Stars like the Sun oscillate in a rich spectrum of acoustic modes excited by near-surface convection. When such a star leaves the main sequence and its core contracts while the envelope expands, the buoyancy frequency in the core rises so much that gravity modes of the core and acoustic modes of the envelope come to share the same frequency range. The two families couple through the evanescent region between the cavities, producing mixed modes that oscillate as gravity waves in the core and as sound waves in the envelope. The relative weight of the two behaviours varies from mode to mode: modes that are mostly trapped in the core are dense in period and have low visibility, whereas modes that are mostly acoustic follow the regular comb of the pressure modes and are easily observed. Space photometry from missions such as Kepler and TESS has resolved thousands of these modes in subgiants and red giants.

Mixed modes are the only direct probe of the deep interior of these stars. Their frequency pattern is well described asymptotically by a small set of global parameters: the large frequency separation of the acoustic comb, the period spacing of the gravity modes, the strength of the coupling between the two cavities and phase offsets of the two families. This framework allows an observed spectrum to be interpreted directly, without constructing a numerical stellar model, which is why asymptotic seismology has become the standard tool for characterising the cores of evolved stars.

Rotation lifts the degeneracy of the modes of a given degree with respect to the azimuthal order and splits each mode into a multiplet. In subgiants and red giants the splitting of a mixed mode depends on both the core and the envelope rotation rates. Measurements of dipolar mixed-mode splittings have shown that the cores of these stars rotate far more slowly than models of angular-momentum transport predict, a discrepancy that motivates the search for additional seismic constraints. Modes of higher angular degree sample different depths and would tighten these constraints, but their multiplets are weaker and, crucially, the multiplets of neighbouring modes can approach each other closely enough that the standard treatment of rotation as a small perturbation acting on isolated modes breaks down.

When two oscillation modes are nearly degenerate in frequency, rotation couples them and the resulting multiplets are no longer symmetric about their central component. Modelling such near-degenerate multiplets requires treating the coupled modes together rather than one at a time. This is the situation encountered for the quadrupolar mixed modes of young red giants, which is the setting of the problem considered here.

## Problem

Evolved solar-like stars oscillate in mixed modes that behave as pressure modes in the envelope and as gravity modes in the core, and the rotational splitting of these modes is the main probe of the internal rotation of subgiants and red giants. Quadrupolar ($\ell = 2$) mixed modes carry complementary information but are rarely exploited: their coupling to the core is weak, and the rotational multiplets of adjacent radial orders lie close enough in frequency that rotation couples them, distorting the multiplets into asymmetric patterns that defeat the usual symmetric-splitting analysis. A recent asymptotic formalism describes this near-degeneracy interaction of quadrupolar mixed modes directly in the mixed-mode basis, so that the asymmetric multiplets can be predicted from the global asymptotic seismic parameters alone, without any numerical stellar model. Use that formalism exactly as formulated there, including every convention of its asymptotic description and of the way it treats the rotational eigenvalue problem. The inputs are the global asymptotic parameters of the star, two observed $m = 0$ quadrupolar mixed-mode frequencies and the core and envelope rotation rates; the output studied here is the near-degeneracy asymmetry of one rotational multiplet.

Consider a young red giant with the following characteristics (all frequencies are cyclic frequencies):

- large frequency separation $\Delta\nu = 26.50\ \mu\mathrm{Hz}$; the radial mode of the radial order of interest lies at $\nu_0 = 371.204\ \mu\mathrm{Hz}$ and the $\ell = 2$ small separation is $\delta\nu_{02} = 3.104\ \mu\mathrm{Hz}$; treat the $\ell = 2$ pure acoustic modes as an exactly uniform comb of spacing $\Delta\nu$ that passes through $\nu_0 - \delta\nu_{02}$ (no curvature terms)
- asymptotic period spacing of the $\ell = 2$ gravity modes $\Delta\Pi_2 = 60.850\ \mathrm{s}$, with pure gravity-mode periods $(n + 1/2 + \varepsilon_{g,2})\,\Delta\Pi_2$; the gravity phase $\varepsilon_{g,2} \in [0, 1)$ and the $\ell = 2$ coupling factor $q_2 \in (0, 1)$ are not known and must be inferred
- two observed $\ell = 2$, $m = 0$ mixed modes of the same radial-order region, at $\nu_a = 366.805\ \mu\mathrm{Hz}$ and $\nu_b = 368.675\ \mu\mathrm{Hz}$; both are exact mixed modes of the asymptotic description (no observational error) and the inferred parameters must reproduce them exactly
- two-zone rotation with core rate $\Omega_{\mathrm{core}}/2\pi = 745\ \mathrm{nHz}$ and envelope rate $\Omega_{\mathrm{env}}/2\pi = 61\ \mathrm{nHz}$, with rotation entering the rotational operator of that formalism at leading order in the rotation rate, and with all mode-inertia quantities taken from the asymptotic description
- the coupled set of modes consists of every $\ell = 2$ mixed mode whose unperturbed (non-rotating) frequency lies in the window $[\nu_0 - 1.5\,\Delta\nu,\ \nu_0 + 1.5\,\Delta\nu]$; the near-degeneracy interaction is retained between every pair of modes of that set, and the rotational eigenvalue problem is solved exactly in that basis, in the form in which that formalism poses it (no perturbative expansion in the rotation rate and no truncation to a pair of modes); each perturbed component is attributed to the unperturbed mode that dominates its eigenvector

Report the following for this star: the exact frequencies of the $m = +2$ and $m = -2$ components of multiplets $a$ and $b$ and the asymmetry $\nu_{+2} + \nu_{-2} - 2\nu_{m=0}$ of each of the two multiplets, where $\nu_{m=0}$ denotes the frequency of the $m = 0$ component, with a physical interpretation of their signs; the same asymmetry for the multiplet of the mode of the window with the smallest trapping fraction, together with that mode's unperturbed frequency and trapping fraction; and, as supporting quantities (in any order), the trapping fractions $\zeta_a$ and $\zeta_b$ of the two observed modes, the dimensionless core-cavity and envelope-cavity near-degeneracy coupling coefficients between modes $a$ and $b$ as defined in that formalism, the number of $\ell = 2$ mixed modes in the window with the unperturbed frequencies of the mode immediately below $\nu_a$ and of the mode immediately above $\nu_b$, and the inferred coupling factor $q_2$ and gravity phase $\varepsilon_{g,2}$.

Your final answer must be a single number: the asymmetry $\nu_{+2} + \nu_{-2} - 2\nu_{m=0}$ of multiplet $a$ (the multiplet whose $m = 0$ component is at $\nu_a$), expressed in nHz. Between the <final_answer> tags place only that value, written as one finite decimal number with at least six significant figures (for example 123.456 or -0.0123456), with no units, symbols, words, equations or any other text.

Output Format Requirements:
Scientific reasoning wrapped in <reasoning>...</reasoning> tags.
Enough intermediate calculations to justify the deterministic computation, without turning the response into a general pipeline summary.
A single final numeric answer wrapped in <final_answer>...</final_answer> tags.

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

infer_coupling_factor

Goal
----
Implement infer_coupling_factor, which recovers the p-g coupling factor q of
the quadrupolar (l = 2) mixed modes of an evolved solar-like star from the
frequencies of two observed m = 0 mixed modes of the same radial-order region.

```python
def infer_coupling_factor(nu_a: float, nu_b: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    '''Coupling factor q from two m = 0 mixed-mode frequencies.

    Parameters
    ----------
    nu_a : float
        Frequency of the first observed m = 0 mixed mode, in microhertz.
    nu_b : float
        Frequency of the second observed m = 0 mixed mode, in microhertz;
        must differ from nu_a.
    delta_nu : float
        Large frequency separation of the acoustic comb, in microhertz,
        strictly positive.
    nu_p : float
        Frequency of one l = 2 pure acoustic mode, in microhertz; the comb is
        nu_p + k * delta_nu for every integer k.
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive.

    Returns
    -------
    q : float
        The unique coupling factor in (0, 1) for which a single gravity
        phase makes both observed modes exact solutions of the coupling
        relation.

    Raises
    ------
    ValueError
        If delta_nu or delta_pi is not strictly positive, if nu_a equals
        nu_b, or if no such coupling factor, or more than one, exists in
        (0, 1).
    '''
    return q
```

### Step 2

infer_gravity_phase

Goal
----
Implement infer_gravity_phase, which recovers the gravity-mode phase offset
eps_g of the quadrupolar mixed modes from one observed m = 0 mixed-mode
frequency once the coupling factor q is known.

```python
def infer_gravity_phase(nu_a: float, q: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    '''Gravity phase eps_g in [0, 1) from one m = 0 mixed-mode frequency.

    Parameters
    ----------
    nu_a : float
        Frequency of an observed m = 0 mixed mode, in microhertz, strictly positive.
    q : float
        Coupling factor of the l = 2 mixed modes, in the open interval (0, 1).
    delta_nu : float
        Large frequency separation of the acoustic comb, in microhertz,
        strictly positive.
    nu_p : float
        Frequency of one l = 2 pure acoustic mode, in microhertz.
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive.

    Returns
    -------
    eps_g : float
        The gravity phase in [0, 1) for which nu_a satisfies the coupling
        relation with the supplied q.

    Raises
    ------
    ValueError
        If q is not in (0, 1), or if nu_a, delta_nu or delta_pi is not
        strictly positive.
    '''
    return eps_g
```

### Step 3

find_mixed_modes

Goal
----
Implement find_mixed_modes, which returns every unperturbed (non-rotating)
quadrupolar mixed-mode frequency of the asymptotic model inside a frequency
window.

```python
def find_mixed_modes(nu_lo: float, nu_hi: float, q: float, eps_g: float, delta_nu: float, nu_p: float, delta_pi: float) -> "np.ndarray":
    '''All unperturbed l = 2 mixed-mode frequencies in [nu_lo, nu_hi].

    Parameters
    ----------
    nu_lo : float
        Lower edge of the window, in microhertz, strictly positive.
    nu_hi : float
        Upper edge of the window, in microhertz, strictly greater than nu_lo.
    q : float
        Coupling factor in (0, 1).
    eps_g : float
        Gravity phase (any real value; only its fractional part matters).
    delta_nu : float
        Large frequency separation of the acoustic comb, in microhertz,
        strictly positive.
    nu_p : float
        Frequency of one l = 2 pure acoustic mode, in microhertz.
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive.

    Returns
    -------
    nu_modes : np.ndarray
        One-dimensional float array of the mixed-mode frequencies in the
        window, in microhertz, sorted in increasing order, each accurate to
        better than 1e-9 microhertz. Empty if the window holds no mode.

    Raises
    ------
    ValueError
        If nu_lo is not strictly positive, if nu_hi <= nu_lo, if q is not in
        (0, 1), or if delta_nu or delta_pi is not strictly positive.
    '''
    return nu_modes
```

### Step 4

compute_trapping_fraction

Goal
----
Implement compute_trapping_fraction, which evaluates the asymptotic trapping
fraction zeta of a quadrupolar mixed mode, the ratio of the mode inertia in
the gravity-mode cavity to the total mode inertia.

```python
def compute_trapping_fraction(nu: float, q: float, delta_nu: float, nu_p: float, delta_pi: float) -> float:
    '''Asymptotic trapping fraction zeta of an l = 2 mixed mode.

    Parameters
    ----------
    nu : float
        Frequency of an unperturbed mixed mode of the model, in microhertz,
        strictly positive.
    q : float
        Coupling factor in (0, 1).
    delta_nu : float
        Large frequency separation of the acoustic comb, in microhertz,
        strictly positive.
    nu_p : float
        Frequency of one l = 2 pure acoustic mode, in microhertz.
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive.

    Returns
    -------
    zeta : float
        Trapping fraction in the open interval (0, 1).

    Raises
    ------
    ValueError
        If nu, delta_nu or delta_pi is not strictly positive, or if q is not
        in (0, 1).
    '''
    return zeta
```

### Step 5

compute_core_coupling

Goal
----
Implement compute_core_coupling, which evaluates the asymptotic core-cavity
near-degeneracy coupling coefficient gamma_c between two distinct quadrupolar
mixed modes of an evolved solar-like star.

```python
def compute_core_coupling(nu_i: float, nu_j: float, zeta_i: float, zeta_j: float, delta_pi: float) -> float:
    '''Asymptotic core near-degeneracy coupling coefficient gamma_c between two l = 2 mixed modes.

    Parameters
    ----------
    nu_i : float
        Unperturbed frequency of the first mode, in microhertz, strictly positive.
    nu_j : float
        Unperturbed frequency of the second mode, in microhertz, strictly
        positive and different from nu_i.
    zeta_i : float
        Trapping fraction of the first mode, in (0, 1].
    zeta_j : float
        Trapping fraction of the second mode, in (0, 1].
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive (the reduced spacing sqrt(6) delta_pi is formed
        internally).

    Returns
    -------
    gamma_c : float
        Dimensionless coupling coefficient, symmetric under exchange of the
        two modes.

    Raises
    ------
    ValueError
        If nu_i or nu_j is not strictly positive, if nu_i equals nu_j, if a
        trapping fraction is outside (0, 1], or if delta_pi is not strictly
        positive.
    '''
    return gamma_c
```

### Step 6

assemble_rotation_matrix

Goal
----
Implement assemble_rotation_matrix, which builds the rotational coupling
matrix of a set of quadrupolar mixed modes for one azimuthal order m in a
two-zone rotation model (core rotating at the cyclic rate nu_core, envelope
at nu_env).

```python
def assemble_rotation_matrix(nu_modes: "np.ndarray", q: float, delta_nu: float, nu_p: float, delta_pi: float, nu_core: float, nu_env: float, m: int) -> "np.ndarray":
    '''Rotational coupling matrix R (in microhertz) of N unperturbed l = 2 mixed modes.

    Parameters
    ----------
    nu_modes : np.ndarray
        One-dimensional array of N >= 1 distinct unperturbed mixed-mode
        frequencies, in microhertz, strictly positive.
    q : float
        Coupling factor in (0, 1).
    delta_nu : float
        Large frequency separation of the acoustic comb, in microhertz,
        strictly positive.
    nu_p : float
        Frequency of one l = 2 pure acoustic mode, in microhertz.
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive.
    nu_core : float
        Core rotation rate Omega_core / (2 pi), in nanohertz.
    nu_env : float
        Envelope rotation rate Omega_env / (2 pi), in nanohertz.
    m : int
        Azimuthal order, an integer with |m| <= 2.

    Returns
    -------
    rot_matrix : np.ndarray
        Symmetric float array of shape (N, N), in microhertz, with the
        first-order splitting terms on the diagonal and the near-degeneracy
        coupling terms off the diagonal; it is identically zero for m = 0
        and changes sign with m.

    Raises
    ------
    ValueError
        If nu_modes is not a one-dimensional array of distinct strictly
        positive values, if m is not an integer with |m| <= 2, or if q,
        delta_nu or delta_pi is outside its admissible range.
    '''
    return rot_matrix
```

### Step 7

solve_rotational_multiplets

Goal
----
Implement solve_rotational_multiplets, which computes the perturbed
frequencies of a set of coupled quadrupolar mixed modes for one azimuthal
order by solving the rotating eigenvalue problem exactly in the basis of the
unperturbed modes.

```python
def solve_rotational_multiplets(nu_modes: "np.ndarray", rot_matrix: "np.ndarray") -> "np.ndarray":
    '''Perturbed frequencies of N coupled mixed modes for one azimuthal order.

    Parameters
    ----------
    nu_modes : np.ndarray
        One-dimensional array of N >= 1 distinct unperturbed mixed-mode
        frequencies, in microhertz, strictly positive.
    rot_matrix : np.ndarray
        Symmetric real array of shape (N, N), in microhertz, the rotational
        coupling matrix for the azimuthal order considered.

    Returns
    -------
    nu_perturbed : np.ndarray
        Float array of shape (N,), in microhertz: entry i is the positive
        eigenfrequency whose eigenvector is dominated by unperturbed mode i.

    Raises
    ------
    ValueError
        If nu_modes is not a one-dimensional array of distinct strictly
        positive values, if rot_matrix is not a real symmetric (N, N) array,
        or if the dominant-component rule does not attribute exactly one
        positive eigenfrequency to every unperturbed mode.
    '''
    return nu_perturbed
```

### Step 8

compute_multiplet_asymmetry

Goal
----
Implement compute_multiplet_asymmetry, the end-to-end pipeline that predicts
the near-degeneracy asymmetry of one rotational multiplet of quadrupolar
mixed modes from two observed m = 0 frequencies, the global seismic
parameters and the two-zone rotation rates.

```python
def compute_multiplet_asymmetry(nu_a: float, nu_b: float, delta_nu: float, nu_p: float, delta_pi: float, nu_core: float, nu_env: float, nu_lo: float, nu_hi: float, nu_target: float, m: int = 2) -> float:
    '''Near-degeneracy asymmetry nu(+m) + nu(-m) - 2 nu(0) of one l = 2 multiplet, in nanohertz.

    Parameters
    ----------
    nu_a : float
        Frequency of the first observed m = 0 mixed mode, in microhertz.
    nu_b : float
        Frequency of the second observed m = 0 mixed mode, in microhertz,
        different from nu_a.
    delta_nu : float
        Large frequency separation of the acoustic comb, in microhertz,
        strictly positive.
    nu_p : float
        Frequency of one l = 2 pure acoustic mode, in microhertz.
    delta_pi : float
        Asymptotic period spacing of the l = 2 gravity modes, in seconds,
        strictly positive.
    nu_core : float
        Core rotation rate Omega_core / (2 pi), in nanohertz.
    nu_env : float
        Envelope rotation rate Omega_env / (2 pi), in nanohertz.
    nu_lo : float
        Lower edge of the window of coupled modes, in microhertz, strictly positive.
    nu_hi : float
        Upper edge of the window, in microhertz, greater than nu_lo.
    nu_target : float
        Frequency, in microhertz, identifying the multiplet: the unperturbed
        mode closest to it is used.
    m : int, optional
        Positive azimuthal order of the pair of components, 1 or 2 (default 2).

    Returns
    -------
    asymmetry : float
        nu(+m) + nu(-m) - 2 nu(0) for the selected multiplet, in nanohertz;
        it has the same value for m and -m.

    Raises
    ------
    ValueError
        If m is not 1 or 2, if the window contains no mixed mode, or if any
        upstream step rejects its input (see the earlier functions).
    '''
    return asymmetry
```

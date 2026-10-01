# Chemistry-Computational_Chemistry-74

## Background

## Chemistry under strong light-matter coupling

When the electric field of a small optical cavity is concentrated on a molecular transition strongly enough, photon and molecular excitation mix into hybrid states called polaritons. Experiments since the mid-2010s have reported that such coupling can change reaction rates, product branching and energy transfer even in the dark, which has made "polaritonic chemistry" an active and contested field. Condensed-phase electron transfer is a natural test case, because its rate is set by a small number of parameters that a cavity can plausibly touch: the electronic coupling between donor and acceptor, the driving force, the reorganization energy of the surrounding medium, and the time scale on which that medium fluctuates.

## Light-matter Hamiltonians and the self-energy

A molecule in a single-mode cavity is usually described in the dipole gauge, obtained from the minimal-coupling Hamiltonian by the Power-Zienau-Woolley transformation. The molecular dipole couples linearly to the cavity field, and a quadratic dipole self-energy term appears alongside it. That term is often dropped, but it is what keeps the ground state bounded and the description gauge invariant, and it matters as soon as the light-matter coupling becomes comparable to the photon energy. Once the dipole operator is projected on a donor and an acceptor state, its diagonal elements (the permanent dipoles of the two states) make the cavity field modulate the energy gap, while its off-diagonal element (the transition dipole) lets the field drive the transfer itself. Permanent dipoles in donor and acceptor states can point in opposite directions, since the transferred charge reverses the charge distribution.

## Beyond the golden rule

Most rate theories for cavity-modified electron transfer treat the donor-acceptor coupling to second order, in the spirit of Marcus theory, with the cavity entering as extra vibronic-like sidebands. Such expressions assume weak coupling and a memoryless environment. The hierarchical equations of motion avoid both assumptions for harmonic environments whose correlation functions are sums of exponentials: the influence of each environment is carried by a hierarchy of auxiliary density matrices that is exact when untruncated and converges systematically with depth. Overdamped solvent-like environments give real decay rates, while a discrete intramolecular vibration that rings for several periods before it is damped gives complex, oscillating terms, which the hierarchy has to treat with some care. Real molecules add a further complication that simple models omit. Dipole moments change with nuclear geometry, so vibrations can modulate how strongly the molecule talks to the cavity, which ties the electrons, the vibrations and the photon together in a single interaction and can make the cavity effect depend on the cavity parameters in a non-monotonic way.

## Problem

Placing a molecule in an optical cavity can change how fast an electron moves from a donor to an acceptor. A recent numerically exact study showed that the cavity effect changes character once the molecular dipoles depend on nuclear coordinates, because that dependence ties the electronic states, a vibration and the photon together in one three-body coupling; it examined a high-frequency intramolecular mode that modulates the dipoles while a separate low-frequency bath modulates the energy gap. Your task is to predict how one lossy cavity mode changes the forward transfer rate of a single donor-acceptor molecule when its permanent-dipole, transition-dipole and three-body couplings to the cavity act together. The model is a synthetic composite inspired by that study, not a system the study reports: it joins the study's minimal dipole-gauge model, dipole self-energy included, to its coordinate-dependent three-body coupling, and the two conventions this needs are chosen here and stated below, so they are not to be recovered from the study.

The donor D lies at 0 cm^-1 and the acceptor A at -100 cm^-1, with electronic coupling <D|H|A> = +30 cm^-1. A harmonic bath with a Drude-Lorentz spectral density (reorganization energy 50 cm^-1, cutoff W = 1 / (100 fs)) acts only through Q_1|A><A|, where Q_1 = sum_j lambda_j (b_j + b_j^dagger) is its collective coordinate. The cavity mode, of photon energy 100 cm^-1 and annihilation operator a, couples in the dipole gauge through hbar omega_c (a - a^dagger) sum_{X,Y} i x_XY |X><Y| together with the dipole self-energy of the dipole operator projected on {D, A}; here x_XY is the projection of <X|mu|Y> on the cavity polarization divided by sqrt(2 hbar omega_c V epsilon_0), with x_DD = +0.2, x_AA = -0.1 and x_DA = x_AD = +0.3. A separate, independent intramolecular vibration is an underdamped Brownian oscillator of frequency 500 cm^-1, damping rate Gamma = 1 / (100 fs) (the friction constant, so its oscillating correlation terms decay at Gamma / 2) and reorganization energy 1 cm^-1, with coordinate Q_2. It shifts each of the four x_XY to x_XY + Q_2 / (hbar omega_c) in the light-matter coupling term only, so the three-body coupling uses the same dipole-gauge quadrature i(a - a^dagger) as the constant dipoles, where the study writes its three-body term with (a + a^dagger). The vibration enters neither the self-energy nor the energies, so the self-energy of the coordinate-dependent term is left out as in the study, while the projected self-energy of the constant dipoles, which the study's three-body analysis does not include, is kept. Both baths are at 300 K, and photons leave the cavity through the Lindblad term kappa (a rho a^dagger - {a^dagger a, rho} / 2), with kappa = 2 pi c (10 cm^-1), into a zero-temperature continuum.

Propagate from |D><D| times the photon vacuum |0><0|, with both baths in equilibrium, using the hierarchical equations of motion. Represent the correlation function <Q(t) Q(0)> / hbar^2 of the low-frequency bath by its Drude term alone and that of the vibration by its two oscillating terms alone, with no Matsubara terms for either. Truncate the hierarchy at total depth 10 over all three terms with deeper auxiliary operators set to zero, let the photon damping act on every auxiliary operator, keep the cavity Fock states 0 to 5, and advance time by fixed-step fourth-order Runge-Kutta with a 1 fs step. Convert energies to angular frequencies with c = 2.99792458 x 10^-2 cm ps^-1 and use k_B = 0.6950348 cm^-1 K^-1. Obtain the forward and backward rates k_DA and k_AD by an unweighted least-squares fit of the donor population, sampled every 10 fs from 0 to 5 ps, to reversible first-order two-state kinetics that start from p_D = 1 with non-negative rates. Obtain the cavity-free forward rate by the same procedure with every x_XY, the vibration and kappa removed.

Report R = k_DA(with cavity) / k_DA(without cavity) as the final answer to at least four decimal places. In the reasoning, give k_DA and k_AD with the cavity and the forward rate without it in ps^-1, the donor population at 1 ps with the cavity, the ratio R recomputed without the vibration and again with kappa = 0, and the coupling and the energy gap between |D,0> and |A,0> once the self-energy is included.

Using external primary scientific literature beyond the supplied paper, also report:
- For a collective polariton-mediated electron-transfer model with single-molecule cavity coupling g_c = 4 meV, acceptor reorganization energy lambda_A = 200 meV, donor-acceptor coupling V_DA = 5 meV, Delta G = 0 and T = 300 K, the interval of molecule numbers over which the cavity enhances the forward rate relative to the cavity-free reaction and the largest reported enhancement factor.
- For an ultrastrong vibrational-polariton electron-transfer model at g_R = g_P = omega_R = omega_P, the reported leading coefficients c_0 and c_2 in the hybrid ground-state expansion, including their signs.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number. Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-fold candidate tables.

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

01_drude_bath_exponents

Goal
----
Exponential decomposition of the correlation function of an overdamped Drude-Lorentz bath: the Drude term and the first Matsubara terms, as complex amplitudes and decay rates in picosecond units.

```python
def drude_bath_exponents(reorg_cm: float, tau_bath_fs: float, temperature_k: float, n_matsubara: int) -> "np.ndarray":
    '''Drude-Lorentz bath correlation function as a finite sum of exponentials.

    Parameters
    ----------
    reorg_cm : float
        Reorganization energy lambda_0 of the bath in cm^-1, non-negative.
    tau_bath_fs : float
        Bath relaxation time in fs; the Drude cutoff is W = 1 / tau_bath.
    temperature_k : float
        Temperature in K.
    n_matsubara : int
        Number of Matsubara terms kept after the Drude term.

    Returns
    -------
    exponents : numpy.ndarray
        Real array of shape (n_matsubara + 1, 4). Row k holds (Re c_k, Im c_k, Re gamma_k,
        Im gamma_k) of the term c_k exp(-gamma_k t) of C(t) = <Q(t)Q(0)>/hbar^2,
        with c_k in ps^-2 and gamma_k in ps^-1.
        Row 0 is the Drude term (gamma_0 = W); rows 1..n_matsubara are the
        Matsubara terms in increasing order of frequency.

    Raises
    ------
    ValueError
        If reorg_cm is negative or not finite, tau_bath_fs or temperature_k is not
        positive and finite, n_matsubara is not a non-negative integer, or a kept
        Matsubara frequency coincides with W to relative precision 1e-12.
    '''
    return exponents
```

### Step 2

02_underdamped_mode_exponents

Goal
----
Exponential decomposition of the correlation function of an underdamped intramolecular vibration treated as a Brownian oscillator: the oscillating pair of terms and the first Matsubara terms, with complex decay rates.

```python
def underdamped_mode_exponents(reorg_cm: float, mode_cm: float, tau_damp_fs: float, temperature_k: float, n_matsubara: int) -> "np.ndarray":
    '''Brownian-oscillator (underdamped mode) correlation function as a finite sum of exponentials.

    Parameters
    ----------
    reorg_cm : float
        Reorganization energy lambda of the mode bath in cm^-1, non-negative.
    mode_cm : float
        Mode frequency hbar omega_0 in cm^-1, positive.
    tau_damp_fs : float
        Damping time in fs; the damping rate is Gamma = 1 / tau_damp.
    temperature_k : float
        Temperature in K.
    n_matsubara : int
        Number of Matsubara terms kept after the oscillating pair.

    Returns
    -------
    exponents : numpy.ndarray
        Real array of shape (n_matsubara + 2, 4). Row k holds (Re c_k, Im c_k, Re gamma_k,
        Im gamma_k) of the term c_k exp(-gamma_k t) of C(t) = <Q(t)Q(0)>/hbar^2,
        with c_k in ps^-2 and gamma_k in ps^-1.
        Rows 0 and 1 are the oscillating pair, row 0 the term whose rate has a
        negative imaginary part and row 1 the term whose rate has a positive
        imaginary part; rows 2..n_matsubara + 1 are the Matsubara terms in
        increasing order of frequency.

    Raises
    ------
    ValueError
        If reorg_cm is negative or not finite, mode_cm, tau_damp_fs or
        temperature_k is not positive and finite, n_matsubara is not a
        non-negative integer, or the mode is not underdamped (omega_0 <= Gamma / 2).
    '''
    return exponents
```

### Step 3

03_dipole_gauge_hamiltonian

Goal
----
System Hamiltonian of a donor-acceptor pair coupled to one cavity mode in the dipole gauge, including the dipole self-energy, on a truncated photon Fock basis.

```python
def dipole_gauge_hamiltonian(driving_force_cm: float, electronic_coupling_cm: float, cavity_energy_cm: float, x_dd: float, x_aa: float, x_da: float, n_photon: int) -> "np.ndarray":
    '''Dipole-gauge Hamiltonian of the donor-acceptor pair plus one cavity mode.

    Parameters
    ----------
    driving_force_cm : float
        Driving force F in cm^-1; E_D = 0 and E_A = -F.
    electronic_coupling_cm : float
        Electronic coupling V_DA between D and A in cm^-1.
    cavity_energy_cm : float
        Photon energy hbar omega_c in cm^-1, positive.
    x_dd, x_aa, x_da : float
        Real dimensionless dipole projections x_DD, x_AA and x_DA = x_AD.
    n_photon : int
        Number of photon Fock states kept (0 .. n_photon - 1), at least 1.

    Returns
    -------
    hamiltonian : numpy.ndarray
        Complex Hermitian array of shape (2 n_photon, 2 n_photon) in cm^-1, basis
        index e * n_photon + m with e = 0 (D), 1 (A) and photon number m.

    Raises
    ------
    ValueError
        If n_photon is not a positive integer, cavity_energy_cm is not positive
        and finite, or any other argument is not finite.
    '''
    return hamiltonian
```

### Step 4

04_bath_coupling_operators

Goal
----
System operators that multiply the two bath coordinates: the acceptor projector for the low-frequency bath and the three-body electron-vibration-photon operator for the intramolecular mode.

```python
def bath_coupling_operators(n_photon: int) -> "np.ndarray":
    '''System operators V_1 and V_2 that multiply the bath coordinates Q_1 and Q_2.

    Parameters
    ----------
    n_photon : int
        Number of photon Fock states kept, at least 1.

    Returns
    -------
    coupling_ops : numpy.ndarray
        Complex array of shape (2, 2 n_photon, 2 n_photon), dimensionless, basis
        index e * n_photon + m as in step 03: coupling_ops[0] = V_1 (low-frequency
        bath) and coupling_ops[1] = V_2 (intramolecular mode), both Hermitian.

    Raises
    ------
    ValueError
        If n_photon is not a positive integer.
    '''
    return coupling_ops
```

### Step 5

05_system_liouvillian

Goal
----
Superoperator of the system alone: coherent evolution under the dipole-gauge Hamiltonian plus Lindblad photon loss from the cavity into a zero-temperature continuum.

```python
def system_liouvillian(hamiltonian_cm: "np.ndarray", kappa_cm: float, n_photon: int) -> "np.ndarray":
    '''System Liouvillian with Lindblad photon loss, in rad/ps.

    Parameters
    ----------
    hamiltonian_cm : numpy.ndarray
        Hermitian (2 n_photon, 2 n_photon) Hamiltonian in cm^-1 (basis of step 03).
    kappa_cm : float
        Photon loss rate expressed as hbar kappa in cm^-1, non-negative.
    n_photon : int
        Number of photon Fock states in the basis.

    Returns
    -------
    liouvillian : numpy.ndarray
        Complex array of shape (d^2, d^2) with d = 2 n_photon, in ps^-1, acting on
        row-major vectorised density matrices.

    Raises
    ------
    ValueError
        If n_photon is not a positive integer, the Hamiltonian is not a Hermitian
        array of shape (2 n_photon, 2 n_photon) (tolerance 1e-10), or kappa_cm is
        negative or not finite.
    '''
    return liouvillian
```

### Step 6

06_heom_propagate

Goal
----
Hierarchical equations of motion for a system coupled to several independent harmonic baths whose correlation functions are finite sums of exponentials with real or complex rates: propagation of the reduced density matrix with fixed-step fourth-order Runge-Kutta.

```python
def heom_propagate(liouvillian: "np.ndarray", coupling_ops: "np.ndarray", exponents: "np.ndarray", bath_index: "np.ndarray", depth: int, rho0: "np.ndarray", dt_ps: float, n_steps: int, stride: int) -> "np.ndarray":
    '''Reduced density matrix from the hierarchical equations of motion with several baths.

    Parameters
    ----------
    liouvillian : numpy.ndarray
        (d^2, d^2) system superoperator in ps^-1 acting on row-major vectorised
        d x d matrices (step 05).
    coupling_ops : numpy.ndarray
        (B, d, d) array of Hermitian system operators V_b, one per bath.
    exponents : numpy.ndarray
        (K, 4) array of rows (Re c_k, Im c_k, Re gamma_k, Im gamma_k), in ps^-2
        and ps^-1, for all exponential terms of all baths.
    bath_index : numpy.ndarray
        (K,) integer array; bath_index[k] is the bath b to which term k belongs,
        so C_b(t) = sum over k with bath_index[k] = b of c_k exp(-gamma_k t).
    depth : int
        Hierarchy depth: auxiliary operators with index sum <= depth are kept.
    rho0 : numpy.ndarray
        (d, d) initial reduced density matrix.
    dt_ps : float
        Runge-Kutta time step in ps.
    n_steps : int
        Number of steps, a non-negative multiple of stride.
    stride : int
        Output every stride steps.

    Returns
    -------
    frames : numpy.ndarray
        Complex array of shape (n_steps // stride + 1, d, d): the reduced density
        matrix at times 0, stride * dt, ..., n_steps * dt.

    Raises
    ------
    ValueError
        If the shapes of liouvillian, coupling_ops and rho0 are inconsistent, a
        coupling operator is not Hermitian (tolerance 1e-10), exponents is not a
        finite array of shape (K, 4) with K >= 1, bath_index does not assign each
        term to an existing bath, the rates of some bath are not closed under
        complex conjugation (to within 1e-9 max(1, |gamma|)), depth is not a
        non-negative integer, dt_ps is not positive and finite, or n_steps is not
        a non-negative integer multiple of the positive integer stride.
    '''
    return frames
```

### Step 7

07_donor_population_trace

Goal
----
Donor population versus time for the cavity-coupled donor-acceptor pair with a low-frequency bath and an intramolecular mode, starting in the donor state with the cavity in its vacuum.

```python
def donor_population_trace(driving_force_cm: float, electronic_coupling_cm: float, cavity_energy_cm: float, x_dd: float, x_aa: float, x_da: float, kappa_cm: float, reorg_cm: float, tau_bath_fs: float, mode_reorg_cm: float, mode_cm: float, tau_mode_fs: float, temperature_k: float, n_matsubara_bath: int, n_matsubara_mode: int, depth: int, n_photon: int, t_max_ps: float, dt_ps: float, sample_ps: float) -> "np.ndarray":
    '''Donor population p_D(t) of the cavity-coupled donor-acceptor model.

    Parameters
    ----------
    driving_force_cm, electronic_coupling_cm, cavity_energy_cm : float
        F, V_DA and hbar omega_c in cm^-1 (step 03).
    x_dd, x_aa, x_da : float
        Dimensionless dipole projections (step 03).
    kappa_cm : float
        Photon loss rate as hbar kappa in cm^-1 (step 05).
    reorg_cm, tau_bath_fs : float
        Reorganization energy (cm^-1) and relaxation time (fs) of the
        low-frequency Drude-Lorentz bath.
    mode_reorg_cm, mode_cm, tau_mode_fs : float
        Reorganization energy (cm^-1), frequency (cm^-1) and damping time (fs)
        of the intramolecular mode; mode_reorg_cm = 0 removes the mode.
    temperature_k : float
        Temperature of both baths in K.
    n_matsubara_bath, n_matsubara_mode : int
        Matsubara terms kept for the low-frequency bath and for the mode.
    depth : int
        Hierarchy depth.
    n_photon : int
        Photon Fock states kept.
    t_max_ps, dt_ps, sample_ps : float
        Final time, integration step and sampling interval in ps; t_max must be a
        multiple of sample_ps and sample_ps a multiple of dt_ps.

    Returns
    -------
    p_donor : numpy.ndarray
        Real array of length round(t_max / sample) + 1: p_D at t = 0, sample, ...,
        t_max.

    Raises
    ------
    ValueError
        If t_max_ps, dt_ps or sample_ps is not positive and finite, if the three
        times are not commensurate as described, if mode_reorg_cm is negative or
        not finite, or if any argument is rejected by steps 01 to 06.
    '''
    return p_donor
```

### Step 8

08_fit_transfer_rates

Goal
----
Forward and backward electron-transfer rate constants from a donor-population trace by least-squares fitting to reversible first-order two-state kinetics.

```python
def fit_transfer_rates(times_ps: "np.ndarray", p_donor: "np.ndarray") -> "np.ndarray":
    '''Least-squares rate constants of reversible two-state kinetics.

    Parameters
    ----------
    times_ps : numpy.ndarray
        1-D array of sample times in ps, starting at 0 and strictly increasing.
    p_donor : numpy.ndarray
        1-D array of donor populations at those times, same length (>= 3).

    Returns
    -------
    rates : numpy.ndarray
        Array [k_DA, k_AD] in ps^-1 minimising the sum of squared residuals.

    Raises
    ------
    ValueError
        If the arrays are not one dimensional of equal length at least 3, contain
        non-finite values, or the times do not start at 0 and increase strictly.
    '''
    return rates
```

### Step 9

09_cavity_rate_enhancement

Goal
----
Orchestrator: ratio of the forward electron-transfer rate with the lossy cavity, including the three-body coupling through the intramolecular mode, to the forward rate of the same molecule without the cavity.

```python
def cavity_rate_enhancement(driving_force_cm: float, electronic_coupling_cm: float, cavity_energy_cm: float, x_dd: float, x_aa: float, x_da: float, kappa_cm: float, reorg_cm: float, tau_bath_fs: float, mode_reorg_cm: float, mode_cm: float, tau_mode_fs: float, temperature_k: float, n_matsubara_bath: int, n_matsubara_mode: int, depth: int, n_photon: int, t_max_ps: float, dt_ps: float, sample_ps: float) -> float:
    '''Cavity enhancement R = k_DA(cavity) / k_DA(no cavity) of the forward transfer rate.

    Parameters
    ----------
    driving_force_cm, electronic_coupling_cm, cavity_energy_cm : float
        F, V_DA and hbar omega_c in cm^-1.
    x_dd, x_aa, x_da : float
        Dimensionless dipole projections.
    kappa_cm : float
        Photon loss rate as hbar kappa in cm^-1.
    reorg_cm, tau_bath_fs : float
        Low-frequency bath reorganization energy (cm^-1) and relaxation time (fs).
    mode_reorg_cm, mode_cm, tau_mode_fs : float
        Intramolecular mode reorganization energy (cm^-1), frequency (cm^-1) and
        damping time (fs).
    temperature_k : float
        Temperature in K.
    n_matsubara_bath, n_matsubara_mode : int
        Matsubara terms kept for the bath and for the mode.
    depth : int
        Hierarchy depth.
    n_photon : int
        Photon Fock states kept in the cavity run.
    t_max_ps, dt_ps, sample_ps : float
        Final time, integration step and sampling interval in ps (step 07).

    Returns
    -------
    ratio : float
        R = k_DA with the cavity divided by k_DA without it.

    Raises
    ------
    ValueError
        If any argument is rejected by steps 01 to 08, or if the cavity-free
        forward rate is zero.
    '''
    return ratio
```

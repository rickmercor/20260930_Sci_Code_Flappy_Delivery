# Physics-Condensed_Matter_Physics-4

## Background

Stacking two transition-metal dichalcogenide monolayers with a small twist angle creates a moiré superlattice whose period is many times the atomic lattice constant. In type-II heterobilayers such as MoSe₂/WSe₂ the lowest optical excitations are interlayer excitons, with the electron and the hole in different layers, a permanent out-of-plane dipole and long lifetimes. The local atomic registry of the two layers changes across the moiré cell and with it the band-edge energies, so the exciton centre of mass moves in a periodic potential. Its free parabolic dispersion then splits into mini-bands separated by gaps. At small twist angles these mini-bands become flat and localise excitons, while at larger angles they approach the free-particle picture.

After optical excitation, excitons are created far above the band minimum and cool by emitting phonons. In a mini-band landscape, cooling can stall when the energy gaps between mini-bands do not match the available phonon energies, a relaxation bottleneck known from quantum dots and superlattices. Whether excitons then move faster or slower depends on where the non-thermal population sits in momentum space, because transport is controlled by group velocities and scattering times rather than by energy alone. Describing this requires following energy relaxation and spatial propagation together, including the broadening of states by their own scattering.

## Problem

Twisted heterobilayers of transition-metal dichalcogenides host interlayer excitons whose centre-of-mass motion is folded by the moiré potential into mini-bands, some of them nearly flat, and whether such excitons spread quickly or slowly after optical excitation decides their use in excitonic devices. A 2025 microscopic study of twisted hBN-encapsulated MoSe₂/WSe₂ solved a Boltzmann transport equation in the moiré-exciton basis, with phonon scattering, collisional broadening and the full two-dimensional mini-band structure, and found that at small twist angles and low temperature a phonon relaxation bottleneck keeps excitons hot and diffusion faster than a thermal (Boltzmann) population would allow. The task is one concrete, deterministic instance of that calculation for an R-stacked MoSe₂/WSe₂ bilayer with electron in MoSe₂ and hole in WSe₂, twisted by 3° at a lattice temperature of 10 K. The inputs are the material parameters and numerical conventions below; the output is the diffusion coefficient of the exciton population 100 ps after a hot excitation.

The calculation needs four ingredients, each built as in that study. First, the interlayer 1s exciton sees a moiré potential assembled from the registry-dependent K-point band-edge shifts of the conduction band of the electron layer and the valence band of the hole layer, each averaged over the exciton's relative-motion density. Second, the zone-folded exciton Hamiltonian gives mini-band energies, eigenvectors and group velocities on a grid of the moiré Brillouin zone. Third, deformation-potential exciton–phonon coupling, projected onto moiré exciton states, gives phonon-assisted transition rates whose energy conservation is broadened self-consistently by the dephasings of the states involved. Fourth, the spatially homogeneous limit of the transport equation relaxes the hot population, and a relaxation-time description turns the resulting occupation, velocities and scattering times into a diffusion coefficient.

Compute the moiré-exciton mini-bands, the self-consistent dephasings, the occupation of every mini-band state 100 ps after excitation, and from it the diffusion coefficient in cm²/s. For comparison, evaluate the same relaxation-time expression with the same rates for a Boltzmann occupation at 10 K.

Material parameters and conventions:

- Lattice constants 0.327 nm (MoSe₂) and 0.325 nm (WSe₂); build the moiré lattice from 0.327 nm with the exact twist geometry a_M = a/[2 sin(θ/2)], neglecting the lattice mismatch. Use primitive moiré reciprocal vectors b1 = G(√3/2, −1/2) and b2 = G(0, 1).
- Masses m_e = 0.64 (MoSe₂ conduction band, K) and m_h = 0.51 (WSe₂ valence band, K), in units of m₀.
- Registry parameters (γ1, γ2) of the R-type MoSe₂/WSe₂ bilayer: MoSe₂ conduction band (−4.389, −6.178) meV, WSe₂ valence band (−1.467, −5.856) meV.
- Interlayer 1s exciton: binding energy 173 meV. Model the relative motion as the 2D hydrogenic 1s state with amplitude ∝ exp(−r/a_B) and fix a_B from the binding energy through the virial relation of that state.
- Plane waves Q + i b1 + j b2 with max(|i|, |j|, |i − j|) ≤ 2. Keep the lowest 6 mini-bands at each Q on the shifted grid Q = ((u + ½)/9) b1 + ((v + ½)/9) b2, u, v = 0…8. Measure energies from the lowest mini-band energy on this grid.
- Phonons, one bath per layer, each coupling to its own carrier. MoSe₂: v_LA = 4.1 nm/ps, D1 = 3.4 eV, D0 = 5.2×10⁸ eV/cm, LO energy 36.6 meV, molar mass 253.86 g/mol. WSe₂: v_LA = 3.3 nm/ps, D1 = 2.1 eV, D0 = 3.1×10⁸ eV/cm, A1 energy 30.8 meV, molar mass 341.76 g/mol. Take each layer's areal mass density as one formula unit per hexagonal cell of that layer's own lattice constant. The acoustic branch is linear and the optical branch dispersionless; an acoustic phonon of zero momentum transfer does not contribute. Momentum transfers include umklapp vectors i b1 + j b2 with max(|i|, |j|, |i − j|) ≤ 2.
- Energy conservation: replace the delta function by a normalised Gaussian exp(−(x/w)²)/(w√π) of the energy mismatch x, with width w equal to the sum of the dephasings of the initial and final states. Obtain the dephasings self-consistently by plain fixed-point iteration from 1 meV for every state, stopping at the first update whose largest change is below 10⁻⁸ meV.
- Excitation: equal occupation of every mini-band state whose energy lies within 60 ± 3.5 meV (inclusive). Propagate the homogeneous master equation exactly to t = 100 ps.
- Constants: ħ = 0.6582119569 meV ps, ħ²/(2m₀) = 38.09982 meV nm², k_B = 0.08617333262 meV/K, N_A = 6.02214076×10²³ mol⁻¹.

## What to report

- The exciton radius a_B in nm.
- The complex moiré amplitude Θ of the interlayer exciton, defined by V(R) = Σ_s Θ exp(i s·R) + c.c. over s = b1, b2, −(b1 + b2), in meV.
- The width of the lowest mini-band, its gap to the next mini-band, and the number of mini-band states in the excitation window.
- The mean self-consistent dephasing over all states, in meV.
- The total occupation of the lowest and of the second-lowest mini-band at 100 ps.
- The diffusion coefficient for the Boltzmann occupation, and the ratio of the relaxed to the Boltzmann coefficient.
- The diffusion coefficient of the relaxed population at 100 ps in cm²/s, to four decimal places; this is the number that goes in the final-answer tag.
- Name the sources you relied on and state the relations you took from each one.

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
from scipy.linalg import expm
```

### Step 1

moire_reciprocal_lattice

Goal
----
Compute the two primitive reciprocal-lattice vectors of the moire superlattice formed by twisting two layers with a common lattice constant.

```python
def moire_reciprocal_lattice(theta_deg: float, a_lattice: float) -> "np.ndarray":
    """Return the moire reciprocal-lattice vectors for a twisted hexagonal bilayer.

    Parameters
    ----------
    theta_deg : float
        Twist angle in degrees, 0 < theta_deg < 60.
    a_lattice : float
        Monolayer lattice constant in nm, positive.

    Returns
    -------
    b_vectors : np.ndarray
        Float array of shape (2, 2); row 0 is b1 = G (sqrt(3)/2, -1/2), row 1 is b2 = G (0, 1),
        where G is the length of a primitive moire reciprocal vector, in nm^-1.

    Raises
    ------
    ValueError
        If theta_deg is not strictly between 0 and 60 or a_lattice is not positive.
    """
    return b_vectors
```

### Step 2

interlayer_moire_coupling

Goal
----
Compute the complex Fourier amplitude of the moire potential felt by the centre of mass of a 1s interlayer exciton.

```python
def interlayer_moire_coupling(gamma_c: "np.ndarray", gamma_v: "np.ndarray", m_e: float, m_h: float,
                              e_b: float, g0: float) -> "np.ndarray":
    """Return the exciton moire amplitude Theta on the first-shell moire vectors b1, b2 and -(b1 + b2).

    Parameters
    ----------
    gamma_c : np.ndarray
        (gamma1, gamma2) of the conduction band of the electron layer, meV.
    gamma_v : np.ndarray
        (gamma1, gamma2) of the valence band of the hole layer, meV.
    m_e, m_h : float
        Electron and hole effective masses in units of the free-electron mass.
    e_b : float
        1s binding energy of the interlayer exciton, meV, positive.
    g0 : float
        Length of a first-shell moire reciprocal vector, nm^-1.

    Returns
    -------
    theta : np.ndarray
        Float array [Re Theta, Im Theta] in meV. The exciton potential is
        V(R) = sum over the three vectors g of Theta exp(i g.R) + complex conjugate.
    """
    return theta
```

### Step 3

moire_exciton_bands

Goal
----
Diagonalise the zone-folded moire exciton Hamiltonian on a grid of the moire Brillouin zone and return mini-band energies, eigenvectors and squared group velocities.

```python
def moire_exciton_bands(b_vectors: "np.ndarray", theta_coupling: "np.ndarray", m_e: float, m_h: float,
                        n_shell: int, nk: int, n_bands: int) -> tuple:
    """Return the moire exciton mini-bands on the shifted nk x nk grid.

    Parameters
    ----------
    b_vectors : np.ndarray
        (2, 2) moire reciprocal vectors, rows b1 and b2, nm^-1.
    theta_coupling : np.ndarray
        [Re Theta, Im Theta] of the exciton moire amplitude, meV.
    m_e, m_h : float
        Electron and hole masses in units of the free-electron mass.
    n_shell : int
        Plane-wave cutoff, max(|i|, |j|, |i - j|) <= n_shell.
    nk : int
        Grid points per reciprocal direction.
    n_bands : int
        Number of lowest mini-bands kept at each grid point.

    Returns
    -------
    result : tuple
        (q_pts, energies, coeffs, v2, basis):
        q_pts (nk*nk, 2) grid momenta in nm^-1, index k = u*nk + v;
        energies (nk*nk, n_bands) ascending eigenvalues in meV, no offset removed;
        coeffs (nk*nk, n_pw, n_bands) complex normalised eigenvectors in the plane-wave order, column n
        belonging to energies[k, n];
        v2 (nk*nk, n_bands) squared group-velocity magnitudes in nm^2/ps^2;
        basis (n_pw, 2) integer (i, j) pairs in the plane-wave order.

    Raises
    ------
    ValueError
        If n_bands exceeds the number of plane waves.
    """
    return q_pts, energies, coeffs, v2, basis
```

### Step 4

phonon_matrix_elements

Goal
----
Evaluate the squared exciton-phonon coupling of one carrier's layer at a set of momentum transfers, for the longitudinal acoustic and the optical deformation-potential channels.

```python
def phonon_matrix_elements(q: "np.ndarray", mass_ratio: float, a_b: float, v_la: float, d1: float, d0: float,
                           e_op: float, a_lattice: float, molar_mass: float) -> "np.ndarray":
    """Return area-scaled squared couplings and the acoustic phonon energy at momentum transfers q.

    Parameters
    ----------
    q : np.ndarray
        Momentum-transfer magnitudes, nm^-1, any shape, non-negative.
    mass_ratio : float
        Fraction of the relative coordinate by which this carrier sits away from the exciton centre of mass.
    a_b : float
        Exciton radius a_B of the 1s state, nm.
    v_la : float
        Longitudinal acoustic sound velocity, nm/ps.
    d1 : float
        Acoustic deformation potential, meV.
    d0 : float
        Optical deformation potential, meV/nm.
    e_op : float
        Optical phonon energy, meV.
    a_lattice : float
        Lattice constant of the carrier's layer, nm.
    molar_mass : float
        Molar mass of one formula unit of the layer, g/mol.

    Returns
    -------
    elements : np.ndarray
        Array of shape (3,) + q.shape: [A |g_ac|^2 (meV^2 nm^2), hbar Omega_ac (meV), A |g_op|^2 (meV^2 nm^2)].
    """
    return elements
```

### Step 5

scattering_weights

Goal
----
Assemble the broadening-independent weights of phonon-assisted scattering between moire exciton states, resolved by umklapp vector, layer and emission or absorption.

```python
def scattering_weights(q_pts: "np.ndarray", coeffs: "np.ndarray", basis: "np.ndarray", b_vectors: "np.ndarray",
                       m_e: float, m_h: float, e_b: float, temperature: float) -> tuple:
    """Return the scattering weights of every ordered pair of moire exciton states.

    States are flattened as s = k * n_bands + n, with k the grid index of q_pts.

    Parameters
    ----------
    q_pts : np.ndarray
        (N_k, 2) grid momenta, nm^-1.
    coeffs : np.ndarray
        (N_k, n_pw, n_bands) complex eigenvector coefficients in the order of basis.
    basis : np.ndarray
        (n_pw, 2) integer plane-wave labels (i, j).
    b_vectors : np.ndarray
        (2, 2) moire reciprocal vectors, rows b1, b2, nm^-1.
    m_e, m_h : float
        Electron and hole masses, units of the free-electron mass.
    e_b : float
        1s interlayer binding energy, meV.
    temperature : float
        Lattice temperature, K; k_B = 0.08617333262 meV/K.

    Returns
    -------
    weights : tuple
        (w_ac, omega, w_op, e_op):
        w_ac (S, S, 19, 2, 2), index [s_initial, s_final, d, layer, channel], channel 0 = emission
        (kernel centred at E_final = E_initial - hbar Omega), channel 1 = absorption, in meV/ps;
        omega (N_k, N_k, 19, 2), index [k_initial, k_final, d, layer], acoustic phonon energy in meV;
        w_op (S, S, 2, 2), index [s_initial, s_final, layer, channel], optical weights summed over d, meV/ps;
        e_op (2,) optical phonon energies of layers 0 and 1, meV.
    """
    return w_ac, omega, w_op, e_op
```

### Step 6

transition_rate_matrix

Goal
----
Build the phonon-assisted transition-rate matrix between moire exciton states for given state dephasings, replacing strict energy conservation by a broadened kernel.

```python
def transition_rate_matrix(energies: "np.ndarray", weights: tuple, gamma: "np.ndarray") -> "np.ndarray":
    """Return R with R[f, i] the phonon-assisted rate from state i into state f.

    Parameters
    ----------
    energies : np.ndarray
        (N_k, n_bands) mini-band energies, meV; state s = k * n_bands + n.
    weights : tuple
        (w_ac, omega, w_op, e_op) in the layout returned by scattering_weights.
    gamma : np.ndarray
        Dephasing of every state, meV, shape (N_k * n_bands,) or (N_k, n_bands), positive.

    Returns
    -------
    rates : np.ndarray
        (S, S) float array in 1/ps, S = N_k * n_bands, column i holding the rates out of state i.
    """
    return rates
```

### Step 7

self_consistent_dephasing

Goal
----
Solve for the state dephasings that are consistent with the broadened scattering rates they produce.

```python
def self_consistent_dephasing(energies: "np.ndarray", weights: tuple, gamma0: float, tol: float,
                              max_iter: int) -> "np.ndarray":
    """Return the converged dephasing of every state.

    Parameters
    ----------
    energies : np.ndarray
        (N_k, n_bands) mini-band energies, meV.
    weights : tuple
        (w_ac, omega, w_op, e_op) as returned by scattering_weights.
    gamma0 : float
        Uniform starting dephasing, meV, positive.
    tol : float
        Convergence threshold on the largest absolute update, meV.
    max_iter : int
        Maximum number of updates.

    Returns
    -------
    gamma : np.ndarray
        (N_k * n_bands,) dephasings in meV: the first update whose largest change is below tol.

    Raises
    ------
    ValueError
        If no update meets tol within max_iter updates.
    """
    return gamma
```

### Step 8

relaxed_distribution

Goal
----
Propagate the homogeneous occupation of the moire exciton states from a hot excitation window through the phonon master equation to a given time.

```python
def relaxed_distribution(rates: "np.ndarray", energies: "np.ndarray", e_center: float, half_width: float,
                         t_eval: float) -> "np.ndarray":
    """Return the state occupations at time t_eval after the excitation.

    Parameters
    ----------
    rates : np.ndarray
        (S, S) rates in 1/ps, rates[f, i] from state i into state f.
    energies : np.ndarray
        State energies in meV, any shape with S entries in the flattened state order.
    e_center : float
        Centre of the excitation window above the lowest energy, meV.
    half_width : float
        Half-width of the window, meV.
    t_eval : float
        Evaluation time, ps.

    Returns
    -------
    occupation : np.ndarray
        (S,) occupations at t_eval.

    Raises
    ------
    ValueError
        If no state lies inside the excitation window.
    """
    return occupation
```

### Step 9

rta_diffusion_coefficient

Goal
----
Compute the exciton diffusion coefficient of a given occupation of the moire mini-band states in the relaxation-time approximation.

```python
def rta_diffusion_coefficient(v2: "np.ndarray", rates: "np.ndarray", occupation: "np.ndarray") -> float:
    """Return the relaxation-time-approximation diffusion coefficient.

    Parameters
    ----------
    v2 : np.ndarray
        Squared group-velocity magnitudes, nm^2/ps^2, S entries in the flattened state order.
    rates : np.ndarray
        (S, S) rates in 1/ps, rates[f, i] from state i into state f.
    occupation : np.ndarray
        Non-negative occupations, S entries, not necessarily normalised.

    Returns
    -------
    d : float
        Diffusion coefficient in cm^2/s.
    """
    return d
```

### Step 10

moire_exciton_diffusion

Goal
----
Compute the diffusion coefficient of hot-excited interlayer excitons in a twisted R-stacked MoSe2/WSe2 heterobilayer after phonon relaxation through the moire mini-bands.

```python
def moire_exciton_diffusion(theta_deg: float, temperature: float, nk: int, e_center: float, half_width: float,
                            t_eval: float) -> float:
    """Return the diffusion coefficient of the relaxed moire exciton population.

    Parameters
    ----------
    theta_deg : float
        Twist angle, degrees.
    temperature : float
        Lattice temperature, K.
    nk : int
        Grid points per moire reciprocal direction.
    e_center : float
        Centre of the uniform excitation window above the lowest mini-band energy, meV.
    half_width : float
        Half-width of the excitation window, meV.
    t_eval : float
        Time after excitation at which the occupation is evaluated, ps.

    Returns
    -------
    d : float
        Diffusion coefficient in cm^2/s.

    Raises
    ------
    ValueError
        If no state lies in the excitation window or the dephasing iteration does not converge.
    """
    return d
```

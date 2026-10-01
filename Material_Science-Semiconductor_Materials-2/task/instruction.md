# Material_Science-Semiconductor_Materials-2

## Background

Atomically thin semiconductors combine a high two-dimensional density of states with gate-tunable carrier populations, so their conductivity and thermopower are sensitive to the microscopic relaxation channels. When the conduction-band minima of a hexagonal crystal lie on the Γ-K lines rather than at the zone center or the zone corners, symmetry leaves more than one acoustic deformation-potential component in each valley, and the longitudinal (LA) and transverse (TA) couplings of a valley depend on the phonon direction relative to the valley axis instead of reducing to one scalar constant.

The surrounding dielectric has two competing effects. Its static permittivity modifies image charges and the finite-temperature carrier response, and so changes both charged-defect scattering and the free-carrier screening inside the sheet. A polar dielectric also supplies surface-optical modes that exchange finite energy with carriers through a long-ranged field. Their absorption and emission processes sample different final-state wave vectors and switch on at the mode energies, so they must be screened on their own inelastic momentum-transfer grids.

Real samples rarely come with a known carrier density or defect density. Resistance and Hall measurements taken at several temperatures on a gated sample constrain both quantities, but only through a transport model that contains every relaxation channel present in the measured stack, and a channel whose strength is misjudged is absorbed into the inferred defect density.

The balance between phonon and charged-defect scattering is summarized by an environment-dependent critical impurity density, at which the separately limited conductivities are equal. Comparing a sample's defect density with that critical density in the encapsulation chosen for its thermoelectric power factor tells whether cleaner material or further dielectric engineering is the better route to improvement.

## Problem

## Setup

A gated n-type hexagonal monolayer Hall bar lies between two identical SiO2 layers; its conduction band has six equivalent valleys, one on each \(\Gamma\)-K line and each holding one spin state, the gate holds one unknown sheet density \(n_s\) at every temperature, the only static defects are singly charged midplane impurities of unknown areal density \(N_I\), and the same sheet is subsequently placed between each of four identical dielectric pairs.

## Inputs

Constants are \(e=1.602176634\times10^{-19}\) C, \(\hbar=1.054571817\times10^{-34}\) J s, \(k_B=1.380649\times10^{-23}\) J K\(^{-1}\), \(m_e=9.1093837015\times10^{-31}\) kg, and \(\epsilon_0=8.8541878128\times10^{-12}\) F m\(^{-1}\).

| layer quantity | value |
|---|:---:|
| thickness \(a\), permittivity \(\epsilon_s\) | 6.60 \(\mathring{\mathrm A}\), 7.60 |
| Kane band of each valley \(m^*\), \(\alpha\) | \(0.405m_e\), 0.700 eV\(^{-1}\) |
| valley deformation potentials \(\Xi_d+\Xi_u\), \(\Xi_p\) | 7.20 eV, 2.63 eV |
| elastic constants \(c_{11}\), \(c_{12}\) | 132.7 N m\(^{-1}\), 33.0 N m\(^{-1}\) |

| symmetric stack | \(\kappa_0\) | \(\kappa_\infty\) | \(\Omega_1\) (meV) | \(\Omega_2\) (meV) |
|---|:---:|:---:|:---:|:---:|
| SiO2 / layer / SiO2 | 3.90 | 2.50 | 55.60 | 138.10 |
| AlN / layer / AlN | 9.14 | 4.80 | 81.40 | 88.50 |
| Al2O3 / layer / Al2O3 | 12.53 | 3.20 | 48.18 | 71.41 |
| HfO2 / layer / HfO2 | 23.00 | 5.03 | 12.40 | 48.35 |

The SiO2 Hall bar gives the sheet resistance \(\rho_\square\) and the magnitude of the weak-field Hall slope \(|d\rho_{xy}/dB|\) below, each with a 0.3% relative standard uncertainty.

| \(T\) (K) | \(\rho_\square\) (\(\Omega\)) | \(|d\rho_{xy}/dB|\) (\(\Omega\) T\(^{-1}\)) |
|:---:|:---:|:---:|
| 200 | 753.6 | 66.16 |
| 250 | 1334 | 70.96 |
| 300 | 2221 | 74.19 |

## Physical model

- **Band and transport:** use the source articles' Kane-band density of states, group velocity, weak-field Boltzmann conductivity tensor, transport moments, Seebeck coefficient, finite-temperature polarizability, finite-thickness image-charge form factors, screened charged-impurity rate, and two-interface surface-optical-phonon rate; integrate the density over \([0,0.8]\) eV to fix \(\mu\), the polarizability over \([0,1.0]\) eV without renormalizing the truncated thermal weight, and transport over \([10^{-6},0.8]\) eV split at the two mode energies, converting electronvolts to joules inside SI expressions.

- **Acoustic phonons:** in each valley frame take \(x\) along that valley's \(\Gamma\)-K direction and use the primary article's generalized tensor obtained from \(\Xi_d+\Xi_u\) and \(\Xi_p\), with \(c_{66}=(c_{11}-c_{12})/2\), six-valley static RPA screening, and intravalley scattering only; average the rate uniformly over incident electron direction \(\psi\), using \(q=2k\sin(\phi/2)\) and \(\theta=\psi+\phi/2+\pi/2\) for counterclockwise scattering angle \(\phi\), rather than selecting one electron direction.

- **Electrostatic and rate conventions:** use \(p(z)=2\cos^2(\pi z/a)/a\) on \(|z|\le a/2\), \(\epsilon_e=\kappa_0\), and the dielectric article's image-charge and RPA expressions; the finite-temperature polarizability uses the constant band-edge prefactor \(6m^*/(2\pi\hbar^2)\) with no \((1+2\alpha E)\) factor, elastic acoustic and impurity rates use the initial-state Kane factor \(m^*(1+2\alpha E)\), and the surface-optical rate uses the constant band-edge final-state prefactor \(m^*\), no final-state Pauli factor, screening reevaluated at each inelastic momentum transfer, \(\Theta(x)=1\) only for \(x>0\), and the two identical interfaces counted once.

- **Reported conductivities:** \(\sigma_{ph}\) uses acoustic plus surface-optical scattering, \(\sigma_I\) uses impurity scattering alone, and \(N_{cr}\) is the impurity density for which \(\sigma_I=\sigma_{ph}\), determined from full energy-resolved transport rather than rates only at \(E=\mu\).

## Task

Choose \(n_s\in[1.0,3.0]\times10^{13}\) cm\(^{-2}\) and \(N_I\in[1.0\times10^{11},1.0\times10^{13}]\) cm\(^{-2}\) to minimize the unweighted sum over all six measurements of squared natural-log model/data ratios; with that fit, define \(P_i=10^3S_i^2\sigma_i\) at 300 K and report the **dimensionless ratio \(N_I/N_{cr}\)** for the stack with largest \(P_i\).

Briefly include, with computed values to four significant figures:

- the constants \(A_s\) and \(B_u\), in eV\(^2\)/(N m\(^{-1}\)), for which your \(\langle\tilde D_{LA}^2/c_{11}+\tilde D_{TA}^2/c_{66}\rangle_\psi\) equals \(A_s/\varepsilon_{2D}^2+B_u\), and the fitted \(n_s\) and \(N_I\)
- the maximizing stack with its \(P\), \(\sigma\), \(S\) and \(N_{cr}\), and its \(\Gamma_{ac}\), \(\Gamma_I\) and \(\Gamma_{SO}\) evaluated directly at \(E=\mu\), not at a quadrature node, and 300 K
- the primary article's predicted maximal room-temperature power factor of monolayer MoS2 with its comparison to commercial thermoelectrics, and the speed-ups it reports for its two methods
- the dielectric article's compromise dielectrics with their reason, and its empirical relation between critical impurity density and environment permittivity

## Numerical conventions

The reference settings are:

- Gauss-Legendre quadrature for every density, polarizability, angular, and transport integral; increase the orders until every requested four-significant-figure checkpoint is stable and the final ratio changes by less than \(10^{-4}\)
- the density bracket \([-0.5,0.8]\) eV with \(10^{-30}\) J absolute and \(10^{-14}\) relative tolerance
- a bounded trust-region least-squares fit in \((\ln n_s,\ln N_I)\) with a three-point difference Jacobian, started at the midpoint of the logarithmic bounds and converged to \(10^{-12}\) relative tolerance
- the first stack in table order if the largest \(P_i\) is shared, and no additional cutoff, broadening, screening factor, final-state occupation factor, or intervalley process

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

solve kane state

Goal
----
Determine finite-temperature band occupation for the compact MoS2 model.

```python
def solve_kane_state(temperature_k: float, density_cm2: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, energy_max_ev: float, quadrature_order: int) -> 'np.ndarray':
    """Return the chemical potential and Fermi wave vector.

    Parameters
    ----------
    temperature_k : float
        Temperature in kelvin.
    density_cm2 : float
        Electron sheet density in cm^-2.
    mass_ratio : float
        Band-edge effective mass divided by the electron mass.
    alpha_ev_inv : float
        Nonparabolicity in eV^-1.
    degeneracy : float
        Combined spin and valley degeneracy.
    energy_max_ev : float
        Upper energy bound in eV.
    quadrature_order : int
        Gauss-Legendre order.

    Returns
    -------
    numpy.ndarray
        Two entries: chemical potential in eV and k_F in m^-1.

    Raises
    ------
    ValueError
        For nonfinite or nonpositive temperature, density, mass ratio,
        degeneracy, or energy maximum; a negative or nonfinite alpha_ev_inv;
        a quadrature order that is not a positive integer (booleans
        included); or a density whose chemical potential lies outside the
        fixed root bracket [-0.5 eV, energy_max_ev] (too dilute a sheet for
        the temperature, or more carriers than the energy interval holds).
    """
    return None
```

### Step 2

resolve acoustic tensor

Goal
----
Contract an in-plane deformation-potential tensor into LA and TA band-edge shifts.

```python
def resolve_acoustic_tensor(phonon_angle_rad: 'np.ndarray', dp_tensor_ev: 'Sequence[float]') -> 'np.ndarray':
    """Return the longitudinal and transverse band-edge shifts per unit strain.

    For a phonon wave vector at angle theta, measured counterclockwise from
    the x axis of the supplied tensor, the in-plane longitudinal and
    transverse acoustic polarizations of a two-dimensional hexagonal crystal
    are parallel and perpendicular to the wave vector. With s = sin(theta)
    and c = cos(theta), the contracted deformation potentials are

        D_LA = Xi_xx c^2 + Xi_yy s^2 + 2 Xi_xy s c,
        D_TA = (Xi_yy - Xi_xx) s c + Xi_xy (c^2 - s^2).

    Parameters
    ----------
    phonon_angle_rad : numpy.ndarray
        Phonon wave-vector directions theta in radians, measured
        counterclockwise from the tensor's x axis; any nonempty array of
        finite values, which is flattened in C order.
    dp_tensor_ev : sequence of float
        The three in-plane tensor components (Xi_xx, Xi_yy, Xi_xy) in eV.

    Returns
    -------
    numpy.ndarray
        Shape (2, n_angle), where n_angle is the number of supplied angles.
        Row 0 is D_LA and row 1 is D_TA at each direction in the flattened
        order, both in eV.

    Raises
    ------
    ValueError
        For an angle array that is empty or holds a value that is not a
        finite number, or a tensor that is not three finite numbers.
    """
    return None
```

### Step 3

evaluate polarizability

Goal
----
Evaluate the thermally broadened two-dimensional polarizability.

```python
def evaluate_polarizability(q_over_kf: 'np.ndarray', k_fermi_m_inv: float, temperature_k: float, chemical_potential_ev: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, energy_max_ev: float, quadrature_order: int) -> 'np.ndarray':
    """Return finite-temperature static polarizability.

    The zero-temperature response uses the constant band-edge prefactor
    g*m/(2*pi*hbar^2), without a Kane (1 + 2*alpha*E) multiplier.

    Parameters
    ----------
    q_over_kf : numpy.ndarray
        Positive scattering wave vectors divided by k_F.
    k_fermi_m_inv : float
        Fermi wave vector in m^-1 from the band-occupation stage.
    temperature_k : float
        Temperature in kelvin.
    chemical_potential_ev : float
        Chemical potential relative to the band edge in eV.
    mass_ratio : float
        Effective mass divided by the electron mass.
    alpha_ev_inv : float
        Nonparabolicity in eV^-1.
    degeneracy : float
        Combined spin and valley degeneracy.
    energy_max_ev : float
        Upper thermal-average energy in eV.
    quadrature_order : int
        Gauss-Legendre order for the thermal average.

    Returns
    -------
    numpy.ndarray
        Polarizability in J^-1 m^-2 with the input shape.

    Raises
    ------
    ValueError
        For empty, nonfinite, or nonpositive q_over_kf; nonfinite or
        nonpositive k_fermi_m_inv, temperature, mass ratio, degeneracy, or
        energy maximum; a nonfinite chemical potential; a negative or
        nonfinite alpha_ev_inv; or a quadrature order that is not a
        positive integer (booleans included).
    """
    return None
```

### Step 4

screen dielectric kernel

Goal
----
Apply closed-form image-charge form factors and free-carrier screening.

```python
def screen_dielectric_kernel(q_m_inv: 'np.ndarray', polarizability_j_inv_m2: 'np.ndarray', thickness_angstrom: float, epsilon_layer: float, epsilon_environment: float) -> 'np.ndarray':
    """Return the screened impurity potential and dielectric response.

    With x = q a, the ground-state profile p(z) = 2 cos^2(pi z / a) / a on
    |z| <= a/2 gives the closed forms
    A(x) = 2 (1 - exp(-x/2)) / x + 2 x (1 + exp(-x/2)) / (x^2 + 4 pi^2),
    C(x) = 8 pi^2 sinh(x/2) / (x (x^2 + 4 pi^2)), and
    Phi(x) = [3 x + (8 pi^2 x + 32 pi^4 g(x)) / (x^2 + 4 pi^2)] / (x^2 + 4 pi^2)
    with g(x) = (x - 1 + exp(-x)) / x^2. With gamma = (eps_layer - eps_env) /
    (eps_layer + eps_env) and r = gamma exp(-x), the impurity and carrier form
    factors are F_I = A + 2 r C / (1 - r) and F_ee = Phi + 2 r C^2 / (1 - r).

    Parameters
    ----------
    q_m_inv : numpy.ndarray
        Positive two-dimensional wave-vector transfers in m^-1.
    polarizability_j_inv_m2 : numpy.ndarray
        Static polarizability with the same shape as q_m_inv.
    thickness_angstrom : float
        Layer thickness a in angstrom.
    epsilon_layer, epsilon_environment : float
        Relative permittivities of the layer and symmetric environment.

    Returns
    -------
    numpy.ndarray
        First row: screened potential U = e^2 F_I / (2 eps_0 eps_layer q
        eps_2D) in J m^2; second row: dimensionless dielectric response
        eps_2D = 1 + e^2 Pi F_ee / (2 eps_0 eps_layer q). Each row retains
        the input array shape.

    Raises
    ------
    ValueError
        For empty or mismatched q and polarizability arrays; nonfinite or
        nonpositive q; nonfinite or negative polarizability; nonfinite or
        nonpositive thickness or permittivities; or any q with q * a above
        700, where the hyperbolic image-charge factors overflow double
        precision.
    """
    return None
```

### Step 5

compute acoustic rates

Goal
----
Compute incidence-averaged, partially screened acoustic momentum relaxation.

```python
def compute_acoustic_rates(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', incidence_weights: 'np.ndarray', band_shift_ev: 'np.ndarray', dielectric_response: 'np.ndarray', temperature_k: float, mass_ratio: float, alpha_ev_inv: float, c11_n_m: float, c12_n_m: float) -> 'np.ndarray':
    """Return incidence-averaged acoustic momentum-relaxation rates.

    The four rows of band_shift_ev hold, for every scattering-angle node i
    and incidence node j, the LA shift that is divided by the dielectric
    response (row 0), the LA shift that is not (row 1), the TA shift that is
    divided by the dielectric response (row 2), and the TA shift that is not
    (row 3), all in eV. With eps = dielectric_response[n, i] at energy node
    n, the event couplings are

        D_LA = row0 / eps + row1,    D_TA = row2 / eps + row3,

    and the rate is

        Gamma(E_n) = m* k_B T (1 + 2 alpha E_n) / (2 pi hbar^3)
                     * sum_i w_i (1 - cos phi_i)
                       * sum_j u_j / (2 pi) * [D_LA^2 / c11 + D_TA^2 / c66],

    with c66 = (c11 - c12) / 2, w_i = angle_weights, u_j =
    incidence_weights, and the shifts converted from eV to J.

    Parameters
    ----------
    energy_ev : numpy.ndarray
        Nonempty one-dimensional array of nonnegative carrier energies in eV.
    scattering_angle_rad, angle_weights : numpy.ndarray
        One-dimensional scattering-angle nodes phi_i and positive weights of
        the same length.
    incidence_weights : numpy.ndarray
        Nonempty one-dimensional positive weights u_j of the average over the
        incident direction; their sum is used as given.
    band_shift_ev : numpy.ndarray
        Shape (4, n_angle, n_incidence), the four shift rows defined above.
    dielectric_response : numpy.ndarray
        Positive dimensionless response with shape (n_energy, n_angle).
    temperature_k, mass_ratio : float
        Positive temperature in kelvin and effective mass divided by the
        electron mass.
    alpha_ev_inv : float
        Nonnegative nonparabolicity in eV^-1.
    c11_n_m, c12_n_m : float
        Two-dimensional elastic constants in N m^-1 with c11 > c12 >= 0.

    Returns
    -------
    numpy.ndarray
        Acoustic momentum-relaxation rates in s^-1, one per energy node.

    Raises
    ------
    ValueError
        For an empty or non-one-dimensional energy or incidence-weight
        array, a non-one-dimensional angle array, angle weights not matching
        the angles, a shift array whose shape is not
        (4, n_angle, n_incidence), a dielectric shape other than
        (n_energy, n_angle), nonfinite entries, negative energies,
        nonpositive angle or incidence weights or dielectric response,
        nonpositive or nonfinite temperature or mass, a negative or
        nonfinite alpha_ev_inv, or nonfinite elastic constants or elastic
        constants violating c11 > c12 >= 0.
    """
    return None
```

### Step 6

integrate impurity rates

Goal
----
Integrate the screened Coulomb rate coefficient over scattering angle.

```python
def integrate_impurity_rate_coefficient(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', screened_potential_j_m2: 'np.ndarray', mass_ratio: float, alpha_ev_inv: float) -> 'np.ndarray':
    """Return the charged-impurity rate per sheet density in cm^-2.

    The elastic prefactor uses the initial-state Kane density-of-states
    multiplier (1 + 2*alpha*E).

    Parameters
    ----------
    energy_ev : numpy.ndarray
        Carrier energies in eV.
    scattering_angle_rad : numpy.ndarray
        Scattering angles in radians.
    angle_weights : numpy.ndarray
        Quadrature weights over zero to two pi.
    screened_potential_j_m2 : numpy.ndarray
        Screened potential with shape (energy, angle), in J m^2.
    mass_ratio : float
        Effective mass divided by the electron mass.
    alpha_ev_inv : float
        Nonparabolicity in eV^-1.

    Returns
    -------
    numpy.ndarray
        Momentum-relaxation-rate coefficients in s^-1 cm^2.

    Raises
    ------
    ValueError
        For non-one-dimensional energy or angle arrays, weights not matching
        the angles, a potential shape other than (n_energy, n_angle),
        nonfinite entries, negative energies, nonpositive weights or
        potential values, a nonpositive or nonfinite mass ratio, or a
        negative or nonfinite alpha_ev_inv.
    """
    return None
```

### Step 7

compute surface optical rates

Goal
----
Compute screened surface-optical-phonon momentum relaxation.

```python
def compute_surface_optical_rates(energy_ev: 'np.ndarray', scattering_angle_rad: 'np.ndarray', angle_weights: 'np.ndarray', mode_energy_mev: 'np.ndarray', absorption_q_m_inv: 'np.ndarray', emission_q_m_inv: 'np.ndarray', absorption_k_ratio: 'np.ndarray', emission_k_ratio: 'np.ndarray', absorption_dielectric: 'np.ndarray', emission_dielectric: 'np.ndarray', temperature_k: float, mass_ratio: float, thickness_angstrom: float, epsilon_static: float, epsilon_high_frequency: float, interface_count: int) -> 'np.ndarray':
    """Return the summed absorption and emission rates in s^-1.

    Use a constant band-edge final-state mass prefactor, without a Kane
    multiplier or a final-state Pauli occupation factor.

    Parameters
    ----------
    energy_ev : numpy.ndarray
        Positive initial carrier energies in eV.
    scattering_angle_rad, angle_weights : numpy.ndarray
        Angular quadrature nodes and weights over zero to two pi.
    mode_energy_mev : numpy.ndarray
        Surface-optical mode energies in meV.
    absorption_q_m_inv, emission_q_m_inv : numpy.ndarray
        Momentum transfers with shape (mode, energy, angle).
    absorption_k_ratio, emission_k_ratio : numpy.ndarray
        Final-to-initial wave-vector ratios with shape (mode, energy).
    absorption_dielectric, emission_dielectric : numpy.ndarray
        Static carrier dielectric responses on the two inelastic grids.
    temperature_k, mass_ratio, thickness_angstrom : float
        Temperature, effective-mass ratio, and layer thickness.
    epsilon_static, epsilon_high_frequency : float
        Static and high-frequency permittivities of the dielectric.
    interface_count : int
        Number of identical dielectric interfaces contributing modes.

    Returns
    -------
    numpy.ndarray
        Total surface-optical momentum-relaxation rate in s^-1.

    Raises
    ------
    ValueError
        For empty, non-one-dimensional, or nonpositive energies; empty or
        non-one-dimensional angles, angles outside [0, 2*pi], or weights
        that are not positive and matching; empty, non-one-dimensional, or
        nonpositive mode energies; inconsistent grid shapes; nonfinite entries;
        nonpositive momentum transfers or dielectric responses; negative
        wave-vector ratios; nonpositive or nonfinite material inputs;
        epsilon_static below epsilon_high_frequency; an interface_count
        that is not a positive integer (booleans included); a momentum
        transfer with a * q above 700, where sinh^2(aq/2) overflows double
        precision; or supplied arrays for which the summed rate is negative
        at some energy (the rate is returned without clipping, so this can
        only come from angular factors 1 - (k'/k) cos(phi) that are not
        outweighed).
    """
    return None
```

### Step 8

compute transport moments

Goal
----
Convert energy-resolved relaxation rates into thermoelectric and Hall responses.

```python
def compute_transport_moments(energy_ev: 'np.ndarray', energy_weights_ev: 'np.ndarray', chemical_potential_ev: float, acoustic_rate_s_inv: 'np.ndarray', impurity_rate_s_inv: 'np.ndarray', surface_optical_rate_s_inv: 'np.ndarray', temperature_k: float, mass_ratio: float, alpha_ev_inv: float, degeneracy: float, thickness_angstrom: float) -> 'np.ndarray':
    """Return conductivity, Seebeck coefficient, power factor, and Hall response.

    Parameters
    ----------
    energy_ev, energy_weights_ev : numpy.ndarray
        One-dimensional energy nodes and positive quadrature weights in eV.
        Energies must be nonnegative.
    chemical_potential_ev : float
        Chemical potential relative to the band edge in eV.
    acoustic_rate_s_inv, impurity_rate_s_inv, surface_optical_rate_s_inv : numpy.ndarray
        Nonnegative independent momentum-relaxation rates in s^-1 on the
        energy nodes. Their sum must be positive at every node.
    temperature_k, mass_ratio, degeneracy : float
        Positive temperature in kelvin, band-edge mass ratio, and combined
        spin and valley degeneracy.
    alpha_ev_inv : float
        Nonnegative nonparabolicity in eV^-1.
    thickness_angstrom : float
        Positive thickness used to convert sheet conductance to S m^-1.

    Returns
    -------
    numpy.ndarray
        Four entries: conductivity in S m^-1, Seebeck coefficient in
        microvolt K^-1, power factor in mW m^-1 K^-2, and |sigma_xy|/B in
        microsiemens T^-1, the magnitude of the sheet Hall conductivity per
        unit perpendicular magnetic field to first order in B, obtained from
        the same isotropic relaxation-time Boltzmann equation for this band
        and these rates with the Lorentz force included.

    Raises
    ------
    ValueError
        For mismatched or non-one-dimensional arrays, nonfinite values,
        negative energies or rates, nonpositive weights or total rate,
        nonpositive temperature, mass, degeneracy, or thickness, a negative
        alpha_ev_inv, or a grid on which the conductivity moment L_0 is zero
        (an empty grid, every node at the band edge, or a thermal window that
        underflows at every node).
    """
    return None
```

### Step 9

infer sheet state

Goal
----
Infer the gate-held sheet density and midplane defect density from Hall-bar data.

```python
def infer_sheet_state(screened_dp_tensors_ev: 'Sequence[Sequence[float]]', unscreened_dp_tensors_ev: 'Sequence[Sequence[float]]', temperatures_k: 'Sequence[float]'=(200.0, 250.0, 300.0), sheet_resistance_ohm: 'Sequence[float]'=(753.6, 1334.0, 2221.0), hall_slope_ohm_per_tesla: 'Sequence[float]'=(66.16, 70.96, 74.19), epsilon_static: 'float'=3.9, epsilon_high_frequency: 'float'=2.5, mode_energy_mev: 'Sequence[float]'=(55.6, 138.1), density_bounds_cm2: 'tuple[float, float]'=(10000000000000.0, 30000000000000.0), impurity_bounds_cm2: 'tuple[float, float]'=(100000000000.0, 10000000000000.0), mass_ratio: 'float'=0.405, alpha_ev_inv: 'float'=0.7, degeneracy: 'float'=6.0, c11_n_m: 'float'=132.7, c12_n_m: 'float'=33.0, thickness_angstrom: 'float'=6.6, epsilon_layer: 'float'=7.6, density_order: 'int'=200, transport_order_per_segment: 'int'=64, angle_order: 'int'=64, polarizability_order: 'int'=100, transport_energy_max_ev: 'float'=0.8, polarizability_energy_max_ev: 'float'=1.0, interface_count: 'int'=2) -> 'np.ndarray':
    """Return the least-squares sheet density and charged-defect density.

    The two tensor-pair arguments fix the acoustic stage. Row 0 of each pair
    is an in-plane tensor (Xi_xx, Xi_yy, Xi_xy) in eV that is contracted by
    the acoustic-tensor stage and whose LA row is used; row 1 is a tensor
    whose TA row is used. The screened pair supplies the shifts that are
    divided by the dielectric response and the unscreened pair supplies the
    shifts that are not. Both scattering angles phi_i and incident
    directions psi_j use the Gauss-Legendre rule of order angle_order on
    [0, 2 pi], and the tensors are contracted at the phonon directions
    theta_ij = psi_j + phi_i/2 + pi/2, measured counterclockwise from the
    tensor x axis.

    Parameters
    ----------
    screened_dp_tensors_ev, unscreened_dp_tensors_ev : array_like
        Shape (2, 3) arrays of finite tensor components in eV: row 0 for the
        LA shift and row 1 for the TA shift, as described above. These
        arguments have no defaults.
    temperatures_k : sequence of float
        At least two distinct positive temperatures in kelvin.
    sheet_resistance_ohm, hall_slope_ohm_per_tesla : sequence of float
        Positive sheet resistance in ohm and positive magnitude of the
        weak-field Hall slope d(rho_xy)/dB in ohm T^-1, one value per
        temperature and in the same order.
    epsilon_static, epsilon_high_frequency : float
        Static and high-frequency permittivities of the identical dielectric
        on both sides of the sheet, with epsilon_static >= epsilon_high_frequency > 0.
    mode_energy_mev : sequence of float
        Distinct surface-optical mode energies of that dielectric in meV,
        each above 1e-3 meV (the lower end of the transport grid) and below
        transport_energy_max_ev.
    density_bounds_cm2, impurity_bounds_cm2 : pair of float
        Finite search intervals (lower, upper) with 0 < lower < upper for the
        sheet density and the midplane impurity density, both in cm^-2.
    mass_ratio, alpha_ev_inv, degeneracy : float
        Kane band parameters: positive mass ratio, nonnegative
        nonparabolicity in eV^-1, positive degeneracy.
    c11_n_m, c12_n_m : float
        Two-dimensional elastic constants in N m^-1 with c11 > c12 >= 0.
    thickness_angstrom, epsilon_layer : float
        Positive layer thickness in angstrom and layer permittivity.
    density_order, transport_order_per_segment, angle_order, polarizability_order : int
        Positive integer quadrature orders (booleans are rejected).
    transport_energy_max_ev, polarizability_energy_max_ev : float
        Upper energies in eV of the transport grid (above 1e-6 eV) and of
        the thermal polarizability average.
    interface_count : int
        Positive integer number of identical dielectric interfaces.

    Returns
    -------
    numpy.ndarray
        Two entries: sheet electron density and midplane charged-impurity
        density, both in cm^-2.

    Notes
    -----
    The fit uses scipy.optimize.least_squares with method="trf",
    jac="3-point", xtol=ftol=gtol=1e-12, max_nfev=200, started at the
    midpoint of the logarithmic bounds. Implementations that follow this
    call agree only to about 1e-9 relative, because the solver's
    termination reacts to rounding in the residuals, so the tests compare
    the natural logarithms of the two densities rounded to six decimals.

    Raises
    ------
    ValueError
        For fewer than two temperatures, repeated, nonpositive, or nonfinite
        temperatures, or data that are not positive, finite, and matched one
        to one with the temperatures; mode energies that are empty, not positive, not
        distinct, at or below 1e-3 meV, or at or above
        transport_energy_max_ev; search intervals that are not finite pairs
        with 0 < lower < upper; a tensor-pair argument that is not a (2, 3)
        array of finite numbers; nonfinite material parameters or energy
        maxima, permittivities violating
        epsilon_static >= epsilon_high_frequency > 0, nonpositive mass,
        degeneracy, thickness, or layer permittivity, a negative
        alpha_ev_inv, or elastic constants violating c11 > c12 >= 0;
        nonpositive energy maxima or a transport maximum at
        or below 1e-6 eV; orders or interface_count that are not positive
        integers (booleans included); a trial state whose Fermi wave vector
        is zero at a tabulated temperature (chemical potential at or below
        the band edge); a least-squares solve that does not report
        convergence; or any error raised by stages 01 through 08 on the
        trial grids or by the least-squares solver (for example a negative
        summed surface-optical rate when angle_order is 2).
    """
    return None
```

### Step 10

select encapsulation defect ratio

Goal
----
Transfer the inferred sheet to each encapsulation and return the defect ratio of the best one.

```python
def select_encapsulation_defect_ratio(screened_dp_tensors_ev: 'Sequence[Sequence[float]]', unscreened_dp_tensors_ev: 'Sequence[Sequence[float]]', temperatures_k: 'Sequence[float]'=(200.0, 250.0, 300.0), sheet_resistance_ohm: 'Sequence[float]'=(753.6, 1334.0, 2221.0), hall_slope_ohm_per_tesla: 'Sequence[float]'=(66.16, 70.96, 74.19), calibration_candidate_index: 'int'=0, target_temperature_k: 'float'=300.0, candidate_epsilon_static: 'Sequence[float]'=(3.9, 9.14, 12.53, 23.0), candidate_epsilon_high: 'Sequence[float]'=(2.5, 4.8, 3.2, 5.03), candidate_mode_energy_mev: 'Sequence[Sequence[float]]'=((55.6, 138.1), (81.4, 88.5), (48.18, 71.41), (12.4, 48.35)), density_bounds_cm2: 'tuple[float, float]'=(10000000000000.0, 30000000000000.0), impurity_bounds_cm2: 'tuple[float, float]'=(100000000000.0, 10000000000000.0), mass_ratio: 'float'=0.405, alpha_ev_inv: 'float'=0.7, degeneracy: 'float'=6.0, c11_n_m: 'float'=132.7, c12_n_m: 'float'=33.0, thickness_angstrom: 'float'=6.6, epsilon_layer: 'float'=7.6, density_order: 'int'=200, transport_order_per_segment: 'int'=64, angle_order: 'int'=64, polarizability_order: 'int'=100, transport_energy_max_ev: 'float'=0.8, polarizability_energy_max_ev: 'float'=1.0, interface_count: 'int'=2) -> float:
    """Return N_I / N_cr for the candidate stack with the largest power factor.

    Parameters
    ----------
    screened_dp_tensors_ev, unscreened_dp_tensors_ev : array_like
        Shape (2, 3) tensor pairs in eV with the layout and meaning used by
        the inference stage (row 0 for the LA shift, row 1 for the TA shift),
        passed unchanged to the inference stage and used in the same way,
        with the same Gauss-Legendre scattering and incidence nodes and the
        same phonon directions theta_ij = psi_j + phi_i/2 + pi/2, at the
        target temperature. These arguments have no defaults.
    temperatures_k, sheet_resistance_ohm, hall_slope_ohm_per_tesla : sequence of float
        Hall-bar data of the sheet in its calibration stack, in the layout
        required by the sheet-state inference stage.
    calibration_candidate_index : int
        Zero-based row of the candidate tables that describes the stack in
        which the Hall-bar data were taken (booleans are rejected).
    target_temperature_k : float
        Positive temperature in kelvin at which every candidate is compared.
    candidate_epsilon_static, candidate_epsilon_high : sequence of float
        One static and one high-frequency permittivity per candidate stack,
        with static >= high-frequency > 0; at least two candidates.
    candidate_mode_energy_mev : sequence of sequence of float
        Rectangular table with one row of distinct surface-optical mode
        energies in meV per candidate, each above 1e-3 meV and below
        transport_energy_max_ev.
    density_bounds_cm2, impurity_bounds_cm2 : pair of float
        Search intervals passed unchanged to the inference stage.
    mass_ratio, alpha_ev_inv, degeneracy, c11_n_m, c12_n_m,
    thickness_angstrom, epsilon_layer, density_order, transport_order_per_segment,
    angle_order, polarizability_order, transport_energy_max_ev,
    polarizability_energy_max_ev, interface_count :
        Material, quadrature, and interface parameters with the same meaning
        and domains as in the inference stage; the same values and the same
        grid construction are used for the calibration and for every
        candidate.

    Returns
    -------
    float
        With the inferred sheet density n_s and charged-impurity density N_I
        unchanged, every candidate is evaluated at target_temperature_k and
        the first candidate in table order with the largest
        P = 10^3 S^2 sigma is selected. For that candidate, sigma_ph is the
        conductivity with the acoustic and surface-optical rates only and
        sigma_I is the conductivity with the impurity rate at N_I only; the
        returned value is N_I / N_cr = sigma_ph / sigma_I, where N_cr is the
        impurity density at which the impurity-only conductivity equals
        sigma_ph.

    Notes
    -----
    The tests compare the natural logarithm of the returned value rounded
    to six decimals, because the inferred sheet state is reproducible only
    to about 1e-9 relative.

    Raises
    ------
    ValueError
        For a tensor-pair argument that is not a (2, 3) array of finite
        numbers; a calibration index that is not an integer (booleans
        included) or lies outside the candidate table; a nonpositive or
        nonfinite target temperature; fewer than two candidates, mismatched
        or ragged candidate tables, empty mode rows, or nonfinite entries;
        permittivities violating static >= high-frequency > 0; mode energies
        that are not positive, not distinct within a row, at or below
        1e-3 meV, or at or above transport_energy_max_ev, or a
        transport_energy_max_ev that is not a finite number; a zero Fermi
        wave vector of the inferred sheet at target_temperature_k (chemical
        potential at or below the band edge); or any error raised by the
        inference stage or by stages 01 through 08 at target_temperature_k
        (for example a zero conductivity moment when the target is so cold
        that the thermal window underflows at every transport node).
    """
    return None
```

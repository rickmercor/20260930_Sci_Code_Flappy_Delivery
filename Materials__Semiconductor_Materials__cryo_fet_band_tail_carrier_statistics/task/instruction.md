# cryo_fet_band_tail_carrier_statistics

## Problem

In a scaled silicon field-effect heterostructure cooled to millikelvin temperature, a single electron is held in a strained silicon quantum well between SiGe barriers, beneath a gate that defines a quantum dot. Biaxial tensile strain pushes four of the six conduction-band valleys of silicon far up in energy and leaves two, at $\pm k_0$ on the growth axis, nearly degenerate; only the confinement couples them, and the valley splitting it produces decides whether the two lowest electron states can serve as a spin doublet at all. Oscillating germanium profiles grown into the well are used to enlarge the splitting, but they vary on the scale of a few atomic monolayers, which is where envelope-function models resting on a slowly varying confinement potential lose their justification. Compute the valley splitting of the ground state of the heterostructure below in a multi-valley envelope-function description that makes no slowly varying potential approximation: the microscopic wave function is expanded in the lattice-periodic Bloch factors at the two valley minima, and each valley's envelope contains plane waves only from that valley's own sector of the Brillouin zone. The dot is laterally large compared with the lattice and its in-plane confinement is weak, so the problem reduces to the growth direction $z$.

The coordinate $z$ points up from the substrate towards the gate. The silicon well occupies $-h < z < 0$, so $z = 0$ is its upper interface, and $h$ is 75 monolayers of thickness $a_{\mathrm{Si}}/4$. The germanium fraction of the stack is

$$X(z) = X_b (1 - \Xi(z)) + \frac{X_w}{2} (1 + \cos(2 \pi z / \lambda)) \Xi(z),$$

$$\Xi(z) = \frac{1}{2} \left( \tanh\left(\frac{z + h}{\sigma_l}\right) - \tanh\left(\frac{z}{\sigma_u}\right) \right),$$

a smoothed well of barrier fraction $X_b$ carrying an in-well oscillation of amplitude $X_w$ and period $\lambda$. The conduction-band potential energy of the electron is

$$U(z) = \Delta E_c X(z) - e F z,$$

with $\Delta E_c$ the band offset per unit germanium fraction and $F$ the vertical gate field, which pulls the electron towards the upper interface. The SiGe above the well is a spacer that ends at an impenetrable gate dielectric at $z = 30$ nm, and the SiGe buffer below the well is thick enough that its extent does not matter.

Use the following configuration.

- lattice constants $a_{\mathrm{Si}} = 0.543$ nm and $a_{\mathrm{Ge}} = 0.565$ nm; the well is pseudomorphic on a relaxed buffer of germanium fraction $x = 0.3$, whose lattice constant is $a_{\mathrm{Si}} + b x (1 - x) + (a_{\mathrm{Ge}} - a_{\mathrm{Si}}) x^{2}$ with $b = 0.0200326$ nm
- silicon elastic constants $C_{11} = 167.5$ GPa and $C_{12} = 65.0$ GPa; in addition to the biaxial growth strain the well carries an in-plane shear strain $\varepsilon_{xy} = 1.0 \times 10^{-3}$, with $\varepsilon_{xz} = \varepsilon_{yz} = 0$
- valley minimum at $k_0 = 0.8394 \times 2 \pi / a_{\mathrm{Si}}$, longitudinal effective mass $m_l = 0.909 m_0$
- $\Delta E_c = 0.5$ eV, $X_b = 0.3$, $\sigma_u = \sigma_l = 0.5$ nm
- $X_w = 0.15$ and $\lambda = 1.629$ nm, twelve monolayers
- $F = 3.0$ mV/nm
- intervalley Bloch-factor overlap sums $C_n = \sum_{G, G'} c^{*}_{+}(G) c_{-}(G') \delta_{G - G', n G_0}$, where $c_{\pm}(G)$ are the coefficients of $\exp(\mathrm{i} G \cdot r)$ in the plane-wave expansions of the Bloch factors at the two valley minima and $G_0$ is the reciprocal-lattice vector of the strained crystal, with positive growth-direction component, that maps one valley onto the zone image of the other; at the stated strain $C_{-4} = 1.84 \times 10^{-3}$, $C_{-3} = 5.99 \times 10^{-4}$, $C_{-2} = -2.44 \times 10^{-2}$, $C_{-1} = -3.18 \times 10^{-2}$, $C_0 = -0.221$, $C_1 = -4.02 \times 10^{-4}$, $C_2 = 1.58 \times 10^{-3}$, $C_3 = 1.04 \times 10^{-5}$ and $C_4 = 3.35 \times 10^{-5}$, and orders beyond $|n| = 4$ are negligible

Report the valley splitting in meV to at least four decimal places. The answer is graded within $0.012$ meV.

Report also, inside the reasoning: the growth-direction strain $\varepsilon_{zz}$; the position of the Brillouin-zone boundary along the growth direction, and twice the distance $k_1$ from the valley minimum to that boundary, both in units of $2 \pi / a_{\mathrm{Si}}$; the lowest and the first excited growth-direction levels of a single valley, before the two valleys are coupled, in meV, measured in the zero of $U$ with the band-edge energy and the in-plane zero-point energy left out; the expectation value of $z$ in the lowest of those states, in nm; and which order $n$ of the overlap sums carries the intervalley coupling, with the reason the splitting at this period would nearly vanish in a crystal without shear strain.

Output Format Requirements:
Emit <final_answer> immediately, then <reasoning>. Do not write a long derivation before the tags.
You must emit exactly one finite decimal inside <final_answer>...</final_answer>, even if the value is approximate or you are unsure.
Rules:
- The tags are required. Do not omit them or leave them empty.
- The tagged value must be a finite decimal (examples: 0.4847, 0.22, 1.05). Not NaN, not Inf, not a fraction string, not a vector, not prose.
- Put only that one number between the tags. No units, no words, no extra lines.
Keep <reasoning> short (a few hundred words). Show only the few scalars that determine the final number.
Do not paste the input matrices, full coefficient vectors, per-iteration paths, or per-candidate tables.

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

step_01_strained_valley_geometry

Goal
----
The quantum well of a Si/SiGe field-effect heterostructure is a thin silicon layer grown pseudomorphically on a relaxed SiGe buffer. The silicon adopts the in-plane lattice constant of the buffer, so it is stretched biaxially in the growth plane and, to keep its stress free along the growth direction, contracts along [001]. For a cubic crystal the in-plane strain is eps_par = (a_sub - a_Si) / a_Si and the tetragonal response along the growth direction is eps_zz = -2 (C12 / C11) eps_par, with C11 and C12 the elastic constants of silicon. The shear components that this growth geometry produces are zero; a small shear strain in the plane, when present, enters later only through the Bloch-factor data and not through the geometry computed here.

The lattice constant of the relaxed alloy buffer is not the linear interpolation between silicon and germanium. It carries a quadratic bowing correction, a_sub(x) = a_Si + b x (1 - x) + (a_Ge - a_Si) x^2, with x the germanium fraction of the buffer and b the bowing length.

Strain matters to the valley problem because it moves the Brillouin zone. The six conduction-band minima of silicon lie on the Delta lines near the X points; biaxial tensile strain lowers the two valleys along the growth direction, at plus and minus k0 on the z axis, far below the four in-plane valleys, so the low-energy physics of the well is a two-valley problem. Along the growth direction the reciprocal-lattice vector that joins the images of these two valleys is G0 = (I - eps)(b1 + b2), which for the strain tensor above has the single non-zero component G0z = (4 pi / a_Si)(1 - eps_zz). Its half, G0z / 2 = (2 pi / a_Si)(1 - eps_zz), is the zone boundary along [001] in the strained crystal, and the valley-specific sector of the +k0 valley along the growth direction is the open interval of wave numbers between zero and that boundary. The valley minimum itself is at k0, quoted as a fraction of 2 pi / a_Si of the unstrained crystal, and k1 = G0z / 2 - k0 is its distance to the strained zone boundary, the wave number whose double, 2 k1, characterises the long-period resonance of an oscillating germanium profile.

```python
def strained_valley_geometry(
    lattice_si: float,
    lattice_ge: float,
    bowing: float,
    substrate_ge_fraction: float,
    c11: float,
    c12: float,
    valley_fraction: float,
) -> dict:
    """Strain the silicon layer to its buffer and locate the valley sector along the growth direction.

    Parameters
    ----------
    lattice_si : float
        Lattice constant of silicon in nm, above zero.
    lattice_ge : float
        Lattice constant of germanium in nm, above zero.
    bowing : float
        Bowing length of the alloy lattice constant in nm.
    substrate_ge_fraction : float
        Germanium fraction of the relaxed buffer, between zero and one.
    c11 : float
        Elastic constant C11 of silicon in GPa, above zero.
    c12 : float
        Elastic constant C12 of silicon in GPa, above zero and below C11.
    valley_fraction : float
        Valley minimum k0 as a fraction of 2 pi / a_Si, inside the strained sector.

    Returns
    -------
    dict
        Under the keys eps_parallel, eps_zz, zone_vector, sector_edge, valley_wavenumber and edge_distance.

    Raises
    ------
    ValueError
        When any argument fails to be finite, when a lattice or elastic constant fails to be above zero, when C12 fails to be below C11, when the buffer fraction falls outside the closed interval from zero to one, or when the valley minimum fails to lie strictly inside the strained sector.
    """
    return
```

### Step 2

step_02_wiggle_well_supercell

Goal
----
The confinement the electron sees along the growth direction z is set by the germanium profile of the stack and by the vertical gate field. The silicon well occupies -h < z < 0, so z = 0 is its upper interface, and the SiGe barrier above it is a spacer that ends at an impenetrable gate dielectric at z = d. The nominal germanium fraction is X(z) = X_b (1 - Xi(z)) + X_mod(z), where Xi(z) = (1/2)[tanh((z + h) / sigma_l) - tanh(z / sigma_u)] is the smoothed well shape, equal to one deep inside the well and zero deep in either barrier, sigma_u and sigma_l are the widths of the upper and lower interfaces, and X_b is the barrier germanium fraction. A wiggle well adds germanium inside the well only, X_mod(z) = (X_w / 2)(1 + cos(q z)) Xi(z), with amplitude X_w and modulation wave number q = 2 pi / lambda for a stated oscillation period lambda. The well thickness h is given in silicon monolayers of a_Si / 4 each.

The conduction-band potential energy follows from a linear band offset in the germanium fraction and from the field, U(z) = Delta E_c X(z) - e F z, in electronvolts when z is in nm and the field F is given in volts per nm; with F above zero the energy falls with height, so the electron is pulled towards the upper interface and the spacer. This zero of energy, the conduction-band edge of the silicon well with the field potential vanishing at the upper interface, is the one the task states.

The equations of the next stages are solved on a periodic supercell whose Fourier grid is matched to the strained crystal. With G0z the growth-direction reciprocal-lattice vector of the strained crystal, N_FBZ grid wave numbers per Brillouin zone and N_BZ resolved zones, the cell length is L = 2 pi N_FBZ / G0z, the cell holds N = N_BZ N_FBZ points at spacing dz = L / N, and the grid wave numbers are k_m = 2 pi m / L for m = -N/2, ..., N/2 - 1 in the discrete Fourier transform order 0, 1, ..., N/2 - 1, -N/2, ..., -1. With N_FBZ even, both the zone centre and the zone boundary G0z / 2 fall exactly on grid points, which is what makes an open valley sector a well-defined set of grid wave numbers. The cell is placed to end at the dielectric: its points are z_j = d - L + j dz for j = 0, ..., N - 1. The periodic continuation then joins the low potential just below the dielectric to the high potential at the bottom of the buffer, a barrier that stands in for the hard wall and is invisible to a state confined in the well.

```python
def wiggle_well_supercell(
    zone_vector: float,
    n_bz: int,
    n_fbz: int,
    spacer: float,
    well_monolayers: float,
    lattice_si: float,
    sigma_upper: float,
    sigma_lower: float,
    x_barrier: float,
    x_wiggle: float,
    wiggle_period: float,
    band_offset: float,
    field: float,
) -> dict:
    """Lay out the periodic supercell and evaluate the wiggle-well germanium profile and confinement energy on it.

    Parameters
    ----------
    zone_vector : float
        Strained growth-direction reciprocal-lattice vector G0z in reciprocal nm, above zero.
    n_bz : int
        Number of resolved Brillouin zones, at least two.
    n_fbz : int
        Grid wave numbers per zone, even and at least four.
    spacer : float
        Distance from the upper interface to the dielectric in nm, above zero.
    well_monolayers : float
        Well thickness in monolayers of a_Si / 4, above zero.
    lattice_si : float
        Lattice constant of silicon in nm, above zero.
    sigma_upper : float
        Upper interface width in nm, above zero.
    sigma_lower : float
        Lower interface width in nm, above zero.
    x_barrier : float
        Barrier germanium fraction, between zero and one.
    x_wiggle : float
        Amplitude of the in-well oscillation, between zero and one.
    wiggle_period : float
        Oscillation period in nm, above zero.
    band_offset : float
        Conduction-band offset per unit germanium fraction in eV.
    field : float
        Vertical field in volts per nm.

    Returns
    -------
    dict
        Under the keys z, wavenumbers, ge_fraction, well_indicator, potential, spacing and length.

    Raises
    ------
    ValueError
        When any float argument fails to be finite, when a length, width, period or the zone vector fails to be above zero, when a germanium fraction falls outside the closed interval from zero to one, when n_bz fails to be an integer of at least two, when n_fbz fails to be an even integer of at least four, or when the cell fails to be longer than the well and the spacer together.
    """
    return
```

### Step 3

step_03_sector_projected_potential

Goal
----
In a multi-valley envelope-function description without a slowly varying potential approximation, the microscopic wave function is expanded in the Bloch factors at the valley minima, and each valley's envelope is required to contain plane waves only from that valley's own sector of the Brillouin zone. That restriction is what makes the decomposition unique, and it changes how the confinement potential acts. The potential no longer multiplies the envelope point by point: its matrix elements are taken between band-limited functions, so it becomes a non-local integral operator whose kernel is the valley-sector projector sandwiched around U. Reduced to the growth direction for a dot that is large compared with the lattice, the single-valley envelope problem for the +k0 valley is posed on the wave numbers K of the open sector S+ = (0, G0z / 2), and the potential couples two of them through its Fourier components at K - K' + n G0z for every integer n, weighted by the Bloch-factor overlap sums B_n of the valley.

The n = 0 term is the ordinary Fourier component of U between two sector wave numbers. The terms with n not zero carry the short-wavelength content of the potential, at wave numbers a whole reciprocal-lattice vector or more away, back into the sector; this back-folding exists because the Bloch factors are lattice periodic and not constant. B_0 = 1 by the normalisation of the Bloch factors, B_(-n) is the complex conjugate of B_n, and that symmetry is what keeps the operator Hermitian for a real potential.

On the supercell of the previous stage the sector is the set of grid wave numbers k_m = 2 pi m / L with m = 1, ..., N_FBZ / 2 - 1; the zone centre m = 0 and the boundary m = N_FBZ / 2 are excluded, so that the +k0 and -k0 sectors do not share a point. The Fourier component of the potential at a grid wave number kappa is defined on the cell as U_hat(kappa) = (1 / N) sum_j U(z_j) exp(-i kappa z_j), with the actual positions z_j, so that the operator refers to plane waves exp(i K z) in the physical coordinate. The operator then has elements

M(K, K') = sum_n B_n U_hat(K - K' + n G0z),

and every wave number it needs, K - K' + n G0z for the orders supplied, must lie strictly inside the resolved band |kappa| < N_BZ G0z / 2, since a component beyond it would be aliased onto a different one.

```python
def sector_projected_potential(
    z: np.ndarray,
    potential: np.ndarray,
    zone_vector: float,
    n_fbz: int,
    orders,
    backfold,
) -> dict:
    """Assemble the back-folded, valley-sector-projected potential operator on the +k0 sector.

    Parameters
    ----------
    z : np.ndarray
        Uniform supercell positions in nm, N of them, N a multiple of n_fbz.
    potential : np.ndarray
        Real confinement energy in eV on those positions.
    zone_vector : float
        Strained growth-direction reciprocal-lattice vector G0z in reciprocal nm, consistent with the cell.
    n_fbz : int
        Grid wave numbers per zone, even and at least four.
    orders : sequence of int
        Back-folding orders, containing zero and closed under negation.
    backfold : sequence of complex
        Coefficients B_n, with B_0 = 1 and B_(-n) the conjugate of B_n.

    Returns
    -------
    dict
        Under the keys indices, wavenumbers, matrix and hermitian_defect.

    Raises
    ------
    ValueError
        When the grid fails to be uniform or to be a whole number of zones consistent with the zone vector, when the potential fails to be real, finite and of matching length, when n_fbz fails to be an even integer of at least four, when the orders and coefficients differ in length, contain duplicates, lack zero, fail to be closed under negation, have B_0 other than one or B_(-n) other than the conjugate of B_n, or when a required wave number falls outside the resolved band.
    """
    return
```

### Step 4

step_04_nonlocal_valley_ground_state

Goal
----
With the projected potential operator of the previous stage in hand, the single-valley envelope problem of the +k0 valley is an ordinary Hermitian eigenproblem on the sector wave numbers. Write the band-limited envelope as F(z) = sum over K in S+ of c_K exp(i K z) / sqrt(L). The kinetic energy along the growth direction is diagonal in K, but it is measured from the valley minimum and not from the zone centre: the dispersion near the minimum is hbar^2 (K - k0)^2 / (2 m_l), with m_l the longitudinal effective mass. Constant energies, the band edge E_c and the zero-point energy of the weak in-plane confinement, only shift every eigenvalue together and are left out, so the eigenvalues are growth-direction energies in the zero of U. The Hamiltonian over the sector is therefore

H(K, K') = hbar^2 (K - k0)^2 / (2 m_l) delta(K, K') + M(K, K'),

with M the back-folded projected potential of the previous stage. Its lowest eigenvector is the ground state of the valley; the -k0 valley needs no separate solve, because its envelope on the mirror sector is the complex conjugate of this one under K to -K.

The valley-scale oscillation is carried by F, whose plane waves sit near k0. The coupling between the valleys is written in terms of the slowly varying envelope f(z) = exp(-i k0 z) F(z), which is what this stage returns, sampled on the supercell. Because the sector restriction is enforced on the coefficients, f contains no plane wave that belongs to the other valley. In general f is complex: the potential of an asymmetric stack is not even, so its Fourier components are complex and so is the eigenvector.

Two conventions make the output unique. The envelope is normalised in the continuum sense, dz sum_j |f(z_j)|^2 = 1, which on the supercell is the same as unit norm of the coefficient vector. Its global phase, which no physical quantity depends on, is fixed by making f real and positive at the grid point where |f| is largest. The stage also reports the first excited eigenvalue, the mean position <z> and the spread, the square root of <z^2> - <z>^2, all taken with the weight |f|^2 dz, so that a state that has left the well can be recognised.

```python
def nonlocal_valley_ground_state(
    z: np.ndarray,
    sector_wavenumbers: np.ndarray,
    potential_operator: np.ndarray,
    valley_wavenumber: float,
    longitudinal_mass: float,
) -> dict:
    """Solve the sector-restricted single-valley envelope problem and return its ground state.

    Parameters
    ----------
    z : np.ndarray
        Uniform supercell positions in nm.
    sector_wavenumbers : np.ndarray
        Increasing sector wave numbers in reciprocal nm, whole multiples of 2 pi / L.
    potential_operator : np.ndarray
        Hermitian projected potential in eV over those wave numbers.
    valley_wavenumber : float
        Valley minimum k0 in reciprocal nm, inside the range of the sector wave numbers.
    longitudinal_mass : float
        Longitudinal effective mass in free-electron masses, above zero.

    Returns
    -------
    dict
        Under the keys energy, excited_energy, envelope, mean_position, spread and sector_size.

    Raises
    ------
    ValueError
        When the grid fails to be uniform and finite, when the sector wave numbers fail to be finite, increasing, above zero and on the grid of the cell, when the operator fails to be a finite Hermitian matrix of matching size, when the valley wave number fails to lie strictly inside the range of the sector, or when the mass fails to be finite and above zero.
    """
    return
```

### Step 5

step_05_intervalley_coupling

Goal
----
The two valley ground states are degenerate at zeroth order, one the complex conjugate of the other, and the intervalley part of the Hamiltonian lifts the degeneracy. At first order in degenerate perturbation theory the problem closes on the two-dimensional subspace spanned by the two valley states, the effective Hamiltonian there is [[0, Delta], [Delta*, 0]] with a complex intervalley coupling Delta, the two eigenvalues are plus and minus |Delta|, and the valley splitting is E_VS = 2 |Delta|.

Written in terms of the slowly varying envelopes, the coupling is a sum over the harmonics that the Bloch factors of the two valleys can exchange. The valley-coupling selection rule G - G' = n G0 between the plane-wave coefficients of the two Bloch factors defines the overlap sums C_n, and each order n picks out the component of the confinement at wave number 2 k0 + n G0z. For the growth-direction problem,

Delta = sum_n C_n integral dz exp(-i (2 k0 + n G0z) z) conj(f_+(z)) U(z) f_-(z),

and since the envelope of the -k0 valley is the complex conjugate of that of the +k0 valley, f_- = conj(f_+), the integrand carries the square of conj(f_+) and not |f_+|^2. The integral is taken on the supercell as dz times the sum over grid points; for an envelope confined well inside the cell this is the trapezoidal rule of a function that vanishes at both ends of the cell and it converges faster than any power of dz.

The order n = 0 samples the confinement at 2 k0, the short-period resonance of an oscillating germanium profile. The order n = -1 samples it at 2 k0 - G0z = -2 k1, the long-period resonance at twice the distance from the valley to the strained zone boundary, and its coefficient is proportional to the in-plane shear strain, so that this channel is closed in an unsheared crystal. The stage reports the contribution of each order separately, so that the dominant channel of a given profile can be identified.

The same formula accepts any envelope and any weighting function in place of U, which later stages use: a real envelope from a conventional calculation, and a unit weight, which measures how the coupling responds to a constant added to the confinement.

```python
def intervalley_coupling(
    z: np.ndarray,
    envelope: np.ndarray,
    potential: np.ndarray,
    valley_wavenumber: float,
    zone_vector: float,
    orders,
    coefficients,
) -> dict:
    """Evaluate the first-order intervalley coupling of a valley envelope and the valley splitting it implies.

    Parameters
    ----------
    z : np.ndarray
        Uniform supercell positions in nm.
    envelope : np.ndarray
        Slowly varying envelope of the +k0 valley, normalised in the continuum sense.
    potential : np.ndarray
        Real weighting function in eV, usually the confinement energy.
    valley_wavenumber : float
        Valley minimum k0 in reciprocal nm, above zero.
    zone_vector : float
        Strained growth-direction reciprocal-lattice vector G0z in reciprocal nm, above zero.
    orders : sequence of int
        Orders n, without repetition.
    coefficients : sequence of complex
        Overlap sums C_n.

    Returns
    -------
    dict
        Under the keys delta_real, delta_imag, delta_abs, splitting, contribution_real, contribution_imag and dominant_order.

    Raises
    ------
    ValueError
        When the grid fails to be uniform and finite, when the envelope or weight fails to be finite or to match the grid, when the weight is complex, when the envelope fails to be normalised, when k0 or G0z fails to be finite and above zero, or when the orders and coefficients fail to be integers and finite values of equal length without repetition.
    """
    return
```

### Step 6

step_06_local_valley_ground_state

Goal
----
The conventional envelope-function model of a silicon quantum well drops the valley-sector projection. The truncated delta function of the sector is replaced by an ordinary delta function, the rapidly oscillating Bloch factors are averaged to one over the intravalley matrix elements, and the potential then acts on the envelope by plain multiplication. The single-valley problem becomes the familiar one-dimensional effective-mass equation

E f(z) = -(hbar^2 / (2 m_l)) f''(z) + U(z) f(z),

solved directly in position space, again with constant energies left out. Its Hamiltonian is real, so the ground-state envelope can be chosen real, and the same envelope serves both valleys. Nothing in this equation confines the Fourier content of f to the valley sector: once shifted to the valley, F(z) = exp(i k0 z) f(z) may carry plane waves beyond the strained zone boundary G0z / 2 or below the zone centre, which belong to the other valley. For a slowly varying potential that content is negligible; a germanium profile that varies on the scale of a few monolayers makes it larger.

The equation is solved on the same periodic supercell as the band-limited problem, with the kinetic operator applied spectrally over the full grid: the discrete Fourier transform of f is multiplied by hbar^2 k^2 / (2 m_l) at every grid wave number k and transformed back. The dense real symmetric Hamiltonian that results is diagonalised and its two lowest states are taken. The same two conventions as for the band-limited solve fix the output: dz sum_j f(z_j)^2 = 1, and f positive at the grid point where |f| is largest.

The stage also measures how much of the solution lies outside its sector. With N_FBZ = L G0z / (2 pi) grid wave numbers per zone, the sector is the set of discrete Fourier indices m = 1, ..., N_FBZ / 2 - 1, and the leakage is the fraction of sum_m |F_hat(m)|^2 carried by every other index, where F_hat is the discrete Fourier transform of exp(i k0 z_j) f(z_j).

```python
def local_valley_ground_state(
    z: np.ndarray,
    potential: np.ndarray,
    longitudinal_mass: float,
    valley_wavenumber: float,
    zone_vector: float,
) -> dict:
    """Solve the conventional local effective-mass equation on the supercell and measure its sector leakage.

    Parameters
    ----------
    z : np.ndarray
        Uniform supercell positions in nm, an even number of them.
    potential : np.ndarray
        Real confinement energy in eV on those positions.
    longitudinal_mass : float
        Longitudinal effective mass in free-electron masses, above zero.
    valley_wavenumber : float
        Valley minimum k0 in reciprocal nm, inside the open sector.
    zone_vector : float
        Strained growth-direction reciprocal-lattice vector G0z in reciprocal nm, above zero.

    Returns
    -------
    dict
        Under the keys energy, excited_energy, envelope, mean_position, spread and leakage.

    Raises
    ------
    ValueError
        When the grid fails to be uniform, finite and of even length, when the cell fails to hold an even whole number of grid wave numbers per zone, when the potential fails to be real, finite and of matching length, when the mass fails to be finite and above zero, or when k0 fails to lie strictly inside the open sector.
    """
    return
```

### Step 7

step_07_spectral_filter_envelope

Goal
----
A constant added to the confinement, U to U + U0, is a change of energy reference and nothing else, so every physical prediction must be blind to it. The coupling of stage 5 is linear in the weighting function, so under that change Delta moves to Delta + U0 R, where

R = sum_n C_n integral dz exp(-i (2 k0 + n G0z) z) conj(f(z))^2

is stage 5 evaluated with a unit weight. The splitting is blind to the reference exactly when R = 0. Whether it is depends on the envelope and not on the potential. Write conj(f)^2 = exp(2 i k0 z) conj(F)^2 with F = exp(i k0 z) f. If every plane wave of F lies in the open sector (0, G0z / 2), the wave numbers of conj(F)^2 lie in the open interval (-G0z, 0), and the n-th harmonic of R picks out its plane wave at n G0z, a whole multiple of G0z that the open interval never contains; every term vanishes and R is identically zero. An envelope with plane waves outside the sector has no such protection.

This stage measures R for a given envelope and then applies the cheapest repair available to an envelope computed without the sector restriction: project it onto its sector after the fact. With F_hat the discrete Fourier transform of exp(i k0 z_j) f(z_j) on the supercell, the filtered envelope keeps the indices m = 1, ..., N_FBZ / 2 - 1, zeroes all others, transforms back, removes the carrier again, and is renormalised so that dz sum_j |f|^2 = 1. The retained weight is the squared norm of the projection divided by that of the original, before renormalisation. The filtered envelope is complex in general even when the input is real, because its spectrum is no longer symmetric about the valley. Its R is evaluated as well, and on the grid it vanishes to rounding for the same reason as in the continuum: the wave numbers of conj(F)^2 are sums of two sector grid wave numbers, strictly between zero and G0z in magnitude, and no such sum is congruent to a whole multiple of G0z modulo the resolved band N_BZ G0z.

Filtering is not the same construction as solving the band-limited eigenproblem. It restores the protection of R, but the filtered function is not an eigenstate of the band-limited Hamiltonian, so its coupling approximates the band-limited result rather than reproducing it.

```python
def spectral_filter_envelope(
    z: np.ndarray,
    envelope: np.ndarray,
    valley_wavenumber: float,
    zone_vector: float,
    orders,
    coefficients,
) -> dict:
    """Measure the reference response of an envelope, project it onto its valley sector and measure again.

    Parameters
    ----------
    z : np.ndarray
        Uniform supercell positions in nm.
    envelope : np.ndarray
        Slowly varying envelope, normalised in the continuum sense.
    valley_wavenumber : float
        Valley minimum k0 in reciprocal nm, inside the open sector.
    zone_vector : float
        Strained growth-direction reciprocal-lattice vector G0z in reciprocal nm, above zero.
    orders : sequence of int
        Orders n, without repetition.
    coefficients : sequence of complex
        Overlap sums C_n.

    Returns
    -------
    dict
        Under the keys envelope, retained_weight, ambiguity_real, ambiguity_imag, ambiguity_abs and filtered_ambiguity_abs.

    Raises
    ------
    ValueError
        When the grid, envelope, orders or coefficients fail the checks of the coupling stage, when the cell fails to hold an even whole number of grid wave numbers per zone, when k0 fails to lie strictly inside the open sector, or when the projection retains no weight.
    """
    return
```

### Step 8

step_08_exact_valley_splitting

Goal
----
This stage answers the question the whole chain exists for: the valley splitting of the ground state of a strained silicon quantum well with an engineered germanium profile, computed in a multi-valley envelope-function description without a slowly varying potential approximation, in which each valley's envelope is band limited to its own sector of the Brillouin zone. It starts from the device and material description and reruns every earlier stage in order.

1. Stage 1 strains the silicon layer to the relaxed buffer and returns eps_zz, the strained growth-direction reciprocal-lattice vector G0z, the open sector (0, G0z / 2), the valley wave number k0 and its distance k1 to the strained zone boundary.

2. Stage 2 lays out the periodic supercell matched to G0z and evaluates the wiggle-well germanium profile and the confinement energy U(z) = Delta E_c X(z) - e F z on it.

3. Stage 3 assembles the back-folded, sector-projected potential operator on the open sector of the +k0 valley.

4. Stage 4 adds the kinetic energy measured from the valley minimum, solves the band-limited single-valley eigenproblem on that operator, and returns the ground-state envelope, its energy and the first excited energy.

5. Stage 5 evaluates the first-order intervalley coupling Delta of that envelope and the splitting 2 |Delta|, with the contribution of each harmonic order.

6. Stage 6 solves the conventional local effective-mass equation for the same stack on the same cell, and stage 5 evaluates its coupling, as the baseline the band-limited construction replaces.

7. Stage 7 measures, for the band-limited and for the local envelope, the response R of the coupling to a constant added to the confinement, and projects the local envelope onto its sector, after which stage 5 evaluates the coupling of the filtered envelope.

The graded quantity is the band-limited splitting in meV. The local and filtered results, the two responses, the dominant harmonic and its share, and the geometry of the strained zone are returned alongside it, because they are what distinguishes the construction from its approximations.

The ground state of a field-biased stack must be a state of the well. Stage 4 returns the lowest eigenvalue of the cell, which is that state only when the spacer is short enough that the field cannot pull the energy below the well ground state anywhere under the dielectric; this stage therefore rejects a result whose mean position lies more than 2 nm outside the silicon layer.

```python
def exact_valley_splitting(
    lattice_si: float,
    lattice_ge: float,
    bowing: float,
    substrate_ge_fraction: float,
    c11: float,
    c12: float,
    valley_fraction: float,
    longitudinal_mass: float,
    band_offset: float,
    x_barrier: float,
    x_wiggle: float,
    well_monolayers: float,
    sigma_upper: float,
    sigma_lower: float,
    wiggle_period: float,
    field: float,
    spacer: float,
    orders,
    coefficients,
    backfold,
    n_bz: int,
    n_fbz: int,
) -> dict:
    """Compute the band-limited valley splitting of a strained Si/SiGe wiggle well from the device description.

    Parameters
    ----------
    lattice_si : float
        Lattice constant of silicon in nm.
    lattice_ge : float
        Lattice constant of germanium in nm.
    bowing : float
        Bowing length of the alloy lattice constant in nm.
    substrate_ge_fraction : float
        Germanium fraction of the relaxed buffer.
    c11 : float
        Elastic constant C11 of silicon in GPa.
    c12 : float
        Elastic constant C12 of silicon in GPa.
    valley_fraction : float
        Valley minimum as a fraction of 2 pi / a_Si.
    longitudinal_mass : float
        Longitudinal effective mass in free-electron masses.
    band_offset : float
        Conduction-band offset per unit germanium fraction in eV.
    x_barrier : float
        Barrier germanium fraction.
    x_wiggle : float
        Amplitude of the in-well germanium oscillation.
    well_monolayers : float
        Well thickness in monolayers.
    sigma_upper : float
        Upper interface width in nm.
    sigma_lower : float
        Lower interface width in nm.
    wiggle_period : float
        Oscillation period in nm.
    field : float
        Vertical field in volts per nm.
    spacer : float
        Distance from the upper interface to the dielectric in nm.
    orders : sequence of int
        Orders n, containing zero and closed under negation.
    coefficients : sequence of complex
        Intervalley overlap sums C_n.
    backfold : sequence of complex
        Intravalley back-folding coefficients B_n.
    n_bz : int
        Resolved Brillouin zones.
    n_fbz : int
        Grid wave numbers per zone.

    Returns
    -------
    dict
        Under the keys splitting_mev, ground_energy_mev, excited_energy_mev, mean_position, spread, eps_zz, sector_edge_fraction, long_period_fraction, dominant_order, dominant_share, local_splitting_mev, local_ground_energy_mev, local_leakage, filtered_splitting_mev, local_ambiguity_abs and exact_ambiguity_abs.

    Raises
    ------
    ValueError
        When any stage rejects its inputs, or when the band-limited ground state lies more than 2 nm outside the silicon layer.
    """
    return
```

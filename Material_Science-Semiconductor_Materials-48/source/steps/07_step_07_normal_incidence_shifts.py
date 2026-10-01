"""
Step 07 - Density-induced shifts of the lower and upper polariton at normal incidence.

At mean-field level the interactions of a thermal polariton gas renormalise the energy of a probed polariton at normal incidence (Q' = 0), and transforming the carrier equations of motion into the polariton basis while keeping only branch-diagonal occupations gives each branch nu its own shift. The occupation per mode n_nu(Q) is the Boltzmann distribution of step 06, normalised so that its integral over d^2Q/(2 pi)^2, summed over both branches, is the total polariton density n.

Exchange enters through the exciton content of both the probed and the occupied states: Delta E_exch(nu) = W X_nu(0)^2 n <X^2>, with W from step 03 and <X^2> from step 06. Phase-space filling enters through the light-matter coupling and therefore through exciton-photon amplitude products: Delta E_psf(nu) = -(g/psi(0)) [X_nu(0) C_nu(0) B + X_nu(0)^2 D], where B is the sum over both branches of the integral of d^2Q/(2 pi)^2 n_mu(Q) I(Q) X_mu(Q)^2, D the same with X_mu(Q)^2 replaced by X_mu(Q) C_mu(Q), I(Q) the overlap of step 04, and psi(0) = sum_i c_i the real-space amplitude at the origin, which converts g into the electron-photon matrix element.

The first phase-space-filling term describes how the probed polariton is affected by the occupied excitons, and the second how the occupied states feel the reduced coupling. With the sign convention of step 05 the first term blueshifts the lower branch, while the second can blueshift the upper branch when the occupied states are strongly photonic. The step returns (Delta E_LP, Delta E_UP), each the sum of both channels, in meV. The momentum integrals share the two scales of step 06 and must be converged to the same accuracy.

Returns
-------
numpy.ndarray of shape (2,), the normal-incidence shifts (Delta E_LP, Delta E_UP) in meV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def normal_incidence_shifts(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float, exciton_energy: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float, density: float) -> np.ndarray:
    '''Normal-incidence LP and UP shifts produced by a thermal polariton gas.

    Parameters
    ----------
    exponents : array_like
        One-dimensional array of finite positive Gaussian exponents a_i, nm^-2.
    coefficients : array_like
        Coefficients c_i of the 1s amplitude psi(r) = sum_i c_i exp(-a_i r^2),
        same length as exponents, with sum_i c_i > 0.
    m_e : float
        Electron effective mass in units of m0, > 0.
    m_h : float
        Hole effective mass in units of m0, > 0.
    kappa : float
        Mean dielectric constant of the surrounding media, > 0.
    r0 : float
        Screening length of the sheet in nm, >= 0.
    exciton_energy : float
        Exciton energy at Q = 0 in eV, > 0.
    cavity_detuning : float
        E_C(0) - E_X(0) in eV; E_C(0) must be positive.
    cavity_index : float
        Effective refractive index of the cavity mode, > 0.
    coupling : float
        Real exciton-photon coupling g in eV, > 0.
    temperature : float
        Temperature in K, > 0.
    density : float
        Total polariton density in nm^-2, > 0 (1 nm^-2 = 1e14 cm^-2).

    Returns
    -------
    shifts : numpy.ndarray
        Array (Delta E_LP, Delta E_UP) of the normal-incidence shifts in meV,
        each the sum of the exchange and phase-space-filling contributions.

    Raises
    ------
    ValueError
        If temperature or density is not finite and positive, if sum_i c_i is not
        positive, or if any input is invalid as in steps 03 to 06.
    '''
    return shifts

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _oracle_normal_incidence_shifts(exponents: npt.ArrayLike, coefficients: npt.ArrayLike, m_e: float, m_h: float, kappa: float, r0: float, exciton_energy: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float, density: float) -> np.ndarray:
    """Density-induced normal-incidence shifts of the lower and upper polariton, meV."""
    import numpy as np
    hb2m0 = 0.0380998
    hbarc = 197.32698
    kb = 8.617333e-5
    temperature = float(temperature)
    density = float(density)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(density) or density <= 0.0:
        raise ValueError("density must be finite and positive")
    c = np.asarray(coefficients, dtype=float)
    psi0 = float(np.sum(c))
    if not np.isfinite(psi0) or psi0 <= 0.0:
        raise ValueError("the amplitude at the origin, sum of coefficients, must be positive")
    mass_total = float(m_e) + float(m_h)
    b0 = _oracle_polariton_branches(np.array([0.0]), exciton_energy, mass_total, cavity_detuning, cavity_index, coupling)[:, 0]
    def _nodes():
        """Gauss-Legendre panels on [0, qmax] resolving the light cone and the thermal reservoir."""
        from numpy.polynomial.legendre import leggauss
        ec0 = exciton_energy + cavity_detuning
        q_ph = cavity_index * np.sqrt(2.0 * ec0 * (abs(cavity_detuning) + 2.0 * coupling)) / hbarc
        q_th = np.sqrt(mass_total * kb * temperature / hb2m0)
        qmax = max(40.0 * q_th, 10.0)
        edges = np.unique(np.concatenate([[0.0], np.geomspace(q_ph * 1e-3, q_ph * 30.0, 24),
                                          np.geomspace(q_ph * 30.0, qmax, 24)]))
        x, wx = leggauss(48)
        nodes, weights = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            nodes.append(0.5 * (hi - lo) * x + 0.5 * (hi + lo))
            weights.append(0.5 * (hi - lo) * wx)
        return np.concatenate(nodes), np.concatenate(weights)

    w_ex = _oracle_exchange_constant(exponents, coefficients, kappa, r0)
    q, wq = _nodes()
    br = _oracle_polariton_branches(q, exciton_energy, mass_total, cavity_detuning, cavity_index, coupling)
    ov = _oracle_saturation_overlap(exponents, coefficients, m_e, m_h, q)
    kt = kb * temperature
    occ_lp = np.exp(-(br[0] - b0[0]) / kt)
    occ_up = np.exp(-(br[1] - b0[0]) / kt)
    meas = q * wq / (2.0 * np.pi)
    fug = density / np.sum(meas * (occ_lp + occ_up))
    n_lp = fug * occ_lp
    n_up = fug * occ_up
    x_mean = _oracle_thermal_exciton_fraction(exciton_energy, mass_total, cavity_detuning, cavity_index, coupling, temperature)
    dens_x = density * x_mean
    dens_ix = np.sum(meas * ov * (n_lp * br[2] ** 2 + n_up * br[4] ** 2))
    dens_ixc = np.sum(meas * ov * (n_lp * br[2] * br[3] + n_up * br[4] * br[5]))
    shifts = []
    for x0, c0 in ((b0[2], b0[3]), (b0[4], b0[5])):
        exch = w_ex * x0 ** 2 * dens_x
        psf = -(coupling / psi0) * (x0 * c0 * dens_ix + x0 ** 2 * dens_ixc)
        shifts.append(1.0e3 * (exch + psf))
    return np.array(shifts)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: cavity 30 meV below the exciton at 50 K and 1e12 cm^-2 ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.02, 0.1, 0.5, 2.5, 12.5])\n'
                      'coefficients = np.array([-0.00484781934334, 0.131037470592, 0.300175291225, 0.0798788920878, 0.0234948262371])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'kappa = 4.5\n'
                      'r0 = 4.15\n'
                      'exciton_energy = 1.9\n'
                      'cavity_detuning = -0.03\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 50.0\n'
                      'density = 0.01\n'),
            "call": 'normal_incidence_shifts(exponents, coefficients, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "gold_call": '_oracle_normal_incidence_shifts(exponents, coefficients, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "tol": 1e-06,
        },
        # --- Boundary: room temperature, reservoir-dominated population ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.02, 0.1, 0.5, 2.5, 12.5])\n'
                      'coefficients = np.array([-0.00484781934334, 0.131037470592, 0.300175291225, 0.0798788920878, 0.0234948262371])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'kappa = 4.5\n'
                      'r0 = 4.15\n'
                      'exciton_energy = 1.9\n'
                      'cavity_detuning = -0.03\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 300.0\n'
                      'density = 0.01\n'),
            "call": 'normal_incidence_shifts(exponents, coefficients, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "gold_call": '_oracle_normal_incidence_shifts(exponents, coefficients, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "tol": 1e-06,
        },
        # --- Edge: zero detuning at 10 K, where the two phase-space-filling terms nearly cancel for the upper branch ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.02, 0.1, 0.5, 2.5, 12.5])\n'
                      'coefficients = np.array([-0.00484781934334, 0.131037470592, 0.300175291225, 0.0798788920878, 0.0234948262371])\n'
                      'm_e = 0.47\n'
                      'm_h = 0.54\n'
                      'kappa = 4.5\n'
                      'r0 = 4.15\n'
                      'exciton_energy = 1.9\n'
                      'cavity_detuning = 0.0\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 10.0\n'
                      'density = 0.01\n'),
            "call": 'normal_incidence_shifts(exponents, coefficients, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "gold_call": '_oracle_normal_incidence_shifts(exponents, coefficients, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "tol": 1e-06,
        },
        # --- Edge: cavity 50 meV below the exciton at 10 K, where phase-space filling blueshifts the upper branch ---
        {
            "setup": ('import numpy as np\n'
                      'exponents = np.array([0.05, 0.4, 3.2, 25.6])\n'
                      'coefficients = np.array([-0.00951763743232, 0.169820649119, 1.06859210545, 0.17307684644])\n'
                      'm_e = 1.0\n'
                      'm_h = 1.2\n'
                      'kappa = 1.0\n'
                      'r0 = 2.0\n'
                      'exciton_energy = 1.9\n'
                      'cavity_detuning = -0.05\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 10.0\n'
                      'density = 0.005\n'),
            "call": 'normal_incidence_shifts(exponents, coefficients, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "gold_call": '_oracle_normal_incidence_shifts(exponents, coefficients, m_e, m_h, kappa, r0, exciton_energy, cavity_detuning, cavity_index, coupling, temperature, density)',
            "tol": 1e-06,
        },
    ]

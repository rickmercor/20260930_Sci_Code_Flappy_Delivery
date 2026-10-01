"""
Step 06 - Mean exciton fraction of a thermal polariton gas.

Relaxation in a microcavity fills the polariton dispersion rather than a single state, and the nonlinear response depends on how the population divides between the few strongly photonic states inside the light cone and the far larger number of exciton-like states at large momentum. Take the population as a Boltzmann distribution over the low-density dispersion of both branches of step 05, with the continuous two-dimensional density of states. The quantity that controls the exchange response is the density-weighted mean exciton fraction <X^2>: the sum over both branches of the integral of Q dQ exp(-E_nu(Q)/(k_B T)) X_nu(Q)^2, divided by the same sum without the factor X_nu(Q)^2.

The light cone is a tiny region of momentum space, of order 1e-3 nm^-1 against thermal momenta of tenths of an inverse nanometre, yet it lies several meV lower in energy, so the integral has to resolve both scales; a uniform momentum grid misses the light cone altogether. Use k_B = 8.617333e-5 eV/K and converge the result to about one part in 1e9.

Returns
-------
float, the density-weighted mean exciton fraction of the thermal polariton gas
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def thermal_exciton_fraction(exciton_energy: float, total_mass: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float) -> float:
    '''Density-weighted mean exciton fraction of a Boltzmann polariton gas.

    Parameters
    ----------
    exciton_energy : float
        Exciton energy at Q = 0 in eV, > 0.
    total_mass : float
        Total exciton mass in units of m0, > 0.
    cavity_detuning : float
        E_C(0) - E_X(0) in eV; E_C(0) must be positive.
    cavity_index : float
        Effective refractive index of the cavity mode, > 0.
    coupling : float
        Real exciton-photon coupling g in eV, > 0.
    temperature : float
        Temperature in K, finite and > 0.

    Returns
    -------
    fraction : float
        <X^2>, the Boltzmann-weighted mean of X_nu(Q)^2 over both branches and the
        whole momentum plane, between 0 and 1.

    Raises
    ------
    ValueError
        If temperature is not finite and positive, or if the dispersion
        parameters are invalid as in step 05.
    '''
    return fraction

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt


def _oracle_thermal_exciton_fraction(exciton_energy: float, total_mass: float, cavity_detuning: float, cavity_index: float, coupling: float, temperature: float) -> float:
    """Mean exciton fraction of a Boltzmann polariton gas on the low-density dispersion."""
    import numpy as np
    hb2m0 = 0.0380998
    hbarc = 197.32698
    kb = 8.617333e-5
    temperature = float(temperature)
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    mass_total = float(total_mass)
    e0 = _oracle_polariton_branches(np.array([0.0]), exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)[0, 0]
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

    q, wq = _nodes()
    br = _oracle_polariton_branches(q, exciton_energy, total_mass, cavity_detuning, cavity_index, coupling)
    kt = kb * temperature
    occ_lp = np.exp(-(br[0] - e0) / kt)
    occ_up = np.exp(-(br[1] - e0) / kt)
    meas = q * wq
    total = np.sum(meas * (occ_lp + occ_up))
    return float(np.sum(meas * (occ_lp * br[2] ** 2 + occ_up * br[4] ** 2)) / total)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: cavity 30 meV below the exciton at 50 K, population split between light cone and reservoir ---
        {
            "setup": ('import numpy as np\n'
                      'exciton_energy = 1.9\n'
                      'total_mass = 1.01\n'
                      'cavity_detuning = -0.03\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 50.0\n'),
            "call": 'thermal_exciton_fraction(exciton_energy, total_mass, cavity_detuning, cavity_index, coupling, temperature)',
            "gold_call": '_oracle_thermal_exciton_fraction(exciton_energy, total_mass, cavity_detuning, cavity_index, coupling, temperature)',
            "tol": 1e-08,
        },
        # --- Boundary: 10 K, population almost entirely inside the light cone ---
        {
            "setup": ('import numpy as np\n'
                      'exciton_energy = 1.9\n'
                      'total_mass = 1.01\n'
                      'cavity_detuning = -0.03\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 10.0\n'),
            "call": 'thermal_exciton_fraction(exciton_energy, total_mass, cavity_detuning, cavity_index, coupling, temperature)',
            "gold_call": '_oracle_thermal_exciton_fraction(exciton_energy, total_mass, cavity_detuning, cavity_index, coupling, temperature)',
            "tol": 1e-08,
        },
        # --- Edge: room temperature, population almost entirely in the exciton reservoir ---
        {
            "setup": ('import numpy as np\n'
                      'exciton_energy = 1.9\n'
                      'total_mass = 1.01\n'
                      'cavity_detuning = -0.03\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 300.0\n'),
            "call": 'thermal_exciton_fraction(exciton_energy, total_mass, cavity_detuning, cavity_index, coupling, temperature)',
            "gold_call": '_oracle_thermal_exciton_fraction(exciton_energy, total_mass, cavity_detuning, cavity_index, coupling, temperature)',
            "tol": 1e-08,
        },
        # --- Normal: zero detuning at 20 K, close to the crossover out of the light cone ---
        {
            "setup": ('import numpy as np\n'
                      'exciton_energy = 1.9\n'
                      'total_mass = 1.01\n'
                      'cavity_detuning = 0.0\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 20.0\n'),
            "call": 'thermal_exciton_fraction(exciton_energy, total_mass, cavity_detuning, cavity_index, coupling, temperature)',
            "gold_call": '_oracle_thermal_exciton_fraction(exciton_energy, total_mass, cavity_detuning, cavity_index, coupling, temperature)',
            "tol": 1e-08,
        },
        # --- Edge: cavity 50 meV above the exciton, where the lower branch is exciton-like everywhere ---
        {
            "setup": ('import numpy as np\n'
                      'exciton_energy = 1.9\n'
                      'total_mass = 1.01\n'
                      'cavity_detuning = 0.05\n'
                      'cavity_index = 1.8\n'
                      'coupling = 0.02\n'
                      'temperature = 77.0\n'),
            "call": 'thermal_exciton_fraction(exciton_energy, total_mass, cavity_detuning, cavity_index, coupling, temperature)',
            "gold_call": '_oracle_thermal_exciton_fraction(exciton_energy, total_mass, cavity_detuning, cavity_index, coupling, temperature)',
            "tol": 1e-08,
        },
    ]

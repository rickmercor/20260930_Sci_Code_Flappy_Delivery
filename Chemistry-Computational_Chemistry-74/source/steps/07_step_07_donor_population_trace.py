"""
Donor population versus time for the cavity-coupled donor-acceptor pair with a low-frequency bath and an intramolecular mode, starting in the donor state with the cavity in its vacuum.

Assemble the full model from the previous steps: the dipole-gauge system Hamiltonian, the two system-bath coupling operators, the Lindblad system Liouvillian, the Drude-Lorentz exponents of the low-frequency bath (bath 0, coupled through V_1) and the Brownian-oscillator exponents of the intramolecular mode (bath 1, coupled through V_2). A mode with zero reorganization energy is absent and contributes no terms. The initial state is |D><D| times the photon vacuum |0><0|, with both baths in equilibrium.

The donor population is the trace of the reduced density matrix over all photon states within the D block. It is sampled from t = 0 to t_max at intervals of sample_ps, with the hierarchy integrated at the time step dt_ps.

Returns
-------
numpy.ndarray of length round(t_max / sample) + 1: the donor population at the sampled times
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_donor_population_trace(driving_force_cm: float, electronic_coupling_cm: float,
                                   cavity_energy_cm: float, x_dd: float, x_aa: float, x_da: float,
                                   kappa_cm: float, reorg_cm: float, tau_bath_fs: float,
                                   mode_reorg_cm: float, mode_cm: float, tau_mode_fs: float,
                                   temperature_k: float, n_matsubara_bath: int, n_matsubara_mode: int,
                                   depth: int, n_photon: int, t_max_ps: float, dt_ps: float,
                                   sample_ps: float) -> "np.ndarray":
    import numpy as np
    if not (np.isfinite(t_max_ps) and np.isfinite(dt_ps) and np.isfinite(sample_ps)) or min(t_max_ps, dt_ps, sample_ps) <= 0.0:
        raise ValueError("t_max, dt and the sampling interval must be positive")
    stride = int(round(sample_ps / dt_ps))
    n_steps = int(round(t_max_ps / dt_ps))
    if abs(stride * dt_ps - sample_ps) > 1e-9 * sample_ps or abs(n_steps * dt_ps - t_max_ps) > 1e-9 * t_max_ps or n_steps % stride:
        raise ValueError("t_max must be a multiple of the sampling interval, which must be a multiple of dt")
    if not np.isfinite(mode_reorg_cm) or mode_reorg_cm < 0.0:
        raise ValueError("mode reorganization energy must be finite and non-negative")
    n = int(n_photon)
    ham = _oracle_dipole_gauge_hamiltonian(driving_force_cm, electronic_coupling_cm, cavity_energy_cm,
                                           x_dd, x_aa, x_da, n)
    ops = _oracle_bath_coupling_operators(n)
    liou = _oracle_system_liouvillian(ham, kappa_cm, n)
    ex_b = _oracle_drude_bath_exponents(reorg_cm, tau_bath_fs, temperature_k, n_matsubara_bath)
    ex_m = _oracle_underdamped_mode_exponents(mode_reorg_cm, mode_cm, tau_mode_fs, temperature_k,
                                              n_matsubara_mode)
    if mode_reorg_cm > 0.0:
        ex = np.vstack([ex_b, ex_m])
        index = np.array([0] * ex_b.shape[0] + [1] * ex_m.shape[0])
    else:
        ex = ex_b
        index = np.zeros(ex_b.shape[0], dtype=int)
    rho0 = np.zeros((2 * n, 2 * n), dtype=complex)
    rho0[0, 0] = 1.0
    frames = _oracle_heom_propagate(liou, ops, ex, index, depth, rho0, dt_ps, n_steps, stride)
    return np.real(np.einsum('tii->t', frames[:, :n, :n]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: all couplings and the intramolecular mode, depth 3, two photon states ---
        {
            "setup": """import numpy as np
F = 100.0
Vda = 30.0
wc = 100.0
xdd = 0.2
xaa = -0.1
xda = 0.3
kap = 10.0
lam = 50.0
tau = 100.0
lamv = 2.0
wv = 500.0
tauv = 100.0
T = 300.0
nmb = 1
nmv = 0
depth = 3
nph = 2
tmax = 0.2
dt = 0.001
samp = 0.02
""",
            "call": "donor_population_trace(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "gold_call": "_oracle_donor_population_trace(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "tol": 1e-09,
        },
        # --- Boundary: no cavity coupling, no mode and one photon state ---
        {
            "setup": """import numpy as np
F = 100.0
Vda = 30.0
wc = 100.0
xdd = 0.0
xaa = 0.0
xda = 0.0
kap = 0.0
lam = 50.0
tau = 100.0
lamv = 0.0
wv = 500.0
tauv = 100.0
T = 300.0
nmb = 1
nmv = 0
depth = 4
nph = 1
tmax = 0.3
dt = 0.001
samp = 0.03
""",
            "call": "donor_population_trace(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "gold_call": "_oracle_donor_population_trace(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "tol": 1e-09,
        },
        # --- Edge: cold baths, three-body coupling through the mode and loss only ---
        {
            "setup": """import numpy as np
F = 100.0
Vda = 30.0
wc = 100.0
xdd = 0.0
xaa = 0.0
xda = 0.0
kap = 20.0
lam = 50.0
tau = 100.0
lamv = 5.0
wv = 400.0
tauv = 100.0
T = 150.0
nmb = 1
nmv = 1
depth = 2
nph = 3
tmax = 0.1
dt = 0.0005
samp = 0.01
""",
            "call": "donor_population_trace(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "gold_call": "_oracle_donor_population_trace(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "tol": 1e-09,
        },
        # --- Invalid: final time not a multiple of the sampling interval ---
        {
            "setup": """import numpy as np
F = 100.0
Vda = 30.0
wc = 100.0
xdd = 0.2
xaa = -0.1
xda = 0.3
kap = 10.0
lam = 50.0
tau = 100.0
lamv = 2.0
wv = 500.0
tauv = 100.0
T = 300.0
nmb = 1
nmv = 0
depth = 3
nph = 2
tmax = 0.1
dt = 0.001
samp = 0.015
def run_model():
    try:
        donor_population_trace(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_donor_population_trace(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

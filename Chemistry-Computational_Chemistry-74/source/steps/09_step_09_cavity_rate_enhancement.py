"""
Orchestrator: ratio of the forward electron-transfer rate with the lossy cavity, including the three-body coupling through the intramolecular mode, to the forward rate of the same molecule without the cavity.

The cavity effect is measured by R = k_DA(with cavity) / k_DA(without cavity). Both rates come from the same procedure: the donor-population trace of step 07 from t = 0 to t_max and the least-squares fit of step 08 over all its samples.

Without the cavity there is no light-matter coupling at all: every dipole projection and the loss rate are zero, and the three-body term, which only acts through the cavity field, disappears with them, so the intramolecular mode no longer couples to the molecule. The reference is therefore the bare donor-acceptor pair in the low-frequency bath with the same hierarchy settings.

Returns
-------
float: R = k_DA(cavity) / k_DA(no cavity)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

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

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_cavity_rate_enhancement(driving_force_cm: float, electronic_coupling_cm: float,
                                    cavity_energy_cm: float, x_dd: float, x_aa: float, x_da: float,
                                    kappa_cm: float, reorg_cm: float, tau_bath_fs: float,
                                    mode_reorg_cm: float, mode_cm: float, tau_mode_fs: float,
                                    temperature_k: float, n_matsubara_bath: int, n_matsubara_mode: int,
                                    depth: int, n_photon: int, t_max_ps: float, dt_ps: float,
                                    sample_ps: float) -> float:
    import numpy as np
    cav = _oracle_donor_population_trace(driving_force_cm, electronic_coupling_cm, cavity_energy_cm,
                                         x_dd, x_aa, x_da, kappa_cm, reorg_cm, tau_bath_fs,
                                         mode_reorg_cm, mode_cm, tau_mode_fs, temperature_k,
                                         n_matsubara_bath, n_matsubara_mode, depth, n_photon,
                                         t_max_ps, dt_ps, sample_ps)
    bare = _oracle_donor_population_trace(driving_force_cm, electronic_coupling_cm, cavity_energy_cm,
                                          0.0, 0.0, 0.0, 0.0, reorg_cm, tau_bath_fs,
                                          0.0, mode_cm, tau_mode_fs, temperature_k,
                                          n_matsubara_bath, n_matsubara_mode, depth, 1,
                                          t_max_ps, dt_ps, sample_ps)
    n_steps = int(round(t_max_ps / dt_ps))
    stride = int(round(sample_ps / dt_ps))
    times = dt_ps * np.arange(0, n_steps + 1, stride)
    k_cav = _oracle_fit_transfer_rates(times, cav)
    k_bare = _oracle_fit_transfer_rates(times, bare)
    if k_bare[0] <= 0.0:
        raise ValueError("the cavity-free forward rate vanished; the ratio is undefined")
    return float(k_cav[0] / k_bare[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: all couplings and the mode, depth 3, two photon states, 1 ps ---
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
tmax = 1.0
dt = 0.001
samp = 0.02
""",
            "call": "cavity_rate_enhancement(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "gold_call": "_oracle_cavity_rate_enhancement(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "tol": 0.0005,
        },
        # --- Boundary: no cavity coupling and no mode, the ratio is one ---
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
depth = 3
nph = 2
tmax = 1.0
dt = 0.002
samp = 0.02
""",
            "call": "cavity_rate_enhancement(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "gold_call": "_oracle_cavity_rate_enhancement(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "tol": 0.0005,
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
lamv = 4.0
wv = 400.0
tauv = 100.0
T = 150.0
nmb = 1
nmv = 0
depth = 3
nph = 3
tmax = 1.0
dt = 0.001
samp = 0.02
""",
            "call": "cavity_rate_enhancement(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "gold_call": "_oracle_cavity_rate_enhancement(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "tol": 0.0005,
        },
        # --- Normal: transition dipole alone, no mode and no loss ---
        {
            "setup": """import numpy as np
F = 100.0
Vda = 30.0
wc = 100.0
xdd = 0.0
xaa = 0.0
xda = 0.5
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
nph = 3
tmax = 1.5
dt = 0.002
samp = 0.03
""",
            "call": "cavity_rate_enhancement(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "gold_call": "_oracle_cavity_rate_enhancement(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)",
            "tol": 0.0005,
        },
        # --- Invalid: a negative reorganization energy for the intramolecular mode ---
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
lamv = -1.0
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
def run_model():
    try:
        cavity_rate_enhancement(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_cavity_rate_enhancement(F, Vda, wc, xdd, xaa, xda, kap, lam, tau, lamv, wv, tauv, T, nmb, nmv, depth, nph, tmax, dt, samp)
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

"""
Cosine of the Bloch phase of a plasma wave in a two-strip lateral plasmonic crystal at a given frequency.

The crystal is a two-dimensional electron layer whose every period L is split into strip 1 of width fL and strip 2 of width (1 - f)L. Strip n carries the equilibrium electron density N_n and sits under its own ideal metal plane at distance h_n; the semiconductor below the layer has permittivity eps_s and the spacer between the layer and the plane above strip n has permittivity kappa_n, so the two strips can differ in density, in gate distance and in spacer permittivity. Damping is neglected throughout.

The crystal is described by Kronig-Penney matching. Inside strip n a plasma wave of frequency omega is a superposition of waves with the local wave number q_n(omega) of the screened dispersion of the previous step, evaluated with N_n, h_n and kappa_n, and eps_n(q) = [eps_s + kappa_n coth(q h_n)]/2 is the corresponding effective permittivity. Its density perturbation is delta N_n = A_n exp(i q_n x) + B_n exp(-i q_n x), the linearised continuity equation fixes the ac velocity, v_n = omega/(q_n N_n) (A_n exp(i q_n x) - B_n exp(-i q_n x)), and the ac potential is related to the density perturbation as in an infinite layer of the same kind, phi_n = K_n delta N_n with K_n = 2 pi e^2 / (q_n eps_n(q_n)). At every boundary between the strips the ac particle current N v and the ac potential are continuous, and after one period the wave must reproduce itself up to the Bloch factor exp(i k L), with k the quasimomentum.

A solution with quasimomentum k exists at frequency omega only when cos(k L) equals half the trace of the transfer matrix that carries the pair (potential, particle current) across one full period. That half-trace does not depend on how the two components are normalised or on where the period starts, and it is the quantity returned. Allowed bands are the frequency ranges where its magnitude does not exceed 1.

All internal work is in Gaussian units with e = 4.80320471e-10 statC, the free-electron mass m_0 = 9.1093837015e-28 g and hbar = 1.054571817e-27 erg s. The frequency is given in GHz.

Returns
-------
float: cos(k L), the half-trace of the one-period transfer matrix at the given frequency
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def bloch_phase_cosine(frequency_ghz: float, period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float) -> float:
    '''Half-trace of the one-period transfer matrix of the two-strip plasmonic crystal.

    Parameters
    ----------
    frequency_ghz : float
        Wave frequency omega/(2 pi) in GHz.
    period_um : float
        Period L of the crystal in micrometres.
    fill : float
        Filling factor f, the fraction of each period occupied by strip 1; 0 < f < 1.
    density1 : float
        Electron density N_1 of strip 1 in cm^-2.
    density2 : float
        Electron density N_2 of strip 2 in cm^-2.
    gate1_nm : float
        Distance h_1 between the electron layer and the metal plane above strip 1, in nm.
    gate2_nm : float
        Distance h_2 between the electron layer and the metal plane above strip 2, in nm.
    eps_substrate : float
        Permittivity eps_s of the semiconductor below the electron layer.
    eps_spacer1 : float
        Permittivity kappa_1 of the spacer between the electron layer and the metal plane above strip 1.
    eps_spacer2 : float
        Permittivity kappa_2 of the spacer between the electron layer and the metal plane above strip 2.
    mass_ratio : float
        Electron effective mass in units of the free-electron mass.

    Returns
    -------
    cos_kl : float
        cos(k L) of the Bloch wave at this frequency; values outside [-1, 1] mark a band gap.

    Raises
    ------
    ValueError
        If any physical argument is not a positive finite number or if fill is not
        strictly between 0 and 1.
    '''
    return cos_kl

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _sm67_check_cell(period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio):
    import numpy as np
    _sm67_require_positive(dict(period_um=period_um, density1=density1, density2=density2, gate1_nm=gate1_nm,
                                gate2_nm=gate2_nm, eps_substrate=eps_substrate, eps_spacer1=eps_spacer1,
                                eps_spacer2=eps_spacer2, mass_ratio=mass_ratio))
    if not np.isfinite(fill) or not (0.0 < fill < 1.0):
        raise ValueError("fill must lie strictly between 0 and 1")


def _sm67_cell_phases(omega, period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio):
    L = period_um * 1e-4
    q1 = _sm67_q_solve(omega, density1, gate1_nm * 1e-7, eps_substrate, eps_spacer1, mass_ratio)
    q2 = _sm67_q_solve(omega, density2, gate2_nm * 1e-7, eps_substrate, eps_spacer2, mass_ratio)
    eta = (_sm67_screened_eps(q2, gate2_nm * 1e-7, eps_substrate, eps_spacer2)
           / _sm67_screened_eps(q1, gate1_nm * 1e-7, eps_substrate, eps_spacer1))
    return q1 * fill * L, q2 * (1.0 - fill) * L, eta


def _oracle_bloch_phase_cosine(frequency_ghz: float, period_um: float, fill: float, density1: float, density2: float, gate1_nm: float,
                      gate2_nm: float, eps_substrate: float, eps_spacer1: float, eps_spacer2: float,
                      mass_ratio: float) -> float:
    import numpy as np
    _sm67_check_cell(period_um, fill, density1, density2, gate1_nm, gate2_nm, eps_substrate, eps_spacer1, eps_spacer2, mass_ratio)
    q1 = _oracle_local_plasmon_wavenumber(frequency_ghz, density1, gate1_nm, eps_substrate, eps_spacer1, mass_ratio)
    q2 = _oracle_local_plasmon_wavenumber(frequency_ghz, density2, gate2_nm, eps_substrate, eps_spacer2, mass_ratio)
    L = period_um * 1e-4
    eta = (_sm67_screened_eps(q2, gate2_nm * 1e-7, eps_substrate, eps_spacer2)
           / _sm67_screened_eps(q1, gate1_nm * 1e-7, eps_substrate, eps_spacer1))
    a1, a2 = q1 * fill * L, q2 * (1.0 - fill) * L
    z = 0.5 * (eta + 1.0 / eta)
    return float(np.cos(a1) * np.cos(a2) - z * np.sin(a1) * np.sin(a2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # --- Normal: benchmark crystal close to its impedance-matched frequency ---
        {
            "setup": """import numpy as np
""",
            "call": "bloch_phase_cosine(500.0, 6.0, 0.44, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)",
            "gold_call": "_oracle_bloch_phase_cosine(500.0, 6.0, 0.44, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)",
            "tol": 1e-09,
        },
        # --- Normal: benchmark crystal at higher frequency ---
        {
            "setup": """import numpy as np
""",
            "call": "bloch_phase_cosine(800.0, 6.0, 0.2, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)",
            "gold_call": "_oracle_bloch_phase_cosine(800.0, 6.0, 0.2, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)",
            "tol": 1e-09,
        },
        # --- Boundary: equal gate distances and spacers, the gated-gated crystal ---
        {
            "setup": """import numpy as np
""",
            "call": "bloch_phase_cosine(150.0, 8.0, 0.5, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067)",
            "gold_call": "_oracle_bloch_phase_cosine(150.0, 8.0, 0.5, 4.0e11, 1.6e12, 200.0, 200.0, 12.8, 12.8, 12.8, 0.067)",
            "tol": 1e-09,
        },
        # --- Edge: a far second plane that makes strip 2 effectively ungated, strong density contrast ---
        {
            "setup": """import numpy as np
""",
            "call": "bloch_phase_cosine(400.0, 8.0, 0.3, 1.0e11, 1.6e12, 320.0, 50000.0, 12.8, 12.8, 12.8, 0.067)",
            "gold_call": "_oracle_bloch_phase_cosine(400.0, 8.0, 0.3, 1.0e11, 1.6e12, 320.0, 50000.0, 12.8, 12.8, 12.8, 0.067)",
            "tol": 1e-09,
        },
        # --- Edge: low-permittivity spacer on strip 2 only ---
        {
            "setup": """import numpy as np
""",
            "call": "bloch_phase_cosine(900.0, 6.0, 0.62, 3.0e11, 3.0e12, 100.0, 300.0, 12.8, 12.8, 4.0, 0.067)",
            "gold_call": "_oracle_bloch_phase_cosine(900.0, 6.0, 0.62, 3.0e11, 3.0e12, 100.0, 300.0, 12.8, 12.8, 4.0, 0.067)",
            "tol": 1e-09,
        },
        # --- Invalid: filling factor above one ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        bloch_phase_cosine(500.0, 6.0, 1.2, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_bloch_phase_cosine(500.0, 6.0, 1.2, 3.0e11, 3.0e12, 300.0, 1500.0, 12.8, 12.8, 7.0, 0.067)
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

"""
Fermi level of the two-dimensional electron gas relative to the conduction band edge at the interface,
for a given sheet density, from a self-consistent Schrodinger-Poisson solution of the GaN channel.

The channel is the GaN side of the heterointerface, z >= 0, with the interface treated as an impenetrable
wall. Electrons of effective mass m* move in a conduction band that their own charge bends; the field at
the interface is tied to the electron sheet by Gauss's law and vanishes deep in the buffer, so no barrier
quantity enters. At zero temperature every subband below the Fermi level is occupied with the
two-dimensional density of states m* / (pi hbar^2), and the total density fixes E_F.

The object is defined on a uniform grid z_j = j h, j = 0 .. M-1, spanning a box of length L, h = L/(M-1).
The envelope functions vanish at both ends of the box, the kinetic operator is the second-order central
difference, and each envelope is normalised so that h * sum_j |psi_i(z_j)|^2 = 1. The electron density is
n(z_j) = sum_i N_i |psi_i(z_j)|^2 with N_i = (m* / (pi hbar^2)) (E_F - E_i) over the occupied subbands, and
the field and band edge follow from plain sums, F_j = (q / eps_GaN) h sum_(k >= j) n(z_k) and
E_C(z_j) - E_C(0) = q h sum_(k < j) F_k. The lowest four levels are computed, which covers every density
in this task. Schrodinger and Poisson are iterated to self-consistency; the converged Fermi level does not
depend on the mixing scheme and is reproducible to better than 1e-8 eV. At zero density the Fermi level
is placed at the ground level of the empty box.

Inputs: sheet_density: float >= 0, n_s in m^-2 effective_mass_ratio: float > 0, m* / m_e in the channel gan: GaN parameter dictionary keyed 'a', 'psp', 'e31', 'e33', 'c13', 'c33' and 'eps_r'; only 'eps_r' (relative
  permittivity) is used here box_length: float > 0, L in m (default 40e-9) grid_points: int >= 5, M (default 401)

 Returns: float, Fermi level above the interface conduction band edge in eV

 Raises: ValueError if sheet_density is negative, effective_mass_ratio or box_length is not positive, or grid_points is smaller than 5. RuntimeError if the self-consistent iteration fails to converge.

Returns
-------
float, Fermi level above the interface conduction band edge in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def channel_fermi_level(sheet_density: float, effective_mass_ratio: float, gan: dict,
                        box_length: float = 40e-9, grid_points: int = 401) -> float:
    """Self-consistent Schrodinger-Poisson Fermi level of the 2DEG above the interface band edge, in eV.
 
    Zero-temperature filling of the subbands of the hard-wall GaN channel on the uniform grid described in
    the step background. Raises ValueError if sheet_density < 0, effective_mass_ratio <= 0, box_length <= 0
    or grid_points < 5.
    """
    return 0.0  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.linalg import eigh_tridiagonal

def _sp_sweep(potential, sheet_density, m_eff, eps, h, m_pts, diag0, off):
    """One Schrodinger-Poisson sweep on the grid: potential -> (new potential, E_F, levels, occupied count)."""
    HBAR = 1.054571817e-34
    Q = 1.602176634e-19
    levels, vecs = eigh_tridiagonal(diag0 + potential[1:-1], off, select='i', select_range=(0, 3))
    dos = m_eff / (np.pi * HBAR * HBAR)
    k = 1
    e_f = sheet_density / dos + levels[0]
    for k in range(1, 5):
        e_f = (sheet_density / dos + levels[:k].sum()) / k
        if k == 4 or e_f <= levels[k]:
            break
    occ = dos * np.clip(e_f - levels[:k], 0.0, None)
    psi = np.zeros((m_pts, k))
    psi[1:-1, :] = vecs[:, :k]
    psi /= np.sqrt(h * np.sum(psi * psi, axis=0))
    dens = (psi * psi * occ).sum(axis=1)
    field = (Q / eps) * h * np.cumsum(dens[::-1])[::-1]
    new_potential = np.concatenate(([0.0], Q * h * np.cumsum(field)[:-1]))
    return new_potential, e_f, levels, k

def _oracle_channel_fermi_level(sheet_density: float, effective_mass_ratio: float, gan: dict,
                                box_length: float = 40e-9, grid_points: int = 401) -> float:
    Q = 1.602176634e-19
    EPS0 = 8.8541878128e-12
    HBAR = 1.054571817e-34
    ME = 9.1093837015e-31
    if not (sheet_density >= 0.0):
        raise ValueError('sheet density must be non-negative')
    if not (effective_mass_ratio > 0.0):
        raise ValueError('effective mass ratio must be positive')
    if not (box_length > 0.0):
        raise ValueError('box length must be positive')
    m_pts = int(grid_points)
    if m_pts < 5:
        raise ValueError('grid needs at least five points')
    m_eff = effective_mass_ratio * ME
    eps = gan['eps_r'] * EPS0
    h = box_length / (m_pts - 1)
    kin = HBAR * HBAR / (2.0 * m_eff * h * h)
    diag0 = 2.0 * kin * np.ones(m_pts - 2)
    off = -kin * np.ones(m_pts - 3)
    if sheet_density == 0.0:
        levels, _ = eigh_tridiagonal(diag0, off, select='i', select_range=(0, 0))
        return float(levels[0] / Q)
    tol = 1e-10 * Q
    beta = 0.3
    potential = np.zeros(m_pts)
    hist_x, hist_r = [], []
    e_f_old = None
    for _ in range(3000):
        new_potential, e_f, levels, k = _sp_sweep(potential, sheet_density, m_eff, eps, h, m_pts, diag0, off)
        resid = new_potential - potential
        if e_f_old is not None and abs(e_f - e_f_old) < tol and np.max(np.abs(resid)) < tol:
            return float(e_f / Q)
        e_f_old = e_f
        hist_x.append(potential.copy())
        hist_r.append(resid.copy())
        hist_x, hist_r = hist_x[-6:], hist_r[-6:]
        step = beta * resid
        if len(hist_r) > 1:      # Anderson acceleration on the last few sweeps
            d_r = np.array([hist_r[i + 1] - hist_r[i] for i in range(len(hist_r) - 1)]).T
            d_x = np.array([hist_x[i + 1] - hist_x[i] for i in range(len(hist_x) - 1)]).T
            with np.errstate(all='ignore'):
                gamma = np.linalg.lstsq(d_r, resid, rcond=None)[0]
                corr = (d_x + beta * d_r) @ gamma
            if np.all(np.isfinite(corr)):
                step = step - corr
        potential = potential + step
    raise RuntimeError('Schrodinger-Poisson iteration did not converge')

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    s = "import numpy as np\nfrom scipy.linalg import eigh_tridiagonal\nGAN = {'a': 3.189e-10, 'psp': -0.034, 'e31': -0.34, 'e33': 0.67, 'c13': 106.0e9, 'c33': 398.0e9, 'eps_r': 8.9}\nALN = {'a': 3.112e-10, 'psp': -0.090, 'e31': -0.53, 'e33': 1.50, 'c13': 108.0e9, 'c33': 373.0e9, 'eps_r': 8.5}\n"
    return [
        {"setup": s,
         "call": "round(channel_fermi_level(6.0304e16, 0.22, GAN), 6)",
         "gold_call": "round(_oracle_channel_fermi_level(6.0304e16, 0.22, GAN), 6)"},
        {"setup": s,
         "call": "round(channel_fermi_level(1.0e15, 0.22, GAN), 6)",
         "gold_call": "round(_oracle_channel_fermi_level(1.0e15, 0.22, GAN), 6)"},
        {"setup": s,
         "call": "round(channel_fermi_level(2.0e17, 0.20, GAN), 6)",
         "gold_call": "round(_oracle_channel_fermi_level(2.0e17, 0.20, GAN), 6)"},
        {"setup": s,
         "call": "round(channel_fermi_level(9.0e16, 0.22, GAN, 30.0e-9, 301), 6)",
         "gold_call": "round(_oracle_channel_fermi_level(9.0e16, 0.22, GAN, 30.0e-9, 301), 6)"},
        {"setup": s,
         "call": "round(channel_fermi_level(0.0, 0.22, GAN), 8)",
         "gold_call": "round(_oracle_channel_fermi_level(0.0, 0.22, GAN), 8)"},
        {"setup": s + "def run_model():\n    try:\n        channel_fermi_level(-1.0e16, 0.22, GAN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef run_oracle():\n    try:\n        _oracle_channel_fermi_level(-1.0e16, 0.22, GAN)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n",
         "call": "run_model()", "gold_call": "run_oracle()"},
    ]

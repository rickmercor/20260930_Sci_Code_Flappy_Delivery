"""
Run the whole analysis on one monolayer and report it. Build the geometry and the zone once and pass them down. Take the band gap on the declared grid; the lowest vertical transition at the zone centre; the effective masses of the highest occupied band at the zone centre towards the edge midpoint and towards the corner, and of the lowest empty band at the edge midpoint towards the zone centre and on towards the corner, all with the declared fit spacing and sample count; the single mass describing that anisotropic lower valley as a whole; then build the transition list along the first Cartesian axis and from it both pi times the zone-averaged quantum metric and, on a uniform grid of n_omega photon energies spanning the declared window, the conductivity and its integrated spectral weight; and finally report how far apart those last two are, as a percentage of the first. The source fixes a convention here that the natural reading does not; follow the source.

The source proposes this comparison as a test that the optical response and the geometry of the occupied states are two views of the same thing. Reporting the gaps and the masses alongside it places the parametrisation against the source's own tabulated electronic structure, so a reader can tell whether the optical agreement rests on a band structure that is itself right. The vertical transition at the zone centre is also what says where the frequency window may honestly begin.

Returns
-------
ndarray of shape (10,): the band gap in eV; the lowest vertical transition energy at the zone centre in eV; the two masses of the highest occupied band at the zone centre, towards the edge midpoint then towards the corner; the two masses of the lowest empty band at the edge midpoint, towards the zone centre then towards the corner; the single mass describing that valley; pi times the zone-averaged quantum metric in Angstrom^2; the integrated optical spectral weight in Angstrom^2; and the signed percentage by which the second exceeds the first.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tb_optics_report(a: float, theta: float, params: "np.ndarray", n_grid: int, n_occupied: int, eta: float, omega_min: float, omega_max: float, n_omega: int, dk: float, n_points: int) -> "np.ndarray":
    """Run the whole analysis on one monolayer and report it. Build the geometry and the zone once and pass them down. Take the band gap on the declared grid; the lowest vertical transition at the zone centre; the effective masses of the highest occupied band at the zone centre towards the edge midpoint and towards the corner, and of the lowest empty band at the edge midpoint towards the zone centre and on towards the corner, all with the declared fit spacing and sample count; the single mass describing that anisotropic lower valley as a whole; then build the transition list along the first Cartesian axis and from it both pi times the zone-averaged quantum metric and, on a uniform grid of n_omega photon energies spanning the declared window, the conductivity and its integrated spectral weight; and finally report how far apart those last two are, as a percentage of the first. The source fixes a convention here that the natural reading does not; follow the source.

    Returns
    -------
    ndarray of shape (10,): the band gap in eV; the lowest vertical transition energy at the zone centre in eV; the two masses of the highest occupied band at the zone centre, towards the edge midpoint then towards the corner; the two masses of the lowest empty band at the edge midpoint, towards the zone centre then towards the corner; the single mass describing that valley; pi times the zone-averaged quantum metric in Angstrom^2; the integrated optical spectral weight in Angstrom^2; and the signed percentage by which the second exceeds the first.

    Raises
    ------
    ValueError: if the frequency window does not satisfy 0 < omega_min < omega_max, if n_omega is less than two, if the quantum metric vanishes, or if any forwarded value is invalid.
    """
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

HBAR2_OVER_ME = 7.6199682     # hbar^2 / m_e in eV Angstrom^2


def _oracle_tb_optics_report(a: float, theta: float, params: "np.ndarray", n_grid: int, n_occupied: int, eta: float,
                             omega_min: float, omega_max: float, n_omega: int, dk: float, n_points: int) -> "np.ndarray":
    if not (0.0 < omega_min < omega_max):
        raise ValueError("the frequency window must satisfy 0 < omega_min < omega_max")
    if n_omega < 2:
        raise ValueError("n_omega must be at least 2")
    # built once here and passed down, so every later step sees the same geometry and the
    # same zone rather than rebuilding them from a and theta.
    vec = _oracle_hopping_vectors(a, theta)
    bz = _oracle_brillouin_zone(a)
    G, M, K = bz[2], bz[3], bz[4]
    gap = _oracle_indirect_gap(vec, bz, params, n_grid, n_occupied)
    # the lowest VERTICAL transition, which is the absorption edge and therefore what fixes
    # where the frequency window below may honestly begin.
    e0 = np.linalg.eigvalsh(_oracle_bloch_hamiltonian(G, vec, params, -1))
    direct = e0[n_occupied] - e0[n_occupied - 1]
    mv1 = _oracle_effective_mass(vec, params, n_occupied - 1, G, M - G, dk, n_points)
    mv2 = _oracle_effective_mass(vec, params, n_occupied - 1, G, K - G, dk, n_points)
    mc1 = _oracle_effective_mass(vec, params, n_occupied, M, G - M, dk, n_points)
    mc2 = _oracle_effective_mass(vec, params, n_occupied, M, K - M, dk, n_points)
    # CONVENTION (source, Table V): the dos mass is the GEOMETRIC mean of the two principal
    # masses, not the arithmetic one.
    mdos = np.sqrt(mc1 * mc2)
    tr = _oracle_interband_transitions(vec, bz, params, n_grid, n_occupied, 0)
    piG = np.pi * _oracle_quantum_metric(tr, n_grid)
    w = np.linspace(omega_min, omega_max, n_omega)
    sig = _oracle_optical_conductivity(w, tr, n_grid, eta)
    wopt = _oracle_spectral_weight(w, sig)
    if piG <= 0.0:
        raise ValueError("the quantum metric vanished; the sum rule cannot be tested")
    return np.array([gap, direct, mv1, mv2, mc1, mc2, mdos, piG, wopt,
                     100.0 * (wopt - piG) / piG])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])",
         "call": "tb_optics_report(3.69, 0.5965, ZR, 30, 6, 0.01, 1.0, 14.0, 13001, 0.01, 7)",
         "gold_call": "_oracle_tb_optics_report(3.69, 0.5965, ZR, 30, 6, 0.01, 1.0, 14.0, 13001, 0.01, 7)"},   # normal
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])",
         "call": "tb_optics_report(3.69, 0.5965, ZR, 6, 6, 0.05, 1.0, 14.0, 1301, 0.02, 3)",
         "gold_call": "_oracle_tb_optics_report(3.69, 0.5965, ZR, 6, 6, 0.05, 1.0, 14.0, 1301, 0.02, 3)"},   # boundary
        {"setup": "import numpy as np\nHF=np.array([-1.1472,-2.0304,1.6231,-5.6310,-4.7654,0.1490,0.1999,-0.1820,1.0371,-0.2387,-1.6199,0.9885,-0.2041,-0.0082,0.0909,0.0122,-0.0205])",
         "call": "tb_optics_report(3.652, 0.5993, HF, 12, 6, 0.001, 1.5, 16.0, 2901, 0.005, 9)",
         "gold_call": "_oracle_tb_optics_report(3.652, 0.5993, HF, 12, 6, 0.001, 1.5, 16.0, 2901, 0.005, 9)"},   # edge
        {"setup": "import numpy as np\nZR=np.array([-1.5007,-2.2542,1.6748,-5.3555,-4.5714,0.0188,0.1288,-0.1255,0.9063,-0.1974,-1.4874,0.9276,-0.1167,-0.0165,0.0605,-0.0078,-0.0109])\ndef _c():\n    try:\n        tb_optics_report(3.69, 0.5965, ZR, 6, 6, 0.01, 14.0, 1.0, 1301, 0.01, 7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\ndef _g():\n    try:\n        _oracle_tb_optics_report(3.69, 0.5965, ZR, 6, 6, 0.01, 14.0, 1.0, 1301, 0.01, 7)\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2",
         "call": "_c()",
         "gold_call": "_g()"},   # invalid input
    ]

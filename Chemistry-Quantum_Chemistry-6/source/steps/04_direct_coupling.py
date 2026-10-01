"""
Compute the direct electronic coupling between the reactant state, which carries the excitation on the donor, and the product state, which carries it on the acceptor.

The direct coupling has a long-ranged part and a short-ranged part. The long-ranged part is the Coulomb interaction of the two transition densities, taken in its leading dipole-dipole form. With d_D and d_A the donor and acceptor transition dipoles, R the length of the donor to acceptor displacement vector and a hat marking a unit vector, it is

J = kappa * |d_D| * |d_A| / R**3 * dipole_unit_factor,

kappa = d_D_hat . d_A_hat - 3 * (d_D_hat . R_hat) * (d_A_hat . R_hat),

where kappa is the dimensionless orientation factor and dipole_unit_factor is the energy in eV of one debye squared per angstrom cubed. The orientation factor enters the coupling to the first power; its square belongs to a rate. The short-ranged part is the exchange interaction, whose magnitude decays exponentially with the same distance,

K = k0 * exp(-beta * R).

How the two parts combine depends on the spin of the excitation. For a transfer between singlet excitations the two determinants differ by a spin-paired exchange of orbitals and the direct coupling is J - K. For a transfer between triplet excitations the Coulomb integral vanishes by spin orthogonality and the direct coupling is -K. The two parts can be comparable in size, so the combination rule decides whether they reinforce or cancel.

Returns
-------
float, the direct electronic coupling between the reactant and product states in eV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def direct_coupling(d_donor: np.ndarray, d_acceptor: np.ndarray, r_vector: np.ndarray,
                    dipole_unit_factor: float, k0: float, beta: float,
                    spin_case: str) -> float:
    '''Direct electronic coupling between the reactant and product states.

    Parameters
    ----------
    d_donor : np.ndarray
        (3,) transition dipole of the donor excitation in debye.
    d_acceptor : np.ndarray
        (3,) transition dipole of the acceptor excitation in debye.
    r_vector : np.ndarray
        (3,) donor to acceptor displacement in angstrom.
    dipole_unit_factor : float
        Positive energy in eV of one debye squared per angstrom cubed.
    k0 : float
        Positive exchange prefactor in eV.
    beta : float
        Positive exchange decay constant in inverse angstrom.
    spin_case : str
        "singlet" or "triplet".

    Returns
    -------
    coupling : float
        The direct electronic coupling in eV: J - K for a singlet transfer and
        -K for a triplet transfer.

    Raises
    ------
    ValueError
        If a dipole or displacement vector is not a finite non-zero vector of
        length three, if dipole_unit_factor, k0 or beta is not positive and
        finite, or if spin_case is neither "singlet" nor "triplet".
    '''
    return coupling  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _unit_vector(v, name):
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f"{name} must be a finite vector of length three")
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        raise ValueError(f"{name} must not be the zero vector")
    return arr / norm, norm


def _oracle_direct_coupling(d_donor: np.ndarray, d_acceptor: np.ndarray,
                            r_vector: np.ndarray, dipole_unit_factor: float,
                            k0: float, beta: float, spin_case: str) -> float:
    """Reference implementation."""
    ud, nd = _unit_vector(d_donor, "d_donor")
    ua, na = _unit_vector(d_acceptor, "d_acceptor")
    ur, r = _unit_vector(r_vector, "r_vector")
    for value, name in ((dipole_unit_factor, "dipole_unit_factor"), (k0, "k0"),
                        (beta, "beta")):
        if not (np.isfinite(float(value)) and float(value) > 0.0):
            raise ValueError(f"{name} must be positive and finite")
    if not isinstance(spin_case, str) or spin_case not in ("singlet", "triplet"):
        raise ValueError('spin_case must be "singlet" or "triplet"')
    kappa = float(ud @ ua - 3.0 * (ud @ ur) * (ua @ ur))
    coulomb = kappa * nd * na / r ** 3 * float(dipole_unit_factor)
    exchange = float(k0) * np.exp(-float(beta) * r)
    if spin_case == "singlet":
        return float(coulomb - exchange)
    return float(-exchange)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# normal: a skewed donor-acceptor geometry, singlet excitations
dd = np.array([1.85, 0.60, -0.40])
da = np.array([-0.95, 1.55, 0.35])
rv = np.array([5.20, 2.40, -1.10])
""",
            "call": 'direct_coupling(dd, da, rv, 0.624151, 17.55, 1.495, "singlet")',
            "gold_call": '_oracle_direct_coupling(dd, da, rv, 0.624151, 17.55, 1.495, "singlet")',
        },
        {
            "setup": """import numpy as np
# normal: the same geometry with triplet excitations, where only exchange survives
dd = np.array([1.85, 0.60, -0.40])
da = np.array([-0.95, 1.55, 0.35])
rv = np.array([5.20, 2.40, -1.10])
""",
            "call": 'direct_coupling(dd, da, rv, 0.624151, 17.55, 1.495, "triplet")',
            "gold_call": '_oracle_direct_coupling(dd, da, rv, 0.624151, 17.55, 1.495, "triplet")',
        },
        {
            "setup": """import numpy as np
# boundary: both dipoles lie along the separation, the head-to-tail geometry
# whose orientation factor is -2
dd = np.array([0.0, 0.0, 2.10])
da = np.array([0.0, 0.0, 1.30])
rv = np.array([0.0, 0.0, 6.50])
""",
            "call": 'direct_coupling(dd, da, rv, 0.624151, 12.0, 1.20, "singlet")',
            "gold_call": '_oracle_direct_coupling(dd, da, rv, 0.624151, 12.0, 1.20, "singlet")',
        },
        {
            "setup": """import numpy as np
# edge: dipoles perpendicular to each other and to the separation, so the
# Coulomb part vanishes and the singlet coupling is the exchange part alone
dd = np.array([1.40, 0.0, 0.0])
da = np.array([0.0, 0.0, 0.90])
rv = np.array([0.0, 4.80, 0.0])
""",
            "call": 'direct_coupling(dd, da, rv, 0.624151, 17.55, 1.495, "singlet")',
            "gold_call": '_oracle_direct_coupling(dd, da, rv, 0.624151, 17.55, 1.495, "singlet")',
        },
        {
            "setup": """import numpy as np
# edge: a long separation, where the exchange part has decayed away and the
# coupling is the dipole-dipole term
dd = np.array([2.50, -0.30, 0.80])
da = np.array([1.10, 1.90, -0.60])
rv = np.array([-14.0, 9.0, 3.0])
""",
            "call": 'direct_coupling(dd, da, rv, 0.624151, 17.55, 1.495, "singlet")',
            "gold_call": '_oracle_direct_coupling(dd, da, rv, 0.624151, 17.55, 1.495, "singlet")',
        },
        {
            "setup": """import numpy as np
dd = np.array([0.0, 0.0, 0.0])
da = np.array([-0.95, 1.55, 0.35])
rv = np.array([5.20, 2.40, -1.10])
def run(f):
    try:
        f(dd, da, rv, 0.624151, 17.55, 1.495, "singlet"); return 0      # zero dipole
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(direct_coupling)",
            "gold_call": "run(_oracle_direct_coupling)",
        },
        {
            "setup": """import numpy as np
dd = np.array([1.85, 0.60, -0.40])
da = np.array([-0.95, 1.55, 0.35])
rv = np.array([5.20, 2.40, -1.10])
def run(f):
    try:
        f(dd, da, rv, 0.624151, 17.55, 1.495, "quintet"); return 0      # unknown spin case
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(direct_coupling)",
            "gold_call": "run(_oracle_direct_coupling)",
        },
        {
            "setup": """import numpy as np
dd = np.array([1.85, 0.60, -0.40])
da = np.array([-0.95, 1.55, 0.35])
rv = np.array([5.20, 2.40, -1.10])
def run(f):
    try:
        f(dd, da, rv, 0.624151, -19.5, 1.495, "singlet"); return 0      # negative prefactor
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(direct_coupling)",
            "gold_call": "run(_oracle_direct_coupling)",
        },
    ]

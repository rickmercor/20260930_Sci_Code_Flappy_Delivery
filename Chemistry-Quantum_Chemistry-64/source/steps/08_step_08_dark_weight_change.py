"""
Step 08 - Pulse-induced change in the weight of a dark determinant of a hydrogen chain (orchestrator).

Pulse-induced change in the configuration weight of a singly excited
determinant of a hydrogen chain (orchestrator).

The chain has n_atoms hydrogen atoms (nuclear charge 1) on the z axis with equal
nearest-neighbour spacing, centred on the origin: atom k sits at
z_k = (k - (n_atoms - 1) / 2) * spacing, k = 0 .. n_atoms - 1. Every atom
carries the given contracted s shells. With an even number of atoms the
closed-shell restricted Hartree-Fock determinant with n_atoms / 2 doubly
occupied canonical orbitals is the reference (the orbital step fixes the SCF
and the orbital signs).

The correlated treatment works in spin orbitals built from the canonical
spatial orbitals by interleaving the spins: spin orbital 2p is spatial orbital
p with spin alpha and 2p + 1 is the same orbital with spin beta. The reference
occupies spin orbitals 0 .. n_atoms - 1. In this basis the core Hamiltonian
(kinetic plus nuclear attraction) and the matrix of the electronic z coordinate
are diagonal in spin, the physicist integrals are
<pq|rs> = (p r | q s) when p and r share a spin and q and s share a spin, and
zero otherwise, with (..|..) the spatial repulsion integrals over the orbitals
of the spin orbitals, the antisymmetrised integrals are
<pq||rs> = <pq|rs> - <pq|sr>, and the Fock matrix of the reference is the core
Hamiltonian plus the sum over occupied spin orbitals i of <pi||qi>.

The coupled-cluster ground state is the initial condition at
t_start = -t_foot / 2. The pulse of the propagation step is centred at time 0,
so the envelope switches on at t_start and off at +t_foot / 2, and the state is
propagated over the whole pulse, n_steps = t_foot / dt steps (this ratio must
be an integer). Configuration weights are evaluated for the ground state and
for the final state.

The requested quantity concerns the spatial excitation from canonical orbital
occ (occupied) to canonical orbital vir (virtual), counted from zero. Its
weight is the sum of the weights of the two singly excited determinants that
promote an alpha electron from occ to vir or a beta electron from occ to vir.
The function returns that weight at the end of the pulse minus its value in the
coupled-cluster ground state.

Returns
-------
float, the pulse-induced change of the spin-summed weight of the excitation occ -> vir (dimensionless)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def dark_weight_change(n_atoms: int, spacing: float, exponents: list, coefficients: list, e0: float, omega: float, t_foot: float, n_env: int, dt: float, occ: int, vir: int) -> float:
    '''Change of the spin-summed configuration weight of the excitation occ -> vir across a laser pulse.

    Parameters
    ----------
    n_atoms : int
        Even number of hydrogen atoms in the chain, at least 2.
    spacing : float
        Nearest-neighbour distance in bohr, strictly positive.
    exponents : list
        One 1D array of primitive exponents per contracted s shell (the same
        shells on every atom).
    coefficients : list
        One 1D array of contraction coefficients per shell (normalised
        primitives), in the same order as exponents.
    e0 : float
        Peak field amplitude in atomic units.
    omega : float
        Carrier angular frequency in hartree, non-negative.
    t_foot : float
        Foot-to-foot duration of the envelope in atomic units of time; the pulse
        is centred at time 0.
    n_env : int
        Positive integer exponent of the cosine envelope.
    dt : float
        Runge-Kutta step; t_foot / dt must be an integer within a relative
        tolerance of 1e-9.
    occ : int
        Occupied canonical spatial orbital, 0 <= occ < n_atoms / 2.
    vir : int
        Virtual canonical spatial orbital, n_atoms / 2 <= vir < n_bf.

    Returns
    -------
    delta_weight : float
        Weight of the two spin-orbital determinants occ -> vir at t = t_foot / 2
        minus their weight in the coupled-cluster ground state (dimensionless).

    Raises
    ------
    ValueError
        If n_atoms is not an even integer of at least 2, if spacing is not a
        finite positive number, if the basis or pulse parameters violate the
        conditions of the earlier steps, if t_foot / dt is not an integer, or if
        occ or vir is out of its range.
    '''
    return delta_weight

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _spin_orbital_operators(C, hcore, zmat, eri, n_occ):
    """Spin-orbital Fock matrix, z matrix and <pq||rs> in the canonical orbitals (interleaved spins)."""
    n = C.shape[1]
    h_mo = C.T @ hcore @ C
    z_mo = C.T @ zmat @ C
    e_mo = np.einsum('pqrs,pi,qj,rk,sl->ijkl', eri, C, C, C, C, optimize=True)
    nso = 2 * n
    sp = np.arange(nso) // 2
    same = (np.arange(nso)[:, None] % 2 == np.arange(nso)[None, :] % 2).astype(float)
    h = h_mo[np.ix_(sp, sp)] * same
    z = z_mo[np.ix_(sp, sp)] * same
    phys = e_mo[np.ix_(sp, sp, sp, sp)].transpose(0, 2, 1, 3) * same[:, None, :, None] * same[None, :, None, :]
    g = phys - phys.transpose(0, 1, 3, 2)
    ne = 2 * n_occ
    f = h + np.einsum('piqi->pq', g[:, :ne, :, :ne])
    return f, z, g


def _oracle_dark_weight_change(n_atoms: int, spacing: float, exponents: list, coefficients: list, e0: float, omega: float, t_foot: float, n_env: int, dt: float, occ: int, vir: int) -> float:
    if isinstance(n_atoms, bool) or not isinstance(n_atoms, (int, np.integer)) or n_atoms < 2 or n_atoms % 2:
        raise ValueError("n_atoms must be an even integer of at least 2")
    if isinstance(spacing, bool) or not isinstance(spacing, (int, float, np.integer, np.floating)) or not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("spacing must be a finite positive number")
    for name, val in (('t_foot', t_foot), ('dt', dt)):
        if isinstance(val, bool) or not isinstance(val, (int, float, np.integer, np.floating)) or not np.isfinite(val) or val <= 0.0:
            raise ValueError("%s must be a finite positive number" % name)
    ratio = float(t_foot) / float(dt)
    n_steps = int(round(ratio))
    if n_steps < 1 or abs(ratio - n_steps) > 1e-9 * max(1.0, ratio):
        raise ValueError("t_foot / dt must be a positive integer")
    n_occ = int(n_atoms) // 2
    zk = (np.arange(n_atoms) - 0.5 * (n_atoms - 1)) * float(spacing)
    coords = np.stack([np.zeros(n_atoms), np.zeros(n_atoms), zk], axis=1)
    charges = np.ones(n_atoms)
    ints = _oracle_s_type_one_electron_integrals(coords, charges, exponents, coefficients)
    eri = _oracle_s_type_electron_repulsion(coords, exponents, coefficients)
    orbitals = _oracle_rhf_canonical_orbitals(coords, charges, exponents, coefficients, n_occ)
    n_bf = orbitals.shape[1]
    for name, val in (('occ', occ), ('vir', vir)):
        if isinstance(val, bool) or not isinstance(val, (int, np.integer)):
            raise ValueError("%s must be an integer" % name)
    if not 0 <= occ < n_occ or not n_occ <= vir < n_bf:
        raise ValueError("occ must be occupied and vir virtual in the reference")
    f, z, g = _spin_orbital_operators(orbitals[1:], ints[1] + ints[2], ints[3], eri, n_occ)
    ne, nso = 2 * n_occ, 2 * n_bf
    y0 = _oracle_ccsd_lambda_ground_state(f, g, ne)
    yf = _oracle_tdccsd_propagate(f, z, g, ne, y0, e0, omega, 0.0, t_foot, n_env, -0.5 * float(t_foot), float(dt), n_steps)
    L = yf.size // 2
    w_start = _oracle_configuration_weights(y0.astype(complex), ne, nso)
    w_end = _oracle_configuration_weights(yf[:L] + 1j * yf[L:], ne, nso)
    v = nso - ne
    cols = [1 + (2 * occ) * v + (2 * vir - ne), 1 + (2 * occ + 1) * v + (2 * vir + 1 - ne)]
    return float(np.sum(w_end[cols]) - np.sum(w_start[cols]))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: four-atom chain, minimal basis, a short pulse near the first bright transition ---
        {
            "setup": """import numpy as np
import copy
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
""",
            "call": "dark_weight_change(4, 1.6, copy.deepcopy(exponents), copy.deepcopy(coefficients), 0.06, 0.6, 10.0, 4, 0.1, 0, 2)",
            "gold_call": "_oracle_dark_weight_change(4, 1.6, copy.deepcopy(exponents), copy.deepcopy(coefficients), 0.06, 0.6, 10.0, 4, 0.1, 0, 2)",
            "tol": 1e-9,
        },
        # --- Boundary: no field, so the state stays the ground state and the change vanishes ---
        {
            "setup": """import numpy as np
import copy
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
""",
            "call": "dark_weight_change(4, 1.6, copy.deepcopy(exponents), copy.deepcopy(coefficients), 0.0, 0.6, 4.0, 2, 0.1, 1, 3)",
            "gold_call": "_oracle_dark_weight_change(4, 1.6, copy.deepcopy(exponents), copy.deepcopy(coefficients), 0.0, 0.6, 4.0, 2, 0.1, 1, 3)",
            "tol": 1e-9,
        },
        # --- Edge: two electrons (where the weights are exact populations), split-valence basis, strong pulse ---
        {
            "setup": """import numpy as np
import copy
exponents = [np.array([18.7311370, 2.8253944, 0.6401217]), np.array([0.1612778])]
coefficients = [np.array([0.03349460, 0.23472695, 0.81375733]), np.array([1.0])]
""",
            "call": "dark_weight_change(2, 1.4, copy.deepcopy(exponents), copy.deepcopy(coefficients), 0.25, 0.55, 8.0, 6, 0.05, 0, 3)",
            "gold_call": "_oracle_dark_weight_change(2, 1.4, copy.deepcopy(exponents), copy.deepcopy(coefficients), 0.25, 0.55, 8.0, 6, 0.05, 0, 3)",
            "tol": 1e-9,
        },
        # --- Invalid: an odd number of atoms has no closed-shell reference ---
        {
            "setup": """import numpy as np
import copy
exponents = [np.array([3.42525091, 0.62391373, 0.16885540])]
coefficients = [np.array([0.15432897, 0.53532814, 0.44463454])]
def run_model():
    try:
        dark_weight_change(3, 1.6, copy.deepcopy(exponents), copy.deepcopy(coefficients), 0.05, 0.5, 4.0, 2, 0.1, 0, 1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_dark_weight_change(3, 1.6, copy.deepcopy(exponents), copy.deepcopy(coefficients), 0.05, 0.5, 4.0, 2, 0.1, 0, 1)
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

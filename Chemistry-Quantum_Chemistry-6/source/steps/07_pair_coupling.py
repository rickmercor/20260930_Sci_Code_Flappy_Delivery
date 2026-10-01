"""
Assemble the total vibronic coupling of one reactant-product pair at that pair's own crossing point: the direct interaction plus the whole sum over pathways through the unpopulated states.

In the electronically nonadiabatic limit a vibronic matrix element is the electronic coupling between two diabatic electronic states, taken as independent of the proton coordinate, times the overlap of the two proton vibrational functions. Functions belonging to different electronic states solve different Schroedinger equations and are not orthogonal, so every such overlap is generally non-zero; each one is the sum over the grid of the product of the two functions times the grid spacing. The functions of the unpopulated states are those of the crossing geometry, because their proton potentials move with the bath, while the reactant and product functions are the same everywhere.

The direct part of the coupling is the direct electronic coupling times the overlap of the pair's own two functions. The indirect part collects every pathway that leaves the reactant level, passes through one or more unpopulated vibronic states in succession and arrives on the product level. With x and y running over the unpopulated vibronic states ordered by virtual_states and then by level,

L[x] = C(I, j) * S(I mu, j xi),    R[y] = C(k, II) * S(k eta, II nu),

W[x, y] = C(j, k) * S(j xi, k eta) for j != k and zero when j and k coincide, and D the diagonal matrix of the pair's energy denominators at its crossing point q, each one the vibronic energy of the product level of the pair minus that of the visited state with both carried up their bath wells,

D(j, xi) = [E(II, nu) + (k / 2) * (q - q_II)**2] - [E(j, xi; q) + (k / 2) * (q - q_j)**2],

where q_II = (2 * lambda / k) ** 0.5 is the product displacement, E(II, nu) and E(j, xi; q) are both measured at the bottom of their own bath wells, and E(j, xi; q) belongs to the tilted potential of the crossing geometry. A pathway through one state contributes L D^-1 R, through two states L D^-1 W D^-1 R, and so on, so the pathways of every length sum to

V_indirect = L (D - W)^-1 R,

which exists only while one pass through the manifold is contractive, that is while the largest eigenvalue modulus of D^-1 W stays below one. A pair whose series does not converge makes the assembly invalid and is an error rather than a number, and so is an energy denominator that is effectively zero. Every retained level of every unpopulated state enters, a pathway may revisit a state after passing through another one, and a pathway entering on one state and leaving from another differs from its reverse.

Returns
-------
float, the total vibronic coupling of the pair in eV at its crossing point
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pair_coupling(crossing: float, reorganization: float, force_constant: float,
                  bath_minima: dict, e_product_level: float, virtual_levels: dict,
                  virtual_wavefunctions: dict, reactant_wavefunction: np.ndarray,
                  product_wavefunction: np.ndarray, spacing: float, virtual_states: tuple,
                  electronic_couplings: dict, direct: float) -> float:
    '''Total vibronic coupling of one pair at its crossing point.

    Parameters
    ----------
    crossing : float
        Bath coordinate of the pair's crossing point.
    reorganization : float
        Positive reorganisation energy lambda in eV.
    force_constant : float
        Positive bath force constant k in eV.
    bath_minima : dict
        Maps every label in virtual_states to that state's bath minimum.
    e_product_level : float
        Vibronic energy in eV of the product level of the pair at the bottom
        of the product bath well.
    virtual_levels : dict
        Maps every label in virtual_states to a one-dimensional np.ndarray of
        that state's vibronic energies in eV at the crossing geometry,
        measured at the bottom of its own bath well.
    virtual_wavefunctions : dict
        Maps every label in virtual_states to an np.ndarray of shape
        (n_grid, n_levels) whose columns are that state's proton vibrational
        functions at the crossing geometry.
    reactant_wavefunction : np.ndarray
        (n_grid,) proton vibrational function of the reactant level of the pair.
    product_wavefunction : np.ndarray
        (n_grid,) proton vibrational function of the product level of the pair.
    spacing : float
        Positive grid spacing in angstrom.
    virtual_states : tuple of str
        Labels of the unpopulated electronic states.
    electronic_couplings : dict
        Maps a pair of state labels to their electronic coupling in eV; one of
        (a, b) and (b, a) is enough. The couplings of the reactant and of the
        product with every unpopulated state, and of every pair of distinct
        unpopulated states, must be present.
    direct : float
        Direct electronic coupling between the reactant and product states in eV.

    Returns
    -------
    coupling : float
        Total vibronic coupling of the pair in eV.

    Raises
    ------
    ValueError
        If any input is missing, malformed or out of range, if the reactant or
        product function does not live on the shared grid, if an energy
        denominator is smaller than 1e-12 eV in magnitude, or if the largest
        eigenvalue modulus of one pass through the manifold is one or more.
    '''
    return coupling  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_pair_coupling(crossing: float, reorganization: float, force_constant: float,
                          bath_minima: dict, e_product_level: float, virtual_levels: dict,
                          virtual_wavefunctions: dict, reactant_wavefunction: np.ndarray,
                          product_wavefunction: np.ndarray, spacing: float,
                          virtual_states: tuple, electronic_couplings: dict,
                          direct: float) -> float:
    """Reference implementation."""
    v_direct = float(direct)
    if not np.isfinite(v_direct):
        raise ValueError("direct must be finite")
    radius = _oracle_pathway_convergence(crossing, reorganization, force_constant,
                                         bath_minima, e_product_level, virtual_levels,
                                         virtual_wavefunctions, spacing, virtual_states,
                                         electronic_couplings)
    if radius >= 1.0:
        raise ValueError("the sum over pathways does not converge for this pair")
    denominators, link, basis, waves, h = _pair_manifold(
        crossing, reorganization, force_constant, bath_minima, e_product_level,
        virtual_levels, virtual_wavefunctions, spacing, virtual_states, electronic_couplings)
    reactant = np.asarray(reactant_wavefunction, dtype=float)
    product = np.asarray(product_wavefunction, dtype=float)
    grid = waves[basis[0][0]].shape[0]
    for arr, name in ((reactant, "reactant_wavefunction"), (product, "product_wavefunction")):
        if arr.ndim != 1 or arr.size != grid or not np.isfinite(arr).all():
            raise ValueError(f"{name} must be a finite one-dimensional array on the shared grid")
    left = np.array([_coupling(electronic_couplings, "I", j) * h * float(reactant @ waves[j][:, i])
                     for j, i in basis])
    right = np.array([_coupling(electronic_couplings, j, "II") * h * float(waves[j][:, i] @ product)
                      for j, i in basis])
    indirect = float(left @ np.linalg.solve(np.diag(denominators) - link, right))
    return float(v_direct * h * float(reactant @ product) + indirect)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# normal: three unpopulated states with three levels each, all coupled to one
# another, at a crossing point beyond the product well
dx = 0.009
rng = np.random.default_rng(17)
lev = {"CT1": np.array([3.5822, 3.8095, 3.8624]),
       "CT2": np.array([4.0664, 4.3266, 4.5408]),
       "B": np.array([3.9218, 4.0762, 4.1445])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(41, 3))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
wr = rng.normal(0.0, 1.0, size=41); wr /= np.sqrt((wr ** 2).sum() * dx)
wp = rng.normal(0.0, 1.0, size=41); wp /= np.sqrt((wp ** 2).sum() * dx)
vel = {("I", "CT1"): 0.0180, ("CT1", "II"): -0.0170, ("I", "CT2"): 0.0080,
       ("CT2", "II"): 0.0200, ("I", "B"): 0.0080, ("B", "II"): -0.0160,
       ("CT1", "CT2"): 0.0700, ("B", "CT1"): 0.1100, ("B", "CT2"): -0.0400}
qmin = {"CT1": 1.2, "CT2": 2.0, "B": 0.0}
""",
            "call": 'pair_coupling(0.6923, 0.2735, 0.8, qmin, 3.4302, lev, psi, wr, wp, dx, ("CT1", "CT2", "B"), vel, -7.9e-4)',
            "gold_call": '_oracle_pair_coupling(0.6923, 0.2735, 0.8, qmin, 3.4302, lev, psi, wr, wp, dx, ("CT1", "CT2", "B"), vel, -7.9e-4)',
        },
        {
            "setup": """import numpy as np
# normal: the reactant and product uncoupled from the manifold, so only the
# direct part survives
dx = 0.009
rng = np.random.default_rng(19)
lev = {"CT1": np.array([3.60, 3.80]), "CT2": np.array([4.10, 4.35])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(41, 2))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
wr = rng.normal(0.0, 1.0, size=41); wr /= np.sqrt((wr ** 2).sum() * dx)
wp = rng.normal(0.0, 1.0, size=41); wp /= np.sqrt((wp ** 2).sum() * dx)
vel = {("I", "CT1"): 0.0, ("CT1", "II"): 0.0, ("I", "CT2"): 0.0,
       ("CT2", "II"): 0.0, ("CT1", "CT2"): 0.0600}
qmin = {"CT1": 1.1, "CT2": 1.7}
""",
            "call": 'pair_coupling(0.55, 0.2735, 0.8, qmin, 3.4302, lev, psi, wr, wp, dx, ("CT1", "CT2"), vel, 2.4e-3)',
            "gold_call": '_oracle_pair_coupling(0.55, 0.2735, 0.8, qmin, 3.4302, lev, psi, wr, wp, dx, ("CT1", "CT2"), vel, 2.4e-3)',
        },
        {
            "setup": """import numpy as np
# boundary: a single unpopulated state, so every pathway visits it exactly once
dx = 0.015
rng = np.random.default_rng(5)
lev = {"CT1": np.array([3.66, 3.80, 3.87])}
w = rng.normal(0.0, 1.0, size=(29, 3))
psi = {"CT1": w / np.sqrt((w ** 2).sum(axis=0) * dx)}
wr = rng.normal(0.0, 1.0, size=29); wr /= np.sqrt((wr ** 2).sum() * dx)
wp = rng.normal(0.0, 1.0, size=29); wp /= np.sqrt((wp ** 2).sum() * dx)
vel = {("I", "CT1"): 0.020, ("CT1", "II"): -0.019}
""",
            "call": 'pair_coupling(0.40, 0.30, 1.0, {"CT1": 1.1}, 3.43, lev, psi, wr, wp, dx, ("CT1",), vel, -1.0e-3)',
            "gold_call": '_oracle_pair_coupling(0.40, 0.30, 1.0, {"CT1": 1.1}, 3.43, lev, psi, wr, wp, dx, ("CT1",), vel, -1.0e-3)',
        },
        {
            "setup": """import numpy as np
# edge: a zero direct coupling, so the whole answer is the pathway sum, with
# ladders of different lengths on the unpopulated states
dx = 0.012
rng = np.random.default_rng(29)
lev = {"CT1": np.array([3.60, 3.75]), "CT2": np.array([4.10, 4.35, 4.50, 4.62]),
       "B": np.array([3.90, 4.05, 4.20])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(37, lev[a].size))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
wr = rng.normal(0.0, 1.0, size=37); wr /= np.sqrt((wr ** 2).sum() * dx)
wp = rng.normal(0.0, 1.0, size=37); wp /= np.sqrt((wp ** 2).sum() * dx)
vel = {("I", "CT1"): 0.020, ("CT1", "II"): -0.019, ("I", "CT2"): 0.009,
       ("CT2", "II"): 0.0225, ("I", "B"): 0.007, ("B", "II"): -0.015,
       ("CT1", "CT2"): 0.050, ("B", "CT1"): 0.080, ("B", "CT2"): -0.030}
qmin = {"CT1": 0.7, "CT2": -1.1, "B": 0.3}
""",
            "call": 'pair_coupling(0.48, 0.31, 0.65, qmin, 3.44, lev, psi, wr, wp, dx, ("CT1", "CT2", "B"), vel, 0.0)',
            "gold_call": '_oracle_pair_coupling(0.48, 0.31, 0.65, qmin, 3.44, lev, psi, wr, wp, dx, ("CT1", "CT2", "B"), vel, 0.0)',
        },
        {
            "setup": """import numpy as np
# edge: pathways that must revisit a state, with two unpopulated states coupled
# strongly to each other but only one of them coupled to the reactant
dx = 0.010
rng = np.random.default_rng(31)
lev = {"CT1": np.array([3.62, 3.79]), "CT2": np.array([4.05, 4.28])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(33, 2))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
wr = rng.normal(0.0, 1.0, size=33); wr /= np.sqrt((wr ** 2).sum() * dx)
wp = rng.normal(0.0, 1.0, size=33); wp /= np.sqrt((wp ** 2).sum() * dx)
vel = {("I", "CT1"): 0.025, ("CT1", "II"): 0.0, ("I", "CT2"): 0.0,
       ("CT2", "II"): 0.021, ("CT1", "CT2"): 0.090}
qmin = {"CT1": 1.0, "CT2": 1.6}
""",
            "call": 'pair_coupling(0.60, 0.28, 0.9, qmin, 3.43, lev, psi, wr, wp, dx, ("CT1", "CT2"), vel, 5.0e-4)',
            "gold_call": '_oracle_pair_coupling(0.60, 0.28, 0.9, qmin, 3.43, lev, psi, wr, wp, dx, ("CT1", "CT2"), vel, 5.0e-4)',
        },
        {
            "setup": """import numpy as np
dx = 0.014
rng = np.random.default_rng(41)
lev = {"CT1": np.array([3.60, 3.75]), "CT2": np.array([3.62, 3.78])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(31, 2))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
wr = rng.normal(0.0, 1.0, size=31); wr /= np.sqrt((wr ** 2).sum() * dx)
wp = rng.normal(0.0, 1.0, size=31); wp /= np.sqrt((wp ** 2).sum() * dx)
vel = {("I", "CT1"): 0.02, ("CT1", "II"): -0.019, ("I", "CT2"): 0.009,
       ("CT2", "II"): 0.0225, ("CT1", "CT2"): 2.40}
qmin = {"CT1": 1.0, "CT2": 1.5}
def run(f):
    try:
        f(0.50, 0.28, 0.8, qmin, 3.43, lev, psi, wr, wp, dx, ("CT1", "CT2"), vel, 1e-3)
        return 0                              # the pathway sum diverges
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(pair_coupling)",
            "gold_call": "run(_oracle_pair_coupling)",
        },
        {
            "setup": """import numpy as np
dx = 0.014
rng = np.random.default_rng(43)
lev = {"CT1": np.array([3.60, 3.75])}
w = rng.normal(0.0, 1.0, size=(31, 2))
psi = {"CT1": w / np.sqrt((w ** 2).sum(axis=0) * dx)}
wr = rng.normal(0.0, 1.0, size=31); wr /= np.sqrt((wr ** 2).sum() * dx)
wp = rng.normal(0.0, 1.0, size=31); wp /= np.sqrt((wp ** 2).sum() * dx)
def run(f):
    try:
        f(0.5, 0.28, 0.8, {"CT1": 1.0}, 3.43, lev, psi, wr, wp, dx, ("CT1",),
          {("I", "CT1"): 0.02}, 1e-3)
        return 0                              # the CT1 to product coupling is missing
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(pair_coupling)",
            "gold_call": "run(_oracle_pair_coupling)",
        },
        {
            "setup": """import numpy as np
dx = 0.014
rng = np.random.default_rng(47)
lev = {"CT1": np.array([3.60, 3.75])}
w = rng.normal(0.0, 1.0, size=(31, 2))
psi = {"CT1": w / np.sqrt((w ** 2).sum(axis=0) * dx)}
wr = rng.normal(0.0, 1.0, size=29); wr /= np.sqrt((wr ** 2).sum() * dx)
wp = rng.normal(0.0, 1.0, size=31); wp /= np.sqrt((wp ** 2).sum() * dx)
vel = {("I", "CT1"): 0.02, ("CT1", "II"): 0.01}
def run(f):
    try:
        f(0.5, 0.28, 0.8, {"CT1": 1.0}, 3.43, lev, psi, wr, wp, dx, ("CT1",), vel, 1e-3)
        return 0                              # the reactant function is on a shorter grid
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(pair_coupling)",
            "gold_call": "run(_oracle_pair_coupling)",
        },
    ]

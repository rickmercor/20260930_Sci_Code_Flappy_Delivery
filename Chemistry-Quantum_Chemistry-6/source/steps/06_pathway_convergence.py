"""
Measure, for one reactant-product vibronic pair at its own crossing point, whether repeated passage through the unpopulated electronic states is contractive. The sum over pathways of every length has a finite limit only when it is, so this has to hold before any coupling built from that sum means anything.

At the crossing point q of the pair, the energy denominator of the unpopulated vibronic state (j, xi) is the vibronic energy of the product level of the pair minus that of the visited state, both carried up their bath wells,

D(j, xi) = [E(II, nu) + (k / 2) * (q - q_II)**2] - [E(j, xi; q) + (k / 2) * (q - q_j)**2],

with q_II = (2 * lambda / k) ** 0.5 the product displacement. Because the two energies of the pair are equal at its crossing point, referring the denominators to the reactant level instead gives the same numbers. The levels E(j, xi; q) and the proton vibrational functions of the unpopulated states are the ones belonging to that geometry, which are supplied here; they are not the equilibrium ones, since the proton potentials of those states move with the bath.

One pass through the manifold is the operation M whose rows and columns run over the unpopulated vibronic states x = (j, xi), ordered by virtual_states and then by level:

M[x, y] = C(j, k) * S(j xi, k eta) / D(x)   for j != k,   and 0 when j and k coincide.

Here C(j, k) is the electronic coupling between the two electronic states and S is the overlap of the two proton vibrational functions, the sum over the grid of their product times the grid spacing. Vibronic states of one electronic state are orthogonal and carry no electronic coupling to each other, so no step goes from a state straight back into itself, while a pathway may return to a state it left earlier; every retained level of every unpopulated state takes part.

The number returned is the largest eigenvalue modulus of M. Below one the sum over pathways converges and its limit is the coupling; at one or above the expansion in the electronic couplings has broken down. A denominator that is effectively zero is an error.

Returns
-------
float, the largest eigenvalue modulus of the one-pass operation through the unpopulated manifold for this pair
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def pathway_convergence(crossing: float, reorganization: float, force_constant: float,
                        bath_minima: dict, e_product_level: float, virtual_levels: dict,
                        virtual_wavefunctions: dict, spacing: float, virtual_states: tuple,
                        electronic_couplings: dict) -> float:
    '''Contraction measure of one pass through the unpopulated manifold for one pair.

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
    spacing : float
        Positive grid spacing in angstrom.
    virtual_states : tuple of str
        Labels of the unpopulated electronic states.
    electronic_couplings : dict
        Maps a pair of state labels to their electronic coupling in eV; one of
        (a, b) and (b, a) is enough. Every pair of distinct labels in
        virtual_states must be present.

    Returns
    -------
    radius : float
        Largest eigenvalue modulus of the one-pass operation for this pair.

    Raises
    ------
    ValueError
        If crossing or e_product_level is not finite, if reorganization,
        force_constant or spacing is not positive and finite, if a level set,
        wave function set, bath minimum or coupling is missing or malformed,
        if virtual_states is empty, repeats a label or names the reactant or
        product, or if an energy denominator is smaller than 1e-12 eV in
        magnitude.
    '''
    return radius  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _virtual_labels(virtual_states):
    virtual = tuple(virtual_states)
    if len(virtual) == 0:
        raise ValueError("at least one unpopulated state is required")
    if len(set(virtual)) != len(virtual):
        raise ValueError("virtual_states must not repeat a label")
    if "I" in virtual or "II" in virtual:
        raise ValueError("the reactant and the product cannot be unpopulated states")
    return virtual


def _waves(wavefunctions, label):
    if label not in wavefunctions:
        raise ValueError(f"the proton vibrational functions of state {label} are missing")
    arr = np.asarray(wavefunctions[label], dtype=float)
    if arr.ndim != 2 or arr.size == 0 or not np.isfinite(arr).all():
        raise ValueError(f"the proton vibrational functions of state {label} must be a "
                         "non-empty finite two-dimensional array")
    return arr


def _coupling(electronic_couplings, a, b):
    for key in ((a, b), (b, a)):
        if key in electronic_couplings:
            value = float(electronic_couplings[key])
            if not np.isfinite(value):
                raise ValueError(f"the electronic coupling between {a} and {b} must be finite")
            return value
    raise ValueError(f"the electronic coupling between {a} and {b} is missing")


def _pair_manifold(crossing, reorganization, force_constant, bath_minima, e_product_level,
                   virtual_levels, virtual_wavefunctions, spacing, virtual_states,
                   electronic_couplings):
    """Energy denominators and one-step matrix of the unpopulated manifold for one pair."""
    lam, k, q_product = _bath_pair(reorganization, force_constant)
    q = float(crossing)
    if not np.isfinite(q) or not np.isfinite(float(e_product_level)):
        raise ValueError("crossing and e_product_level must be finite")
    h = float(spacing)
    if not (np.isfinite(h) and h > 0.0):
        raise ValueError("spacing must be positive and finite")
    if not isinstance(bath_minima, dict) or not isinstance(virtual_levels, dict) \
            or not isinstance(virtual_wavefunctions, dict) or not isinstance(electronic_couplings, dict):
        raise ValueError("bath_minima, virtual_levels, virtual_wavefunctions and "
                         "electronic_couplings must be dictionaries")
    virtual = _virtual_labels(virtual_states)
    waves, energies, minima, basis = {}, [], [], []
    for label in virtual:
        if label not in bath_minima:
            raise ValueError(f"the bath minimum of state {label} is missing")
        q_j = float(bath_minima[label])
        if not np.isfinite(q_j):
            raise ValueError(f"the bath minimum of state {label} must be finite")
        if label not in virtual_levels:
            raise ValueError(f"the vibronic energies of state {label} are missing")
        levels = _level_array(virtual_levels[label], f"the vibronic energies of state {label}")
        waves[label] = _waves(virtual_wavefunctions, label)
        if waves[label].shape[1] != levels.size:
            raise ValueError(f"state {label} has a different number of functions and levels")
        energies.extend(levels.tolist())
        minima.extend([q_j] * levels.size)
        basis.extend((label, i) for i in range(levels.size))
    grid = waves[virtual[0]].shape[0]
    if any(w.shape[0] != grid for w in waves.values()):
        raise ValueError("every state's functions must live on the same grid")
    product_energy = float(e_product_level) + 0.5 * k * (q - q_product) ** 2
    denominators = product_energy - (np.array(energies) + 0.5 * k * (q - np.array(minima)) ** 2)
    if np.any(np.abs(denominators) < 1e-12):
        raise ValueError("a vanishing energy denominator was encountered")
    link = np.zeros((len(basis),) * 2)
    for x, (j, i) in enumerate(basis):
        for y, (kk, l) in enumerate(basis):
            if j != kk:
                link[x, y] = _coupling(electronic_couplings, j, kk) * h * float(
                    waves[j][:, i] @ waves[kk][:, l])
    return denominators, link, basis, waves, h


def _oracle_pathway_convergence(crossing: float, reorganization: float, force_constant: float,
                                bath_minima: dict, e_product_level: float,
                                virtual_levels: dict, virtual_wavefunctions: dict,
                                spacing: float, virtual_states: tuple,
                                electronic_couplings: dict) -> float:
    """Reference implementation."""
    denominators, link, _, _, _ = _pair_manifold(
        crossing, reorganization, force_constant, bath_minima, e_product_level,
        virtual_levels, virtual_wavefunctions, spacing, virtual_states, electronic_couplings)
    one_pass = link / denominators[:, None]
    return float(np.max(np.abs(np.linalg.eigvals(one_pass))))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
# normal: three unpopulated states with three levels each at a crossing point
# out beyond the product well
dx = 0.009
rng = np.random.default_rng(11)
lev = {"CT1": np.array([3.5822, 3.8095, 3.8624]),
       "CT2": np.array([4.0664, 4.3266, 4.5408]),
       "B": np.array([3.9218, 4.0762, 4.1445])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(41, 3))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
vel = {("CT1", "CT2"): 0.0700, ("B", "CT1"): 0.1100, ("B", "CT2"): -0.0400}
qmin = {"CT1": 1.2, "CT2": 2.0, "B": 0.0}
""",
            "call": 'pathway_convergence(0.6923, 0.2735, 0.8, qmin, 3.4302, lev, psi, dx, ("CT1", "CT2", "B"), vel)',
            "gold_call": '_oracle_pathway_convergence(0.6923, 0.2735, 0.8, qmin, 3.4302, lev, psi, dx, ("CT1", "CT2", "B"), vel)',
        },
        {
            "setup": """import numpy as np
# normal: the ground pair's crossing point, much closer to the reactant well,
# where the same manifold sits at different denominators
dx = 0.009
rng = np.random.default_rng(11)
lev = {"CT1": np.array([3.6096, 3.8021, 3.8639]),
       "CT2": np.array([4.1104, 4.3706, 4.5406]),
       "B": np.array([3.9014, 4.0918, 4.1329])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(41, 3))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
vel = {("CT1", "CT2"): 0.0700, ("B", "CT1"): 0.1100, ("B", "CT2"): -0.0400}
qmin = {"CT1": 1.2, "CT2": 2.0, "B": 0.0}
""",
            "call": 'pathway_convergence(0.1974, 0.2735, 0.8, qmin, 3.1028, lev, psi, dx, ("CT1", "CT2", "B"), vel)',
            "gold_call": '_oracle_pathway_convergence(0.1974, 0.2735, 0.8, qmin, 3.1028, lev, psi, dx, ("CT1", "CT2", "B"), vel)',
        },
        {
            "setup": """import numpy as np
# boundary: the couplings among the unpopulated states switched off, so nothing
# propagates and the radius is exactly zero
dx = 0.009
rng = np.random.default_rng(3)
lev = {"CT1": np.array([3.66, 3.80, 3.87]), "CT2": np.array([4.20, 4.44, 4.53]),
       "B": np.array([3.89, 4.10, 4.12])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(41, 3))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
vel = {("CT1", "CT2"): 0.0, ("B", "CT1"): 0.0, ("B", "CT2"): 0.0}
qmin = {"CT1": 1.2, "CT2": 2.0, "B": 0.0}
""",
            "call": 'pathway_convergence(0.55, 0.2735, 0.8, qmin, 3.4302, lev, psi, dx, ("CT1", "CT2", "B"), vel)',
            "gold_call": '_oracle_pathway_convergence(0.55, 0.2735, 0.8, qmin, 3.4302, lev, psi, dx, ("CT1", "CT2", "B"), vel)',
        },
        {
            "setup": """import numpy as np
# edge: a single unpopulated state cannot step anywhere, so its manifold is
# not connected to itself at all
dx = 0.015
rng = np.random.default_rng(5)
w = rng.normal(0.0, 1.0, size=(29, 3))
psi = {"CT1": w / np.sqrt((w ** 2).sum(axis=0) * dx)}
lev = {"CT1": np.array([3.66, 3.80, 3.87])}
""",
            "call": 'pathway_convergence(0.40, 0.30, 1.0, {"CT1": 1.1}, 3.43, lev, psi, dx, ("CT1",), {})',
            "gold_call": '_oracle_pathway_convergence(0.40, 0.30, 1.0, {"CT1": 1.1}, 3.43, lev, psi, dx, ("CT1",), {})',
        },
        {
            "setup": """import numpy as np
# edge: ladders of different lengths and the same manifold listed in two
# orders, which must not move the eigenvalues
dx = 0.010
rng = np.random.default_rng(23)
lev = {"CT1": np.array([3.60, 3.75]), "CT2": np.array([4.10, 4.35]),
       "B": np.array([3.90, 4.05, 4.20])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(33, lev[a].size))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
vel = {("CT1", "CT2"): 0.050, ("B", "CT1"): 0.060, ("B", "CT2"): -0.020}
qmin = {"CT1": 1.3, "CT2": 1.8, "B": 0.2}
def both(f):
    return np.array([f(0.62, 0.28, 0.9, qmin, 3.43, lev, psi, dx, ("CT1", "CT2", "B"), vel),
                     f(0.62, 0.28, 0.9, qmin, 3.43, lev, psi, dx, ("B", "CT2", "CT1"), vel)])
""",
            "call": "both(pathway_convergence)",
            "gold_call": "both(_oracle_pathway_convergence)",
        },
        {
            "setup": """import numpy as np
# boundary: couplings inside the manifold large enough that one pass is no
# longer a contraction, so the radius comes back above one and is returned
dx = 0.014
rng = np.random.default_rng(41)
lev = {"CT1": np.array([3.60, 3.75]), "CT2": np.array([3.62, 3.78])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(31, 2))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
vel = {("CT1", "CT2"): 2.40}
qmin = {"CT1": 1.0, "CT2": 1.5}
""",
            "call": 'pathway_convergence(0.50, 0.28, 0.8, qmin, 3.43, lev, psi, dx, ("CT1", "CT2"), vel)',
            "gold_call": '_oracle_pathway_convergence(0.50, 0.28, 0.8, qmin, 3.43, lev, psi, dx, ("CT1", "CT2"), vel)',
        },
        {
            "setup": """import numpy as np
dx = 0.014
rng = np.random.default_rng(3)
lev = {"CT1": np.array([3.43, 3.75]), "CT2": np.array([4.10, 4.35])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(31, 2))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
vel = {("CT1", "CT2"): 0.05}
def run(f):
    try:
        f(0.0, 0.28, 0.8, {"CT1": 0.0, "CT2": 1.5}, 3.43, lev, psi, dx, ("CT1", "CT2"), vel)
        return 0                              # a vanishing energy denominator
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(pathway_convergence)",
            "gold_call": "run(_oracle_pathway_convergence)",
        },
        {
            "setup": """import numpy as np
dx = 0.014
rng = np.random.default_rng(9)
lev = {"CT1": np.array([3.60, 3.75]), "CT2": np.array([4.10, 4.35])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(31, 2))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
def run(f):
    try:
        f(0.5, 0.28, 0.8, {"CT1": 1.0, "CT2": 1.5}, 3.43, lev, psi, dx, ("CT1", "CT2"),
          {("I", "CT1"): 0.02})
        return 0                              # the CT1 to CT2 coupling is missing
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(pathway_convergence)",
            "gold_call": "run(_oracle_pathway_convergence)",
        },
        {
            "setup": """import numpy as np
dx = 0.014
rng = np.random.default_rng(13)
lev = {"CT1": np.array([3.60, 3.75]), "CT2": np.array([4.10, 4.35])}
psi = {}
for a in lev:
    w = rng.normal(0.0, 1.0, size=(31, 3))
    psi[a] = w / np.sqrt((w ** 2).sum(axis=0) * dx)
vel = {("CT1", "CT2"): 0.05}
def run(f):
    try:
        f(0.5, 0.28, 0.8, {"CT1": 1.0, "CT2": 1.5}, 3.43, lev, psi, dx, ("CT1", "CT2"), vel)
        return 0                              # three functions against two levels
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run(pathway_convergence)",
            "gold_call": "run(_oracle_pathway_convergence)",
        },
    ]

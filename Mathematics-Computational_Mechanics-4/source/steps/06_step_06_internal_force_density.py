"""
Accumulate the internal peridynamic force density at every material point for a given displacement field and bond state.

Each directed bond contributes a central pairwise force along its deformed direction, obtained by differentiating that bond's micro-potential with respect to the deformed bond vector, so the coefficient multiplying the stretch is the micro-potential constant supplied to this function, used unchanged. Broken bonds, whose state is zero, contribute nothing.

Because the internal potential is a double sum over points and their families, its variation places two contributions on every directed bond: one on the bond's owner and one on its neighbour, equal in magnitude and opposite in direction as force densities, but weighted by different quadrature volumes because the two points occupy voxels of different size. A point therefore accumulates contributions both from the bonds it owns and from the bonds that reach it as a neighbour, and neither set may be omitted when horizons vary in space.

The returned quantity is a force density, that is, the internal term of mass density times acceleration; it does not include any external body force. It has one row per material point, in the lattice ordering.

The influence function is the power law omega(r) = r ** (-influence_exponent), and micromodulus is the constant appearing in the micro-potential, as returned by the calibration step.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; bonds does not have shape (B, 2); any bond index lies outside the range of lattice; micromodulus does not have shape (N,) matching lattice; micromodulus contains non-finite or non-positive values; bond_states does not have shape (B,) matching bonds; bond_states contains a value other than 0 or 1; displacement does not have shape (N, 2) matching lattice; displacement contains non-finite values; influence_exponent is not finite or is greater than or equal to 3; any bond has non-positive reference length; any bond has non-positive deformed length.

In a variational formulation the equations of motion are not postulated as a balance of pairwise forces but obtained by making the discrete action stationary. The internal potential energy is a double sum, over every point and over every member of that point's family, of the bond micro-potential weighted by both voxel volumes. Taking the variation with respect to the position of one point collects two distinct kinds of term: those in which the point appears as the owner of a bond, and those in which it appears as some other point's neighbour. The second group is exactly the dual family, and it is the variational origin of what was previously introduced as a corrective construction for varying horizons.

Under a uniform horizon the two groups coincide, the pairwise force is antisymmetric, and the familiar single-sum equation of motion is recovered with the two contributions absorbed into a redefined micro-modulus that is twice the one appearing in the double-sum micro-potential. Under a varying horizon the two groups differ, and dropping the dual contribution leaves a point near a refinement interface pulled by neighbours it does not pull back. Those unbalanced interactions are the ghost forces: they violate linear momentum balance and act like an artificial boundary that partially reflects stress waves and deflects crack paths.

The quadrature weights matter as much as the force law. Each contribution carries the volume of the point being summed over, not the volume of the point receiving it, so at an interface between coarse and fine regions the action and the reaction are weighted by different volumes. This asymmetry is what makes the total force vanish when summed over the body with the correct weights, and total linear momentum is therefore a sensitive check on whether the accumulation has been assembled correctly.

Returns
-------
np.ndarray, an (N, 2) float array of internal force density per point, in the lattice ordering
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def internal_force_density(lattice: np.ndarray, bonds: np.ndarray,
                           micromodulus: np.ndarray, bond_states: np.ndarray,
                           displacement: np.ndarray,
                           influence_exponent: float) -> np.ndarray:
    '''Accumulate internal force density at every point.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour].
    micromodulus : np.ndarray
        (N,) float array of micro-potential constants, one per point.
    bond_states : np.ndarray
        (B,) array of bond states, 1 for intact and 0 for broken.
    displacement : np.ndarray
        (N, 2) float array of current displacements.
    influence_exponent : float
        Exponent a in the influence function omega(r) = r ** (-a). Must be < 3.

    Returns
    -------
    force : np.ndarray
        (N, 2) float array of internal force density at each point, in the
        lattice ordering.
    '''
    force = np.zeros((len(np.asarray(lattice)), 2), dtype=float)
    return force  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_internal_force_density(lattice: np.ndarray, bonds: np.ndarray,
                                   micromodulus: np.ndarray, bond_states: np.ndarray,
                                   displacement: np.ndarray,
                                   influence_exponent: float) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    bd = np.asarray(bonds)
    cm = np.asarray(micromodulus, dtype=float)
    mu = np.asarray(bond_states, dtype=float)
    u = np.asarray(displacement, dtype=float)
    a = float(influence_exponent)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    if bd.ndim != 2 or bd.shape[1] != 2:
        raise ValueError("bonds must have shape (B, 2)")
    if bd.shape[0] > 0 and (bd.min() < 0 or bd.max() >= lat.shape[0]):
        raise ValueError("bond indices out of range for lattice")
    if cm.ndim != 1 or cm.shape[0] != lat.shape[0]:
        raise ValueError("micromodulus must have shape (N,) matching lattice")
    if not np.all(np.isfinite(cm)) or np.any(cm <= 0.0):
        raise ValueError("micromodulus must be finite and > 0")
    if mu.ndim != 1 or mu.shape[0] != bd.shape[0]:
        raise ValueError("bond_states must have shape (B,) matching bonds")
    if not np.all((mu == 0.0) | (mu == 1.0)):
        raise ValueError("bond_states must contain only the values 0 and 1")
    if u.shape != (lat.shape[0], 2):
        raise ValueError("displacement must have shape (N, 2) matching lattice")
    if not np.all(np.isfinite(u)):
        raise ValueError("displacement must be finite")
    if not np.isfinite(a) or a >= 3.0:
        raise ValueError("influence_exponent must be finite and < 3")

    force = np.zeros((lat.shape[0], 2), dtype=float)
    if bd.shape[0] == 0:
        return force

    own = bd[:, 0].astype(np.int64)
    nbr = bd[:, 1].astype(np.int64)
    position = lat[:, :2] + u
    d = position[nbr] - position[own]
    R = np.hypot(lat[nbr, 0] - lat[own, 0], lat[nbr, 1] - lat[own, 1])
    if np.any(R <= 0.0):
        raise ValueError("bonds must have strictly positive reference length")
    r = np.hypot(d[:, 0], d[:, 1])
    if np.any(r <= 0.0):
        raise ValueError("bonds must have strictly positive deformed length")

    s = (r - R) / R
    f = (cm[nbr] * R ** (-a) * mu * s / r)[:, None] * d
    np.add.at(force, own, f * lat[nbr, 4][:, None])
    np.add.at(force, nbr, -f * lat[own, 4][:, None])
    return force

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: production system under a smooth non-affine displacement ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            if (Y[i] - cy) * (Y[j] - cy) < 0.0:
                t = (cy - Y[i]) / (Y[j] - Y[i])
                if X[i] + t * (X[j] - X[i]) <= cxm:
                    continue
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 0.02], [0.02, 0.04]], [0.002, 0.001], 0.02, 3.015)
bonds = _bonds(lattice, 0.0101, 0.0151)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
bond_states = np.ones(len(bonds), dtype=float)
displacement = np.stack([2e-5 * np.sin(70.0 * lattice[:, 0]),
                         3e-5 * np.cos(90.0 * lattice[:, 1])], axis=1)
influence_exponent = 2.0
""",
            "call": "internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, influence_exponent)",
            "gold_call": "_oracle_internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, influence_exponent)",
        },
        # --- Boundary: undeformed configuration, every stretch vanishes ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            if (Y[i] - cy) * (Y[j] - cy) < 0.0:
                t = (cy - Y[i]) / (Y[j] - Y[i])
                if X[i] + t * (X[j] - X[i]) <= cxm:
                    continue
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 0.02], [0.02, 0.04]], [0.002, 0.001], 0.02, 3.015)
bonds = _bonds(lattice, 0.0101, 0.0151)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
bond_states = np.ones(len(bonds), dtype=float)
displacement = np.zeros((len(lattice), 2), dtype=float)
influence_exponent = 2.0
""",
            "call": "internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, influence_exponent)",
            "gold_call": "_oracle_internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, influence_exponent)",
        },
        # --- Edge: asymmetric families across a refinement interface, with broken bonds ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 0.5], [0.5, 1.0]], [0.5, 0.25], 0.5, 2.0)
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = _cm(72e9, lattice[:, 3], 2.0)
bond_states = np.ones(len(bonds), dtype=float)
bond_states[::3] = 0.0
displacement = np.stack([1e-3 * lattice[:, 0], -5e-4 * lattice[:, 1]], axis=1)
influence_exponent = 2.0
""",
            "call": "internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, influence_exponent)",
            "gold_call": "_oracle_internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, influence_exponent)",
        },
        # --- Edge: uniform compression, every stretch negative ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 4.0]], [1.0], 4.0, 1.5)
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = _cm(1.0, lattice[:, 3], 0.0)
bond_states = np.ones(len(bonds), dtype=float)
displacement = -0.01 * lattice[:, :2]
influence_exponent = 0.0
""",
            "call": "internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, influence_exponent)",
            "gold_call": "_oracle_internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, influence_exponent)",
        },
        # --- Invalid: displacement with three components ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 4.0]], [1.0], 4.0, 1.5)
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = _cm(1.0, lattice[:, 3], 0.0)
bond_states = np.ones(len(bonds), dtype=float)
displacement = np.zeros((len(lattice), 3), dtype=float)
def run_model():
    try:
        internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: bond state outside {0, 1} ---
        {
            "setup": """import numpy as np
def _mk(xb, sp, h, mr):
    rows = []
    for r in range(len(sp)):
        d = sp[r]; lo = xb[r][0]
        nx = int(round((xb[r][1] - lo) / d)); ny = int(round(h / d))
        for ix in range(nx):
            for iy in range(ny):
                rows.append((lo + (ix + 0.5) * d, (iy + 0.5) * d, d, mr * d, d * d))
    return np.asarray(rows, dtype=float)
def _bonds(lat, cy, cxm):
    X, Y, dl = lat[:, 0], lat[:, 1], lat[:, 3]
    o, n = [], []
    for i in range(len(lat)):
        r = np.hypot(X - X[i], Y - Y[i])
        for j in np.where((r > 0.0) & (r <= dl[i] * (1.0 + 1e-12)))[0]:
            o.append(i); n.append(j)
    return np.asarray([o, n], dtype=np.int64).T.reshape(-1, 2)
def _cm(E, d, a):
    return 8.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
lattice = _mk([[0.0, 4.0]], [1.0], 4.0, 1.5)
bonds = _bonds(lattice, -10.0, 0.0)
micromodulus = _cm(1.0, lattice[:, 3], 0.0)
bond_states = np.full(len(bonds), 2.0, dtype=float)
displacement = np.zeros((len(lattice), 2), dtype=float)
def run_model():
    try:
        internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: deformed configuration collapses a bond to zero length ---
        {
            "setup": """import numpy as np
lattice = np.array([[0.0, 0.0, 1.0, 1.5, 1.0], [1.0, 0.0, 1.0, 1.5, 1.0]], dtype=float)
bonds = np.array([[0, 1], [1, 0]], dtype=np.int64)
micromodulus = np.array([1.0, 1.0], dtype=float)
bond_states = np.ones(2, dtype=float)
displacement = np.array([[0.5, 0.0], [-0.5, 0.0]], dtype=float)
def run_model():
    try:
        internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_internal_force_density(lattice, bonds, micromodulus, bond_states, displacement, 0.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]

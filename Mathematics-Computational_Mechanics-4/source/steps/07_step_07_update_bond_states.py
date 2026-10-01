"""
Apply the irreversible bond-failure criterion to the current deformed configuration and return the updated bond states.

For every directed bond the current stretch is computed from the deformed positions, and the bond fails when the square of that stretch reaches or exceeds the squared critical stretch supplied for it. Comparing squares rather than signed stretches is what the energy-based criterion requires, since the energy stored in a bond does not depend on the sign of its deformation.

Failure is irreversible: a bond whose incoming state is zero remains zero regardless of its current stretch, and is never restored.

A bond is exempt from failure when either of the points it joins is marked in no_fail_points. Exempt bonds retain their incoming state.

Bond states are returned in the bond ordering, with the value 1 for intact and 0 for broken. The incoming array is not modified.

Raises ValueError if: lattice does not have shape (N, 5) with N >= 2; lattice contains non-finite values; bonds does not have shape (B, 2); any bond index lies outside the range of lattice; bond_states does not have shape (B,) matching bonds; bond_states contains a value other than 0 or 1; critical_stretch_squared does not have shape (B,) matching bonds; critical_stretch_squared contains non-finite or non-positive values; displacement does not have shape (N, 2) matching lattice; displacement contains non-finite values; no_fail_points does not have shape (N,) matching lattice; no_fail_points contains a value other than 0 or 1; any bond has non-positive reference length; any bond has non-positive deformed length.

Damage in peridynamics is carried entirely by a Boolean state attached to each bond. A bond that has failed transmits no force and contributes nothing to the internal energy, so a crack is represented as a locus across which the connectivity has been removed rather than as a geometric surface. Because bonds fail independently and locally, crack initiation, branching, coalescence, and arrest all emerge from the same rule without any separate criterion for the direction or speed of propagation.

Failure is assumed irreversible: once a bond has broken it cannot heal even if the deformation subsequently relaxes. This is the discrete counterpart of the thermodynamic irreversibility of brittle fracture, and it means the state is genuinely history dependent. Evaluating the criterion at every update of the owning interaction, rather than once per global step, is what makes that history faithful when different regions advance at different rates.

Whether the criterion is applied to the signed stretch or to its square is not a matter of convention. A threshold on the signed stretch is a statement about extension alone and can never break a bond in compression. A threshold derived from the stored energy density is quadratic, so a sufficiently compressed bond stores as much energy as an equally extended one and fails on the same footing. Which behaviour is correct follows from how the critical value was calibrated, and mixing a quadratic calibration with a signed test, or the reverse, changes the set of bonds that break and therefore the crack path.

Regions where failure is suppressed are a standard modelling device. Loaded boundaries and the surfaces where tractions are applied as equivalent body forces experience the peridynamic surface effect, in which points near a free surface have incomplete families and therefore artificially low stiffness. Without suppression, damage nucleates spuriously at those boundaries instead of at the feature under study.

Returns
-------
np.ndarray, a (B,) float array of updated bond states taking only the values 0.0 and 1.0, in the bond ordering
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def update_bond_states(lattice: np.ndarray, bonds: np.ndarray, bond_states: np.ndarray,
                       critical_stretch_squared: np.ndarray, displacement: np.ndarray,
                       no_fail_points: np.ndarray) -> np.ndarray:
    '''Apply the irreversible failure criterion and return updated bond states.

    Parameters
    ----------
    lattice : np.ndarray
        (N, 5) float array with columns [x, y, spacing, horizon, volume].
    bonds : np.ndarray
        (B, 2) integer array with columns [owner, neighbour].
    bond_states : np.ndarray
        (B,) array of incoming bond states, 1 for intact and 0 for broken.
    critical_stretch_squared : np.ndarray
        (B,) float array of squared critical stretch, one per directed bond.
    displacement : np.ndarray
        (N, 2) float array of current displacements.
    no_fail_points : np.ndarray
        (N,) array marking points exempt from failure, 1 for exempt and 0 otherwise.

    Returns
    -------
    updated_states : np.ndarray
        (B,) float array of updated bond states, in the bond ordering.
    '''
    updated_states = np.asarray(bond_states, dtype=float).copy()
    return updated_states  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_update_bond_states(lattice: np.ndarray, bonds: np.ndarray, bond_states: np.ndarray,
                               critical_stretch_squared: np.ndarray, displacement: np.ndarray,
                               no_fail_points: np.ndarray) -> np.ndarray:
    """Reference implementation."""
    lat = np.asarray(lattice, dtype=float)
    bd = np.asarray(bonds)
    mu = np.asarray(bond_states, dtype=float)
    sc2 = np.asarray(critical_stretch_squared, dtype=float)
    u = np.asarray(displacement, dtype=float)
    nf = np.asarray(no_fail_points, dtype=float)
    if lat.ndim != 2 or lat.shape[1] != 5 or lat.shape[0] < 2:
        raise ValueError("lattice must have shape (N, 5) with N >= 2")
    if not np.all(np.isfinite(lat)):
        raise ValueError("lattice must be finite")
    if bd.ndim != 2 or bd.shape[1] != 2:
        raise ValueError("bonds must have shape (B, 2)")
    if bd.shape[0] > 0 and (bd.min() < 0 or bd.max() >= lat.shape[0]):
        raise ValueError("bond indices out of range for lattice")
    if mu.ndim != 1 or mu.shape[0] != bd.shape[0]:
        raise ValueError("bond_states must have shape (B,) matching bonds")
    if not np.all((mu == 0.0) | (mu == 1.0)):
        raise ValueError("bond_states must contain only the values 0 and 1")
    if sc2.ndim != 1 or sc2.shape[0] != bd.shape[0]:
        raise ValueError("critical_stretch_squared must have shape (B,) matching bonds")
    if not np.all(np.isfinite(sc2)) or np.any(sc2 <= 0.0):
        raise ValueError("critical_stretch_squared must be finite and > 0")
    if u.shape != (lat.shape[0], 2):
        raise ValueError("displacement must have shape (N, 2) matching lattice")
    if not np.all(np.isfinite(u)):
        raise ValueError("displacement must be finite")
    if nf.ndim != 1 or nf.shape[0] != lat.shape[0]:
        raise ValueError("no_fail_points must have shape (N,) matching lattice")
    if not np.all((nf == 0.0) | (nf == 1.0)):
        raise ValueError("no_fail_points must contain only the values 0 and 1")
    if bd.shape[0] == 0:
        return mu.copy()

    own = bd[:, 0].astype(np.int64)
    nbr = bd[:, 1].astype(np.int64)
    R = np.hypot(lat[nbr, 0] - lat[own, 0], lat[nbr, 1] - lat[own, 1])
    if np.any(R <= 0.0):
        raise ValueError("bonds must have strictly positive reference length")
    position = lat[:, :2] + u
    d = position[nbr] - position[own]
    r = np.hypot(d[:, 0], d[:, 1])
    if np.any(r <= 0.0):
        raise ValueError("bonds must have strictly positive deformed length")

    s = (r - R) / R
    allowed = (nf[own] == 0.0) & (nf[nbr] == 0.0)
    breaks = (mu == 1.0) & allowed & (s * s >= sc2)
    updated_states = mu.copy()
    updated_states[breaks] = 0.0
    return updated_states

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: tensile displacement, a substantial fraction of bonds fails ---
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
def _sc2(lat, bd, E, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    d = lat[n, 3]
    cm = 16.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
    return 3.0 * Gc / (cm * d ** 3 * R ** (1.0 - a))
lattice = _mk([[0.0, 0.02], [0.02, 0.04]], [0.002, 0.001], 0.02, 3.015)
bonds = _bonds(lattice, 0.0101, 0.0151)
critical_stretch_squared = _sc2(lattice, bonds, 72e9, 135.0, 2.0)
bond_states = np.ones(len(bonds), dtype=float)
no_fail_points = ((lattice[:, 1] < 0.0021) | (lattice[:, 1] > 0.0179)).astype(float)
displacement = np.stack([np.zeros(len(lattice)), 1.2e-3 * lattice[:, 1]], axis=1)
""",
            "call": "update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)",
            "gold_call": "_oracle_update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)",
        },
        # --- Boundary: deformation far below every threshold, nothing fails ---
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
def _sc2(lat, bd, E, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    d = lat[n, 3]
    cm = 16.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
    return 3.0 * Gc / (cm * d ** 3 * R ** (1.0 - a))
lattice = _mk([[0.0, 0.02], [0.02, 0.04]], [0.002, 0.001], 0.02, 3.015)
bonds = _bonds(lattice, 0.0101, 0.0151)
critical_stretch_squared = _sc2(lattice, bonds, 72e9, 135.0, 2.0)
bond_states = np.ones(len(bonds), dtype=float)
no_fail_points = ((lattice[:, 1] < 0.0021) | (lattice[:, 1] > 0.0179)).astype(float)
displacement = np.stack([np.zeros(len(lattice)), 1.0e-6 * lattice[:, 1]], axis=1)
""",
            "call": "update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)",
            "gold_call": "_oracle_update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)",
        },
        # --- Edge: uniform contraction, every stretch negative ---
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
def _sc2(lat, bd, E, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    d = lat[n, 3]
    cm = 16.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
    return 3.0 * Gc / (cm * d ** 3 * R ** (1.0 - a))
lattice = _mk([[0.0, 0.02], [0.02, 0.04]], [0.002, 0.001], 0.02, 3.015)
bonds = _bonds(lattice, 0.0101, 0.0151)
critical_stretch_squared = _sc2(lattice, bonds, 72e9, 135.0, 2.0)
bond_states = np.ones(len(bonds), dtype=float)
no_fail_points = ((lattice[:, 1] < 0.0021) | (lattice[:, 1] > 0.0179)).astype(float)
displacement = -2.0e-3 * lattice[:, :2]
""",
            "call": "update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)",
            "gold_call": "_oracle_update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)",
        },
        # --- Edge: irreversibility, pre-broken bonds under zero deformation ---
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
def _sc2(lat, bd, E, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    d = lat[n, 3]
    cm = 16.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
    return 3.0 * Gc / (cm * d ** 3 * R ** (1.0 - a))
lattice = _mk([[0.0, 0.02], [0.02, 0.04]], [0.002, 0.001], 0.02, 3.015)
bonds = _bonds(lattice, 0.0101, 0.0151)
critical_stretch_squared = _sc2(lattice, bonds, 72e9, 135.0, 2.0)
bond_states = np.ones(len(bonds), dtype=float)
bond_states[:100] = 0.0
no_fail_points = ((lattice[:, 1] < 0.0021) | (lattice[:, 1] > 0.0179)).astype(float)
displacement = np.zeros((len(lattice), 2), dtype=float)
""",
            "call": "update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)",
            "gold_call": "_oracle_update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)",
        },
        # --- Edge: every point exempt, failure entirely suppressed ---
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
def _sc2(lat, bd, E, Gc, a):
    o, n = bd[:, 0], bd[:, 1]
    R = np.hypot(lat[n, 0] - lat[o, 0], lat[n, 1] - lat[o, 1])
    d = lat[n, 3]
    cm = 16.0 * E / (5.0 * np.pi * (d ** (3.0 - a) / (3.0 - a)))
    return 3.0 * Gc / (cm * d ** 3 * R ** (1.0 - a))
lattice = _mk([[0.0, 0.02], [0.02, 0.04]], [0.002, 0.001], 0.02, 3.015)
bonds = _bonds(lattice, 0.0101, 0.0151)
critical_stretch_squared = _sc2(lattice, bonds, 72e9, 135.0, 2.0)
bond_states = np.ones(len(bonds), dtype=float)
no_fail_points = np.ones(len(lattice), dtype=float)
displacement = np.stack([np.zeros(len(lattice)), 1.2e-3 * lattice[:, 1]], axis=1)
""",
            "call": "update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)",
            "gold_call": "_oracle_update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)",
        },
        # --- Invalid: non-positive critical stretch ---
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
lattice = _mk([[0.0, 4.0]], [1.0], 4.0, 1.5)
bonds = _bonds(lattice, -10.0, 0.0)
critical_stretch_squared = np.zeros(len(bonds), dtype=float)
bond_states = np.ones(len(bonds), dtype=float)
no_fail_points = np.zeros(len(lattice), dtype=float)
displacement = np.zeros((len(lattice), 2), dtype=float)
def run_model():
    try:
        update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: no_fail_points outside {0, 1} ---
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
lattice = _mk([[0.0, 4.0]], [1.0], 4.0, 1.5)
bonds = _bonds(lattice, -10.0, 0.0)
critical_stretch_squared = np.full(len(bonds), 1e-6, dtype=float)
bond_states = np.ones(len(bonds), dtype=float)
no_fail_points = np.full(len(lattice), 2.0, dtype=float)
displacement = np.zeros((len(lattice), 2), dtype=float)
def run_model():
    try:
        update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_update_bond_states(lattice, bonds, bond_states, critical_stretch_squared, displacement, no_fail_points)
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

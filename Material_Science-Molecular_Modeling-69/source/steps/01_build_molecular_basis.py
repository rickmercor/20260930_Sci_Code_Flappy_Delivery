"""
Construct localized ordered molecular electron-hole states.

The molecular model includes distinct local-exciton and charge-transfer

energies. Positive exciton binding lowers the local-exciton energy. The

ground-state energy is zero. Lengths are nm; energies are eV.

Returns
-------
np.ndarray (N*N,6): [e,h,E(0),separation,label,dE/ds] per pair
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_molecular_basis(
    positions: "np.ndarray",
    homo: "np.ndarray",
    lumo: "np.ndarray",
    binding: "np.ndarray",
    j0: float,
    rj: float,
    field: "np.ndarray",
    disorder: "np.ndarray",
    rthr: float,
    direction: "np.ndarray",
) -> "np.ndarray":
    """Build localized pair energies and their electric-field slopes.
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Finite distinct molecular sites in nm; N >= 2, no periodic boundary.
    homo, lumo, binding : ndarray, shape (N,)
        Finite site energies in eV; binding is nonnegative.
    j0, rj : float
        Positive Mataga energy in eV and range in nm.
    field, direction : ndarray, shape (2,)
        Finite base field and direction in F(s) = field + s * direction.
        The field and s use V/nm; direction is dimensionless and may be zero.
    disorder : ndarray, shape (N, N)
        Finite ordered-pair energy corrections in eV.
    rthr : float
        Positive electron-hole separation threshold for extraction in nm.
    
    Returns
    -------
    ndarray, shape (N*N, 6)
        Row e*N+h is [e, h, E(0), separation, label, dE/ds].
        E(s) = lumo[e] - homo[h] + disorder[e,h] minus binding[e] for e=h,
        or minus j0/(1+separation/rj) otherwise, minus
        (positions[h]-positions[e]) dot F(s) in both cases.
        Labels are LE=0 for e=h, CS=2 for non-LE separation >= rthr,
        and CT=1 otherwise. Energies are affine functions of s.
    
    Raises
    ------
    ValueError
        For nonfinite, malformed or out-of-domain inputs specified above.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite(value, name):
    import numpy as np

    try:
        out = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(name + " must be numeric") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(name + " must be finite")
    return out


def _positive(value, name, zero=False):
    out = _finite(value, name)
    if out.ndim != 0 or (out < 0 if zero else out <= 0):
        raise ValueError(name + " must be a valid nonnegative/positive scalar")
    return float(out)


def _positions(positions):
    import numpy as np

    pos = _finite(positions, "positions")
    if pos.ndim != 2 or pos.shape[0] < 2 or pos.shape[1] != 2:
        raise ValueError("positions must have shape (N, 2), N >= 2")
    dist = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    if np.any(dist[np.triu_indices(len(pos), 1)] <= 0):
        raise ValueError("molecular sites must be distinct")
    return pos, dist


def _oracle_build_molecular_basis(
    positions: "np.ndarray",
    homo: "np.ndarray",
    lumo: "np.ndarray",
    binding: "np.ndarray",
    j0: float,
    rj: float,
    field: "np.ndarray",
    disorder: "np.ndarray",
    rthr: float,
    direction: "np.ndarray",
) -> "np.ndarray":
    import numpy as np

    pos, dist = _positions(positions)
    n = len(pos)
    homo, lumo, binding = [
        _finite(a, "site array") for a in (homo, lumo, binding)
    ]
    disorder, field = _finite(disorder, "disorder"), _finite(field, "field")
    if any(a.shape != (n,) for a in (homo, lumo, binding)):
        raise ValueError("site arrays must have shape (N,)")
    if disorder.shape != (n, n) or field.shape != (2,) or np.any(binding < 0):
        raise ValueError("invalid disorder, field or binding")
    j0, rj, rthr = [_positive(v, "scale") for v in (j0, rj, rthr)]
    e, h = np.indices((n, n)).reshape(2, -1)
    radius = dist[e, h]
    energy = lumo[e] - homo[h] + disorder[e, h]
    energy -= np.where(e == h, binding[e], j0 / (1 + radius / rj))
    energy -= (pos[h] - pos[e]) @ field
    label = np.where(e == h, 0, np.where(radius >= rthr, 2, 1))
    direction = _finite(direction, "direction")
    if direction.shape != (2,):
        raise ValueError("direction must have shape (2,)")
    slope = -(pos[h] - pos[e]) @ direction
    return np.column_stack((e, h, energy, radius, label, slope))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
)
homo = np.zeros(6)
lumo = np.full(6, 0.95)
binding = np.full(6, 0.24)
e, h = np.indices((6, 6))
disorder = (
    0.040 * np.cos(2 * np.pi * (e + 1) / 7)
    + 0.025 * np.sin(2 * np.pi * (h + 1) / 7)
    + 0.012 * np.sin((e + 1) * (h + 2))
)
field = np.array([0.045, 0.0])
direction = np.array([1.0, 0.0])
illumination = 1 + 0.2 * np.cos(np.arange(6) + 1)
parameters = np.array(
    [
        0.60,
        0.30,
        2.0,
        np.sqrt(2),
        1.0,
        0.40,
        0.35,
        0.004,
        0.003,
        0.002,
        0.025,
        0.105,
        0.032,
        0.090,
        0.060,
        310.0,
        12.0,
        40.0,
        0.020,
        0.001,
        10.0,
    ]
)
""",
            "call": """
build_molecular_basis(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    parameters[0],
    parameters[1],
    field.copy(),
    disorder.copy(),
    parameters[2],
    direction.copy(),
)
""",
            "gold_call": """
_oracle_build_molecular_basis(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    parameters[0],
    parameters[1],
    field.copy(),
    disorder.copy(),
    parameters[2],
    direction.copy(),
)
""",
            "tol": 1e-06,
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
homo = np.array([0.01, -0.01, 0.02])
lumo = np.array([0.94, 0.98, 0.91])
binding = np.array([0.23, 0.25, 0.22])
disorder = np.array(
    [[0.02, -0.01, 0.03], [0.01, 0.0, -0.02], [-0.01, 0.015, 0.01]]
)
field = np.array([0.018, -0.01])
direction = np.array([0.6, 0.8])
illumination = np.array([1.0, 0.4, 1.7])
parameters = np.array(
    [
        0.60,
        0.30,
        1.8,
        1.01,
        1.0,
        0.40,
        0.35,
        0.004,
        0.003,
        0.002,
        0.025,
        0.105,
        0.032,
        0.090,
        0.060,
        330.0,
        10.0,
        35.0,
        0.020,
        0.001,
        6.0,
    ]
)
""",
            "call": """
build_molecular_basis(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    parameters[0],
    parameters[1],
    field.copy(),
    disorder.copy(),
    parameters[2],
    direction.copy(),
)
""",
            "gold_call": """
_oracle_build_molecular_basis(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    parameters[0],
    parameters[1],
    field.copy(),
    disorder.copy(),
    parameters[2],
    direction.copy(),
)
""",
            "tol": 1e-06,
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
)
homo = np.zeros(6)
lumo = np.full(6, 0.95)
binding = np.full(6, 0.24)
e, h = np.indices((6, 6))
disorder = (
    0.040 * np.cos(2 * np.pi * (e + 1) / 7)
    + 0.025 * np.sin(2 * np.pi * (h + 1) / 7)
    + 0.012 * np.sin((e + 1) * (h + 2))
)
field = np.array([0.045, 0.0])
direction = np.array([1.0, 0.0])
illumination = 1 + 0.2 * np.cos(np.arange(6) + 1)
parameters = np.array(
    [
        0.60,
        0.30,
        2.0,
        np.sqrt(2),
        1.0,
        0.40,
        0.35,
        0.004,
        0.003,
        0.002,
        0.025,
        0.105,
        0.032,
        0.090,
        0.060,
        310.0,
        12.0,
        40.0,
        0.020,
        0.001,
        10.0,
    ]
)
direction[:] = 0.0
parameters[2] = 1.0
""",
            "call": """
build_molecular_basis(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    parameters[0],
    parameters[1],
    field.copy(),
    disorder.copy(),
    parameters[2],
    direction.copy(),
)
""",
            "gold_call": """
_oracle_build_molecular_basis(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    parameters[0],
    parameters[1],
    field.copy(),
    disorder.copy(),
    parameters[2],
    direction.copy(),
)
""",
            "tol": 1e-06,
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
)
homo = np.zeros(6)
lumo = np.full(6, 0.95)
binding = np.full(6, 0.24)
e, h = np.indices((6, 6))
disorder = (
    0.040 * np.cos(2 * np.pi * (e + 1) / 7)
    + 0.025 * np.sin(2 * np.pi * (h + 1) / 7)
    + 0.012 * np.sin((e + 1) * (h + 2))
)
field = np.array([0.045, 0.0])
direction = np.array([1.0, 0.0])
illumination = 1 + 0.2 * np.cos(np.arange(6) + 1)
parameters = np.array(
    [
        0.60,
        0.30,
        2.0,
        np.sqrt(2),
        1.0,
        0.40,
        0.35,
        0.004,
        0.003,
        0.002,
        0.025,
        0.105,
        0.032,
        0.090,
        0.060,
        310.0,
        12.0,
        40.0,
        0.020,
        0.001,
        10.0,
    ]
)
direction = np.zeros(3)


def check_domain(fn):
    try:
        fn(
            positions.copy(),
            homo.copy(),
            lumo.copy(),
            binding.copy(),
            parameters[0],
            parameters[1],
            field.copy(),
            disorder.copy(),
            parameters[2],
            direction.copy(),
        )
    except ValueError:
        return 1
    return 0
""",
            "call": "check_domain(build_molecular_basis)",
            "gold_call": "check_domain(_oracle_build_molecular_basis)",
            "tol": 1e-06,
        },
    ]

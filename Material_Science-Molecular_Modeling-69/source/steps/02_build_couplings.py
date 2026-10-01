"""
Construct the perturbative molecular transfer couplings.

Two local excitons couple through a dipole interaction. Every other

allowed transfer moves exactly one carrier. These couplings enter rates;

they are not diagonalized into delocalized eigenstates in this limit.

Returns
-------
np.ndarray (N*N,N*N): real symmetric transfer amplitudes
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_couplings(
    positions: "np.ndarray",
    cutoff: float,
    a: float,
    rd: float,
    rt: float,
    d0: float,
    te: float,
    th: float,
) -> "np.ndarray":
    """Construct the perturbative molecular transfer amplitudes.
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Finite distinct molecular coordinates in nm; N >= 2.
    cutoff, a, rd, rt : float
        Positive lengths in nm. An edge is active for
        0 < r <= cutoff + 1e-12. Require 1+(r-a)/rd > 0 on active edges.
    d0, te, th : float
        Nonnegative dipole, electron and hole coupling amplitudes in eV.
    
    Returns
    -------
    ndarray, shape (N*N, N*N)
        Real symmetric amplitudes in e*N+h order with zero diagonal.
        Distinct LE states couple by d0/[1+(r-a)/rd]^3. Every other
        allowed transition moves exactly one carrier, with amplitude
        te*exp[-(r-a)/rt] for electrons or th*exp[-(r-a)/rt] for holes.
        Apply the edge cutoff to the moving carrier's site distance.
        All other simultaneous two-carrier transitions are zero.
    
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


def _oracle_build_couplings(
    positions: "np.ndarray",
    cutoff: float,
    a: float,
    rd: float,
    rt: float,
    d0: float,
    te: float,
    th: float,
) -> "np.ndarray":
    import numpy as np

    pos, dist = _positions(positions)
    n = len(pos)
    cutoff, a, rd, rt = [_positive(v, "length") for v in (cutoff, a, rd, rt)]
    d0, te, th = [_positive(v, "coupling", zero=True) for v in (d0, te, th)]
    active = (dist > 0) & (dist <= cutoff + 1e-12)
    if np.any(1 + (dist[active] - a) / rd <= 0):
        raise ValueError(
            ("dipole denominator must be positive on " "active edges")
        )
    dip = np.zeros_like(dist)
    hop = np.zeros_like(dist)
    dip[active] = d0 / (1 + (dist[active] - a) / rd) ** 3
    hop[active] = np.exp(-(dist[active] - a) / rt)
    matrix = te * np.kron(hop, np.eye(n)) + th * np.kron(np.eye(n), hop)
    le = np.arange(n) * (n + 1)
    matrix[np.ix_(le, le)] = dip
    return matrix

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    ("Return three scientific cases and one " "domain check.")
    return [
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
)
e, h = np.indices((6, 6))
homo = np.zeros(6)
lumo = np.full(6, 0.95)
binding = np.full(6, 0.24)
disorder = (
    0.040 * np.cos(2 * np.pi * (e + 1) / 7)
    + 0.025 * np.sin(2 * np.pi * (h + 1) / 7)
    + 0.012 * np.sin((e + 1) * (h + 2))
)
field = np.array([0.045, 0.0])
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
build_couplings(positions.copy(), *parameters[3:10])
""",
            "gold_call": """
_oracle_build_couplings(positions.copy(), *parameters[3:10])
""",
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
)
e, h = np.indices((6, 6))
homo = np.zeros(6)
lumo = np.full(6, 0.95)
binding = np.full(6, 0.24)
disorder = (
    0.040 * np.cos(2 * np.pi * (e + 1) / 7)
    + 0.025 * np.sin(2 * np.pi * (h + 1) / 7)
    + 0.012 * np.sin((e + 1) * (h + 2))
)
field = np.array([0.045, 0.0])
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
parameters[3] = 1.0
parameters[8] = 0.0
""",
            "call": """
build_couplings(positions.copy(), *parameters[3:10])
""",
            "gold_call": """
_oracle_build_couplings(positions.copy(), *parameters[3:10])
""",
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
)
e, h = np.indices((6, 6))
homo = np.zeros(6)
lumo = np.full(6, 0.95)
binding = np.full(6, 0.24)
disorder = (
    0.040 * np.cos(2 * np.pi * (e + 1) / 7)
    + 0.025 * np.sin(2 * np.pi * (h + 1) / 7)
    + 0.012 * np.sin((e + 1) * (h + 2))
)
field = np.array([0.045, 0.0])
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
positions[5] += [0.1, 0.2]
parameters[7] = 0.0
""",
            "call": """
build_couplings(positions.copy(), *parameters[3:10])
""",
            "gold_call": """
_oracle_build_couplings(positions.copy(), *parameters[3:10])
""",
        },
        {
            "setup": """
import numpy as np
from scipy.special import eval_genlaguerre, gammaln

positions = np.array(
    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]
)
e, h = np.indices((6, 6))
homo = np.zeros(6)
lumo = np.full(6, 0.95)
binding = np.full(6, 0.24)
disorder = (
    0.040 * np.cos(2 * np.pi * (e + 1) / 7)
    + 0.025 * np.sin(2 * np.pi * (h + 1) / 7)
    + 0.012 * np.sin((e + 1) * (h + 2))
)
field = np.array([0.045, 0.0])
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
parameters[6] = 0.0


def check_domain(fn):
    try:
        fn(positions.copy(), *parameters[3:10])
    except ValueError:
        return 1
    return 0
""",
            "call": "check_domain(build_couplings)",
            "gold_call": "check_domain(_oracle_build_couplings)",
        },
    ]

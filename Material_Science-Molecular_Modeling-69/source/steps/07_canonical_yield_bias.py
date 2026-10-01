"""
Compute the ordinary second field derivative of the canonical-minus-full charge-yield bias by composing the six preceding steps.

Compose the six preceding steps. The molecular rate response uses

thermal oscillator overlaps; the coupled population response includes

the full driven model and canonical reconstruction. The two yield

responses give the signed curvature of the canonical-minus-full bias.

Returns
-------
float: the ordinary second derivative at s=0 of 100*(eta_canonical(s)-eta_full(s)), in percentage points/(V/nm)^2.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def canonical_bias_curvature(
    positions: "np.ndarray",
    homo: "np.ndarray",
    lumo: "np.ndarray",
    binding: "np.ndarray",
    disorder: "np.ndarray",
    field: "np.ndarray",
    illumination: "np.ndarray",
    parameters: "np.ndarray",
    direction: "np.ndarray",
) -> float:
    """Compute the second field derivative of canonical yield bias.
    
    Parameters
    ----------
    positions : ndarray, shape (N, 2)
        Finite distinct sites in nm, N >= 2, as in Step 1.
    homo, lumo, binding : ndarray, shape (N,)
        Site energies in eV, as in Step 1; binding is nonnegative.
    disorder : ndarray, shape (N, N)
        Finite ordered-pair energy disorder in eV.
    field, direction : ndarray, shape (2,)
        F(s)=field+s*direction, as in Step 1; direction may be zero.
    illumination : ndarray, shape (N,)
        Finite nonnegative LE generation with positive total.
    parameters : ndarray, shape (21,)
        Ordered finite values:
        [j0, rj, rthr, cutoff, a, rd, rt, d0, te, th, low_x, high_x,
         low_p, high_p, quantum, temperature, nmax, mmax, vx, vct, kext].
        Units and domains are those of Steps 1–6. The two inclusive
        vibrational cutoffs must be nonnegative integer-valued numbers.
    
    Returns
    -------
    float
        The ordinary second derivative at s=0 of
        100*(eta_canonical(s)-eta_full(s)), in percentage points/(V/nm)^2.
        Compose the previous six steps using their stated conventions.
        Geometry, pool labels and every non-field input remain fixed.
    
    Raises
    ------
    ValueError
        For invalid inputs required by Steps 1–6, malformed parameters,
        fractional cutoffs, empty pools, nonpositive source, or a singular
        or sinkless base network.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_canonical_bias_curvature(
    positions: "np.ndarray",
    homo: "np.ndarray",
    lumo: "np.ndarray",
    binding: "np.ndarray",
    disorder: "np.ndarray",
    field: "np.ndarray",
    illumination: "np.ndarray",
    parameters: "np.ndarray",
    direction: "np.ndarray",
) -> float:
    import numpy as np

    p = _finite(parameters, "parameters")
    if p.shape != (21,):
        raise ValueError("parameters must have shape (21,)")
    if any(x != int(x) or x < 0 for x in p[16:18]):
        raise ValueError("vibrational cutoffs must be nonnegative integers")
    basis = _oracle_build_molecular_basis(
        positions,
        homo,
        lumo,
        binding,
        p[0],
        p[1],
        field,
        disorder,
        p[2],
        direction,
    )
    coupling = _oracle_build_couplings(positions, *p[3:10])
    kinetics = _oracle_build_kinetics_response(
        basis,
        coupling,
        *p[10:16],
        int(p[16]),
        int(p[17]),
        p[18],
        p[19],
        p[3],
        p[20],
        illumination
    )
    population = _oracle_coupled_population_response(
        basis[:, [2, 5]], basis[:, 4], kinetics, p[15]
    )
    full = _oracle_charge_yield_response(
        kinetics, population[0], basis[:, 4]
    )
    canonical = _oracle_charge_yield_response(
        kinetics, population[1], basis[:, 4]
    )
    return float(100 * (canonical[2, 0] - full[2, 0]))

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
canonical_bias_curvature(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    disorder.copy(),
    field.copy(),
    illumination.copy(),
    parameters.copy(),
    direction.copy(),
)
""",
            "gold_call": """
_oracle_canonical_bias_curvature(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    disorder.copy(),
    field.copy(),
    illumination.copy(),
    parameters.copy(),
    direction.copy(),
)
""",
            "tol": 2e-05,
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
canonical_bias_curvature(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    disorder.copy(),
    field.copy(),
    illumination.copy(),
    parameters.copy(),
    direction.copy(),
)
""",
            "gold_call": """
_oracle_canonical_bias_curvature(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    disorder.copy(),
    field.copy(),
    illumination.copy(),
    parameters.copy(),
    direction.copy(),
)
""",
            "tol": 2e-05,
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
""",
            "call": """
canonical_bias_curvature(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    disorder.copy(),
    field.copy(),
    illumination.copy(),
    parameters.copy(),
    direction.copy(),
)
""",
            "gold_call": """
_oracle_canonical_bias_curvature(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    disorder.copy(),
    field.copy(),
    illumination.copy(),
    parameters.copy(),
    direction.copy(),
)
""",
            "tol": 2e-05,
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
direction = np.array([-0.7, 0.4])
parameters[15] = 290.0
illumination *= 2.5
""",
            "call": """
canonical_bias_curvature(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    disorder.copy(),
    field.copy(),
    illumination.copy(),
    parameters.copy(),
    direction.copy(),
)
""",
            "gold_call": """
_oracle_canonical_bias_curvature(
    positions.copy(),
    homo.copy(),
    lumo.copy(),
    binding.copy(),
    disorder.copy(),
    field.copy(),
    illumination.copy(),
    parameters.copy(),
    direction.copy(),
)
""",
            "tol": 2e-05,
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
parameters[16] = 2.5


def check_domain(fn):
    try:
        fn(
            positions.copy(),
            homo.copy(),
            lumo.copy(),
            binding.copy(),
            disorder.copy(),
            field.copy(),
            illumination.copy(),
            parameters.copy(),
            direction.copy(),
        )
    except ValueError:
        return 1
    return 0
""",
            "call": "check_domain(canonical_bias_curvature)",
            "gold_call": "check_domain(_oracle_canonical_bias_curvature)",
            "tol": 2e-05,
        },
    ]

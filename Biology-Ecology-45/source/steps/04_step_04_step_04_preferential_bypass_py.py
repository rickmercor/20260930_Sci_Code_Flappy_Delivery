"""
Route top-layer dissolved solute through a rainfall-driven macropore event. A correct output caps removal at available dissolved mass, deposits the adsorbed fraction by layer thickness, and exports the remainder; a failing result violates solute conservation or the paper's preferential-flow construction.

The paper adds a parsimonious macropore pathway in which a fraction of rainfall bypasses the matrix and carries dissolved solute. A calibrated adsorption fraction retains part of that bypass mass along the profile.

Returns
-------
np.ndarray with shape (L + 3,), containing [bypass water, bypass mass, L deposited masses, exported mass]
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def preferential_bypass(
    aqueous_concentration: float,
    available_dissolved_mass: float,
    rainfall: float,
    retention_current: float,
    retention_maximum: float,
    movement_fraction: float,
    adsorption_fraction: float,
    layer_thickness: "np.ndarray",
) -> "np.ndarray":
    """Route one solute through a preferential-flow event.
 
    Returns
    -------
    routing : np.ndarray
        [preferential water, bypass mass, L deposited masses, exported mass].
 
    Raises
    ------
    ValueError
        If array shape, finiteness, nonnegativity, retention ordering, or the
        [0,1] fraction contracts are violated.
    """
    return routing

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np
 
def _oracle_preferential_bypass(
    aqueous_concentration: float,
    available_dissolved_mass: float,
    rainfall: float,
    retention_current: float,
    retention_maximum: float,
    movement_fraction: float,
    adsorption_fraction: float,
    layer_thickness: "np.ndarray",
) -> "np.ndarray":
    thickness = np.asarray(layer_thickness, dtype=float)
    values = [aqueous_concentration, available_dissolved_mass, rainfall, retention_current, retention_maximum, movement_fraction, adsorption_fraction]
    if thickness.ndim != 1 or thickness.size == 0 or not np.all(np.isfinite(thickness)) or np.any(thickness <= 0.0):
        raise ValueError("layer_thickness must be a nonempty positive finite 1D array")
    if not all(np.isfinite(x) for x in values):
        raise ValueError("preferential-flow inputs must be finite")
    if aqueous_concentration < 0.0 or available_dissolved_mass < 0.0 or rainfall < 0.0 or retention_current < 0.0 or retention_maximum <= 0.0:
        raise ValueError("preferential-flow magnitudes are outside their allowed ranges")
    if retention_current > retention_maximum:
        raise ValueError("retention_current cannot exceed retention_maximum")
    if not 0.0 <= movement_fraction <= 1.0 or not 0.0 <= adsorption_fraction <= 1.0:
        raise ValueError("movement and adsorption fractions must lie in [0, 1]")
    wcrk = movement_fraction * rainfall * retention_current / retention_maximum
    bypass = min(wcrk * aqueous_concentration, available_dissolved_mass)
    deposited = bypass * adsorption_fraction * thickness / np.sum(thickness)
    exported = bypass - float(np.sum(deposited))
    return np.concatenate(([wcrk, bypass], deposited, [exported])).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {"setup": "import numpy as np\nz=np.array([1.,4.,10.])\n", "call": "preferential_bypass(.8,2.,20.,50.,80.,.5,.08,z)", "gold_call": "_oracle_preferential_bypass(.8,2.,20.,50.,80.,.5,.08,z.copy())"},
        {"setup": "import numpy as np\nz=np.array([2.])\n", "call": "preferential_bypass(.8,2.,0.,0.,80.,.5,.08,z)", "gold_call": "_oracle_preferential_bypass(.8,2.,0.,0.,80.,.5,.08,z.copy())"},
        {"setup": "import numpy as np\nz=np.array([1.,1.])\n", "call": "preferential_bypass(10.,.2,40.,80.,80.,1.,0.,z)", "gold_call": "_oracle_preferential_bypass(10.,.2,40.,80.,80.,1.,0.,z.copy())"},
        {"setup": """import numpy as np
z=np.array([1.,2.])
def run(fn):
    try: fn(1.,1.,2.,1.,2.,1.1,.2,z); return 0
    except ValueError: return 1
""", "call": "run(preferential_bypass)", "gold_call": "run(_oracle_preferential_bypass)"},
        {"setup": """import numpy as np
z=np.array([])
def run(fn):
    try: fn(1.,1.,2.,1.,2.,.5,.2,z); return 0
    except ValueError: return 1
""", "call": "run(preferential_bypass)", "gold_call": "run(_oracle_preferential_bypass)"},
    ]

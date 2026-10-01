"""
Build the axial eigenvalues and their squared norms for a through-silicon via that is adiabatic on its bottom face and held at the ambient temperature by a heat sink on its top face.

The transient temperature field of a layered through-silicon via is governed by the axisymmetric heat conduction equation in cylindrical coordinates, written for the excess temperature theta = T - T_ext so that the sink condition becomes homogeneous. Separating theta(r, z, t) = R(r) Z(z) Gamma(t) sends the axial factor onto the harmonic equation Z'' + eta^2 Z = 0. The axial problem is the only part of the separation that is common to every material layer, because the axial boundary conditions are applied to the whole cross section at once rather than layer by layer, so a single set of axial eigenvalues serves all layers and the layer index can be dropped from eta.

The via is thermally insulated at z = 0, which kills the sine branch and leaves Z(z) = cos(eta z), and it is clamped to the ambient temperature at z = h by the heat sink, which forces cos(eta h) = 0. The admissible axial eigenvalues are therefore the odd quarter-wave numbers eta_m = (2m - 1) pi / (2 h) for m = 1, 2, 3, ..., a set whose lowest member has a quarter wavelength equal to the full via height and therefore sets the slowest axial relaxation the structure can support.

Every projection of an initial condition onto this basis is normalized by the squared norm N_zm = integral of cos^2(eta_m z) over z from 0 to h. Because eta_m h is an odd multiple of pi/2, the oscillatory correction sin(2 eta_m h) / (4 eta_m) vanishes identically, and the norm collapses to h/2 for every mode, which is what makes the axial projection a single scalar rather than a mode-dependent quadrature.

Returns
-------
np.ndarray of shape (n_modes, 2), float: axial eigenvalues in column 0 and squared norms in column 1.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_axial_eigenmodes(height: float, n_modes: int) -> np.ndarray:
    """Axial eigenvalues and squared norms of the via temperature expansion.

    Parameters
    ----------
    height : float
        Total via height h in metres (h > 0).
    n_modes : int
        Number of axial modes to return (n_modes >= 1).

    Returns
    -------
    modes : np.ndarray
        Array of shape (n_modes, 2). Column 0 holds the axial eigenvalues
        eta_m in inverse metres, ordered from the smallest upward. Column 1
        holds the squared norms N_zm in metres.

    Raises
    ------
    ValueError
        If height is not a finite number greater than 0, or if n_modes
        is not an integer greater than or equal to 1.
    """
    return modes  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_axial_eigenmodes(height: float, n_modes: int) -> np.ndarray:
    # Local imports keep the oracle self-contained when the harness
    # executes it in isolation.
    import numpy as np

    if not (isinstance(height, (int, float)) and np.isfinite(height)
            and float(height) > 0.0):
        raise ValueError("height must be a finite number > 0")
    if not (isinstance(n_modes, (int, np.integer)) and not isinstance(n_modes, bool)
            and int(n_modes) >= 1):
        raise ValueError("n_modes must be an integer >= 1")

    height = float(height)
    n_modes = int(n_modes)

    order = np.arange(1, n_modes + 1, dtype=float)
    # Insulated at z = 0 and held at ambient at z = h gives odd quarter waves.
    eigenvalues = (2.0 * order - 1.0) * np.pi / (2.0 * height)
    # The oscillatory part of the norm vanishes because eta_m h is an odd
    # multiple of pi / 2, so every squared norm is exactly h / 2.
    norms = np.full(n_modes, 0.5 * height, dtype=float)

    return np.column_stack([eigenvalues, norms])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark 200 um via, the eight axial modes the pipeline retains (normal scenario) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    # The harness compares plain numbers, so each array this step returns is
    # reduced to two descriptors of order one: a normalized weighted sum that
    # pins down its shape, and the log of its weighted magnitude that pins down
    # its scale. Both stay of order one whatever the component is worth, so an
    # error in a small component cannot hide behind a large one.
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
height = 200.0e-6
""",
            "call": "digest(compute_axial_eigenmodes(height, 8))",
            "gold_call": "digest(_oracle_compute_axial_eigenmodes(height, 8))",
        },
        # --- Edge: thin 50 um die with 24 modes, the top eigenvalue near 1.5e6 per metre ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
height = 50.0e-6
""",
            "call": "digest(compute_axial_eigenmodes(height, 24))",
            "gold_call": "digest(_oracle_compute_axial_eigenmodes(height, 24))",
        },
        # --- Boundary: a single mode, the quarter wave of a 1 mm via ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-12
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
height = 1.0e-3
""",
            "call": "digest(compute_axial_eigenmodes(height, 1))",
            "gold_call": "digest(_oracle_compute_axial_eigenmodes(height, 1))",
        },
        # --- Consistency: every mode vanishes on the sink plane and every squared norm is the integral of cos^2 ---
        # The eigenvalues must be the successive odd quarter waves (cos(eta h) = 0, the
        # lowest one pi / (2 h), neighbours pi / h apart so none is skipped), and column 1
        # must equal the integral of cos^2(eta z) over [0, h], taken here by Gauss-Legendre
        # quadrature, which the sink condition collapses to h / 2 for every mode.
        {
            "setup": """import numpy as np
height = 120.0e-6
def check(fn):
    modes = np.asarray(fn(height, 12), dtype=float)
    eta = modes[:, 0]
    sink = (np.max(np.abs(np.cos(eta * height))) < 1e-12
            and abs(eta[0] * 2.0 * height / np.pi - 1.0) < 1e-12
            and np.max(np.abs(np.diff(eta) * height / np.pi - 1.0)) < 1e-12)
    x, w = np.polynomial.legendre.leggauss(64)
    z = 0.5 * height * (x + 1.0)
    quad = 0.5 * height * (np.cos(np.outer(eta, z)) ** 2 @ w)
    norm = modes.shape == (12, 2) and np.max(np.abs(modes[:, 1] - quad) / quad) < 1e-12
    return int(sink), int(norm)
""",
            "call": "check(compute_axial_eigenmodes)",
            "gold_call": "check(_oracle_compute_axial_eigenmodes)",
        },
        # --- Invalid: zero via height ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_axial_eigenmodes(0.0, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_axial_eigenmodes(0.0, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: infinite via height ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_axial_eigenmodes(np.inf, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_axial_eigenmodes(np.inf, 4)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: fewer than one mode requested ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_axial_eigenmodes(200.0e-6, 0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_axial_eigenmodes(200.0e-6, 0)
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

"""
Compute the zero-temperature (present-day) masses of the two axion flavor fields from the axion decay constant and the model's dimensionless mixing ratios.

The QCD-coupled flavor field couples to gluons through the standard QCD axion mechanism, so its zero-temperature mass is fixed entirely by chiral perturbation theory once the axion decay constant fa is chosen: m_{a,0}^2 = (m_pi^2 f_pi^2 / fa^2) * (m_u m_d)/(m_u+m_d)^2, where m_pi and f_pi are the neutral pion mass and decay constant, and m_u, m_d are the up- and down-quark masses. This is a standard, model-independent relation from chiral perturbation theory, not specific to any particular axion-mixing scenario.

The second flavor field's mass, mS, is a free parameter of the model, conventionally expressed through the dimensionless ratio Rm = mS / m_{a,0}, so that mS = Rm * m_{a,0} once m_{a,0} is known.

Given values to use for the standard hadronic inputs: m_pi = 0.135 GeV, f_pi = 0.092 GeV, m_u = 2.16e-3 GeV, m_d = 4.67e-3 GeV.

Returns
-------
tuple of two floats: (m_{a,0}, mS), both in GeV.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compute_axion_mass_scales(fa: float, Rm: float) -> tuple:
    """Zero-temperature axion mass scales m_{a,0} and mS.

    Parameters
    ----------
    fa : float
        Axion decay constant (GeV), a finite number > 0.
    Rm : float
        Mass ratio mS / m_{a,0}, a finite number > 0.

    Returns
    -------
    masses : tuple
        (m_a0, mS): two floats, both finite and > 0, in GeV.

    Raises
    ------
    ValueError
        If fa or Rm is not a finite number > 0.
    """
    return masses  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_axion_mass_scales(fa: float, Rm: float) -> tuple:
    import numpy as np

    if not (isinstance(fa, (int, float, np.floating)) and np.isfinite(fa) and float(fa) > 0.0):
        raise ValueError("fa must be a finite number > 0")
    if not (isinstance(Rm, (int, float, np.floating)) and np.isfinite(Rm) and float(Rm) > 0.0):
        raise ValueError("Rm must be a finite number > 0")

    fa = float(fa)
    Rm = float(Rm)

    m_pi = 0.135
    f_pi = 0.092
    m_u = 2.16e-3
    m_d = 4.67e-3

    m_a0 = np.sqrt((m_pi**2 * f_pi**2 / fa**2) * (m_u * m_d) / (m_u + m_d) ** 2)
    mS = Rm * m_a0

    return (m_a0, mS)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Valid: benchmark scenario ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-30
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
fa = 1.0e12
Rm = 0.10
""",
            "call": "digest(compute_axion_mass_scales(fa, Rm))",
            "gold_call": "digest(_oracle_compute_axion_mass_scales(fa, Rm))",
        },
        # --- Valid: a much smaller decay constant and different ratio ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-30
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
fa = 5.0e9
Rm = 2.5
""",
            "call": "digest(compute_axion_mass_scales(fa, Rm))",
            "gold_call": "digest(_oracle_compute_axion_mass_scales(fa, Rm))",
        },
        # --- Boundary: Rm = 1 (equal-mass flavor fields) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-30
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
fa = 1.0e11
Rm = 1.0
""",
            "call": "digest(compute_axion_mass_scales(fa, Rm))",
            "gold_call": "digest(_oracle_compute_axion_mass_scales(fa, Rm))",
        },
        # --- Valid: a smaller decay constant, fa = 1e11 GeV (m_a0 of about 5.8e-14 GeV) ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-30
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
fa = 1.0e11
Rm = 0.1
""",
            "call": "digest(compute_axion_mass_scales(fa, Rm))",
            "gold_call": "digest(_oracle_compute_axion_mass_scales(fa, Rm))",
        },
        # --- Valid: fa doubled to 2e11 GeV (m_a0 of about 2.9e-14 GeV); with the previous case this
        # pins the 1/fa scaling of m_a0 ---
        {
            "setup": """import numpy as np
def digest(*parts):
    total = 0.0
    for k, part in enumerate(parts):
        x = np.asarray(part, dtype=float).ravel()
        w = 1.0 + 0.5 * np.arange(x.size) / max(x.size, 1)
        mag = float(np.dot(np.abs(x), w)) + 1e-30
        total += (k + 1.0) * float(np.dot(x, w)) / mag + np.log(mag)
    return float(total / len(parts))
fa = 2.0e11
Rm = 0.1
""",
            "call": "digest(compute_axion_mass_scales(fa, Rm))",
            "gold_call": "digest(_oracle_compute_axion_mass_scales(fa, Rm))",
        },
        # --- Consistency: mS/m_a0 exactly reproduces Rm ---
        {
            "setup": """import numpy as np
def check(fn):
    m_a0, mS = fn(3.7e10, 0.37)
    return int(abs(mS/m_a0 - 0.37) < 1e-9)
""",
            "call": "check(compute_axion_mass_scales)",
            "gold_call": "check(_oracle_compute_axion_mass_scales)",
        },
        # --- Invalid: non-positive decay constant ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_axion_mass_scales(0.0, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_axion_mass_scales(0.0, 0.1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: negative Rm ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        compute_axion_mass_scales(1e11, -0.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_compute_axion_mass_scales(1e11, -0.5)
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

"""
Step 1: Evaluate the deformed Z3 clock coefficients f, g1, g2.

The deformed Z3 clock interaction of the model is parameterized by two positive real numbers r and s through three scalar coefficients: f and g1 are complex, while g2 is real. At r = s = 1, all three coefficients vanish and the undeformed Potts interaction is recovered, providing a useful boundary test.

Returns
-------
# tuple, (f, g1, g2) with f and g1 complex and g2 a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================
def deformation_coefficients(r: float, s: float) -> tuple[complex, complex, float]:
    '''Return the deformation coefficients (f, g1, g2).
 
    Parameters
    ----------
    r, s : float
        Positive finite deformation parameters.
 
    Returns
    -------
    result : tuple[complex, complex, float]
        (f, g1, g2) with f, g1 complex and g2 a native Python float.
 
    Raises
    ------
    ValueError
        If r or s is non-finite or <= 0.
    '''
    return result  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

# =============================================================================
# ORACLE SOLUTION
# =============================================================================
import numpy as np
def _oracle_deformation_coefficients(r: float, s: float) -> tuple[complex, complex, float]:
    import numpy as np
    r = float(r); s = float(s)
    if not np.isfinite(r) or not np.isfinite(s) or r <= 0.0 or s <= 0.0:
        raise ValueError("r and s must be finite and > 0")
    w = np.exp(2j*np.pi/3.0); wb = np.conj(w)
    f  = -(2.0/9.0)*(2.0*(r*s + w*r/s**2 + wb*s/r**2) - (1.0/(r*s) + wb*r**2/s + w*s**2/r))
    g1 = -(2.0/9.0)*(w*(r**2/s + s/r**2) + wb*(s**2/r + r/s**2) + r*s + 1.0/(r*s))
    g2 =  (1.0/9.0)*(3.0 + 1.0/(r*s) + s**2/r + r**2/s - 2.0*(r*s + s/r**2 + r/s**2))
    return complex(f), complex(g1), float(np.real(g2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        # normal
        {"setup": "import numpy as np\nr=0.347; s=0.783", "call": "np.asarray(deformation_coefficients(r,s), dtype=complex)",
         "gold_call": "np.asarray(_oracle_deformation_coefficients(r,s), dtype=complex)"},
        # boundary: undeformed point, all coefficients vanish
        {"setup": "import numpy as np\nr=1.0; s=1.0", "call": "np.asarray(deformation_coefficients(r,s), dtype=complex)",
         "gold_call": "np.asarray(_oracle_deformation_coefficients(r,s), dtype=complex)"},
        # edge: strongly deformed, large 1/r^2 terms
        {"setup": "import numpy as np\nr=0.1; s=0.2", "call": "np.asarray(deformation_coefficients(r,s), dtype=complex)",
         "gold_call": "np.asarray(_oracle_deformation_coefficients(r,s), dtype=complex)"},
    ]

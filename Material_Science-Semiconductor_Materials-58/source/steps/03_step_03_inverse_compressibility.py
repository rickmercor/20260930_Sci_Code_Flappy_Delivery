"""
Step 03 - Inverse compressibility of the dilute exciton condensate.

Inverse compressibility of the dilute exciton condensate.

When the bare gap E_G of the two-band model drops below the exciton binding energy E_b, excitons condense and the
ground-state energy per unit area, expanded in the exciton density n_ex, reads

    E(n_ex) = (E_G - E_b) n_ex + (1/2) (d mu_ex / d n_ex) n_ex^2 + ...,

so that at equilibrium E_b - E_G = (d mu_ex/d n_ex) n_ex. The inverse compressibility d mu_ex/d n_ex
follows from a systematic expansion of the self-consistent mean-field (Hartree-Fock) equations in powers of
the exciton amplitude. It is a functional of the bound-state wave function phi of the previous step alone: it
combines the geometric interlayer capacitance, the intraband (same-layer) exchange energy gained by the two
occupation clouds, and the interband exchange between the pairing amplitude and the interlayer kernel. The exact
combination, including its relative signs and factors of two, is deliberately not restated here; it follows
from that expansion. For a monolayer it is a pure number times
Ry* (a_B*)^2; for d > 0 it also grows with the capacitive charging term 8 pi d.

Both exchange terms are four-dimensional integrals with the singular Coulomb kernel; evaluate them through the
exchange integral of step 1 applied to functions built from phi, with the same 1e-9 relative accuracy.

Inputs: d >= 0 (a_B*). Output: d mu_ex/d n_ex in Ry* (a_B*)^2 as a float. Raises ValueError if d is negative or
not finite.

Returns
-------
float, d mu_ex/d n_ex in Ry* (a_B*)^2
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def inverse_compressibility(d: float) -> float:
    '''Inverse exciton compressibility d mu_ex/d n_ex of the dilute condensate.

    Parameters
    ----------
    d : float
        Interlayer distance in units of a_B*, >= 0.

    Returns
    -------
    result : float
        d mu_ex/d n_ex in units of Ry* (a_B*)^2, converged to a relative accuracy of 1e-9.

    Raises
    ------
    ValueError
        If d is negative or not finite.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_inverse_compressibility(d: float) -> float:
    """Reference implementation: Eq. (A8) of the source, 1/C_G - 2 [<phi^2, V_0 phi^2> - <phi^3, V_d phi>]."""
    d = _check_scalar(d, "d", nonneg=True)
    op = _operator(d)
    op0 = _operator(0.0)
    Eb, phi = _exciton(d)
    intra = np.sum(op["mu"] * phi ** 2 * (op0["M"] @ phi ** 2))
    inter = np.sum(op["mu"] * phi ** 3 * (op["M"] @ phi))
    return float(8 * np.pi * d - 2 * (intra - inter))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: monolayer, the number quoted in the source (6.0566) ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "inverse_compressibility(0.0)",
            "gold_call": "_oracle_inverse_compressibility(0.0)",
            "tol": 1e-6,
        },
        # --- Normal: the heterobilayer of the task ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "inverse_compressibility(0.25)",
            "gold_call": "_oracle_inverse_compressibility(0.25)",
            "tol": 1e-6,
        },
        # --- Boundary: half a Bohr radius ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "inverse_compressibility(0.5)",
            "gold_call": "_oracle_inverse_compressibility(0.5)",
            "tol": 1e-6,
        },
        # --- Edge: one Bohr radius, capacitive term dominant ---
        {
            "setup": "import numpy as np\n"
                     "from scipy.special import ellipk, ellipe\n"
                     "from scipy.linalg import eigh\n",
            "call": "inverse_compressibility(1.0)",
            "gold_call": "_oracle_inverse_compressibility(1.0)",
            "tol": 1e-6,
        },
    ]

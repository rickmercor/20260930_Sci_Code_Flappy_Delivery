"""
For a filled, spin-unpolarized hydrogenic p shell, run the complete reference-plus-ePC pipeline on the fixed radial grid and return the primary-paper ePC W'_inf result.

The final audit combines the standard shell ingredients, the older PC reference and both ePC functionals. Step 4 reaches the primary-paper W_inf branches of Step 3, while Step 6 reaches the primary-paper W'_inf branches of Step 5. The final ePC signs provide consistency checks and W'_inf is the requested quantity.

Returns
-------
a float, the ePC W'_inf in hartree. Raise ``ValueError`` for invalid principal quantum number, nonpositive Z, nonfinite PC reference values, or violated ePC sign constraints.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def p_shell_epc_wprime(n_principal: int, Z: float) -> float:
    """Return the primary-paper ePC W'_inf of a filled hydrogenic p shell.

    Use M=20001 radii geomspaced from 1e-6/Z to
    60*n_principal^2/Z bohr. Run ``p_shell_semilocal_ingredients``, the
    companion ``pc_strong_interaction`` reference, ``epc_w_inf`` and
    ``epc_wprime_inf``. The latter two reach their paper-defined branch
    functions in Steps 3 and 5. Require both returned PC reference values to be finite; otherwise raise ValueError.
    Require ePC W_inf<0 and W'_inf>=0 and return
    W'_inf.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_p_shell_epc_wprime(n_principal: int, Z: float) -> float:
    if n_principal not in (2,3,4):
        raise ValueError("n_principal must be 2, 3 or 4.")
    if Z<=0.0:
        raise ValueError("Z must be positive.")
    r=np.geomspace(1e-6,60.0*n_principal**2,20001)/Z
    ing=_oracle_p_shell_semilocal_ingredients(n_principal,Z,r)
    pc_reference = _oracle_pc_strong_interaction(r,ing[2],ing[3])
    if not np.all(np.isfinite(pc_reference)):
        raise ValueError("The PC reference audit requires two finite values.")
    w_inf=_oracle_epc_w_inf(r,ing[2],ing[6],ing[7])
    w_prime=_oracle_epc_wprime_inf(r,ing[2],ing[6],ing[7])
    if w_inf>=0.0 or w_prime<0.0:
        raise ValueError("The ePC model requires W_inf < 0 and W'_inf >= 0.")
    return w_prime

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict]:

    """Return the canonical 4p shell and further shells and charges."""

    return [

        # Normal: canonical fixture, 4p shell, Z = 10.

        {

            "setup": "",

            "call": "p_shell_epc_wprime(4, 10.0)",

            "gold_call": "_oracle_p_shell_epc_wprime(4, 10.0)",

        },

        # Normal: 2p shell, Z = 7.

        {

            "setup": "",

            "call": "p_shell_epc_wprime(2, 7.0)",

            "gold_call": "_oracle_p_shell_epc_wprime(2, 7.0)",

        },

        # Normal: 3p shell, Z = 5.

        {

            "setup": "",

            "call": "p_shell_epc_wprime(3, 5.0)",

            "gold_call": "_oracle_p_shell_epc_wprime(3, 5.0)",

        },

        # Edge: 2p shell, non-integer Z = 10.5.

        {

            "setup": "",

            "call": "p_shell_epc_wprime(2, 10.5)",

            "gold_call": "_oracle_p_shell_epc_wprime(2, 10.5)",

        },

        # Edge: diffuse 4p shell, low non-integer Z = 4.5.

        {

            "setup": "",

            "call": "p_shell_epc_wprime(4, 4.5)",

            "gold_call": "_oracle_p_shell_epc_wprime(4, 4.5)",

        },

    ]

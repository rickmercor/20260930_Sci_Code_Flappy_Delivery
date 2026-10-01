"""
Construct the (ny,ny) CSC matrix representing multiplication by the positive index-squared profile (C+B*cos(k_mod*x))**2 on the periodic Fourier grid. Here mx is the integer Fourier shift associated with k_mod. Include the diagonal and shifts +/-mx and +/-2*mx, wrapping indices modulo ny and adding all coincident contributions. The Signature is a neutral sparse-matrix stub; the computation belongs in Gold.

The squared profile has constant coefficient C**2+B**2/2, first-harmonic coefficients B*C, and second-harmonic coefficients B**2/4. Its Fourier representation is generally a diagonal plus four shifted contributions, not a tridiagonal matrix in the original ordering. When shifts coincide with each other or the diagonal, their coefficients accumulate. M_tilde is dimensionless and positive index-squared multiplication; it is not I-M_tilde or its negative.

Returns
-------
return csc_matrix((ny, ny), dtype=complex)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
from scipy.sparse import csc_matrix


def build_modulation_matrix(ny, mx, B, C):
    """Return the (ny, ny) CSC matrix for multiplication by
    (C+B*cos(mx*2*pi*x/L))**2 on a periodic Fourier collocation grid.
    ny is a positive integer, mx is an integer Fourier-index shift, and
    B and C are finite real scalars. Add C**2+B**2/2 on the diagonal,
    B*C at shifts +/-mx, and B**2/4 at shifts +/-2*mx. Accumulate all
    collisions modulo ny, including shifts that wrap to the diagonal.
    """
    return csc_matrix((ny, ny), dtype=complex)

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.sparse import coo_matrix


def _oracle_build_modulation_matrix(ny, mx, B, C):
    """Build the Fourier matrix for the squared refractive index.

    Periodic shifts contribute at 0, +/-mx, and +/-2*mx. Converting
    COO to CSC sums entries when shifts coincide, including at zero.
    """
    if not isinstance(ny, (int, np.integer)) or ny < 1:
        raise ValueError("ny must be a positive integer.")
    if not isinstance(mx, (int, np.integer)):
        raise ValueError("mx must be an integer.")

    mx = int(mx)
    rows = []
    cols = []
    values = []

    for row_index in range(ny):
        rows.append(row_index)
        cols.append(row_index)
        values.append(C**2 + B**2 / 2.0)

        rows.extend([row_index, row_index])
        cols.extend([(row_index + mx) % ny, (row_index - mx) % ny])
        values.extend([B * C, B * C])

        rows.extend([row_index, row_index])
        cols.extend([
            (row_index + 2 * mx) % ny,
            (row_index - 2 * mx) % ny,
        ])
        values.extend([B**2 / 4.0, B**2 / 4.0])

    return coo_matrix(
        (values, (rows, cols)), shape=(ny, ny), dtype=complex
    ).tocsc()

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
ny = 8
mx = 1
B = 0.1
C = 1.4


def run_model():
    return build_modulation_matrix(
        ny=ny,
        mx=mx,
        B=B,
        C=C
    ).toarray()


def run_gold():
    return _oracle_build_modulation_matrix(
        ny=ny,
        mx=mx,
        B=B,
        C=C
    ).toarray()
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
# The +mx and -mx shifts coincide.
ny = 8
mx = ny // 2
B = 0.1
C = 1.4


def run_model():
    return build_modulation_matrix(
        ny=ny,
        mx=mx,
        B=B,
        C=C
    ).toarray()


def run_gold():
    return _oracle_build_modulation_matrix(
        ny=ny,
        mx=mx,
        B=B,
        C=C
    ).toarray()
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """
# All periodic shifts wrap to the diagonal.
ny = 8
mx = ny
B = 0.1
C = 1.4


def run_model():
    return build_modulation_matrix(
        ny=ny,
        mx=mx,
        B=B,
        C=C
    ).toarray()


def run_gold():
    return _oracle_build_modulation_matrix(
        ny=ny,
        mx=mx,
        B=B,
        C=C
    ).toarray()
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]

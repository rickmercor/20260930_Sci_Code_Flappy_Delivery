"""
Compute percent lattice mismatch for electrode/channel matching cells.

Vertical heterostructure cells are built by choosing supercell repeats that minimize the relative lattice mismatch between electrode and channel lattices.

Returns
-------
float — mismatch δ in percent
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def lattice_mismatch(
    a_electrode: float,
    n_electrode: int,
    a_channel: float,
    n_channel: int,
) -> float:
    """Return lattice mismatch δ in percent.

    δ = 100*|n_electrode*a_electrode - n_channel*a_channel|/(n_channel*a_channel).

    Raises
    ------
    ValueError
        If any lattice constant is not positive, either repeat count is
        less than 1, or any input is non-finite.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_lattice_mismatch(
    a_electrode: float,
    n_electrode: int,
    a_channel: float,
    n_channel: int,
) -> float:
    a_electrode = float(a_electrode)
    a_channel = float(a_channel)
    n_electrode = int(n_electrode)
    n_channel = int(n_channel)
    if a_electrode <= 0.0 or a_channel <= 0.0:
        raise ValueError("Lattice constants must be positive.")
    if n_electrode < 1 or n_channel < 1:
        raise ValueError("Supercell repeats must be >= 1.")
    if not all(np.isfinite(v) for v in (a_electrode, a_channel)):
        raise ValueError("Lattice constants must be finite.")
    num = abs(n_electrode * a_electrode - n_channel * a_channel)
    den = n_channel * a_channel
    return float(100.0 * num / den)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np",
            "call": "lattice_mismatch(4.127, 1, 4.19, 1)",
            "gold_call": "_oracle_lattice_mismatch(4.127, 1, 4.19, 1)",
        },
        {
            "setup": "import numpy as np",
            "call": "lattice_mismatch(4.19, 1, 4.19, 1)",
            "gold_call": "_oracle_lattice_mismatch(4.19, 1, 4.19, 1)",
        },
        {
            "setup": "import numpy as np",
            "call": "lattice_mismatch(3.33, 2, 4.19, 1)",
            "gold_call": "_oracle_lattice_mismatch(3.33, 2, 4.19, 1)",
        },
    ]

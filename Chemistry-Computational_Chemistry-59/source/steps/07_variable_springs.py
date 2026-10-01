"""
Return the energy-scaled spring constant of every adjacent pair of images of a band, following the workflow's variable-spring scheme for its nudged elastic bands. There is one spring per adjacent pair, so a band of $M$ images yields $M-1$ constants.

A nudged elastic band holds its images apart with springs acting along the local tangent. With one constant for the whole band the images distribute themselves at roughly equal arc length, which spends most of them on the flat approaches and leaves the saddle, the only part of the path anyone wants, resolved by two or three images. Letting the spring constants depend on the energy profile of the band changes that distribution, and because the energy profile changes as the band relaxes, such constants have to be recomputed as the optimisation proceeds rather than fixed once.



Conventions fixed by this task. Where the source's scheme distinguishes springs lying above and below its reference energy, a spring whose attributed energy is exactly equal to the reference takes the minimum constant, and so does every spring of a band whose maximum energy does not exceed the reference.

No closed form is supplied for this step beyond the convention above: the $M-1$ constants are $k_0,\\dots,k_{M-2}$, with $k_i$ belonging to the spring between images $i$ and $i+1$, each lying between $k_{\\min}$ and $k_{\\max}$.

Returns
-------
A one-dimensional `numpy` array of length $M-1$ holding the spring constant of each adjacent image pair in eV/A$^2$.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def variable_spring_constants(energies: "np.ndarray", k_min: float = 0.1,
                              k_max: float = 4.0) -> "np.ndarray":
    """Energy-scaled nudged-elastic-band spring constants.

    Parameters
    ----------
    energies : numpy.ndarray
        Image energies in eV, shape (M,) with M >= 2, ordered from the
        initial state to the final state.
    k_min : float
        Positive spring constant in eV/A^2 assigned at and below the
        reference energy.
    k_max : float
        Spring constant in eV/A^2 assigned at the band maximum; it must
        exceed k_min.

    Returns
    -------
    k : numpy.ndarray
        Spring constants of the M - 1 adjacent image pairs, in eV/A^2.

    Raises
    ------
    ValueError
        If energies holds fewer than two images, or if k_max does not exceed
        k_min or either is not positive.
    """
    return k

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_variable_spring_constants(energies: "np.ndarray",
                                      k_min: float = 0.1,
                                      k_max: float = 4.0) -> "np.ndarray":
    energies = np.asarray(energies, dtype=float).reshape(-1)
    if energies.size < 2:
        raise ValueError("energies must hold at least two images")
    if not (k_max > k_min > 0.0):
        raise ValueError("k_max must exceed k_min and both must be positive")

    e_ref = max(float(energies[0]), float(energies[-1]))
    e_max = float(energies.max())
    e_pair = np.maximum(energies[:-1], energies[1:])
    dk = float(k_max) - float(k_min)
    k = np.full(e_pair.shape, float(k_max) - dk)
    hot = e_pair > e_ref
    if np.any(hot):
        k[hot] = k_max - dk * (e_max - e_pair[hot]) / (e_max - e_ref)
    return k

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of step test specifications (setup/call/gold_call)."""
    return [
        # normal: a twenty-image profile with an interior maximum and an
        # endothermic final state, which is what sets the reference energy
        {"setup": """import numpy as np
x = np.linspace(0.0, 1.0, 20)
e = -33.8 + 1.4 * np.sin(np.pi * x) ** 2 + 0.24 * x
""",
         "call": "variable_spring_constants(e)",
         "gold_call": "_oracle_variable_spring_constants(e)"},
        # normal: the same profile with a different constant range
        {"setup": """import numpy as np
x = np.linspace(0.0, 1.0, 20)
e = -33.8 + 1.4 * np.sin(np.pi * x) ** 2 + 0.24 * x
""",
         "call": "variable_spring_constants(e, 0.5, 1.5)",
         "gold_call": "_oracle_variable_spring_constants(e, 0.5, 1.5)"},
        # boundary: a monotonically rising profile, whose maximum is the final
        # state, so the reference energy equals the band maximum and every
        # spring falls in the lower branch
        {"setup": """import numpy as np
e = np.linspace(-2.0, 1.0, 12)
""",
         "call": "variable_spring_constants(e)",
         "gold_call": "_oracle_variable_spring_constants(e)"},
        # boundary: a flat profile, where the reference energy and the maximum
        # coincide and no spring can exceed the reference
        {"setup": """import numpy as np
e = np.full(9, -4.25)
""",
         "call": "variable_spring_constants(e)",
         "gold_call": "_oracle_variable_spring_constants(e)"},
        # boundary: a profile whose peak sits on the very first interior image,
        # so the first spring already carries the maximum
        {"setup": """import numpy as np
e = np.array([-1.0, 3.0, 2.0, 1.0, 0.5, 0.0, -0.5, -0.8])
""",
         "call": "variable_spring_constants(e, 0.1, 4.0)",
         "gold_call": "_oracle_variable_spring_constants(e, 0.1, 4.0)"},
        # edge: two images, which carry a single spring and no interior energy
        {"setup": """import numpy as np
e = np.array([-1.5, -0.25])
""",
         "call": "variable_spring_constants(e)",
         "gold_call": "_oracle_variable_spring_constants(e)"},
        # invalid: a band of one image
        {"setup": """import numpy as np
e = np.array([0.0])
def run_model():
    try:
        variable_spring_constants(e)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_variable_spring_constants(e)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
        # invalid: a maximum constant below the minimum
        {"setup": """import numpy as np
e = np.array([0.0, 1.0, 0.5])
def run_model():
    try:
        variable_spring_constants(e, 4.0, 0.1)
        return 0
    except ValueError:
        return 1
def run_oracle():
    try:
        _oracle_variable_spring_constants(e, 4.0, 0.1)
        return 0
    except ValueError:
        return 1
""",
         "call": "run_model()",
         "gold_call": "run_oracle()"},
    ]

"""
Call build_ness_symbols, fourier_covariance, partial_transpose_jet, symplectic_absolute_jet, balance_contour_jet, active_log_jet, and negativity_width_response in that order; supply both transposed covariance and absolute jets to balance_contour_jet.

The bias response combines the chiral thermal state, partial transposition, the polar geometry of a Gaussian contour, and normalization of a spatial moment. Propagating a first-order jet carries the physical response through each transformation without choosing a derivative of a degenerate mode basis.

Returns
-------
float, the complete right bias derivative of the normalized negativity width as a native Python float.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_ness_negativity_response(mass: float, coupling: float, beta_mean: float, bias: float, nq: int, positions: "np.ndarray", transpose_sites: "np.ndarray", center: float) -> float:
    """Compose the preceding seven public functions to evaluate the response.

    Parameters
    ----------
    mass : float
        Finite strictly positive oscillator frequency parameter.
    coupling : float
        Finite strictly positive nearest-neighbor spring constant.
    beta_mean : float
        Finite positive mean inverse temperature b.
    bias : float
        Finite inverse-temperature bias delta, satisfying abs(delta) < b. The left and right inverse temperatures are b-delta and b+delta. Differentiate with respect to delta holding every other input fixed.
    nq : int
        Even integer >= 8, defining q_l=-pi+2*pi*(l+0.5)/nq. The finite midpoint quadrature defines the observable exactly.
    positions : np.ndarray
        Real finite one-dimensional array of distinct integer-valued lattice sites, with at least one site and span strictly smaller than nq; boolean entries are invalid.
    transpose_sites : np.ndarray
        One-dimensional real array of distinct integer-valued local indices into positions, all between 0 and len(positions)-1. It may be empty. Boolean entries are invalid. Reverse only their momentum coordinates.
    center : float
        Fixed finite real scalar origin for the normalized second moment. Boolean scalar parameters are invalid. Use the preceding public functions in their listed order, keeping all-q-then-all-p ordering and the active-log threshold convention. Inputs are not mutated.

    Returns
    -------
    response : float
        Native Python float: the right derivative of the normalized spatial second moment of the negativity contour. The total negativity must exceed 1e-14. Compute analytic directional jets through the pipeline; finite differences are reserved for independent verification.

    Raises
    ------
    ValueError
        If any parameter violates its stated contract, a required positive definite matrix is invalid, a thermal symbol is nonfinite in floating point, or the total negativity is at most 1e-14. The numerical symmetry and threshold tolerances are inherited from preceding steps.
    """
    return response

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_ness_negativity_response(mass: float, coupling: float, beta_mean: float, bias: float, nq: int, positions: "np.ndarray", transpose_sites: "np.ndarray", center: float) -> float:
    symbols = _oracle_build_ness_symbols(mass, coupling, beta_mean, bias, nq)
    covariance = _oracle_fourier_covariance(symbols, positions)
    transposed = _oracle_partial_transpose_jet(covariance, transpose_sites)
    absolute = _oracle_symplectic_absolute_jet(transposed)
    contour = _oracle_balance_contour_jet(transposed, absolute)
    logarithm = _oracle_active_log_jet(contour)
    return _oracle_negativity_width_response(logarithm, positions, center)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\nx=np.arange(-3,3); t=np.array([3,4,5])", "call": "compute_ness_negativity_response(.45,1.,3.2,.9,128,x.copy(),t.copy(),-.5)", "gold_call": "_oracle_compute_ness_negativity_response(.45,1.,3.2,.9,128,x.copy(),t.copy(),-.5)"},
        {"setup": "import numpy as np\nx=np.arange(-3,3); t=np.array([3,4,5])", "call": "compute_ness_negativity_response(.45,1.,3.2,0.,128,x.copy(),t.copy(),-.5)", "gold_call": "_oracle_compute_ness_negativity_response(.45,1.,3.2,0.,128,x.copy(),t.copy(),-.5)"},
        {"setup": "import numpy as np\nx=np.arange(-3,3); t=np.array([3,4,5])", "call": "compute_ness_negativity_response(.45,1.,3.2,-.9,128,x.copy(),t.copy(),-.5)", "gold_call": "_oracle_compute_ness_negativity_response(.45,1.,3.2,-.9,128,x.copy(),t.copy(),-.5)"},
        {"setup": "import numpy as np\nx=np.array([-3,-2,-1,0,1,2]); t=np.array([1,3,5])", "call": "compute_ness_negativity_response(.45,1.,6.,1.2,64,x.copy(),t.copy(),-.5)", "gold_call": "_oracle_compute_ness_negativity_response(.45,1.,6.,1.2,64,x.copy(),t.copy(),-.5)"},
        {"setup": "import numpy as np\nx=np.arange(-3,3); t=np.array([],dtype=int)\ndef rejected(fn):\n    try:\n        fn(.45,1.,3.2,.9,128,x.copy(),t.copy(),-.5)\n    except ValueError:\n        return 1\n    return 0", "call": "rejected(compute_ness_negativity_response)", "gold_call": "rejected(_oracle_compute_ness_negativity_response)"},
        {"setup": "import numpy as np\nx=np.arange(-3,3); t=np.array([3,4,5])\ndef rejected(fn):\n    try:\n        fn(.45,1.,3.2,3.2,128,x.copy(),t.copy(),-.5)\n    except ValueError:\n        return 1\n    return 0", "call": "rejected(compute_ness_negativity_response)", "gold_call": "rejected(_oracle_compute_ness_negativity_response)"}
    ]

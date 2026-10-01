"""
Evaluate the donor-acceptor-distance profiles used by the rate model.

Energy coefficients are ordered from the constant term through the quartic
term about ``reference_distance``.  Coupling parameters contain the leading
amplitude followed by the linear and quadratic distance terms.  Apply the
strict critical-distance convention: distances below the threshold are
inactive and equality remains active.  Return the three energy profiles and
the nonnegative coupling in the column order declared by the signature.
Invalid or non-finite inputs raise ``ValueError``.

Returns
-------
np.ndarray of shape (n, 4), with one row per input distance and columns [V_reactant, V_MECP, V_product, |V_ab|], all in E_h.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_distance_profiles(
    distance: 'np.ndarray',
    reactant_coefficients: 'np.ndarray',
    mecp_coefficients: 'np.ndarray',
    product_coefficients: 'np.ndarray',
    coupling_parameters: 'np.ndarray',
    reference_distance: float,
    critical_distance: float,
) -> 'np.ndarray':
    '''Return the three energy profiles and critical-distance-masked coupling.

    Parameters
    ----------
    distance
        Finite, strictly increasing donor-acceptor distances in bohr with
        shape ``(n,)``.
    reactant_coefficients, mecp_coefficients, product_coefficients
        Finite quartic coefficients ``[E0, a1, a2, a3, a4]`` in the
        corresponding ``E_h bohr^-k`` units.
    coupling_parameters
        Finite parameters ``[V0, c1, c2]`` in
        ``[E_h, bohr^-1, bohr^-2]`` with ``V0 >= 0``.
    reference_distance
        Finite real origin in bohr used in
        ``y = distance - reference_distance``.
    critical_distance
        Finite real strict lower gate in bohr for the coupling.

    Returns
    -------
    np.ndarray
        Array in ``E_h`` with columns
        ``[V_reactant, V_MECP, V_product, |V_ab|]``.

    Raises
    ------
    ValueError
        If an input has the wrong shape or domain, is non-finite, or makes the
        exponential coupling overflow.
    '''
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _step3_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def _step3_numeric_vector(value: np.ndarray, length: int | None, name: str) -> np.ndarray:
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.size == 0
        or (length is not None and array.size != length)
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or np.any(~np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real vector")
    return np.asarray(array, dtype=float)


def _step3_polynomial(coefficients: np.ndarray, offset: np.ndarray) -> np.ndarray:
    return (
        coefficients[0]
        + coefficients[1] * offset
        + coefficients[2] * offset**2
        + coefficients[3] * offset**3
        + coefficients[4] * offset**4
    )


def _oracle_evaluate_distance_profiles(
    distance: 'np.ndarray',
    reactant_coefficients: 'np.ndarray',
    mecp_coefficients: 'np.ndarray',
    product_coefficients: 'np.ndarray',
    coupling_parameters: 'np.ndarray',
    reference_distance: float,
    critical_distance: float,
) -> 'np.ndarray':
    r = _step3_numeric_vector(distance, None, "distance")
    reactant = _step3_numeric_vector(reactant_coefficients, 5, "reactant_coefficients")
    mecp = _step3_numeric_vector(mecp_coefficients, 5, "mecp_coefficients")
    product = _step3_numeric_vector(product_coefficients, 5, "product_coefficients")
    coupling = _step3_numeric_vector(coupling_parameters, 3, "coupling_parameters")
    if r.size > 1 and np.any(np.diff(r) <= 0.0):
        raise ValueError("distance must be strictly increasing")
    reference = _step3_real_scalar(reference_distance, "reference_distance")
    critical = _step3_real_scalar(critical_distance, "critical_distance")
    if coupling[0] < 0.0:
        raise ValueError("V0 must be nonnegative")
    y = r - reference
    reactant_energy = _step3_polynomial(reactant, y)
    mecp_energy = _step3_polynomial(mecp, y)
    product_energy = _step3_polynomial(product, y)
    vibronic_coupling = coupling[0] * np.exp(coupling[1] * y + coupling[2] * y**2)
    vibronic_coupling = np.where(r < critical, 0.0, vibronic_coupling)
    result = np.column_stack(
        [reactant_energy, mecp_energy, product_energy, vibronic_coupling]
    )
    if np.any(~np.isfinite(result)):
        raise ValueError("profile evaluation produced a non-finite value")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": "import numpy as np\nr=np.array([4.4,4.7,5.0]); vr=np.array([0.,0.,.02,0.,.001]); vm=np.array([.004,-.001,.01,0.,.002]); vp=np.array([.001,.0002,.018,0.,.001]); c=np.array([.0004,-2.,-.3])",
            "call": "evaluate_distance_profiles(r.copy(),vr.copy(),vm.copy(),vp.copy(),c.copy(),4.8,4.5)",
            "gold_call": "_oracle_evaluate_distance_profiles(r.copy(),vr.copy(),vm.copy(),vp.copy(),c.copy(),4.8,4.5)",
            "tol": 1e-13,
        },
        {
            "setup": "import numpy as np\nr=np.array([4.5,4.6]); vr=np.zeros(5); vm=np.array([.003,0.,0.,0.,0.]); vp=np.zeros(5); c=np.array([.0002,-1.,0.])",
            "call": "evaluate_distance_profiles(r.copy(),vr.copy(),vm.copy(),vp.copy(),c.copy(),4.5,4.5)",
            "gold_call": "_oracle_evaluate_distance_profiles(r.copy(),vr.copy(),vm.copy(),vp.copy(),c.copy(),4.5,4.5)",
            "tol": 1e-13,
        },
        {
            "setup": "import numpy as np\nr=np.array([5.0]); vr=np.array([.1,.2,.3,.4,.5]); vm=vr+.01; vp=vr-.02; c=np.array([0.,-3.,-.5])",
            "call": "evaluate_distance_profiles(r.copy(),vr.copy(),vm.copy(),vp.copy(),c.copy(),5.0,4.0)",
            "gold_call": "_oracle_evaluate_distance_profiles(r.copy(),vr.copy(),vm.copy(),vp.copy(),c.copy(),5.0,4.0)",
            "tol": 1e-13,
        },
        {
            "setup": "import numpy as np\nr=np.array([4.5,4.5]); vr=np.zeros(5); vm=np.ones(5); vp=np.zeros(5); c=np.array([.001,-1.,0.])\ndef check(fn):\n try: fn(r.copy(),vr.copy(),vm.copy(),vp.copy(),c.copy(),4.5,4.4)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(evaluate_distance_profiles)",
            "gold_call": "check(_oracle_evaluate_distance_profiles)",
            "tol": 0.0,
        },
    ]

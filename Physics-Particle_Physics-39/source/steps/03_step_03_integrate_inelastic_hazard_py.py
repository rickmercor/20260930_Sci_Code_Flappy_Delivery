"""
Integrate the finite-temperature splitting and merging channels over the supplied momentum-fraction quadrature to obtain the total inelastic hazard.

The finite-temperature Sudakov factor is controlled by the sum of the splitting and merging channels. When the momentum-fraction integral is represented by discrete quadrature nodes $z_j$ with weights $w_j$, the integrated channel rates are

$$

\lambda_{\mathrm{split}}(p)=\sum_j w_j\Gamma_1(z_j,p),\qquad \lambda_{\mathrm{merge}}(p)=\sum_j w_j\Gamma_2(z_j,p).

$$

The total inelastic hazard is

$$

\lambda(p)=\lambda_{\mathrm{split}}(p)+\lambda_{\mathrm{merge}}(p).

$$

Returns
-------
np.ndarray of shape (3,), containing the integrated splitting hazard, merging hazard, and total inelastic hazard in GeV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def integrate_inelastic_hazard(
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
) -> "np.ndarray":
    """Integrate the finite-temperature inelastic shower rates.

    Parameters
    ----------
    momentum : float
        Gluon momentum in GeV.
    temperature : float
        Medium temperature in GeV.
    alpha_s : float
        Strong coupling constant.
    c_a : float
        Gluon color factor.
    qhat : float
        Transverse-momentum broadening coefficient in GeV^3.
    z_nodes : np.ndarray
        One-dimensional quadrature nodes.
    weights : np.ndarray
        Quadrature weights corresponding to z_nodes.

    Returns
    -------
    hazards : np.ndarray
        Array of shape (3,) containing the integrated splitting hazard,
        merging hazard, and their sum, in GeV.
    """
    return hazards

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_integrate_inelastic_hazard(
    momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
) -> "np.ndarray":
    z_nodes = np.asarray(z_nodes, dtype=float)
    weights = np.asarray(weights, dtype=float)

    split_hazard = 0.0
    merge_hazard = 0.0

    for z, weight in zip(z_nodes, weights):
        rates = _oracle_compute_thermal_channel_rates(
            float(z),
            momentum,
            temperature,
            alpha_s,
            c_a,
            qhat,
        )
        split_hazard += float(weight) * float(rates[0])
        merge_hazard += float(weight) * float(rates[1])

    total_hazard = split_hazard + merge_hazard

    return np.array(
        [split_hazard, merge_hazard, total_hazard],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
z_nodes = np.array([
    0.228146046218401,
    0.338459206968295,
    0.5,
    0.661540793031705,
    0.771853953781599,
], dtype=float)
weights = np.array([
    0.071078065516857,
    0.143588601149810,
    0.170666666666667,
    0.143588601149810,
    0.071078065516857,
], dtype=float)""",
            "call": "integrate_inelastic_hazard(3.0, 0.5, 0.3, 3.0, 0.015, z_nodes.copy(), weights.copy())",
            "gold_call": "_oracle_integrate_inelastic_hazard(3.0, 0.5, 0.3, 3.0, 0.015, z_nodes.copy(), weights.copy())",
        },
        {
            "setup": """import numpy as np
z_nodes = np.array([
    0.228146046218401,
    0.338459206968295,
    0.5,
    0.661540793031705,
    0.771853953781599,
], dtype=float)
weights = np.array([
    0.071078065516857,
    0.143588601149810,
    0.170666666666667,
    0.143588601149810,
    0.071078065516857,
], dtype=float)""",
            "call": "integrate_inelastic_hazard(1.015377620904885, 0.5, 0.3, 3.0, 0.015, z_nodes.copy(), weights.copy())",
            "gold_call": "_oracle_integrate_inelastic_hazard(1.015377620904885, 0.5, 0.3, 3.0, 0.015, z_nodes.copy(), weights.copy())",
        },
        {
            "setup": """import numpy as np
z_nodes = np.array([0.25, 0.5, 0.75], dtype=float)
weights = np.array([0.15, 0.30, 0.15], dtype=float)""",
            "call": "integrate_inelastic_hazard(2.5, 0.4, 0.25, 3.0, 0.012, z_nodes.copy(), weights.copy())",
            "gold_call": "_oracle_integrate_inelastic_hazard(2.5, 0.4, 0.25, 3.0, 0.012, z_nodes.copy(), weights.copy())",
        },
        {
            "setup": """import numpy as np
z_nodes = np.array([0.21, 0.34, 0.58, 0.76], dtype=float)
weights = np.array([0.07, 0.12, 0.19, 0.22], dtype=float)""",
            "call": "integrate_inelastic_hazard(2.2, 0.55, 0.27, 3.0, 0.013, z_nodes.copy(), weights.copy())",
            "gold_call": "_oracle_integrate_inelastic_hazard(2.2, 0.55, 0.27, 3.0, 0.013, z_nodes.copy(), weights.copy())",
        },
    ]

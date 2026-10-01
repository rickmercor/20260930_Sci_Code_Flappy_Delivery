"""
Determine the two daughter momenta and their finite-temperature interaction hazards after the designated splitting.

A splitting with momentum fraction $z_\star$ produces daughter momenta $p_1=z_\star p_0$ and $p_2=(1-z_\star)p_0$. Each daughter subsequently experiences its own finite-temperature competition between splitting and merging. Because the exclusive history requires both daughters to survive without another inelastic interaction, the post-splitting survival factor is controlled by the sum of their total hazards.

Returns
-------
np.ndarray of shape (5,), containing p1, p2, lambda(p1), lambda(p2), and their summed daughter hazard, all in GeV
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_daughter_shower_state(
    z_star: float,
    parent_momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
) -> "np.ndarray":
    """Evaluate daughter momenta and their total inelastic hazards.

    Parameters
    ----------
    z_star : float
        Momentum fraction carried by the first daughter.
    parent_momentum : float
        Parent gluon momentum in GeV.
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
    state : np.ndarray
        Array of shape (5,) containing [p1, p2, lambda1, lambda2,
        lambda1 + lambda2], with momenta and hazards in GeV.
    """
    return state

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_daughter_shower_state(
    z_star: float,
    parent_momentum: float,
    temperature: float,
    alpha_s: float,
    c_a: float,
    qhat: float,
    z_nodes: "np.ndarray",
    weights: "np.ndarray",
) -> "np.ndarray":
    p1 = z_star * parent_momentum
    p2 = (1.0 - z_star) * parent_momentum

    hazard_1 = _oracle_integrate_inelastic_hazard(
        p1,
        temperature,
        alpha_s,
        c_a,
        qhat,
        z_nodes,
        weights,
    )[2]

    hazard_2 = _oracle_integrate_inelastic_hazard(
        p2,
        temperature,
        alpha_s,
        c_a,
        qhat,
        z_nodes,
        weights,
    )[2]

    return np.array(
        [p1, p2, hazard_1, hazard_2, hazard_1 + hazard_2],
        dtype=float,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nz_nodes = np.array([0.228146046218401, 0.338459206968295, 0.5, 0.661540793031705, 0.771853953781599], dtype=float)\nweights = np.array([0.071078065516857, 0.143588601149810, 0.170666666666667, 0.143588601149810, 0.071078065516857], dtype=float)",
            "call": "compute_daughter_shower_state(0.338459206968295, 3.0, 0.5, 0.3, 3.0, 0.015, z_nodes.copy(), weights.copy())",
            "gold_call": "_oracle_compute_daughter_shower_state(0.338459206968295, 3.0, 0.5, 0.3, 3.0, 0.015, z_nodes.copy(), weights.copy())",
        },
        {
            "setup": "import numpy as np\nz_nodes = np.array([0.228146046218401, 0.338459206968295, 0.5, 0.661540793031705, 0.771853953781599], dtype=float)\nweights = np.array([0.071078065516857, 0.143588601149810, 0.170666666666667, 0.143588601149810, 0.071078065516857], dtype=float)",
            "call": "compute_daughter_shower_state(0.5, 3.0, 0.5, 0.3, 3.0, 0.015, z_nodes.copy(), weights.copy())",
            "gold_call": "_oracle_compute_daughter_shower_state(0.5, 3.0, 0.5, 0.3, 3.0, 0.015, z_nodes.copy(), weights.copy())",
        },
        {
            "setup": "import numpy as np\nz_nodes = np.array([0.25, 0.5, 0.75], dtype=float)\nweights = np.array([0.15, 0.30, 0.15], dtype=float)",
            "call": "compute_daughter_shower_state(0.25, 1.4, 0.55, 0.28, 3.0, 0.014, z_nodes.copy(), weights.copy())",
            "gold_call": "_oracle_compute_daughter_shower_state(0.25, 1.4, 0.55, 0.28, 3.0, 0.014, z_nodes.copy(), weights.copy())",
        },
    ]

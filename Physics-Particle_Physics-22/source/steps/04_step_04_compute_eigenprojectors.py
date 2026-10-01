"""
Construct the two real spectral matrices from electron-row cofactors.

For a simple eigenvalue, the projector is



$$

Q_i=\frac{\operatorname{adj}(\lambda_i I-K)}{\prod_{j\ne i}(\lambda_i-\lambda_j)}.

$$



Only $i=2,3$ are needed after removing a common propagation phase.

Their first rows are determined by the two coefficient rows:

$Q_{ee}=(\lambda_i^2-S_{ee}\lambda_i+T_{ee})/D_i$,

$Q_{e\mu}=(-S_{e\mu}\lambda_i+T_{e\mu})/D_i$, and

$Q_{e\tau}=(-S_{e\tau}\lambda_i+T_{e\tau})/D_i$.

Here $D_i=\prod_{j\ne i}(\lambda_i-\lambda_j)$.

Complete a symmetric matrix with $Q_{\mu\mu}=Q_{e\mu}^2/Q_{ee}$,

$Q_{\mu\tau}=Q_{e\mu}Q_{e\tau}/Q_{ee}$, and

$Q_{\tau\tau}=1-Q_{ee}-Q_{\mu\mu}$.

This trace-one completion defines the output when approximate eigenvalues are supplied.

Returns
-------
A real dimensionless array of shape (2, 3, 3) containing the second and third spectral projectors.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_eigenprojectors(
    eigenvalues_ev2: "np.ndarray", coefficients: "np.ndarray"
) -> "np.ndarray":
    r"""Construct the two real spectral matrices from electron-row cofactors.

    Parameters
    ----------
    eigenvalues_ev2 : np.ndarray
        Finite real shape (3,) array of strictly increasing eigenvalues.
    coefficients : np.ndarray
        Finite real shape (2, 3) array of electron-row S and T coefficients.

    Returns
    -------
    projectors : np.ndarray
        Real dimensionless array of shape (2, 3, 3), ordered as Q_2, Q_3.

    Raises
    ------
    ValueError
        If shapes, reality, finiteness, or strict eigenvalue ordering fail,
        if an electron diagonal entry is nonpositive, or if the completed
        matrices have nonfinite entries. The inputs must describe simple
        eigenvalues with nonzero electron projections.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compute_eigenprojectors(
    eigenvalues_ev2: "np.ndarray", coefficients: "np.ndarray"
) -> "np.ndarray":
    roots = _real_array(eigenvalues_ev2, (3,), "eigenvalues_ev2")
    coeff = _real_array(coefficients, (2, 3), "coefficients")
    if np.any(np.diff(roots) <= 0.0):
        raise ValueError("eigenvalues must be strictly increasing")
    projectors = np.empty((2, 3, 3))
    for out, index in enumerate((1, 2)):
        value = roots[index]
        denominator = np.prod(value - np.delete(roots, index))
        electron = (-value * coeff[0] + coeff[1]) / denominator
        electron[0] += value * value / denominator
        ee, em, et = electron
        if ee <= 0.0:
            raise ValueError("electron projection must be positive")
        mm = em * em / ee
        mt = em * et / ee
        tt = 1.0 - ee - mm
        projectors[out] = [[ee, em, et], [em, mm, mt], [et, mt, tt]]
    if not np.all(np.isfinite(projectors)):
        raise ValueError("projectors must be finite")
    return projectors

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return representative numerical and invalid-input cases."""
    return [
        {
            "setup": "import numpy as np\nspec_model = compute_matter_spectrum(0.00204, 0.307, 0.02195, 7.49e-5, 2.534e-3, 0)\nspec_gold = _oracle_compute_matter_spectrum(0.00204, 0.307, 0.02195, 7.49e-5, 2.534e-3, 0)\nroots_model, coeff_model = spec_model[0], spec_model[1:]\nroots_gold, coeff_gold = spec_gold[0], spec_gold[1:]\n",
            "call": "compute_eigenprojectors(roots_model.copy(), coeff_model.copy())",
            "gold_call": "_oracle_compute_eigenprojectors(roots_gold.copy(), coeff_gold.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nspec_model = compute_matter_spectrum(0.0, 0.307, 0.02195, 7.49e-5, 2.534e-3)\nspec_gold = _oracle_compute_matter_spectrum(0.0, 0.307, 0.02195, 7.49e-5, 2.534e-3)\nroots_model, coeff_model = spec_model[0], spec_model[1:]\nroots_gold, coeff_gold = spec_gold[0], spec_gold[1:]\n",
            "call": "compute_eigenprojectors(roots_model.copy(), coeff_model.copy())",
            "gold_call": "_oracle_compute_eigenprojectors(roots_gold.copy(), coeff_gold.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nspec_model = compute_matter_spectrum(-0.004, 0.4, 0.03, 8e-5, 2.4e-3)\nspec_gold = _oracle_compute_matter_spectrum(-0.004, 0.4, 0.03, 8e-5, 2.4e-3)\nroots_model, coeff_model = spec_model[0], spec_model[1:]\nroots_gold, coeff_gold = spec_gold[0], spec_gold[1:]\n",
            "call": "compute_eigenprojectors(roots_model.copy(), coeff_model.copy())",
            "gold_call": "_oracle_compute_eigenprojectors(roots_gold.copy(), coeff_gold.copy())",
            "tol": 1e-09,
        },
        {
            "setup": "import numpy as np\nroots = np.array([0.0, 1.0, 1.0])\ncoeff = np.ones((2, 3))\ndef _raises_value_error(function):\n    try:\n        function()\n    except ValueError:\n        return 1\n    return 0\n",
            "call": "_raises_value_error(lambda: compute_eigenprojectors(roots.copy(), coeff.copy()))",
            "gold_call": "_raises_value_error(lambda: _oracle_compute_eigenprojectors(roots.copy(), coeff.copy()))",
            "tol": 0.0,
        },
    ]

"""
Advance the maximum scalar history and residual deviatoric plastic tensor using the active or frozen branch.

Irreversibility is imposed by projecting the instantaneous candidate onto the previous maximum:



$$

h=\\max(h_{old},\\hat h).

$$



The branch indicator uses a strict comparison,



$$

A=\\mathbf 1_{\\hat h>h_{old}},

$$



so equality belongs to the inactive branch. The derivative passed to the active constitutive response is



$$

g_p=A\\hat h'.

$$



The residual deviatoric tensor is updated radially only when the maximum grows:



$$

E_p=\\begin{cases}

\\dfrac{h}{q}e, & A=1,\\\\

E_{p,old}, & A=0.

\\end{cases}

$$



On unloading and sub-maximum reloading, both $h$ and $E_p$ remain frozen. Retaining the tensor is essential because the scalar maximum does not encode residual direction. Reconstructing $E_p$ from the current deviator while $h$ is frozen would rotate the residual state without a new history increment. The activity output uses numerical values 0.0 and 1.0 so every public return remains directly testable.

Returns
-------
Return a tuple containing the dimensionless history, plastic tensor, branch derivative, and numerical activity flags as arrays of shapes (n_elements,), (n_elements, 3, 3), (n_elements,), and (n_elements,), respectively.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def irreversible_state(
    deviator: np.ndarray,
    equivalent_strain: np.ndarray,
    candidate: np.ndarray,
    derivative: np.ndarray,
    previous_history: np.ndarray,
    previous_plastic: np.ndarray,
) -> tuple:
    r"""Advance scalar maximum history and the residual deviatoric tensor.

    All inputs must be finite with matching element counts. Tensors must be symmetric
    and traceless within absolute tolerance 1e-10. Scalar strains/history must be
    nonnegative, derivatives in [0, 1], and active cells must have q > 0. Tensor norms
    must agree with q and previous_history within atol=1e-10, rtol=1e-8. Invalid inputs
    raise ValueError.

    Parameters
    ----------
    deviator : np.ndarray
        Symmetric traceless e, shape (n_elements, 3, 3).
    equivalent_strain : np.ndarray
        q >= 0, shape (n_elements,).
    candidate : np.ndarray
        Nonnegative instantaneous candidate, shape (n_elements,).
    derivative : np.ndarray
        Candidate derivative in [0, 1], shape (n_elements,).
    previous_history : np.ndarray
        Nonnegative stored scalar history, shape (n_elements,).
    previous_plastic : np.ndarray
        Symmetric traceless residual tensor, shape (n_elements, 3, 3).

    Raises
    ------
    ValueError
        If inputs are nonfinite or have inconsistent shapes; scalar strain, candidate,
        or history is negative; a derivative lies outside [0, 1]; a tensor is not
        symmetric and traceless; a tensor norm disagrees with its scalar measure; or
        an active state has zero equivalent strain.

    Returns
    -------
    tuple
        (history, plastic, branch_derivative, active): dimensionless arrays of shapes
        (n_elements,), (n_elements, 3, 3), (n_elements,), (n_elements,); active contains
        float 0.0 or 1.0.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _finite_array(value):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError("numeric inputs must be finite")
    return array


def _matched_vectors(*values):
    arrays = tuple(_finite_array(value) for value in values)
    if not arrays or arrays[0].ndim != 1 or arrays[0].size == 0:
        raise ValueError("expected nonempty one-dimensional arrays")
    if any(array.shape != arrays[0].shape for array in arrays):
        raise ValueError("vector shapes must match")
    return arrays


def _symmetric_tensors(value, count):
    array = _finite_array(value)
    if array.shape != (count, 3, 3):
        raise ValueError("expected shape (n_elements, 3, 3)")
    if not np.allclose(array, array.swapaxes(1, 2), rtol=0.0, atol=1e-10):
        raise ValueError("tensors must be symmetric to absolute tolerance 1e-10")
    return array


def _oracle_irreversible_state(
    deviator: np.ndarray,
    equivalent_strain: np.ndarray,
    candidate: np.ndarray,
    derivative: np.ndarray,
    previous_history: np.ndarray,
    previous_plastic: np.ndarray,
) -> tuple:
    q, candidate, derivative, old = _matched_vectors(
        equivalent_strain, candidate, derivative, previous_history
    )
    deviator = _symmetric_tensors(deviator, len(q))
    plastic_old = _symmetric_tensors(previous_plastic, len(q))
    if np.any(q < 0) or np.any(candidate < 0) or np.any(old < 0):
        raise ValueError("strains and histories must be nonnegative")
    if np.any(derivative < 0) or np.any(derivative > 1):
        raise ValueError("candidate slopes must lie in [0, 1]")
    for tensor, magnitude in [(deviator, q), (plastic_old, old)]:
        if not np.allclose(
            np.trace(tensor, axis1=1, axis2=2), 0.0, atol=1e-10, rtol=0.0
        ):
            raise ValueError("deviatoric tensors must be traceless")
        norm = np.sqrt(2.0 / 3.0 * np.sum(tensor**2, axis=(1, 2)))
        if not np.allclose(norm, magnitude, atol=1e-10, rtol=1e-8):
            raise ValueError("tensor equivalent norm disagrees with scalar history")
    active = candidate > old
    if np.any(active & (q <= 0)):
        raise ValueError("active states require positive equivalent strain")
    history = np.maximum(old, candidate)
    plastic = plastic_old.copy()
    plastic[active] = (history[active] / q[active])[:, None, None] * deviator[active]
    branch_derivative = np.where(active, derivative, 0.0)
    return history, plastic, branch_derivative, active.astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return normal, boundary, edge, and invalid-input cases."""
    return [
        {
            "setup": """
import numpy as np
B = np.diag([1., -.5, -.5])
q = np.array([.12, .06])
e = q[:, None, None] * B
old = np.array([.02, .08])
Ep = old[:, None, None] * np.diag([-.5, 1., -.5])
candidate = np.array([.07, .02])
derivative = np.array([.9, .2])
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in irreversible_state(e, q, candidate, derivative, old, Ep)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_irreversible_state(e, q, candidate, derivative, old, Ep)])",
        },
        {
            "setup": """
import numpy as np
B = np.diag([1., -.5, -.5])
q = np.array([.12, .06])
e = q[:, None, None] * B
old = np.array([.02, .08])
Ep = old[:, None, None] * np.diag([-.5, 1., -.5])
candidate = np.array([.07, .02])
derivative = np.array([.9, .2])
candidate = old.copy()
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in irreversible_state(e, q, candidate, derivative, old, Ep)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_irreversible_state(e, q, candidate, derivative, old, Ep)])",
        },
        {
            "setup": """
import numpy as np
B = np.diag([1., -.5, -.5])
q = np.array([.12, .06])
e = q[:, None, None] * B
old = np.array([.02, .08])
Ep = old[:, None, None] * np.diag([-.5, 1., -.5])
candidate = np.array([.07, .02])
derivative = np.array([.9, .2])
q[:] = 0
e[:] = 0
candidate[:] = 0
derivative[:] = 0
""",
            "call": "np.concatenate([np.asarray(x).reshape(-1) for x in irreversible_state(e, q, candidate, derivative, old, Ep)])",
            "gold_call": "np.concatenate([np.asarray(x).reshape(-1) for x in _oracle_irreversible_state(e, q, candidate, derivative, old, Ep)])",
        },
        {
            "setup": """
import numpy as np
B = np.diag([1., -.5, -.5])
q = np.array([.12, .06])
e = q[:, None, None] * B
old = np.array([.02, .08])
Ep = old[:, None, None] * np.diag([-.5, 1., -.5])
candidate = np.array([.07, .02])
derivative = np.array([.9, .2])
old[0] = -.1
def _invalid_status(function):
    try:
        function(e, q, candidate, derivative, old, Ep)
    except ValueError:
        return 1.0
    except Exception:
        return 2.0
    return 0.0
""",
            "call": "_invalid_status(irreversible_state)",
            "gold_call": "_invalid_status(_oracle_irreversible_state)",
        },
    ]

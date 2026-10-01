"""
Evaluate the structural compliance of the converged equilibrium state and solve

the adjoint system that the design derivative of that compliance requires.

Compliance is the work done by the applied load on the converged displacement,

so for a load that does not depend on the design it is the inner product of the

external nodal force vector with the converged nodal displacement. Its

derivative with respect to a design variable therefore requires the derivative

of the displacement, and obtaining that by differentiating through the Newton

iteration history is both expensive and unnecessary: the implicit function

theorem applied to the converged residual, which vanishes identically as a

function of the design, expresses the displacement derivative through the

inverse of the tangent stiffness acting on the partial derivative of the

residual with respect to the design. Contracting that expression with the

external force on the left defines an adjoint field, obtained once from the

transposed tangent system driven by the external force, after which the

derivative with respect to any design variable is a single inner product with

the corresponding residual derivative. In the large-deformation setting the

adjoint field is not the displacement field, because the converged displacement

solves a nonlinear balance rather than the linear system governed by the tangent

at the solution; identifying the two, as the self-adjoint argument of

small-strain compliance minimization allows, gives a different and incorrect

sensitivity. Constrained degrees of freedom carry no load in the adjoint

problem, so their entries in the right-hand side are set to zero and the

constrained rows of the tangent return zero adjoint components.

Returns
-------
tuple (compliance, adjoint) of a native Python float and an np.ndarray of shape (n_nodes, 2) and float64
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def compliance_and_adjoint(stiffness: np.ndarray, external_forces: np.ndarray,
                           displacement: np.ndarray, fixed_nodes: np.ndarray):
    """Evaluate the compliance and solve for its adjoint field.

    Parameters
    ----------
    stiffness : np.ndarray
        Constrained tangent stiffness, shape (2 n_nodes, 2 n_nodes).
    external_forces : np.ndarray
        External nodal forces, shape (n_nodes, 2).
    displacement : np.ndarray
        Converged nodal displacement, shape (n_nodes, 2).
    fixed_nodes : np.ndarray
        Boolean mask of constrained nodes, shape (n_nodes,).

    Returns
    -------
    objective : tuple
        The pair (compliance, adjoint), holding the work done by the external
        forces on the converged displacement as a float and the adjoint field
        of the compliance functional of shape (n_nodes, 2).

    Raises
    ------
    ValueError
        If `external_forces` is not of shape (n_nodes, 2), if `displacement`
        does not have that same shape, if `stiffness` is not of shape
        (2 n_nodes, 2 n_nodes), or if `fixed_nodes` is not of shape (n_nodes,).
    """
    return objective

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_compliance_and_adjoint(stiffness: np.ndarray, external_forces: np.ndarray,
                                   displacement: np.ndarray, fixed_nodes: np.ndarray):
    """Reference implementation."""
    stiffness = np.asarray(stiffness, dtype=float)
    forces = np.asarray(external_forces, dtype=float)
    displacement = np.asarray(displacement, dtype=float)
    fixed = np.asarray(fixed_nodes, dtype=bool)
    if forces.ndim != 2 or forces.shape[1] != 2:
        raise ValueError("external_forces must have shape (n_nodes, 2)")
    n_nodes = forces.shape[0]
    if displacement.shape != forces.shape:
        raise ValueError("displacement must have the same shape as external_forces")
    if stiffness.shape != (2 * n_nodes, 2 * n_nodes):
        raise ValueError("stiffness must have shape (2 n_nodes, 2 n_nodes)")
    if fixed.shape != (n_nodes,):
        raise ValueError("fixed_nodes must have shape (n_nodes,)")
    right_hand_side = forces.ravel().copy()
    right_hand_side[np.repeat(fixed, 2)] = 0.0
    adjoint = np.linalg.solve(stiffness.T, right_hand_side).reshape(n_nodes, 2)
    compliance = float(np.sum(forces * displacement))
    return compliance, adjoint

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

_STIFFNESS = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 2.5, -0.4, -0.9, 0.2],
                       [0.0, 0.0, -0.4, 3.1, 0.3, -1.1],
                       [0.0, 0.0, -0.9, 0.3, 1.8, -0.2],
                       [0.0, 0.0, 0.2, -1.1, -0.2, 2.4]])
_FIXED = np.array([True, False, False])


def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: a loaded structure with one node constrained ---
        {
            "setup": """import numpy as np
external_forces = np.array([[0.0, -0.0005], [0.0, -0.0012], [0.0004, -0.0009]])
displacement = np.array([[0.0, 0.0], [0.031, -0.214], [0.057, -0.398]])
fixed_nodes = _FIXED.copy()

def _report(pair):
    return [round(float(pair[0]), 10), np.round(pair[1], 10).tolist()]

def run_model():
    return _report(compliance_and_adjoint(
        _STIFFNESS, external_forces, displacement, fixed_nodes))

def run_oracle():
    return _report(_oracle_compliance_and_adjoint(
        _STIFFNESS, external_forces, displacement, fixed_nodes))
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Boundary case: no external load, so compliance and adjoint vanish ---
        {
            "setup": """import numpy as np
external_forces = np.zeros((3, 2))
displacement = np.array([[0.0, 0.0], [0.031, -0.214], [0.057, -0.398]])
fixed_nodes = _FIXED.copy()

def _report(pair):
    return [round(float(pair[0]), 10), np.round(pair[1], 10).tolist()]

def run_model():
    return _report(compliance_and_adjoint(
        _STIFFNESS, external_forces, displacement, fixed_nodes))

def run_oracle():
    return _report(_oracle_compliance_and_adjoint(
        _STIFFNESS, external_forces, displacement, fixed_nodes))
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
        # --- Edge case: a stiffness matrix inconsistent with the node count ---
        {
            "setup": """import numpy as np
external_forces = np.zeros((3, 2))
displacement = np.zeros((3, 2))
fixed_nodes = _FIXED.copy()
bad_stiffness = _STIFFNESS[:4, :4].copy()
def run_model():
    try:
        compliance_and_adjoint(bad_stiffness, external_forces, displacement, fixed_nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_compliance_and_adjoint(bad_stiffness, external_forces, displacement, fixed_nodes)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

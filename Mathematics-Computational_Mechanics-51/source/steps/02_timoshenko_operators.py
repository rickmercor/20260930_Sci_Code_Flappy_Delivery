"""
Assemble the block operators of the source's beam equations of motion for one edge. Return a stack of three 6x6 matrices: first the block-diagonal coefficient matrix pairing the force resultants with the moment resultants, second the block-diagonal coefficient matrix pairing the displacement with the rotation, and third the source's tangent-coupling operator, the linear map built from the edge tangent that couples the two three-component halves of the state in the way the source's equations of motion require. Its structure is the source's convention, so recover it from the paper rather than assuming a symmetric or diagonal form.

The beam model on an edge relates internal force and moment resultants to gradients of the displacement and rotation. Beyond those gradients the tangent direction itself enters the relations, which is what separates a shear-deformable beam from a bending-only one.

Returns
-------
return (3, 6, 6) float64: the two block coefficient matrices and the source's tangent coupling
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def timoshenko_operators(i_hat, Cn, Cm, Cu, Cr):
    """i_hat: (3,) unit tangent; Cn, Cm, Cu, Cr: (3, 3) symmetric positive
    definite blocks. Returns (3, 6, 6) float64: the two block-diagonal
    coefficient matrices followed by the tangent-coupling operator."""
    return np.zeros((3, 6, 6))

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Gold oracle: timoshenko_operators."""

import numpy as np
import warnings
warnings.filterwarnings("ignore")


def _cross(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _oracle_timoshenko_operators(i_hat, Cn, Cm, Cu, Cr):
    """Block operators of Eq 2.2a-b: Cq, Cy and the source's i-cross coupling."""
    i_hat = np.asarray(i_hat, dtype=np.float64)
    if i_hat.shape != (3,) or not np.isfinite(i_hat).all():
        raise ValueError("i_hat must be a finite 3-vector")
    if abs(float(np.linalg.norm(i_hat)) - 1.0) > 1e-9:
        raise ValueError("i_hat must be a unit tangent")
    for C in (Cn, Cm, Cu, Cr):
        if np.asarray(C).shape != (3, 3):
            raise ValueError("coefficient blocks must be 3x3")
    Cq = np.zeros((6, 6)); Cq[:3, :3] = Cn; Cq[3:, 3:] = Cm
    Cy = np.zeros((6, 6)); Cy[:3, :3] = Cu; Cy[3:, 3:] = Cr
    IX = np.zeros((6, 6)); IX[3:, :3] = _cross(i_hat)
    return np.stack([Cq, Cy, IX])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": 'import numpy as np\ni_hat=np.array([0.,0.,1.]);Cn=np.eye(3);Cm=2*np.eye(3);Cu=np.eye(3);Cr=np.eye(3)', "call": 'timoshenko_operators(i_hat, Cn, Cm, Cu, Cr)', "gold_call": '_oracle_timoshenko_operators(i_hat, Cn, Cm, Cu, Cr)', "tol": 1e-12},
        {"setup": 'import numpy as np\ni_hat=np.array([1.,1.,1.])/np.sqrt(3)\nw=np.array([1.,2.,3.]);w=w/np.linalg.norm(w);O=np.outer(w,w)\nCn=np.eye(3)+0.6*O;Cm=np.eye(3)+0.9*O;Cu=np.eye(3)+0.4*O;Cr=np.eye(3)+0.7*O', "call": 'timoshenko_operators(i_hat, Cn, Cm, Cu, Cr)', "gold_call": '_oracle_timoshenko_operators(i_hat, Cn, Cm, Cu, Cr)', "tol": 1e-12},
        {"setup": 'import numpy as np\ni_hat=np.array([0.6,-0.8,0.])\nCn=np.diag([1.,2.,3.]);Cm=np.diag([2.,1.,4.]);Cu=np.diag([1.,1.,2.]);Cr=np.diag([3.,1.,1.])', "call": 'timoshenko_operators(i_hat, Cn, Cm, Cu, Cr)', "gold_call": '_oracle_timoshenko_operators(i_hat, Cn, Cm, Cu, Cr)', "tol": 1e-12},
    ]

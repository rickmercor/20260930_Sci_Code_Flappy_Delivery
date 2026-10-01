"""
Compute one requested reconstructed non-mortar interface increment for the complete benchmark dual-mortar problem.

Use the complete source coupling, condensation, reduced solve, and eliminated-interface reconstruction from the preceding steps.

Return the signed component of the reconstructed non-mortar interface increment identified by the supplied zero-based target index.

Do not round intermediate values.

The non-mortar interface coordinates removed during static condensation are not independent unknowns of the reduced system.

After solving the retained coordinates, their physical increments are recovered from the mortar projection. A requested interface response must therefore be taken from the reconstructed field rather than directly from the reduced solution vector.

Returns
-------
A native Python float containing the signed reconstructed non-mortar interface displacement increment at target_nonmortar_dof.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def dual_mortar_target_increment(
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
    Knn: np.ndarray,
    K11: np.ndarray,
    K22: np.ndarray,
    Kn1: np.ndarray,
    Kn2: np.ndarray,
    Rn: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    target_nonmortar_dof: int,
) -> float:
    """Return one reconstructed non-mortar Newton increment."""

    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_dual_mortar_target_increment(
    length_x: float,
    length_y: float,
    n_cells_x: int,
    n_cells_y: int,
    Knn: np.ndarray,
    K11: np.ndarray,
    K22: np.ndarray,
    Kn1: np.ndarray,
    Kn2: np.ndarray,
    Rn: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
    target_nonmortar_dof: int,
) -> float:

    (
        delta_n,
        delta_1,
        delta_2,
        residual_inf,
    ) = _oracle_solve_dual_mortar_increment(
        length_x,
        length_y,
        n_cells_x,
        n_cells_y,
        Knn,
        K11,
        K22,
        Kn1,
        Kn2,
        Rn,
        R1,
        R2,
    )

    return float(
        delta_1[
            int(target_nonmortar_dof)
        ]
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    common = r"""
import numpy as np

Knn=np.array([
[22.,-3.,1.5],
[-3.,20.,-2.],
[1.5,-2.,18.]])

K11=np.array([
[18.,-2.,.5,-1.],
[-2.,17.,-1.,.4],
[.5,-1.,16.,-1.5],
[-1.,.4,-1.5,15.]])

K22=np.array([
[21.5,-2,0,0,-1.5,0,0,0,0,0,0,0],
[-2,23.5,-2,0,0,-1.5,0,0,0,0,0,0],
[0,-2,23.5,-2,0,0,-1.5,0,0,0,0,0],
[0,0,-2,21.5,0,0,0,-1.5,0,0,0,0],
[-1.5,0,0,0,23,-2,0,0,-1.5,0,0,0],
[0,-1.5,0,0,-2,25,-2,0,0,-1.5,0,0],
[0,0,-1.5,0,0,-2,25,-2,0,0,-1.5,0],
[0,0,0,-1.5,0,0,-2,23,0,0,0,-1.5],
[0,0,0,0,-1.5,0,0,0,21.5,-2,0,0],
[0,0,0,0,0,-1.5,0,0,-2,23.5,-2,0],
[0,0,0,0,0,0,-1.5,0,0,-2,23.5,-2],
[0,0,0,0,0,0,0,-1.5,0,0,-2,21.5]],
dtype=float)

Kn1=np.array([
[.8,-.4,.2,.5],
[-.3,.7,-.5,.1],
[.4,.2,.6,-.7]])

Kn2=np.array([
[.20,-.10,.05,.12,.08,-.04,.03,.06,.02,-.01,.04,.05],
[-.05,.18,-.08,.03,-.02,.11,-.04,.07,.01,.06,-.03,.09],
[.10,.02,.15,-.06,.05,.04,.12,-.03,.07,.01,.09,-.02]])

Rn=np.array([1.2,-.7,.9])
R1=np.array([.55,-.35,.25,-.45])
R2=np.array([.15,-.20,.10,-.05,.30,-.25,.12,-.18,.22,-.14,.08,-.11])
"""

    return [
        {
            "setup": common + r"""
length_x=2.4
length_y=1.6
n_cells_x=3
n_cells_y=2
target_nonmortar_dof=2
""",
            "call": """
dual_mortar_target_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2,target_nonmortar_dof)
""",
            "gold_call": """
_oracle_dual_mortar_target_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2,target_nonmortar_dof)
""",
            "tol": 5e-11,
        },
        {
            "setup": common + r"""
length_x=2.
length_y=1.5
n_cells_x=2
n_cells_y=3
target_nonmortar_dof=0

Knn=1.1*Knn
K11=.9*K11
K22=1.2*K22
Kn1=.8*Kn1
Kn2=1.1*Kn2
Rn=np.array([-.8,.6,-.4])
R1=np.array([.2,.1,-.3,.5])
R2=.5*R2[::-1]
""",
            "call": """
dual_mortar_target_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2,target_nonmortar_dof)
""",
            "gold_call": """
_oracle_dual_mortar_target_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2,target_nonmortar_dof)
""",
            "tol": 5e-11,
        },
        {
            "setup": common + r"""
length_x=3.
length_y=1.2
n_cells_x=5
n_cells_y=1
target_nonmortar_dof=3

Knn=.95*Knn
K11=1.15*K11
K22=.85*K22
Kn1=-.6*Kn1
Kn2=.7*Kn2
Rn=np.array([.5,.2,-1.])
R1=np.array([-.4,.3,.15,-.2])
R2=np.array([
-.12,.09,-.04,.18,-.07,.11,
-.15,.05,.13,-.08,.06,-.10])
""",
            "call": """
dual_mortar_target_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2,target_nonmortar_dof)
""",
            "gold_call": """
_oracle_dual_mortar_target_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2,target_nonmortar_dof)
""",
            "tol": 5e-11,
        },
        {
            "setup": common + r"""
length_x=2.4
length_y=1.6
n_cells_x=3
n_cells_y=2
target_nonmortar_dof=1

Knn=1.3*Knn
K11=1.05*K11
K22=.9*K22
Kn1=1.2*Kn1
Kn2=.85*Kn2
Rn=np.array([.3,-1.1,.7])
R1=np.array([.1,-.2,.4,-.1])
R2=np.array([
.05,.04,-.03,.02,-.06,.07,
-.02,.08,-.09,.03,.01,-.04])
""",
            "call": """
dual_mortar_target_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2,target_nonmortar_dof)
""",
            "gold_call": """
_oracle_dual_mortar_target_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2,target_nonmortar_dof)
""",
            "tol": 5e-11,
        },
    ]

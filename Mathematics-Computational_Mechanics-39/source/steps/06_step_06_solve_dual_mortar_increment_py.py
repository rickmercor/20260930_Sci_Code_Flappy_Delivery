"""
Solve the complete statically condensed dual-mortar Newton increment for the supplied interface geometry and uncoupled tangent/residual blocks.

Construct the source mortar projection, form the source condensed Newton system, solve the retained linear system directly, and reconstruct the eliminated non-mortar interface increment using the mortar projection.

Return the uninvolved increment, reconstructed non-mortar increment, mortar-side increment, and the infinity norm of the reduced Newton residual.

Do not round intermediate values.

Static condensation reduces the constrained system to a smaller set of independent coordinates.

After the retained increment has been solved, eliminated interface coordinates can be recovered from the projection relation used during condensation. A residual norm provides a numerical consistency check on the reduced solve.

Returns
-------
A tuple (delta_n, delta_1, delta_2, residual_inf).  delta_n contains the uninvolved retained increment.  delta_1 contains the reconstructed four-component non-mortar interface increment.  delta_2 contains the retained mortar-side increment.  residual_inf is a native float containing the infinity norm of the reduced Newton residual.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_dual_mortar_increment(
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return retained and reconstructed Newton increments."""

    return (
        np.zeros(Knn.shape[0], dtype=float),
        np.zeros(4, dtype=float),
        np.zeros(K22.shape[0], dtype=float),
        0.0,
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_dual_mortar_increment(
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
):

    (
        D,
        A,
        P,
        n_gauss,
    ) = _oracle_assemble_dual_mortar_projection(
        length_x,
        length_y,
        n_cells_x,
        n_cells_y,
    )

    (
        K_reduced,
        R_reduced,
    ) = _oracle_condense_dual_mortar_newton(
        P,
        Knn,
        K11,
        K22,
        Kn1,
        Kn2,
        Rn,
        R1,
        R2,
    )

    delta_reduced = np.linalg.solve(
        K_reduced,
        -R_reduced,
    )

    n = np.asarray(Knn).shape[0]

    delta_n = (
        delta_reduced[:n]
        .copy()
    )

    delta_2 = (
        delta_reduced[n:]
        .copy()
    )

    delta_1 = (
        P @ delta_2
    )

    residual_inf = float(
        np.max(
            np.abs(
                K_reduced
                @ delta_reduced
                + R_reduced
            )
        )
    )

    return (
        delta_n,
        delta_1,
        delta_2,
        residual_inf,
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
""",
            "call": """
(lambda z: np.concatenate((
z[0],z[1],z[2],[z[3]]
)))(solve_dual_mortar_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "gold_call": """
(lambda z: np.concatenate((
z[0],z[1],z[2],[z[3]]
)))(_oracle_solve_dual_mortar_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "tol": 5e-11,
        },
        {
            "setup": common + r"""
length_x=2.
length_y=1.5
n_cells_x=2
n_cells_y=3

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
(lambda z: np.concatenate((z[0],z[1],z[2],[z[3]])))(
solve_dual_mortar_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "gold_call": """
(lambda z: np.concatenate((z[0],z[1],z[2],[z[3]])))(
_oracle_solve_dual_mortar_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "tol": 5e-11,
        },
        {
            "setup": common + r"""
length_x=3.
length_y=1.2
n_cells_x=5
n_cells_y=1

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
(lambda z: np.concatenate((z[0],z[1],z[2],[z[3]])))(
solve_dual_mortar_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "gold_call": """
(lambda z: np.concatenate((z[0],z[1],z[2],[z[3]])))(
_oracle_solve_dual_mortar_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "tol": 5e-11,
        },
        {
            "setup": common + r"""
length_x=2.4
length_y=1.6
n_cells_x=3
n_cells_y=2

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
(lambda z: np.concatenate((z[0],z[1],z[2],[z[3]])))(
solve_dual_mortar_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "gold_call": """
(lambda z: np.concatenate((z[0],z[1],z[2],[z[3]])))(
_oracle_solve_dual_mortar_increment(
length_x,length_y,n_cells_x,n_cells_y,
Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "tol": 5e-11,
        },
    ]

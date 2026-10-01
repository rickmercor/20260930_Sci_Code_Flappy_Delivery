"""
Construct the statically condensed Newton system corresponding to the cited dual-mortar elimination procedure.

Use the supplied mortar projection and uncoupled tangent/residual blocks.

Eliminate the non-mortar interface degrees of freedom according to the source relation and retain the uninvolved and mortar-side coordinates.

Return the reduced tangent matrix and reduced residual in that retained ordering.

A mortar constraint can be used to express one interface field in terms of the opposing interface field.

Substituting this relationship into the unconstrained Newton system removes both the eliminated interface coordinates and the Lagrange-multiplier unknowns. The resulting reduced tangent contains projection-induced coupling terms.

Returns
-------
A tuple (K_reduced, R_reduced).  The retained ordering is [uninvolved coordinates, mortar-side coordinates].  K_reduced has shape (n+n2,n+n2).  R_reduced has shape (n+n2,).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def condense_dual_mortar_newton(
    P: np.ndarray,
    Knn: np.ndarray,
    K11: np.ndarray,
    K22: np.ndarray,
    Kn1: np.ndarray,
    Kn2: np.ndarray,
    Rn: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source statically condensed tangent and residual."""

    n = np.asarray(Knn).shape[0]
    n2 = np.asarray(K22).shape[0]

    return (
        np.zeros(
            (n + n2, n + n2),
            dtype=float,
        ),
        np.zeros(
            n + n2,
            dtype=float,
        ),
    )

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_condense_dual_mortar_newton(
    P: np.ndarray,
    Knn: np.ndarray,
    K11: np.ndarray,
    K22: np.ndarray,
    Kn1: np.ndarray,
    Kn2: np.ndarray,
    Rn: np.ndarray,
    R1: np.ndarray,
    R2: np.ndarray,
):

    P = np.asarray(
        P,
        dtype=float,
    )

    Knn = np.asarray(
        Knn,
        dtype=float,
    )

    K11 = np.asarray(
        K11,
        dtype=float,
    )

    K22 = np.asarray(
        K22,
        dtype=float,
    )

    Kn1 = np.asarray(
        Kn1,
        dtype=float,
    )

    Kn2 = np.asarray(
        Kn2,
        dtype=float,
    )

    top_right = (
        Kn2
        + Kn1 @ P
    )

    bottom_left = (
        Kn2.T
        + P.T @ Kn1.T
    )

    bottom_right = (
        K22
        + P.T @ K11 @ P
    )

    K_reduced = np.block(
        [
            [
                Knn,
                top_right,
            ],
            [
                bottom_left,
                bottom_right,
            ],
        ]
    )

    R_reduced = np.concatenate(
        (
            np.asarray(
                Rn,
                dtype=float,
            ),
            np.asarray(
                R2,
                dtype=float,
            )
            - P.T
            @ np.asarray(
                R1,
                dtype=float,
            ),
        )
    )

    return (
        K_reduced,
        R_reduced,
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            "setup": """
import numpy as np
P=np.array([[.6,.3,.1],[.1,.4,.5]])
Knn=np.array([[5.,-.4],[-.4,4.]])
K11=np.array([[3.,-.2],[-.2,2.5]])
K22=np.array([[4.,-.3,0.],[-.3,3.5,-.2],[0.,-.2,3.]])
Kn1=np.array([[.2,-.1],[.05,.3]])
Kn2=np.array([[.1,.04,-.02],[-.03,.08,.06]])
Rn=np.array([.4,-.2])
R1=np.array([.1,-.15])
R2=np.array([.05,-.08,.12])
""",
            "call": """
(lambda z: np.concatenate((z[0].ravel(),z[1])))(
condense_dual_mortar_newton(
P,Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "gold_call": """
(lambda z: np.concatenate((z[0].ravel(),z[1])))(
_oracle_condense_dual_mortar_newton(
P,Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "tol": 1e-12,
        },
        {
            "setup": """
import numpy as np
P=np.array([[.7,.3],[.25,.75]])
Knn=np.array([[6.]])
K11=np.array([[4.,-.5],[-.5,3.]])
K22=np.array([[5.,-.4],[-.4,4.5]])
Kn1=np.array([[.2,-.1]])
Kn2=np.array([[.05,.12]])
Rn=np.array([-.3])
R1=np.array([.2,-.1])
R2=np.array([.06,-.04])
""",
            "call": """
(lambda z: np.concatenate((z[0].ravel(),z[1])))(
condense_dual_mortar_newton(
P,Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "gold_call": """
(lambda z: np.concatenate((z[0].ravel(),z[1])))(
_oracle_condense_dual_mortar_newton(
P,Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "tol": 1e-12,
        },
        {
            "setup": """
import numpy as np
P=np.array([[.4,.3,.2,.1],[.1,.2,.3,.4]])
Knn=np.array([[7.,-.5,.2],[-.5,6.,-.3],[.2,-.3,5.]])
K11=np.array([[3.5,-.2],[-.2,3.2]])
K22=np.diag([4.,4.5,5.,5.5])
Kn1=np.array([[.1,.2],[-.2,.05],[.12,-.08]])
Kn2=np.array([
[.04,.02,-.01,.03],
[-.02,.05,.06,-.03],
[.01,-.04,.02,.07]])
Rn=np.array([.2,-.4,.3])
R1=np.array([-.1,.25])
R2=np.array([.03,-.02,.05,-.01])
""",
            "call": """
(lambda z: np.concatenate((z[0].ravel(),z[1])))(
condense_dual_mortar_newton(
P,Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "gold_call": """
(lambda z: np.concatenate((z[0].ravel(),z[1])))(
_oracle_condense_dual_mortar_newton(
P,Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "tol": 1e-12,
        },
        {
            "setup": """
import numpy as np
P=np.array([
[.6,.3,.1],
[.2,.5,.3],
[.1,.25,.65]])
Knn=np.array([[5.,-.2],[-.2,4.5]])
K11=np.array([
[4.,-.3,.1],
[-.3,3.8,-.2],
[.1,-.2,3.6]])
K22=np.array([
[5.,-.4,.1],
[-.4,4.8,-.3],
[.1,-.3,4.6]])
Kn1=np.array([[.1,-.05,.08],[-.03,.12,-.06]])
Kn2=np.array([[.04,.07,-.02],[-.01,.03,.05]])
Rn=np.array([-.2,.35])
R1=np.array([.15,-.1,.05])
R2=np.array([-.04,.06,.02])
""",
            "call": """
(lambda z: np.concatenate((z[0].ravel(),z[1])))(
condense_dual_mortar_newton(
P,Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "gold_call": """
(lambda z: np.concatenate((z[0].ravel(),z[1])))(
_oracle_condense_dual_mortar_newton(
P,Knn,K11,K22,Kn1,Kn2,Rn,R1,R2))
""",
            "tol": 1e-12,
        },
    ]

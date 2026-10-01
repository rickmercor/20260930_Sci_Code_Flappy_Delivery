"""
Compute the constrained generalized displacement increment produced by an incremental generalized load about a prescribed nonlinear Q4BW state.

Construct the consistent tangent at the supplied current state using the preceding step.

Impose zero increments at the supplied constrained zero-based generalized coordinates, solve the reduced tangent system for the remaining free increments, and restore the complete twenty-component incremental vector with zeros at all constrained coordinates.

Return the signed increment associated with target_dof.

Use a direct linear solve and do not round intermediate values.

A consistent nonlinear tangent describes the first-order response of a structure about its current configuration.

After essential constraints remove rigid and prescribed modes, the tangent can be restricted to the free generalized coordinates. Solving the reduced incremental system provides the local Newton or perturbation response to a small additional generalized load.

Returns
-------
A native Python float containing the signed generalized displacement increment at target_dof.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_q4bw_increment(
    a: float,
    b: float,
    A: np.ndarray,
    D: np.ndarray,
    As: np.ndarray,
    NT: np.ndarray,
    MT: np.ndarray,
    q: np.ndarray,
    delta_p: np.ndarray,
    fixed_dofs: np.ndarray,
    target_dof: int,
) -> float:
    """Return the signed increment at target_dof."""

    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_q4bw_increment(
    a: float,
    b: float,
    A: np.ndarray,
    D: np.ndarray,
    As: np.ndarray,
    NT: np.ndarray,
    MT: np.ndarray,
    q: np.ndarray,
    delta_p: np.ndarray,
    fixed_dofs: np.ndarray,
    target_dof: int,
) -> float:

    delta_p = np.asarray(
        delta_p,
        dtype=float,
    )

    fixed_dofs = np.asarray(
        fixed_dofs,
        dtype=int,
    )

    (
        residual,
        K,
        potential,
    ) = _oracle_assemble_q4bw_tangent(
        a,
        b,
        A,
        D,
        As,
        NT,
        MT,
        q,
    )

    free_mask = np.ones(
        20,
        dtype=bool,
    )

    free_mask[
        fixed_dofs
    ] = False

    free = np.flatnonzero(
        free_mask
    )

    if target_dof not in free:
        raise ValueError(
            "target_dof must be free"
        )

    Kff = K[
        np.ix_(
            free,
            free,
        )
    ]

    dq = np.zeros(
        20,
        dtype=float,
    )

    dq[free] = np.linalg.solve(
        Kff,
        delta_p[free],
    )

    return float(
        dq[target_dof]
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    common = """
a=2.4
b=1.6
A=np.array([
[65281.52476946175,19980.10937164915,0.0],
[19980.10937164915,14616.71670598327,0.0],
[0.0,0.0,21953.51704911001]])
D=np.array([
[3481.6813210379605,1065.6058331546217,1292.1589332648687],
[1065.6058331546217,779.5582243191081,462.9215011685319],
[1292.1589332648687,462.9215011685319,1170.8542426192012]])
As=np.array([[2833.333333333334,0.0],[0.0,2500.0]])
NT=np.array([3.457656015440701,7.220169418829079,0.0])
MT=np.zeros(3)
q=np.array([
.018,.010,-.006,.027,.006,-.004,.041,-.003,.009,.022,-.005,.007,
.0015,-.0008,.0024,-.0011,.0032,.0017,.0011,.0013])
"""

    return [
        {
            "setup": common + """
fixed_dofs=np.array([0,1,2,3,9,12,13,15],dtype=int)
delta_p=np.zeros(20)
delta_p[[6,8,10,14,17,19]]=[-8.0,1.8,-2.3,3.1,-1.7,2.4]
target_dof=6
""",
            "call": """
solve_q4bw_increment(
a,b,A,D,As,NT,MT,q,delta_p,fixed_dofs,target_dof)
""",
            "gold_call": """
_oracle_solve_q4bw_increment(
a,b,A,D,As,NT,MT,q,delta_p,fixed_dofs,target_dof)
""",
            "tol": 5e-10,
        },
        {
            "setup": common + """
fixed_dofs=np.array([0,1,2,3,6,12,13,15],dtype=int)
delta_p=np.zeros(20)
delta_p[[4,5,7,10,14,18]]=[1.2,-2.1,3.4,-1.5,2.2,-0.9]
target_dof=10
""",
            "call": """
solve_q4bw_increment(
a,b,A,D,As,NT,MT,q,delta_p,fixed_dofs,target_dof)
""",
            "gold_call": """
_oracle_solve_q4bw_increment(
a,b,A,D,As,NT,MT,q,delta_p,fixed_dofs,target_dof)
""",
            "tol": 5e-10,
        },
        {
            "setup": common + """
fixed_dofs=np.array([0,1,2,3,4,9,10,12,13,15],dtype=int)
delta_p=np.zeros(20)
delta_p[[5,6,8,11,16,19]]=[-1.8,-4.2,2.6,0.7,1.5,-1.1]
target_dof=6
""",
            "call": """
solve_q4bw_increment(
a,b,A,D,As,NT,MT,q,delta_p,fixed_dofs,target_dof)
""",
            "gold_call": """
_oracle_solve_q4bw_increment(
a,b,A,D,As,NT,MT,q,delta_p,fixed_dofs,target_dof)
""",
            "tol": 5e-10,
        },
        {
            "setup": common + """
fixed_dofs=np.array([0,1,2,6,9,12,13,15],dtype=int)
delta_p=np.zeros(20)
delta_p[[3,4,7,8,17,18]]=[-2.5,1.0,2.0,-1.7,0.9,-1.3]
target_dof=7
""",
            "call": """
solve_q4bw_increment(
a,b,A,D,As,NT,MT,q,delta_p,fixed_dofs,target_dof)
""",
            "gold_call": """
_oracle_solve_q4bw_increment(
a,b,A,D,As,NT,MT,q,delta_p,fixed_dofs,target_dof)
""",
            "tol": 5e-10,
        },
    ]

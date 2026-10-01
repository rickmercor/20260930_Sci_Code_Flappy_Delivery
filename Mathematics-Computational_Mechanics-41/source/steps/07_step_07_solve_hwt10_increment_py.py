"""
Compute a constrained linearized incremental displacement using the condensed nonlinear HWT10 tangent.



Construct the tangent at the supplied current state using the preceding HWT10 steps. Remove the prescribed displacement degrees of freedom, solve the reduced tangent system for the free displacement increment, and restore the full 30-component displacement vector with zero increments at the constrained coordinates.



Return the signed displacement increment associated with target_dof.



Use a direct linear solve. Do not replace the HWT10 tangent by a standard displacement-element stiffness or by a pseudoinverse.

After element-local mixed variables have been condensed, the resulting tangent acts only on nodal displacement degrees of freedom.



Essential boundary conditions remove rigid-body motions and define a nonsingular reduced system. Solving the reduced tangent equation gives the first-order incremental structural response about the prescribed nonlinear state.

Returns
-------
A native Python float containing the signed displacement increment at target_dof.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def solve_hwt10_increment(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
    load: np.ndarray,
    fixed_dofs: np.ndarray,
    target_dof: int,
) -> float:
    """Return the signed tangent displacement increment at target_dof."""

    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_solve_hwt10_increment(
    coords: np.ndarray,
    disp: np.ndarray,
    young: float,
    poisson: float,
    load: np.ndarray,
    fixed_dofs: np.ndarray,
    target_dof: int,
) -> float:

    load = np.asarray(
        load,
        dtype=float,
    )

    if load.shape != (30,):
        raise ValueError(
            "load must have shape (30,)"
        )

    fixed_dofs = np.asarray(
        fixed_dofs,
        dtype=int,
    )

    if (
        fixed_dofs.ndim != 1
        or np.unique(
            fixed_dofs
        ).size
        != fixed_dofs.size
    ):
        raise ValueError(
            "invalid fixed_dofs"
        )

    if np.any(
        (fixed_dofs < 0)
        | (fixed_dofs >= 30)
    ):
        raise ValueError(
            "fixed DOF outside range"
        )

    target_dof = int(
        target_dof
    )

    if not (
        0 <= target_dof < 30
    ):
        raise ValueError(
            "invalid target_dof"
        )

    if target_dof in set(
        fixed_dofs.tolist()
    ):
        raise ValueError(
            "target DOF is fixed"
        )

    (
        K_tangent,
        K_material,
        K_geometric,
    ) = _oracle_assemble_hwt10_condensed_tangent(
        coords,
        disp,
        young,
        poisson,
    )

    free_mask = np.ones(
        30,
        dtype=bool,
    )

    free_mask[
        fixed_dofs
    ] = False

    free = np.flatnonzero(
        free_mask
    )

    K_ff = K_tangent[
        np.ix_(
            free,
            free,
        )
    ]

    du = np.zeros(
        30,
        dtype=float,
    )

    du[free] = np.linalg.solve(
        K_ff,
        load[free],
    )

    return float(
        du[target_dof]
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():

    fixed = """
fixed_dofs=np.array([0,1,2,4,5,8],dtype=int)
"""

    return [
        {
            "setup": fixed + """
coords=np.array([
[0,0,0],[1.2,.1,0],[.1,1.1,.05],[0,.15,.95],
[.62,.02,.03],[.68,.62,-.02],[.02,.58,.04],
[-.03,.08,.50],[.58,.15,.50],[.06,.62,.52]],float)
X,Y,Z=coords[:,0],coords[:,1],coords[:,2]
disp=np.column_stack([
.08*X+.025*Y+.015*X*Z,
-.035*Y+.020*Z+.010*X*Y,
.060*Z-.018*X+.012*Y*Z])
young=1000.0
poisson=.30
load=np.zeros(30)
load[[3,7,11,29]]=[3,-2,-12,-5]
target_dof=29
""",
            "call": """
solve_hwt10_increment(
coords,disp,young,poisson,
load,fixed_dofs,target_dof)
""",
            "gold_call": """
_oracle_solve_hwt10_increment(
coords,disp,young,poisson,
load,fixed_dofs,target_dof)
""",
            "tol": 5e-10,
        },
        {
            "setup": fixed + """
coords=np.array([
[0,0,0],[1,.08,.02],[.04,.92,.10],[.06,.02,1.05],
[.5,.04,.01],[.52,.5,.06],[.02,.46,.05],
[.03,.01,.525],[.53,.05,.535],[.05,.47,.575]],float)
X,Y,Z=coords[:,0],coords[:,1],coords[:,2]
disp=.55*np.column_stack([
.08*X+.025*Y+.015*X*Z,
-.035*Y+.020*Z+.010*X*Y,
.060*Z-.018*X+.012*Y*Z])
young=800.0
poisson=.25
load=np.zeros(30)
load[[3,7,11,26]]=[1.5,-1,-7,-2.5]
target_dof=11
""",
            "call": """
solve_hwt10_increment(
coords,disp,young,poisson,
load,fixed_dofs,target_dof)
""",
            "gold_call": """
_oracle_solve_hwt10_increment(
coords,disp,young,poisson,
load,fixed_dofs,target_dof)
""",
            "tol": 5e-10,
        },
        {
            "setup": fixed + """
coords=np.array([
[0,0,0],[.85,-.05,.03],[.12,1.15,-.02],[-.04,.1,.88],
[.445,-.04,.025],[.475,.57,-.01],[.075,.585,-.005],
[-.03,.065,.46],[.425,.015,.45],[.025,.63,.445]],float)
X,Y,Z=coords[:,0],coords[:,1],coords[:,2]
disp=.70*np.column_stack([
.08*X+.025*Y+.015*X*Z,
-.035*Y+.020*Z+.010*X*Y,
.060*Z-.018*X+.012*Y*Z])
young=1500.0
poisson=.32
load=np.zeros(30)
load[[3,7,11,29,25]]=[4,-3,-10,-6,2]
target_dof=29
""",
            "call": """
solve_hwt10_increment(
coords,disp,young,poisson,
load,fixed_dofs,target_dof)
""",
            "gold_call": """
_oracle_solve_hwt10_increment(
coords,disp,young,poisson,
load,fixed_dofs,target_dof)
""",
            "tol": 5e-10,
        },
        {
            "setup": fixed + """
coords=np.array([
[0,0,0],[1.2,.1,0],[.1,1.1,.05],[0,.15,.95],
[.62,.02,.03],[.68,.62,-.02],[.02,.58,.04],
[-.03,.08,.50],[.58,.15,.50],[.06,.62,.52]],float)
disp=np.zeros_like(coords)
young=600.0
poisson=.45
load=np.zeros(30)
load[[3,7,11,29]]=[2,-1,-8,-4]
target_dof=29
""",
            "call": """
solve_hwt10_increment(
coords,disp,young,poisson,
load,fixed_dofs,target_dof)
""",
            "gold_call": """
_oracle_solve_hwt10_increment(
coords,disp,young,poisson,
load,fixed_dofs,target_dof)
""",
            "tol": 5e-10,
        },
    ]

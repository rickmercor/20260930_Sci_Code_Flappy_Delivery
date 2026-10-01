"""
Runs the complete component-wise strain-space hyperreduction pipeline.

The final computation depends on the consistency across components, nonlinear constitutive state construction, exact constitutive differentiation, weighted hyperreduction, global assembly, and Newton solution.

Returns
-------
float, the first coordinate after one exact end-to-end Newton update
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def first_newton_coordinate(y: np.ndarray, d: np.ndarray,
                            component_maps: np.ndarray, point_sets: list,
                            B1: np.ndarray, B2: np.ndarray,
                            L1: np.ndarray, L2: np.ndarray,
                            f_ext: np.ndarray,
                            c1: float, c2: float, kappa: float):
    """Run the complete component-wise strain-space hyperreduction pipeline.

    Parameters
    ----------
    y : np.ndarray
        Initial global reduced state with shape (2,).
    d : np.ndarray
        Boundary parameters with shape (2,).
    component_maps : np.ndarray
        Component coordinate maps with shape (n, 2, 2).
    point_sets : list[np.ndarray]
        One selected-point array per component, with rows
        [a, b, ell1, ell2, weight].
    B1 : np.ndarray
        First reduced deformation-mode matrix with shape (3, 3).
    B2 : np.ndarray
        Second reduced deformation-mode matrix with shape (3, 3).
    L1 : np.ndarray
        First boundary-lifting matrix with shape (3, 3).
    L2 : np.ndarray
        Second boundary-lifting matrix with shape (3, 3).
    f_ext : np.ndarray
        External reduced load with shape (2,).
    c1 : float
        First positive Mooney-Rivlin material coefficient.
    c2 : float
        Second positive Mooney-Rivlin material coefficient.
    kappa : float
        Positive bulk-modulus parameter.

    Returns
    -------
    value : float
        First coordinate of the reduced state after one exact undamped Newton update.

    Raises
    ------
    ValueError : If any supplied data violate the dimensional, finiteness, positivity,
        constitutive-admissibility, or solvability requirements of the
        end-to-end reduced-order pipeline.
    """
    return value

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_first_newton_coordinate(y, d, component_maps, point_sets,
                                    B1, B2, L1, L2, f_ext,
                                    c1, c2, kappa):
    local_residuals = []
    local_jacobians = []

    for C, points in zip(component_maps, point_sets):
        F_points, phis = _oracle_component_point_kinematics(
            C, y, d, points, B1, B2, L1, L2
        )

        stresses = []
        tangent_actions = []
        for F, point in zip(F_points, points):
            a, b = point[0], point[1]
            directions = np.stack((a * B1, b * B2))
            state, differential_state = _oracle_mooney_rivlin_state(
                F, directions
            )
            P, dP = _oracle_mooney_rivlin_constitutive(
                F, directions, state, differential_state, c1, c2, kappa
            )
            stresses.append(P)
            tangent_actions.append(dP)

        r_local, K_local = _oracle_component_hyperreduced_system(
            phis,
            np.asarray(points)[:, 4],
            np.asarray(stresses),
            np.asarray(tangent_actions)
        )
        local_residuals.append(r_local)
        local_jacobians.append(K_local)

    r, K = _oracle_global_reduced_system(
        component_maps,
        np.asarray(local_residuals),
        np.asarray(local_jacobians),
        f_ext
    )
    y_new, diagnostics = _oracle_newton_update_diagnostics(y, r, K)
    return float(y_new[0])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {
            # Case 1: prompt configuration
            'setup':"""import numpy as np
y=np.array([.018,-.012])
d=np.array([.04,-.03])
component_maps=np.array([[[1.,.35],[-.25,.9]],[[.8,-.4],[.3,1.1]]])
point_sets=[np.array([[1.,.7,.8,-.2,.55],[-.6,1.1,.3,.9,.85]]),np.array([[.9,-.8,1.2,.4,.65],[1.3,.5,-.4,1.,.95]])]
B1=np.array([[.6,.15,0],[.15,-.2,.05],[0,.05,.1]])
B2=np.array([[-.1,.2,.04],[.2,.5,0],[.04,0,-.3]])
L1=np.array([[.4,.1,0],[.1,-.1,.02],[0,.02,-.2]])
L2=np.array([[-.2,0,.05],[0,.3,.08],[.05,.08,.1]])
f_ext=np.array([.08,-.03])
c1,c2,kappa=1.3,.7,250.""",
            'call':'first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)',
            'gold_call':'_oracle_first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)'
        },
        {
            # Case 2: changed state, boundary values, and load
            'setup':"""import numpy as np
y=np.array([.006,.009])
d=np.array([.015,.02])
component_maps=np.array([[[1.,.35],[-.25,.9]],[[.8,-.4],[.3,1.1]]])
point_sets=[np.array([[1.,.7,.8,-.2,.55],[-.6,1.1,.3,.9,.85]]),np.array([[.9,-.8,1.2,.4,.65],[1.3,.5,-.4,1.,.95]])]
B1=np.array([[.6,.15,0],[.15,-.2,.05],[0,.05,.1]])
B2=np.array([[-.1,.2,.04],[.2,.5,0],[.04,0,-.3]])
L1=np.array([[.4,.1,0],[.1,-.1,.02],[0,.02,-.2]])
L2=np.array([[-.2,0,.05],[0,.3,.08],[.05,.08,.1]])
f_ext=np.array([-.03,.04])
c1,c2,kappa=1.3,.7,250.""",
            'call':'first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)',
            'gold_call':'_oracle_first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)'
        },
        {
            # Case 3: three-component pipeline
            'setup':"""import numpy as np
y=np.array([-.004,.011])
d=np.array([.02,-.01])
component_maps=np.array([[[.9,.1],[-.15,1.05]],[[1.1,-.2],[.25,.85]],[[.7,.3],[-.2,1.2]]])
point_sets=[np.array([[1.,.7,.8,-.2,.4],[-.6,1.1,.3,.9,1.1]]),np.array([[.9,-.8,1.2,.4,.75],[1.3,.5,-.4,1.,.6]]),np.array([[.5,.9,.2,-.5,.7],[-.8,.4,.6,.3,.9]])]
B1=np.array([[.6,.15,0],[.15,-.2,.05],[0,.05,.1]])
B2=np.array([[-.1,.2,.04],[.2,.5,0],[.04,0,-.3]])
L1=np.array([[.4,.1,0],[.1,-.1,.02],[0,.02,-.2]])
L2=np.array([[-.2,0,.05],[0,.3,.08],[.05,.08,.1]])
f_ext=np.array([.08,-.03])
c1,c2,kappa=1.1,.5,180.""",
            'call':'first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)',
            'gold_call':'_oracle_first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)'
        },
        {
            # Case 4: invalid component map shape
            'setup':"""import numpy as np
y=np.array([.018,-.012])
d=np.array([.04,-.03])
component_maps=np.array([np.eye(3)])
point_sets=[np.array([[1.,.7,.8,-.2,.55]])]
B1=B2=L1=L2=np.eye(3)
f_ext=np.array([.08,-.03])
c1,c2,kappa=1.3,.7,250.
def run_model():
    try:
        first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call':'run_model()',
            'gold_call':'run_oracle()'
        },
        {
            # Case 5: invalid nonpositive hyperreduction weight
            'setup':"""import numpy as np
y=np.array([.018,-.012])
d=np.array([.04,-.03])
component_maps=np.array([np.eye(2)])
point_sets=[np.array([[1.,.7,.8,-.2,0.]])]
B1=B2=L1=L2=np.eye(3)
f_ext=np.array([.08,-.03])
c1,c2,kappa=1.3,.7,250.
def run_model():
    try:
        first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call':'run_model()',
            'gold_call':'run_oracle()'
        },
        {
            # Case 6: invalid nonpositive material parameter
            'setup':"""import numpy as np
y=np.array([.018,-.012])
d=np.array([.04,-.03])
component_maps=np.array([np.eye(2)])
point_sets=[np.array([[1.,.7,.8,-.2,.55]])]
B1=B2=L1=L2=np.eye(3)
f_ext=np.array([.08,-.03])
c1,c2,kappa=0.,.7,250.
def run_model():
    try:
        first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_oracle():
    try:
        _oracle_first_newton_coordinate(y,d,component_maps,point_sets,B1,B2,L1,L2,f_ext,c1,c2,kappa)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2""",
            'call':'run_model()',
            'gold_call':'run_oracle()'
        }
    ]

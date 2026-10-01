"""
Run the block-alternating calibration loop for the non-equilibrium branch and

return the adapted upper endpoint of the dissipation-potential interpolation

domain.

The two blocks of the scheme are the constitutive representation and the

interpolation domains, and they are alternated rather than optimised together

because doing both at once leaves a scaling ambiguity: a wider domain

compensated by smaller coefficients reproduces almost the same constitutive

response, which creates flat directions and ill-conditions the inverse problem.

One outer iteration therefore holds the representation fixed, runs the forward

solve over the entire stretch history to find out which invariants the branch

actually visits, moves each domain endpoint towards the smooth upper-tail

statistic of those samples under the relaxation factor, and finally transfers

the representation onto the moved basis so that the identified response survives

the change of domain. Within a forward solve, each increment advances the

internal variable through the trial state, the implicit branch update and the

refreshed inverse viscous right Cauchy-Green tensor, and the invariants the three

constitutive functions are interrogated on are recorded. The energy bookkeeping

of the branch is carried alongside: the free energy it stores at an increment is

psi1(I1_e) + psi2(I2_e), which the normalisation f(x_1) = 0 and the non-negative

boundary slope keep non-negative on every admissible elastic state, while the

reduced dissipation accumulated along the way must also stay non-negative. Those

two conditions are the defining property of the generalized standard material

structure and the reason no admissibility constraint has to be imposed on the

identified functions. Only the three

non-equilibrium functions carry adaptive domains, because their arguments are set

by the internal evolution rather than directly by the applied deformation, and

the reported scalar is the upper endpoint reached by the dissipation-potential

domain, in the units of the deviatoric stress invariant J_tau.

Returns
-------
float, the adapted upper endpoint of the dissipation-potential interpolation domain as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_full_pipeline(n_outer: int = 3, alpha_smooth: float = 5.0,
                      eta: float = 0.5, dt: float = 1.0,
                      stretch_rate: float = 0.05,
                      stretch_max: float = 2.0) -> float:
    '''Run the block-alternating calibration and return the adapted endpoint.

    Parameters
    ----------
    n_outer : int
        Number of outer iterations of the block-alternating scheme.
    alpha_smooth : float
        Smoothing parameter of the upper-tail statistic.
    eta : float
        Relaxation factor of the endpoint update.
    dt : float
        Time increment of the stretch history.
    stretch_rate : float
        Rate of the prescribed stretch.
    stretch_max : float
        Peak stretch of the loading and unloading history.

    Returns
    -------
    x_end_phi : float
        Adapted upper endpoint of the dissipation-potential interpolation
        domain after n_outer outer iterations.

    Raises
    ------
    ValueError
        If n_outer is not a non-negative integer, if stretch_max is not greater
        than one, or if the branch violates thermodynamic admissibility by
        storing a negative free energy or accumulating a negative reduced
        dissipation.
    '''
    return x_end_phi

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

_PSI1_INITIAL = (3.0, 3.02, (-1.0, -2.0, -2.5, -3.0))
_PSI2_INITIAL = (0.0, 0.05, (-2.0, -3.0, -3.5, -4.0))
_PHI_INITIAL = (0.0, 0.10, (-4.0, -5.0, -5.5, -6.0))
_ADMISSIBILITY_TOLERANCE = 1e-12


def _oracle_run_full_pipeline(n_outer: int = 3, alpha_smooth: float = 5.0,
                              eta: float = 0.5, dt: float = 1.0,
                              stretch_rate: float = 0.05,
                              stretch_max: float = 2.0) -> float:
    """Reference implementation chaining every earlier step."""
    if not (isinstance(n_outer, int) and not isinstance(n_outer, bool)
            and n_outer >= 0):
        raise ValueError("n_outer must be a non-negative integer")
    if not float(stretch_max) > 1.0:
        raise ValueError("stretch_max must be greater than one")

    n_half = int(round((float(stretch_max) - 1.0) / (float(stretch_rate) * float(dt))))
    ramp = np.arange(1, n_half + 1)
    stretch_history = np.concatenate([1.0 + float(stretch_rate) * float(dt) * ramp,
                                      float(stretch_max)
                                      - float(stretch_rate) * float(dt) * ramp])

    functions = [list(_PSI1_INITIAL), list(_PSI2_INITIAL), list(_PHI_INITIAL)]
    for _ in range(n_outer):
        psi1 = tuple(functions[0])
        psi2 = tuple(functions[1])
        phi = tuple(functions[2])
        cv_inverse = np.ones(3)
        samples = [np.empty(stretch_history.size) for _ in range(3)]
        dissipation = 0.0
        for index, stretch in enumerate(stretch_history):
            beta_trial, stretch_squared = _oracle_uniaxial_kinematics(
                float(stretch), cv_inverse)
            beta_e = _oracle_local_branch_update(beta_trial, float(dt), psi1, psi2,
                                                 phi)
            cv_inverse = beta_e / stretch_squared
            first, second = _oracle_isochoric_invariants(beta_e)
            tau_neq, J_tau = _oracle_branch_kirchhoff_stress(beta_e, psi1, psi2)
            d_e = _oracle_viscous_flow_rate(tau_neq, J_tau, phi)
            dissipation += float(dt) * float(np.dot(tau_neq, d_e))
            stored = float(
                _oracle_curvature_spline_values(psi1[0], psi1[1], psi1[2],
                                                [first])[0][0]
                + _oracle_curvature_spline_values(psi2[0], psi2[1], psi2[2],
                                                  [second])[0][0])
            if stored < -_ADMISSIBILITY_TOLERANCE:
                raise ValueError("the branch stores a negative free energy")
            samples[0][index] = first
            samples[1][index] = second
            samples[2][index] = J_tau
        if dissipation < -_ADMISSIBILITY_TOLERANCE:
            raise ValueError("the accumulated reduced dissipation is negative")

        moved = [_oracle_adapt_domain_endpoint(functions[i][1], samples[i],
                                               float(alpha_smooth), float(eta))
                 for i in range(3)]
        for i in range(3):
            functions[i][2] = _oracle_project_spline_parameters(
                functions[i][0], functions[i][1], functions[i][2], moved[i],
                samples[i])
            functions[i][1] = moved[i]
    return float(functions[2][1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal scenario: the configuration given in the prompt ---
        {
            "setup": "n_outer = 3\n",
            "call": "round(float(run_full_pipeline(n_outer=n_outer)), 10)",
            "gold_call": "round(float(_oracle_run_full_pipeline(n_outer=n_outer)), 10)",
        },
        # --- Boundary case: no outer iteration, where the endpoint is still the
        # initial one, and a single outer iteration ---
        {
            "setup": "n_outer = 0\n",
            "call": "[round(float(run_full_pipeline(n_outer=k)), 10) for k in (n_outer, 1)]",
            "gold_call": "[round(float(_oracle_run_full_pipeline(n_outer=k)), 10) for k in (n_outer, 1)]",
        },
        # --- Edge case: a negative outer iteration count must raise ValueError ---
        {
            "setup": """def run_model():
    try:
        run_full_pipeline(n_outer=-1)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_run_full_pipeline(n_outer=-1)
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

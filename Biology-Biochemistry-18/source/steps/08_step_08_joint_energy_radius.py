"""
Find the least simultaneous mismatch over the shared Euclidean uncertainty region.

Concentration and driving-force restrictions are linear, while the shared intrinsic factors occupy a Euclidean ball. The minimum uniform mismatch is a convex calibration problem; componentwise factor bounds alone define a different region.

Returns
-------
One nonnegative float, or positive infinity for infeasibility
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def joint_energy_radius(system: "np.ndarray", n_concentrations: int,
                        n_factors: int, radius: float) -> float:
    """
    system is a finite augmented [D|d] array as in step 7, with variables
    (concentrations, shared factors, delta). n_concentrations and n_factors
    are nonnegative integer counts; the last variable is delta>=0. radius
    is finite and nonnegative, constraining the Euclidean norm of all
    n_factors together. Return the minimum delta, or positive infinity
    if the feasible region is empty. Bounded concentration domains are
    encoded in system. Zero factors or zero radius are allowed. Invalid
    dimensions, nonfinite data or negative radius raise ValueError.
    Unresolved solver status or optimality gap raises RuntimeError.
    Finite answers are accepted within absolute tolerance 0.000002.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_joint_energy_radius(system: "np.ndarray", n_concentrations: int,
                        n_factors: int, radius: float) -> float:
    import numpy as np
    from scipy.optimize import linprog, minimize, LinearConstraint
    data = np.asarray(system, dtype=float)
    nx, k = int(n_concentrations), int(n_factors)
    if (nx != n_concentrations or k != n_factors or min(nx, k) < 0
            or data.ndim != 2 or data.shape[1] != nx+k+2 or not np.isfinite(data).all()
            or not np.isfinite(radius) or radius < 0):
        raise ValueError('Finite half-spaces, dimensions and nonnegative radius required')
    mat, rhs = data[:, :-1].copy(), data[:, -1].copy()
    width = nx+k+1
    cost = np.zeros(width)
    cost[-1] = 1.
    bounds = [(None, None)]*nx + [(-radius, radius)]*k + [(0., None)]
    lp_options = {'primal_feasibility_tolerance': 1e-10,
                  'dual_feasibility_tolerance': 1e-10}
    upper = np.inf
    for iteration in range(180):
        relaxed = linprog(cost, A_ub=mat, b_ub=rhs, bounds=bounds,
                          method='highs', options=lp_options)
        if relaxed.status == 2:
            return float('inf')
        if not relaxed.success:
            raise RuntimeError('Calibration linear relaxation failed')
        v = relaxed.x
        norm = np.linalg.norm(v[nx:nx+k])
        if norm <= radius + 1e-10:
            return float(max(0., relaxed.fun))
        projected = v[nx:nx+k] * (radius / norm)
        fixed_bounds = [(None, None)]*nx + [(u, u) for u in projected] + [(0., None)]
        witness = linprog(cost, A_ub=mat, b_ub=rhs, bounds=fixed_bounds,
                          method='highs', options=lp_options)
        if witness.success:
            upper = min(upper, float(witness.fun))
        if iteration == 0 or iteration % 12 == 0:
            def _cone(x):
                return radius**2 - x[nx:nx+k] @ x[nx:nx+k]
            def _cone_jac(x):
                out = np.zeros(width)
                out[nx:nx+k] = -2.*x[nx:nx+k]
                return out
            solved = minimize(lambda x: x[-1], v, jac=lambda x: cost,
                bounds=bounds, constraints=[LinearConstraint(mat, -np.inf, rhs),
                    {'type':'ineq', 'fun':_cone, 'jac':_cone_jac}],
                method='SLSQP', options={'ftol':2e-12, 'maxiter':600})
            if (np.max(mat@solved.x-rhs, initial=0.) <= 2e-8
                    and np.linalg.norm(solved.x[nx:nx+k]) <= radius+1e-9):
                upper = min(upper, float(solved.fun))
        if upper - relaxed.fun <= 3e-8:
            return float(max(0., upper))
        cut = np.zeros(width)
        cut[nx:nx+k] = v[nx:nx+k]/norm
        mat = np.vstack([mat, cut])
        rhs = np.r_[rhs, radius]
    raise RuntimeError('Calibration optimality gap unresolved')

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\ns=np.array([[1.,1.,-1.,-2.],[0.,0.,-1.,0.]])',
      'call': 'joint_energy_radius(s,0,2,.5)',
      'gold_call': '_oracle_joint_energy_radius(s,0,2,.5)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               's=np.array([[1.,-1.,-1.],[-1.,0.,0.],[1.,0.,2.],[0.,-1.,0.]])',
      'call': 'joint_energy_radius(s,1,0,0.)',
      'gold_call': '_oracle_joint_energy_radius(s,1,0,0.)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\ns=np.array([[1.,-1.,-2.],[0.,-1.,0.]])',
      'call': 'joint_energy_radius(s,0,1,0.)',
      'gold_call': '_oracle_joint_energy_radius(s,0,1,0.)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\ns=np.array([[1.,0.,-2.]])',
      'call': 'float(np.isposinf(joint_energy_radius(s,0,1,.5)))',
      'gold_call': 'float(np.isposinf(_oracle_joint_energy_radius(s,0,1,.5)))',
      'tol': 0.0},
     {'setup': 'import numpy as np\n'
               's=np.array([[0.,-1.,0.]])\n'
               '\n'
               'def _raises_value_error(fn, arguments):\n'
               '    try:\n'
               '        fn(*arguments)\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    return 0.0\n',
      'call': '_raises_value_error(joint_energy_radius, (s,0,1,-1.,))',
      'gold_call': '_raises_value_error(_oracle_joint_energy_radius, (s,0,1,-1.,))',
      'tol': 0.0}]

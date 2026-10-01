"""
Select a nonlinear profile-curvature lower bound.

The author-derived profile equation minimizes curvature over the coefficient-covariance ellipsoid at each composition. Its crossing is nonlinear in composition and replaces the easier local root-minus-z-standard-error approximation before AICc-supported model selection.

Returns
-------
return np.array([lower_bound, root, profile_half_width, coefficient_count, delta_aicc, third_derivative, local_standard_error], dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def spinodal_lower_confidence(rk_ensemble, temperature_k,
                              z_score=1.645, delta_aicc=4.0):
    """Select a profile bound across covariance-robust feasible RK orders.

    Input is the (3,36) output of `common_tangent_screen`. Retain triple-
    feasible models within `delta_aicc` of the best feasible AICc. Evaluate
    q(x)-z_score*sqrt(b2(x).T@Cov@b2(x)) on 513 inclusive nodes from 0.05 to
    the stored root; require one strict crossing and bisect to 1e-12. Here q
    is full curvature and b2 is the active RK second-derivative basis. Choose
    the smallest crossing, then smaller coefficient count. Return float64(7):
    [profile_lower_bound,nominal_root,profile_half_width,coefficient_count,
    Delta_AICc_from_best,third_derivative_at_root,local_delta_standard_error].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_spinodal_lower_confidence(rk_ensemble, temperature_k,
                                      z_score=1.645, delta_aicc=4.0):
    """Choose the most conservative uncertainty bound across supported feasible RK orders."""
    import math
    import numpy as np
    from numpy.polynomial import Polynomial
    ensemble = np.asarray(rk_ensemble, dtype=float)
    temperature = float(temperature_k)
    z_value = float(z_score)
    delta_limit = float(delta_aicc)
    if ensemble.shape != (3, 36):
        raise ValueError("rk_ensemble must have shape (3,36)")
    if not np.all(np.isfinite(ensemble[:, :3])) or not math.isfinite(temperature + z_value + delta_limit):
        raise ValueError("inputs must be finite")
    if temperature <= 0.0 or z_value < 0.0 or delta_limit < 0.0:
        raise ValueError("invalid temperature, z score, or AICc support width")
    counts = ensemble[:, 0].astype(int)
    if not np.array_equal(counts, np.array([2, 3, 4])) or np.any(ensemble[:, 2] != np.round(ensemble[:, 2])):
        raise ValueError("candidate rows must be ordered counts 2,3,4 with binary feasibility")
    feasible = (ensemble[:, 2].astype(bool) & ensemble[:, 24].astype(bool)
                & ensemble[:, 30].astype(bool))
    if not np.any(feasible):
        raise ValueError("no physically feasible Redlich-Kister candidate")
    best_aicc = float(np.min(ensemble[feasible, 1]))
    supported = feasible & (ensemble[:, 1] - best_aicc <= delta_limit + 1.0e-12)
    k_b_ev_per_k = 8.617333262145e-5
    base = Polynomial([0.0, 1.0, -1.0])
    zeta_poly = Polynomial([-1.0, 2.0])
    basis = [base * zeta_poly ** order for order in range(4)]
    candidates = []
    for row_index in np.flatnonzero(supported):
        row = ensemble[row_index]
        count = int(row[0])
        root = float(row[3])
        coefficients = row[4:4 + count]
        covariance = row[8:24].reshape(4, 4)[:count, :count]
        if not math.isfinite(root) or np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
            raise ValueError("supported candidate has invalid root or covariance")
        second_basis = np.array([poly.deriv(2)(root) for poly in basis[:count]])
        third = k_b_ev_per_k * temperature * (
            -1.0 / root ** 2 + 1.0 / (1.0 - root) ** 2
        ) + sum(float(c * poly.deriv(3)(root))
                for c, poly in zip(coefficients, basis[:count]))
        standard_error = float(np.sqrt(second_basis @ covariance @ second_basis) / abs(third))
        if z_value == 0.0:
            lower_bound = root
        else:
            grid = np.linspace(0.05, root, 513)
            profile = []
            for point in grid:
                b2 = np.array([poly.deriv(2)(point) for poly in basis[:count]])
                mean_curvature = k_b_ev_per_k * temperature * (
                    1.0 / point + 1.0 / (1.0 - point)
                ) + float(coefficients @ b2)
                profile.append(mean_curvature - z_value * np.sqrt(b2 @ covariance @ b2))
            profile = np.asarray(profile)
            crossings = np.flatnonzero(profile[:-1] * profile[1:] < 0.0)
            if crossings.size != 1:
                continue
            lo, hi = float(grid[crossings[0]]), float(grid[crossings[0] + 1])
            f_lo = float(profile[crossings[0]])
            for _ in range(100):
                midpoint = 0.5 * (lo + hi)
                b2 = np.array([poly.deriv(2)(midpoint) for poly in basis[:count]])
                mean_curvature = k_b_ev_per_k * temperature * (
                    1.0 / midpoint + 1.0 / (1.0 - midpoint)
                ) + float(coefficients @ b2)
                f_mid = float(mean_curvature - z_value * np.sqrt(b2 @ covariance @ b2))
                if hi - lo <= 1.0e-12:
                    break
                if f_lo * f_mid <= 0.0:
                    hi = midpoint
                else:
                    lo, f_lo = midpoint, f_mid
            lower_bound = 0.5 * (lo + hi)
        if not (0.0 < lower_bound <= root):
            continue
        profile_half_width = root - lower_bound
        candidates.append((lower_bound, count, root, profile_half_width,
                           float(row[1] - best_aicc), float(third), standard_error))
    if not candidates:
        raise ValueError("no supported candidate has a physical profile bound")
    chosen = min(candidates, key=lambda value: (value[0], value[1]))
    return np.array([chosen[0], chosen[2], chosen[3], float(chosen[1]),
                     chosen[4], chosen[5], chosen[6]], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nx=np.array([0.08, 0.22, 0.38, 0.55, 0.72, 0.88],float);L=np.array([0.12, 0.02, -0.01, 0.004],float);u=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),u,u*u,u*u*u));y=A@L+np.random.default_rng(270).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(290).normal(size=(x.size,x.size));C=(Q@Q.T)*1e-07+np.eye(x.size)*1e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));rk=_oracle_fit_redlich_kister(m,520);ens=_oracle_common_tangent_screen(rk,520,sigma_radius=0.75)', 'call': 'spinodal_lower_confidence(ens,520,z_score=1.645,delta_aicc=4.0)', 'gold_call': '_oracle_spinodal_lower_confidence(ens,520,z_score=1.645,delta_aicc=4.0)'}, {'setup': 'import numpy as np\nx=np.array([0.06, 0.18, 0.31, 0.47, 0.63, 0.78, 0.92],float);L=np.array([0.1, -0.03, 0.02, -0.006],float);u=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),u,u*u,u*u*u));y=A@L+np.random.default_rng(271).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(291).normal(size=(x.size,x.size));C=(Q@Q.T)*2e-07+np.eye(x.size)*2e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));rk=_oracle_fit_redlich_kister(m,450);ens=_oracle_common_tangent_screen(rk,450,sigma_radius=1.0)', 'call': 'spinodal_lower_confidence(ens,450,z_score=1.0,delta_aicc=2.0)', 'gold_call': '_oracle_spinodal_lower_confidence(ens,450,z_score=1.0,delta_aicc=2.0)'}, {'setup': 'import numpy as np\nx=np.array([0.1, 0.24, 0.4, 0.58, 0.74, 0.9],float);L=np.array([0.15, 0.0, 0.0, 0.0],float);u=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),u,u*u,u*u*u));y=A@L+np.random.default_rng(272).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(292).normal(size=(x.size,x.size));C=(Q@Q.T)*5e-08+np.eye(x.size)*5e-08;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));rk=_oracle_fit_redlich_kister(m,600);ens=_oracle_common_tangent_screen(rk,600,sigma_radius=1.25)', 'call': 'spinodal_lower_confidence(ens,600,z_score=1.96,delta_aicc=6.0)', 'gold_call': '_oracle_spinodal_lower_confidence(ens,600,z_score=1.96,delta_aicc=6.0)'}, {'setup': 'import numpy as np\nx=np.array([0.05, 0.16, 0.29, 0.43, 0.57, 0.71, 0.84, 0.95],float);L=np.array([0.11, 0.04, 0.015, -0.003],float);u=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),u,u*u,u*u*u));y=A@L+np.random.default_rng(273).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(293).normal(size=(x.size,x.size));C=(Q@Q.T)*3e-07+np.eye(x.size)*3e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));rk=_oracle_fit_redlich_kister(m,450);ens=_oracle_common_tangent_screen(rk,450,sigma_radius=0.5)', 'call': 'spinodal_lower_confidence(ens,450,z_score=1.28,delta_aicc=1.0)', 'gold_call': '_oracle_spinodal_lower_confidence(ens,450,z_score=1.28,delta_aicc=1.0)'}, {'setup': 'import numpy as np\nx=np.array([0.09, 0.21, 0.35, 0.5, 0.66, 0.81, 0.93],float);L=np.array([0.14, -0.02, 0.03, 0.008],float);u=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),u,u*u,u*u*u));y=A@L+np.random.default_rng(274).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(294).normal(size=(x.size,x.size));C=(Q@Q.T)*1.5e-07+np.eye(x.size)*1.5e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));rk=_oracle_fit_redlich_kister(m,700);ens=_oracle_common_tangent_screen(rk,700,sigma_radius=1.1)', 'call': 'spinodal_lower_confidence(ens,700,z_score=1.645,delta_aicc=4.0)', 'gold_call': '_oracle_spinodal_lower_confidence(ens,700,z_score=1.645,delta_aicc=4.0)'}, {'setup': 'import numpy as np\nx=np.array([0.07, 0.19, 0.33, 0.48, 0.62, 0.76, 0.89],float);L=np.array([0.13, 0.01, -0.025, 0.005],float);u=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),u,u*u,u*u*u));y=A@L+np.random.default_rng(275).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(295).normal(size=(x.size,x.size));C=(Q@Q.T)*8e-08+np.eye(x.size)*8e-08;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));rk=_oracle_fit_redlich_kister(m,500);ens=_oracle_common_tangent_screen(rk,500,sigma_radius=0.9)', 'call': 'spinodal_lower_confidence(ens,500,z_score=0.5,delta_aicc=10.0)', 'gold_call': '_oracle_spinodal_lower_confidence(ens,500,z_score=0.5,delta_aicc=10.0)'}, {'setup': 'import numpy as np\nx=np.array([0.11, 0.25, 0.39, 0.54, 0.69, 0.83, 0.94],float);L=np.array([0.09, 0.05, 0.01, -0.01],float);u=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),u,u*u,u*u*u));y=A@L+np.random.default_rng(276).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(296).normal(size=(x.size,x.size));C=(Q@Q.T)*4e-07+np.eye(x.size)*4e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));rk=_oracle_fit_redlich_kister(m,400);ens=_oracle_common_tangent_screen(rk,400,sigma_radius=0.65)', 'call': 'spinodal_lower_confidence(ens,400,z_score=2.0,delta_aicc=3.0)', 'gold_call': '_oracle_spinodal_lower_confidence(ens,400,z_score=2.0,delta_aicc=3.0)'}, {'setup': 'import numpy as np\nx=np.array([0.04, 0.14, 0.27, 0.41, 0.59, 0.73, 0.86, 0.96],float);L=np.array([0.125, -0.035, 0.018, 0.006],float);u=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),u,u*u,u*u*u));y=A@L+np.random.default_rng(277).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(297).normal(size=(x.size,x.size));C=(Q@Q.T)*2.5e-07+np.eye(x.size)*2.5e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));rk=_oracle_fit_redlich_kister(m,550);ens=_oracle_common_tangent_screen(rk,550,sigma_radius=0.85)', 'call': 'spinodal_lower_confidence(ens,550,z_score=1.4,delta_aicc=4.5)', 'gold_call': '_oracle_spinodal_lower_confidence(ens,550,z_score=1.4,delta_aicc=4.5)'}]

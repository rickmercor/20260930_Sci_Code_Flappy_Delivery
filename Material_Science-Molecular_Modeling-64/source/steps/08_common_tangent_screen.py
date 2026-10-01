"""
Solve nominal and covariance-robust liquid-liquid coexistence certificates.

The paper diagnoses phase separation from the mixing-free-energy convex hull. The author-derived Cholesky sigma-point extension makes coexistence a consequential uncertainty-feasibility condition: every perturbed coefficient vector must retain one enclosed lower spinodal and the envelope enters the final certificate.

Returns
-------
return np.asarray(rows_with_nominal_and_robust_certificates, dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def common_tangent_screen(rk_ensemble, temperature_k,
                          left_interval=(0.01,0.49),
                          right_interval=(0.51,0.99),
                          starts_per_axis=9, tolerance=1.0e-10,
                          sigma_radius=1.0):
    """Screen nominal and coefficient-uncertain coexistence consistency.

    Input is (3,24) from `fit_redlich_kister`. Solve nominal common tangents
    from the evenly spaced start-grid product. Use the exact 2x2 Newton
    Jacobian and first in-bounds halving among powers 0..13; select by smaller
    alpha, larger beta, residual, then slope. Require residual<=10*tolerance
    and alpha<stored_root<0.5<beta. For C=LL.T, then process c and ordered
    c+sigma_radius*L[:,j], c-sigma_radius*L[:,j]. Recompute each 257-node root
    and tangent by continuation from the preceding accepted solution. Every
    point must retain one enclosed root; the envelope is max(alpha),min(beta).

    Returns float64 shape (3,36): the original 24 columns followed by
    [phase_feasible,x_alpha,x_beta,tangent_slope,max_residual,width,
    robust_feasible,robust_alpha,robust_beta,robust_worst_residual,
    robust_min_root_margin,robust_max_root_shift]. Infeasible robust
    diagnostics use numeric zeros.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_common_tangent_screen(rk_ensemble, temperature_k,
                                  left_interval=(0.01, 0.49),
                                  right_interval=(0.51, 0.99),
                                  starts_per_axis=9, tolerance=1.0e-10,
                                  sigma_radius=1.0):
    """Append nominal and covariance-robust common-tangent certificates."""
    import math
    import numpy as np
    from numpy.polynomial import Polynomial
    ensemble = np.asarray(rk_ensemble, dtype=float)
    temperature = float(temperature_k)
    la, lb = map(float, left_interval)
    ra, rb = map(float, right_interval)
    starts = int(starts_per_axis)
    tol = float(tolerance)
    radius = float(sigma_radius)
    if ensemble.shape != (3, 24) or not np.all(np.isfinite(ensemble)):
        raise ValueError("rk_ensemble must be a finite (3,24) array")
    if temperature <= 0.0 or not (0.0 < la < lb < ra < rb < 1.0):
        raise ValueError("invalid temperature or disjoint composition intervals")
    if starts < 3 or tol <= 0.0 or not math.isfinite(radius) or radius < 0.0:
        raise ValueError("starts_per_axis>=3, positive tolerance, and nonnegative sigma_radius are required")
    kb = 8.617333262145e-5
    base = Polynomial([0.0, 1.0, -1.0])
    zeta = Polynomial([-1.0, 2.0])
    basis = [base * zeta ** order for order in range(4)]
    output = []
    for row in ensemble:
        count = int(row[0])
        coeff = row[4:4 + count]
        covariance = row[8:24].reshape(4, 4)[:count, :count]
        polys = basis[:count]

        def values(x, active_coeff):
            g = kb * temperature * (x * np.log(x) + (1.0 - x) * np.log(1.0 - x))
            gp = kb * temperature * np.log(x / (1.0 - x))
            gpp = kb * temperature * (1.0 / x + 1.0 / (1.0 - x))
            for c, p in zip(active_coeff, polys):
                g += c * p(x); gp += c * p.deriv(1)(x); gpp += c * p.deriv(2)(x)
            return float(g), float(gp), float(gpp)

        def curvature_root(active_coeff):
            grid = np.linspace(0.05, 0.49, 257)
            second = kb * temperature * (1.0 / grid + 1.0 / (1.0 - grid))
            for c, p in zip(active_coeff, polys):
                second = second + c * p.deriv(2)(grid)
            crossings = np.flatnonzero(second[:-1] * second[1:] < 0.0)
            if crossings.size != 1:
                return None
            lo, hi = float(grid[crossings[0]]), float(grid[crossings[0] + 1])
            f_lo = float(second[crossings[0]])
            for _ in range(100):
                midpoint = 0.5 * (lo + hi)
                f_mid = values(midpoint, active_coeff)[2]
                if hi - lo <= 1.0e-12:
                    break
                if f_lo * f_mid <= 0.0:
                    hi = midpoint
                else:
                    lo, f_lo = midpoint, f_mid
            return 0.5 * (lo + hi)

        def newton(active_coeff, x0, y0):
            x, y = float(x0), float(y0)
            for _ in range(80):
                gx, gpx, gppx = values(x, active_coeff)
                gy, gpy, gppy = values(y, active_coeff)
                sec = (gy - gx) / (y - x)
                f = np.array([gpx - sec, gpy - sec])
                if np.max(np.abs(f)) <= tol:
                    break
                dsx = (sec - gpx) / (y - x)
                dsy = (gpy - sec) / (y - x)
                jac = np.array([[gppx - dsx, -dsy], [-dsx, gppy - dsy]])
                try:
                    step = np.linalg.solve(jac, f)
                except np.linalg.LinAlgError:
                    break
                accepted = False
                for power in range(14):
                    scale = 0.5 ** power
                    xn, yn = x - scale * step[0], y - scale * step[1]
                    if la <= xn <= lb and ra <= yn <= rb and xn < yn:
                        x, y = float(xn), float(yn); accepted = True; break
                if not accepted:
                    break
            gx, gpx, _ = values(x, active_coeff)
            gy, gpy, _ = values(y, active_coeff)
            sec = (gy - gx) / (y - x)
            residual = max(abs(gpx - sec), abs(gpy - sec))
            if residual <= 10.0 * tol:
                return (x, y, residual, sec)
            return None

        solutions = []
        if int(round(row[2])) == 1:
            for x0 in np.linspace(la, lb, starts):
                for y0 in np.linspace(ra, rb, starts):
                    solved = newton(coeff, x0, y0)
                    if solved is not None:
                        x_sol, y_sol, residual_sol, slope_sol = solved
                        solutions.append((x_sol, -y_sol, residual_sol, slope_sol))
        feasible = int(bool(solutions))
        xa = xb = slope = residual = width = 0.0
        if feasible:
            xa, neg_xb, residual, slope = min(solutions)
            xb = -neg_xb
            width = xb - xa
            if not (xa < row[3] < 0.5 < xb):
                feasible = 0
        robust = 0
        robust_alpha = robust_beta = robust_residual = robust_margin = max_root_shift = 0.0
        if feasible and np.min(np.linalg.eigvalsh(covariance)) > 0.0:
            cholesky = np.linalg.cholesky(covariance)
            sigma_points = [coeff.copy()]
            for column in range(count):
                sigma_points.append(coeff + radius * cholesky[:, column])
                sigma_points.append(coeff - radius * cholesky[:, column])
            certificates = [(float(row[3]), xa, xb, residual)]
            continuation = (xa, xb)
            robust = 1
            for perturbed in sigma_points[1:]:
                perturbed_root = curvature_root(perturbed)
                solved = None if perturbed_root is None else newton(
                    perturbed, continuation[0], continuation[1]
                )
                if solved is None:
                    robust = 0
                    break
                perturbed_alpha, perturbed_beta, perturbed_residual, _ = solved
                if not (perturbed_alpha < perturbed_root < 0.5 < perturbed_beta):
                    robust = 0
                    break
                certificates.append((perturbed_root, perturbed_alpha,
                                     perturbed_beta, perturbed_residual))
                continuation = (perturbed_alpha, perturbed_beta)
            if robust:
                roots = np.array([item[0] for item in certificates])
                alphas = np.array([item[1] for item in certificates])
                betas = np.array([item[2] for item in certificates])
                robust_alpha = float(np.max(alphas))
                robust_beta = float(np.min(betas))
                robust_residual = float(np.max([item[3] for item in certificates]))
                robust_margin = float(min(np.min(roots) - robust_alpha,
                                          robust_beta - np.max(roots)))
                max_root_shift = float(np.max(np.abs(roots - row[3])))
                if not (robust_margin > 0.0 and robust_alpha < 0.5 < robust_beta):
                    robust = 0
        if not robust:
            robust_alpha = robust_beta = robust_residual = robust_margin = max_root_shift = 0.0
        output.append(np.concatenate((
            row, np.array([feasible, xa, xb, slope, residual, width,
                           robust, robust_alpha, robust_beta, robust_residual,
                           robust_margin, max_root_shift])
        )))
    return np.asarray(output, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nx=np.array([0.08, 0.22, 0.38, 0.55, 0.72, 0.88],float);L=np.array([0.12, 0.02, -0.01, 0.004],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(170).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(190).normal(size=(x.size,x.size));C=(Q@Q.T)*1e-07+np.eye(x.size)*1e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));ens=_oracle_fit_redlich_kister(m,520)', 'call': 'common_tangent_screen(ens,520,sigma_radius=0.5)', 'gold_call': '_oracle_common_tangent_screen(ens,520,sigma_radius=0.5)'}, {'setup': 'import numpy as np\nx=np.array([0.06, 0.18, 0.31, 0.47, 0.63, 0.78, 0.92],float);L=np.array([0.1, -0.03, 0.02, -0.006],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(171).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(191).normal(size=(x.size,x.size));C=(Q@Q.T)*2e-07+np.eye(x.size)*2e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));ens=_oracle_fit_redlich_kister(m,450)', 'call': 'common_tangent_screen(ens,450,sigma_radius=0.75)', 'gold_call': '_oracle_common_tangent_screen(ens,450,sigma_radius=0.75)'}, {'setup': 'import numpy as np\nx=np.array([0.1, 0.24, 0.4, 0.58, 0.74, 0.9],float);L=np.array([0.15, 0.0, 0.0, 0.0],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(172).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(192).normal(size=(x.size,x.size));C=(Q@Q.T)*5e-08+np.eye(x.size)*5e-08;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));ens=_oracle_fit_redlich_kister(m,600)', 'call': 'common_tangent_screen(ens,600,sigma_radius=1.0)', 'gold_call': '_oracle_common_tangent_screen(ens,600,sigma_radius=1.0)'}, {'setup': 'import numpy as np\nx=np.array([0.05, 0.16, 0.29, 0.43, 0.57, 0.71, 0.84, 0.95],float);L=np.array([0.11, 0.04, 0.015, -0.003],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(173).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(193).normal(size=(x.size,x.size));C=(Q@Q.T)*3e-07+np.eye(x.size)*3e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));ens=_oracle_fit_redlich_kister(m,350)', 'call': 'common_tangent_screen(ens,350,sigma_radius=1.25)', 'gold_call': '_oracle_common_tangent_screen(ens,350,sigma_radius=1.25)'}, {'setup': 'import numpy as np\nx=np.array([0.09, 0.21, 0.35, 0.5, 0.66, 0.81, 0.93],float);L=np.array([0.14, -0.02, 0.03, 0.008],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(174).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(194).normal(size=(x.size,x.size));C=(Q@Q.T)*1.5e-07+np.eye(x.size)*1.5e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));ens=_oracle_fit_redlich_kister(m,700)', 'call': 'common_tangent_screen(ens,700,sigma_radius=1.5)', 'gold_call': '_oracle_common_tangent_screen(ens,700,sigma_radius=1.5)'}, {'setup': 'import numpy as np\nx=np.array([0.07, 0.19, 0.33, 0.48, 0.62, 0.76, 0.89],float);L=np.array([0.13, 0.01, -0.025, 0.005],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(175).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(195).normal(size=(x.size,x.size));C=(Q@Q.T)*8e-08+np.eye(x.size)*8e-08;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));ens=_oracle_fit_redlich_kister(m,500)', 'call': 'common_tangent_screen(ens,500,sigma_radius=0.9)', 'gold_call': '_oracle_common_tangent_screen(ens,500,sigma_radius=0.9)'}, {'setup': 'import numpy as np\nx=np.array([0.11, 0.25, 0.39, 0.54, 0.69, 0.83, 0.94],float);L=np.array([0.09, 0.05, 0.01, -0.01],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(176).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(196).normal(size=(x.size,x.size));C=(Q@Q.T)*4e-07+np.eye(x.size)*4e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));ens=_oracle_fit_redlich_kister(m,400)', 'call': 'common_tangent_screen(ens,400,sigma_radius=0.65)', 'gold_call': '_oracle_common_tangent_screen(ens,400,sigma_radius=0.65)'}, {'setup': 'import numpy as np\nx=np.array([0.04, 0.14, 0.27, 0.41, 0.59, 0.73, 0.86, 0.96],float);L=np.array([0.125, -0.035, 0.018, 0.006],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(177).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(197).normal(size=(x.size,x.size));C=(Q@Q.T)*2.5e-07+np.eye(x.size)*2.5e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)));ens=_oracle_fit_redlich_kister(m,550)', 'call': 'common_tangent_screen(ens,550,sigma_radius=0.85)', 'gold_call': '_oracle_common_tangent_screen(ens,550,sigma_radius=0.85)'}]

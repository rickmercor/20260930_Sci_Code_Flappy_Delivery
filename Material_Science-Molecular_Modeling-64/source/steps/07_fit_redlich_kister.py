"""
Fit and physically screen correlated Redlich-Kister model orders.

Coefficient counts 2, 3, and 4 compete by AICc after augmented-covariance GLS; a dense curvature scan prevents a statistically favored but spinodally ambiguous model from entering the supported set.

Returns
-------
return np.asarray(candidate_rows, dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def fit_redlich_kister(augmented_certificate, temperature_k,
                       bracket=(0.05, 0.49), grid_points=257):
    """Fit and physically screen a Redlich-Kister order ensemble.

    `augmented_certificate` has shape (N,N+4). Fit coefficient counts 2, 3,
    and 4 by correlated GLS, compute AICc, and mark a candidate feasible only
    when an odd `grid_points` scan finds exactly one strict sign change of the
    full ideal-plus-excess second derivative inside `bracket`; refine that
    root to 1e-12 by bisection. Returns shape (3,24), rows ordered by counts
    [2,3,4], and columns [count,AICc,feasible,root,L0..L3,
    flattened_row_major_4x4_covariance], zero-padding unused coefficients and
    covariance entries. An infeasible root uses the numeric sentinel 0.0 and
    must never enter the supported set.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_fit_redlich_kister(augmented_certificate, temperature_k,
                               bracket=(0.05, 0.49), grid_points=257):
    """Fit and screen Redlich-Kister orders with GLS, AICc, and spinodal feasibility."""
    import math
    import numpy as np
    from numpy.polynomial import Polynomial
    certificate = np.asarray(augmented_certificate, dtype=float)
    temperature = float(temperature_k)
    left, right = map(float, bracket)
    grid_points = int(grid_points)
    if certificate.ndim != 2 or certificate.shape[0] < 6 or certificate.shape[1] != certificate.shape[0] + 4:
        raise ValueError("augmented_certificate must have shape (N,N+4) with N>=6")
    if not np.all(np.isfinite(certificate)) or not math.isfinite(temperature):
        raise ValueError("inputs must be finite")
    if temperature <= 0.0 or not (0.0 < left < right < 1.0) or grid_points < 33 or grid_points % 2 == 0:
        raise ValueError("invalid temperature, bracket, or odd grid_points")
    n = certificate.shape[0]
    x = certificate[:, 0]
    response = certificate[:, 1]
    covariance = certificate[:, 2:2 + n]
    if np.any(x <= 0.0) or np.any(x >= 1.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("certificate compositions must increase strictly in (0,1)")
    if np.min(np.linalg.eigvalsh(covariance)) <= 0.0:
        raise ValueError("response covariance must be positive definite")
    inverse = np.linalg.inv(covariance)
    k_b_ev_per_k = 8.617333262145e-5
    base = Polynomial([0.0, 1.0, -1.0])
    zeta_poly = Polynomial([-1.0, 2.0])
    basis_polynomials = [base * zeta_poly ** order for order in range(4)]
    grid = np.linspace(left, right, grid_points)
    rows = []
    for count in (2, 3, 4):
        design = np.column_stack([poly(x) for poly in basis_polynomials[:count]])
        normal_inverse = np.linalg.inv(design.T @ inverse @ design)
        coefficients = normal_inverse @ design.T @ inverse @ response
        residual = response - design @ coefficients
        chi_square = float(residual @ inverse @ residual)
        denominator = n - count - 1
        if denominator <= 0:
            raise ValueError("too few compositions for AICc candidates")
        aicc = chi_square + 2.0 * count + 2.0 * count * (count + 1.0) / denominator

        second = k_b_ev_per_k * temperature * (1.0 / grid + 1.0 / (1.0 - grid))
        for coefficient, poly in zip(coefficients, basis_polynomials[:count]):
            second = second + coefficient * poly.deriv(2)(grid)
        crossings = np.flatnonzero(second[:-1] * second[1:] < 0.0)
        feasible = int(crossings.size == 1)
        root = 0.0
        if feasible:
            lo = float(grid[crossings[0]])
            hi = float(grid[crossings[0] + 1])
            f_lo = float(second[crossings[0]])
            for _ in range(100):
                midpoint = 0.5 * (lo + hi)
                f_mid = k_b_ev_per_k * temperature * (1.0 / midpoint + 1.0 / (1.0 - midpoint))
                f_mid += sum(float(c * poly.deriv(2)(midpoint))
                             for c, poly in zip(coefficients, basis_polynomials[:count]))
                if hi - lo <= 1.0e-12:
                    break
                if f_lo * f_mid <= 0.0:
                    hi = midpoint
                else:
                    lo = midpoint
                    f_lo = f_mid
            root = 0.5 * (lo + hi)
        padded_coefficients = np.zeros(4)
        padded_coefficients[:count] = coefficients
        padded_covariance = np.zeros((4, 4))
        padded_covariance[:count, :count] = normal_inverse
        rows.append(np.concatenate((
            np.array([float(count), aicc, float(feasible), root]),
            padded_coefficients,
            padded_covariance.ravel(),
        )))
    return np.asarray(rows, dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nx=np.array([0.08, 0.22, 0.38, 0.55, 0.72, 0.88],float);L=np.array([0.12, 0.02, -0.01, 0.004],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(70).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(90).normal(size=(x.size,x.size));C=(Q@Q.T)*1e-07+np.eye(x.size)*1e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)))', 'call': 'fit_redlich_kister(m,520)', 'gold_call': '_oracle_fit_redlich_kister(m,520)'}, {'setup': 'import numpy as np\nx=np.array([0.06, 0.18, 0.31, 0.47, 0.63, 0.78, 0.92],float);L=np.array([0.1, -0.03, 0.02, -0.006],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(71).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(91).normal(size=(x.size,x.size));C=(Q@Q.T)*2e-07+np.eye(x.size)*2e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)))', 'call': 'fit_redlich_kister(m,450)', 'gold_call': '_oracle_fit_redlich_kister(m,450)'}, {'setup': 'import numpy as np\nx=np.array([0.1, 0.24, 0.4, 0.58, 0.74, 0.9],float);L=np.array([0.15, 0.0, 0.0, 0.0],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(72).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(92).normal(size=(x.size,x.size));C=(Q@Q.T)*5e-08+np.eye(x.size)*5e-08;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)))', 'call': 'fit_redlich_kister(m,600)', 'gold_call': '_oracle_fit_redlich_kister(m,600)'}, {'setup': 'import numpy as np\nx=np.array([0.05, 0.16, 0.29, 0.43, 0.57, 0.71, 0.84, 0.95],float);L=np.array([0.11, 0.04, 0.015, -0.003],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(73).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(93).normal(size=(x.size,x.size));C=(Q@Q.T)*3e-07+np.eye(x.size)*3e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)))', 'call': 'fit_redlich_kister(m,350)', 'gold_call': '_oracle_fit_redlich_kister(m,350)'}, {'setup': 'import numpy as np\nx=np.array([0.09, 0.21, 0.35, 0.5, 0.66, 0.81, 0.93],float);L=np.array([0.14, -0.02, 0.03, 0.008],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(74).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(94).normal(size=(x.size,x.size));C=(Q@Q.T)*1.5e-07+np.eye(x.size)*1.5e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)))', 'call': 'fit_redlich_kister(m,700)', 'gold_call': '_oracle_fit_redlich_kister(m,700)'}, {'setup': 'import numpy as np\nx=np.array([0.07, 0.19, 0.33, 0.48, 0.62, 0.76, 0.89],float);L=np.array([0.13, 0.01, -0.025, 0.005],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(75).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(95).normal(size=(x.size,x.size));C=(Q@Q.T)*8e-08+np.eye(x.size)*8e-08;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)))', 'call': 'fit_redlich_kister(m,500)', 'gold_call': '_oracle_fit_redlich_kister(m,500)'}, {'setup': 'import numpy as np\nx=np.array([0.11, 0.25, 0.39, 0.54, 0.69, 0.83, 0.94],float);L=np.array([0.09, 0.05, 0.01, -0.01],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(76).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(96).normal(size=(x.size,x.size));C=(Q@Q.T)*4e-07+np.eye(x.size)*4e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)))', 'call': 'fit_redlich_kister(m,400)', 'gold_call': '_oracle_fit_redlich_kister(m,400)'}, {'setup': 'import numpy as np\nx=np.array([0.04, 0.14, 0.27, 0.41, 0.59, 0.73, 0.86, 0.96],float);L=np.array([0.125, -0.035, 0.018, 0.006],float);z=2*x-1;A=x[:,None]*(1-x)[:,None]*np.column_stack((np.ones(x.size),z,z*z,z*z*z));y=A@L+np.random.default_rng(77).normal(scale=2e-5,size=x.size);Q=np.random.default_rng(97).normal(size=(x.size,x.size));C=(Q@Q.T)*2.5e-07+np.eye(x.size)*2.5e-07;m=np.column_stack((x,y,C,np.full(x.size,.002),np.full(x.size,.25)))', 'call': 'fit_redlich_kister(m,550)', 'gold_call': '_oracle_fit_redlich_kister(m,550)'}]

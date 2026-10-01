"""
Calibrate and add a composition-correlated finite-size uncertainty model.

The paper identifies finite-size error as material. Paired-cell residuals profile the declared exponential-kernel grid, which is transformed through e_mix(x)=e(x)-x e(1) before addition.

Returns
-------
return np.column_stack((augmented, selected_sigma_column, selected_length_column)).astype(float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def augment_finite_size_covariance(mixing_certificate,
                                   calibration_compositions,
                                   calibration_residuals_ev,
                                   sigma_grid_ev,
                                   length_grid,
                                   noise_sigma_ev):
    """Profile and add finite-size uncertainty correlated across composition.

    `mixing_certificate` has shape (N,N+2), with x in column 0, excess
    response in column 1, and the NxN sampling covariance in columns 2..N+1.
    The paired-cell residuals at `calibration_compositions` are scored for
    every Cartesian-product pair in the strictly increasing positive sigma
    and length grids using log(det(C))+r.T@solve(C,r), where
    C=sigma^2*exp(-|xa-xb|/length)+noise_sigma_ev^2*I. Select the lexicographic
    minimum (objective,sigma,length). Return a float64 ndarray of shape
    (N,N+4): the original x and response, the augmented NxN covariance, then
    the selected sigma and length repeated down the final two columns.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_augment_finite_size_covariance(mixing_certificate,
                                           calibration_compositions,
                                           calibration_residuals_ev,
                                           sigma_grid_ev,
                                           length_grid,
                                           noise_sigma_ev):
    """Profile a finite-size GP kernel and transform it through endmember subtraction."""
    import math
    import numpy as np
    certificate = np.asarray(mixing_certificate, dtype=float)
    calibration_x = np.asarray(calibration_compositions, dtype=float)
    residuals = np.asarray(calibration_residuals_ev, dtype=float)
    sigma_grid = np.asarray(sigma_grid_ev, dtype=float)
    length_candidates = np.asarray(length_grid, dtype=float)
    noise = float(noise_sigma_ev)
    if certificate.ndim != 2 or certificate.shape[1] != certificate.shape[0] + 2:
        raise ValueError("mixing_certificate must have shape (N,N+2)")
    if calibration_x.ndim != 1 or residuals.shape != calibration_x.shape or calibration_x.size < 3:
        raise ValueError("finite-size calibration vectors must have one common length >=3")
    if sigma_grid.ndim != 1 or length_candidates.ndim != 1 or sigma_grid.size < 2 or length_candidates.size < 2:
        raise ValueError("both hyperparameter grids must contain at least two values")
    arrays = (certificate, calibration_x, residuals, sigma_grid, length_candidates)
    if any(not np.all(np.isfinite(value)) for value in arrays) or not math.isfinite(noise):
        raise ValueError("inputs must be finite")
    if (np.any(np.diff(calibration_x) <= 0.0) or np.any(calibration_x <= 0.0)
            or np.any(calibration_x > 1.0)):
        raise ValueError("calibration compositions must increase in (0,1]")
    if (np.any(sigma_grid <= 0.0) or np.any(np.diff(sigma_grid) <= 0.0)
            or np.any(length_candidates <= 0.0) or np.any(np.diff(length_candidates) <= 0.0)
            or noise <= 0.0):
        raise ValueError("hyperparameter grids and noise sigma must be positive")
    x = certificate[:, 0]
    if np.any(x <= 0.0) or np.any(x >= 1.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("certificate compositions must increase strictly in (0,1)")

    best = None
    distances = np.abs(calibration_x[:, None] - calibration_x[None, :])
    identity = np.eye(calibration_x.size)
    for sigma in sigma_grid:
        for length in length_candidates:
            covariance = sigma * sigma * np.exp(-distances / length) + noise * noise * identity
            sign, logdet = np.linalg.slogdet(covariance)
            if sign <= 0.0:
                continue
            objective = float(logdet + residuals @ np.linalg.solve(covariance, residuals))
            candidate = (objective, float(sigma), float(length))
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("no positive-definite finite-size candidate")
    _, selected_sigma, selected_length = best
    points = np.concatenate((x, np.array([1.0])))
    kernel = selected_sigma ** 2 * np.exp(
        -np.abs(points[:, None] - points[None, :]) / selected_length
    )
    added = (kernel[:-1, :-1]
             - x[:, None] * kernel[-1, :-1][None, :]
             - kernel[:-1, -1][:, None] * x[None, :]
             + np.outer(x, x) * kernel[-1, -1])
    augmented = certificate.copy()
    augmented[:, 2:] = certificate[:, 2:] + added
    return np.column_stack((
        augmented,
        np.full(x.size, selected_sigma),
        np.full(x.size, selected_length),
    )).astype(float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\nrng=np.random.default_rng(101);x=np.linspace(.08,.92,4);y=np.linspace(-.02,.03,4);A=rng.normal(size=(4,4));C=(A@A.T)*1e-8+np.eye(4)*2e-8;m=np.column_stack((x,y,C));cx=np.linspace(.1,1.,4);cr=rng.normal(scale=.0015,size=4);sg=np.array([.0007,.0011,.0017,.0024,.0033]);lg=np.array([.05,.11,.23,.47,.9]);noise=0.00025', 'call': 'augment_finite_size_covariance(m,cx,cr,sg,lg,noise)', 'gold_call': '_oracle_augment_finite_size_covariance(m,cx,cr,sg,lg,noise)'}, {'setup': 'import numpy as np\nrng=np.random.default_rng(102);x=np.linspace(.08,.92,5);y=np.linspace(-.02,.03,5);A=rng.normal(size=(5,5));C=(A@A.T)*1e-8+np.eye(5)*2e-8;m=np.column_stack((x,y,C));cx=np.linspace(.1,1.,5);cr=rng.normal(scale=.0015,size=5);sg=np.array([.0007,.0011,.0017,.0024,.0033]);lg=np.array([.05,.11,.23,.47,.9]);noise=0.00028000000000000003', 'call': 'augment_finite_size_covariance(m,cx,cr,sg,lg,noise)', 'gold_call': '_oracle_augment_finite_size_covariance(m,cx,cr,sg,lg,noise)'}, {'setup': 'import numpy as np\nrng=np.random.default_rng(103);x=np.linspace(.08,.92,6);y=np.linspace(-.02,.03,6);A=rng.normal(size=(6,6));C=(A@A.T)*1e-8+np.eye(6)*2e-8;m=np.column_stack((x,y,C));cx=np.linspace(.1,1.,4);cr=rng.normal(scale=.0015,size=4);sg=np.array([.0007,.0011,.0017,.0024,.0033]);lg=np.array([.05,.11,.23,.47,.9]);noise=0.00031', 'call': 'augment_finite_size_covariance(m,cx,cr,sg,lg,noise)', 'gold_call': '_oracle_augment_finite_size_covariance(m,cx,cr,sg,lg,noise)'}, {'setup': 'import numpy as np\nrng=np.random.default_rng(104);x=np.linspace(.08,.92,4);y=np.linspace(-.02,.03,4);A=rng.normal(size=(4,4));C=(A@A.T)*1e-8+np.eye(4)*2e-8;m=np.column_stack((x,y,C));cx=np.linspace(.1,1.,6);cr=rng.normal(scale=.0015,size=6);sg=np.array([.0007,.0011,.0017,.0024,.0033]);lg=np.array([.05,.11,.23,.47,.9]);noise=0.00034', 'call': 'augment_finite_size_covariance(m,cx,cr,sg,lg,noise)', 'gold_call': '_oracle_augment_finite_size_covariance(m,cx,cr,sg,lg,noise)'}, {'setup': 'import numpy as np\nrng=np.random.default_rng(105);x=np.linspace(.08,.92,7);y=np.linspace(-.02,.03,7);A=rng.normal(size=(7,7));C=(A@A.T)*1e-8+np.eye(7)*2e-8;m=np.column_stack((x,y,C));cx=np.linspace(.1,1.,5);cr=rng.normal(scale=.0015,size=5);sg=np.array([.0007,.0011,.0017,.0024,.0033]);lg=np.array([.05,.11,.23,.47,.9]);noise=0.00037', 'call': 'augment_finite_size_covariance(m,cx,cr,sg,lg,noise)', 'gold_call': '_oracle_augment_finite_size_covariance(m,cx,cr,sg,lg,noise)'}, {'setup': 'import numpy as np\nrng=np.random.default_rng(106);x=np.linspace(.08,.92,5);y=np.linspace(-.02,.03,5);A=rng.normal(size=(5,5));C=(A@A.T)*1e-8+np.eye(5)*2e-8;m=np.column_stack((x,y,C));cx=np.linspace(.1,1.,7);cr=rng.normal(scale=.0015,size=7);sg=np.array([.0007,.0011,.0017,.0024,.0033]);lg=np.array([.05,.11,.23,.47,.9]);noise=0.0004', 'call': 'augment_finite_size_covariance(m,cx,cr,sg,lg,noise)', 'gold_call': '_oracle_augment_finite_size_covariance(m,cx,cr,sg,lg,noise)'}, {'setup': 'import numpy as np\nrng=np.random.default_rng(107);x=np.linspace(.08,.92,6);y=np.linspace(-.02,.03,6);A=rng.normal(size=(6,6));C=(A@A.T)*1e-8+np.eye(6)*2e-8;m=np.column_stack((x,y,C));cx=np.linspace(.1,1.,6);cr=rng.normal(scale=.0015,size=6);sg=np.array([.0007,.0011,.0017,.0024,.0033]);lg=np.array([.05,.11,.23,.47,.9]);noise=0.00043000000000000004', 'call': 'augment_finite_size_covariance(m,cx,cr,sg,lg,noise)', 'gold_call': '_oracle_augment_finite_size_covariance(m,cx,cr,sg,lg,noise)'}, {'setup': 'import numpy as np\nrng=np.random.default_rng(108);x=np.linspace(.08,.92,8);y=np.linspace(-.02,.03,8);A=rng.normal(size=(8,8));C=(A@A.T)*1e-8+np.eye(8)*2e-8;m=np.column_stack((x,y,C));cx=np.linspace(.1,1.,6);cr=rng.normal(scale=.0015,size=6);sg=np.array([.0007,.0011,.0017,.0024,.0033]);lg=np.array([.05,.11,.23,.47,.9]);noise=0.00046', 'call': 'augment_finite_size_covariance(m,cx,cr,sg,lg,noise)', 'gold_call': '_oracle_augment_finite_size_covariance(m,cx,cr,sg,lg,noise)'}]

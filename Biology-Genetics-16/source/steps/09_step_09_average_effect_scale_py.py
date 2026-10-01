"""
'''Generalised-least-squares scale coefficient for the mean average effects.




    The mean average effects are modelled as the coefficient times `contrast`, so

    replicate m has expected allele-frequency change equal to the coefficient times

    operators[m] applied to `contrast`, with residual covariance

    drift_covariances[m]. Fit the single coefficient across all replicates jointly

    by generalised least squares, weighting each replicate by the inverse of its

    own residual covariance.




    Parameters

    ----------

    delta_p_by_replicate : np.ndarray

        (n_replicates, n_loci) observed allele-frequency changes, one row per

        replicate, loci in a consistent order.

    operators : list

        Length-n_replicates list of (n_loci, n_loci) expected-change operators.

    drift_covariances : list

        Length-n_replicates list of (n_loci, n_loci) symmetric positive-definite

        drift covariance matrices.

    contrast : np.ndarray

        (n_loci,) reference-allele frequency contrast p - q in the base population.




    Returns

    -------

    coefficient : float

        Native Python float.




    Raises

    ------

    ValueError

        If delta_p_by_replicate is not a finite 2D array with at least one row, if

        operators or drift_covariances is not a list whose length matches the

        number of replicates, if any matrix is not square with size equal to the

        locus count, if contrast is not a finite 1D array of that length, if any

        drift covariance is not symmetric within atol=1e-12 or is not positive

        definite, or if the accumulated information is zero.

    '''

Average effects cannot be estimated locus by locus: over a handful of generations

the change at any single locus is dominated by drift. What can be estimated is the

scale of a low-dimensional model for the average effects across the genome. Under

mutation-selection-drift balance the average effect at a locus is expected to grow

with the reference-allele frequency contrast p - q, so a one-parameter model in

which the mean average effects are proportional to that contrast captures the

signal with a single coefficient.




Because the replicate observations differ in precision - drift variance scales

inversely with the variance effective size, and the replicates here differ in

census size by more than tenfold - the coefficient must be fitted by generalised

least squares against each replicate's own drift covariance, not by an unweighted

pooling of replicates. Each replicate also has its own expected-change operator,

because the erosion of disequilibrium depends on that replicate's effective size.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def average_effect_scale(delta_p_by_replicate: np.ndarray,
                         operators: list, drift_covariances: list,
                         contrast: np.ndarray) -> float:
    '''Generalised-least-squares scale coefficient for the mean average effects.

    The mean average effects are modelled as the coefficient times `contrast`, so
    replicate m has expected allele-frequency change equal to the coefficient times
    operators[m] applied to `contrast`, with residual covariance
    drift_covariances[m]. Fit the single coefficient across all replicates jointly
    by generalised least squares, weighting each replicate by the inverse of its
    own residual covariance.

    Parameters
    ----------
    delta_p_by_replicate : np.ndarray
        (n_replicates, n_loci) observed allele-frequency changes, one row per
        replicate, loci in a consistent order.
    operators : list
        Length-n_replicates list of (n_loci, n_loci) expected-change operators.
    drift_covariances : list
        Length-n_replicates list of (n_loci, n_loci) symmetric positive-definite
        drift covariance matrices.
    contrast : np.ndarray
        (n_loci,) reference-allele frequency contrast p - q in the base population.

    Returns
    -------
    coefficient : float
        Native Python float.

    Raises
    ------
    ValueError
        If delta_p_by_replicate is not a finite 2D array with at least one row, if
        operators or drift_covariances is not a list whose length matches the
        number of replicates, if any matrix is not square with size equal to the
        locus count, if contrast is not a finite 1D array of that length, if any
        drift covariance is not symmetric within atol=1e-12 or is not positive
        definite, or if the accumulated information is zero.
    '''
    return coefficient  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_average_effect_scale(delta_p_by_replicate: np.ndarray,
                                 operators: list, drift_covariances: list,
                                 contrast: np.ndarray) -> float:
    Y = np.asarray(delta_p_by_replicate, dtype=float)
    if Y.ndim != 2 or Y.shape[0] < 1 or Y.shape[1] < 1:
        raise ValueError("delta_p_by_replicate must be a 2D array with >= 1 row")
    if not np.all(np.isfinite(Y)):
        raise ValueError("delta_p_by_replicate must be finite")
    m_rep, n_loci = Y.shape
    if not isinstance(operators, list) or len(operators) != m_rep:
        raise ValueError("operators must be a list with one matrix per replicate")
    if not isinstance(drift_covariances, list) or len(drift_covariances) != m_rep:
        raise ValueError("drift_covariances must be a list with one matrix per replicate")
    x = np.asarray(contrast, dtype=float)
    if x.ndim != 1 or x.size != n_loci or not np.all(np.isfinite(x)):
        raise ValueError("contrast must be a finite 1D array with one entry per locus")

    numerator = 0.0
    information = 0.0
    for m in range(m_rep):
        Lm = np.asarray(operators[m], dtype=float)
        Dm = np.asarray(drift_covariances[m], dtype=float)
        for name, A in (("operator", Lm), ("drift covariance", Dm)):
            if A.ndim != 2 or A.shape[0] != A.shape[1] or A.shape[0] != n_loci:
                raise ValueError(f"every {name} must be square with size n_loci")
            if not np.all(np.isfinite(A)):
                raise ValueError(f"every {name} must be finite")
        if not np.allclose(Dm, Dm.T, rtol=0.0, atol=1e-12):
            raise ValueError("every drift covariance must be symmetric within atol=1e-12")
        # LAPACK's Cholesky accepts a numerically singular matrix whose smallest
        # factor diagonal merely underflows, which would silently return a finite
        # but meaningless solve, so test conditioning directly and scale-invariantly.
        eig = np.linalg.eigvalsh(Dm)
        if eig[-1] <= 0.0 or eig[0] <= 1e-12 * eig[-1]:
            raise ValueError("every drift covariance must be positive definite "
                             "with condition number below 1e12")
        chol = np.linalg.cholesky(Dm)
        z = Lm @ x
        w = np.linalg.solve(chol.T, np.linalg.solve(chol, np.column_stack([z, Y[m]])))
        numerator += float(z @ w[:, 1])
        information += float(z @ w[:, 0])
    if information == 0.0:
        raise ValueError("accumulated information is zero; the coefficient is not identified")
    return float(numerator / information)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    _SETUP = """import numpy as np
Y = np.array([
 [-0.080508, -0.088382, -0.092691, -0.079427, -0.029581,  0.007478,  0.057723,  0.081924,  0.063215,  0.058229],
 [-0.087222, -0.098856, -0.084800, -0.045451, -0.039817, -0.008562,  0.026788,  0.041907,  0.049555,  0.057158],
 [-0.104796, -0.131458, -0.065336, -0.087962, -0.065417, -0.008970,  0.015046,  0.058879,  0.065851,  0.072945],
 [-0.076533, -0.060456, -0.062452, -0.045712, -0.024884,  0.022225,  0.068221,  0.082374,  0.104465,  0.091738],
 [-0.063602, -0.083285, -0.104352, -0.102128, -0.025376,  0.053171,  0.009892,  0.118097,  0.088889,  0.056976],
 [-0.079714, -0.159038, -0.173297, -0.097500, -0.076308,  0.005407,  0.028781,  0.104331,  0.044838,  0.049813]])
rng = np.random.default_rng(2043)
Z = rng.standard_normal((4000, 10))
X = np.empty_like(Z); X[:, 0] = Z[:, 0]
s = np.sqrt(1 - 0.90 ** 2)
for i in range(1, 10):
    X[:, i] = 0.90 * X[:, i - 1] + s * Z[:, i]
thr = np.array([ 1.0364334,  0.7461852,  0.5084881,  0.2967378,  0.0976349,
                -0.0976349, -0.2967378, -0.5084881, -0.7461852, -1.0364334])
H = (X > thr[None, :]).astype(float)
d = np.concatenate([[0.0], np.cumsum(-0.5 * np.log1p(-2.0 * np.full(9, 0.05)))])
R = 0.5 * (1.0 - np.exp(-2.0 * np.abs(d[:, None] - d[None, :])))
p = H.mean(axis=0)
a, b = H[0::2] - p[None, :], H[1::2] - p[None, :]
Lg = 0.25 * (a.T @ a + b.T @ b) / a.shape[0]
Lx = 0.25 * (a.T @ b + b.T @ a) / a.shape[0]
Lt = Lg + (R / (1.0 - R)) * Lx
census = [2000, 1200, 700, 400, 250, 150]
ops, cov = [], []
for N in census:
    ne, nE = float(N), 4.0 * N / (2.0 + 6.0)
    f0, dr = 1.0 - 1.0 / (2.0 * 50.0), 1.0 - 1.0 / (2.0 * ne)
    Nt = [((1.0 - R) ** t) * f0 * (dr ** (t - 1)) for t in (1, 2)]
    Lm = sum(n_t * Lt for n_t in Nt)
    Dm = Lt * sum(((1.0 - R) / nE) * n_t for n_t in Nt)
    ops.append(Lm); cov.append(Dm)
x = 2.0 * p - 1.0
"""
    return [
        # --- normal: all six shipped replicates ---
        {
            "setup": _SETUP,
            "call": "average_effect_scale(Y, ops, cov, x)",
            "gold_call": "_oracle_average_effect_scale(Y, ops, cov, x)",
        },
        # --- boundary: a single replicate, so the fit is exactly determined ---
        {
            "setup": _SETUP,
            "call": "average_effect_scale(Y[:1], ops[:1], cov[:1], x)",
            "gold_call": "_oracle_average_effect_scale(Y[:1], ops[:1], cov[:1], x)",
        },
        # --- edge: two replicates of very unequal precision, where an unweighted
        #     pooling and a GLS fit give visibly different coefficients
        #     (alone, replicate 1 implies 0.5 and replicate 2 implies 2.0;
        #     GLS gives 0.5015, unweighted pooling 1.25) ---
        {
            "setup": """import numpy as np
Y = np.array([[0.02, -0.02],
              [0.08, -0.08]])
L1 = np.array([[0.10, 0.02], [0.02, 0.10]])
ops = [L1, L1]
cov = [np.array([[1e-5, 0.0], [0.0, 1e-5]]), np.array([[1e-2, 0.0], [0.0, 1e-2]])]
x = np.array([0.5, -0.5])
""",
            "call": "average_effect_scale(Y, ops, cov, x)",
            "gold_call": "_oracle_average_effect_scale(Y, ops, cov, x)",
        },
    ]

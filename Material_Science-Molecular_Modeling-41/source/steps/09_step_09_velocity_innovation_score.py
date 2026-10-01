"""
Evaluate the stationary Gaussian innovation likelihood of probe velocities.

A held-out velocity trajectory is scored by Gaussian innovations. Irregular observation intervals and hidden correlations enter every prediction.

Returns
-------
Finite Python float, negative log likelihood in nats per scalar     normalized velocity. Negative values are allowed for a density.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def velocity_innovation_score(Sigma: "np.ndarray", transitions: "np.ndarray", noise_covariances: "np.ndarray", observations: "np.ndarray", observation_covariance: "np.ndarray") -> float:
    """Evaluate the stationary Gaussian innovation likelihood of probe velocities.

    Parameters
    ----------
    Sigma : finite real stationary covariance (q,q), positive definite
    transitions, noise_covariances : finite real arrays (m-1,q,q)
    observations : finite real array (m,d), m>=1, 1<=d<=q
        Dimensionless velocities v*sqrt(mass/(k_B*T)).
    observation_covariance : finite real symmetric positive definite (d,d)
        Independent measurement-error covariance in those same units.

    Contract
    --------
    Evaluate the complete joint stationary Gaussian likelihood of all
    measurements of the first d state coordinates, with zero mean and
    initial state covariance Sigma. Use each supplied transition and
    process covariance for its corresponding interval and include the
    independent measurement covariance, the initial observation and
    Gaussian normalization constants. Divide by m*d. Equivalent dense
    joint-covariance and sequential-conditioning calculations are valid.
    Symmetry tolerance is 1e-7; positive definite means minimum >1e-10;
    process covariances may be semidefinite down to -1e-8.

    Returns
    -------
    result
        Finite Python float, negative log likelihood in nats per scalar
        normalized velocity. Negative values are allowed for a density.

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, nonsymmetric or
        nonpositive Sigma/R, nonsymmetric or non-semidefinite Q under
        the stated tolerances, or a nonpositive innovation covariance.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_velocity_innovation_score(Sigma: "np.ndarray", transitions: "np.ndarray", noise_covariances: "np.ndarray", observations: "np.ndarray", observation_covariance: "np.ndarray") -> float:
    """Evaluate the stationary Gaussian innovation likelihood of probe velocities.

    Parameters
    ----------
    Sigma : finite real stationary covariance (q,q), positive definite
    transitions, noise_covariances : finite real arrays (m-1,q,q)
    observations : finite real array (m,d), m>=1, 1<=d<=q
        Dimensionless velocities v*sqrt(mass/(k_B*T)).
    observation_covariance : finite real symmetric positive definite (d,d)
        Independent measurement-error covariance in those same units.

    Contract
    --------
    Evaluate the complete joint stationary Gaussian likelihood of all
    measurements of the first d state coordinates, with zero mean and
    initial state covariance Sigma. Use each supplied transition and
    process covariance for its corresponding interval and include the
    independent measurement covariance, the initial observation and
    Gaussian normalization constants. Divide by m*d. Equivalent dense
    joint-covariance and sequential-conditioning calculations are valid.
    Symmetry tolerance is 1e-7; positive definite means minimum >1e-10;
    process covariances may be semidefinite down to -1e-8.

    Returns
    -------
    result
        Finite Python float, negative log likelihood in nats per scalar
        normalized velocity. Negative values are allowed for a density.

    Raises
    ------
    ValueError
        Complex/nonfinite input, incompatible shapes, nonsymmetric or
        nonpositive Sigma/R, nonsymmetric or non-semidefinite Q under
        the stated tolerances, or a nonpositive innovation covariance.
    """
    import numpy as np
    if any((np.iscomplexobj(x) for x in [Sigma, transitions, noise_covariances, observations, observation_covariance])):
        raise ValueError('real input')
    S0, T, Q, y, R = [np.asarray(x, float) for x in [Sigma, transitions, noise_covariances, observations, observation_covariance]]
    if y.ndim != 2 or len(y) < 1 or y.shape[1] < 1 or (S0.ndim != 2) or (S0.shape[0] != S0.shape[1]):
        raise ValueError('data shapes')
    m, d = y.shape
    q = len(S0)
    if d > q or R.shape != (d, d) or T.shape != (m - 1, q, q) or (Q.shape != T.shape) or (not all((np.isfinite(x).all() for x in [S0, T, Q, y, R]))):
        raise ValueError('data shapes')
    for x in [S0, R]:
        if np.max(abs(x - x.T)) > 1e-07 or np.min(np.linalg.eigvalsh((x + x.T) / 2)) <= 1e-10:
            raise ValueError('positive covariance')
    for x in Q:
        if np.max(abs(x - x.T)) > 1e-07 or np.min(np.linalg.eigvalsh((x + x.T) / 2)) < -1e-08:
            raise ValueError('transition noise')
    mean = np.zeros(q)
    P = S0.copy()
    total = 0.0
    for k, row in enumerate(y):
        residual = row - mean[:d]
        S = P[:d, :d] + R
        S = (S + S.T) / 2
        try:
            L = np.linalg.cholesky(S)
        except np.linalg.LinAlgError as e:
            raise ValueError('innovation covariance') from e
        u = np.linalg.solve(L, residual)
        total += 0.5 * (d * np.log(2 * np.pi) + 2 * np.log(np.diag(L)).sum() + u @ u)
        gain = np.linalg.solve(L.T, np.linalg.solve(L, P[:, :d].T)).T
        mean = mean + gain @ residual
        P = P - gain @ S @ gain.T
        P = (P + P.T) / 2
        if k < m - 1:
            mean = T[k] @ mean
            P = T[k] @ P @ T[k].T + Q[k]
            P = (P + P.T) / 2
    return float(total / (m * d))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return independent numerical test specifications."""
    common_0 = """import copy
import numpy as np
from scipy.linalg import expm, block_diag

def near(actual, expected, atol=2e-06, rtol=2e-06):
    a = np.asarray(actual)
    b = np.asarray(expected)
    return int(a.shape == b.shape and np.isfinite(a).all() and np.allclose(a, b, atol=atol, rtol=rtol))

def true_drift(case=0):
    if case == 2:
        return (np.array([[0.0, 1.4, 0.2], [-1.4, -0.8, 1.1], [-0.2, -1.1, -1.2]]), 1)
    B = np.array([[1.2, 0.25], [-0.35, 1.05], [0.7, -0.55], [0.4, 0.8]])
    J = np.array([[0, 1.3, -0.2, 0.4], [-1.3, 0, 0.65, -0.3], [0.2, -0.65, 0, 0.9], [-0.4, 0.3, -0.9, 0.0]])
    A = np.block([[np.zeros((2, 2)), B.T], [-B, J - np.diag([0.6, 1.1, 0.85, 1.4])]])
    return (A * (0.8 if case == 1 else 1.0), 2)

def dense_score(A, d, observations, intervals, R):
    times = np.r_[0.0, np.cumsum(intervals)]
    m = len(times)
    C = np.empty((m * d, m * d))
    for i in range(m):
        for j in range(i + 1):
            v = expm((times[i] - times[j]) * A)[:d, :d] + (R if i == j else 0)
            C[i * d:(i + 1) * d, j * d:(j + 1) * d] = v
            C[j * d:(j + 1) * d, i * d:(i + 1) * d] = v.T
    sign, logdet = np.linalg.slogdet(C)
    y = observations.ravel()
    return float((len(y) * np.log(2 * np.pi) + logdet + y @ np.linalg.solve(C, y)) / (2 * len(y)))
'Deterministic synthetic probe data; not measurements reported in the article.'

def make_inputs(case=0, seed=20260912, count=24):
    import numpy as np
    from scipy.linalg import expm
    if case == 2:
        A = np.array([[0.0, 1.4, 0.2], [-1.4, -0.8, 1.1], [-0.2, -1.1, -1.2]])
        d = 1
    else:
        B = np.array([[1.2, 0.25], [-0.35, 1.05], [0.7, -0.55], [0.4, 0.8]])
        J = np.array([[0, 1.3, -0.2, 0.4], [-1.3, 0, 0.65, -0.3], [0.2, -0.65, 0, 0.9], [-0.4, 0.3, -0.9, 0.0]])
        A0 = J - np.diag([0.6, 1.1, 0.85, 1.4])
        A = np.block([[np.zeros((2, 2)), B.T], [-B, A0]])
        d = 2
        if case == 1:
            A *= 0.8
    tau = 0.4
    mass = 5.0
    thermal_energy = 1.0
    Y = np.array([expm(k * tau * A)[:d, :d] for k in range(128)])
    cvv = thermal_energy / mass * Y
    intervals = 0.22 + 0.04 * (np.arange(count - 1) % 5)
    times = np.r_[0.0, np.cumsum(intervals)]
    R = 0.008 * np.array([[1.0, 0.3], [0.3, 1.7]]) if d == 2 else np.array([[0.011]])
    covariance = np.empty((count * d, count * d))
    for i in range(count):
        for j in range(i + 1):
            block = expm((times[i] - times[j]) * A)[:d, :d]
            if i == j:
                block = block + R
            covariance[i * d:(i + 1) * d, j * d:(j + 1) * d] = block
            covariance[j * d:(j + 1) * d, i * d:(i + 1) * d] = block.T
    rng = np.random.default_rng(seed)
    normalized = (np.linalg.cholesky(covariance) @ rng.standard_normal(count * d)).reshape(count, d)
    return dict(cvv=cvv, tau=tau, mass=mass, thermal_energy=thermal_energy, observations=normalized * np.sqrt(thermal_energy / mass), intervals=intervals, observation_covariance=R, rho=1.4, grid_size=64, fit_tolerance=1e-11, max_support=18, rank_tolerance=1e-07)
"""
    common_1 = """import copy
import numpy as np
from scipy.linalg import expm, block_diag

def error_code(function, *args, **kwargs):
    try:
        function(*args, **kwargs)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
'Deterministic synthetic probe data; not measurements reported in the article.'

def make_inputs(case=0, seed=20260912, count=24):
    import numpy as np
    from scipy.linalg import expm
    if case == 2:
        A = np.array([[0.0, 1.4, 0.2], [-1.4, -0.8, 1.1], [-0.2, -1.1, -1.2]])
        d = 1
    else:
        B = np.array([[1.2, 0.25], [-0.35, 1.05], [0.7, -0.55], [0.4, 0.8]])
        J = np.array([[0, 1.3, -0.2, 0.4], [-1.3, 0, 0.65, -0.3], [0.2, -0.65, 0, 0.9], [-0.4, 0.3, -0.9, 0.0]])
        A0 = J - np.diag([0.6, 1.1, 0.85, 1.4])
        A = np.block([[np.zeros((2, 2)), B.T], [-B, A0]])
        d = 2
        if case == 1:
            A *= 0.8
    tau = 0.4
    mass = 5.0
    thermal_energy = 1.0
    Y = np.array([expm(k * tau * A)[:d, :d] for k in range(128)])
    cvv = thermal_energy / mass * Y
    intervals = 0.22 + 0.04 * (np.arange(count - 1) % 5)
    times = np.r_[0.0, np.cumsum(intervals)]
    R = 0.008 * np.array([[1.0, 0.3], [0.3, 1.7]]) if d == 2 else np.array([[0.011]])
    covariance = np.empty((count * d, count * d))
    for i in range(count):
        for j in range(i + 1):
            block = expm((times[i] - times[j]) * A)[:d, :d]
            if i == j:
                block = block + R
            covariance[i * d:(i + 1) * d, j * d:(j + 1) * d] = block
            covariance[j * d:(j + 1) * d, i * d:(i + 1) * d] = block.T
    rng = np.random.default_rng(seed)
    normalized = (np.linalg.cholesky(covariance) @ rng.standard_normal(count * d)).reshape(count, d)
    return dict(cvv=cvv, tau=tau, mass=mass, thermal_energy=thermal_energy, observations=normalized * np.sqrt(thermal_energy / mass), intervals=intervals, observation_covariance=R, rho=1.4, grid_size=64, fit_tolerance=1e-11, max_support=18, rank_tolerance=1e-07)
data = make_inputs(count=1)
y = data['observations']
R = data['observation_covariance']
"""
    return [
        {
            'setup': common_0 + """data = make_inputs(0, count=24)
A, d = true_drift(0)
dt = data['intervals']
q = len(A)
T = np.array([expm(h * A) for h in dt]).reshape(len(dt), q, q)
Q = np.array([np.eye(q) - v @ v.T for v in T]).reshape(len(dt), q, q)
y = data['observations'] * np.sqrt(5.0)
R = data['observation_covariance']
Sigma = np.eye(q)
expected = dense_score(A, d, y, dt, R)
""",
            'call': 'near(velocity_innovation_score(copy.deepcopy(Sigma), copy.deepcopy(T), copy.deepcopy(Q), copy.deepcopy(y), copy.deepcopy(R)), expected, atol=1e-08, rtol=1e-08)',
            'gold_call': 'near(_oracle_velocity_innovation_score(copy.deepcopy(Sigma), copy.deepcopy(T), copy.deepcopy(Q), copy.deepcopy(y), copy.deepcopy(R)), expected, atol=1e-08, rtol=1e-08)',
        },
        {
            'setup': common_0 + """data = make_inputs(1, count=7)
A, d = true_drift(1)
dt = data['intervals']
q = len(A)
T = np.array([expm(h * A) for h in dt]).reshape(len(dt), q, q)
Q = np.array([np.eye(q) - v @ v.T for v in T]).reshape(len(dt), q, q)
y = data['observations'] * np.sqrt(5.0)
R = data['observation_covariance']
Sigma = np.eye(q)
expected = dense_score(A, d, y, dt, R)
""",
            'call': 'near(velocity_innovation_score(copy.deepcopy(Sigma), copy.deepcopy(T), copy.deepcopy(Q), copy.deepcopy(y), copy.deepcopy(R)), expected, atol=1e-08, rtol=1e-08)',
            'gold_call': 'near(_oracle_velocity_innovation_score(copy.deepcopy(Sigma), copy.deepcopy(T), copy.deepcopy(Q), copy.deepcopy(y), copy.deepcopy(R)), expected, atol=1e-08, rtol=1e-08)',
        },
        {
            'setup': common_0 + """data = make_inputs(2, count=1)
A, d = true_drift(2)
dt = data['intervals']
q = len(A)
T = np.array([expm(h * A) for h in dt]).reshape(len(dt), q, q)
Q = np.array([np.eye(q) - v @ v.T for v in T]).reshape(len(dt), q, q)
y = data['observations'] * np.sqrt(5.0)
R = data['observation_covariance']
Sigma = np.eye(q)
expected = dense_score(A, d, y, dt, R)
""",
            'call': 'near(velocity_innovation_score(copy.deepcopy(Sigma), copy.deepcopy(T), copy.deepcopy(Q), copy.deepcopy(y), copy.deepcopy(R)), expected, atol=1e-08, rtol=1e-08)',
            'gold_call': 'near(_oracle_velocity_innovation_score(copy.deepcopy(Sigma), copy.deepcopy(T), copy.deepcopy(Q), copy.deepcopy(y), copy.deepcopy(R)), expected, atol=1e-08, rtol=1e-08)',
        },
        {
            'setup': common_1,
            'call': 'error_code(velocity_innovation_score, copy.deepcopy(np.eye(2)), copy.deepcopy(np.empty((0, 2, 2))), copy.deepcopy(np.empty((0, 2, 2))), copy.deepcopy(y), copy.deepcopy(-R))',
            'gold_call': 'error_code(_oracle_velocity_innovation_score, copy.deepcopy(np.eye(2)), copy.deepcopy(np.empty((0, 2, 2))), copy.deepcopy(np.empty((0, 2, 2))), copy.deepcopy(y), copy.deepcopy(-R))',
        },
        {
            'setup': common_1,
            'call': 'error_code(velocity_innovation_score, copy.deepcopy(np.eye(2).astype(complex)), copy.deepcopy(np.empty((0, 2, 2))), copy.deepcopy(np.empty((0, 2, 2))), copy.deepcopy(y), copy.deepcopy(R))',
            'gold_call': 'error_code(_oracle_velocity_innovation_score, copy.deepcopy(np.eye(2).astype(complex)), copy.deepcopy(np.empty((0, 2, 2))), copy.deepcopy(np.empty((0, 2, 2))), copy.deepcopy(y), copy.deepcopy(R))',
        },
        {
            'setup': common_0 + """data = make_inputs(0, count=24)
A, d = true_drift(0)
dt = data['intervals']
q = len(A)
T = np.array([expm(h * A) for h in dt]).reshape(len(dt), q, q)
Q = np.array([np.eye(q) - v @ v.T for v in T]).reshape(len(dt), q, q)
y = data['observations'] * np.sqrt(5.0)
R = data['observation_covariance']
Sigma = np.eye(q)
expected = dense_score(A, d, y, dt, R)
U = block_diag(np.eye(d),np.array([[1.8,.2,0.,0.],[0.,.7,.1,0.],[0.,0.,1.4,.3],[.1,0.,0.,.6]]))
Ui = np.linalg.inv(U)
Sigma = U @ U.T
T = np.array([U @ v @ Ui for v in T])
Q = np.array([U @ v @ U.T for v in Q])
""",
            'call': 'velocity_innovation_score(copy.deepcopy(Sigma.copy()), copy.deepcopy(T.copy()), copy.deepcopy(Q.copy()), copy.deepcopy(y.copy()), copy.deepcopy(R.copy()))',
            'gold_call': '_oracle_velocity_innovation_score(copy.deepcopy(Sigma.copy()), copy.deepcopy(T.copy()), copy.deepcopy(Q.copy()), copy.deepcopy(y.copy()), copy.deepcopy(R.copy()))',
            'tol': 1e-09,
        },
    ]

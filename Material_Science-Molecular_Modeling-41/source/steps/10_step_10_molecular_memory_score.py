"""
Reconstruct an inertial vector memory model and score a held-out trajectory.

The final routine reconstructs the vector memory model and evaluates the trajectory density in normalized velocity units.

Returns
-------
Finite Python float: stationary negative log likelihood per scalar     normalized velocity, in nats.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def molecular_memory_score(cvv: "np.ndarray", tau: float, mass: float, thermal_energy: float, observations: "np.ndarray", intervals: "np.ndarray", observation_covariance: "np.ndarray", rho: float = 1.4, grid_size: int = 64, fit_tolerance: float = 0.001, max_support: int = 18, rank_tolerance: float = 1e-07) -> float:
    """Reconstruct an inertial vector memory model and score a held-out trajectory.

    Parameters
    ----------
    cvv : real (n,d,d), n>=4, normalized zero-lag mass*cvv[0]/k_B*T=I
    tau : positive correlation sample spacing
    mass, thermal_energy : positive mass and k_B*T
    observations : real (m,d), m>=1, physical velocities
    intervals : real (m-1,), increments in (0,3] between observations
    observation_covariance : real positive definite (d,d)
        Measurement covariance of normalized velocities, already scaled.
    rho=1.4, grid_size=64, fit_tolerance=0.001, max_support=18,
    rank_tolerance=1e-7 : numerical parameters
        Their domains and conventions are defined in the earlier steps.

    Definition
    ----------
    Execute correlation_transform, shared_rational_fit, continuous_poles,
    constrained_residues, real_velocity_embedding, singular_memory_reduction,
    regular_lure, complete_discrete_embedding, and velocity_innovation_score,
    in that order. Multiply observations by sqrt(mass/thermal_energy).
    The likelihood is a density in normalized velocity coordinates; do
    not add a change-of-units Jacobian. Preserve every cross correlation.
    Inputs must admit the stable positive-real inertial realization with
    q>=2*d specified by those steps. All their rejection conditions apply.
    Final score comparisons use absolute tolerance 2e-6. A numerically
    equivalent realization is accepted. The fitted parameters, covariance,
    and noise must be derived from these inputs, without stored answers.
    Import required packages inside the function.

    Returns
    -------
    result
        Finite Python float: stationary negative log likelihood per scalar
        normalized velocity, in nats.

    Raises
    ------
    ValueError
        Any invalid input or failed admissibility condition declared by
        an earlier step; in addition, observation dimension must equal the
        cvv dimension and intervals must have length len(observations)-1.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_molecular_memory_score(cvv: "np.ndarray", tau: float, mass: float, thermal_energy: float, observations: "np.ndarray", intervals: "np.ndarray", observation_covariance: "np.ndarray", rho: float = 1.4, grid_size: int = 64, fit_tolerance: float = 0.001, max_support: int = 18, rank_tolerance: float = 1e-07) -> float:
    """Reconstruct an inertial vector memory model and score a held-out trajectory.

    Parameters
    ----------
    cvv : real (n,d,d), n>=4, normalized zero-lag mass*cvv[0]/k_B*T=I
    tau : positive correlation sample spacing
    mass, thermal_energy : positive mass and k_B*T
    observations : real (m,d), m>=1, physical velocities
    intervals : real (m-1,), increments in (0,3] between observations
    observation_covariance : real positive definite (d,d)
        Measurement covariance of normalized velocities, already scaled.
    rho=1.4, grid_size=64, fit_tolerance=0.001, max_support=18,
    rank_tolerance=1e-7 : numerical parameters
        Their domains and conventions are defined in the earlier steps.

    Definition
    ----------
    Execute correlation_transform, shared_rational_fit, continuous_poles,
    constrained_residues, real_velocity_embedding, singular_memory_reduction,
    regular_lure, complete_discrete_embedding, and velocity_innovation_score,
    in that order. Multiply observations by sqrt(mass/thermal_energy).
    The likelihood is a density in normalized velocity coordinates; do
    not add a change-of-units Jacobian. Preserve every cross correlation.
    Inputs must admit the stable positive-real inertial realization with
    q>=2*d specified by those steps. All their rejection conditions apply.
    Final score comparisons use absolute tolerance 2e-6. A numerically
    equivalent realization is accepted. The fitted parameters, covariance,
    and noise must be derived from these inputs, without stored answers.
    Import required packages inside the function.

    Returns
    -------
    result
        Finite Python float: stationary negative log likelihood per scalar
        normalized velocity, in nats.

    Raises
    ------
    ValueError
        Any invalid input or failed admissibility condition declared by
        an earlier step; in addition, observation dimension must equal the
        cvv dimension and intervals must have length len(observations)-1.
    """
    import numpy as np
    Y, z, F = _oracle_correlation_transform(cvv, mass, thermal_energy, rho, grid_size)
    if np.iscomplexobj(observations):
        raise ValueError('real observations')
    obs = np.asarray(observations, float)
    if obs.ndim != 2 or len(obs) < 1 or obs.shape[1] != Y.shape[1] or (np.shape(intervals) != (len(obs) - 1,)):
        raise ValueError('observations')
    indices, w = _oracle_shared_rational_fit(z, F, fit_tolerance, max_support)
    rates = _oracle_continuous_poles(z[indices], w, tau)
    residues = _oracle_constrained_residues(Y, tau, rates)
    A = _oracle_real_velocity_embedding(rates, residues, rank_tolerance)
    d = Y.shape[1]
    A1, B1, C1, D1, X = _oracle_singular_memory_reduction(A, d)
    S1, L1, K1 = _oracle_regular_lure(A1, B1, C1, D1)
    Sigma, T, Q = _oracle_complete_discrete_embedding(A, d, X, S1, L1, K1, intervals)
    return _oracle_velocity_innovation_score(Sigma, T, Q, obs * np.sqrt(mass / thermal_energy), observation_covariance)

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
data = make_inputs()
"""
    return [
        {
            'setup': common_0 + """data = make_inputs(0, count=24)
A, d = true_drift(0)
expected = dense_score(A, d, data['observations'] * np.sqrt(5.0), data['intervals'], data['observation_covariance'])
""",
            'call': 'near(molecular_memory_score(**copy.deepcopy(data)), expected, atol=2e-06, rtol=0.0)',
            'gold_call': 'near(_oracle_molecular_memory_score(**copy.deepcopy(data)), expected, atol=2e-06, rtol=0.0)',
        },
        {
            'setup': common_0 + """data = make_inputs(1, count=7)
A, d = true_drift(1)
expected = dense_score(A, d, data['observations'] * np.sqrt(5.0), data['intervals'], data['observation_covariance'])
""",
            'call': 'near(molecular_memory_score(**copy.deepcopy(data)), expected, atol=2e-06, rtol=0.0)',
            'gold_call': 'near(_oracle_molecular_memory_score(**copy.deepcopy(data)), expected, atol=2e-06, rtol=0.0)',
        },
        {
            'setup': common_0 + """data = make_inputs(2, count=1)
A, d = true_drift(2)
expected = dense_score(A, d, data['observations'] * np.sqrt(5.0), data['intervals'], data['observation_covariance'])
""",
            'call': 'near(molecular_memory_score(**copy.deepcopy(data)), expected, atol=2e-06, rtol=0.0)',
            'gold_call': 'near(_oracle_molecular_memory_score(**copy.deepcopy(data)), expected, atol=2e-06, rtol=0.0)',
        },
        {
            'setup': common_1 + """data['observations'] = data['observations'].astype(complex)
""",
            'call': 'error_code(molecular_memory_score, **copy.deepcopy(data))',
            'gold_call': 'error_code(_oracle_molecular_memory_score, **copy.deepcopy(data))',
        },
        {
            'setup': common_1 + """data['intervals'] = np.array([0.2])
""",
            'call': 'error_code(molecular_memory_score, **copy.deepcopy(data))',
            'gold_call': 'error_code(_oracle_molecular_memory_score, **copy.deepcopy(data))',
        },
        {
            'setup': """import copy
import numpy as np
data = {
    'cvv': np.array([[[0.2, 0.0], [0.0, 0.2]],
     [[0.16945275355461512, -0.0031061281066597324], [0.006318887056165058, 0.17215784336029188]],
     [[0.10228698581513448, -0.02294982511747461], [0.032306787674341315, 0.11678141426416602]],
     [[0.03535322878232866, -0.05537457355094818], [0.06718706424579374, 0.0664658578601601]],
     [[-0.012070503948953008, -0.08250120144868953], [0.09182743610774419, 0.03442788237776536]],
     [[-0.036567907394091034, -0.08967786383321005], [0.09367154682294565, 0.01629099765430479]],
     [[-0.04805115675008539, -0.07612552988109926], [0.07524453360736394, 0.0013469836284512177]],
     [[-0.052980487861631634, -0.04976334460550338], [0.04744033651063496, -0.014399023985480316]],
     [[-0.053573674685929565, -0.021844282384545315], [0.02084220030053809, -0.029663465237752137]],
     [[-0.04940535547554183, -1.4279943422836207e-05], [0.00035011170274973304, -0.03935188077887899]],
     [[-0.03915422712825922, 0.013814255171149876], [-0.01236497699941597, -0.04059198783712678]],
     [[-0.02496308971383378, 0.021156445919811882], [-0.020382843812545563, -0.03462103679445504]],
     [[-0.009249846254845411, 0.024686491046292593], [-0.024208678295770638, -0.023564731597636782]],
     [[0.004365438477382304, 0.025193834506066343], [-0.02508201946078254, -0.012288311204434854]],
     [[0.01358990676609818, 0.023031162770754537], [-0.02327059893817632, -0.003142777389224482]],
     [[0.018122885841297225, 0.01836972622396861], [-0.0184703043130576, 0.003931289913838741]],
     [[0.01873215766456137, 0.01183830314128188], [-0.01192020618640197, 0.008058969618742067]],
     [[0.016716952695203933, 0.004982808835726663], [-0.005172428744607777, 0.01061720087710268]],
     [[0.013218619742006825, -0.0009571182041697382], [0.0006421196033715314, 0.011361386530192908]],
     [[0.009086035078043791, -0.005229656083456315], [0.004915736183000062, 0.010582115084983463]],
     [[0.004830636209681205, -0.007547881529941231], [0.007565730075036082, 0.00851557517510024]],
     [[0.0007359987473980553, -0.008467173841718117], [0.007831263449807235, 0.005906063360175109]],
     [[-0.0022509390924260733, -0.007806776358392653], [0.007599086011296447, 0.002895564879486651]],
     [[-0.004721944583818917, -0.006203691780404964], [0.006293874655016793, 0.00017913119095960882]],
     [[-0.00563401891759615, -0.004506168716168017], [0.004532862513282481, -0.0018762386088536347]],
     [[-0.005623675093181799, -0.0026182951153545355], [0.002659209082206636, -0.002925842514936929]],
     [[-0.004788167623683006, -0.0005524811008411267], [0.0008970014639894886, -0.0033943354542712305]],
     [[-0.003616693700975458, 0.0006469192711624382], [-0.0007522478167353058, -0.0033516718421862162]],
     [[-0.002188476128449707, 0.0019439810271163801], [-0.0018834007125281215, -0.0027859983711293447]],
     [[-0.0007038256616400821, 0.0024633839224239553], [-0.002480797344634457, -0.0022709111605531012]],
     [[0.00020349630111010425, 0.002615696077141865], [-0.0024982768937633917, -0.001307052561986828]],
     [[0.0010832116729455527, 0.0023144301388396396], [-0.0021782710761860244, -0.0006492020796481774]],
     [[0.001704387842520097, 0.0016631006641346282], [-0.0017135325449676266, 0.00022722265144433808]],
     [[0.001624242152888081, 0.001021942449047799], [-0.0011219081065389103, 0.0006523458305635018]],
     [[0.0013264597869771804, 0.0005174761288357546], [-0.0005212118783658866, 0.0010132293824609802]],
     [[0.0013492189942925154, -0.000192073875027788], [1.4237017069651186e-05, 0.0012086499590651793]],
     [[0.0009909357975836257, -0.0005332406949934581], [0.00042307402351765555, 0.0009771426260838099]],
     [[0.000643101226689506, -0.0005553599790969417], [0.0006929626647005008, 0.0008231842460808193]],
     [[-1.0662035861166535e-05, -0.0006297788738246896], [0.00066564486982282, 0.0006112383690790388]],
     [[-8.74821614196258e-05, -0.000716593450257982], [0.0007616126615676449, 0.0004374551791716873]],
     [[-0.00033061558141287046, -0.0007187465600888608],
      [0.0005826897811688773, -6.257120921320456e-05]],
     [[-0.0005061538341057612, -0.0004606460547027496], [0.00053767976402595, -5.492366477692096e-05]],
     [[-0.0005010055318834733, -0.00025829400952672365],
      [0.0002890113954020254, -0.00020581486787535043]],
     [[-0.00047366194329932873, -2.8981293198352755e-05],
      [0.00014968437858105997, -0.00042411977425599117]],
     [[-0.00035024813171045244, 1.6434569195682753e-05],
      [-5.501974776150669e-05, -0.00039362309952963053]],
     [[-0.00013892955296638274, 4.686758526523353e-05],
      [-0.00020568669053105287, -0.00028285054789021217]],
     [[-0.0002003103943955653, 0.00039531752712794065],
      [-7.316636950648031e-05, -0.0002731729693580255]],
     [[2.2286696831127793e-05, 0.00015437860236684287],
      [-0.0001544354845687408, -9.563983878576827e-05]],
     [[0.0001440272157608188, 0.0001939925799713992],
      [-0.00019804613463913988, -3.690729057163481e-05]],
     [[0.00010252431489315893, 0.0001400003979238398],
      [-0.0001385116171799568, 2.4140381372999342e-05]],
     [[0.0001363407089332727, 0.00021272843964391242], [-6.406958715635323e-05, 7.958828852769995e-05]],
     [[0.00011630478588597312, -6.936128147111542e-05],
      [-1.8557028207100188e-05, 0.00025453608482457274]],
     [[6.22604773256754e-05, -4.1767845117361375e-05], [0.00010027246529721863, 0.0001165975462666937]],
     [[0.0001605395181351869, -5.408087416581433e-05], [0.00010328705484075705, 8.427168474534932e-05]],
     [[0.00010008097506184958, -0.00010731245895343017],
      [9.447450128903084e-05, 0.00014116820291291575]],
     [[-4.9150276985951924e-05, -0.00010609248660087736],
      [5.044545510272651e-05, 1.412408325984388e-05]],
     [[4.830971150547375e-05, -5.2118054146629344e-05],
      [0.00014691153985706433, 8.147231055826554e-05]],
     [[3.5216990260324576e-05, -5.740398182903633e-05],
      [2.992906091353924e-05, -0.0001678243295094388]],
     [[-2.142839148494098e-05, -3.844705509572901e-05],
      [-0.00011402473901201629, -0.0001409368712290601]],
     [[2.5578362040290127e-05, -8.660659896232096e-05],
      [0.00014428477844329399, 2.9892099298629904e-05]],
     [[-4.518713570257886e-05, 8.658337816386911e-05],
      [-1.7130929539254325e-06, -3.55135403653932e-05]],
     [[-1.5743338701147348e-05, 1.715407116781407e-05],
      [-2.2317017717851454e-05, -7.321969621187974e-05]],
     [[-6.24064550504235e-05, 3.0034007594653694e-05],
      [1.0421647162102535e-05, -7.322195707014262e-05]],
     [[-3.0135586111838027e-05, 2.333347818979117e-05],
      [-8.960611284407689e-05, -0.0001427388512389588]],
     [[3.763187750506885e-05, -3.089800736402216e-05],
      [-6.514831619996491e-05, -3.288379083309858e-05]],
     [[-1.2309791989966141e-05, -5.9158697383808875e-06],
      [-0.0001051774794279742, 9.423545818588685e-05]],
     [[-7.12066443649222e-05, -6.392464320962927e-05],
      [-9.312783442542194e-05, -7.521094145958501e-05]],
     [[1.4747072008726796e-05, 4.2474303883829376e-05], [6.40441282655833e-06, 1.981232320425809e-05]],
     [[2.505086805117769e-05, -1.0083372167592046e-05],
      [-2.137116682327815e-05, 3.157126405865931e-05]],
     [[-1.4363204030243868e-05, -1.4181274490354373e-05],
      [-9.024570388196097e-06, 8.234475658919713e-05]],
     [[4.358272132169892e-06, 5.972696569630652e-05], [1.880384036766271e-05, -4.7629632721311394e-05]],
     [[9.021311663896799e-05, 4.10473002886764e-05], [-6.411202634197987e-06, -8.595663954717196e-06]],
     [[-2.6222302165776125e-05, 1.189316154983247e-05],
      [-5.186911866133515e-05, 5.8073649290162005e-05]],
     [[-3.754797458576825e-05, -4.337158583213995e-05],
      [-3.0575129026285615e-07, 1.5043052756945144e-05]],
     [[-5.230600888614886e-07, -1.962725504434252e-05],
      [-7.295639387778146e-05, -2.446771720031154e-05]],
     [[-4.140288344963852e-05, 8.139750654482538e-05],
      [-3.601760617374465e-05, 1.5595674761334436e-05]],
     [[-3.571381856937219e-05, -5.506392012621399e-05],
      [-2.068400262475203e-05, 2.6654262625442626e-05]],
     [[8.486360858553457e-06, 2.2695217986930107e-05],
      [2.2944169412917867e-05, -8.503129724927262e-06]],
     [[8.910145697119443e-06, -2.7502930323898968e-05],
      [-2.6268555576725443e-05, -1.764460509942499e-05]],
     [[3.6714412863273056e-06, 4.358859443419548e-06],
      [-6.653291827063059e-06, 0.00010942534406397968]],
     [[-4.218862305083614e-06, 4.6782236827428836e-05],
      [-1.3941027638269417e-05, 4.7602860091795876e-05]],
     [[3.857757687181387e-05, -5.942653235718807e-06], [-1.19333109701094e-05, -7.395522538611696e-06]],
     [[3.8516849227630255e-07, -1.469311634794489e-05],
      [1.1469265810564096e-05, 6.542695560985904e-06]],
     [[8.785228140505579e-05, 6.138228884729674e-05],
      [-3.1628770201456747e-05, -5.917858506993576e-06]],
     [[7.471261312927948e-05, -2.192725750417227e-05],
      [-1.5544526056373188e-05, 3.489277101879881e-05]],
     [[3.79386352000514e-05, 2.7010485591287206e-05],
      [-2.2889964009117745e-05, -3.6083213836026665e-05]],
     [[8.156261071118911e-05, -1.6272000363119138e-05], [4.627895823582638e-05, 7.847118446969751e-05]],
     [[2.2860974538761207e-05, -1.4141919688006165e-05],
      [-4.001706628272636e-05, -7.302413299342011e-05]],
     [[1.8229104686196623e-05, 2.9479897141023017e-05],
      [1.2212558950728554e-05, 3.9128968995768535e-05]],
     [[-2.251559122461778e-06, 1.1918302253010489e-05],
      [3.4608254906484577e-05, -1.4480563031320118e-05]],
     [[2.222461157843098e-05, 4.602814137559041e-05],
      [-1.4863227706421655e-05, -3.464693079977688e-05]],
     [[1.1408027860383027e-05, -4.185027164871985e-05], [4.304678293009884e-05, 5.964120822948625e-06]],
     [[6.12494353188849e-05, -2.6007808053192416e-05],
      [-5.845231823022385e-05, -1.9566610694596033e-05]],
     [[3.2566842710366366e-05, 1.6777430826097137e-05],
      [-1.372275924821406e-05, -2.188898074738862e-05]],
     [[-5.637250310627042e-05, -2.62470658955132e-05],
      [3.071051220982115e-05, -1.3089120040882971e-05]],
     [[3.642534188890838e-05, 1.7572595380545352e-05], [3.9120027499985e-05, -1.4423170846364087e-05]],
     [[-1.4299087744479837e-05, 1.8535549680679776e-05],
      [-2.3288970298525078e-05, 9.515129395076562e-06]],
     [[-1.1357724123202616e-05, 5.88640823503951e-05], [1.0164899664124045e-05, 6.402909322379169e-06]],
     [[-1.3551733885690897e-05, -3.578453849905521e-05],
      [-3.1283674702305404e-05, -3.9374639050833186e-05]],
     [[6.417052270730976e-05, 2.1310466076685957e-05],
      [1.5464319676770254e-05, 1.3563275313497464e-05]],
     [[-2.201953653610577e-05, 1.178232746244161e-05], [2.1098045659187664e-05, -2.83176147528608e-05]],
     [[3.962249887041102e-05, 5.396108986690356e-05],
      [-2.2024136435024297e-05, -1.720105690887811e-06]],
     [[4.155651892090912e-06, -7.5818866508571404e-06],
      [-2.7133422342654934e-05, 8.10468392302828e-06]],
     [[1.1676037453984336e-05, -9.138721886691081e-06],
      [-2.7882086532147694e-05, 5.513964350417517e-06]],
     [[3.2002852338094223e-06, -1.7140538399323085e-05],
      [-8.094542689575464e-06, 1.340120869283062e-06]],
     [[-2.980684627323806e-07, 1.1716419130637584e-05],
      [-1.6206632817093632e-05, -4.731342504399433e-05]],
     [[-2.3158355626351732e-05, 2.710589318325983e-05], [5.264225007809326e-05, 4.124338615847594e-05]],
     [[1.0290761526286932e-05, 4.1515553887657316e-05],
      [-6.294504971748244e-05, 4.3643154408683365e-05]],
     [[-2.1409151236450712e-05, -1.0235777121399537e-05],
      [2.7635894154029304e-05, 1.4742028843041657e-05]],
     [[2.0741323336249596e-06, 1.5700713707866398e-06],
      [2.1680618667031612e-05, 2.994048005784835e-05]],
     [[1.6294243331539282e-06, -2.0826359928999633e-05],
      [2.7100375812963935e-06, -1.4626823832537979e-05]],
     [[-1.3361057533918406e-06, -1.5739266376055273e-05],
      [-1.4299336095268268e-05, 1.2189101049520563e-05]],
     [[7.25167944028748e-06, 2.6829799182446905e-05],
      [-2.7048571531943143e-05, 2.6521897562414144e-06]],
     [[-3.17044162624041e-06, 2.225123591869783e-05], [4.001806111426734e-06, 1.907243215885327e-05]],
     [[2.8881753603962484e-05, -8.298154271469826e-07],
      [-4.8265089210566464e-05, -2.823407490766972e-05]],
     [[2.0379687915479157e-05, 3.6854661440476254e-06],
      [-4.50252676606985e-06, 4.5976385947080476e-05]],
     [[-3.5366085249669806e-06, -3.539479228542985e-05],
      [2.052162066992743e-05, 3.4151067360658445e-05]],
     [[2.731714049776435e-06, -1.826350473882273e-05], [3.396541345137203e-05, 3.087081083574963e-05]],
     [[1.0614363222310238e-05, -3.155146282661812e-06],
      [-1.630762736184292e-05, -8.542453939331966e-06]],
     [[-2.2440704597324105e-05, -1.530636449161752e-05],
      [-6.772118654896633e-06, 8.10273719522035e-06]],
     [[1.19228659488092e-06, -2.027235404168733e-05], [-3.565112117731127e-06, 2.843439320956975e-06]],
     [[-1.2229911092714656e-05, -6.764466860102689e-08],
      [2.0824019026878105e-05, -2.0378801397439018e-05]],
     [[2.601326098229826e-05, -1.6327422493467892e-05],
      [-4.09096423632767e-06, -1.2796899034336425e-05]],
     [[1.5045833127684287e-05, 1.6666491039754603e-05],
      [-2.680538849628366e-07, 5.4337728578602895e-06]],
     [[3.0219458135166468e-05, -3.51967441406466e-05], [3.441366515601803e-05, 1.5197801936123761e-06]],
     [[4.495594750668476e-06, 1.890943851237289e-06], [6.626018485508577e-06, -5.510260973659996e-06]],
     [[-1.2083128833352775e-05, 3.8258955624314664e-05],
      [-1.103947572401589e-05, -6.207780531760683e-06]],
     [[1.2839974366087394e-05, 2.170310522139568e-06],
      [-1.4386189273414595e-05, -1.6968770273904225e-05]]], dtype=float),
    'tau': 0.4,
    'mass': 5.0,
    'thermal_energy': 1.0,
    'observations': np.array([[0.5164846654625908, 0.20130516002576324], [0.6337259009390972, 0.4720131379597723],
     [0.7452604227057262, 0.5077125414823482], [0.8147575451481152, 0.6061955494637903],
     [0.5627002676258996, 0.596849741670902], [0.2918826919485984, 0.6393957137990919],
     [0.14818510601615853, 0.6250724811303531], [-0.021659149255744966, 0.6248065703643814],
     [-0.3003608695936576, 0.6863868156319217], [-0.48138316031851064, 0.5710454338354477],
     [-0.674287706174415, 0.6204914835116102], [-0.6499272013346309, 0.4377196418488011],
     [-0.6509116948331924, 0.14082401859968596], [-0.723928169933927, -0.06439161054128857],
     [-0.6057395715165842, -0.061074791738953534], [-0.40782199086916904, -0.15086016695850402],
     [-0.35625064929434114, -0.149774759721444], [-0.3741949223688179, -0.18022312151511505],
     [-0.2601212184770717, -0.3610384397980052], [0.011395460334694988, -0.6431715418321573],
     [0.17692853767991987, -0.6953602419050365], [0.2337240870973197, -0.6835101741386617],
     [0.3397446404640645, -0.7515667486087267], [0.43028097402357973, -0.6803542182364358]], dtype=float),
    'intervals': np.array([0.22, 0.26, 0.3, 0.33999999999999997, 0.38, 0.22, 0.26, 0.3, 0.33999999999999997, 0.38, 0.22, 0.26,
     0.3, 0.33999999999999997, 0.38, 0.22, 0.26, 0.3, 0.33999999999999997, 0.38, 0.22, 0.26, 0.3], dtype=float),
    'observation_covariance': np.array([[0.008, 0.0024], [0.0024, 0.0136]], dtype=float),
    'rho': 1.4,
    'grid_size': 64,
    'fit_tolerance': 0.001,
    'max_support': 18,
    'rank_tolerance': 1e-07,
}
""",
            'call': 'molecular_memory_score(**copy.deepcopy({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}))',
            'gold_call': '_oracle_molecular_memory_score(**copy.deepcopy({k: v.copy() if isinstance(v, np.ndarray) else v for k, v in data.items()}))',
            'tol': 2e-06,
        },
    ]

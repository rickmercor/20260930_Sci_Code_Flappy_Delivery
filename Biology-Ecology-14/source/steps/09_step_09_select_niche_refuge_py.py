"""
Select the refuge plan that maximizes robust reserve protection per unit effort. Returns
-------
return score

The selected plan must remain feasible, globally stable and above every reserve density under the same correlated disturbance model.

Returns
-------
return score
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_niche_refuge(resource_matrix: "numpy.ndarray", k: float,
                        base_competition: "numpy.ndarray", base_designs: "numpy.ndarray",
                        loadings: "numpy.ndarray", reserves: "numpy.ndarray",
                        covariance: "numpy.ndarray", upper_efforts: "numpy.ndarray",
                        coupling: "numpy.ndarray", breadth_response: "numpy.ndarray",
                        buffer: float, minimum_effort: float) -> float:
    """Return the largest eligible robust climate radius per unit effort.

    Construct the noncircular niche geometry, condition the competition and
    regulation panels, enforce both coexistence certificates with the buffer,
    and optimize each plan on its full closed effort interval. A single
    correlated ellipsoidal climate disturbance acts in all factors. Plans
    with an empty interval or negative best reserve radius are unavailable.
    Values within 1e-10 tie in favor of the earliest plan; return zero when no
    plan is available.

    Raises
    ------
    ValueError
        If the ecological arrays are nonfinite or dimensionally misaligned,
        plan slopes or effort bounds are not strictly positive, reserves are
        negative, buffer is not greater than one, minimum_effort is not
        positive, covariance is not symmetric positive definite, or the
        required population-response system is singular or has zero climate
        response.
    """
    return score

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from numpy.polynomial import Chebyshev as _Chebyshev


def _canonical_refuge(interactions):
    mean = interactions.mean(axis=0)
    return np.vstack([interactions-mean[None, :], mean])


def _response_refuge(packed, regulation, loadings, effort):
    centered, mean = packed[:-1], packed[-1]
    try:
        solved = np.linalg.solve(effort*np.eye(len(regulation))+centered,
                                 np.column_stack([np.ones(len(regulation)), loadings]))
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular canonical response system") from exc
    denominator = 1.0+mean@solved[:, 0]
    if abs(denominator) <= 1e-13:
        raise ValueError("singular full response system")
    corrected = solved-np.outer(solved[:, 0], mean@solved)/denominator
    return corrected/regulation[:, None]


def _radii_refuge(response, reserves, cholesky):
    lengths = np.linalg.norm(response[:, 1:]@cholesky, axis=1)
    if np.any(lengths == 0.0):
        raise ValueError("every population needs a nonzero climate response")
    return (response[:, 0]-reserves)/lengths


def _optimize_refuge_exact(packed_panel, regulation, loadings, reserves,
                           cholesky, lower, upper):
    matrices = packed_panel[:, :-1]+packed_panel[:, -1, None, :]
    transformed_loadings = loadings@cholesky

    def _values(effort):
        return np.concatenate([
            _radii_refuge(_response_refuge(packed, regulation, forcing, effort),
                          reserves, cholesky)
            for packed, forcing in zip(packed_panel, loadings)
        ])

    if lower == upper:
        radius = float(_values(lower).min())
        return np.array([lower, radius, radius/lower], dtype=np.float64)
    n = len(regulation)
    nodes = np.cos((np.arange(n+1)+0.5)*np.pi/(n+1))
    efforts = (upper+lower)/2.0+(upper-lower)/2.0*nodes
    numerators, variances = [], []
    for packed, interactions, forcing in zip(packed_panel, matrices, transformed_loadings):
        samples = []
        for effort in efforts:
            response = _response_refuge(packed, regulation, forcing, effort)
            response[:, 0] -= reserves
            samples.append(response*np.linalg.det(effort*np.eye(n)+interactions))
        samples = np.asarray(samples)
        samples /= max(1e-100, np.max(np.abs(samples)))
        for i in range(n):
            numerators.append(_Chebyshev.fit(nodes, samples[:, i, 0], n, domain=[-1, 1]))
            terms = [_Chebyshev.fit(nodes, samples[:, i, j], n-1, domain=[-1, 1])
                     for j in range(1, forcing.shape[1]+1)]
            variances.append(sum((term*term for term in terms), _Chebyshev([0.0]))
                             *_Chebyshev([(lower+upper)/2.0, (upper-lower)/2.0])**2)
    candidates = [-1.0, 1.0]

    def _add_roots(polynomial):
        coefficients = polynomial.coef.copy()
        while (len(coefficients) > 1
               and abs(coefficients[-1]) < 1e-12*max(1e-100, np.max(np.abs(coefficients)))):
            coefficients = coefficients[:-1]
        roots = _Chebyshev(coefficients).roots()
        candidates.extend(roots.real[(np.abs(roots.imag) < 1e-7)
                                     & (np.abs(roots.real) < 1.0)])

    for numerator, variance in zip(numerators, variances):
        _add_roots(2.0*numerator.deriv()*variance-numerator*variance.deriv())
    for i in range(len(numerators)):
        for j in range(i):
            _add_roots(numerators[i]**2*variances[j]-numerators[j]**2*variances[i])
    candidates = np.asarray(candidates, dtype=np.float64)
    candidate_efforts = np.sort((upper+lower)/2.0+(upper-lower)/2.0*candidates)
    scores = np.asarray([_values(effort).min()/effort for effort in candidate_efforts])
    best = scores.max()
    chosen = np.flatnonzero(scores >= best-1e-12)[0]
    effort = float(candidate_efforts[chosen])
    radius = float(_values(effort).min())
    return np.array([effort, radius, radius/effort], dtype=np.float64)


def _oracle_select_niche_refuge(resource_matrix: "numpy.ndarray", k: float,
                                base_competition: "numpy.ndarray", base_designs: "numpy.ndarray",
                                loadings: "numpy.ndarray", reserves: "numpy.ndarray",
                                covariance: "numpy.ndarray", upper_efforts: "numpy.ndarray",
                                coupling: "numpy.ndarray", breadth_response: "numpy.ndarray",
                                buffer: float, minimum_effort: float) -> float:
    N = np.asarray(resource_matrix, dtype=np.float64)
    base_B = np.asarray(base_competition, dtype=np.float64)
    base_D = np.asarray(base_designs, dtype=np.float64)
    U = np.asarray(loadings, dtype=np.float64)
    ell = np.asarray(reserves, dtype=np.float64)
    Sigma = np.asarray(covariance, dtype=np.float64)
    upper = np.asarray(upper_efforts, dtype=np.float64)
    coupling = np.asarray(coupling, dtype=np.float64)
    response = np.asarray(breadth_response, dtype=np.float64)
    if (base_B.ndim != 3 or base_B.shape[0] < 1 or base_B.shape[1] != base_B.shape[2]
            or base_D.ndim != 2 or base_D.shape[1] != base_B.shape[1] or base_D.shape[0] < 1
            or U.ndim != 3 or U.shape[:2] != base_B.shape[:2] or U.shape[2] < 1
            or ell.shape != (base_B.shape[1],) or Sigma.shape != (U.shape[2], U.shape[2])
            or upper.shape != (base_D.shape[0],) or coupling.shape != (base_B.shape[0], 2)
            or response.shape != (base_D.shape[0],)
            or any(not np.all(np.isfinite(x)) for x in
                   (N, base_B, base_D, U, ell, Sigma, upper, coupling, response))
            or np.any(base_D <= 0.0) or np.any(ell < 0.0) or np.any(upper <= 0.0)
            or not np.isfinite(buffer) or buffer <= 1.0
            or not np.isfinite(minimum_effort) or minimum_effort <= 0.0):
        raise ValueError("finite aligned ecological inputs and valid policy constants are required")
    try:
        cholesky = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be symmetric positive definite") from exc
    if np.max(np.abs(Sigma-Sigma.T)) > 1e-10:
        raise ValueError("covariance must be symmetric positive definite")

    geometry = _oracle_noncircular_niche_geometry(N, k)
    competition = _oracle_effective_competition(base_B, geometry, coupling)
    regulation = _oracle_effective_regulation(base_D, geometry, response)
    plan_scores = []
    for plan, (slopes, high) in enumerate(zip(regulation, upper)):
        certificates = _oracle_coexistence_certificates(competition, slopes)
        low = max(float(minimum_effort), float(buffer)*float(certificates[:, :2].max()))
        if low > high:
            plan_scores.append((plan, -np.inf))
            continue
        packed = np.asarray([_canonical_refuge(condition/slopes[None, :])
                             for condition in competition])
        optimum = _optimize_refuge_exact(packed, slopes, U, ell, cholesky, low, float(high))
        plan_scores.append((plan, float(optimum[2]) if optimum[1] >= 0.0 else -np.inf))
    finite = [item for item in plan_scores if np.isfinite(item[1])]
    if not finite:
        return 0.0
    best = max(score for _, score in finite)
    return float(next(score for _, score in finite if score >= best-1e-10))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = """import numpy as np
N=np.array([[7,39,11,9,0,6,0],[39,0,25,0,31,4,0],[0,0,0,21,37,10,0],[0,0,0,0,14,0,0],[0,0,0,37,0,0,0],[0,38,0,0,0,0,0]],float)
B=np.array([[[0,.515,.858,1.099,.88,.244],[.218,0,.792,.794,.891,.134],[1.02,.121,0,.869,.803,.199],[.396,.345,.452,0,.859,.543],[.876,.526,.397,.614,0,.387],[.398,.691,.511,.179,.331,0]],[[0,.685,1.068,1.013,.914,.3],[.205,0,.669,.87,.757,.15],[1.302,.093,0,.783,.697,.25],[.292,.416,.471,0,.952,.72],[.828,.54,.443,.606,0,.406],[.372,.636,.487,.133,.421,0]],[[0,.628,1.112,1.418,1.106,.223],[.226,0,1.133,.992,1.014,.177],[1.352,.158,0,1.025,.93,.232],[.337,.409,.347,0,1.136,.366],[1.115,.593,.438,.666,0,.469],[.465,.774,.403,.219,.304,0]]],float)
D=np.array([[.614,1.117,.817,1.164,1.254,.973],[1.6049,.9328,1.1209,.9394,1.4058,.7073],[.595,1.436,.855,.827,.9,.781],[.922,1.044,.85,1.174,1.472,.922],[1.222,1.284,.928,.63,1.255,.553],[1.044,1.404,.966,.579,.665,.716]],float)
U=np.array([[[.306,.292],[-.313,.23],[-.395,.32],[.382,.177],[.361,.427],[-.416,-.493]],[[-.198,.303],[-.129,.568],[-.205,-.096],[.252,.326],[-.297,.106],[-.056,-.095]],[[-.008,.031],[-.332,.017],[.395,.785],[.069,.69],[.083,-.445],[-.045,-.223]]],float)
ell=np.array([.032,.045,.036,.027,.041,.034])
Sigma=np.array([[1.,-.35777087639996635],[-.35777087639996635,.8]])
upper=np.array([8.,7.6,8.4,7.9,8.2,7.7])
coupling=np.array([[.8,-.5],[-.35,.7],[1.1,.45]])
q=np.array([-1.1,.8,-.6,1.25,-.9,.55])
args=(N,10000.,B,D,U,ell,Sigma,upper,coupling,q,1.8,.2)
"""


    return [
        {
            "setup": setup,
            "call": "select_niche_refuge(*args)",
            "gold_call": "_oracle_select_niche_refuge(*args)",
            "tol": 2e-6,
        },
        {
            "setup": setup+"\nargs=list(args);args[7]=np.array([2.,2.,2.,2.,2.,2.]);args=tuple(args)",
            "call": "select_niche_refuge(*args)",
            "gold_call": "_oracle_select_niche_refuge(*args)",
            "tol": 2e-6,
        },
        {
            "setup": setup+"\nargs=list(args);args[5]=np.ones(6)*10.;args=tuple(args)",
            "call": "select_niche_refuge(*args)",
            "gold_call": "_oracle_select_niche_refuge(*args)",
            "tol": 2e-6,
        },
        {
            "setup": setup+"\nargs=list(args);args[6]=np.array([[1.,2.],[2.,1.]]);args=tuple(args)\ndef run(fn):\n    try: fn(*args)\n    except ValueError: return 1.0\n    return 0.0",
            "call": "run(select_niche_refuge)",
            "gold_call": "run(_oracle_select_niche_refuge)",
            "tol": 0.0,
        },
    ]

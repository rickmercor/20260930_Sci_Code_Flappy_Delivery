"""
Select the refuge plan that survives the resource-state stress audit.

The nominal winner is not accepted unless the same plan also wins enough
attenuation scenarios after the complete ecological chain is recomputed.

Returns
-------
return score

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def select_stress_tested_refuge(resource_matrix: "numpy.ndarray", k: float,
                                base_competition: "numpy.ndarray", base_designs: "numpy.ndarray",
                                loadings: "numpy.ndarray", reserves: "numpy.ndarray",
                                covariance: "numpy.ndarray", upper_efforts: "numpy.ndarray",
                                coupling: "numpy.ndarray", breadth_response: "numpy.ndarray",
                                buffer: float, minimum_effort: float,
                                attenuation: float, minimum_wins: int,
                                regret_weight: float) -> float:
    """Return the largest eligible stress-adjusted refuge score.

    First compute every plan's nominal optimized reserve-radius-per-effort
    score. Then form one stress scenario for each occupied resource state by
    multiplying that complete resource column by attenuation and recomputing
    the entire niche, competition, regulation, certificate and optimization
    chain. A plan wins a stress scenario when its score is within 1e-10 of
    that scenario's largest finite score. Its scenario regret is the largest
    finite score minus its own score. The tail regret is the mean of its two
    largest stress regrets. A plan is eligible only when it wins at least
    minimum_wins stress scenarios. Its decision score is its nominal score
    minus regret_weight times its tail regret. Choose the largest finite
    decision score, with values within 1e-10 tied in favour of the earliest
    plan, and return zero when no plan is eligible.

    Raises
    ------
    ValueError
        If attenuation is not strictly between zero and one, minimum_wins is
        not an integer from one through the number of occupied states,
        regret_weight is negative or nonfinite, or any earlier ecological
        contract is violated.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _stress_plan_panel(resource_matrix, k, base_competition, base_designs,
                       loadings, reserves, covariance, upper_efforts,
                       coupling, breadth_response, buffer, minimum_effort):
    geometry = _oracle_noncircular_niche_geometry(resource_matrix, k)
    competition = _oracle_effective_competition(base_competition, geometry, coupling)
    regulation = _oracle_effective_regulation(base_designs, geometry, breadth_response)
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance must be symmetric positive definite") from exc
    rows = []
    for slopes, high in zip(regulation, upper_efforts):
        certificates = _oracle_coexistence_certificates(competition, slopes)
        low = max(float(minimum_effort), float(buffer)*float(certificates[:, :2].max()))
        if low > high:
            rows.append([np.nan, np.nan, -np.inf])
            continue
        packed = np.asarray([_canonical_refuge(condition/slopes[None, :])
                             for condition in competition])
        optimum = _optimize_refuge_exact(packed, slopes, loadings, reserves,
                                          cholesky, low, float(high))
        if optimum[1] < 0.0:
            optimum[2] = -np.inf
        rows.append(optimum)
    return np.asarray(rows, dtype=np.float64)


def _oracle_select_stress_tested_refuge(resource_matrix: "numpy.ndarray", k: float,
                                        base_competition: "numpy.ndarray", base_designs: "numpy.ndarray",
                                        loadings: "numpy.ndarray", reserves: "numpy.ndarray",
                                        covariance: "numpy.ndarray", upper_efforts: "numpy.ndarray",
                                        coupling: "numpy.ndarray", breadth_response: "numpy.ndarray",
                                        buffer: float, minimum_effort: float,
                                        attenuation: float, minimum_wins: int,
                                        regret_weight: float) -> float:
    N = np.asarray(resource_matrix, dtype=np.float64)
    if (N.ndim != 2 or N.shape[0] < 3 or N.shape[1] < 2
            or not np.all(np.isfinite(N)) or np.any(N < 0.0)):
        raise ValueError("resource_matrix must be a finite nonnegative matrix")
    occupied = np.flatnonzero(N.sum(axis=0) > 0.0)
    if (not np.isfinite(attenuation) or not 0.0 < attenuation < 1.0
            or not isinstance(minimum_wins, (int, np.integer))
            or minimum_wins < 1 or minimum_wins > len(occupied)
            or not np.isfinite(regret_weight) or regret_weight < 0.0):
        raise ValueError("invalid stress-audit controls")

    common = (k, np.asarray(base_competition, dtype=np.float64),
              np.asarray(base_designs, dtype=np.float64),
              np.asarray(loadings, dtype=np.float64),
              np.asarray(reserves, dtype=np.float64),
              np.asarray(covariance, dtype=np.float64),
              np.asarray(upper_efforts, dtype=np.float64),
              np.asarray(coupling, dtype=np.float64),
              np.asarray(breadth_response, dtype=np.float64),
              buffer, minimum_effort)
    nominal = _stress_plan_panel(N, *common)
    stress = []
    for state in occupied:
        changed = N.copy()
        changed[:, state] *= attenuation
        stress.append(_stress_plan_panel(changed, *common))
    stress = np.asarray(stress, dtype=np.float64)
    scores = stress[:, :, 2]
    if np.any(~np.any(np.isfinite(scores), axis=1)):
        return 0.0
    best = np.max(np.where(np.isfinite(scores), scores, -np.inf), axis=1)
    decisions = np.full(scores.shape[1], -np.inf, dtype=np.float64)
    for plan in range(scores.shape[1]):
        if not np.isfinite(nominal[plan, 2]) or np.any(~np.isfinite(scores[:, plan])):
            continue
        regrets = best-scores[:, plan]
        wins = int(np.count_nonzero(regrets <= 1e-10))
        if wins < minimum_wins:
            continue
        tail = float(np.sort(regrets)[-2:].mean()) if len(regrets) > 1 else float(regrets[0])
        decisions[plan] = nominal[plan, 2]-regret_weight*tail
    if not np.any(np.isfinite(decisions)):
        return 0.0
    best_decision = float(np.max(decisions))
    chosen = int(np.flatnonzero(decisions >= best_decision-1e-10)[0])
    return float(decisions[chosen])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = """import numpy as np
N=np.array([[7,39,11,9,0,6,0],[39,0,25,0,31,4,0],[0,0,0,21,37,10,0],[0,0,0,0,14,0,0],[0,0,0,37,0,0,0],[0,38,0,0,0,0,0]],float)
B=np.array([[[0,.515,.858,1.099,.88,.244],[.218,0,.792,.794,.891,.134],[1.02,.121,0,.869,.803,.199],[.396,.345,.452,0,.859,.543],[.876,.526,.397,.614,0,.387],[.398,.691,.511,.179,.331,0]],[[0,.685,1.068,1.013,.914,.3],[.205,0,.669,.87,.757,.15],[1.302,.093,0,.783,.697,.25],[.292,.416,.471,0,.952,.72],[.828,.54,.443,.606,0,.406],[.372,.636,.487,.133,.421,0]],[[0,.628,1.112,1.418,1.106,.223],[.226,0,1.133,.992,1.014,.177],[1.352,.158,0,1.025,.93,.232],[.337,.409,.347,0,1.136,.366],[1.115,.593,.438,.666,0,.469],[.465,.774,.403,.219,.304,0]]],float)
D=np.array([[.614,1.117,.817,1.164,1.254,.973],[1.6049,.9328,1.1209,.9394,1.4058,.7073],[.595,1.436,.855,.827,.9,.781],[.922,1.044,.85,1.174,1.472,.922],[1.222,1.284,.928,.63,1.255,.553],[1.044,1.404,.966,.579,.665,.716]],float)
U=np.array([[[.306,.292],[-.313,.23],[-.395,.32],[.382,.177],[.361,.427],[-.416,-.493]],[[-.198,.303],[-.129,.568],[-.205,-.096],[.252,.326],[-.297,.106],[-.056,-.095]],[[-.008,.031],[-.332,.017],[.395,.785],[.069,.69],[.083,-.445],[-.045,-.223]]],float)
ell=np.array([.032,.045,.036,.027,.041,.034]);Sigma=np.array([[1.,-.35777087639996635],[-.35777087639996635,.8]])
upper=np.array([8.,7.6,8.4,7.9,8.2,7.7]);coupling=np.array([[.8,-.5],[-.35,.7],[1.1,.45]]);q=np.array([-1.1,.8,-.6,1.25,-.9,.55])
args=(N,10000.,B,D,U,ell,Sigma,upper,coupling,q,1.8,.2,.35,4,.4)
"""
    return [
        {
            "setup": setup,
            "call": "select_stress_tested_refuge(*args)",
            "gold_call": "_oracle_select_stress_tested_refuge(*args)",
            "tol": 2e-9,
        },
        {
            "setup": setup+"\nargs=list(args);args[12]=.5;args[13]=2;args=tuple(args)",
            "call": "select_stress_tested_refuge(*args)",
            "gold_call": "_oracle_select_stress_tested_refuge(*args)",
            "tol": 2e-9,
        },
        {
            "setup": setup+"\nargs=list(args);args[13]=6;args=tuple(args)",
            "call": "select_stress_tested_refuge(*args)",
            "gold_call": "_oracle_select_stress_tested_refuge(*args)",
            "tol": 0.0,
        },
        {
            "setup": setup+"\nargs=list(args);args[7]=np.ones(6)*2.;args=tuple(args)",
            "call": "select_stress_tested_refuge(*args)",
            "gold_call": "_oracle_select_stress_tested_refuge(*args)",
            "tol": 0.0,
        },
        {
            "setup": setup+"\nargs=list(args);args[12]=1.0;args=tuple(args)\ndef run(fn):\n    try: fn(*args)\n    except ValueError: return 1.0\n    return 0.0",
            "call": "run(select_stress_tested_refuge)",
            "gold_call": "run(_oracle_select_stress_tested_refuge)",
            "tol": 0.0,
        },
    ]

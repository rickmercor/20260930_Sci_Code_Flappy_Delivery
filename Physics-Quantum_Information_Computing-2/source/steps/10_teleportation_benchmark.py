"""
Compose the distribution-calibrated robust teleportation benchmark.

Final orchestrator. The submitted implementation must call and combine all nine preceding public functions, directly or transitively, using their returned values. Form the actual damping probabilities from the physical exposures before applying the orientation. Optimize both record assessments, each against its own classical baseline, and return recorded optimum minus erased optimum.

Returns
-------
return result
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def teleportation_benchmark(observations: np.ndarray, thresholds: np.ndarray, profiles: np.ndarray, scenarios: np.ndarray, priors: np.ndarray, budget: float, target: float, tail_min: float) -> float:
    """Compose the distribution-calibrated robust teleportation benchmark.

    observations, thresholds : ndarray
        Calibration arrays of shapes (2+T,) and (T,), with the infer_damping contract.
    profiles : ndarray, shape (C,4)
        Rows [orientation,e_A,e_B,cost]; orientation in {0,1}, positive exposures,
        nonnegative cost; C>0. Orientation 1 exchanges the resulting dampings.
    scenarios : ndarray, shape (S,2)
        Positive physical exposure multipliers [r_A,r_B]; S>0.
    priors : ndarray, shape (K,2)
        Beta importance [alpha,beta] rows, shapes >=1; K>0.
    budget, target, tail_min : float
        Nonnegative cost ceiling, finite fidelity threshold, and reliability
        floor in [0,1]. The event is F>target. Use the exposure formula and
        common-policy objective in the global background.
    Returns
    -------
    float
        Signed difference J_R-J_E of separately optimized worst-case advantages,
        each subject to its own cost and per-scenario reliability constraints.
        Compose the full public pipeline, including infer_damping and
        optimize_policy. An infeasible design raises ValueError.

    All quantities are dimensionless. Inputs are preserved.
    Raises
    ------
    ValueError
        For invalid input shapes or data outside the stated domain.
    """
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
from scipy.integrate import quad
from scipy.special import betaln
from scipy.optimize import least_squares, linprog

def _oracle_teleportation_benchmark(observations: np.ndarray, thresholds: np.ndarray, profiles: np.ndarray, scenarios: np.ndarray, priors: np.ndarray, budget: float, target: float, tail_min: float) -> float:
    profiles = np.asarray(profiles, dtype=float)
    scenarios = np.asarray(scenarios, dtype=float)
    priors = np.asarray(priors, dtype=float)
    if profiles.ndim != 2 or profiles.shape[1] != 4 or scenarios.ndim != 2 or (scenarios.shape[1] != 2) or (not len(profiles)) or (not len(scenarios)):
        raise ValueError('Expected nonempty profile (C,4) and scenario (S,2) arrays.')
    if not np.isfinite(profiles).all() or not np.isfinite(scenarios).all() or np.any(profiles[:, 1:3] <= 0) or np.any(scenarios <= 0) or np.any(~np.isin(profiles[:, 0], [0, 1])) or (not np.isfinite(target)):
        raise ValueError('Invalid orientations, exposure multipliers, or target.')
    pair = _oracle_infer_damping(observations, thresholds)
    scores = []
    for mode in (0, 1):
        base = _oracle_classical_reference(priors, np.array([], dtype=float), mode)
        quality = []
        tails = []
        for exposure in scenarios:
            q = []
            t = []
            for (orientation, ea, eb, _) in profiles:
                damp = 1 - (1 - pair) ** (np.array([ea, eb]) * exposure)
                if orientation:
                    damp = damp[::-1]
                rho = _oracle_damped_resource(*damp)
                L = _oracle_conditional_fidelity_law(rho, mode)
                q.append(_oracle_importance_expectations(L, priors) - base)
                t.append(1 - _oracle_fidelity_cdf(L, np.array([target]))[0])
            quality.extend(np.array(q).T)
            tails.append(t)
        result = _oracle_optimize_policy(np.array(quality), np.array(tails), profiles[:, 3], budget, tail_min)
        scores.append(result[0])
    return float(scores[0] - scores[1])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'observations=np.array([0.7231882775473688, 0.5572218954564145, 0.20852676659117841, '
               '0.38807975204329076, 0.6633375335863507], dtype=float)\n'
               'thresholds=np.array([0.55, 0.7, 0.85], dtype=float)\n'
               'profiles=np.array([[0, 1, 1, 0]], dtype=float)\n'
               'scenarios=np.array([[1, 1]], dtype=float)\n'
               'priors=np.array([[2, 1]], dtype=float)',
      'call': 'teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, 0, 0.8, 0)',
      'gold_call': '_oracle_teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, '
                   '0, 0.8, 0)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.7231882775473688, 0.5246163746252991, 0.0, 0.35444029382514236, '
               '0.9999999999999998], dtype=float)\n'
               'thresholds=np.array([0.55, 0.7, 0.85], dtype=float)\n'
               'profiles=np.array([[0, 1, 1, 0], [1, 1, 1, 0]], dtype=float)\n'
               'scenarios=np.array([[1, 1]], dtype=float)\n'
               'priors=np.array([[2, 1], [6, 1]], dtype=float)',
      'call': 'teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, 0, 0.8, 0)',
      'gold_call': '_oracle_teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, '
                   '0, 0.8, 0)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.6252795558988649, 0.45802813889173716, 0.152640058775516, '
               '0.43001741309454333, 0.8201101174886166], dtype=float)\n'
               'thresholds=np.array([0.3, 0.6, 0.9], dtype=float)\n'
               'profiles=np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 0.2, 1.0]], dtype=float)\n'
               'scenarios=np.array([[1, 1]], dtype=float)\n'
               'priors=np.array([[3, 1], [7, 1]], dtype=float)',
      'call': 'teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, 0.3, 0.8, 0)',
      'gold_call': '_oracle_teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, '
                   '0.3, 0.8, 0)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.6252795558988649, 0.39504247463471376, 0.0, 0.7038892266354584, '
               '0.9835463780003222], dtype=float)\n'
               'thresholds=np.array([0.52, 0.65, 0.8], dtype=float)\n'
               'profiles=np.array([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.05], [0.0, 0.3, 0.3, 0.8]], '
               'dtype=float)\n'
               'scenarios=np.array([[0.9, 1.2], [1.2, 0.9]], dtype=float)\n'
               'priors=np.array([[2, 1], [8, 1]], dtype=float)',
      'call': 'teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, 0.4, 0.7, '
              '0.1)',
      'gold_call': '_oracle_teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, '
                   '0.4, 0.7, 0.1)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.749933329999333, 0.5692774995555574, 0.07849250407879499, '
               '0.3776887155912776, 0.9999999999999999], dtype=float)\n'
               'thresholds=np.array([0.6, 0.75, 0.85], dtype=float)\n'
               'profiles=np.array([[0.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, 0.05], [0.0, 0.4, 0.4, 0.8]], '
               'dtype=float)\n'
               'scenarios=np.array([[1, 1]], dtype=float)\n'
               'priors=np.array([[2.0, 1.0], [12.0, 2.5]], dtype=float)',
      'call': 'teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, 0.2, 0.85, 0)',
      'gold_call': '_oracle_teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, '
                   '0.2, 0.85, 0)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.8924572710215865, 0.7980367565068479, 0.17170758629017224, '
               '0.9999999999999997, 0.9999999999999997], dtype=float)\n'
               'thresholds=np.array([0.85, 0.93, 0.98], dtype=float)\n'
               'profiles=np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.3, 0.3, 1.0]], dtype=float)\n'
               'scenarios=np.array([[1, 1]], dtype=float)\n'
               'priors=np.array([[1, 8], [2, 5]], dtype=float)',
      'call': 'teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, 0.5, 0.8, 0)',
      'gold_call': '_oracle_teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, '
                   '0.5, 0.8, 0)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.6728564615223281, 0.49100192876364895, 0.049107657954781773, '
               '0.40478708742856784, 0.8898200250677972], dtype=float)\n'
               'thresholds=np.array([0.3, 0.65, 0.9], dtype=float)\n'
               'profiles=np.array([[0.0, 1.0, 1.0, 0.0], [0.0, 0.2, 0.2, 1.0]], dtype=float)\n'
               'scenarios=np.array([[0.8, 1.1], [1.1, 0.8]], dtype=float)\n'
               'priors=np.array([[2, 1], [5, 1]], dtype=float)',
      'call': 'teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, 0.3, 0.7, 0)',
      'gold_call': '_oracle_teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, '
                   '0.3, 0.7, 0)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'observations=np.array([0.7316264038725353, 0.5601691381686221, 0.7014132758927534, '
               '0.16305444991419757, 0.7014132758927534, 0.35408732210301164], dtype=float)\n'
               'thresholds=np.array([0.85, 0.55, 0.85, 0.7], dtype=float)\n'
               'profiles=np.array([[0.0, 0.4, 0.4, 0.2], [0.0, 0.4, 0.4, 0.2]], dtype=float)\n'
               'scenarios=np.array([[1.1, 0.9], [0.9, 1.1]], dtype=float)\n'
               'priors=np.array([[2, 1], [7, 1]], dtype=float)',
      'call': 'teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, 0.2, 0.8, 0)',
      'gold_call': '_oracle_teleportation_benchmark(observations, thresholds, profiles, scenarios, priors, '
                   '0.2, 0.8, 0)',
      'tol': 2e-06},
     {'setup': 'import numpy as np\n'
               'def _exception_code(fn):\n'
               '    try:\n'
               '        fn()\n'
               '    except ValueError:\n'
               '        return 1.0\n'
               '    except Exception:\n'
               '        return 2.0\n'
               '    return 0.0',
      'call': '_exception_code(lambda: teleportation_benchmark(np.ones(5), np.ones(3), np.ones((2,3)), '
              'np.ones((1,2)), np.ones((1,2)), 0., .8, 0.))',
      'gold_call': '_exception_code(lambda: _oracle_teleportation_benchmark(np.ones(5), np.ones(3), '
                   'np.ones((2,3)), np.ones((1,2)), np.ones((1,2)), 0., .8, 0.))',
      'tol': 0.0}]

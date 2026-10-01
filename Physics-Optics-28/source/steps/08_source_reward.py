"""
Return the optimized pooled reward for one declared optical-source configuration.

The benchmark configuration is the paper-grounded three-mode source specified
in the main problem. The diagnostic configurations use the same projected
generators, count-selected Gaussian controls, incident-attempt weights and
pooled physical reward. These named configurations are the supported domain
of this final orchestration step; the preceding steps have their own input
domains. No optimizer coordinates are returned.

Returns
-------
One finite real value within 0.0004 of the configured bounded-domain optimum, including both evaluation and optimization errors.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def source_reward(configuration: str = "benchmark") -> float:
    """Return one finite real optimal reward for the named source configuration.

    configuration is a string in {"benchmark", "vacuum", "common", "adaptive",
    "rare_detector"}; it defaults to "benchmark". Other values or types raise
    ValueError. Each configuration is completely specified below. Angles and
    phases are in radians; mixer rows act on (0,1),(1,2),(0,2) in that order.

    "benchmark": d=28; squeezers=((.50,.18),(.41,2.72),(.46,.39));
    mixers=((.63,.27),(.51,1.03),(.37,-.42)); first_counts=(1,2);
    second_counts=(2,1); efficiencies=(.99,.98,.95,.99,.99);
    target_spec=(sqrt(6),.5,0); rmax=.5; adaptive=True.

    "vacuum": d=3; all three squeezer rows=(0,0);
    mixers=((.31,.2),(.27,-.4),(.19,.7)); first_counts=(1,);
    second_counts=(0,); efficiencies=(.96,.91,.93,.89,.94);
    target_spec=(1.1,.2,.1); rmax=.18; adaptive=True.

    "common" and "adaptive": d=4;
    squeezers=((.27,.1),(.18,-.5),(.33,.7));
    mixers=((.34,-.1),(.22,.6),(.41,-.4)); first_counts=(0,1);
    second_counts=(1,0); efficiencies=(.91,.88,.93,.90,.95);
    target_spec=(1.3,.22,.1); rmax=.15. adaptive is False for "common"
    and True for "adaptive".

    "rare_detector": d=3 with the benchmark squeezer and mixer arrays;
    first_counts=(1,); second_counts=(2,); efficiencies=(1,1,1,1e-200,1);
    target_spec=(.9,0,0); rmax=0; adaptive=True.

    The five efficiencies are first detector, the two retained memories,
    second detector, final signal loss. target_spec=(alpha,r,phase) means
    the normalized truncated odd coherent-state superposition at real
    alpha, followed by the projected single-mode squeeze (r,phase).
    The controller and pooled reward have the optimal_reward conventions.
    Preserve first-record weights relative to incident preparation attempts.
    A positive success probability has no floor, even if its final numeric
    representation underflows. Use the same shared C(H) state interpretation.

    Returns
    -------
    One finite real value within 0.0004 of the configured bounded-domain
    optimum, including both evaluation and optimization errors.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
import numpy as np

def _oracle_source_reward(configuration: str = "benchmark") -> float:
    if not isinstance(configuration, str) or configuration not in (
            "benchmark", "vacuum", "common", "adaptive", "rare_detector"):
        raise ValueError('unknown source configuration')
    d = 28
    squeezers = np.array([[.50, .18], [.41, 2.72], [.46, .39]])
    mixers = np.array([[.63, .27], [.51, 1.03], [.37, -.42]])
    first_counts, second_counts = (1, 2), (2, 1)
    efficiencies = np.array([.99, .98, .95, .99, .99])
    target_spec, rmax, adaptive = (6**.5, .5, 0.), .5, True
    if configuration == "vacuum":
        d, squeezers = 3, np.zeros((3, 2))
        mixers = np.array([[.31, .2], [.27, -.4], [.19, .7]])
        first_counts, second_counts = (1,), (0,)
        efficiencies = np.array([.96, .91, .93, .89, .94])
        target_spec, rmax = (1.1, .2, .1), .18
    elif configuration in ("common", "adaptive"):
        d = 4
        squeezers = np.array([[.27, .1], [.18, -.5], [.33, .7]])
        mixers = np.array([[.34, -.1], [.22, .6], [.41, -.4]])
        first_counts, second_counts = (0, 1), (1, 0)
        efficiencies = np.array([.91, .88, .93, .90, .95])
        target_spec, rmax = (1.3, .22, .1), .15
        adaptive = configuration == "adaptive"
    elif configuration == "rare_detector":
        d = 3
        first_counts, second_counts = (1,), (2,)
        efficiencies = np.array([1., 1., 1., 1e-200, 1.])
        target_spec, rmax = (.9, 0., 0.), 0.
    states = _oracle_retained_states(d, squeezers, mixers, first_counts, efficiencies[:3])
    alpha, r, phase = target_spec
    core = np.zeros(d, complex)
    for n in range(1, d, 2):
        core[n] = alpha**(n-1)/math.sqrt(float(math.factorial(n)))
    core /= np.linalg.norm(core)
    target = _oracle_gaussian_gate(d, 'squeeze', [r, phase])@core
    baseline_moments = _oracle_policy_statistics(
        states, target, np.zeros((len(first_counts), 8)), second_counts, efficiencies[3:])
    baseline_probability, baseline_overlap = baseline_moments.sum(axis=0)
    baseline_reward = (0. if baseline_probability == 0 else
                       baseline_probability+baseline_overlap/baseline_probability)
    reward = float(_oracle_optimal_reward(
        states, target, second_counts, efficiencies[3:], rmax, adaptive))
    if reward+5e-10 < baseline_reward:
        raise RuntimeError('optimized reward is below a valid baseline policy')
    return reward

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'def _case_encode(value):\n'
           '    raw=np.asarray(value)\n'
           '    if raw.shape!=() or raw.dtype.kind not in "iuf" or not np.isfinite(raw):\n'
           '        return np.array([np.nan])\n'
           '    return np.array([float(raw)*(1e-9/.0004)])\n',
  'call': "_case_encode(source_reward('vacuum'))",
  'gold_call': "_case_encode(_oracle_source_reward('vacuum'))"},
 {'setup': 'def _case_encode(value):\n'
           '    raw=np.asarray(value)\n'
           '    if raw.shape!=() or raw.dtype.kind not in "iuf" or not np.isfinite(raw):\n'
           '        return np.array([np.nan])\n'
           '    return np.array([float(raw)*(1e-9/.0004)])\n',
  'call': "_case_encode(source_reward('common'))",
  'gold_call': "_case_encode(_oracle_source_reward('common'))"},
 {'setup': 'def _case_encode(value):\n'
           '    raw=np.asarray(value)\n'
           '    if raw.shape!=() or raw.dtype.kind not in "iuf" or not np.isfinite(raw):\n'
           '        return np.array([np.nan])\n'
           '    return np.array([float(raw)*(1e-9/.0004)])\n',
  'call': "_case_encode(source_reward('adaptive'))",
  'gold_call': "_case_encode(_oracle_source_reward('adaptive'))"},
 {'setup': 'def _case_encode(value):\n'
           '    raw=np.asarray(value)\n'
           '    if raw.shape!=() or raw.dtype.kind not in "iuf" or not np.isfinite(raw):\n'
           '        return np.array([np.nan])\n'
           '    return np.array([float(raw)*(1e-9/.0004)])\n',
  'call': "_case_encode(source_reward('rare_detector'))",
  'gold_call': "_case_encode(_oracle_source_reward('rare_detector'))"},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: source_reward("unlisted"))',
  'gold_call': '_case_value_error(lambda: _oracle_source_reward("unlisted"))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: source_reward(None))',
  'gold_call': '_case_value_error(lambda: _oracle_source_reward(None))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: source_reward(1))',
  'gold_call': '_case_value_error(lambda: _oracle_source_reward(1))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: source_reward(True))',
  'gold_call': '_case_value_error(lambda: _oracle_source_reward(True))'},
 {'setup': 'def _case_value_error(thunk):\n'
           '    try:\n'
           '        thunk()\n'
           '    except ValueError:\n'
           '        return True\n'
           '    return False\n',
  'call': '_case_value_error(lambda: source_reward(["benchmark"]))',
  'gold_call': '_case_value_error(lambda: _oracle_source_reward(["benchmark"]))'}]

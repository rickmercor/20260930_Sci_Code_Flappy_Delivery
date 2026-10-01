"""
Step 5: propagate one queue-based cascading-fragment realization.

Consume one entry draw, keep its physical parameters and nuisance point fixed within the realization, and process a first-in-first-out queue.  $cascade=False$ still permits the initial breakup but prevents its children from breaking.  Retain only impacts with mass at least `.01 kg`, count all breakup events, and repeat realization metadata on every retained descendant.

Returns
-------
Return a nonempty finite `(n_impacts,8)` array ordered `(east,north,mass,uS,uD,uBeta,uWind,breakup_count)`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Step 5: propagate one queue-based cascading-fragment realization."""
import numpy as np

def simulate_realization(rng=None, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2):
    """Simulate one FIFO fragmentation realization.

    Parameters
    ----------
    rng : numpy.random.Generator or None
        Advancing stream with standard_normal and uniform, default PCG64(761903).
    kick_power : float
        Finite nonnegative radius-ratio exponent inside the kick square root.
    momentum_correct, cascade : bool
        Kick centering and repeated-breakup switches. With cascade=False,
        the first breakup remains enabled when max_depth>0.
    rtol, max_step : float
        Finite positive RK45 controls.
    max_depth : int
        Finite nonnegative breakup depth limit, not bool.

    Returns
    -------
    ndarray, shape (n_impacts,8)
        (east,north,mass,uS,uD,uBeta,uWind,breakups) in retained FIFO order.

    Raises
    ------
    ValueError
        For nonfinite or out-of-domain controls, nonboolean switches, or
        rng lacking either random-number method.
    RuntimeError
        If propagation fails or the realization has no retained impacts.
    Notes
    -----
    Comparison fixtures round east/north to 0.1 m and mass to 1e-6 kg;
    array shape, FIFO row order, nuisance values, and discrete breakup count
    remain exact. The supplied numerical controls define the discrete target.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 5: propagate one queue-based cascading-fragment realization."""
import numpy as np

def _oracle_simulate_realization(rng=None, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2):
    if not np.all(np.isfinite([kick_power, rtol, max_step, max_depth])):
        raise ValueError('simulation controls must be finite')
    if kick_power < 0 or rtol <= 0 or max_step <= 0 or (max_depth < 0) or (int(max_depth) != max_depth) or isinstance(max_depth, (bool, np.bool_)):
        raise ValueError('invalid simulation control domain')
    if not isinstance(momentum_correct, (bool, np.bool_)) or not isinstance(cascade, (bool, np.bool_)):
        raise ValueError('momentum_correct and cascade must be boolean')
    if rng is None:
        rng = np.random.Generator(np.random.PCG64(761903))
    if not hasattr(rng, 'standard_normal') or not hasattr(rng, 'uniform'):
        raise ValueError('rng must provide standard_normal and uniform')
    entry = _oracle_sample_entry(rng)
    strength, dust, largest, beta, wind_scale, cd_scale, sigma_abl = entry[7:14]
    nuisance = entry[14:18]
    queue = [np.concatenate((entry[:7], [strength, 0.0]))]
    impacts = []
    breakup_count = 0
    while queue:
        current = queue.pop(0)
        terminal = _oracle_propagate_fragment(current, wind_scale, cd_scale, sigma_abl, rtol, max_step, max_depth)
        event_code = int(round(terminal[0]))
        if event_code == 2:
            breakup_count += 1
            parent_depth = current[8] if cascade else float(max_depth)
            parent = np.concatenate((terminal[1:8], [current[7], parent_depth]))
            children = _oracle_partition_fragment_cloud(parent, dust, largest, beta, wind_scale, rng, 0.16, 15000000.0, 0.2, kick_power, momentum_correct)
            queue.extend(children)
        elif event_code == 0 and terminal[7] >= 0.01:
            impacts.append([terminal[1], terminal[2], terminal[7]])
    if not impacts:
        raise RuntimeError('realization produced no surviving impacts')
    impacts = np.asarray(impacts, dtype=float)
    repeated_nuisance = np.repeat(nuisance[None, :], len(impacts), axis=0)
    repeated_breakups = np.full((len(impacts), 1), float(breakup_count))
    return np.hstack((impacts, repeated_nuisance, repeated_breakups))

def _reported_impacts(value):
    """Apply only the disclosed measurement-resolution comparison map."""
    result = np.asarray(value, dtype=float).copy()
    if result.ndim == 2 and result.shape[1] >= 3:
        result[:, :2] = np.round(result[:, :2], 1)
        result[:, 2] = np.round(result[:, 2], 6)
    return result

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Exercise two random streams and the single-breakup branch."""
    return [{'setup': 'rng_candidate=np.random.Generator(np.random.PCG64(761903)); rng_oracle=np.random.Generator(np.random.PCG64(761903))', 'call': '_reported_impacts(simulate_realization(rng_candidate))', 'gold_call': '_reported_impacts(_oracle_simulate_realization(rng_oracle))'}, {'setup': 'rng_candidate=np.random.Generator(np.random.PCG64(1701)); rng_oracle=np.random.Generator(np.random.PCG64(1701))', 'call': '_reported_impacts(simulate_realization(rng_candidate, max_step=0.5))', 'gold_call': '_reported_impacts(_oracle_simulate_realization(rng_oracle, max_step=0.5))'}, {'setup': 'rng_candidate=np.random.Generator(np.random.PCG64(761903)); rng_oracle=np.random.Generator(np.random.PCG64(761903))', 'call': '_reported_impacts(simulate_realization(rng_candidate,cascade=False,kick_power=1.,momentum_correct=False))', 'gold_call': '_reported_impacts(_oracle_simulate_realization(rng_oracle,cascade=False,kick_power=1.,momentum_correct=False))'}, {'setup': 'rng_candidate=np.random.default_rng(47); rng_oracle=np.random.default_rng(47)\n', 'call': '_reported_impacts(simulate_realization(rng_candidate,kick_power=.8,rtol=1e-8,max_step=.9,max_depth=0))', 'gold_call': '_reported_impacts(_oracle_simulate_realization(rng_oracle,kick_power=.8,rtol=1e-8,max_step=.9,max_depth=0))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(rng=object()))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(rng=object()))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(kick_power=-1.))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(kick_power=-1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(kick_power=np.nan))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(kick_power=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(kick_power=np.inf))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(kick_power=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(rtol=0.))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(rtol=0.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(rtol=np.nan))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(rtol=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(max_step=0.))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(max_step=0.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(max_step=np.inf))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(max_step=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(max_depth=-1.))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(max_depth=-1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(max_depth=.5))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(max_depth=.5))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(max_depth=True))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(max_depth=True))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(max_depth=np.nan))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(max_depth=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(cascade=1))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(cascade=1))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: simulate_realization(momentum_correct=1))', 'gold_call': 'rejects(lambda: _oracle_simulate_realization(momentum_correct=1))'}]

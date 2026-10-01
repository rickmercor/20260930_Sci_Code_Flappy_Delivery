"""
Step 6: assemble the proposal cloud and retain realization metadata.

Create exactly one `Generator(PCG64(seed))`, pass the same advancing stream through all requested realizations, and insert the zero-based realization label after mass.  Do not reseed between realizations or branches.

Returns
-------
Return a nonempty finite `(n_impacts,9)` array ordered `(east,north,mass,run,uS,uD,uBeta,uWind,breakup_count)`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Step 6: assemble the proposal cloud and retain realization metadata."""
import numpy as np

def build_impact_cloud(seed=761903, runs=8, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2):
    """Assemble independent entry realizations from one advancing stream.

    Parameters
    ----------
    seed, runs : int
        Finite integers, not bool. seed>=0 and 1<=runs<=1000.
    kick_power : float
        Finite nonnegative radius-ratio exponent inside the kick square root.
    momentum_correct, cascade : bool
        Kick centering and repeated-breakup switches.
    rtol, max_step : float
        Finite positive RK45 controls.
    max_depth : int
        Finite nonnegative breakup depth limit, not bool.

    Returns
    -------
    ndarray, shape (n_impacts,9)
        (east,north,mass,run,uS,uD,uBeta,uWind,breakups), zero-based run labels.

    Raises
    ------
    ValueError
        For any nonfinite or out-of-domain numeric control or nonboolean switch.
    RuntimeError
        If a realization fails or has no retained impacts.
    Notes
    -----
    Differential comparisons use absolute tolerances of 5e-3 m for east and
    north and 5e-6 kg for mass. Row count, row order, run labels, nuisance
    values, and breakup counts are exact.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 6: assemble the proposal cloud and retain realization metadata."""
import numpy as np

def _oracle_build_impact_cloud(seed=761903, runs=8, kick_power=2.0, momentum_correct=True, cascade=True, rtol=2e-10, max_step=0.35, max_depth=2):
    if not np.all(np.isfinite([seed, runs])):
        raise ValueError('seed and runs must be finite integers')
    if isinstance(seed, (bool, np.bool_)) or isinstance(runs, (bool, np.bool_)) or int(seed) != seed or (int(runs) != runs) or (seed < 0):
        raise ValueError('seed must be a nonnegative integer and runs an integer')
    seed = int(seed)
    runs = int(runs)
    if runs < 1 or runs > 1000:
        raise ValueError('runs must be between one and one thousand')
    if not np.isfinite(seed):
        raise ValueError('seed must be finite')
    rng = np.random.Generator(np.random.PCG64(seed))
    rows = []
    for realization in range(runs):
        impacts = _oracle_simulate_realization(rng, kick_power, momentum_correct, cascade, rtol, max_step, max_depth)
        labels = np.full((len(impacts), 1), float(realization))
        rows.append(np.hstack((impacts[:, :3], labels, impacts[:, 3:])))
    return np.vstack(rows)

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np

def test_cases():
    """Cover corrected, legacy, and varied-seed proposal clouds."""
    return [{'setup': 'def packed(c):\n    return np.column_stack((c[:, :2], 1000.0*c[:, 2], 1e9*c[:, 3:]))', 'call': 'packed(build_impact_cloud(runs=1))', 'gold_call': 'packed(_oracle_build_impact_cloud(runs=1))', 'tol': 0.005}, {'setup': 'def packed(c):\n    return np.column_stack((c[:, :2], 1000.0*c[:, 2], 1e9*c[:, 3:]))', 'call': 'packed(build_impact_cloud(seed=1701, runs=1, max_step=0.5))', 'gold_call': 'packed(_oracle_build_impact_cloud(seed=1701, runs=1, max_step=0.5))', 'tol': 0.005}, {'setup': 'def packed(c):\n    return np.column_stack((c[:, :2], 1000.0*c[:, 2], 1e9*c[:, 3:]))', 'call': 'packed(build_impact_cloud(runs=2,kick_power=1.,cascade=False,momentum_correct=False))', 'gold_call': 'packed(_oracle_build_impact_cloud(runs=2,kick_power=1.,cascade=False,momentum_correct=False))', 'tol': 0.005}, {'setup': 'def packed(c):\n    return np.column_stack((c[:, :2], 1000.0*c[:, 2], 1e9*c[:, 3:]))', 'call': 'packed(build_impact_cloud(seed=47,runs=1,kick_power=.8,rtol=1e-8,max_step=.9,max_depth=1))', 'gold_call': 'packed(_oracle_build_impact_cloud(seed=47,runs=1,kick_power=.8,rtol=1e-8,max_step=.9,max_depth=1))', 'tol': 0.005}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(seed=-1.))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(seed=-1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(seed=.5))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(seed=.5))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(seed=True))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(seed=True))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(seed=np.nan))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(seed=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(seed=np.inf))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(seed=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(seed=-np.inf))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(seed=-np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(runs=0))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(runs=0))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(runs=1001))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(runs=1001))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(runs=.5))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(runs=.5))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(runs=True))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(runs=True))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(runs=np.nan))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(runs=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\n', 'call': 'rejects(lambda: build_impact_cloud(runs=np.inf))', 'gold_call': 'rejects(lambda: _oracle_build_impact_cloud(runs=np.inf))'}]

"""
Step 3: propagate one fragment to impact, depletion, or breakup.

Interpret the input as `(state7,strength,depth)`. Integrate stage 2 on `[0,5000]` with SciPy RK45, supplied `rtol` and $max_step$, and fixed $atol=(1e-5,1e-5,1e-5,1e-6,1e-6,1e-6,1e-12)$. Terminal events are ground $z=0$ with direction $-1$, depletion `m-.001=0` with direction $-1$, and, only when integer depth is below $max_depth$, breakup $ram_pressure-strength=0$ with direction $+1$. Keep that event-list order, scan $y_events$ in the same order, and force terminal impact altitude to zero.

Returns
-------
Return a finite length-8 vector with event code `0`, `1`, or `2` followed by the terminal seven-state vector.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

"""Step 3: propagate one fragment to impact, depletion, or breakup."""
import numpy as np
from scipy.integrate import solve_ivp

def propagate_fragment(fragment, wind_scale, cd_scale, sigma_abl, rtol=2e-10, max_step=0.35, max_depth=2):
    """Propagate a fragment to the first terminal event.

    Parameters
    ----------
    fragment : array_like, shape (9,)
        Finite (state7,strength,depth), positive mass and strength, and
        nonnegative integer depth.
    wind_scale, cd_scale, sigma_abl : float
        Finite scales with domains >=0, >0, >=0, respectively, as in stage 2.
    rtol, max_step : float
        Finite positive RK45 relative tolerance and maximum time step in s.
    max_depth : int
        Finite nonnegative integer, not bool, below which breakup is enabled.

    Returns
    -------
    ndarray, shape (8,)
        Event code (0 ground, 1 depletion, 2 breakup), then terminal state7.

    Raises
    ------
    ValueError
        For invalid shape, nonfinite data or controls, nonpositive mass,
        strength, rtol, max_step or drag scale, negative wind or ablation,
        or a depth outside its integer domain.
    RuntimeError
        If integration fails or no terminal event occurs by 5000 s.
    Notes
    -----
    Finite terminal-state components are compared with absolute tolerance
    5e-3; the event code is exact. The supplied numerical controls define
    the discrete target.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

"""Step 3: propagate one fragment to impact, depletion, or breakup."""
import numpy as np
from scipy.integrate import solve_ivp

def _oracle_propagate_fragment(fragment, wind_scale, cd_scale, sigma_abl, rtol=2e-10, max_step=0.35, max_depth=2):
    fragment = np.asarray(fragment, dtype=float)
    if fragment.shape != (9,) or not np.all(np.isfinite(fragment)):
        raise ValueError('fragment must contain state, strength, and depth')
    if fragment[6] <= 0.0 or fragment[7] <= 0.0 or fragment[8] < 0.0 or (fragment[8] != np.floor(fragment[8])):
        raise ValueError('fragment mass, strength, and depth are invalid')
    if not np.all(np.isfinite([rtol, max_step, max_depth])):
        raise ValueError('integration controls must be finite')
    if rtol <= 0.0 or max_step <= 0.0 or max_depth < 0 or (int(max_depth) != max_depth) or isinstance(max_depth, (bool, np.bool_)):
        raise ValueError('integration controls are invalid')
    state0 = fragment[:7]
    strength = float(fragment[7])
    depth = int(round(fragment[8]))

    def rhs(_, state):
        return _oracle_atmospheric_rhs(state, wind_scale, cd_scale, sigma_abl)[:7]

    def ground(_, state):
        return state[2]

    def depleted(_, state):
        return state[6] - 0.001
    ground.terminal = True
    ground.direction = -1
    depleted.terminal = True
    depleted.direction = -1
    events = [ground, depleted]
    if depth < int(max_depth):

        def breakup(_, state):
            return _oracle_atmospheric_rhs(state, wind_scale, cd_scale, sigma_abl)[13] - strength
        breakup.terminal = True
        breakup.direction = 1
        events.append(breakup)
    solution = solve_ivp(rhs, (0.0, 5000.0), state0, method='RK45', rtol=rtol, atol=np.array([1e-05, 1e-05, 1e-05, 1e-06, 1e-06, 1e-06, 1e-12]), max_step=max_step, events=events)
    if not solution.success:
        raise RuntimeError(solution.message)
    for event_index, event_states in enumerate(solution.y_events):
        if len(event_states):
            terminal = event_states[0].copy()
            if event_index == 0:
                terminal[2] = 0.0
                return np.concatenate(([0.0], terminal))
            if event_index == 1:
                return np.concatenate(([1.0], terminal))
            return np.concatenate(([2.0], terminal))
    raise RuntimeError('trajectory terminated without a terminal event')

# =============================================================================
# TEST CASES
# =============================================================================

import numpy as np
from scipy.integrate import solve_ivp

def test_cases():
    """Exercise a luminous breakup, dark-flight impact, and invalid shape."""
    return [{'setup': 'fragment=np.array([0.,0.,100000.,12400.,180.,-5250.,185000.,5.5e5,0.])', 'call': 'propagate_fragment(fragment,1.0,1.0,8e-8,max_depth=0)', 'gold_call': '_oracle_propagate_fragment(fragment,1.0,1.0,8e-8,max_depth=0)', 'tol': 0.005}, {'setup': 'fragment=np.array([0.,0.,100000.,12400.,180.,-5250.,185000.,5.5e5,0.])', 'call': 'propagate_fragment(fragment,1.0,1.0,8e-8)', 'gold_call': '_oracle_propagate_fragment(fragment,1.0,1.0,8e-8)', 'tol': 0.005}, {'setup': 'fragment=np.array([30.,-5.,900.,75.,-9.,-86.,2.5,8e6,2.])', 'call': 'propagate_fragment(fragment,0.9,1.05,8e-8,max_step=1.5)', 'gold_call': '_oracle_propagate_fragment(fragment,0.9,1.05,8e-8,max_step=1.5)', 'tol': 0.005}, {'setup': 'def catches(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nfragment=np.ones(8)', 'call': 'catches(lambda: propagate_fragment(fragment,1.,1.,8e-8))', 'gold_call': 'catches(lambda: _oracle_propagate_fragment(fragment,1.,1.,8e-8))'}, {'setup': 'f=np.array([0.,0.,30000.,12000.,50.,-500.,.0012,1e12,2.])\n', 'call': 'propagate_fragment(f,1.,1.,1e-8,max_step=1e-5)', 'gold_call': '_oracle_propagate_fragment(f,1.,1.,1e-8,max_step=1e-5)', 'tol': 0.005}, {'setup': 'f=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'propagate_fragment(f,.85,1.2,1.1e-7,rtol=8e-9,max_step=.7,max_depth=1)', 'gold_call': '_oracle_propagate_fragment(f,.85,1.2,1.1e-7,rtol=8e-9,max_step=.7,max_depth=1)', 'tol': 0.005}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\nf[6]=0.\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\nf[7]=0.\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\nf[8]=-1.\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\nf[8]=.5\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\nf[0]=np.nan\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,rtol=np.nan))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,rtol=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,rtol=np.inf))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,rtol=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,rtol=-np.inf))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,rtol=-np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,rtol=-1.))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,rtol=-1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,max_step=np.nan))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,max_step=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,max_step=np.inf))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,max_step=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,max_step=-np.inf))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,max_step=-np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,max_step=-1.))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,max_step=-1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,max_depth=np.nan))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,max_depth=np.nan))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,max_depth=np.inf))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,max_depth=np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,max_depth=-np.inf))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,max_depth=-np.inf))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,max_depth=-1.))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,max_depth=-1.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,rtol=0.))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,rtol=0.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,max_step=0.))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,max_step=0.))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,max_depth=.5))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,max_depth=.5))'}, {'setup': 'def rejects(fn):\n    try: fn()\n    except ValueError: return 1.0\n    return 0.0\nf=np.array([11.,-7.,18000.,5300.,70.,-2100.,400.,7e5,1.])\n', 'call': 'rejects(lambda: propagate_fragment(f,1.,1.,8e-8,max_depth=True))', 'gold_call': 'rejects(lambda: _oracle_propagate_fragment(f,1.,1.,8e-8,max_depth=True))'}]

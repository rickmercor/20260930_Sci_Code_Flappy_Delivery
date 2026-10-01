"""
Simulate one exact remaining clock gene-expression path.

The exact process races four reaction clocks against one completion clock per unfinished transcript and preserves all non-firing residuals after each event.

Returns
-------
np.ndarray: length-4 array [M, P, Mstar, event_count] at the final time.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_exact_gene_path(parameters: "np.ndarray", final_time: float,
                             seed: int, max_events: int = 2000000) -> "np.ndarray":
    """Simulate one exact path and return ``[M,P,Mstar,event_count]``.

    Raises
    ------
    ValueError
        If inputs are inadmissible or the event limit is reached.
    """
    return endpoint  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_simulate_exact_gene_path(parameters: "np.ndarray", final_time: float,
                                      seed: int,
                                      max_events: int = 2000000) -> "np.ndarray":
    p = np.asarray(parameters, dtype=float)
    if p.shape != (10,) or not np.all(np.isfinite(p)):
        raise ValueError("parameters must be a finite length-10 vector")
    if not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    if isinstance(max_events, bool) or not isinstance(max_events, (int, np.integer)) or max_events < 1:
        raise ValueError("max_events must be a positive integer")
    rng = np.random.default_rng(int(seed))
    state = np.zeros(3, dtype=int)
    reaction_residuals = rng.exponential(size=4)
    ages, completion_residuals = np.empty(0), np.empty(0)
    # Net change for an initiation, a translation, and the two decays.
    zeta = np.array([[0, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    time, events = 0.0, 0
    while time < final_time:
        prop = _oracle_gene_channel_propensities(state, p)
        reaction_waits = np.full(4, np.inf)
        active = prop > 0.0
        reaction_waits[active] = reaction_residuals[active] / prop[active]
        completion_waits = _oracle_invert_completion_waits(
            ages, completion_residuals, state, p)
        waits = np.concatenate((reaction_waits, completion_waits))
        event = int(np.argmin(waits))
        dt = float(waits[event])
        if not np.isfinite(dt) or time + dt > final_time:
            break
        packed = _oracle_advance_remaining_clocks(
            reaction_residuals, ages, completion_residuals, dt, state, p)
        count = ages.size
        reaction_residuals = packed[:4]
        completion_residuals = packed[4:4 + count]
        ages = packed[4 + count:]
        if event < 4:
            reaction_residuals[event] = rng.exponential()
            state += zeta[event]
            if event == 0:
                state[2] += 1
                ages = np.append(ages, 0.0)
                completion_residuals = np.append(completion_residuals, rng.exponential())
        else:
            index = event - 4
            state += np.array([1, 0, -1])
            ages = np.delete(ages, index)
            completion_residuals = np.delete(completion_residuals, index)
        time += dt
        events += 1
        if events >= max_events:
            raise ValueError("maximum event count reached")
    return np.array([state[0], state[1], state[2], events], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\np=np.array([2,.4,.5,.2,.1,2,5,.2,2,3.])", "call": "float(np.dot(simulate_exact_gene_path(p,5,7),[1,2,3,.01]))", "gold_call": "float(np.dot(_oracle_simulate_exact_gene_path(p,5,7),[1,2,3,.01]))"},
        {"setup": "import numpy as np\np=np.array([1,.5,1,.1,.1,1,5,0,1,2.])", "call": "float(np.dot(simulate_exact_gene_path(p,1,0),[1,2,3,4]))", "gold_call": "float(np.dot(_oracle_simulate_exact_gene_path(p,1,0),[1,2,3,4]))"},
        {"setup": "import numpy as np\np=np.array([1,.5,1,.1,.1,3,5,0,1,2.])", "call": "float(np.sum(simulate_exact_gene_path(p,3,99)))", "gold_call": "float(np.sum(_oracle_simulate_exact_gene_path(p,3,99)))"},
        {"setup": "import numpy as np\np=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])\ndef bad():\n    try: simulate_exact_gene_path(p,0,7); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef bad_gold():\n    try: _oracle_simulate_exact_gene_path(p,0,7); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "bad()", "gold_call": "bad_gold()"},
    ]

"""
Simulate one grouped-delay non-Markovian tau-leap path.

The approximate simulator freezes state over each leap, advances channels by Poisson counts, and stores new initiations as a common-age delay group.

Returns
-------
np.ndarray: length-4 array [M, P, Mstar, step_count] at the final time.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def simulate_tau_gene_path(parameters: "np.ndarray", final_time: float,
                           tau: float, seed: int) -> "np.ndarray":
    """Simulate one approximate path and return ``[M,P,Mstar,step_count]``.

    Raises
    ------
    ValueError
        If parameters, times, or seed are inadmissible.
    """
    return endpoint  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
def _oracle_simulate_tau_gene_path(parameters: "np.ndarray", final_time: float,
                                    tau: float, seed: int) -> "np.ndarray":
    p = np.asarray(parameters, dtype=float)
    if p.shape != (10,) or not np.all(np.isfinite(p)):
        raise ValueError("parameters must be a finite length-10 vector")
    if not np.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("final_time must be positive and finite")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be positive and finite")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an integer")
    rng = np.random.default_rng(int(seed))
    state = np.zeros(3, dtype=int)
    ages = np.empty(0, dtype=float)
    sizes = np.empty(0, dtype=int)
    time = 0.0
    steps = 0
    while time < final_time:
        dt = min(float(tau), float(final_time) - time)
        prop = _oracle_gene_channel_propensities(state, p)
        reaction_counts = rng.poisson(prop * dt)
        means = _oracle_compute_group_completion_means(ages, sizes, dt, state, p)
        completion_counts = rng.poisson(means)
        completion_counts = np.minimum(completion_counts, sizes)
        transcription = int(reaction_counts[0])
        translation = int(reaction_counts[1])
        mrna_loss = min(int(state[0]), int(reaction_counts[2]))
        protein_loss = min(int(state[1]), int(reaction_counts[3]))
        completed = int(np.sum(completion_counts))
        state[0] += completed - mrna_loss
        state[1] += translation - protein_loss
        if sizes.size:
            sizes = sizes - completion_counts
            ages = ages + dt
            keep = sizes > 0
            sizes = sizes[keep]
            ages = ages[keep]
        if transcription > 0:
            sizes = np.append(sizes, transcription)
            ages = np.append(ages, 0.0)
        state[2] = int(np.sum(sizes))
        time += dt
        steps += 1
    return np.array([state[0], state[1], state[2], steps], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {"setup": "import numpy as np\np=np.array([2,.4,.5,.2,.1,2,5,.2,2,3.])", "call": "float(np.dot(simulate_tau_gene_path(p,5,1,7),[1,2,3,.01]))", "gold_call": "float(np.dot(_oracle_simulate_tau_gene_path(p,5,1,7),[1,2,3,.01]))"},
        {"setup": "import numpy as np\np=np.array([1,.5,1,.1,.1,1,5,0,1,2.])", "call": "float(np.dot(simulate_tau_gene_path(p,1,.1,0),[1,2,3,4]))", "gold_call": "float(np.dot(_oracle_simulate_tau_gene_path(p,1,.1,0),[1,2,3,4]))"},
        {"setup": "import numpy as np\np=np.array([1,.5,1,.1,.1,3,5,0,1,2.])", "call": "float(np.sum(simulate_tau_gene_path(p,3,5,99)))", "gold_call": "float(np.sum(_oracle_simulate_tau_gene_path(p,3,5,99)))"},
        {"setup": "import numpy as np\np=np.array([10,.175,1,.08,.05,2.5,10,.5,1.5,5.])\ndef bad():\n    try: simulate_tau_gene_path(p,5,0,7); return 0\n    except ValueError: return 1\n    except Exception: return 2\ndef bad_gold():\n    try: _oracle_simulate_tau_gene_path(p,5,0,7); return 0\n    except ValueError: return 1\n    except Exception: return 2\n", "call": "bad()", "gold_call": "bad_gold()"},
    ]

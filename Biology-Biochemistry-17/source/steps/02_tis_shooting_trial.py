"""
Perform one two-way shooting trial of the problem statement in the ensemble of paths that leave A and cross the given interface of the given interface set, starting from the supplied current path, and return the path kept after the trial: the trial path if it is accepted, the current path otherwise. The segments are generated with the lattice walk of step 1 and every random number comes from the supplied generator in the order the problem statement prescribes. The submitted function must import inside itself whatever it uses.

Paths are integer arrays of lattice sites (i, j), with x_i = -1.45 + 0.1 i and y_j = -0.95 + 0.1 j; the current path is a member of the ensemble. The order parameters are lambda1 = x, lambda2 = x cos 5deg + y sin 5deg and lambda3 = x + 0.1 sin(2 pi y), set_index selects one of them, and a path crosses the interface when the largest value of that order parameter over its frames is greater than it. The shooting frame is 1 + rng.integers(L - 2) for a current path of L frames; the first segment is run from its site, the trial is rejected at once if that segment ends in B or is abandoned at the step cap, and otherwise the second segment is run from the same site; a second segment abandoned at the cap also rejects the trial. The acceptance number rng.random() is drawn only for a trial path that crosses the interface, so a trial consumes a variable number of random numbers.

Returns
-------
return new_path
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def tis_shooting_trial(path: "np.ndarray", set_index: int, interface: float, beta: float,
                       rng: "np.random.Generator", max_steps: int) -> "np.ndarray":
    """One two-way shooting trial in the interface ensemble (set_index, interface).

    Args:
        path: integer array of shape (L, 2), the current path (sites (i, j)), with L >= 3.
        set_index: interface set 1, 2 or 3, selecting the order parameter.
        interface: the interface value that every path of the ensemble must cross.
        beta: inverse temperature.
        rng: numpy Generator, advanced as the problem statement prescribes.
        max_steps: step cap of each segment.

    Returns:
        np.ndarray: integer array of shape (L_kept, 2), the path kept after the trial.
    """
    return new_path

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_tis_shooting_trial(path: "np.ndarray", set_index: int, interface: float, beta: float,
                               rng: "np.random.Generator", max_steps: int) -> "np.ndarray":
    """One two-way shooting trial in the ensemble of paths leaving A that cross the given interface."""
    import numpy as np
    path = np.asarray(path, dtype=int)
    n_old = len(path)
    s = 1 + int(rng.integers(n_old - 2))
    back = _oracle_metropolis_segment(path[s], beta, rng, max_steps)
    if back[-1, 0] > 5:
        return path
    fwd = _oracle_metropolis_segment(path[s], beta, rng, max_steps)
    if 5 < fwd[-1, 0] < 24:
        return path
    new = np.concatenate([back[::-1], fwd[1:]])
    x = -1.45 + 0.1 * new[:, 0]
    y = -0.95 + 0.1 * new[:, 1]
    theta = np.deg2rad(5.0)
    cvs = (x, x * np.cos(theta) + y * np.sin(theta), x + 0.1 * np.sin(2.0 * np.pi * y))
    if cvs[set_index - 1].max() <= interface:
        return path
    if rng.random() < min(1.0, (n_old - 2) / (len(new) - 2)):
        return new
    return path

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = """import numpy as np
def straight(top):
    return np.array([(c, 9) for c in list(range(5, top + 1)) + list(range(top - 1, 4, -1))])
def trial(f, top, s, lam, seed, cap=100000):
    rng = np.random.default_rng(seed)
    out = f(straight(top), s, lam, 10.0, rng, cap)
    return np.asarray(out, dtype=float), rng.random()
"""
    return [
        {   # 2.1 accepted trial whose path reaches B
            "setup": common,
            "call": "trial(tis_shooting_trial, 15, 1, 0.0, 19)",
            "gold_call": "trial(_oracle_tis_shooting_trial, 15, 1, 0.0, 19)",
        },
        {   # 2.2 accepted trial whose path returns to A
            "setup": common,
            "call": "trial(tis_shooting_trial, 15, 1, 0.0, 36)",
            "gold_call": "trial(_oracle_tis_shooting_trial, 15, 1, 0.0, 36)",
        },
        {   # 2.3 a valid trial path rejected by the path-length rule
            "setup": common,
            "call": "trial(tis_shooting_trial, 15, 1, 0.0, 1)",
            "gold_call": "trial(_oracle_tis_shooting_trial, 15, 1, 0.0, 1)",
        },
        {   # 2.4 the first segment enters B: rejected before the second segment is run
            "setup": common,
            "call": "trial(tis_shooting_trial, 15, 1, 0.0, 9)",
            "gold_call": "trial(_oracle_tis_shooting_trial, 15, 1, 0.0, 9)",
        },
        {   # 2.5 the trial path does not cross the interface: rejected without an acceptance draw
            "setup": common,
            "call": "trial(tis_shooting_trial, 15, 1, 0.0, 0)",
            "gold_call": "trial(_oracle_tis_shooting_trial, 15, 1, 0.0, 0)",
        },
        {   # 2.6 edge: the undulating order parameter of set 3 decides where x alone would not
            "setup": common,
            "call": "trial(tis_shooting_trial, 11, 3, -0.4, 59)",
            "gold_call": "trial(_oracle_tis_shooting_trial, 11, 3, -0.4, 59)",
        },
        {   # 2.7 edge: the second segment reaches the step cap
            "setup": common,
            "call": "trial(tis_shooting_trial, 11, 1, -0.4, 4, 20)",
            "gold_call": "trial(_oracle_tis_shooting_trial, 11, 1, -0.4, 4, 20)",
        },
        {   # 2.8 boundary: a five-frame current path, where the path-length rule rejects a 13-frame trial path
            "setup": common,
            "call": "trial(tis_shooting_trial, 7, 1, -0.8, 3)",
            "gold_call": "trial(_oracle_tis_shooting_trial, 7, 1, -0.8, 3)",
        },
    ]

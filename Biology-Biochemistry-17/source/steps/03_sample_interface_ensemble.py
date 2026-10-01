"""
Sample the interface ensemble (set_index, k) of the problem statement by two-way shooting, with the generator, initial path, interface tables and trial of the problem statement, and return, for each of the n_samples stored paths, the largest value over its frames of each of the three order parameters. The submitted function must import inside itself whatever it uses.

The interfaces are -0.8, -0.7, ..., -0.1, 0.0, 0.2 (k = 1..10) for sets 1 and 3 and -0.7, -0.6, ..., 0.0, 0.2 (k = 1..9) for set 2; k counts from 1. The generator is np.random.default_rng((seed, set_index, k)). The initial path runs along the row j = 9 from i = 5 up to the first site whose order parameter of set set_index exceeds the interface, and back to i = 5 over the same sites. The first n_equil trials are discarded; after each of the next n_samples trials the current path is stored, whether or not the trial was accepted. The three columns of the result are the maxima of lambda1 = x, lambda2 = x cos 5deg + y sin 5deg and lambda3 = x + 0.1 sin(2 pi y), in this order, whatever set is sampled.

Returns
-------
return maxima
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def sample_interface_ensemble(set_index: int, k: int, n_equil: int, n_samples: int, beta: float,
                              seed: int, max_steps: int) -> "np.ndarray":
    """Order-parameter maxima of the stored paths of interface ensemble (set_index, k).

    Args:
        set_index: interface set 1, 2 or 3.
        k: interface index within the set, counted from 1.
        n_equil: number of discarded trials.
        n_samples: number of stored paths.
        beta: inverse temperature.
        seed: first entry of the generator seed (seed, set_index, k).
        max_steps: step cap of each segment.

    Returns:
        np.ndarray: float array of shape (n_samples, 3); row n holds the maxima of lambda1, lambda2 and
            lambda3 over the n-th stored path.
    """
    return maxima

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_sample_interface_ensemble(set_index: int, k: int, n_equil: int, n_samples: int, beta: float,
                                      seed: int, max_steps: int) -> "np.ndarray":
    """Maxima of the three order parameters over the stored paths of TIS ensemble (set_index, k)."""
    import numpy as np
    interfaces = ([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2],
                  [-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2],
                  [-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2])
    lam = interfaces[set_index - 1][k - 1]
    rng = np.random.default_rng((seed, set_index, k))
    theta = np.deg2rad(5.0)
    y0 = -0.95 + 0.1 * 9
    top = 5
    while True:
        x0 = -1.45 + 0.1 * top
        value = (x0, x0 * np.cos(theta) + y0 * np.sin(theta), x0 + 0.1 * np.sin(2.0 * np.pi * y0))[set_index - 1]
        if value > lam:
            break
        top += 1
    cols = list(range(5, top + 1)) + list(range(top - 1, 4, -1))
    path = np.array([(c, 9) for c in cols], dtype=int)
    out = np.empty((n_samples, 3))
    for t in range(n_equil + n_samples):
        path = _oracle_tis_shooting_trial(path, set_index, lam, beta, rng, max_steps)
        if t >= n_equil:
            x = -1.45 + 0.1 * path[:, 0]
            y = -0.95 + 0.1 * path[:, 1]
            out[t - n_equil] = (x.max(), (x * np.cos(theta) + y * np.sin(theta)).max(),
                                (x + 0.1 * np.sin(2.0 * np.pi * y)).max())
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {   # 3.1 set 1, a low interface
            "setup": "",
            "call": "sample_interface_ensemble(1, 3, 5, 40, 10.0, 2026, 100000)",
            "gold_call": "_oracle_sample_interface_ensemble(1, 3, 5, 40, 10.0, 2026, 100000)",
        },
        {   # 3.2 the last interface of set 2: the tilted order parameter sets the initial path
            "setup": "",
            "call": "sample_interface_ensemble(2, 9, 3, 25, 10.0, 7, 100000)",
            "gold_call": "_oracle_sample_interface_ensemble(2, 9, 3, 25, 10.0, 7, 100000)",
        },
        {   # 3.3 edge: set 3 without discarded trials, so the first stored path follows the first trial
            "setup": "",
            "call": "sample_interface_ensemble(3, 1, 0, 30, 10.0, 11, 100000)",
            "gold_call": "_oracle_sample_interface_ensemble(3, 1, 0, 30, 10.0, 11, 100000)",
        },
        {   # 3.4 boundary: a single stored path in the highest ensemble of set 1
            "setup": "",
            "call": "sample_interface_ensemble(1, 10, 0, 1, 10.0, 5, 100000)",
            "gold_call": "_oracle_sample_interface_ensemble(1, 10, 0, 1, 10.0, 5, 100000)",
        },
    ]

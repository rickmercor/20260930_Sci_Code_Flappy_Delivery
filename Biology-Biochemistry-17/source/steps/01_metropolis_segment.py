"""
Run the lattice Metropolis walk of the problem statement from a given site, at a given inverse temperature, until the walker first enters state A or state B or a maximum number of steps has been made, drawing the random numbers from the supplied generator exactly as the problem statement prescribes, and return the visited sites in order. The submitted function must import inside itself whatever it uses.

Sites are given by their lattice indices (i, j), with x_i = -1.45 + 0.1 i for i = 0..29 and y_j = -0.95 + 0.1 j for j = 0..19, the free energy is U(x, y) = (x^2 - 1)^2 + y^2, state A is x < -0.9 (i <= 5) and state B is x > 0.9 (i >= 24). At every step d = rng.integers(4) and then u = rng.random() are drawn; d = 0, 1, 2, 3 proposes (i+1, j), (i-1, j), (i, j+1), (i, j-1); an on-lattice proposal is accepted when u < exp(-beta (U_proposed - U_current)); an off-lattice proposal or a rejected one leaves the walker in place, and each step appends one frame. A start inside A or B is returned as the only frame and consumes no random numbers; the walk stops after max_steps steps even if neither state has been reached.

Returns
-------
return frames
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def metropolis_segment(start: "np.ndarray", beta: float, rng: "np.random.Generator",
                       max_steps: int) -> "np.ndarray":
    """Lattice Metropolis walk from start until state A or B is entered.

    Args:
        start: integer array (i, j) of the starting site.
        beta: inverse temperature (non-negative).
        rng: numpy Generator, advanced by two draws per step.
        max_steps: largest number of steps to make.

    Returns:
        np.ndarray: integer array of shape (n_frames, 2), the visited sites (i, j) in order, starting
            with start; n_frames - 1 is the number of steps made.
    """
    return frames

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_metropolis_segment(start: "np.ndarray", beta: float, rng: "np.random.Generator",
                               max_steps: int) -> "np.ndarray":
    """Metropolis walk on the lattice from start until A or B is entered or max_steps steps are made."""
    import numpy as np
    i, j = int(start[0]), int(start[1])
    frames = [(i, j)]
    u_old = ((-1.45 + 0.1 * i) ** 2 - 1.0) ** 2 + (-0.95 + 0.1 * j) ** 2
    steps = 0
    while 5 < i < 24 and steps < max_steps:
        d = int(rng.integers(4))
        u = rng.random()
        ni, nj = i + (1, -1, 0, 0)[d], j + (0, 0, 1, -1)[d]
        if 0 <= ni < 30 and 0 <= nj < 20:
            u_new = ((-1.45 + 0.1 * ni) ** 2 - 1.0) ** 2 + (-0.95 + 0.1 * nj) ** 2
            if u < np.exp(-beta * (u_new - u_old)):
                i, j, u_old = ni, nj, u_new
        frames.append((i, j))
        steps += 1
    return np.array(frames, dtype=int)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    common = """import numpy as np
def walk(f, start, beta, seed, cap):
    rng = np.random.default_rng(seed)
    frames = f(np.array(start), beta, rng, cap)
    return np.asarray(frames, dtype=float), rng.random()
"""
    return [
        {   # 1.1 start near the barrier top at the benchmark temperature, run until A or B
            "setup": common,
            "call": "walk(metropolis_segment, (15, 9), 10.0, 1, 100000)",
            "gold_call": "walk(_oracle_metropolis_segment, (15, 9), 10.0, 1, 100000)",
        },
        {   # 1.2 boundary: a start inside A is returned alone and consumes no random numbers
            "setup": common,
            "call": "walk(metropolis_segment, (5, 3), 10.0, 2, 100000)",
            "gold_call": "walk(_oracle_metropolis_segment, (5, 3), 10.0, 2, 100000)",
        },
        {   # 1.3 edge: the step cap ends the walk before either state is reached
            "setup": common,
            "call": "walk(metropolis_segment, (12, 9), 10.0, 3, 7)",
            "gold_call": "walk(_oracle_metropolis_segment, (12, 9), 10.0, 3, 7)",
        },
        {   # 1.4 edge: a start on the lattice border, where off-lattice proposals leave the walker in place
            "setup": common,
            "call": "walk(metropolis_segment, (14, 0), 10.0, 4, 300)",
            "gold_call": "walk(_oracle_metropolis_segment, (14, 0), 10.0, 4, 300)",
        },
        {   # 1.5 edge: beta = 0 accepts every on-lattice proposal
            "setup": common,
            "call": "walk(metropolis_segment, (22, 19), 0.0, 5, 400)",
            "gold_call": "walk(_oracle_metropolis_segment, (22, 19), 0.0, 5, 400)",
        },
    ]

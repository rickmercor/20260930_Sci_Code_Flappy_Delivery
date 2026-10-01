"""
Run the alignment scheme from the uniform start for a fixed number of iterations.

The alignment is obtained by iterating the two-half-step scheme from a uniform starting matrix in which every entry carries the same mass. The iteration is deterministic, so the result depends only on the inputs and the number of steps, and no convergence test is applied: the loop runs a prescribed number of times. Because each iteration performs a second multiplicative update after the row rescaling and only afterwards rescales the columns, the returned alignment respects the column budgets exactly but may exceed the row budgets slightly. That is a property of the scheme rather than an error, and restoring row feasibility at the end would change every quantity computed from the alignment.

Returns
-------
list of lists of floats, the alignment after the prescribed number of iterations, each entry rounded to 6 decimal places
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def align(A1: list[list[int]], A2: list[list[int]], C: list[list[float]],
          alpha: float, eps: float, gamma: float,
          mu: list[float], nu: list[float], iterations: int) -> list[list[float]]:
    '''Iterate the alignment scheme from the uniform start.
    Parameters
    ----------
    A1, A2 : list[list[int]]
        Undirected adjacency of the first and second web.
    C : list[list[float]]
        Dissimilarity matrix between the two webs.
    alpha, eps, gamma : float
        Tradeoff parameter, self-alignment parameter and step size.
    mu, nu : list[float]
        Species importance budgets of the first and second web.
    iterations : int
        Number of iterations to perform; no early stopping.
    Returns
    -------
    T : list[list[float]]
        Final alignment, each entry rounded to 6 decimal places.
    '''
    T = []
    return T  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_align(A1, A2, C, alpha, eps, gamma, mu, nu, iterations):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    m, n = len(C), len(C[0])
    T = [[1.0 / (m * n)] * n for _ in range(m)]
    for _ in range(iterations):
        T = _oracle_alignment_iteration(A1, A2, C, T, alpha, eps, gamma, mu, nu)
    return [[round(x, 6) for x in row] for row in T]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    setup = '''
A1 = [[0,1,0],[1,0,1],[0,1,0]]
A2 = [[0,1,0],[1,0,1],[0,1,0]]
CS = [[0.2,0.9,0.5],[0.8,0.1,0.6],[0.4,0.7,0.3]]
CE = [[0.0,1.0,1.0],[1.0,0.0,1.0],[1.0,1.0,0.0]]
MU = [1/3, 1/3, 1/3]
T0 = [[1/9]*3 for _ in range(3)]
TA = [[0.10,0.02,0.00],[0.00,0.12,0.03],[0.05,0.00,0.09]]
TB = [[0.08,0.00,0.04],[0.02,0.10,0.00],[0.00,0.06,0.07]]
TC = [[0.11,0.01,0.02],[0.03,0.09,0.00],[0.00,0.04,0.10]]
'''
    return [
        {   # normal: the full iteration count
            "setup": setup,
            "call": "align(A1, A2, CS, 0.7, 5.0, 1.0, MU, MU, 300)",
            "gold_call": "_oracle_align(A1, A2, CS, 0.7, 5.0, 1.0, MU, MU, 300)",
        },
        {   # boundary: a single iteration
            "setup": setup,
            "call": "align(A1, A2, CS, 0.7, 5.0, 1.0, MU, MU, 1)",
            "gold_call": "_oracle_align(A1, A2, CS, 0.7, 5.0, 1.0, MU, MU, 1)",
        },
        {   # edge: no mass reward, so mass decays
            "setup": setup,
            "call": "align(A1, A2, CE, 0.7, 0.0, 1.0, MU, MU, 50)",
            "gold_call": "_oracle_align(A1, A2, CE, 0.7, 0.0, 1.0, MU, MU, 50)",
        },
    ]

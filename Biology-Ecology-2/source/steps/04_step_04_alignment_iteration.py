"""
Apply one iteration of the proximal alignment scheme.

One iteration consists of two half-steps, and they are deliberately not symmetric. Each half-step forms a cost surrogate, multiplies the current alignment entrywise by the exponential of the negated surrogate divided by the step size, and then rescales to respect one of the two marginal budgets. The first surrogate weights the cost by the current alignment before conjugating by the two adjacency structures; the second conjugates first and applies the cost afterwards. Both add a share of the direct-matching cost and subtract a reward proportional to the product of the two importance weights, which is what allows mass to be left unassigned. The rescaling is a projection onto an inequality constraint, so a row or column already within its budget is left untouched rather than being normalised upward.

Returns
-------
list of lists of floats, the alignment after one iteration, each entry rounded to 9 decimal places
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def alignment_iteration(A1: list[list[int]], A2: list[list[int]],
                        C: list[list[float]], T: list[list[float]],
                        alpha: float, eps: float, gamma: float,
                        mu: list[float], nu: list[float]) -> list[list[float]]:
    '''One iteration of the two-half-step proximal alignment scheme.
    Parameters
    ----------
    A1, A2 : list[list[int]]
        Undirected adjacency of the first and second web.
    C : list[list[float]]
        Dissimilarity matrix between the two webs.
    T : list[list[float]]
        Current alignment.
    alpha, eps, gamma : float
        Tradeoff parameter, self-alignment parameter and step size.
    mu, nu : list[float]
        Species importance budgets of the first and second web.
    Returns
    -------
    T_next : list[list[float]]
        Updated alignment, each entry rounded to 9 decimal places.
    '''
    T_next = []
    return T_next  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math
def _oracle_alignment_iteration(A1, A2, C, T, alpha, eps, gamma, mu, nu):
    m, n = len(C), len(C[0])
    def matmul(X, Y):
        p, q, r = len(X), len(Y), len(Y[0])
        return [[sum(X[i][k] * Y[k][j] for k in range(q)) for j in range(r)]
                for i in range(p)]
    def surrogate(M, elementwise_cost):
        if elementwise_cost:
            inner = matmul(matmul(A1, M), A2)
            core = [[C[i][j] * inner[i][j] for j in range(n)] for i in range(m)]
        else:
            core = matmul(matmul(A1, [[C[i][j] * M[i][j] for j in range(n)]
                                      for i in range(m)]), A2)
        return [[alpha * core[i][j] + 0.5 * (1 - alpha) * C[i][j]
                 - 0.5 * eps * mu[i] * nu[j] for j in range(n)] for i in range(m)]
    Q = surrogate(T, False)
    H = [[T[i][j] * math.exp(-Q[i][j] / gamma) for j in range(n)] for i in range(m)]
    rf = [min(mu[i] / s, 1.0) if (s := sum(H[i])) > 0 else 1.0 for i in range(m)]
    H = [[H[i][j] * rf[i] for j in range(n)] for i in range(m)]
    Qp = surrogate(H, True)
    H = [[H[i][j] * math.exp(-Qp[i][j] / gamma) for j in range(n)] for i in range(m)]
    cf = [min(nu[j] / s, 1.0) if (s := sum(H[i][j] for i in range(m))) > 0 else 1.0
          for j in range(n)]
    return [[round(H[i][j] * cf[j], 9) for j in range(n)] for i in range(m)]

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
        {   # normal: one step from the uniform start
            "setup": setup,
            "call": "alignment_iteration(A1, A2, CS, T0, 0.7, 5.0, 1.0, MU, MU)",
            "gold_call": "_oracle_alignment_iteration(A1, A2, CS, T0, 0.7, 5.0, 1.0, MU, MU)",
        },
        {   # boundary: alpha zero, direct matching only
            "setup": setup,
            "call": "alignment_iteration(A1, A2, CS, T0, 0.0, 5.0, 1.0, MU, MU)",
            "gold_call": "_oracle_alignment_iteration(A1, A2, CS, T0, 0.0, 5.0, 1.0, MU, MU)",
        },
        {   # edge: alpha one and no mass reward
            "setup": setup,
            "call": "alignment_iteration(A1, A2, CE, T0, 1.0, 0.0, 1.0, MU, MU)",
            "gold_call": "_oracle_alignment_iteration(A1, A2, CE, T0, 1.0, 0.0, 1.0, MU, MU)",
        },
    ]

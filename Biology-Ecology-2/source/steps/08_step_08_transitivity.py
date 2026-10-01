"""
Measure how much of a species' alignment mass closes a triangle across two other webs.

A species of the focal web is aligned into two other webs at once. Each pair consisting of one image in the first and one in the second either is itself aligned or is not, and the triangle closes only when it is. The transitivity score is the share of the species' paired alignment mass that closes in this sense: the numerator accumulates the product of the two outgoing strengths weighted by the strength linking the two images, and the denominator accumulates the same products without that weighting. The ratio therefore lies between zero and one and asks whether the correspondences a species participates in are mutually consistent, rather than merely strong. A species carrying no paired mass has no ratio at all, which is a degenerate case rather than a score of zero.

Returns
-------
float, the share of the species' paired alignment mass that closes a triangle, rounded to 6 decimal places
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transitivity(T_ip: list[list[float]], T_iq: list[list[float]],
                 T_pq: list[list[float]], j: int) -> float:
    '''Share of a species' paired alignment mass that closes a triangle.
    Parameters
    ----------
    T_ip : list[list[float]]
        Alignment from the focal web to the first other web.
    T_iq : list[list[float]]
        Alignment from the focal web to the second other web.
    T_pq : list[list[float]]
        Alignment between the two other webs.
    j : int
        Species of the focal web being scored.
    Returns
    -------
    score : float
        Closing share, rounded to 6 decimal places.
    '''
    score = 0.0
    return score  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_transitivity(T_ip, T_iq, T_pq, j):
    num = den = 0.0
    for a, x in enumerate(T_ip[j]):
        if x == 0.0:
            continue
        for b, y in enumerate(T_iq[j]):
            if y == 0.0:
                continue
            den += x * y
            num += x * y * T_pq[a][b]
    if den <= 0.0:
        raise ValueError("no alignment mass at species %d: ratio undefined" % j)
    return round(num / den, 6)

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
        {   # normal: a species with mass into both webs
            "setup": setup,
            "call": "transitivity(TA, TB, TC, 0)",
            "gold_call": "_oracle_transitivity(TA, TB, TC, 0)",
        },
        {   # normal: a different species
            "setup": setup,
            "call": "transitivity(TA, TB, TC, 2)",
            "gold_call": "_oracle_transitivity(TA, TB, TC, 2)",
        },
        {   # boundary: every image pair aligned, so the share is one
            "setup": setup,
            "call": "transitivity(TA, TB, [[1.0]*3 for _ in range(3)], 1)",
            "gold_call": "_oracle_transitivity(TA, TB, [[1.0]*3 for _ in range(3)], 1)",
        },
    ]

"""
Evaluate total and event-specific endpoint weights for a selected two-site substitution in an ordered mutant background.

A finite interaction is the change in one substitution's effect after the other substitution changes its background. Exact ordered recombination therefore uses inside and outside quantities from that altered background for both the total ensemble and any selected local-event subset.

Returns
-------
np.ndarray: shape (2, 4), log total and p-as-5'-pair weights for reference, p, q and double endpoints.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def evaluate_ordered_double_endpoints(sequence: str, rule_weights: "np.ndarray",
                                      unpaired: "np.ndarray", pair_factors: "np.ndarray",
                                      log_inside: "np.ndarray", channel_logs: "np.ndarray",
                                      position: int, nucleotide: str,
                                      partner_position: int,
                                      partner_nucleotide: str) -> "np.ndarray":
    """Return log endpoint weights for a selected two-site replacement.

    Let p = ``position`` receive c = ``nucleotide`` and q =
    ``partner_position`` receive d = ``partner_nucleotide``. Row 0 contains
    natural logs of total weights for [reference, p-only, q-only, double]. Row
    1 contains the corresponding logs of the unnormalized event weight in
    which p is the 5' base of a pair. Evaluate the double endpoint by exact
    one-site recombination at p in the background already carrying q to d.
    ``log_inside`` and ``channel_logs`` are reference outputs from steps 1 and
    3. Store ``-inf`` for a zero event weight.

    Returns
    -------
    np.ndarray
        Float array of shape (2, 4), with endpoint columns ordered as above.

    Raises
    ------
    ValueError
        If the grammar or reference outputs are invalid, positions are not
        distinct integers in range, or a replacement is not a one-character
        RNA base different from the base already present.
    """
    return log_endpoints  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_evaluate_ordered_double_endpoints(sequence: str, rule_weights: "np.ndarray",
                                              unpaired: "np.ndarray", pair_factors: "np.ndarray",
                                              log_inside: "np.ndarray", channel_logs: "np.ndarray",
                                              position: int, nucleotide: str,
                                              partner_position: int,
                                              partner_nucleotide: str) -> "np.ndarray":
    x, w, u, pf = _validate_rna_grammar(sequence, rule_weights, unpaired, pair_factors)
    n = len(x)
    inside = np.asarray(log_inside, dtype=float)
    channels = np.asarray(channel_logs, dtype=float)
    if inside.shape != (n + 1, n + 1) or channels.shape != (n, 4, 4) \
            or np.any(np.isnan(inside)) or np.any(np.isposinf(inside)) \
            or np.any(np.isnan(channels)) or np.any(np.isposinf(channels)) \
            or not np.isfinite(inside[0, n]):
        raise ValueError("reference log outputs have the wrong shape or invalid values")
    for site, base in ((position, nucleotide), (partner_position, partner_nucleotide)):
        if isinstance(site, bool) or not isinstance(site, (int, np.integer)) \
                or not 0 <= site < n or not isinstance(base, str) or len(base) != 1 \
                or base not in "ACGU" or base == sequence[int(site)]:
            raise ValueError("each replacement must change a valid position to one RNA base")
    if int(position) == int(partner_position):
        raise ValueError("the two positions must be distinct")

    p, q = int(position), int(partner_position)
    c, d = "ACGU".index(nucleotide), "ACGU".index(partner_nucleotide)
    xp = x[p]
    totals = np.logaddexp.reduce(channels, axis=2)
    unchanged = totals[np.arange(n), x]
    if not np.all(np.isfinite(unchanged)) \
            or not np.allclose(unchanged, inside[0, n], rtol=0.0, atol=1e-9):
        raise ValueError("reference inside and replacement-channel outputs are inconsistent")

    q_background = sequence[:q] + partner_nucleotide + sequence[q + 1:]
    q_inside = _oracle_compute_log_inside_table(q_background, w, u, pf)
    q_outside = _oracle_compute_log_outside_table(q_background, w, u, pf, q_inside)
    q_channels = _oracle_evaluate_single_replacement_channels(
        q_background, w, u, pf, q_inside, q_outside
    )
    q_totals = np.logaddexp.reduce(q_channels, axis=2)
    endpoint_logs = np.array([
        [inside[0, n], totals[p, c], totals[q, d], q_totals[p, c]],
        [channels[p, xp, 0], channels[p, c, 0], q_channels[p, xp, 0], q_channels[p, c, 0]],
    ], dtype=float)
    if np.any(endpoint_logs[1] > endpoint_logs[0] + 1e-9):
        raise ValueError("an event endpoint cannot exceed its corresponding total endpoint")
    return endpoint_logs

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases() -> list[dict[str, str]]:
    """Return scalar-only test specifications."""
    base = (
        "import numpy as np\nw=np.array([0.7,0.44,0.36,0.24,0.8]); u=np.array([[0.28,0.19,0.27,0.26],[0.24,0.26,0.21,0.29]])\n"
        "P=np.zeros((4,4)); P[0,3]=P[3,0]=1.1; P[1,2]=P[2,1]=1.9; P[2,3]=0.5; P[3,2]=0.7\n"
        "def _inputs(s):\n    x=np.array(['ACGU'.index(c) for c in s]); n=len(s); t=np.diag(np.full(n+1,w[4])); o=np.zeros_like(t); o[0,n]=1.0\n    for z in range(1,n+1):\n        for i in range(n-z+1):\n            k=i+z; t[i,k]=w[1]*u[0,x[i]]*t[i+1,k]+w[2]*u[1,x[k-1]]*t[i,k-1]\n            if z>1: t[i,k]+=w[0]*P[x[i],x[k-1]]*t[i+1,k-1]+w[3]*float(np.dot(t[i,i+1:k],t[i+1:k,k]))\n    for z in range(n,0,-1):\n        for i in range(n-z+1):\n            k=i+z; b=o[i,k]; o[i+1,k]+=b*w[1]*u[0,x[i]]; o[i,k-1]+=b*w[2]*u[1,x[k-1]]\n            if z>1: o[i+1,k-1]+=b*w[0]*P[x[i],x[k-1]]; o[i,i+1:k]+=b*w[3]*t[i+1:k,k]; o[i+1:k,k]+=b*w[3]*t[i,i+1:k]\n    R=np.zeros((n,4,4))\n    for p in range(n):\n        for c in range(4):\n            R[p,c,0]=sum(o[p,k]*w[0]*P[c,x[k-1]]*t[p+1,k-1] for k in range(p+2,n+1)); R[p,c,1]=sum(o[i,p+1]*w[0]*P[x[i],c]*t[i+1,p] for i in range(p)); R[p,c,2]=w[1]*u[0,c]*sum(o[p,k]*t[p+1,k] for k in range(p+1,n+1)); R[p,c,3]=w[2]*u[1,c]*sum(o[i,p+1]*t[i,p] for i in range(p+1))\n    L=np.full_like(t,-np.inf); C=np.full_like(R,-np.inf); L[t>0]=np.log(t[t>0]); C[R>0]=np.log(R[R>0]); return L,C\n"
        "def _pin(v):\n    a=np.asarray(v,dtype=float); q=np.where(np.isfinite(a),a,-700.0).ravel(); r=np.arange(1.0,q.size+1.0); return float(np.sum(np.sin(.31*r)*q)+np.sum(np.cos(.13*r)*q*q))\n"
        "def _status(fn):\n    try:\n        fn()\n        return 0\n    except ValueError:\n        return 1\n    except Exception:\n        return 2\n"
    )
    candidate = "evaluate_ordered_double_endpoints(s,w.copy(),u.copy(),P.copy(),L.copy(),C.copy()"
    oracle = "_oracle_evaluate_ordered_double_endpoints(s,w.copy(),u.copy(),P.copy(),L.copy(),C.copy()"
    return [
        {"setup": base + "s='GCAUGGUACGU'; L,C=_inputs(s)\n", "call": "_pin(" + candidate + ",2,'G',10,'C'))",
         "gold_call": "_pin(" + oracle + ",2,'G',10,'C'))"},
        {"setup": base + "s='GCAUGGUACGU'; L,C=_inputs(s)\n", "call": "_pin(" + candidate + ",0,'A',1,'U'))",
         "gold_call": "_pin(" + oracle + ",0,'A',1,'U'))"},
        {"setup": base + "w=np.array([0.5,0.3,0.1,0.2,1.3]); P[2,3]=0.9; s='GGUUCCACAGUAGCUU'; L,C=_inputs(s)\n",
         "call": "_pin(" + candidate + ",0,'C',15,'A'))",
         "gold_call": "_pin(" + oracle + ",0,'C',15,'A'))"},
        {"setup": base + "s='GCAUGGUACGU'; L,C=_inputs(s)\n", "call": "_status(lambda: " + candidate + ",2,'AC',10,'C'))",
         "gold_call": "_status(lambda: " + oracle + ",2,'AC',10,'C'))"},
    ]

"""
Chain the sub-problem functions 01-09 end to end on the tetraploid fragment testbed and return the block-adjusted minimum error correction of the assembly they produce.

The whole measurement is one pass: build the SNP line graph from the fragments, put a genotype-constrained phasing distribution on it, decode one phasing per vertex, stitch those local phasings into a global assembly variant by variant, and score the result against the fragments that produced it. The returned number is what that assembly costs in read allele corrections per called allele.

Returns
-------
float: the block-adjusted minimum error correction of the assembled haplotypes, as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_phapcompass_short_pipeline(fragments: tuple = (
        "0-10-----------", "00111----------", "011------------",
        "0110-----------", "10100----------", "1100-----------",
        "1110-----------", "11110----------", "-010-----------",
        "-111-----------", "--110----------", "-----00000-----",
        "-----0011------", "-----0101------", "-----100-------",
        "-----100-------", "-----1010------", "-----11101-----",
        "-----11111-----", "------0101-----", "------100------",
        "------1001-----", "----------00-0-", "----------000--",
        "----------0000-", "----------0000-", "----------1100-",
        "-----------000-", "-----------100-", "-----------111-",
        "--------------1", "--------------0", "--------------1"),
        genotypes: tuple = (3, 3, 2, 1, 1, 3, 3, 2, 2, 3, 2, 3, 1, 1, 2),
        ploidy: int = 4,
        error_rate: float = 0.02,
        likelihood_weight: float = 1.0 / 12.0,
        mec_weight: float = 10.0 / 12.0,
        agreement_weight: float = 1.0 / 12.0) -> float:
    """Assemble a polyploid haplotype set from fragments and score the result.

    Parameters
    ----------
    fragments : tuple
        Sequence of equal-length strings over the alphabet {'0', '1', '-'}, one
        per aligned fragment, giving the reference allele, the alternate allele
        or no call at each of the heterozygous SNP positions in order.
    genotypes : tuple
        Sequence of integers of the same length as each fragment, holding the
        called count of alternate alleles at each SNP.
    ploidy : int
        Number of haplotypes to reconstruct (ploidy >= 1).
    error_rate : float
        Per-base sequencing error rate, strictly between zero and one.
    likelihood_weight : float
        Non-negative weight of the rescaled read-likelihood term in the
        candidate score.
    mec_weight : float
        Non-negative weight of the rescaled error-correction term, which enters
        the candidate score with a negative sign.
    agreement_weight : float
        Non-negative weight of the rescaled inference-agreement term.

    Returns
    -------
    mec : float
        The block-adjusted minimum error correction of the reconstructed
        assembly, as a native Python float.

    Raises
    ------
    ValueError
        If ``fragments`` is not a non-empty sequence of equal-length strings
        over {'0', '1', '-'}, if ``genotypes`` does not hold one integer per SNP
        in the range from zero to ``ploidy`` inclusive, if ``ploidy`` is not an
        integer greater than zero, if ``error_rate`` is not a finite number
        strictly between zero and one, or if any of the three weights is not a
        finite non-negative number.
    """
    return mec  # placeholder

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_run_phapcompass_short_pipeline(fragments: tuple = (
        "0-10-----------", "00111----------", "011------------",
        "0110-----------", "10100----------", "1100-----------",
        "1110-----------", "11110----------", "-010-----------",
        "-111-----------", "--110----------", "-----00000-----",
        "-----0011------", "-----0101------", "-----100-------",
        "-----100-------", "-----1010------", "-----11101-----",
        "-----11111-----", "------0101-----", "------100------",
        "------1001-----", "----------00-0-", "----------000--",
        "----------0000-", "----------0000-", "----------1100-",
        "-----------000-", "-----------100-", "-----------111-",
        "--------------1", "--------------0", "--------------1"),
        genotypes: tuple = (3, 3, 2, 1, 1, 3, 3, 2, 2, 3, 2, 3, 1, 1, 2),
        ploidy: int = 4,
        error_rate: float = 0.02,
        likelihood_weight: float = 1.0 / 12.0,
        mec_weight: float = 10.0 / 12.0,
        agreement_weight: float = 1.0 / 12.0) -> float:
    # Local imports keep the oracle self-contained when the harness executes it
    # in isolation.
    import numpy as np

    # -- Validate the orchestrator inputs.
    if isinstance(ploidy, bool) or not isinstance(ploidy, (int, np.integer)) or int(ploidy) < 1:
        raise ValueError("ploidy must be an integer greater than zero")
    ploidy = int(ploidy)
    if (isinstance(error_rate, bool)
            or not isinstance(error_rate, (int, float, np.floating, np.integer))
            or not np.isfinite(error_rate) or not 0.0 < float(error_rate) < 1.0):
        raise ValueError("error_rate must be a finite number strictly between zero and one")
    weights = np.array([likelihood_weight, mec_weight, agreement_weight], dtype=float)
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("the three weights must be finite and non-negative")

    rows = list(fragments)
    if len(rows) < 1 or not all(isinstance(row, str) for row in rows):
        raise ValueError("fragments must be a non-empty sequence of strings")
    n_snps = len(rows[0])
    if n_snps < 1 or any(len(row) != n_snps for row in rows):
        raise ValueError("every fragment must be a non-empty string of the same length")
    if any(character not in "01-" for row in rows for character in row):
        raise ValueError("fragments may only contain the characters 0, 1 and -")
    reads = np.array([[-1 if c == "-" else int(c) for c in row] for row in rows], dtype=int)

    calls = np.asarray(genotypes, dtype=float)
    if calls.ndim != 1 or calls.size != n_snps:
        raise ValueError("genotypes must hold one entry per SNP")
    if not np.allclose(calls, np.round(calls), rtol=0.0, atol=1e-12):
        raise ValueError("genotypes entries must be integer valued")
    calls = np.round(calls).astype(int)
    if np.any(calls < 0) or np.any(calls > ploidy):
        raise ValueError("genotypes entries must lie between zero and ploidy inclusive")

    # -- Sub-problem 03: the vertices of the SNP line graph, in genomic order.
    nodes = _oracle_build_snp_line_graph(reads)
    n_vertices = int(nodes.shape[0])

    # -- Sub-problems 01 and 02: the phasing state space and read-evidence
    #    potential of every vertex.
    node_phasings = [_oracle_enumerate_valid_phasings(ploidy, calls[nodes[t]])
                     for t in range(n_vertices)]
    node_potentials = [_oracle_compute_phasing_potentials(reads, node_phasings[t],
                                                          nodes[t], error_rate)
                       for t in range(n_vertices)]

    # -- Sub-problems 01, 02 and 04: every pair of vertices sharing exactly one
    #    SNP spans a three-SNP run whose potential becomes a transition.
    edge_list = []
    transitions = []
    for parent in range(n_vertices):
        for child in range(parent + 1, n_vertices):
            shared = set(nodes[parent].tolist()) & set(nodes[child].tolist())
            if len(shared) != 1:
                continue
            joint = np.array(sorted(set(nodes[parent].tolist()) | set(nodes[child].tolist())),
                             dtype=int)
            joint_phasings = _oracle_enumerate_valid_phasings(ploidy, calls[joint])
            joint_potentials = _oracle_compute_phasing_potentials(reads, joint_phasings,
                                                                  joint, error_rate)
            transitions.append(_oracle_build_transition_matrix(
                nodes[parent], node_phasings[parent], nodes[child], node_phasings[child],
                joint, joint_phasings, joint_potentials))
            edge_list.append((parent, child))
    edges = np.array(edge_list, dtype=int).reshape(-1, 2)

    # -- Sub-problem 05: one decoded phasing per vertex.
    states = _oracle_decode_map_phasings(nodes, node_potentials, edges, transitions)

    # -- Sub-problems 06, 07 and 08: stitch the local phasings into one global
    #    assembly, one variant at a time.
    haplotypes = -np.ones((ploidy, n_snps), dtype=int)
    blocks = np.zeros(n_snps, dtype=int)
    node_phased = np.zeros(n_vertices, dtype=int)
    position_phased = np.zeros(n_snps, dtype=int)
    block_id = 0
    while True:
        move = _oracle_select_next_position(nodes, node_phased, position_phased)
        action = int(move[0])
        if action == 2:
            break
        if action == 1:
            vertex = int(move[2])
            block_id += 1
            phasing = node_phasings[vertex][int(states[vertex])]
            for column, snp in enumerate(nodes[vertex].tolist()):
                if not position_phased[snp]:
                    haplotypes[:, snp] = phasing[:, column]
                    position_phased[snp] = 1
                    blocks[snp] = block_id
            node_phased[vertex] = 1
            continue
        target = int(move[1])
        frontier = np.flatnonzero(move[3:] > 0)
        touching = [t for t in frontier.tolist() if target in nodes[t].tolist()]
        window = np.array(sorted({int(snp) for t in touching
                                  for snp in nodes[t].tolist()}), dtype=int)
        candidates = _oracle_generate_position_candidates(window, haplotypes, calls)
        neighbour_phasings = np.array([node_phasings[t][int(states[t])] for t in touching],
                                      dtype=int)
        scores = _oracle_score_phasing_candidates(candidates, window, reads,
                                                  nodes[touching], neighbour_phasings,
                                                  error_rate, weights)
        best = candidates[int(np.argmax(scores))]
        haplotypes[:, target] = best[:, int(np.flatnonzero(window == target)[0])]
        position_phased[target] = 1
        blocks[target] = block_id
        for t in touching:
            node_phased[t] = 1

    # -- Sub-problem 09: score the finished assembly against the fragments.
    return float(_oracle_compute_block_adjusted_mec(reads, haplotypes, blocks, ploidy))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Integration: the reported tetraploid testbed (normal scenario) ---
        {
            "setup": """import numpy as np
""",
            "call": "round(run_phapcompass_short_pipeline(), 10)",
            "gold_call": "round(_oracle_run_phapcompass_short_pipeline(), 10)",
        },
        # --- Integration: the same fragments scored on the likelihood alone,
        #     which changes the assembly and therefore the reported cost ---
        {
            "setup": """import numpy as np
""",
            "call": "round(run_phapcompass_short_pipeline(likelihood_weight=1.0, mec_weight=0.0, agreement_weight=0.0), 10)",
            "gold_call": "round(_oracle_run_phapcompass_short_pipeline(likelihood_weight=1.0, mec_weight=0.0, agreement_weight=0.0), 10)",
        },
        # --- Integration: a much larger assumed error rate, which flattens the
        #     potentials without changing the graph ---
        {
            "setup": """import numpy as np
""",
            "call": "round(run_phapcompass_short_pipeline(error_rate=0.2), 10)",
            "gold_call": "round(_oracle_run_phapcompass_short_pipeline(error_rate=0.2), 10)",
        },
        # --- Integration (boundary): a small diploid fragment set, which
        #     exercises every step at the lowest ploidy the model allows ---
        {
            "setup": """import numpy as np
frags = ("0101----", "1010----", "01-1----", "----1100", "----0011", "----11-0",
         "--0101--", "--1010--")
gt = (1, 1, 1, 1, 1, 1, 1, 1)
""",
            "call": "round(run_phapcompass_short_pipeline(frags, gt, 2, 0.02), 10)",
            "gold_call": "round(_oracle_run_phapcompass_short_pipeline(frags, gt, 2, 0.02), 10)",
        },
        # --- Integration (edge): a fragment set that joins no pair of SNPs, so
        #     the graph is empty and nothing can be phased at all ---
        {
            "setup": """import numpy as np
frags = ("1---", "-0--", "--1-", "---0")
gt = (2, 2, 2, 2)
""",
            "call": "round(run_phapcompass_short_pipeline(frags, gt, 4, 0.02), 10)",
            "gold_call": "round(_oracle_run_phapcompass_short_pipeline(frags, gt, 4, 0.02), 10)",
        },
        # --- Invalid: a fragment carrying a character outside the alphabet ---
        {
            "setup": """import numpy as np
frags = ("01N1----", "1010----")
gt = (1, 1, 1, 1, 1, 1, 1, 1)
def run_model():
    try:
        run_phapcompass_short_pipeline(frags, gt, 2, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_phapcompass_short_pipeline(frags, gt, 2, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a genotype call exceeding the ploidy ---
        {
            "setup": """import numpy as np
frags = ("0101", "1010")
gt = (1, 1, 1, 5)
def run_model():
    try:
        run_phapcompass_short_pipeline(frags, gt, 2, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_phapcompass_short_pipeline(frags, gt, 2, 0.02)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        # --- Invalid: a negative candidate-scoring weight ---
        {
            "setup": """import numpy as np
def run_model():
    try:
        run_phapcompass_short_pipeline(mec_weight=-1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_run_phapcompass_short_pipeline(mec_weight=-1.0)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]

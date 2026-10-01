"""
Run the full ancestral recombination graph mixed model scan and return the score

statistic reported for the variant under test.

The pipeline turns a set of per-chromosome genealogies into a single association

score: relatedness built from the mutations the graphs carry, a moment-based fit

of the two variance components, a polygenic prediction from every chromosome

except the one under test, and a score for one variant against the phenotype

that prediction leaves behind. Holding the focal chromosome out of the fit is

what keeps the variant being tested from also being used to correct for itself.

Returns
-------
float, the score statistic reported for the variant under test as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def run_full_pipeline(
    chromosome_edges: list,
    chromosome_mutations: list,
    chromosome_spans: list,
    haplotype_pairs: list,
    phenotype: list,
    alpha: float = -0.25,
    min_minor_allele_count: int = 2,
    num_probes: int = 200,
    probe_seed: int = 101,
    focal_index: int = 0,
    focal_variant: int = 0,
    tolerance: float = 1e-5,
    max_iterations: int = 6,
) -> float:
    """Run the whole scan and report the score of one variant.

    Args:
        chromosome_edges: list of per-chromosome edge lists, each entry a
            (child, parent, left, right) tuple.
        chromosome_mutations: list of per-chromosome mutation lists, each entry
            a (child node of the carrying edge, position) tuple.
        chromosome_spans: right end of the coordinate range of each chromosome.
        haplotype_pairs: list of [hap_a, hap_b] pairs of leaf indices, one per
            diploid individual, in the order the individuals are reported.
        phenotype: raw trait values, one per individual.
        alpha: frequency-dependent scaling exponent of the genotype weight.
        min_minor_allele_count: smallest retained minor allele count.
        num_probes: number of probe vectors.
        probe_seed: seed of the probe generator.
        focal_index: index of the chromosome under test.
        focal_variant: column of the focal chromosome's retained variants to
            score.
        tolerance: relative residual tolerance of the solve.
        max_iterations: step budget of the solve.

    Returns:
        float, the score statistic reported for the variant under test.

    Raises:
        ValueError: if the per-chromosome lists disagree in length, if
            focal_index does not name one of the chromosomes, or if
            focal_variant does not name one of its retained variants.
    """
    return statistic

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_full_pipeline(
    chromosome_edges: list,
    chromosome_mutations: list,
    chromosome_spans: list,
    haplotype_pairs: list,
    phenotype: list,
    alpha: float = -0.25,
    min_minor_allele_count: int = 2,
    num_probes: int = 200,
    probe_seed: int = 101,
    focal_index: int = 0,
    focal_variant: int = 0,
    tolerance: float = 1e-5,
    max_iterations: int = 6,
) -> float:
    """Reference implementation."""
    if not (len(chromosome_edges) == len(chromosome_mutations) == len(chromosome_spans)):
        raise ValueError("the per-chromosome lists disagree in length")
    if not 0 <= int(focal_index) < len(chromosome_edges):
        raise ValueError("focal_index does not name one of the chromosomes")

    blocks = []
    for edges, mutations, span in zip(
        chromosome_edges, chromosome_mutations, chromosome_spans
    ):
        genotypes, _clade_end = _oracle_arg_carrier_genotypes(
            edges, mutations, haplotype_pairs, span
        )
        blocks.append(_oracle_standardize_arg_genotypes(
            genotypes, alpha, min_minor_allele_count
        ))
    if not 0 <= int(focal_variant) < blocks[int(focal_index)].shape[1]:
        raise ValueError("focal_variant does not name a retained variant")

    pooled = np.hstack(blocks)
    grm, scale = _oracle_build_arg_grm(pooled)
    trace_r_squared = _oracle_hutchinson_trace_squared(
        pooled, scale, num_probes, probe_seed
    )
    sigma_g_squared, sigma_e_squared = _oracle_rhe_variance_component(
        grm, phenotype, trace_r_squared
    )
    solution, _steps = _oracle_conjugate_gradient_solve(
        blocks, focal_index, sigma_g_squared, sigma_e_squared, phenotype,
        tolerance, max_iterations,
    )
    prediction = _oracle_loco_covariance_product(
        blocks, focal_index, sigma_g_squared, 0.0, solution
    )
    residual = _oracle_blup_residual_phenotype(prediction, phenotype)
    return _oracle_grammar_gamma_statistic(
        blocks[int(focal_index)][:, int(focal_variant)], residual
    )

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    spans = [60.0, 45.0, 50.0]
    edges = [
        [
            (0, 20, 0, 20), (0, 21, 20, 40), (0, 20, 40, 60), (1, 20, 0, 60), (2, 21, 0, 60), (3, 21,
            0, 60), (4, 22, 0, 60), (5, 22, 0, 60), (6, 23, 0, 60), (7, 23, 0, 60), (8, 24, 0, 60),
            (9, 24, 0, 60), (10, 25, 0, 60), (11, 25, 0, 60), (12, 26, 0, 60), (13, 26, 0, 60), (14,
            27, 0, 60), (15, 27, 0, 60), (16, 28, 0, 60), (17, 28, 0, 60), (18, 29, 0, 60), (19, 29,
            0, 60), (20, 30, 0, 60), (21, 30, 0, 60), (22, 31, 0, 60), (23, 31, 0, 60), (24, 32, 0,
            60), (25, 32, 0, 60), (26, 33, 0, 60), (27, 33, 0, 60), (28, 34, 0, 60), (29, 34, 0, 60),
            (30, 35, 0, 60), (31, 35, 0, 60), (32, 36, 0, 40), (32, 37, 40, 60), (33, 36, 0, 60), (34,
            37, 0, 60), (35, 37, 0, 60), (36, 38, 0, 60), (37, 38, 0, 60)
        ],
        [
            (0, 20, 0, 45), (1, 20, 0, 45), (2, 21, 0, 45), (3, 21, 0, 45), (4, 22, 0, 45), (5, 22, 0,
            15), (5, 23, 15, 35), (5, 22, 35, 45), (6, 23, 0, 45), (7, 23, 0, 45), (8, 24, 0, 45), (9,
            24, 0, 45), (10, 25, 0, 45), (11, 25, 0, 45), (12, 26, 0, 35), (12, 28, 35, 45), (13, 26,
            0, 45), (14, 27, 0, 45), (15, 27, 0, 45), (16, 28, 0, 45), (17, 28, 0, 45), (18, 29, 0,
            45), (19, 29, 0, 45), (20, 30, 0, 45), (21, 30, 0, 45), (22, 31, 0, 45), (23, 31, 0, 45),
            (24, 32, 0, 45), (25, 32, 0, 45), (26, 33, 0, 45), (27, 33, 0, 45), (28, 34, 0, 45), (29,
            34, 0, 45), (30, 35, 0, 45), (31, 35, 0, 45), (32, 36, 0, 45), (33, 36, 0, 45), (34, 37,
            0, 45), (35, 37, 0, 45), (36, 38, 0, 45), (37, 38, 0, 45)
        ],
        [
            (0, 20, 0, 50), (1, 20, 0, 50), (2, 21, 0, 50), (3, 21, 0, 50), (4, 22, 0, 50), (5, 22, 0,
            50), (6, 23, 0, 50), (7, 23, 0, 50), (8, 24, 0, 50), (9, 24, 0, 50), (10, 25, 0, 50), (11,
            25, 0, 50), (12, 26, 0, 50), (13, 26, 0, 50), (14, 27, 0, 50), (15, 27, 0, 50), (16, 28,
            0, 50), (17, 28, 0, 50), (18, 29, 0, 10), (18, 28, 10, 30), (18, 29, 30, 50), (19, 29, 0,
            50), (20, 30, 0, 50), (21, 30, 0, 50), (22, 31, 0, 50), (23, 31, 0, 50), (24, 32, 0, 50),
            (25, 32, 0, 50), (26, 33, 0, 50), (27, 33, 0, 50), (28, 34, 0, 50), (29, 34, 0, 50), (30,
            35, 0, 30), (30, 37, 30, 50), (31, 35, 0, 50), (32, 36, 0, 50), (33, 36, 0, 50), (34, 37,
            0, 50), (35, 37, 0, 50), (36, 38, 0, 50), (37, 38, 0, 50)
        ],
    ]
    mutations = [
        [
            (30, 5), (20, 5), (21, 5), (35, 19.999), (35, 20), (21, 25), (36, 10), (37, 39.999), (37,
            40), (31, 45), (32, 30), (33, 50), (3, 12), (11, 33), (17, 55), (24, 52)
        ],
        [
            (31, 5), (22, 5), (23, 5), (35, 14.999), (35, 15), (23, 25), (36, 34.999), (36, 35), (34,
            40), (30, 20), (32, 8), (28, 38), (7, 22), (14, 41), (1, 30), (26, 44)
        ],
        [
            (34, 5), (29, 5), (28, 5), (36, 9.999), (36, 10), (28, 20), (37, 29.999), (37, 30), (35,
            15), (31, 35), (33, 45), (25, 40), (9, 25), (16, 48), (5, 18), (22, 12)
        ],
    ]
    pairs = [[0, 11], [1, 14], [2, 17], [3, 8], [4, 19], [5, 12], [6, 15], [7, 18], [9, 16], [10, 13]]
    trait = [12.26, 12.52, 12.42, 10.39, 9.6, 10.65, 11.57, 10.64, 12.51, 14.0]
    config = """spans = %r
edges = %r
mutations = %r
pairs = %r
trait = %r
""" % (spans, edges, mutations, pairs, trait)
    weak = [11.2, 11.9, 12.4, 11.05, 12.8, 10.4, 11.75, 12.2, 10.9, 11.6]
    args = "edges, mutations, spans, pairs, trait"
    return [
        {
            "setup": config,
            "call": "round(run_full_pipeline(%s), 12)" % args,
            "gold_call": "round(_oracle_run_full_pipeline(%s), 12)" % args,
        },
        {
            "setup": config + "weak = %r\n" % (weak,),
            "call": "round(run_full_pipeline(edges, mutations, spans, pairs, weak), 12)",
            "gold_call": "round(_oracle_run_full_pipeline(edges, mutations, spans, pairs, weak), 12)",
        },
        {
            "setup": config,
            "call": "round(run_full_pipeline(%s, focal_index=1), 12)" % args,
            "gold_call": "round(_oracle_run_full_pipeline(%s, focal_index=1), 12)" % args,
        },
        # A shorter step budget stops the solve further from the exact solution, so the
        # reported score moves with the budget rather than being insensitive to it.
        {
            "setup": config,
            "call": "round(run_full_pipeline(%s, max_iterations=3), 12)" % args,
            "gold_call": "round(_oracle_run_full_pipeline(%s, max_iterations=3), 12)" % args,
        },
        {
            "setup": config,
            "call": "round(run_full_pipeline(%s, focal_index=2, focal_variant=4, alpha=-1.0), 12)" % args,
            "gold_call": "round(_oracle_run_full_pipeline(%s, focal_index=2, focal_variant=4, alpha=-1.0), 12)" % args,
        },
        {
            "setup": config + """
def run_model():
    try:
        run_full_pipeline(edges, mutations, spans, pairs, trait, focal_index=5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_oracle():
    try:
        _oracle_run_full_pipeline(edges, mutations, spans, pairs, trait, focal_index=5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_oracle()",
        },
    ]

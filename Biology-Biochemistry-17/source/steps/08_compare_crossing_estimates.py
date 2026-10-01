"""
Run the whole comparison of the problem statement for the given numbers of discarded and stored trials, inverse temperature, seed and step cap: sample all 29 interface ensembles, and return 1e5 times the probability that a path leaving A which crosses x = -0.8 reaches B, estimated by the joint reweighting of the three sets, by the joint reweighting of sets 1 and 2, by set 1 alone, by the rescaled combination of the three sets, and exactly. The submitted function must import inside itself whatever it uses.

Every ensemble (s, k) is sampled with the generator seed (seed, s, k), and every ensemble contributes n_samples samples. The joint estimates of sets 1 and 2 and of set 1 use only the paths of those sets and their order parameters, in the layout of steps 4 and 5; the rescaled combination is that of step 6 and the exact value that of step 7. The benchmark configuration is n_equil = 100, n_samples = 5000, beta = 10, seed = 2026 and max_steps = 100000.

Returns
-------
return estimates
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compare_crossing_estimates(n_equil: int, n_samples: int, beta: float, seed: int,
                               max_steps: int) -> "np.ndarray":
    """1e5 x the crossing probability into B, given x = -0.8 is crossed, by five estimators.

    Args:
        n_equil: discarded trials per ensemble.
        n_samples: stored paths per ensemble.
        beta: inverse temperature.
        seed: first entry of every generator seed (seed, s, k).
        max_steps: step cap of each segment.

    Returns:
        np.ndarray: float array of length 5, 1e5 times the probability from the joint reweighting of sets
            1-3, the joint reweighting of sets 1-2, set 1 alone, the rescaled combination of sets 1-3,
            and the exact value.
    """
    return estimates

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compare_crossing_estimates(n_equil: int, n_samples: int, beta: float, seed: int,
                                       max_steps: int) -> "np.ndarray":
    """1e5 x [joint (3 sets), joint (sets 1-2), set 1 alone, rescaled (3 sets), exact]."""
    import numpy as np
    interfaces = [np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2]),
                  np.array([-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2]),
                  np.array([-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.2])]
    by_set = [np.concatenate([_oracle_sample_interface_ensemble(i + 1, k + 1, n_equil, n_samples, beta, seed, max_steps)
                              for k in range(len(interfaces[i]))]) for i in range(3)]
    counts = [np.full(len(lams), n_samples) for lams in interfaces]
    out = []
    for m in (3, 2, 1):
        mx = np.concatenate(by_set[:m])[:, :m]
        log_z = _oracle_joint_log_partition_sums(mx, interfaces[:m], counts[:m])
        out.append(np.exp(_oracle_crossing_probability(mx, interfaces[:m], counts[:m], log_z, -0.8, np.array([0.9]))[0]))
    out.append(np.exp(_oracle_rescaled_crossing_probability(by_set, interfaces, counts, -0.8)))
    out.append(np.exp(_oracle_exact_crossing_probability(beta, -0.8)))
    return 1e5 * np.array(out)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [
        {   # 8.1 short run at beta = 3
            "setup": "",
            "call": "compare_crossing_estimates(5, 20, 3.0, 1, 100000)",
            "gold_call": "_oracle_compare_crossing_estimates(5, 20, 3.0, 1, 100000)",
        },
        {   # 8.2 short run at beta = 6
            "setup": "",
            "call": "compare_crossing_estimates(10, 30, 6.0, 3, 100000)",
            "gold_call": "_oracle_compare_crossing_estimates(10, 30, 6.0, 3, 100000)",
        },
        {   # 8.3 boundary: very short equilibration at beta = 2.5
            "setup": "",
            "call": "compare_crossing_estimates(3, 15, 2.5, 11, 100000)",
            "gold_call": "_oracle_compare_crossing_estimates(3, 15, 2.5, 11, 100000)",
        },
    ]

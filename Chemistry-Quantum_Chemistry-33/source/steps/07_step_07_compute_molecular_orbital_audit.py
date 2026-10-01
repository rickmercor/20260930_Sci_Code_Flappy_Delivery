"""
This final orchestrator uses the default signature values shown below and default_rng(seed). Construct every seeded array and analytic weight exactly as specified in the problem statement. Form true antisymmetrized integrals g=0.25*(raw-raw.swapaxes(0,1)-raw.swapaxes(2,3)+raw.transpose(1,0,3,2)) and MP2=g/denominator. Call public Steps 1 through 6 in order. Compute all seventeen diagnostics using the definitions in the problem statement, including mean absolute valid scores and the absolute weighted T2 checksum. Compute the disclosed J, round only J to eight decimal places, and return J, diagnostics, T1, T2, and scores.

Steps 1 through 6 carry the paper-dependent representation, symmetry, locality, readout, and energy decisions. The seeded arrays, analytic weights, checksums, and final scalar reduction are disclosed benchmark conventions.

Returns
-------
tuple : J, seventeen diagnostics, T1, T2, and attention scores.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_molecular_orbital_audit(seed: int = 33027, atom_count: int = 6, occupied_count: int = 3, virtual_count: int = 3, radial_channels: int = 4, hidden_channels: int = 4, heads: int = 3, cutoff: float = 2.25, epsilon: float = 1e-8) -> tuple:
    """Return J, exactly seventeen named diagnostics, T1, T2, and scores.

    Construct the fully disclosed deterministic fixture and call public Steps 1
    through 6 in order. Only J is rounded, to eight decimal places.

    Parameters
    ----------
    seed : int
        Seed for default_rng.
    atom_count : int
        Number of atoms, from 3 through 9.
    occupied_count, virtual_count : int
        Positive occupied and virtual counts, each at most 5.
    radial_channels, hidden_channels : int
        Channel counts, each from 2 through 6.
    heads : int
        Attention heads, from 1 through 4.
    cutoff, epsilon : float
        Positive finite cutoff and stabilizer.

    Returns
    -------
    j : float
        Audit score rounded to eight decimal places.
    diagnostics : dict
        Exactly amplitude_norm, attention_norm, contraction_checksum,
        correlation_energy, correction_ratio, cross_score_max, directed_edges,
        doubles_energy, embedding_checksum, exchange_residual, pair_norm,
        padding_zero, singles_energy, state_norm, t1_checksum, t2_checksum,
        and valid_score_mean.
    t1 : np.ndarray
        Shape (occupied_count,virtual_count).
    t2 : np.ndarray
        Shape (occupied_count,occupied_count,virtual_count,virtual_count).
    scores : np.ndarray
        Shape (heads,P,P), where P=occupied_count+virtual_count.

    Raises
    ------
    ValueError
        If a configuration value or upstream dependency is invalid.
    """
    return None, None, None, None, None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_compute_molecular_orbital_audit(seed: int = 33027, atom_count: int = 6, occupied_count: int = 3, virtual_count: int = 3, radial_channels: int = 4, hidden_channels: int = 4, heads: int = 3, cutoff: float = 2.25, epsilon: float = 1e-8) -> tuple:
    integer_values = (atom_count, occupied_count, virtual_count, radial_channels, hidden_channels, heads)
    if not all(isinstance(z, (int, np.integer)) for z in integer_values):
        raise ValueError("count parameters must be integers")
    if not 3 <= atom_count <= 9 or not 1 <= occupied_count <= 5 or not 1 <= virtual_count <= 5:
        raise ValueError("atom and orbital counts are outside their supported ranges")
    if not 2 <= radial_channels <= 6 or not 2 <= hidden_channels <= 6 or not 1 <= heads <= 4:
        raise ValueError("channel or head count is outside its supported range")
    if not np.isfinite(cutoff) or cutoff <= 0.0 or not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("cutoff and epsilon must be positive and finite")
    rng = np.random.default_rng(int(seed))
    orbital_count = occupied_count + virtual_count
    magnetic_channels = 3
    coefficients = rng.normal(0.0, 0.42, size=(orbital_count, atom_count, radial_channels, magnetic_channels))
    shell_counts = 1 + ((3 * np.arange(atom_count) + 1) % radial_channels)
    embed_weights = np.fromfunction(lambda k, r: (np.sin((k + 1) * (r + 2) / 5.0) + 0.21 * np.cos((k + 2) * (r + 1) / 7.0)) / np.sqrt(radial_channels), (hidden_channels, radial_channels), dtype=float)
    padded, embedded, embedding_checksum = _oracle_embed_localized_orbitals(coefficients, shell_counts, embed_weights)
    fragment_ids = np.zeros(orbital_count, dtype=int)
    fragment_ids[(orbital_count + 1) // 2:] = 1
    q_weights = np.fromfunction(lambda h, j, k: 0.19 * np.sin((h + 1) * (j + 2) * (k + 1) / 9.0), (heads, hidden_channels, hidden_channels), dtype=float)
    k_weights = np.fromfunction(lambda h, j, k: 0.17 * np.cos((h + 2) * (j + 1) * (k + 2) / 11.0), (heads, hidden_channels, hidden_channels), dtype=float)
    v_weights = np.fromfunction(lambda h, j, k: 0.15 * np.sin((h + 3) * (j + 1) + (k + 2) / 8.0), (heads, hidden_channels, hidden_channels), dtype=float)
    head_weights = 0.5 + 0.2 * np.cos(np.arange(heads, dtype=float) + 1.0)
    mixed, scores, output_norms = _oracle_apply_signed_mo_attention(embedded, fragment_ids, q_weights, k_weights, v_weights, head_weights, epsilon)
    angle = 2.0 * np.pi * np.arange(atom_count, dtype=float) / atom_count
    radius = 1.15 + 0.08 * np.sin(3.0 * angle)
    positions = np.column_stack((radius * np.cos(angle), radius * np.sin(angle), 0.22 * np.sin(2.0 * angle + 0.3)))
    linear_weights = np.fromfunction(lambda j, k: 0.075 * np.cos((j + 1) * (k + 2) / 4.0), (hidden_channels, hidden_channels), dtype=float)
    cubic_weights = np.fromfunction(lambda j, k: 0.012 * np.sin((j + 2) * (k + 1) / 5.0), (hidden_channels, hidden_channels), dtype=float)
    updated, aggregate, directed_edges = _oracle_apply_odd_local_mixing(mixed, positions, cutoff, linear_weights, cubic_weights)
    pair_weights_1 = np.fromfunction(lambda j, k: 0.23 * np.cos((j + 1) * (k + 2) / 6.0), (hidden_channels, hidden_channels), dtype=float)
    hidden_width = hidden_channels + 2
    hidden_weights_1 = np.fromfunction(lambda j, h: 0.31 * np.sin((j + 2) * (h + 1) / 7.0), (hidden_channels, hidden_width), dtype=float)
    output_weights_1 = 0.18 * np.cos(np.arange(hidden_width, dtype=float) + 1.0)
    t1, t1_pairs, t1_checksum = _oracle_readout_single_amplitudes(updated, occupied_count, pair_weights_1, hidden_weights_1, output_weights_1, epsilon)
    orbital_energies = np.concatenate((-1.10 - 0.17 * np.arange(occupied_count), 0.25 + 0.21 * np.arange(virtual_count)))
    raw_integrals = rng.normal(0.0, 0.09, size=(occupied_count, occupied_count, virtual_count, virtual_count))
    integrals = 0.25 * (raw_integrals - raw_integrals.swapaxes(0, 1) - raw_integrals.swapaxes(2, 3) + raw_integrals.transpose(1, 0, 3, 2))
    denominators = orbital_energies[:occupied_count, None, None, None] + orbital_energies[None, :occupied_count, None, None] - orbital_energies[None, None, occupied_count:, None] - orbital_energies[None, None, None, occupied_count:]
    mp2 = integrals / denominators
    pair_weights_2 = np.fromfunction(lambda j, k: 0.20 * np.sin((j + 2) * (k + 1) / 5.0), (hidden_channels, hidden_channels), dtype=float)
    hidden_weights_2 = np.fromfunction(lambda j, h: 0.27 * np.cos((j + 1) * (h + 2) / 8.0), (hidden_channels, hidden_width), dtype=float)
    output_weights_2 = 0.14 * np.sin(np.arange(hidden_width, dtype=float) + 1.0)
    t2, correction, exchange_residual = _oracle_readout_double_amplitudes(updated, occupied_count, mp2, pair_weights_2, hidden_weights_2, output_weights_2, epsilon)
    energy, singles_energy, doubles_energy, amplitude_norm, contraction_checksum = _oracle_compute_cc_correlation_audit(t1, t2, integrals)
    cross = fragment_ids[:, None] != fragment_ids[None, :]
    cross_score_max = float(np.max(np.abs(scores[:, cross]))) if np.any(cross) else 0.0
    valid_score_mean = float(np.mean(np.abs(scores[:, ~cross])))
    t2_flat = t2.ravel(order="C")
    t2_checksum = float(np.dot(np.arange(1, t2_flat.size + 1, dtype=float), np.abs(t2_flat)) / t2_flat.size)
    correction_ratio = float(np.linalg.norm(correction) / (np.linalg.norm(mp2) + epsilon))
    state_norm = float(np.linalg.norm(updated))
    attention_norm = float(np.mean(output_norms))
    padding_zero = float(np.max(np.abs(padded * (np.arange(radial_channels)[None, None, :, None] >= shell_counts[None, :, None, None]))))
    diagnostics = {"amplitude_norm": amplitude_norm, "attention_norm": attention_norm, "contraction_checksum": contraction_checksum, "correlation_energy": energy, "correction_ratio": correction_ratio, "cross_score_max": cross_score_max, "directed_edges": float(directed_edges), "doubles_energy": doubles_energy, "embedding_checksum": float(embedding_checksum), "exchange_residual": exchange_residual, "pair_norm": float(np.linalg.norm(t1_pairs)), "padding_zero": padding_zero, "singles_energy": singles_energy, "state_norm": state_norm, "t1_checksum": float(t1_checksum), "t2_checksum": t2_checksum, "valid_score_mean": valid_score_mean}
    raw_j = float(np.exp(-abs(energy)) * (1.0 + valid_score_mean + 0.25 * correction_ratio) / (1.0 + amplitude_norm + 0.02 * state_norm + 0.01 * attention_norm))
    return float(np.round(raw_j, 8)), diagnostics, t1, t2, scores

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    keys = ('amplitude_norm','attention_norm','contraction_checksum','correlation_energy','correction_ratio','cross_score_max','directed_edges','doubles_energy','embedding_checksum','exchange_residual','pair_norm','padding_zero','singles_energy','state_norm','t1_checksum','t2_checksum','valid_score_mean')
    pack = lambda call: "(lambda z: [len(z),int(tuple(sorted(z[1]))==" + repr(keys) + "),*z[2].shape,*z[3].shape,*z[4].shape,*np.concatenate([[z[0]],np.asarray([z[1][k] for k in sorted(z[1])]),z[2].ravel(),z[3].ravel(),z[4].ravel()]).tolist()])(" + call + ")"
    err = "def et(fn,*a):\n try: fn(*a); return 0\n except ValueError: return 1\n"
    return [
        {"tol": 1e-9, "setup": "import numpy as np\na=(33027,6,3,3,4,4,3,2.25,1e-8)", "call": pack("compute_molecular_orbital_audit(*a)"), "gold_call": pack("_oracle_compute_molecular_orbital_audit(*a)")},
        {"tol": 1e-9, "setup": "import numpy as np\na=(33113,3,1,1,2,2,1,3.0,1e-7)", "call": pack("compute_molecular_orbital_audit(*a)"), "gold_call": pack("_oracle_compute_molecular_orbital_audit(*a)")},
        {"tol": 1e-9, "setup": "import numpy as np\na=(33221,8,4,2,6,5,4,1.9,1e-6)", "call": pack("compute_molecular_orbital_audit(*a)"), "gold_call": pack("_oracle_compute_molecular_orbital_audit(*a)")},
        {"tol": 0.0, "setup": err + "a=(33027,2,3,3,4,4,3,2.25,1e-8)", "call": "et(compute_molecular_orbital_audit,*a)", "gold_call": "et(_oracle_compute_molecular_orbital_audit,*a)"},
    ]

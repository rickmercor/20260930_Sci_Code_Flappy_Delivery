"""
Run the end-to-end audit. Build the network of the free duplex and the network of the protein-nucleic acid complex for the supplied configuration by calling the earlier steps in order, take the vibrational spectrum and the per-mode metrics of each network, keep only those modes whose collectivity reaches kappa_min, select among them the mode whose overlap with the first reference field is largest, and return the ratio of the selected eigenvalue of the complex to the selected eigenvalue of the free duplex. The two interaction cutoffs are the ones the source specifies for its two networks. The reference fields are the five probe fields (z^2, 0, 0), (0, z^2, 0), (0, 0, z), (-y, x, 0) and (x, y, 0), evaluated at the bead coordinates of the network under analysis with z measured from the mean height of all beads of that network (protein beads included for the complex) and x, y the bead coordinates themselves, each flattened bead-major into a 3M-vector and normalised; the first field is the bending reference. Raise ValueError if no mode of either network reaches kappa_min.

Binding a protein adds springs to a nucleic acid network and therefore raises the frequency of the deformation it restrains. The comparison is only meaningful between the same physical motion before and after binding, so the mode is chosen by its alignment with a reference deformation rather than by its position in the spectrum, and a collectivity floor keeps the comparison on global motions instead of the local rattling of individual interface beads.

Returns
-------
float: the ratio of the selected complex eigenvalue to the selected free-duplex eigenvalue
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def binding_stiffening_index(
        seq_a: str = "GAGCGCUCAG",
        geometry: tuple = (32.7, 3.10, 121.0, 8.91, 5.84, 2.444),
        prot_coords: tuple = ((-2.588, 9.659, 3.500), (-8.660, -5.000, 14.000),
                              (1.531, 3.696, -3.000), (3.222, -0.424, 31.000),
                              (-9.093, 5.250, 9.500), (-11.897, -1.566, 12.000),
                              (3.827, -9.239, 20.500), (-4.401, -10.625, 18.000)),
        prot_residues: str = "KRDESTLA",
        n_modes: int = 10,
        kappa_min: float = 0.52) -> float:
    """Run the end-to-end audit. Build the network of the free duplex and the network of
    the protein-nucleic acid complex for the supplied configuration by calling the
    earlier steps in order, take the vibrational spectrum and the per-mode metrics of
    each network, keep only those modes whose collectivity reaches kappa_min, select
    among them the mode whose overlap with the first reference field is largest, and
    return the ratio of the selected eigenvalue of the complex to the selected
    eigenvalue of the free duplex. The two interaction cutoffs are the ones the source
    specifies for its two networks. The reference fields are the five probe fields
    (z^2, 0, 0), (0, z^2, 0), (0, 0, z), (-y, x, 0) and (x, y, 0), evaluated at the bead
    coordinates of the network under analysis with z measured from the mean height of
    all beads of that network (protein beads included for the complex) and x, y the bead
    coordinates themselves, each flattened bead-major into a 3M-vector and normalised;
    the first field is the bending reference. Raise ValueError if no mode of either
    network reaches kappa_min.

    Parameters
    ----------
    seq_a : str
        strand A of the duplex, written 5'->3'.
    geometry : tuple of length 6
        helical twist per nucleotide in degrees, helical rise per nucleotide in
        angstrom, angular offset of strand B in degrees, and the radial
        distances of the phosphate, sugar and base beads in angstrom, in that
        order.
    prot_coords : array-like of shape (P, 3)
        protein Calpha coordinates in angstrom, appended after the nucleic-acid
        beads in the order given.
    prot_residues : str
        one one-letter residue code per protein bead, in the same order as
        prot_coords.
    n_modes : int
        how many of the lowest vibrational modes of each network to consider.
    kappa_min : float
        the collectivity degree a mode must reach to stay in the comparison.

    Returns
    -------
    float: the ratio of the selected complex eigenvalue to the selected free-duplex eigenvalue

    Raises
    ------
    ValueError
        if kappa_min does not lie strictly between 0 and 1, if n_modes is not
        positive, or if no mode of either network reaches kappa_min.
    """
    return index

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _reference_fields(coords: "np.ndarray", n_ref: int) -> "np.ndarray":
    """Declared in the prompt: analytic probe fields, column-normalised.

    Columns, in order: bend-x, bend-y, stretch-z, twist, radial breathing, each
    evaluated at the bead coordinates and flattened bead-major.
    """
    a = np.asarray(coords, dtype=float)
    z = a[:, 2] - a[:, 2].mean()
    x, y = a[:, 0], a[:, 1]
    o = np.zeros_like(z)
    fields = [np.c_[z ** 2, o, o], np.c_[o, z ** 2, o], np.c_[o, o, z],
              np.c_[-y, x, o], np.c_[x, y, o]]
    cols = []
    for f in fields[:n_ref]:
        v = f.ravel()
        cols.append(v / np.linalg.norm(v))
    return np.asarray(cols, dtype=float).T


def _oracle_binding_stiffening_index(
        seq_a: str = "GAGCGCUCAG",
        geometry: tuple = (32.7, 3.10, 121.0, 8.91, 5.84, 2.444),
        prot_coords: tuple = ((-2.588, 9.659, 3.500), (-8.660, -5.000, 14.000),
                              (1.531, 3.696, -3.000), (3.222, -0.424, 31.000),
                              (-9.093, 5.250, 9.500), (-11.897, -1.566, 12.000),
                              (3.827, -9.239, 20.500), (-4.401, -10.625, 18.000)),
        prot_residues: str = "KRDESTLA",
        n_modes: int = 10,
        kappa_min: float = 0.52) -> float:
    """Orchestrator. Chains steps 01-07 over the free duplex and the complex."""
    twist_deg, rise, phase_deg, r_p, r_c1, r_c2 = (float(v) for v in geometry)
    if not 0.0 < kappa_min < 1.0:
        raise ValueError("kappa_min must lie strictly between 0 and 1")
    if int(n_modes) <= 0:
        raise ValueError("n_modes must be positive")
    cutoff_na = 11.0
    cutoff_pna = 8.0
    coords = _oracle_bead_network(seq_a, twist_deg, rise, phase_deg, r_p, r_c1, r_c2)
    classes = _oracle_contact_class_matrix(seq_a, coords, cutoff_na)
    k_na = _oracle_nucleic_spring_matrix(seq_a, coords, classes)
    k_complex = _oracle_interface_spring_matrix(seq_a, coords, k_na, prot_coords,
                                                prot_residues, cutoff_pna)
    all_coords = np.vstack([np.asarray(coords, dtype=float),
                            np.asarray(prot_coords, dtype=float)])
    selected = []
    for crd, spr in ((coords, k_na), (all_coords, k_complex)):
        H = _oracle_hessian_matrix(crd, spr)
        w = _oracle_mode_spectrum(H, n_modes)
        metrics = _oracle_mode_subspace_metrics(H, _reference_fields(crd, 5), n_modes)
        eligible = [i for i in range(n_modes) if metrics[i, 0] >= kappa_min]
        if not eligible:
            raise ValueError("no mode clears kappa_min in one of the two networks")
        best = max(eligible, key=lambda i: metrics[i, 1])
        selected.append(float(w[best]))
    return selected[1] / selected[0]

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": """import numpy as np
""",
            "call": "binding_stiffening_index()",
            "gold_call": "_oracle_binding_stiffening_index()",
        },
        {
            "setup": """import numpy as np
prot = np.array([[-2.588, 9.659, 3.500], [-8.660, -5.000, 14.000],
                 [1.531, 3.696, -3.000], [3.222, -0.424, 31.000]])
""",
            "call": "binding_stiffening_index(prot_coords=prot, prot_residues=\"KRDE\")",
            "gold_call": "_oracle_binding_stiffening_index(prot_coords=prot, prot_residues=\"KRDE\")",
        },
        {
            "setup": """import numpy as np
""",
            "call": "binding_stiffening_index(seq_a=\"CGCGAUCGCG\")",
            "gold_call": "_oracle_binding_stiffening_index(seq_a=\"CGCGAUCGCG\")",
        },
        {
            "setup": """import numpy as np
""",
            "call": "binding_stiffening_index(geometry=(32.7, 3.10, 121.0, 8.91, 5.84, 2.200))",
            "gold_call": "_oracle_binding_stiffening_index(geometry=(32.7, 3.10, 121.0, 8.91, 5.84, 2.200))",
        },
        {
            "setup": """import numpy as np
def run_model():
    try:
        binding_stiffening_index(kappa_min=1.5)
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
def run_gold():
    try:
        _oracle_binding_stiffening_index(kappa_min=1.5)
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

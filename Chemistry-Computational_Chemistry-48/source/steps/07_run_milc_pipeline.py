"""
Orchestrate the complete MILC evaluation and return the molecular solvation

free energy.

The production architecture is the paper's multi-input linear correction

of the main text. Do not replace it by a one-parameter volume correction,

by the 628-molecule in-sample Full-set fit, or by the SI analogue-to-

experiment fit. Do not run a new 3D-RISM or MD job. Do not refit the

supplied coefficients.

Returns
-------
return delta_g_milc
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np


def run_milc_pipeline(
    charged_h: np.ndarray,
    charged_c: np.ndarray,
    zero_h: np.ndarray,
    zero_c: np.ndarray,
    conformer_labels: np.ndarray,
    grid_spacing: float,
    rho,
    chi_kt: float,
    coefficients: np.ndarray,
    temperature: float = 298.15,
    k_b: float = 0.00198720425864,
    descriptor_set: str = "sub",
) -> float:
    """
    Execute the complete MILC solvation free-energy pipeline.

    Parameters
    ----------
    charged_h, charged_c : np.ndarray
        Charged-state total and direct correlation fields.
    zero_h, zero_c : np.ndarray
        Zero-charge total and direct correlation fields.
    conformer_labels : np.ndarray
        Conformer identifiers.
    grid_spacing : float
        Cubic grid spacing in Angstrom.
    rho : float or np.ndarray
        Solvent-site number densities in Angstrom^-3.
    chi_kt : float
        Prefactor k_B T κ_T in Angstrom^3.
    coefficients : np.ndarray
        MILC coefficient vector for the selected descriptor set.
    temperature : float
        Temperature in Kelvin.
    k_b : float
        Boltzmann constant in kcal/(mol·K).
    descriptor_set : str
        One of "sub", "hnc", or "full".

    Returns
    -------
    float
        Molecular ΔG_MILC in kcal/mol.

    Raises
    ------
    ValueError
        If any correlation field is non-finite, if dimensions or
        conformer labels are invalid, or if other numerical inputs are
        invalid.
    """
    return delta_g_milc

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _oracle_run_milc_pipeline(
    charged_h,
    charged_c,
    zero_h,
    zero_c,
    conformer_labels,
    grid_spacing,
    rho,
    chi_kt,
    coefficients,
    temperature=298.15,
    k_b=0.00198720425864,
    descriptor_set="sub",
) -> float:
    """Deterministic reference implementation for the complete MILC pipeline."""

    # ============================================================
    # Step 01: Validate inputs
    # ============================================================
    _oracle_validate_rism_data(
        charged_h,
        charged_c,
        zero_h,
        zero_c,
        conformer_labels,
    )

    # ============================================================
    # Step 02: Solvation functionals
    # Charged and zero-charge states are independent evaluations.
    # ============================================================
    charged_terms = _oracle_compute_solvation_terms(
        charged_h,
        charged_c,
        grid_spacing,
        rho,
        temperature,
        k_b,
    )
    zero_terms = _oracle_compute_solvation_terms(
        zero_h,
        zero_c,
        grid_spacing,
        rho,
        temperature,
        k_b,
    )

    if charged_terms.shape != zero_terms.shape:
        raise ValueError(
            "charged and zero-charge solvation terms must have "
            "the same shape"
        )
    if charged_terms.ndim != 2 or charged_terms.shape[1] != 3:
        raise ValueError(
            "solvation terms must have shape (n_conformers, 3)"
        )
    if charged_terms.shape[0] != np.asarray(charged_h).shape[0]:
        raise ValueError(
            "solvation terms must contain one row per conformer"
        )
    if not np.all(np.isfinite(charged_terms)):
        raise ValueError("charged solvation terms contain non-finite values")
    if not np.all(np.isfinite(zero_terms)):
        raise ValueError(
            "zero-charge solvation terms contain non-finite values"
        )

    # ============================================================
    # Step 03: Partial molar volume from the direct correlation
    # ============================================================
    pmv_charged = _oracle_compute_pmv(
        charged_c,
        grid_spacing,
        rho,
        chi_kt,
    )
    pmv_zero = _oracle_compute_pmv(
        zero_c,
        grid_spacing,
        rho,
        chi_kt,
    )

    n_conformers = charged_terms.shape[0]
    pmv_charged = np.asarray(pmv_charged, dtype=float).reshape(-1)
    pmv_zero = np.asarray(pmv_zero, dtype=float).reshape(-1)
    if pmv_charged.shape != (n_conformers,) or pmv_zero.shape != (
        n_conformers,
    ):
        raise ValueError("PMV arrays must have one entry per conformer")
    if not np.all(np.isfinite(pmv_charged)):
        raise ValueError("charged PMV contains non-finite values")
    if not np.all(np.isfinite(pmv_zero)):
        raise ValueError("zero-charge PMV contains non-finite values")

    # ============================================================
    # Step 04: Descriptor matrices
    # ============================================================
    full, sub, hnc = _oracle_build_descriptors(
        charged_terms,
        zero_terms,
        pmv_charged,
        pmv_zero,
    )

    if full.ndim != 2 or full.shape != (n_conformers, 8):
        raise ValueError("Full descriptors must have shape (n_conformers, 8)")
    if sub.ndim != 2 or sub.shape != (n_conformers, 5):
        raise ValueError("Sub descriptors must have shape (n_conformers, 5)")
    if hnc.ndim != 2 or hnc.shape != (n_conformers, 3):
        raise ValueError("HNC descriptors must have shape (n_conformers, 3)")
    if not np.all(np.isfinite(full)):
        raise ValueError("Full descriptors contain non-finite values")
    if not np.all(np.isfinite(sub)):
        raise ValueError("Sub descriptors contain non-finite values")
    if not np.all(np.isfinite(hnc)):
        raise ValueError("HNC descriptors contain non-finite values")

    key = str(descriptor_set).lower()
    if key == "sub":
        descriptors = sub
    elif key == "hnc":
        descriptors = hnc
    elif key == "full":
        descriptors = full
    else:
        raise ValueError("descriptor_set must be 'sub', 'hnc', or 'full'")

    # ============================================================
    # Step 05: Zero-intercept MILC inner product
    # ============================================================
    predictions = _oracle_apply_milc(descriptors, coefficients)

    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    if predictions.shape != (n_conformers,):
        raise ValueError(
            "MILC predictions must contain one value per conformer"
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("MILC predictions contain non-finite values")

    # ============================================================
    # Step 06: Arithmetic conformer average
    # ============================================================
    delta_g_milc = _oracle_average_conformers(predictions)
    return float(delta_g_milc)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return differential test specifications for the MILC pipeline."""
    return [
        {
            "setup": """import numpy as np

def make_field(amp, freq, phase, offset, n_conf=3, n_sites=2, n=3):
    field = np.empty((n_conf, n_sites, n, n, n), dtype=float)
    for p in range(n_conf):
        for s in range(n_sites):
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        arg = (
                            freq * (i + 0.31 * (p + 1))
                            + 0.47 * (j + 0.19 * (s + 1))
                            + 0.29 * (k - 1.0)
                            + phase
                        )
                        field[p, s, i, j, k] = (
                            offset
                            + amp * np.sin(arg)
                            + 0.34 * (p - 1)
                            - 0.22 * (2 * s - 1)
                        )
    return field

charged_h = make_field(0.85, 0.73, 0.18, -0.10)
charged_c = make_field(0.55, 0.61, 0.44, -0.20)
zero_h = make_field(0.60, 0.55, 1.07, -0.25)
zero_c = make_field(0.40, 0.49, 0.83, -0.30)
conformer_labels = np.array(["c0", "c1", "c2"])
grid_spacing = 1.25
rho = np.array([0.033327, 0.066654])
chi_kt = 0.016387412
coefficients = np.array(
    [-0.718451957, 1.62878139, 5.19413792, -3.58716288, -750.513679]
)
temperature = 298.15
k_b = 0.00198720425864
""",
            "call": """run_milc_pipeline(
    charged_h, charged_c, zero_h, zero_c, conformer_labels,
    grid_spacing, rho, chi_kt, coefficients, temperature, k_b, "sub"
)""",
            "gold_call": """_oracle_run_milc_pipeline(
    charged_h, charged_c, zero_h, zero_c, conformer_labels,
    grid_spacing, rho, chi_kt, coefficients, temperature, k_b, "sub"
)"""
        },
        {
            "setup": """import numpy as np

def make_field(amp, freq, phase, offset, n_conf=3, n_sites=2, n=3):
    field = np.empty((n_conf, n_sites, n, n, n), dtype=float)
    for p in range(n_conf):
        for s in range(n_sites):
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        arg = (
                            freq * (i + 0.31 * (p + 1))
                            + 0.47 * (j + 0.19 * (s + 1))
                            + 0.29 * (k - 1.0)
                            + phase
                        )
                        field[p, s, i, j, k] = (
                            offset
                            + amp * np.sin(arg)
                            + 0.34 * (p - 1)
                            - 0.22 * (2 * s - 1)
                        )
    return field

charged_h = make_field(0.85, 0.73, 0.18, -0.10)
charged_c = make_field(0.55, 0.61, 0.44, -0.20)
zero_h = make_field(0.60, 0.55, 1.07, -0.25)
zero_c = make_field(0.40, 0.49, 0.83, -0.30)
conformer_labels = np.array(["c0", "c1", "c2"])
grid_spacing = 1.25
rho = np.array([0.033327, 0.066654])
chi_kt = 0.016387412
coefficients = np.array([0.99554824, 1.50999895, -682.54621964])
temperature = 298.15
k_b = 0.00198720425864
""",
            "call": """run_milc_pipeline(
    charged_h, charged_c, zero_h, zero_c, conformer_labels,
    grid_spacing, rho, chi_kt, coefficients, temperature, k_b, "hnc"
)""",
            "gold_call": """_oracle_run_milc_pipeline(
    charged_h, charged_c, zero_h, zero_c, conformer_labels,
    grid_spacing, rho, chi_kt, coefficients, temperature, k_b, "hnc"
)"""
        },
        {
            "setup": """import numpy as np

shape = (2, 2, 2, 2, 2)
charged_h = np.ones(shape)
charged_c = np.ones(shape)
zero_h = np.ones(shape)
zero_c = np.ones(shape)
charged_h[0, 0, 0, 0, 0] = np.nan
conformer_labels = np.array(["a", "b"])
rho = np.array([0.033327, 0.066654])
coefficients = np.ones(5)

def run_model():
    try:
        run_milc_pipeline(
            charged_h, charged_c, zero_h, zero_c, conformer_labels,
            1.0, rho, 16.0, coefficients
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2

def run_gold():
    try:
        _oracle_run_milc_pipeline(
            charged_h, charged_c, zero_h, zero_c, conformer_labels,
            1.0, rho, 16.0, coefficients
        )
        return 0
    except ValueError:
        return 1
    except Exception:
        return 2
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
        {
            "setup": """import numpy as np

h = np.ones((1, 1, 2, 2, 2))
c = np.full_like(h, 0.2)
labels = np.array(["only"])
coefficients = np.array([0.5, -0.2, 0.1, 0.05, -0.01])

def run_model():
    return run_milc_pipeline(
        h, c, h, c, labels, 1.0, 0.0333, 10.0, coefficients
    )

def run_gold():
    return _oracle_run_milc_pipeline(
        h, c, h, c, labels, 1.0, 0.0333, 10.0, coefficients
    )
""",
            "call": "run_model()",
            "gold_call": "run_gold()"
        },
    ]

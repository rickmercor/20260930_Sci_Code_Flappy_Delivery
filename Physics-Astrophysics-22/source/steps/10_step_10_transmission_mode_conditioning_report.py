"""
Run the chain end to end for the requested background and resolution, and return the frequency of the mode on the positive imaginary axis, its condition number in the energy norm and in the Lebesgue norm, the frequency and both condition numbers of the corresponding mode at a second spacetime dimension, the energy-norm value recomputed at a raised resolution together with the drift between the two, the smallest eigenvalue of the energy Gram matrix, the location returned by the sweep, the certification flag, and the six reported quantities rounded through step 9.

This step runs the whole chain and produces the reported quantities. Step 1 supplies the compactification weight and the reduced potential of the Tangherlini background. Step 2 supplies the Chebyshev-Lobatto grid on the unit interval and its differentiation matrix. Step 3 assembles the two members of the total transmission mode pencil from them, with the first-order member held at exactly twice the derivative so that the scale of the condition numbers is pinned. Step 4 locates the mode on the positive imaginary frequency axis by sweeping the smallest singular value of the shifted pencil at a coarse resolution. Step 5 refines it by two-sided shifted inverse iteration, returning the frequency together with the right eigenvector and the ordinary left null vector. Step 6 certifies the result by resolution independence and by the spectral decay of the eigenfunction. Step 7 builds the Gram matrices of the energy and Lebesgue inner products by exact integration of the cardinal polynomials. Step 8 forms the condition number from the Riesz image of the left vector, the two norms and the pairing. Step 9 rounds the six reported quantities to the precisions the problem asks for.

The orchestration repeats the last stages three times over. The graded value is the energy-norm condition number of the mode at the requested resolution and dimension. The same mode and Gram pair give the Lebesgue-norm value, which is the dimension-independent baseline against which the energy-norm result is read. The same construction run at a second spacetime dimension gives the cross-dimensional comparison, which is meaningful here only because the first-order member of the pencil does not depend on the dimension, so the scale ambiguity that would otherwise spoil such a comparison cancels. Finally the whole chain is repeated at a raised resolution, and the difference between the two energy-norm values is reported as the drift, which is the quantity that distinguishes this mode from the overtones: for the overtones the condition number grows by orders of magnitude under refinement, while for this one it settles.

The quadrature count passed to step 7 is set to the resolution plus half the larger of the two dimensions plus four, which exceeds the exactness requirement of that step for both backgrounds; increasing it further changes no returned value, as the tests of step 7 record. The iteration tolerance and budget passed to step 5 control only how far the refinement is carried and not the value it converges to.

Returns
-------
dict holding the float graded_answer, the energy-norm condition number rounded to six decimal places as step 9 returns it, together with the floats reported_lebesgue, reported_frequency, reported_comparison_energy, reported_comparison_lebesgue and reported_comparison_frequency; the float frequency_real and the float frequency_imag of the mode; the float condition_number_energy, the unrounded graded value; the float condition_number_lebesgue; the float condition_number_comparison, the float condition_number_lebesgue_comparison and the float frequency_imag_comparison for the second dimension; the float condition_number_refined and the float refinement_drift; the float smallest_energy_eigenvalue; the float scan_location; and the int certified.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def transmission_mode_conditioning_report(
    resolution: int,
    refinement: int,
    d: int,
    ell: int,
    s: float,
    comparison_dimension: int,
    scan_resolution: int,
    scan_lower: float,
    scan_upper: float,
    scan_samples: int,
) -> dict:
    """Run the total transmission mode conditioning chain end to end.

    Parameters
    ----------
    resolution : int
        Grid resolution at which the graded value is computed.
    refinement : int
        Increment used for the resolution drift and the certification.
    d : int
        Spacetime dimension.
    ell : int
        Multipole number.
    s : float
        Spin label.
    comparison_dimension : int
        Second spacetime dimension for the cross-dimensional value.
    scan_resolution : int
        Coarse resolution used for the sweep.
    scan_lower : float
        Lower end of the sweep.
    scan_upper : float
        Upper end of the sweep.
    scan_samples : int
        Number of sweep samples.

    Returns
    -------
    dict
        Under the keys graded_answer, reported_lebesgue, reported_frequency,
        reported_comparison_energy, reported_comparison_lebesgue, reported_comparison_frequency,
        frequency_real, frequency_imag, condition_number_energy,
        condition_number_lebesgue, condition_number_comparison,
        condition_number_lebesgue_comparison, frequency_imag_comparison, condition_number_refined,
        refinement_drift, smallest_energy_eigenvalue, scan_location and certified.

    Raises
    ------
    ValueError
        When any argument is outside the range the earlier steps accept, or when the sweep finds no
        mode on the requested interval.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _quadrature_count(resolution, dimensions):
    return int(resolution) + (max(int(v) for v in dimensions) + 1) // 2 + 4


def _condition_pair(resolution, d, ell, s, guess, quadrature):
    """Frequency and the two condition numbers of the mode nearest the guess."""
    pencil = _oracle_transmission_pencil(resolution, d, ell, s)  # noqa: F821
    mode = _oracle_regular_mode(  # noqa: F821
        pencil["operator_a"], pencil["operator_b"], guess, 1e-13, 200
    )
    gram = _oracle_energy_gram_matrix(resolution, d, ell, s, quadrature)  # noqa: F821
    energy = _oracle_condition_number_in_norm(  # noqa: F821
        gram["energy_gram"], pencil["operator_b"], mode["right_vector"], mode["left_vector"]
    )
    lebesgue = _oracle_condition_number_in_norm(  # noqa: F821
        gram["lebesgue_gram"], pencil["operator_b"], mode["right_vector"], mode["left_vector"]
    )
    return {
        "frequency": complex(mode["frequency_real"], mode["frequency_imag"]),
        "energy": float(energy["condition_number"]),
        "lebesgue": float(lebesgue["condition_number"]),
        "smallest": float(gram["smallest_energy_eigenvalue"]),
    }


def _oracle_transmission_mode_conditioning_report(
    resolution: int,
    refinement: int,
    d: int,
    ell: int,
    s: float,
    comparison_dimension: int,
    scan_resolution: int,
    scan_lower: float,
    scan_upper: float,
    scan_samples: int,
) -> dict:
    """Reference implementation."""
    if isinstance(resolution, bool) or not isinstance(resolution, (int, np.integer)):
        raise ValueError("resolution must be an integer")
    if int(resolution) < 8:
        raise ValueError("resolution must be at least eight")
    if isinstance(refinement, bool) or not isinstance(refinement, (int, np.integer)):
        raise ValueError("refinement must be an integer")
    if int(refinement) < 1:
        raise ValueError("refinement must be at least one")
    if isinstance(comparison_dimension, bool) or not isinstance(comparison_dimension, (int, np.integer)):
        raise ValueError("comparison_dimension must be an integer")
    if int(comparison_dimension) < 4:
        raise ValueError("comparison_dimension must be at least four")

    scan_pencil = _oracle_transmission_pencil(scan_resolution, d, ell, s)  # noqa: F821
    sweep = _oracle_imaginary_axis_scan(  # noqa: F821
        scan_pencil["operator_a"], scan_pencil["operator_b"],
        scan_lower, scan_upper, scan_samples,
    )
    location = float(sweep["location"])
    if not np.isfinite(location):
        raise ValueError("the sweep found no mode on the requested interval")
    guess = complex(0.0, location)

    quadrature = _quadrature_count(resolution, (d, comparison_dimension))
    graded = _condition_pair(int(resolution), d, ell, s, guess, quadrature)

    fine_resolution = int(resolution) + int(refinement)
    fine_quadrature = _quadrature_count(fine_resolution, (d, comparison_dimension))
    refined = _condition_pair(fine_resolution, d, ell, s, guess, fine_quadrature)

    comparison_scan_pencil = _oracle_transmission_pencil(  # noqa: F821
        scan_resolution, comparison_dimension, ell, s
    )
    comparison_sweep = _oracle_imaginary_axis_scan(  # noqa: F821
        comparison_scan_pencil["operator_a"], comparison_scan_pencil["operator_b"],
        scan_lower, scan_upper, scan_samples,
    )
    comparison_location = float(comparison_sweep["location"])
    if not np.isfinite(comparison_location):
        raise ValueError("the sweep found no mode for the comparison dimension")
    comparison = _condition_pair(
        int(resolution), comparison_dimension, ell, s, complex(0.0, comparison_location), quadrature
    )

    certification = _oracle_mode_certification(  # noqa: F821
        int(resolution), int(refinement), d, ell, s, guess, 1e-10
    )

    rounded = _oracle_reported_values(  # noqa: F821
        graded["energy"], graded["lebesgue"], float(graded["frequency"].imag),
        comparison["energy"], comparison["lebesgue"], float(comparison["frequency"].imag),
    )

    return {
        "graded_answer": rounded["graded_answer"],
        "reported_lebesgue": rounded["reported_lebesgue"],
        "reported_frequency": rounded["reported_frequency"],
        "reported_comparison_energy": rounded["reported_comparison_energy"],
        "reported_comparison_lebesgue": rounded["reported_comparison_lebesgue"],
        "reported_comparison_frequency": rounded["reported_comparison_frequency"],
        "frequency_real": float(graded["frequency"].real),
        "frequency_imag": float(graded["frequency"].imag),
        "condition_number_energy": graded["energy"],
        "condition_number_lebesgue": graded["lebesgue"],
        "condition_number_comparison": comparison["energy"],
        "condition_number_lebesgue_comparison": comparison["lebesgue"],
        "frequency_imag_comparison": float(comparison["frequency"].imag),
        "condition_number_refined": refined["energy"],
        "refinement_drift": float(abs(refined["energy"] - graded["energy"])),
        "smallest_energy_eigenvalue": graded["smallest"],
        "scan_location": location,
        "certified": int(certification["certified"]),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
def flat(x):
    if isinstance(x, dict):
        return flat([x[k] for k in sorted(x)])
    if isinstance(x, (tuple, list)):
        out = []
        for v in x:
            out.extend(flat(v))
        return tuple(out)
    if hasattr(x, "tolist"):
        return flat(x.tolist())
    if isinstance(x, bool):
        return (int(x),)
    return (x,)
"""

    SETUP = """
import numpy as np
def digest(out):
    return (round(out["frequency_imag"], 10), round(out["condition_number_energy"], 10),
            round(out["condition_number_lebesgue"], 9), round(out["condition_number_comparison"], 9),
            round(out["frequency_imag_comparison"], 10), int(out["certified"]))
"""
    return [
        {
            # the graded configuration at a reduced resolution
            "setup": SETUP + FLAT,
            "call": "flat(digest(transmission_mode_conditioning_report(60, 30, 14, 2, 2.0, 20, 40, 0.3, 4.0, 75)))",
            "gold_call": "flat(digest(_oracle_transmission_mode_conditioning_report(60, 30, 14, 2, 2.0, 20, 40, 0.3, 4.0, 75)))",
        },
        {
            # the reported structure: the mode is purely imaginary, the energy-norm value settles under
            # refinement while staying well below the Lebesgue-norm value, the condition number falls as
            # the dimension rises, the energy Gram matrix is positive definite although the reduced
            # potential is negative at the horizon, and the sweep lands within a sample of the mode
            "setup": SETUP + """
def structure(fn):
    out = fn(60, 30, 14, 2, 2.0, 20, 40, 0.3, 4.0, 75)
    return (int(abs(out["frequency_real"]) < 1e-12),
            int(out["refinement_drift"] < 1e-3),
            int(out["condition_number_energy"] < out["condition_number_lebesgue"]),
            int(out["condition_number_comparison"] < out["condition_number_energy"]),
            int(out["frequency_imag_comparison"] < out["frequency_imag"]),
            int(out["smallest_energy_eigenvalue"] > 0.0),
            int(abs(out["scan_location"] - out["frequency_imag"]) < 0.05),
            int(out["certified"]))
""" + FLAT,
            "call": "flat(structure(transmission_mode_conditioning_report))",
            "gold_call": "flat(structure(_oracle_transmission_mode_conditioning_report))",
        },
        {
            # boundary: a low resolution with the smallest admissible refinement, and a comparison
            # dimension equal to the primary one, for which the two condition numbers coincide
            "setup": SETUP + """
def same_dimension(fn):
    out = fn(30, 1, 14, 2, 2.0, 14, 30, 0.3, 4.0, 60)
    return (int(abs(out["condition_number_comparison"] - out["condition_number_energy"]) < 1e-12),
            int(abs(out["frequency_imag_comparison"] - out["frequency_imag"]) < 1e-12),
            round(out["condition_number_energy"], 9))
""" + FLAT,
            "call": "flat(same_dimension(transmission_mode_conditioning_report))",
            "gold_call": "flat(same_dimension(_oracle_transmission_mode_conditioning_report))",
        },
        {
            "setup": SETUP + """
def verdict(fn, **kw):
    args = dict(resolution=30, refinement=10, d=14, ell=2, s=2.0, comparison_dimension=20,
                scan_resolution=30, scan_lower=0.3, scan_upper=4.0, scan_samples=60)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""" + FLAT,
            "call": "flat((verdict(transmission_mode_conditioning_report, resolution=6), "
                    "verdict(transmission_mode_conditioning_report, refinement=0), "
                    "verdict(transmission_mode_conditioning_report, comparison_dimension=3), "
                    "verdict(transmission_mode_conditioning_report, scan_lower=2.5, scan_upper=3.5), "
                    "verdict(transmission_mode_conditioning_report, s=0.0)))",
            "gold_call": "flat((verdict(_oracle_transmission_mode_conditioning_report, resolution=6), "
                         "verdict(_oracle_transmission_mode_conditioning_report, refinement=0), "
                         "verdict(_oracle_transmission_mode_conditioning_report, comparison_dimension=3), "
                         "verdict(_oracle_transmission_mode_conditioning_report, scan_lower=2.5, scan_upper=3.5), "
                         "verdict(_oracle_transmission_mode_conditioning_report, s=0.0)))",
        },
    ]

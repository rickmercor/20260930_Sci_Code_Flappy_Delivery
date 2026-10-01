"""
Run the complete two-threshold form-factor calculation and combine a non-empty sequence of sheet-specific evaluations into one scalar. The two positive masses set the ordered thresholds, the reference points fix the bounded conformal variable, every pole energy is squared and mapped on its own sheet and the ascending polynomial is multiplied by the reciprocal pole product. The function returns the product of the squared form-factor moduli as a native Python float. Invalid input raises ValueError for non-finite or non-scalar data, non-real masses, geometry or coefficients, non-positive or unordered masses, unknown sheets, malformed or empty pole, coefficient or evaluation sequences, coincident mapped references, singular conformal maps including the outer reciprocal or an outer branch at the lower threshold, overflowing thresholds, squared pole energies or evaluations, an overflowing squared-modulus product or coincidence with a mapped pole or its conjugate.

The normalization point must lie strictly below the lower threshold and its intermediate image must be distinct from both unit endpoints in double precision. Unsupported normalization geometries and constituent map or reciprocal-product range failures propagate as ValueError. A squared modulus outside the finite float range raises ValueError even when the complex form factor itself is finite. The normalization point must be strictly below the lower threshold and its intermediate image must be distinct from both unit endpoints in double precision. Unsupported normalization geometries and numerical range failures in either constituent map or the reciprocal product propagate as ValueError. The normalization point must lie strictly below the lower threshold and its intermediate image must be distinct from both unit endpoints in double precision. Unsupported normalization geometries and numerical range refusals from any earlier step propagate as ValueError. Every individual squared modulus and the running product must be finite; overflow in either is rejected. The product probes a single analytic form factor on several Riemann sheets at once. Its analytic structure is carried by the composed map itself, with no outer function and no Omnes factor, while the reciprocal pole factor supplies the resonances and the polynomial supplies the smooth remainder. Mapping every pole on its quoted sheet preserves independent pole data from sheets that are not contiguous to the physical one.

Returns
-------
float, the product of squared form-factor moduli as a native Python float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

from collections.abc import Sequence
import numpy as np


def run_pipeline(light_mass, heavy_mass, cut_opening, normalization_point, poles, coefficients, evaluations):
    """Return the product of squared form-factor moduli at all evaluations.

    Parameters
    ----------
    light_mass : float
        Finite positive mass of the lighter channel constituent in energy units.
    heavy_mass : float
        Finite positive mass of the heavier channel constituent in energy units.
    cut_opening : float
        Finite real squared energy at which the left-hand cut opens.
    normalization_point : float
        Finite real squared energy at which the bounded variable is normalized.
    poles : sequence of pairs
        Non-empty sequence of complex pole energies and their sheet labels.
    coefficients : sequence of float
        Non-empty sequence of finite real polynomial coefficients in ascending order.
    evaluations : sequence of pairs
        Non-empty sequence of finite squared energies and evaluation sheet labels.

    Returns
    -------
    float
        Product of the squared form-factor moduli as a native Python float.

    Raises
    ------
    ValueError
        If either mass is not a finite positive real scalar, if their thresholds
        are not strictly ordered, if a reference value or coefficient is not a
        finite real scalar, if any energy is not a finite scalar, if any sheet
        is unknown, if a sequence is empty or malformed, if mapped references
        coincide, if a conformal denominator vanishes (including the outer
        reciprocal), if an outer branch is requested at the lower threshold,
        if a threshold, squared pole energy, map, form factor or squared-modulus
        product overflows, or if an evaluation coincides with a mapped pole or
        its conjugate. This includes every individual squared modulus, unsupported
        normalization geometries (point not below the lower threshold or image
        at a unit endpoint) and numerical range refusals in earlier steps.
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from collections.abc import Sequence
import numpy as np


def _oracle_run_pipeline(light_mass, heavy_mass, cut_opening, normalization_point, poles, coefficients, evaluations):
    from collections.abc import Sequence

    lower, upper, opening, normalization = _oracle_channel_geometry(
        light_mass, heavy_mass, cut_opening, normalization_point
    )

    if isinstance(poles, (str, bytes)) or not isinstance(poles, Sequence) or len(poles) == 0:
        raise ValueError("poles must be a non-empty sequence of energy and sheet pairs")
    for index, entry in enumerate(poles):
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
            raise ValueError("poles[" + str(index) + "] must be an energy and sheet pair")

    if isinstance(evaluations, (str, bytes)) or not isinstance(evaluations, Sequence) or len(evaluations) == 0:
        raise ValueError("evaluations must be a non-empty sequence of squared energy and sheet pairs")
    for index, entry in enumerate(evaluations):
        if isinstance(entry, (str, bytes)) or not isinstance(entry, Sequence) or len(entry) != 2:
            raise ValueError("evaluations[" + str(index) + "] must be a squared energy and sheet pair")

    product = 1.0
    for squared_energy, evaluation_sheet in evaluations:
        value = _oracle_form_factor(
            squared_energy,
            evaluation_sheet,
            lower,
            upper,
            opening,
            normalization,
            poles,
            coefficients,
        )
        try:
            product *= abs(value) ** 2
        except OverflowError as exc:
            raise ValueError("the squared-modulus product must be finite") from exc
        if not np.isfinite(product):
            raise ValueError("the squared-modulus product must be finite")
    return float(product)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    status_setup = (
        "import numpy as np\n"
        "def status(fn):\n"
        "    try:\n"
        "        fn()\n"
        "    except ValueError:\n"
        "        return 1\n"
        "    except Exception:\n"
        "        return 2\n"
        "    return 0\n"
    )

    def _status_case(args):
        # call runs the model's function and gold_call runs the oracle through
        # the same try/except, so the exception contract is compared against the
        # oracle rather than asserted against a constant.
        return {
            "setup": status_setup,
            "call": "status(lambda: run_pipeline" + args + ")",
            "gold_call": "status(lambda: _oracle_run_pipeline" + args + ")",
        }

    return [
        {
            "setup": "import numpy as np\npoles = [(0.452 - 0.271j, '21'), (0.998 - 0.029j, '21'), (0.981 - 0.043j, '22')]\ncoefficients = [0.462, -0.236, 0.559, -0.706, 0.228, -0.510, -0.147, 1.088, -0.526, 1.578, -0.268, 0.533]\nevaluations = [(1.15 + 0.12j, '22'), (0.55 + 0.18j, '21')]",
            "call": "round(run_pipeline(0.13957, 0.493677, 0.0, -0.60, poles, coefficients, evaluations), 12)",
            "gold_call": "round(_oracle_run_pipeline(0.13957, 0.493677, 0.0, -0.60, poles, coefficients, evaluations), 12)",
        },
        {
            "setup": "import numpy as np\npoles = [(0.452 - 0.271j, '21')]\ncoefficients = [1.0]\nevaluations = [(-0.60, '11')]",
            "call": "round(run_pipeline(0.13957, 0.493677, 0.0, -0.60, poles, coefficients, evaluations), 12)",
            "gold_call": "round(_oracle_run_pipeline(0.13957, 0.493677, 0.0, -0.60, poles, coefficients, evaluations), 12)",
        },
        {
            "setup": "import numpy as np\npoles = [(0.452 - 0.271j, '21'), (0.981 - 0.043j, '22')]\ncoefficients = [0.0, 0.0, 1.0]\nevaluations = [(2.0 + 0.3j, '12'), (0.2 + 0.1j, '11'), (0.8 - 0.2j, '22')]",
            "call": "round(run_pipeline(0.13957, 0.493677, 0.0, -0.60, poles, coefficients, evaluations), 12)",
            "gold_call": "round(_oracle_run_pipeline(0.13957, 0.493677, 0.0, -0.60, poles, coefficients, evaluations), 12)",
        },
        # valid: the prompt's graded configuration
        {
            "setup": "import numpy as np\npoles = [(0.452 - 0.271j, '21'), (0.998 - 0.029j, '21'), (0.981 - 0.043j, '22')]\ncoefficients = [0.462, -0.236, 0.559, -0.706, 0.228, -0.510, -0.147, 1.088, -0.526, 1.578, -0.268, 0.533]\nevaluations = [(-3.50 + 0.20j, '22'), (-0.75 - 0.16j, '12')]\n",
            "call": "round(run_pipeline(0.13957, 0.493677, 0.0, -0.60, poles, coefficients, evaluations), 6)",
            "gold_call": "round(_oracle_run_pipeline(0.13957, 0.493677, 0.0, -0.60, poles, coefficients, evaluations), 6)",
        },
        # valid: extreme mass hierarchy exercises the stable root form
        {
            "setup": "import numpy as np\npoles = [(0.452 - 0.271j, '21')]\ncoefficients = [0.5, -0.25]\nevaluations = [(0.3 + 0.2j, '21'), (0.3 + 0.2j, '21')]\n",
            "call": "round(run_pipeline(1e-06, 1000.0, 0.0, -1.0, poles, coefficients, evaluations), 8)",
            "gold_call": "round(_oracle_run_pipeline(1e-06, 1000.0, 0.0, -1.0, poles, coefficients, evaluations), 8)",
        },
        # invalid: vanishing outer conformal denominator
        _status_case("(1.0, 1.25, 0.0, 2.734375, [(.5 - .1j, '21')], [1.0], [(2.734375, '22')])"),
        # invalid: complex cut opening
        _status_case("(0.13957, 0.493677, np.complex64(.1 + .2j), -0.60, [(.5 - .1j, '21')], [1.0], [(0.2, '11')])"),
        # invalid: normalization point not below the lower threshold
        _status_case("(0.13957, 0.493677, 0.0, 1e200, [(.5 - .1j, '21')], [1.0], [(0.2, '11')])"),
        # invalid: squared-modulus product overflows
        _status_case("(0.13957, 0.493677, 0.0, -0.60, [(.5 - .1j, '21')], [1e200], [(0.2, '11')])"),
        # invalid: normalization image at a unit endpoint
        _status_case("(0.13957, 0.493677, 0.0, -1e308, [(.5 - .1j, '21')], [1.0], [(0.2, '11')])"),
        # invalid: outer branch at the lower threshold
        _status_case("(1.0, 1.25, 0.0, -0.60, [(.5 - .1j, '21')], [1.0], [(4.0, '22')])"),
        # invalid: negative mass
        _status_case("(-0.13957, 0.493677, 0.0, -0.60, [(.5 - .1j, '21')], [1.0], [(0.2, '11')])"),
        # invalid: masses swapped
        _status_case("(0.493677, 0.13957, 0.0, -0.60, [(.5 - .1j, '21')], [1.0], [(0.2, '11')])"),
        # invalid: empty evaluation sequence
        _status_case("(0.13957, 0.493677, 0.0, -0.60, [(.5 - .1j, '21')], [1.0], [])"),
        # invalid: unknown pole sheet
        _status_case("(0.13957, 0.493677, 0.0, -0.60, [(.5 - .1j, '13')], [1.0], [(0.2, '11')])"),
        # invalid: non-finite evaluation energy
        _status_case("(0.13957, 0.493677, 0.0, -0.60, [(.5 - .1j, '21')], [1.0], [(float('nan'), '11')])"),
        # invalid: evaluation coincides with a mapped pole
        _status_case("(0.2, 0.5, 0.0, -0.60, [(0.5 - 0.1j, '21')], [1.0], [((0.5 - 0.1j)**2, '21')])"),
    ]

"""
Componentwise interpolation of the supplied Kerr-mode state table and pairing with remnant mass.

Linearly interpolate every real state table column independently at requested spins in $[0.70,0.86]$, with no extrapolation. Pair each spin row with its finite positive supplied remnant mass ratio $M_f/M_{\rm ref}$. When the optional mass array is omitted, use one for every row. The tabulated frequencies are the intrinsic dimensionless products $\widehat\omega=M_f\omega$ and are not divided by mass in this stage. The synthetic $\rho_l$ columns feed the response from two 220 parents and the supplied $\kappa_l=-0.20\rho_l$ transfer from unequal parents in step 03.

Returns
-------
`numpy.ndarray` of shape `(n, 16)` with columns $$ (\chi_f,M_f/M_{\rm ref},\Re\widehat\omega_{220},\Im\widehat\omega_{220},\Re\widehat\omega_{320},\Im\widehat\omega_{320},\Re\widehat\omega_{640},\Im\widehat\omega_{640}, \Re\mu_{22},\Im\mu_{22},\Re\mu_{32},\Im\mu_{32}, \Re \rho_4,\Im \rho_4,\Re \rho_5,\Im \rho_5). $$ Every returned component is compared at numerical tolerance `1e-9`.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
_RAW_TABLE = np.array([
    [0.70, 0.532600243597, -0.080792873136, 0.759174723145, -0.084189645815, 1.541269014176, -0.086754879659, 0.997417280443, -0.009984915659, -0.100258037153, 0.008919401626, -0.0146100910845, 0.2685306133210, -0.0000431583755, 0.0057182161510],
    [0.72, 0.541793731483, -0.079990805121, 0.767600177103, -0.083373330711, 1.558104443344, -0.085893956678, 0.997176808792, -0.010342555366, -0.104222397641, 0.008993049256, -0.0135624224605, 0.2594809831015, -0.0000440202685, 0.0058059579770],
    [0.74, 0.551630332971, -0.079092730022, 0.776508654282, -0.082467481710, 1.575854074157, -0.084937290752, 0.996913703333, -0.010696053255, -0.108307676796, 0.009045655703, -0.0125189913410, 0.2504395563260, -0.0000496081975, 0.0058990191325],
    [0.76, 0.562200718998, -0.078081685942, 0.785958791816, -0.081457397033, 1.594626097338, -0.083869410692, 0.996624850763, -0.011042407814, -0.112526148250, 0.009074892257, -0.0114774776610, 0.2413981510995, -0.0000538512135, 0.0059813295865],
    [0.78, 0.573616428376, -0.076936363177, 0.796021310470, -0.080324671683, 1.614550655886, -0.082671004345, 0.996306431311, -0.011377667052, -0.116892351938, 0.009077706061, -0.0104350677610, 0.2323464427360, -0.0000564632375, 0.0060520126375],
    [0.80, 0.586016974889, -0.075629552389, 0.806782753193, -0.079045861888, 1.635786572623, -0.081317547659, 0.995953682283, -0.011696506766, -0.121423923726, 0.009050514766, -0.0093882979190, 0.2232712080030, -0.0000633699605, 0.0061239641660],
    [0.82, 0.599580345727, -0.074125837403, 0.818350853396, -0.077590486692, 1.658530992051, -0.079777259961, 0.995560546571, -0.011991671098, -0.126142547285, 0.008988807405, -0.0083328258905, 0.2141552033065, -0.0000680304500, 0.0061813114210],
    [0.84, 0.614539083929, -0.072378038263, 0.830862496923, -0.075917929225, 1.683033655782, -0.078007941274, 0.995119134957, -0.012253083617, -0.131075225463, 0.008886290872, -0.0072630847515, 0.2049754391135, -0.0000761402500, 0.0062223026810],
    [0.86, 0.631206038694, -0.070321478391, 0.844496006954, -0.073972411405, 1.709618885024, -0.075951859914, 0.994618869600, -0.012466362882, -0.136256982615, 0.008735425951, -0.0061717338255, 0.1957004072510, -0.0000810477215, 0.0062578720305],
], dtype=float)
def interpolate_spin_tables(spins: np.ndarray, remnant_mass_ratios: np.ndarray | None = None) -> np.ndarray:
    """Interpolate state-table columns and pair them with remnant-mass ratios.

    Parameters
    ----------
    spins : array_like, shape (n,)
        Trial dimensionless spins in the closed interval [0.70, 0.86].
    remnant_mass_ratios : array_like, shape (n,), optional
        Finite positive values of M_f/M_ref paired row by row with the spins.
        When omitted, use one for every row.

    Returns
    -------
    numpy.ndarray, shape (n, 16)
        Spin; remnant-mass ratio; intrinsic dimensionless 220, 320, and 640
        frequencies; two parent projections; two
        task-defined synthetic radial response coefficients used with the
        step-03 mixed-parent scaling convention.
        Components are compared at numerical tolerance 1e-9.

    Raises
    ------
    ValueError
        If spins is empty, nonfinite, not one-dimensional, or outside the
        closed interpolation interval, or if remnant_mass_ratios has the wrong
        shape or contains a nonfinite or nonpositive value.
    """
    return None
import numpy as np

# =============================================================================
# GOLD SOLUTION
# =============================================================================

r"""
Componentwise interpolation of the supplied Kerr-mode state table and pairing with remnant mass.

Linearly interpolate every real state-table column independently at requested spins in $[0.70,0.86]$, with no extrapolation. Pair each spin with its supplied finite positive remnant-mass ratio $M_f/M_{\rm ref}$. The tabulated frequencies are the intrinsic dimensionless products $\widehat\omega=M_f\omega$; do not divide them by the mass ratio in this stage.

Returns
-------
`numpy.ndarray` of shape `(n, 16)` with columns

$$
(\chi_f,M_f/M_{\rm ref},\Re\widehat\omega_{220},\Im\widehat\omega_{220},\Re\widehat\omega_{320},\Im\widehat\omega_{320},\Re\widehat\omega_{640},\Im\widehat\omega_{640},
\Re\mu_{22},\Im\mu_{22},\Re\mu_{32},\Im\mu_{32},
\Re \rho_4,\Im \rho_4,\Re \rho_5,\Im \rho_5).
$$

The $\rho_l$ values are task-defined synthetic 220-by-220 radial transfers.
Step 03 also uses the task-defined mixed-parent transfer
$\kappa_l=-0.20\rho_l$.

Every returned component is compared at numerical tolerance `1e-9`.
"""
"""Componentwise interpolation of the supplied Kerr-mode state table."""
import numpy as np
"""Componentwise interpolation of the supplied Kerr-mode state table."""
import numpy as np
_RAW_TABLE = np.array([
    [0.70, 0.532600243597, -0.080792873136, 0.759174723145, -0.084189645815, 1.541269014176, -0.086754879659, 0.997417280443, -0.009984915659, -0.100258037153, 0.008919401626, -0.0146100910845, 0.2685306133210, -0.0000431583755, 0.0057182161510],
    [0.72, 0.541793731483, -0.079990805121, 0.767600177103, -0.083373330711, 1.558104443344, -0.085893956678, 0.997176808792, -0.010342555366, -0.104222397641, 0.008993049256, -0.0135624224605, 0.2594809831015, -0.0000440202685, 0.0058059579770],
    [0.74, 0.551630332971, -0.079092730022, 0.776508654282, -0.082467481710, 1.575854074157, -0.084937290752, 0.996913703333, -0.010696053255, -0.108307676796, 0.009045655703, -0.0125189913410, 0.2504395563260, -0.0000496081975, 0.0058990191325],
    [0.76, 0.562200718998, -0.078081685942, 0.785958791816, -0.081457397033, 1.594626097338, -0.083869410692, 0.996624850763, -0.011042407814, -0.112526148250, 0.009074892257, -0.0114774776610, 0.2413981510995, -0.0000538512135, 0.0059813295865],
    [0.78, 0.573616428376, -0.076936363177, 0.796021310470, -0.080324671683, 1.614550655886, -0.082671004345, 0.996306431311, -0.011377667052, -0.116892351938, 0.009077706061, -0.0104350677610, 0.2323464427360, -0.0000564632375, 0.0060520126375],
    [0.80, 0.586016974889, -0.075629552389, 0.806782753193, -0.079045861888, 1.635786572623, -0.081317547659, 0.995953682283, -0.011696506766, -0.121423923726, 0.009050514766, -0.0093882979190, 0.2232712080030, -0.0000633699605, 0.0061239641660],
    [0.82, 0.599580345727, -0.074125837403, 0.818350853396, -0.077590486692, 1.658530992051, -0.079777259961, 0.995560546571, -0.011991671098, -0.126142547285, 0.008988807405, -0.0083328258905, 0.2141552033065, -0.0000680304500, 0.0061813114210],
    [0.84, 0.614539083929, -0.072378038263, 0.830862496923, -0.075917929225, 1.683033655782, -0.078007941274, 0.995119134957, -0.012253083617, -0.131075225463, 0.008886290872, -0.0072630847515, 0.2049754391135, -0.0000761402500, 0.0062223026810],
    [0.86, 0.631206038694, -0.070321478391, 0.844496006954, -0.073972411405, 1.709618885024, -0.075951859914, 0.994618869600, -0.012466362882, -0.136256982615, 0.008735425951, -0.0061717338255, 0.1957004072510, -0.0000810477215, 0.0062578720305],
], dtype=float)
def _oracle_interpolate_spin_tables(spins: np.ndarray, remnant_mass_ratios: np.ndarray | None = None) -> np.ndarray:
    query = np.asarray(spins, dtype=float)
    if query.ndim != 1 or query.size == 0 or not np.all(np.isfinite(query)):
        raise ValueError("spins must be a nonempty finite one-dimensional array")
    nodes = _RAW_TABLE[:, 0]
    if np.any(query < nodes[0]) or np.any(query > nodes[-1]):
        raise ValueError("spins must lie in the supplied interpolation interval")
    masses = np.ones_like(query) if remnant_mass_ratios is None else np.asarray(remnant_mass_ratios, dtype=float)
    if masses.shape != query.shape or not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
        raise ValueError("remnant_mass_ratios must have finite positive shape (n,)")
    interpolated = np.column_stack([query, masses, *[np.interp(query, nodes, _RAW_TABLE[:, j]) for j in range(1, 15)]])
    return interpolated

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Endpoint, tabulated, and interior interpolation cases."""
    return [
        {
            "setup": "import numpy as np\nspins = np.array([0.70, 0.78, 0.86])",
            "call": "interpolate_spin_tables(spins)",
            "gold_call": "_oracle_interpolate_spin_tables(spins)",
        },
        {
            "setup": "import numpy as np\nspins = np.array([0.73, 0.755, 0.815]); masses = np.array([1.022, 0.981, 1.007])",
            "call": "interpolate_spin_tables(spins, masses)",
            "gold_call": "_oracle_interpolate_spin_tables(spins, masses)",
        },
        {
            "setup": "import numpy as np\ndef catches_value_error(fn):\n    try:\n        fn()\n    except ValueError:\n        return True\n    return False\nspins = np.array([0.74, 0.80])",
            "call": "catches_value_error(lambda: interpolate_spin_tables(np.array([0.69, 0.80]))) and catches_value_error(lambda: interpolate_spin_tables(np.array([]))) and catches_value_error(lambda: interpolate_spin_tables(np.array([[0.75]]))) and catches_value_error(lambda: interpolate_spin_tables(np.array([np.nan]))) and catches_value_error(lambda: interpolate_spin_tables(spins, np.array([1.0]))) and catches_value_error(lambda: interpolate_spin_tables(spins, np.array([1.0, np.nan]))) and catches_value_error(lambda: interpolate_spin_tables(spins, np.array([1.0, 0.0])))",
            "gold_call": "catches_value_error(lambda: _oracle_interpolate_spin_tables(np.array([0.69, 0.80]))) and catches_value_error(lambda: _oracle_interpolate_spin_tables(np.array([]))) and catches_value_error(lambda: _oracle_interpolate_spin_tables(np.array([[0.75]]))) and catches_value_error(lambda: _oracle_interpolate_spin_tables(np.array([np.nan]))) and catches_value_error(lambda: _oracle_interpolate_spin_tables(spins, np.array([1.0]))) and catches_value_error(lambda: _oracle_interpolate_spin_tables(spins, np.array([1.0, np.nan]))) and catches_value_error(lambda: _oracle_interpolate_spin_tables(spins, np.array([1.0, 0.0])))",
        },
    ]

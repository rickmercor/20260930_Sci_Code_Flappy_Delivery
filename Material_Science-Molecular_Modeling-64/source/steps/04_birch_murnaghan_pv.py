"""
Integrate the pure-reference EOS to obtain the finite-volume Gibbs correction.

The paper converts the equal-volume Helmholtz difference to a fixed-pressure Gibbs difference using P*(Vx-VA)-integral P_A(V)dV and a third-order Birch-Murnaghan EOS.

Returns
-------
return np.array([correction, endpoint_pressure, reference_residual], dtype=float)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def birch_murnaghan_pv(volume_solution_a3, volume_reference_a3,
                       pressure_target_gpa, bulk_modulus_gpa,
                       bulk_derivative, volume_zero_a3,
                       quadrature_order=64):
    """Evaluate the third-order Birch-Murnaghan EOS correction.

    Returns a float64 ndarray of length 3 ordered as
    [F_PV_eV_per_atom, P_at_solution_volume_GPa,
    P_at_reference_volume_minus_target_GPa].
    """
    return None

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_birch_murnaghan_pv(volume_solution_a3, volume_reference_a3,
                               pressure_target_gpa, bulk_modulus_gpa,
                               bulk_derivative, volume_zero_a3,
                               quadrature_order=64):
    """Evaluate the paper's EOS free-energy correction with Gauss-Legendre quadrature."""
    import math
    import numpy as np
    volume_solution = float(volume_solution_a3)
    volume_reference = float(volume_reference_a3)
    pressure_target = float(pressure_target_gpa)
    bulk_modulus = float(bulk_modulus_gpa)
    bulk_derivative = float(bulk_derivative)
    volume_zero = float(volume_zero_a3)
    order = int(quadrature_order)
    values = (volume_solution, volume_reference, pressure_target,
              bulk_modulus, bulk_derivative, volume_zero)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("inputs must be finite")
    if min(volume_solution, volume_reference, bulk_modulus, volume_zero) <= 0.0 or order < 8:
        raise ValueError("volumes and modulus must be positive and quadrature_order>=8")

    def pressure(volume):
        eta = (volume_zero / volume) ** (1.0 / 3.0)
        return (1.5 * bulk_modulus * (eta ** 7 - eta ** 5)
                * (1.0 + 0.75 * (bulk_derivative - 4.0) * (eta * eta - 1.0)))

    nodes, weights = np.polynomial.legendre.leggauss(order)
    mapped = 0.5 * (volume_solution - volume_reference) * nodes + 0.5 * (
        volume_solution + volume_reference
    )
    integral_gpa_a3 = 0.5 * (volume_solution - volume_reference) * float(
        weights @ np.array([pressure(v) for v in mapped], dtype=float)
    )
    gpa_a3_to_ev = 0.006241509074460763
    correction = (pressure_target * (volume_solution - volume_reference)
                  - integral_gpa_a3) * gpa_a3_to_ev
    endpoint_pressure = pressure(volume_solution)
    reference_residual = pressure(volume_reference) - pressure_target
    return np.array([correction, endpoint_pressure, reference_residual], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': '', 'call': 'birch_murnaghan_pv(32.4,41.552,0.0,5.143,3.943,41.552,64)', 'gold_call': '_oracle_birch_murnaghan_pv(32.4,41.552,0.0,5.143,3.943,41.552,64)'}, {'setup': '', 'call': 'birch_murnaghan_pv(41.552,41.552,0.0,5.143,3.943,41.552,32)', 'gold_call': '_oracle_birch_murnaghan_pv(41.552,41.552,0.0,5.143,3.943,41.552,32)'}, {'setup': '', 'call': 'birch_murnaghan_pv(30.0,38.0,0.668590247,8.0,4.2,40.0,96)', 'gold_call': '_oracle_birch_murnaghan_pv(30.0,38.0,0.668590247,8.0,4.2,40.0,96)'}, {'setup': '', 'call': 'birch_murnaghan_pv(44.0,40.0,0.0,7.0,4.0,40.0,64)', 'gold_call': '_oracle_birch_murnaghan_pv(44.0,40.0,0.0,7.0,4.0,40.0,64)'}, {'setup': '', 'call': 'birch_murnaghan_pv(39.0,41.552,0.0,5.143,4.0,41.552,16)', 'gold_call': '_oracle_birch_murnaghan_pv(39.0,41.552,0.0,5.143,4.0,41.552,16)'}, {'setup': '', 'call': 'birch_murnaghan_pv(28.0,41.552,0.0,5.143,3.943,41.552,128)', 'gold_call': '_oracle_birch_murnaghan_pv(28.0,41.552,0.0,5.143,3.943,41.552,128)'}, {'setup': '', 'call': 'birch_murnaghan_pv(39.5,40.0,0.103105,6.0,4.5,40.5,48)', 'gold_call': '_oracle_birch_murnaghan_pv(39.5,40.0,0.103105,6.0,4.5,40.5,48)'}, {'setup': '', 'call': 'birch_murnaghan_pv(35.2,42.1,0.25,4.8,3.2,41.7,80)', 'gold_call': '_oracle_birch_murnaghan_pv(35.2,42.1,0.25,4.8,3.2,41.7,80)'}]

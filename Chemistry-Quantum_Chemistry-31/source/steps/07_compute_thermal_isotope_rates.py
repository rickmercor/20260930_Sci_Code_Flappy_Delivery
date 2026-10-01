"""
Thermally average fixed-distance LZ and weak-coupling rate kernels under the
declared finite-order benchmark convention.

``profile_values`` supplies the reactant, crossing, product, and coupling
profiles on a strictly increasing distance grid.  Combine both transmission
branches with the same normalized reactant distance distribution and
activation convention, use the source-defined weak-coupling lower-energy
domain, evaluate each transformed thermal integral at exactly its requested
Gauss-Laguerre order without adaptive extrapolation, and integrate over
distance with the trapezoidal convention.  Return
the two reduced rates, shifted distance partition, and mean distance in the
order declared by the signature.  Inputs outside the finite physical domain,
including quadrature orders outside eight through 100 or activation exponents
of magnitude 600 or more, raise ``ValueError``.

Returns
-------
np.ndarray of shape (4,), ordered as [k_LZ, k_WC, shifted_partition, mean_distance]; the two reduced rates are finite and nonnegative in E_h/hbar, and the last two entries are in bohr.
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_thermal_isotope_rates(
    distance: "np.ndarray",
    profile_values: "np.ndarray",
    beta: float,
    crossing_descriptor: "np.ndarray",
    lz_order: int,
    wc_order: int,
) -> "np.ndarray":
    """Return LZ and WC thermal rates plus two distance-distribution audits.

    Parameters
    ----------
    distance
        Finite strictly increasing distance grid in bohr with shape ``(n,)``.
    profile_values
        Finite array in ``E_h`` with shape ``(n, 4)`` and columns
        ``[V_r, V_MECP, V_product, |V_ab|]`` with nonnegative couplings.
    beta
        Finite strictly positive inverse temperature in ``E_h^-1``.
    crossing_descriptor
        Finite vector ``[mu, gradient_gap, gradient_product_root, curvature]``
        whose four entries are strictly positive; the first three units are
        ``m_e``, ``E_h bohr^-1``, and ``E_h bohr^-1`` respectively.
    lz_order, wc_order
        Integer Gauss-Laguerre orders from eight through 100.

    Returns
    -------
    np.ndarray
        ``[k_LZ, k_WC, shifted_partition, mean_distance]``; the two reduced
        rates are finite and nonnegative in ``E_h/hbar`` and the last two
        entries are in bohr.

    Raises
    ------
    ValueError
        If shapes, orders, or physical domains are invalid, or if an
        activation exponent reaches magnitude 600, or if a thermal rate is
        negative or non-finite.
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np


def _step7_real_scalar(value: float, name: str) -> float:
    array = np.asarray(value)
    if (
        array.ndim != 0
        or not np.issubdtype(array.dtype, np.number)
        or not np.isrealobj(array)
        or not bool(np.isfinite(array))
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(array)


def _step7_trapezoid(values: np.ndarray, grid: np.ndarray) -> float:
    return float(np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(grid)))


def _oracle_compute_thermal_isotope_rates(
    distance: "np.ndarray",
    profile_values: "np.ndarray",
    beta: float,
    crossing_descriptor: "np.ndarray",
    lz_order: int,
    wc_order: int,
) -> "np.ndarray":
    raw_distance = np.asarray(distance)
    raw_profiles = np.asarray(profile_values)
    raw_descriptor = np.asarray(crossing_descriptor)
    arrays = [raw_distance, raw_profiles, raw_descriptor]
    if any(
        not np.issubdtype(value.dtype, np.number)
        or not np.isrealobj(value)
        or np.any(~np.isfinite(value))
        for value in arrays
    ):
        raise ValueError("arrays must be finite, real, and numeric")
    r = np.asarray(raw_distance, dtype=float)
    profiles = np.asarray(raw_profiles, dtype=float)
    descriptor = np.asarray(raw_descriptor, dtype=float)
    beta_value = _step7_real_scalar(beta, "beta")
    if (
        r.ndim != 1
        or r.size < 2
        or profiles.shape != (r.size, 4)
        or descriptor.shape != (4,)
        or np.any(np.diff(r) <= 0.0)
        or np.any(profiles[:, 3] < 0.0)
        or beta_value <= 0.0
        or not isinstance(lz_order, (int, np.integer))
        or isinstance(lz_order, (bool, np.bool_))
        or not isinstance(wc_order, (int, np.integer))
        or isinstance(wc_order, (bool, np.bool_))
        or not 8 <= int(lz_order) <= 100
        or not 8 <= int(wc_order) <= 100
    ):
        raise ValueError("distance, profiles, beta, or descriptor is invalid")
    reduced_mass, gradient_gap, gradient_product_root, curvature = descriptor
    if (
        reduced_mass <= 0.0
        or gradient_gap <= 0.0
        or gradient_product_root <= 0.0
        or curvature <= 0.0
    ):
        raise ValueError("crossing_descriptor must contain four positive values")
    reactant, mecp, product, coupling = profiles.T
    shifted_boltzmann = np.exp(-beta_value * (reactant - np.min(reactant)))
    partition = _step7_trapezoid(shifted_boltzmann, r)
    if not np.isfinite(partition) or partition <= 0.0:
        raise ValueError("the distance partition must be positive and finite")
    distance_probability = shifted_boltzmann / partition
    activation_argument = -beta_value * (mecp - reactant)
    if np.any(np.abs(activation_argument) >= 600.0):
        raise ValueError("activation exponent is outside the supported finite domain")
    lz_integrand = np.empty(r.size, dtype=float)
    wc_integrand = np.empty(r.size, dtype=float)
    for index in range(r.size):
        lz_coefficient = _oracle_integrate_lz_coefficient(
            float(coupling[index]),
            float(gradient_gap),
            float(reduced_mass),
            beta_value,
            lz_order,
        )
        energy_floor = -(
            float(mecp[index]) - max(float(reactant[index]), float(product[index]))
        )
        wc_coefficient = _oracle_integrate_wc_coefficient(
            float(coupling[index]),
            float(gradient_gap),
            float(gradient_product_root),
            float(reduced_mass),
            beta_value,
            energy_floor,
            wc_order,
            True,
        )
        activation = float(np.exp(activation_argument[index]))
        lz_integrand[index] = distance_probability[index] * activation * lz_coefficient
        wc_integrand[index] = distance_probability[index] * activation * wc_coefficient
    lz_rate = _step7_trapezoid(lz_integrand, r)
    wc_rate = _step7_trapezoid(wc_integrand, r)
    mean_distance = _step7_trapezoid(r * distance_probability, r)
    result = np.array([lz_rate, wc_rate, partition, mean_distance], dtype=float)
    if np.any(~np.isfinite(result)) or lz_rate < 0.0 or wc_rate < 0.0:
        raise ValueError("the thermal rate calculation must be finite and nonnegative")
    return result

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    scaled_helper = """def scaled_result(fn,r,p,beta,d,lz_order,wc_order,scale):
 value=fn(r,p,beta,d,lz_order,wc_order)
 if not isinstance(value,np.ndarray) or value.shape!=(4,): return np.full(4,1e100)
 if not np.issubdtype(value.dtype,np.number) or not np.isrealobj(value) or np.any(~np.isfinite(value)): return np.full(4,1e100)
 if value[0] < 0.0 or value[1] < 0.0 or value[2] <= 0.0: return np.full(4,1e100)
 return value/scale"""
    return [
        {
            "setup": "import numpy as np\nfrom scipy.special import airy\n" + scaled_helper + "\nr=np.array([4.5,4.7,4.9,5.1]); vr=.012*(r-4.8)**2; vm=.004+.008*(r-4.9)**2; vp=.0004+.011*(r-4.85)**2; v=.00035*np.exp(-1.8*(r-4.8)); p=np.column_stack([vr,vm,vp,v]); d=np.array([22000.,.09,.045,5e-6]); scale=np.array([1e-11,1e-8,1e-1,1.0])",
            "call": "scaled_result(compute_thermal_isotope_rates,r.copy(),p.copy(),3600.,d.copy(),32,40,scale)",
            "gold_call": "scaled_result(_oracle_compute_thermal_isotope_rates,r.copy(),p.copy(),3600.,d.copy(),32,40,scale)",
            "tol": 1e-8,
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import airy\n" + scaled_helper + "\nr=np.array([4.4,4.6,4.8,5.0]); vr=.01*(r-4.7)**2; vm=.0035+.006*(r-4.8)**2; vp=.0002+.009*(r-4.75)**2; v=np.zeros(4); p=np.column_stack([vr,vm,vp,v]); d=np.array([18000.,.11,.052,4e-6]); scale=np.array([1.0,1.0,1e-1,1.0])",
            "call": "scaled_result(compute_thermal_isotope_rates,r.copy(),p.copy(),2800.,d.copy(),24,32,scale)",
            "gold_call": "scaled_result(_oracle_compute_thermal_isotope_rates,r.copy(),p.copy(),2800.,d.copy(),24,32,scale)",
            "tol": 1e-8,
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import airy\n" + scaled_helper + "\nr=np.array([4.8,5.0]); vr=np.array([0.,.0002]); vm=np.array([.003,.0031]); vp=np.array([.0001,.0002]); v=np.array([.00025,.00018]); p=np.column_stack([vr,vm,vp,v]); d=np.array([15000.,.07,.039,3e-6]); scale=np.array([1e-8,1e-7,1e-1,1.0])",
            "call": "scaled_result(compute_thermal_isotope_rates,r.copy(),p.copy(),2000.,d.copy(),16,24,scale)",
            "gold_call": "scaled_result(_oracle_compute_thermal_isotope_rates,r.copy(),p.copy(),2000.,d.copy(),16,24,scale)",
            "tol": 1e-8,
        },
        {
            "setup": "import numpy as np\nfrom scipy.special import airy\nr=np.array([4.8,4.7]); p=np.ones((2,4)); d=np.ones(4)\ndef check(fn):\n try: fn(r.copy(),p.copy(),3000.,d.copy(),16,16)\n except ValueError: return 1\n except Exception: return 2\n return 0",
            "call": "check(compute_thermal_isotope_rates)",
            "gold_call": "check(_oracle_compute_thermal_isotope_rates)",
            "tol": 0.0,
        },
    ]

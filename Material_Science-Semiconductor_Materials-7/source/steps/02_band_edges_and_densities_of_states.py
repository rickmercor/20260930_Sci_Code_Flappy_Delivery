"""
Return the conduction- and valence-band edge energies and the effective densities of states of the host at a given temperature, on the fixed electron-energy scale used by the defect calculation.

Defect charge-transition levels computed at 0 K are rarely known at processing temperatures, while the band gap of a semiconductor shrinks substantially on heating. A defect-concentration calculation over a cooling path therefore needs a convention for where the two band edges sit at each temperature relative to the fixed transition levels, together with the band densities of states that set the free-carrier concentrations.

Returns
-------
np.ndarray [Ec, Ev, Nc, Nv], band-edge energies (eV) on the fixed scale with zero at the 0 K valence-band maximum and effective densities of states (cm^-3)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def band_edges_and_densities(temperature: float, host: dict) -> "np.ndarray":
    '''Band edges and effective densities of states at one temperature.

    Parameters
    ----------
    temperature : float
        Absolute temperature (K), strictly positive.
    host : dict
        Host description with keys
        "eg0" : band gap at 0 K (eV);
        "varshni_alpha" : Varshni coefficient (eV/K) of the gap temperature dependence;
        "varshni_beta" : Varshni temperature parameter (K);
        "f_cb" : fraction, between 0 and 1, of the band-gap change relative to 0 K that is
        carried by the conduction band, in the sense of the source framework's band-edge
        convention;
        "nc_ref", "nv_ref" : conduction- and valence-band effective densities of states
        (cm^-3) at the reference temperature;
        "t_ref" : reference temperature (K) for nc_ref and nv_ref.
        Other keys are ignored.

    Returns
    -------
    edges : np.ndarray
        Float array [Ec, Ev, Nc, Nv]: conduction- and valence-band edge energies (eV) on the
        framework's fixed energy scale, whose zero is the valence-band maximum at 0 K, and the
        effective densities of states (cm^-3), which scale with the three-halves power of
        temperature from their reference values.

    Raises
    ------
    ValueError
        If temperature is not positive or f_cb lies outside [0, 1].
    '''
    return edges

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_band_edges_and_densities(temperature: float, host: dict) -> "np.ndarray":
    t = float(temperature)
    if not t > 0.0:
        raise ValueError("temperature must be positive")
    f_cb = float(host["f_cb"])
    if not 0.0 <= f_cb <= 1.0:
        raise ValueError("f_cb must lie in [0, 1]")
    d_gap = -float(host["varshni_alpha"]) * t * t / (t + float(host["varshni_beta"]))
    ec = float(host["eg0"]) + f_cb * d_gap
    ev = -(1.0 - f_cb) * d_gap
    scale = (t / float(host["t_ref"])) ** 1.5
    return np.array([ec, ev, float(host["nc_ref"]) * scale, float(host["nv_ref"]) * scale])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    host_setup = """import numpy as np
host = {"eg0": 1.583, "varshni_alpha": 3.4e-4, "varshni_beta": 128.0, "f_cb": 0.78,
        "nc_ref": 7.9e17, "nv_ref": 1.69e19, "t_ref": 296.0, "hw0": 0.0119}
def scaled(v):
    # Put energies and densities of states on comparable scales for the comparison.
    v = np.asarray(v, dtype=float)
    return np.array([v[0], v[1], v[2] / 1e17, v[3] / 1e19])
"""
    raise_setup = host_setup + """
def run(fn, t, h):
    try:
        fn(t, h)
        return 0
    except ValueError:
        return 1
"""
    return [
        # --- Valid: high processing temperature ---
        {
            "setup": host_setup,
            "call": "scaled(band_edges_and_densities(1081.0, dict(host)))",
            "gold_call": "scaled(_oracle_band_edges_and_densities(1081.0, dict(host)))",
        },
        # --- Valid: reference temperature (densities equal the reference values) ---
        {
            "setup": host_setup,
            "call": "scaled(band_edges_and_densities(296.0, dict(host)))",
            "gold_call": "scaled(_oracle_band_edges_and_densities(296.0, dict(host)))",
        },
        # --- Boundary: all of the gap change carried by the valence band ---
        {
            "setup": host_setup + """
h0 = dict(host); h0["f_cb"] = 0.0
""",
            "call": "scaled(band_edges_and_densities(742.0, dict(h0)))",
            "gold_call": "scaled(_oracle_band_edges_and_densities(742.0, dict(h0)))",
        },
        # --- Edge: very low temperature (edges approach their 0 K positions) ---
        {
            "setup": host_setup,
            "call": "scaled(band_edges_and_densities(2.0, dict(host)))",
            "gold_call": "scaled(_oracle_band_edges_and_densities(2.0, dict(host)))",
        },
        # --- Invalid: non-positive temperature ---
        {
            "setup": raise_setup,
            "call": "run(band_edges_and_densities, 0.0, dict(host))",
            "gold_call": "run(_oracle_band_edges_and_densities, 0.0, dict(host))",
        },
    ]

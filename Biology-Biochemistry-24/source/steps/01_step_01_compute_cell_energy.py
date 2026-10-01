"""
Calculate the potential energy of a single cell.

The energy function captures the mechanical resistance of a cell to changes in its volume (area elasticity) and its cortical actomyosin ring/membrane (perimeter line tension). It is given by e_i = 0.5 * [(a_i - 1)^2 + kappa * (p_i - chi)^2].

Returns
-------
energy : float
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def compute_cell_energy(area: float, perimeter: float, kappa: float, chi: float) -> float:
    '''   
    Notes
    -----
    Computes the energy scalar.
    
    Parameters
    ----------
    area : float
        Current area of the cell.
    perimeter : float
        Current perimeter of the cell.
    kappa : float
        Rigidity ratio.
    chi : float
        Preferred shape index.
        
    Returns
    -------
    float
        The computed cell energy.

    Raises
    ------
    ValueError
        If any argument is not a finite real number, or if `area` or `perimeter` is negative.

    '''
    return 0.0

# =============================================================================
# GOLD SOLUTION
# =============================================================================

def _oracle_compute_cell_energy(area: float, perimeter: float, kappa: float, chi: float) -> float:
    import numpy as np
    def _num(name, value):
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a real number, got {value!r}") from exc
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value!r}")
        return value
    area = _num("area", area)
    perimeter = _num("perimeter", perimeter)
    kappa = _num("kappa", kappa)
    chi = _num("chi", chi)
    if area < 0.0:
        raise ValueError(f"area must be non-negative, got {area!r}")
    if perimeter < 0.0:
        raise ValueError(f"perimeter must be non-negative, got {perimeter!r}")
    return float(0.5 * ((area - 1.0)**2 + kappa * (perimeter - chi)**2))

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
        return [
            # --- Normal scenario ---
            {
                "setup": """import numpy as np
area = 1.2
perimeter = 3.8
kappa = 1.0
chi = 3.81""",
                "call": "compute_cell_energy(area, perimeter, kappa, chi)",
                "gold_call": "_oracle_compute_cell_energy(area, perimeter, kappa, chi)"
            },
            # --- Boundary case ---
            {
                "setup": """import numpy as np
area = 1.0
perimeter = 3.81
kappa = 0.0
chi = 3.81""",
                "call": "compute_cell_energy(area, perimeter, kappa, chi)",
                "gold_call": "_oracle_compute_cell_energy(area, perimeter, kappa, chi)"
            },
            # --- Edge case ---
            {
                "setup": """import numpy as np
area = 0.0
perimeter = 0.0
kappa = 10.0
chi = 0.0""",
                "call": "compute_cell_energy(area, perimeter, kappa, chi)",
                "gold_call": "_oracle_compute_cell_energy(area, perimeter, kappa, chi)"
            }
        ]

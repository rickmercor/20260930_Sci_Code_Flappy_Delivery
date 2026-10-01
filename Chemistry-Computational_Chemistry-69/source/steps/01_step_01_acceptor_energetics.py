"""
Step 01 - Vacuum-referenced and hydroxide-referenced acceptor energetics.

Electrochemical potentials are quoted against a reference electrode, but a microscopic rate theory needs free energies on an absolute scale, and the two differ by a constant that has to be put in by hand. The standard hydrogen electrode sits a fixed distance below a free electron at rest in vacuum, so a reduction potential quoted versus that electrode converts into the free energy of taking an electron out of vacuum and putting it on the acceptor.

That absolute number is not the driving force of the reaction studied here. The electron does not come from vacuum; it comes from a hydroxide ion sitting in water, and the hydroxide couple has its own potential. The free energy of the concerted transfer in bulk water is set by the difference of the two potentials, and that difference is what the rest of the calculation builds on.

Keeping the two apart matters. The absolute quantity is large and negative for almost any acceptor, which makes every reaction look downhill if it is mistaken for the driving force. The bulk reaction free energy is the one that is strongly positive for a sluggish acceptor, and its sign is exactly the puzzle the interface has to resolve.

Returns
-------
numpy.ndarray of shape (2,), free energies in kcal/mol
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np
import numpy.typing as npt


def acceptor_energetics(E0_acceptor: float) -> np.ndarray:
    '''Vacuum-referenced and hydroxide-referenced free energies of an acceptor.

    Parameters
    ----------
    E0_acceptor : float
        Standard one-electron reduction potential of the acceptor couple,
        in volts versus the standard hydrogen electrode.

    Returns
    -------
    energetics : numpy.ndarray
        Array of shape (2,) in kcal/mol. Element 0 is the free energy change
        for reducing the acceptor with an electron taken from rest in vacuum.
        Element 1 is the standard free energy change of the bulk-water
        reaction in which hydroxide is the electron donor and the hydroxyl
        radical is the oxidised product.

    Raises
    ------
    ValueError
        If E0_acceptor is not finite.
    '''
    return energetics

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np
import numpy.typing as npt

def _oracle_acceptor_energetics(E0_acceptor: float) -> np.ndarray:
    """Vacuum-referenced and hydroxide-referenced acceptor free energies."""
    _F_KCAL = 23.060548
    _E_ABS_SHE = 4.44
    _E0_OH = 1.9
    import numpy as np
    E = float(E0_acceptor)
    if not np.isfinite(E):
        raise ValueError("E0_acceptor must be finite")
    dG_absolute = -_F_KCAL * (E + _E_ABS_SHE)
    dG_bulk = -_F_KCAL * (E - _E0_OH)
    return np.array([dG_absolute, dG_bulk], dtype=float)

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        # --- Normal: the hexaamminecobalt(III) couple of the target system ---
        {
            "setup": "import numpy as np\nE0 = 0.108\n",
            "call": "acceptor_energetics(E0)",
            "gold_call": "_oracle_acceptor_energetics(E0)",
        },
        # --- Normal: a strongly oxidising acceptor above the hydroxyl couple ---
        {
            "setup": "import numpy as np\nE0 = 2.35\n",
            "call": "acceptor_energetics(E0)",
            "gold_call": "_oracle_acceptor_energetics(E0)",
        },
        # --- Boundary: potential of the hydroxyl couple itself, bulk term vanishes ---
        {
            "setup": "import numpy as np\nE0 = 1.90\n",
            "call": "acceptor_energetics(E0)",
            "gold_call": "_oracle_acceptor_energetics(E0)",
        },
        # --- Edge: strongly reducing acceptor near the alkali-metal limit ---
        {
            "setup": "import numpy as np\nE0 = -3.04\n",
            "call": "acceptor_energetics(E0)",
            "gold_call": "_oracle_acceptor_energetics(E0)",
        },
    ]

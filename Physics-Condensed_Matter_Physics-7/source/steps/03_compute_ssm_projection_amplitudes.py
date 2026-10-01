"""
Construct overlaps of spin-$z$ basis words with the two projection sectors used for the periodic Shastry–Sutherland lattice. Bit $x+Ly$ is zero for spin up and one for spin down. The visible helper fixes the oriented diagonal and horizontal dimer covers.

Construct overlaps of spin-$z$ basis words with the two projection sectors used for the periodic Shastry–Sutherland lattice. Bit $x+Ly$ is zero for spin up and one for spin down. Let $L$ be $lattice_size$, reduce neighbor coordinates modulo $L$, and label site $(x,y)$ by $x+Ly$. Order diagonal dimers by family $f=0$ then $f=1$, and within each family by increasing $y$ then increasing $x$. Anchors have even $x$ and $(x+y)$ mod 2 equal to $f$. Their oriented partners are $(x+1,y+1)$ for $f=0$ and $(x-1,y+1)$ for $f=1$. Order horizontal dimers by increasing $y$ then increasing $x$ over anchors with even $x+y$; their partners are $(x+1,y)$ for even $x$ and $(x-1,y)$ for odd $x$. In each oriented pair, the anchor is $i$ and its partner is $j$. The one-triplet coordinates follow this diagonal-dimer order.

Returns
-------
np.ndarray, a float array of shape (n_states, 4 + lattice_size**2//2)
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

import numpy as np

def compute_ssm_projection_amplitudes(
    state_words: np.ndarray,
    lattice_size: int,
) -> np.ndarray:
    """Compute the two Shastry-Sutherland projection-sector amplitudes.

    Parameters
    ----------
    state_words : np.ndarray
        Nonempty one-dimensional nonnegative integer array. Bit
        ``x + lattice_size*y`` is zero for spin up and one for spin down.
    lattice_size : int
        Even periodic linear size in ``[2, 6]``.

    Returns
    -------
    amplitudes : np.ndarray
        Float array with one row per state word. The first four columns are,
        in order, the alternating single-site Q=+1 product, the product of
        diagonal triplets, the product of horizontal singlets, and the
        product of horizontal triplets. The remaining columns contain the
        Q=-1 states with one diagonal triplet, in diagonal-cover order, and
        singlets on every other diagonal dimer.
        The alternating single-site product is unnormalized: its local state
        is (|up> + |down>) for x+y even and (|down> - |up>) for x+y odd.
        Its amplitude contributes a factor -1 at every odd-sublattice site
        whose word bit is 0, and +1 otherwise.

    Raises
    ------
    ValueError
        If the lattice size or encoded words are invalid.
    """
    return amplitudes

# =============================================================================
# GOLD SOLUTION
# =============================================================================

from numbers import Integral

import numpy as np


def _oracle_compute_ssm_projection_amplitudes(
    state_words: np.ndarray,
    lattice_size: int,
) -> np.ndarray:
    from numbers import Integral
    import numpy as np

    if (
        isinstance(lattice_size, bool)
        or not isinstance(lattice_size, Integral)
        or int(lattice_size) < 2
        or int(lattice_size) > 6
        or int(lattice_size) % 2 != 0
    ):
        raise ValueError("lattice_size must be an even integer in [2,6]")
    L = int(lattice_size)
    raw = np.asarray(state_words)
    if (
        raw.ndim != 1
        or raw.size < 1
        or not np.issubdtype(raw.dtype, np.integer)
        or np.issubdtype(raw.dtype, np.bool_)
        or np.any(raw < 0)
    ):
        raise ValueError("state_words must be a nonempty 1D nonnegative integer array")
    limit = 1 << (L * L)
    if any(int(word) >= limit for word in raw):
        raise ValueError("a state word has a bit outside the lattice")

    diagonal = []
    for family in (0, 1):
        for y in range(L):
            for x in range(L):
                if x % 2 != 0 or (x + y) % 2 != family:
                    continue
                jx = (x + 1) % L if family == 0 else (x - 1) % L
                jy = (y + 1) % L
                diagonal.append((x + L * y, jx + L * jy, x, y))
    horizontal = []
    for y in range(L):
        for x in range(L):
            if (x + y) % 2 != 0:
                continue
            jx = (x + 1) % L if x % 2 == 0 else (x - 1) % L
            horizontal.append((x + L * y, jx + L * y, x, y))

    n_diagonal = L * L // 2
    out = np.zeros((raw.size, 4 + n_diagonal), dtype=float)
    for row, encoded in enumerate(raw):
        word = int(encoded)
        spins = np.asarray(
            [-1 if (word >> site) & 1 else 1 for site in range(L * L)],
            dtype=int,
        )

        single_site = 1.0
        for site in range(L * L):
            x = site % L
            y = site // L
            if (x + y) % 2 == 1:
                single_site *= -1.0 if spins[site] == 1 else 1.0
        out[row, 0] = single_site

        diagonal_singlets = []
        diagonal_compatible = True
        diagonal_singlet_product = 1.0
        for i, j, x, y in diagonal:
            if spins[i] == spins[j]:
                diagonal_compatible = False
                break
            phase = -1.0 if (x + y) % 2 else 1.0
            local_singlet = phase * (1.0 if spins[i] == 1 else -1.0)
            diagonal_singlets.append(local_singlet)
            diagonal_singlet_product *= local_singlet
        if diagonal_compatible:
            out[row, 1] = 1.0
            for dimer, local_singlet in enumerate(diagonal_singlets):
                out[row, 4 + dimer] = diagonal_singlet_product / local_singlet

        horizontal_compatible = True
        horizontal_singlet_product = 1.0
        for i, j, _x, _y in horizontal:
            if spins[i] == spins[j]:
                horizontal_compatible = False
                break
            horizontal_singlet_product *= 1.0 if spins[i] == 1 else -1.0
        if horizontal_compatible:
            out[row, 2] = horizontal_singlet_product
            out[row, 3] = 1.0
    return out

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    """Return list of test case specifications."""
    return [
        {
            "setup": "import numpy as np\nwords=np.array([6,9,0,15],dtype=np.int64)",
            "call": "compute_ssm_projection_amplitudes(words.copy(),2)",
            "gold_call": "_oracle_compute_ssm_projection_amplitudes(words.copy(),2)",
        },
        {
            "setup": """import numpy as np
L=4
d=np.array([[0,5],[2,7],[8,13],[10,15],[4,11],[6,9],[12,3],[14,1]],dtype=int)
words=[]
for code in (11,84,157,230):
 word=0
 for a,(i,j) in enumerate(d):
  word |= 1 << (i if ((code>>a)&1) else j)
 words.append(word)
words=np.array(words,dtype=np.int64)""",
            "call": "compute_ssm_projection_amplitudes(words.copy(),L)",
            "gold_call": "_oracle_compute_ssm_projection_amplitudes(words.copy(),L)",
        },
        {
            "setup": "import numpy as np\nwords=np.array([0,65535],dtype=np.int64)",
            "call": "compute_ssm_projection_amplitudes(words.copy(),4)",
            "gold_call": "_oracle_compute_ssm_projection_amplitudes(words.copy(),4)",
        },
        {
            "setup": """import numpy as np
L=6
word=0
for family in (0,1):
 for y in range(L):
  for x in range(L):
   if x%2==0 and (x+y)%2==family:
    jx=(x+1)%L if family==0 else (x-1)%L
    jy=(y+1)%L
    word |= 1 << (jx+L*jy)
words=np.array([word],dtype=np.int64)""",
            "call": "compute_ssm_projection_amplitudes(words.copy(),L)",
            "gold_call": "_oracle_compute_ssm_projection_amplitudes(words.copy(),L)",
        },
        {
            "setup": """import numpy as np
words=np.array([0],dtype=np.int64)
def run_model():
 try:
  compute_ssm_projection_amplitudes(words.copy(),3)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_compute_ssm_projection_amplitudes(words.copy(),3)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
        {
            "setup": """import numpy as np
words=np.array([1<<16],dtype=np.int64)
def run_model():
 try:
  compute_ssm_projection_amplitudes(words.copy(),4)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2
def run_gold():
 try:
  _oracle_compute_ssm_projection_amplitudes(words.copy(),4)
  return 0
 except ValueError:
  return 1
 except Exception:
  return 2""",
            "call": "run_model()",
            "gold_call": "run_gold()",
        },
    ]

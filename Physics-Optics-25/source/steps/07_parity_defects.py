"""
Even and odd parity defects of two real arrays.

The even defect measures how far two arrays are from being equal and the odd
defect how far they are from being exact negatives, each relative to the
complementary combination.

Returns
-------
A finite array (even_defect, odd_defect).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def parity_defects(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the even and odd parity defects of two same-shape real arrays.

    first and second must be finite real numeric arrays (not Boolean or
    complex) with identical shapes and at least one element. With Euclidean
    norms over all elements, the even defect is
    ||first-second|| / max(||first+second||, 1e-12) and the odd defect is
    ||first+second|| / max(||first-second||, 1e-12). Raise ValueError for any
    contract violation. Return np.array([even_defect, odd_defect]).
    """
    return result

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import numpy as np

def _oracle_parity_defects(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = np.asarray(first)
    right = np.asarray(second)
    for value in (left, right):
        if value.dtype.kind not in "iuf":
            raise ValueError("inputs must be real numeric arrays")
    if left.shape != right.shape or left.size < 1:
        raise ValueError("inputs must share one nonempty shape")
    left = left.astype(float)
    right = right.astype(float)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError("inputs must be finite")
    difference = float(np.linalg.norm((left - right).ravel()))
    total = float(np.linalg.norm((left + right).ravel()))
    return np.array([difference / max(total, 1e-12), total / max(difference, 1e-12)])

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    return [{'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'parity_defects(a, b)',
      'gold_call': '_oracle_parity_defects(a, b)'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'parity_defects(c, d)',
      'gold_call': '_oracle_parity_defects(c, d)'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'parity_defects(m.copy(), n.copy())',
      'gold_call': '_oracle_parity_defects(m.copy(), n.copy())'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'parity_defects(m.copy(), m[::-1].copy())',
      'gold_call': '_oracle_parity_defects(m.copy(), m[::-1].copy())'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'parity_defects(z.copy(), z.copy())',
      'gold_call': '_oracle_parity_defects(z.copy(), z.copy())'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': '(lambda r: np.array([np.all(np.isfinite(r)), r[0] > 1e11, r[1] == 0.0], '
              'dtype=int))(parity_defects(c.copy(), (-c).copy()))',
      'gold_call': '(lambda r: np.array([np.all(np.isfinite(r)), r[0] > 1e11, r[1] == 0.0], '
                   'dtype=int))(_oracle_parity_defects(c.copy(), (-c).copy()))'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'parity_defects(np.array([3,1]), np.array([3,-2]))',
      'gold_call': '_oracle_parity_defects(np.array([3,1]), np.array([3,-2]))'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(parity_defects, a.copy(), c.copy())',
      'gold_call': 'rejects(_oracle_parity_defects, a.copy(), c.copy())'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(parity_defects, a[:0].copy(), a[:0].copy())',
      'gold_call': 'rejects(_oracle_parity_defects, a[:0].copy(), a[:0].copy())'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(parity_defects, a.copy(), np.array([0.1, np.nan, 0.2, 0.3, 0.4]))',
      'gold_call': 'rejects(_oracle_parity_defects, a.copy(), np.array([0.1, np.nan, 0.2, 0.3, 0.4]))'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(parity_defects, np.array([np.inf, 0.0]), c.copy())',
      'gold_call': 'rejects(_oracle_parity_defects, np.array([np.inf, 0.0]), c.copy())'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(parity_defects, (c + 1j).copy(), c.copy())',
      'gold_call': 'rejects(_oracle_parity_defects, (c + 1j).copy(), c.copy())'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': 'rejects(parity_defects, np.array([True, False]), c.copy())',
      'gold_call': 'rejects(_oracle_parity_defects, np.array([True, False]), c.copy())'},
     {'setup': 'import numpy as np\n'
               'a=np.array([.31,-.12,.07,.44,-.2]); b=np.array([.29,-.1,.09,.47,-.23])\n'
               'c=np.array([.6,-1.3]); d=np.array([-.55,1.36])\n'
               'm=np.array([[.2,-.4],[.05,.3],[-.1,.6]]); '
               'n=np.array([[-.21,.38],[.02,-.33],[.12,-.58]])\n'
               'z=np.zeros(3)\n'
               'def rejects(fn,*v):\n'
               ' try: fn(*v)\n'
               ' except ValueError: return 1\n'
               ' return 0\n',
      'call': "rejects(parity_defects, np.array(['a', 'b']), c.copy())",
      'gold_call': "rejects(_oracle_parity_defects, np.array(['a', 'b']), c.copy())"}]

"""
This stage assembles the weight matrix and the site numbers from the drainage map, and records the two structural facts the later stages rely on: the out-strength q_i = sum_j w_ij of every patch, which is the total rate factor with which an explorer leaves it, and whether the directed network is strongly connected. Persistence theory for the effective dynamics needs an irreducible colonisation kernel, and the kernel inherits irreducibility from strong connectivity of the dispersal network.

A drainage map is valid only if it has exactly one outlet and every downstream path reaches that outlet without revisiting a patch; a map with a cycle does not describe a river.

A metapopulation lives on a set of habitat patches joined by a weighted, directed dispersal network. The weight w_ij of the link from patch i to patch j sets how readily a mobile explorer moves from i to j, and w_ii = 0. In a river landscape the backbone of that network is the drainage tree: every patch except the outlet drains into exactly one downstream neighbour, explorers drift with the current along the downstream link and move against it along the reverse link with a smaller weight, and a few one-way overland links, carried by a prevailing wind for instance, join otherwise distant headwaters.

Patches differ in the number M_i of colonisable sites they hold. In a river network the habitat a reach offers grows with the area it drains, so the number of sites is taken proportional to the drainage count of the patch, the number of patches whose downstream path passes through it, the patch itself included.

Returns
-------
dict, the assembled dispersal network and patch site numbers, keyed by weights (N by N link weights), sites (the M_i), drainage (integer drainage counts), out_strength (the q_i), n_links (the integer number of non-zero weights), and strongly_connected (the integer 1 if every patch reaches every other along directed links, else 0).
"""

# =============================================================================
# FUNCTION SIGNATURE (shown to the LLM)
# =============================================================================

def build_dispersal_landscape(
    downstream: tuple,
    downstream_weight: float,
    upstream_weight: float,
    extra_links: tuple,
    sites_per_patch_drained: float,
) -> dict:
    """Assemble the directed dispersal network and the patch site numbers of a river landscape.

    Parameters
    ----------
    downstream : sequence of int
        Downstream neighbour of each patch, -1 for the outlet.
    downstream_weight : float
        Weight of each downstream link, above zero.
    upstream_weight : float
        Weight of each upstream link, not below zero.
    extra_links : sequence of (int, int, float)
        Additional one-way links (i, j, w).
    sites_per_patch_drained : float
        Colonisable sites per patch of drainage count, above zero.

    Returns
    -------
    dict
        Under the keys weights, sites, drainage, out_strength, n_links and strongly_connected; strongly_connected is the integer 1 if every patch reaches every other along directed links and 0 otherwise.

    Raises
    ------
    ValueError
        When the drainage map has other than one outlet, points outside the patch set, points a patch at itself or contains a cycle, when a weight or the site density fails to be finite with the required sign, or when an extra link is malformed, repeats an ordered pair or duplicates a river link.
    """
    return

# =============================================================================
# GOLD SOLUTION
# =============================================================================

import math

import numpy as np


def _reaches_all(adjacency: np.ndarray, start: int) -> bool:
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        for nxt in np.nonzero(adjacency[node])[0]:
            nxt = int(nxt)
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == adjacency.shape[0]


def _oracle_build_dispersal_landscape(
    downstream: tuple,
    downstream_weight: float,
    upstream_weight: float,
    extra_links: tuple,
    sites_per_patch_drained: float,
) -> dict:
    """Reference implementation."""
    try:
        parent = [int(p) for p in downstream]
    except (TypeError, ValueError):
        raise ValueError("downstream must be a sequence of integers")
    for raw, p in zip(downstream, parent):
        if isinstance(raw, bool) or float(raw) != p:
            raise ValueError("downstream must be a sequence of integers")
    n = len(parent)
    if n < 2:
        raise ValueError("the landscape needs at least two patches")
    if sum(1 for p in parent if p == -1) != 1:
        raise ValueError("the drainage map must have exactly one outlet")
    for i, p in enumerate(parent):
        if p != -1 and not 0 <= p < n:
            raise ValueError("downstream index outside the patch set")
        if p == i:
            raise ValueError("a patch cannot drain into itself")
    wd = float(downstream_weight)
    wu = float(upstream_weight)
    rho = float(sites_per_patch_drained)
    if not (math.isfinite(wd) and wd > 0.0):
        raise ValueError("downstream_weight must be finite and above zero")
    if not (math.isfinite(wu) and wu >= 0.0):
        raise ValueError("upstream_weight must be finite and not below zero")
    if not (math.isfinite(rho) and rho > 0.0):
        raise ValueError("sites_per_patch_drained must be finite and above zero")

    drainage = np.zeros(n, dtype=int)
    for i in range(n):
        node, steps = i, 0
        while node != -1:
            drainage[node] += 1
            node = parent[node]
            steps += 1
            if steps > n:
                raise ValueError("the drainage map contains a cycle")

    weights = np.zeros((n, n))
    for i, p in enumerate(parent):
        if p != -1:
            weights[i, p] = wd
            weights[p, i] = wu
    river = weights > 0.0
    seen = set()
    for link in extra_links:
        if len(link) != 3:
            raise ValueError("each extra link must be a triple (i, j, w)")
        i, j, w = link
        if isinstance(i, bool) or isinstance(j, bool) or float(i) != int(i) or float(j) != int(j):
            raise ValueError("extra link endpoints must be integers")
        i, j, w = int(i), int(j), float(w)
        if not (0 <= i < n and 0 <= j < n) or i == j:
            raise ValueError("extra link endpoints must be distinct patches")
        if not (math.isfinite(w) and w > 0.0):
            raise ValueError("extra link weights must be finite and above zero")
        if (i, j) in seen:
            raise ValueError("extra links repeat an ordered pair")
        if river[i, j]:
            raise ValueError("extra link duplicates a river link")
        seen.add((i, j))
        weights[i, j] = w

    adjacency = weights > 0.0
    connected = all(_reaches_all(adjacency, s) for s in range(n))
    return {
        "weights": weights,
        "sites": rho * drainage.astype(float),
        "drainage": drainage,
        "out_strength": weights.sum(axis=1),
        "n_links": int(np.count_nonzero(weights)),
        "strongly_connected": int(connected),
    }

# =============================================================================
# TEST CASES
# =============================================================================

def test_cases():
    FLAT = """
    def flat(x):
        # flatten to a tuple of plain numeric terminals
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
        if isinstance(x, float):
            return (round(x, 12),)
        return (x,)
    """
    import textwrap as _textwrap
    FLAT = _textwrap.dedent(FLAT)
    return [
        {
            # the thirteen-patch river with two one-way overland links
            "setup": """
DOWN = (-1, 0, 0, 1, 1, 2, 2, 3, 3, 5, 6, 6, 10)
LINKS = ((7, 12, 0.6), (11, 4, 0.5))
""" + FLAT,
            "call": "flat(build_dispersal_landscape(DOWN, 1.0, 0.3, LINKS, 40.0))",
            "gold_call": "flat(_oracle_build_dispersal_landscape(DOWN, 1.0, 0.3, LINKS, 40.0))",
        },
        {
            # boundary: no upstream movement leaves only the outlet unable to reach anything else,
            # so the network is not strongly connected; a chain drains cumulatively
            "setup": """
DOWN = (-1, 0, 1, 2, 3)
""" + FLAT,
            "call": "flat((build_dispersal_landscape(DOWN, 2.0, 0.0, (), 3.0), build_dispersal_landscape(DOWN, 2.0, 0.0, ((0, 4, 0.1),), 3.0)['strongly_connected']))",
            "gold_call": "flat((_oracle_build_dispersal_landscape(DOWN, 2.0, 0.0, (), 3.0), _oracle_build_dispersal_landscape(DOWN, 2.0, 0.0, ((0, 4, 0.1),), 3.0)['strongly_connected']))",
        },
        {
            # a star whose outlet drains every other patch, with a reversed overland pair
            "setup": """
DOWN = (3, 3, 3, -1, 3, 3)
LINKS = ((0, 1, 0.25), (1, 0, 0.75), (4, 5, 0.5))
""" + FLAT,
            "call": "flat(build_dispersal_landscape(DOWN, 0.9, 0.45, LINKS, 12.5))",
            "gold_call": "flat(_oracle_build_dispersal_landscape(DOWN, 0.9, 0.45, LINKS, 12.5))",
        },
        {
            "setup": """
def verdict(fn, **kw):
    args = dict(downstream=(-1, 0, 0, 1), downstream_weight=1.0, upstream_weight=0.3, extra_links=((3, 2, 0.2),), sites_per_patch_drained=10.0)
    args.update(kw)
    try:
        fn(**args)
    except ValueError:
        return 1
    except Exception:
        return 2
    return 0
""",
            "call": "(verdict(build_dispersal_landscape, downstream=(-1, -1, 0, 1)), verdict(build_dispersal_landscape, downstream=(1, 2, 0, 1)), verdict(build_dispersal_landscape, downstream=(-1, 0, 2, 1)), verdict(build_dispersal_landscape, downstream=(-1, 0, 0, 7)), verdict(build_dispersal_landscape, extra_links=((3, 1, 0.2),)), verdict(build_dispersal_landscape, extra_links=((2, 2, 0.2),)), verdict(build_dispersal_landscape, extra_links=((3, 2, 0.2), (3, 2, 0.4))), verdict(build_dispersal_landscape, upstream_weight=-0.1), verdict(build_dispersal_landscape, sites_per_patch_drained=float('nan')), verdict(build_dispersal_landscape))",
            "gold_call": "(verdict(_oracle_build_dispersal_landscape, downstream=(-1, -1, 0, 1)), verdict(_oracle_build_dispersal_landscape, downstream=(1, 2, 0, 1)), verdict(_oracle_build_dispersal_landscape, downstream=(-1, 0, 2, 1)), verdict(_oracle_build_dispersal_landscape, downstream=(-1, 0, 0, 7)), verdict(_oracle_build_dispersal_landscape, extra_links=((3, 1, 0.2),)), verdict(_oracle_build_dispersal_landscape, extra_links=((2, 2, 0.2),)), verdict(_oracle_build_dispersal_landscape, extra_links=((3, 2, 0.2), (3, 2, 0.4))), verdict(_oracle_build_dispersal_landscape, upstream_weight=-0.1), verdict(_oracle_build_dispersal_landscape, sites_per_patch_drained=float('nan')), verdict(_oracle_build_dispersal_landscape))",
        },
    ]

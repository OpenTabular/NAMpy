"""Soap-film PDE bases, following mgcv R/soap.r and src/soap.c.

The signed grid indices, boundary traversal order and knot RHS indexing are
intentional: changing them changes the upstream finite-difference basis.

References
----------
- Wood, S. N., Bravington, M. V., and Hedley, S. L. (2008). Soap film smoothing.
  https://doi.org/10.1111/j.1467-9868.2008.00665.x
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from .univariate.cr import (
    cyclic_cubic_bd,
    cyclic_cubic_predict_matrix,
    place_knots_through_values,
)
from .univariate.ps import (
    cyclic_pspline_design,
    cyclic_pspline_difference_penalty,
    cyclic_pspline_knots,
)


def process_boundary(boundary):
    """Close boundary loops and compute cumulative segment distances."""
    if not isinstance(boundary, (list, tuple)) or not boundary:
        raise ValueError("bnd must be a nonempty list of boundary loops")
    loops = []
    for loop in boundary:
        x = np.asarray(loop["x"], dtype=float).ravel().copy()
        y = np.asarray(loop["y"], dtype=float).ravel().copy()
        if x.size != y.size or x.size < 3:
            raise ValueError("boundary x and y must have the same length, at least 3")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("boundary coordinates must be finite")
        f = (
            None
            if loop.get("f") is None
            else np.asarray(loop["f"], dtype=float).ravel().copy()
        )
        if f is not None and (f.size != x.size or not np.all(np.isfinite(f))):
            raise ValueError(
                "fixed boundary values must be finite and match coordinates"
            )
        if x[0] != x[-1] or y[0] != y[-1]:
            x, y = np.r_[x, x[0]], np.r_[y, y[0]]
            if f is not None:
                f = np.r_[f, f[0]]
        d = np.r_[0, np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))]
        loops.append({"x": x, "y": y, "d": d, "f": f})
    return loops


def inside_boundary(boundary, x, y):
    """The crossing convention of mgcv src/misc.c::in_out (including edges)."""
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    count = np.zeros(x.shape, dtype=int)
    for loop in boundary:
        bx, by = loop["x"], loop["y"]
        for i in range(len(bx)):
            j = (i + 1) % len(bx)
            x0, x1, y0, y1 = bx[i], bx[j], by[i], by[j]
            if x0 == x1:
                continue
            if x1 < x0:
                x0, x1, y0, y1 = x1, x0, y1, y0
            possible = (x0 < x) & (x1 >= x)
            definite = (y0 <= y) & (y1 <= y)
            mixed = ~((y0 > y) & (y1 > y)) & ~definite
            intercept = (x - x0) * (y1 - y0) / (x1 - x0) + y0
            count += possible & (definite | (mixed & (y >= intercept)))
    return count % 2 == 1


def boundary_grid(boundary, nmax):
    """Port soap.c::boundary, retaining its distance-to-segment arithmetic."""
    x0 = min(np.min(loop["x"]) for loop in boundary)
    x1 = max(np.max(loop["x"]) for loop in boundary)
    y0 = min(np.min(loop["y"]) for loop in boundary)
    y1 = max(np.max(loop["y"]) for loop in boundary)
    if nmax < 3 or int(nmax) != nmax or max(x1 - x0, y1 - y0) <= 0:
        raise ValueError(
            "soap grid needs integer nmax >= 3 and a nondegenerate boundary"
        )
    h = max(x1 - x0, y1 - y0) / (nmax - 1)
    nx = int(nmax) if x1 - x0 > y1 - y0 else int(np.ceil((x1 - x0) / h)) + 1
    ny = int(nmax) if y1 - y0 >= x1 - x0 else int(np.ceil((y1 - y0) / h)) + 1
    # C accumulates coordinates rather than multiplying their indices.
    gx, gy = np.empty(nx), np.empty(ny)
    gx[0], gy[0] = x0, y0
    for i in range(1, nx):
        gx[i] = gx[i - 1] + h
    for i in range(1, ny):
        gy[i] = gy[i - 1] + h
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    G = np.where(inside_boundary(boundary, xx, yy), 1, -nx * ny - 10)
    dto = np.full((nx, ny), -1.0)
    d = np.empty(nx * ny)
    nb = []
    ii = 0
    xb0, yb0 = x0 - h / 2, y0 - h / 2
    for loop in boundary:
        initial = ii
        dist = 0.0
        for seg in range(1, len(loop["x"])):
            ax, ay = loop["x"][seg - 1], loop["y"][seg - 1]
            bx, by = loop["x"][seg], loop["y"][seg]
            # Vertical first, then horizontal, exactly as boundary().
            for axis in (0, 1):
                reversed_segment = (ax >= bx) if axis == 0 else (ay >= by)
                if reversed_segment:
                    sx, sy, ex, ey = bx, by, ax, ay
                else:
                    sx, sy, ex, ey = ax, ay, bx, by
                lower, upper = (sx, ex) if axis == 0 else (sy, ey)
                base = xb0 if axis == 0 else yb0
                j0 = int(np.ceil((lower - base) / h))
                j1 = int(np.floor((upper - base) / h))
                if upper == lower:
                    continue
                gradient = (ey - sy) / (ex - sx) if axis == 0 else (ex - sx) / (ey - sy)
                for j in range(j0, j1 + 1):
                    line = base + j * h
                    cross = (sy if axis == 0 else sx) + (line - lower) * gradient
                    other_base = yb0 if axis == 0 else xb0
                    other = int(np.floor((cross - other_base) / h))
                    cells = (
                        [(j - 1, other), (j, other)]
                        if axis == 0
                        else [(other, j - 1), (other, j)]
                    )
                    for ix, iy in cells:
                        if not (0 <= ix < nx and 0 <= iy < ny):
                            raise ValueError("boundary crosses outside soap grid")
                        if G[ix, iy] > 0 or G[ix, iy] < -nx * ny:
                            G[ix, iy] = -ii
                            ii += 1
                    # C visits the higher-index cell first for distances.
                    for ix, iy in cells[::-1]:
                        vx, vy = ex - sx, ey - sy
                        cx, cy = ix * h + x0 - sx, iy * h + y0 - sy
                        length2 = vy * vy + vx * vx
                        b = (cx * vx + cy * vy) / length2
                        px, py = vx * b + sx, vy * b + sy
                        coordinate = px if axis == 0 else py
                        if coordinate < lower:
                            px, py = sx, sy
                        if coordinate > upper:
                            px, py = ex, ey
                        # Upstream compares translated centre to absolute projection.
                        distance_to = np.sqrt((px - cx) ** 2 + (py - cy) ** 2)
                        if distance_to < dto[ix, iy] or dto[ix, iy] < 0:
                            dto[ix, iy] = distance_to
                            along = np.sqrt((px - sx) ** 2 + (py - sy) ** 2)
                            d[-G[ix, iy]] = dist + (
                                np.sqrt(length2) - along if reversed_segment else along
                            )
            dist += np.sqrt((bx - ax) ** 2 + (by - ay) ** 2)
        nb.append(ii - initial)
    d = d[:ii]
    for ix in range(nx):
        for iy in range(ny):
            if G[ix, iy] > 0:
                G[ix, iy] = ii
                ii += 1
    return G, d, np.asarray(nb), float(x0), float(y0), float(h), ii


def pde_matrix(G, h):
    """Port soap.c::pde_coeffs to a CSC sparse matrix."""
    nx, ny = G.shape
    outside = -nx * ny - 1
    h2 = 1.0 / (h * h)
    rows, cols, values = [], [], []

    def add(i, j, value):
        rows.append(i)
        cols.append(j)
        values.append(value)

    for ix in range(nx):
        for iy in range(ny):
            index = G[ix, iy]
            if index <= outside:
                continue
            if index <= 0:
                add(-index, -index, 1.0)
                continue
            diagonal = 0.0
            if 0 < ix < nx - 1:
                a, b = G[ix - 1, iy], G[ix + 1, iy]
                if a > outside and b > outside:
                    diagonal += 2 * h2
                    add(index, abs(a), -h2)
                    add(index, abs(b), -h2)
            if 0 < iy < ny - 1:
                a, b = G[ix, iy - 1], G[ix, iy + 1]
                if a > outside and b > outside:
                    diagonal += 2 * h2
                    add(index, abs(a), -h2)
                    add(index, abs(b), -h2)
                if diagonal > h2 * 0.5:
                    add(index, index, diagonal)
    return csc_matrix((values, (rows, cols)))


def gridder(x, y, values, G, x0, y0, h):
    """Port soap.c::gridder, including its incomplete-cell nearest neighbour."""
    x, y = np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
    values = np.asarray(values)
    scalar = values.ndim == 1
    if scalar:
        values = values[:, None]
    nx, ny = G.shape
    result = np.full((len(x), values.shape[1]), np.nan)
    for row, (xx, yy) in enumerate(zip(x, y, strict=True)):
        if not np.isfinite(xx) or not np.isfinite(yy):
            continue
        ix, iy = int(np.floor((xx - x0) / h)), int(np.floor((yy - y0) / h))
        cells = [(ix, iy), (ix, iy + 1), (ix + 1, iy + 1), (ix + 1, iy)]
        good = [0 <= a < nx and 0 <= b < ny and G[a, b] >= -nx * ny for a, b in cells]
        local = [
            values[abs(G[a, b])] if ok else None
            for (a, b), ok in zip(cells, good, strict=True)
        ]
        xa, ya = xx - x0 - ix * h, yy - y0 - iy * h
        if all(good):
            g00, g01, g11, g10 = local
            result[row] = (
                g00
                + (g10 - g00) / h * xa
                + (g01 - g00) / h * ya
                + (g11 - g10 - g01 + g00) / (h * h) * xa * ya
            )
        elif any(good):
            distance = 4 * h * h
            for i, ok in enumerate(good):
                if not ok:
                    continue
                if i in (1, 3):
                    ya = h - ya
                if i == 2:
                    xa = h - xa
                candidate = xa * xa + ya * ya
                if i == 0 or candidate < distance:
                    result[row] = local[i]
                    distance = candidate
    return result[:, 0] if scalar else result


@dataclass
class SoapBasisSetup:
    basis_train: np.ndarray
    penalties: list
    ranks: list
    null_space_dim: int
    need_constraint: bool
    offset: np.ndarray | None
    irng: np.ndarray
    boundary: list
    G: np.ndarray
    x0: float
    y0: float
    h: float
    grid_basis: np.ndarray
    grid_offset: np.ndarray | None
    basis: str


def build_soap_basis(
    x, y, *, boundary, knots, k=10, nmax=200, basis="so", boundary_spec=None
):
    """Construct a raw, range-scaled so/sf/sw basis and full-width penalties."""
    if basis not in {"so", "sf", "sw"}:
        raise ValueError("soap basis must be so, sf or sw")
    boundary = process_boundary(boundary)
    if knots is None:
        raise ValueError("knots must be specified for soap")
    if isinstance(knots, dict):
        knots = np.column_stack([knots["x"], knots["y"]])
    knots = np.asarray(knots, dtype=float).reshape(-1, 2)
    if basis != "sf" and len(knots) < 1:
        raise ValueError("need at least one interior knot")
    if basis == "so":
        keep = inside_boundary(boundary, knots[:, 0], knots[:, 1])
        if not np.all(keep):
            warnings.warn("dropping soap knots not inside boundary", stacklevel=2)
            knots = knots[keep]
    ks = np.asarray(k, dtype=int).ravel()
    if len(ks) == 1:
        ks = np.repeat(ks, len(boundary))
    if len(ks) != len(boundary):
        raise ValueError("k and bnd lengths are inconsistent")
    G, d, nb, x0, y0, h, ng = boundary_grid(boundary, nmax)
    lu = splu(pde_matrix(G, h))
    ki = np.zeros(len(knots), dtype=int)
    for index, (kx, ky) in enumerate(knots):
        ix, iy = int(np.rint((kx - x0) / h)), int(np.rint((ky - y0) / h))
        if 0 < ix < G.shape[0] and 0 < iy < G.shape[1]:
            ki[index] = G[ix, iy]
            if ki[index] <= 0:
                raise ValueError(f"knot {index + 1} is on or outside boundary")
    film, wiggly = basis != "sw", basis != "sf"
    grid_blocks, blocks, ranks = [], [], []
    fixed_rhs = np.zeros(ng)
    fixed = False
    ns = 0
    start = 0
    for index, loop in enumerate(boundary):
        stop = start + nb[index]
        if loop["f"] is not None:
            fixed = True
            distances, unique = np.unique(loop["d"], return_index=True)
            fixed_rhs[start:stop] = np.interp(
                np.clip(d[start:stop], distances[0], distances[-1]),
                distances,
                loop["f"][unique],
            )
        else:
            distances = np.r_[d[start:stop], 0.0]
            spec = boundary_spec or {"bs": "cc"}
            kval = int(ks[index])
            if spec.get("bs", "cc") == "cc":
                if boundary_spec is not None and kval < 0:
                    raise ValueError("cc bndSpec requires a non-negative boundary k")
                if kval < 0:
                    kval = 10
                if kval < 4:
                    warnings.warn(
                        "basis dimension, k, increased to minimum possible",
                        stacklevel=2,
                    )
                    kval = 4
                # setup.soap computes but does not pass bndSpec's spaced knots.
                knot_vector = place_knots_through_values(distances, kval)
                BD, _, D = cyclic_cubic_bd(knot_vector)
                B = cyclic_cubic_predict_matrix(distances, knot_vector, BD)[:-1]
                S = D.T @ BD
                S = (S + S.T) / 2
            else:
                orders = np.asarray(spec.get("m", (2, 2)), dtype=int).ravel()
                if len(orders) == 1:
                    orders = np.repeat(orders, 2)
                if kval < 0:
                    kval = max(10, int(orders[0]))
                knot_vector = cyclic_pspline_knots(distances, kval, orders[0])
                B = cyclic_pspline_design(distances, knot_vector, orders[0])[:-1]
                S = cyclic_pspline_difference_penalty(B.shape[1], orders[1])
            if film:
                rhs = np.zeros((ng, B.shape[1]))
                rhs[start:stop] = B
                grid_blocks.append(lu.solve(rhs))
                blocks.append(S)
                ranks.append(B.shape[1] - 1)
                ns += 1
        start = stop
    grid_offset = lu.solve(fixed_rhs) if fixed and film else None
    if wiggly:
        rhs = np.zeros((ng, len(ki)))
        # crunch.knots returns C zero-based indices; soap.basis uses those
        # directly in R's one-based matrix indexing. Preserve that shift.
        for i, index in enumerate(ki):
            if index > 0:
                rhs[index - 1, i] = 1
        green = lu.solve(rhs)
        green /= np.max(green, axis=0)
        blocks.append(green.T @ green * h * h)
        ranks.append(len(ki))
        grid_blocks.append(lu.solve(green))
    grid_basis = np.column_stack(grid_blocks) if grid_blocks else np.empty((ng, 0))
    X = gridder(x, y, grid_basis, G, x0, y0, h)
    inside = inside_boundary(boundary, x, y)
    X[~inside] = np.nan
    if np.isnan(X).any():
        raise ValueError("data outside soap boundary")
    if basis == "sf" and X.shape[1] == 0:
        raise ValueError("no free coefs in sf smooth")
    with np.errstate(divide="ignore", invalid="ignore"):
        irng = 1.0 / np.ptp(X, axis=0)
    if not np.all(np.isfinite(irng)):
        raise ValueError("soap basis ill-conditioned - changing 'xt$nmax' may help")
    X = X * irng
    penalties = []
    start = 0
    for block in blocks:
        stop = start + len(block)
        S = np.zeros((X.shape[1], X.shape[1]))
        S[start:stop, start:stop] = (
            irng[start:stop, None] * block * irng[None, start:stop]
        )
        penalties.append(S)
        start = stop
    offset = None if grid_offset is None else gridder(x, y, grid_offset, G, x0, y0, h)
    if offset is not None:
        # soap.basis applies this sentinel threshold even to signed offsets.
        offset[offset > np.max(grid_offset) * 2] = np.nan
        offset[~inside] = np.nan
    return SoapBasisSetup(
        X,
        penalties,
        ranks,
        ns,
        film and not fixed,
        offset,
        irng,
        boundary,
        G,
        x0,
        y0,
        h,
        grid_basis,
        grid_offset,
        basis,
    )


def predict_soap_basis(x, y, setup):
    """Predict a fitted soap basis, with NA rows outside the original boundary."""
    X = gridder(x, y, setup.grid_basis, setup.G, setup.x0, setup.y0, setup.h)
    inside = inside_boundary(setup.boundary, x, y)
    X[~inside] = np.nan
    X *= setup.irng
    offset = None
    if setup.grid_offset is not None:
        offset = gridder(x, y, setup.grid_offset, setup.G, setup.x0, setup.y0, setup.h)
        offset[offset > np.max(setup.grid_offset) * 2] = np.nan
        offset[~inside] = np.nan
    return X, offset

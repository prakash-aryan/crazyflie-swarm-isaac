"""Face-level geometry extraction from USD scenes.

Mesh bounding boxes are too coarse indoors (the box of an L-shaped desk covers the aisle next to it), so every
mesh face is reduced to its own axis-aligned box in world metres (Z-up).
"""
import numpy as np


def face_boxes(stage, z_band=None, mpu=None, root=None):
    """Per-face axis-aligned boxes (lo, hi), each of shape (N, 3), in metres.

    z_band: optional (zmin, zmax) [m]; faces entirely outside the band are dropped.
    """
    from pxr import Usd, UsdGeom

    mpu = mpu or UsdGeom.GetStageMetersPerUnit(stage) or 1.0
    root = root or stage.GetDefaultPrim() or stage.GetPseudoRoot()
    t = Usd.TimeCode.Default()
    xf_cache = UsdGeom.XformCache(t)
    los, his = [], []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        if UsdGeom.Imageable(prim).ComputeVisibility(t) == UsdGeom.Tokens.invisible:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get(t)
        counts = mesh.GetFaceVertexCountsAttr().Get(t)
        idx = mesh.GetFaceVertexIndicesAttr().Get(t)
        if not pts or not counts or not idx:
            continue
        P = np.asarray(pts, dtype=np.float64)
        M = np.asarray(xf_cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        W = (P @ M[:3, :3] + M[3, :3]) * mpu  # USD uses row vectors
        if z_band is not None and (W[:, 2].max() < z_band[0] or W[:, 2].min() > z_band[1]):
            continue
        counts = np.asarray(counts, dtype=np.int64)
        idx = np.asarray(idx, dtype=np.int64)
        if len(counts) == 0 or counts.sum() != len(idx) or idx.max() >= len(W):
            continue
        V = W[idx]
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        lo = np.minimum.reduceat(V, starts, axis=0)
        hi = np.maximum.reduceat(V, starts, axis=0)
        if z_band is not None:
            keep = (hi[:, 2] >= z_band[0]) & (lo[:, 2] <= z_band[1])
            lo, hi = lo[keep], hi[keep]
        los.append(lo)
        his.append(hi)
    if not los:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(los), np.concatenate(his)


def coverage_mask(stage, prims, grid_x, grid_y, mpu=None):
    """Boolean mask over grid cell centres (grid_x: (nx,), grid_y: (ny,)) of the cells covered by the faces of
    the given mesh prims, projected onto the XY plane."""
    cover = np.zeros((len(grid_x), len(grid_y)), dtype=bool)
    res_x = grid_x[1] - grid_x[0] if len(grid_x) > 1 else 1.0
    res_y = grid_y[1] - grid_y[0] if len(grid_y) > 1 else 1.0
    for prim in prims:
        lo, hi = face_boxes(stage, mpu=mpu, root=prim)
        for a, b in zip(lo, hi):
            i0 = int(np.searchsorted(grid_x, a[0] - 0.5 * res_x))
            i1 = int(np.searchsorted(grid_x, b[0] + 0.5 * res_x))
            j0 = int(np.searchsorted(grid_y, a[1] - 0.5 * res_y))
            j1 = int(np.searchsorted(grid_y, b[1] + 0.5 * res_y))
            cover[i0:i1, j0:j1] = True
    return cover

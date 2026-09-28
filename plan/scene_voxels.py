"""3D occupancy and clearance map of an indoor scene, for validating drone choreography.

Voxelises every mesh face (axis-aligned face boxes) inside a window, optionally marks everything outside the floor
meshes as occupied (GRScenes interiors), closes the grid with the floor and a ceiling plane, and stores the
Euclidean distance to the nearest occupied voxel.

    python scene_voxels.py --scene <office.usd> --out ../maps/office_voxels.npz
    python scene_voxels.py --scene <warehouse.usd> --out ../maps/warehouse_voxels.npz \
        --window -48 -72 -22 -47 --ceiling 9 --no-floor-mask
"""
import argparse
import os
import re
import sys

import numpy as np
from pxr import Usd, UsdGeom
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import usd_geometry  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scene", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--res", type=float, default=0.05)
ap.add_argument("--window", type=float, nargs=4, default=(-4.8, -19.6, 7.1, 2.4), help="x0 y0 x1 y1 [m]")
ap.add_argument("--ceiling", type=float, default=2.5)
ap.add_argument("--no-floor-mask", action="store_true", help="do not restrict the map to the floor meshes")
args = ap.parse_args()

stage = Usd.Stage.Open(args.scene)
mpu = UsdGeom.GetStageMetersPerUnit(stage)
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
floors = []
for prim in ([] if args.no_floor_mask else Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies())):
    if not prim.IsA(UsdGeom.Mesh):
        continue
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    if r.IsEmpty():
        continue
    lo, hi = np.array(r.GetMin()) * mpu, np.array(r.GetMax()) * mpu
    if max(hi[0] - lo[0], hi[1] - lo[1]) > 80:
        continue
    if re.search(r"ground|floor", str(prim.GetPath()).lower()) and hi[2] - lo[2] < 0.5 \
            and min(hi[0] - lo[0], hi[1] - lo[1]) >= 0.5 and abs(hi[2]) < 0.15:
        floors.append(prim)

x0, y0, x1, y1 = args.window
res = args.res
nx, ny, nz = int(round((x1 - x0) / res)), int(round((y1 - y0) / res)), int(round(args.ceiling / res))
occ = np.zeros((nx, ny, nz), dtype=bool)
lo, hi = usd_geometry.face_boxes(stage, z_band=(0.02, args.ceiling), mpu=mpu)
lo_i = np.floor((lo - [x0, y0, 0.0]) / res).astype(int)
hi_i = np.floor((hi - [x0, y0, 0.0]) / res).astype(int)
lo_i = np.clip(lo_i, 0, [nx - 1, ny - 1, nz - 1])
hi_i = np.clip(hi_i, 0, [nx - 1, ny - 1, nz - 1])
keep = (hi[:, 2] > 0.02) & (hi[:, 0] >= x0) & (lo[:, 0] <= x1) & (hi[:, 1] >= y0) & (lo[:, 1] <= y1)
for a, b in zip(lo_i[keep], hi_i[keep]):
    occ[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1] = True
cx = x0 + (np.arange(nx) + 0.5) * res
cy = y0 + (np.arange(ny) + 0.5) * res
if not args.no_floor_mask:
    inside = usd_geometry.coverage_mask(stage, floors, cx, cy, mpu)
    occ |= ~inside[:, :, None]
occ[:, :, 0] = True  # the floor
occ[:, :, -1] = True  # the ceiling
clearance = distance_transform_edt(~occ) * res
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez_compressed(args.out, origin=np.array([x0, y0, 0.0]), res=res, clearance=clearance.astype(np.float16))
print(f"grid {nx}x{ny}x{nz}, occupied {occ.mean():.1%}, saved {args.out}")

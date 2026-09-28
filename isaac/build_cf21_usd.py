"""Build Crazyflie 2.1 USD assets from the MuJoCo Menagerie Crazyflie 2 meshes.

Outputs (in --out):
  cf21_visual.usd  visual model: body parts with Crazyflie 2.1 materials (matte black PCB and propellers, clear
                   motor mounts, chrome motors, silver LiPo, gold header pins), the four propellers as separate
                   rotor prims (rotor_1..rotor_4, pivot on the motor axis, rotation about +Z) and the four arm LEDs
                   (led_1..led_4: front right red, rear blue, front left green).
  cf21_body.usda   physics body: rigid body with the firmware's nominal mass and the Crazyswarm2 inertia, and a box
                   collider whose bottom matches the motor mount feet.

Motor numbering follows the Crazyflie body frame (x forward, y left): M1 front right, M2 back right, M3 back
left, M4 front left.

    python build_cf21_usd.py --meshes ../assets/crazyflie_2 --out ../assets/cf21
"""
import argparse
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

# part name, Menagerie mesh index, material
PARTS = [
    ("frame", 5, "pcb_black"),
    ("battery_holder", 1, "pcb_black"),
    ("header_pins", 2, "gold"),
    ("battery", 3, "lipo_silver"),
    ("motors", 4, "motor_chrome"),
    ("motor_mounts", 6, "mount_clear"),
]
PROPELLERS = 0
MOTOR_SIGNS = {1: (1, -1), 2: (-1, -1), 3: (-1, 1), 4: (1, 1)}

# name: (base colour, roughness, metallic, opacity, emissive colour)
MATERIALS = {
    "pcb_black": ((0.012, 0.012, 0.014), 0.38, 0.0, 1.0, None),
    "prop_black": ((0.012, 0.012, 0.013), 0.38, 0.0, 1.0, None),
    "mount_clear": ((0.9, 0.93, 0.95), 0.18, 0.0, 1.0, None),
    "motor_chrome": ((0.78, 0.79, 0.8), 0.22, 1.0, 1.0, None),
    "lipo_silver": ((0.62, 0.64, 0.66), 0.3, 0.85, 1.0, None),
    "gold": ((0.95, 0.78, 0.42), 0.25, 1.0, 1.0, None),
    "led_blue": ((0.1, 0.25, 1.0), 0.3, 0.0, 1.0, (0.15, 0.35, 1.0)),
    "led_red": ((1.0, 0.1, 0.05), 0.3, 0.0, 1.0, (1.0, 0.08, 0.04)),
    "led_green": ((0.1, 1.0, 0.2), 0.3, 0.0, 1.0, (0.1, 1.0, 0.25)),
}
GLASS = {"mount_clear"}  # rendered with the OmniGlass MDL (clear plastic) instead of UsdPreviewSurface
LED_MATERIAL = {1: "led_red", 2: "led_blue", 3: "led_blue", 4: "led_green"}

# Mass: the firmware's nominal Crazyflie 2.1 mass (CF_MASS, used by the controllers' thrust feed-forward); the
# Crazyswarm2 simulator's 0.034 kg includes motion-capture markers, which the rendered drones do not carry.
# Inertia: Crazyflie 2 system identification (Foerster 2015), as used by Crazyswarm2 (crazyflie_sim/backend/np.py).
MASS = 0.029
INERTIA = (16.571710e-6, 16.655602e-6, 29.261652e-6)


def read_obj(path):
    v, vn, faces = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith("v "):
                v.append([float(c) for c in line.split()[1:4]])
            elif line.startswith("vn "):
                vn.append([float(c) for c in line.split()[1:4]])
            elif line.startswith("f "):
                faces.append([int(c.split("/")[0]) - 1 for c in line.split()[1:]])
    return np.array(v), np.array(vn), faces


def define_mesh(stage, path, points, normals, faces, material, offset=(0.0, 0.0, 0.0)):
    used = sorted({i for f in faces for i in f})
    remap = {old: new for new, old in enumerate(used)}
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy((points[used] - np.array(offset)).astype(np.float32)))
    mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals[used].astype(np.float32)))
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateFaceVertexCountsAttr([len(f) for f in faces])
    mesh.CreateFaceVertexIndicesAttr([remap[i] for f in faces for i in f])
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


def define_glass(stage, path, colour, frosting):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniGlass.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniGlass", "mdl")
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*colour))
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(frosting)
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.49)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    material.CreateVolumeOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return material


def define_material(stage, path, base, roughness, metallic, opacity, emissive):
    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*base))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
    if emissive is not None:
        shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def build_visual(mesh_dir, out_path):
    if os.path.exists(out_path):
        os.remove(out_path)
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/cf21")
    stage.SetDefaultPrim(root.GetPrim())
    UsdGeom.Xform.Define(stage, "/cf21/Looks")
    mats = {name: define_glass(stage, f"/cf21/Looks/{name}", spec[0], spec[1]) if name in GLASS
            else define_material(stage, f"/cf21/Looks/{name}", *spec) for name, spec in MATERIALS.items()}

    UsdGeom.Xform.Define(stage, "/cf21/body")
    for name, index, material in PARTS:
        v, vn, faces = read_obj(os.path.join(mesh_dir, f"cf2_{index}.obj"))
        define_mesh(stage, f"/cf21/body/{name}", v, vn, faces, mats[material])

    # motor axes: centroid of each motor can, by quadrant
    mv, _, _ = read_obj(os.path.join(mesh_dir, "cf2_4.obj"))
    axes = {}
    for m, (sx, sy) in MOTOR_SIGNS.items():
        sel = (np.sign(mv[:, 0]) == sx) & (np.sign(mv[:, 1]) == sy)
        axes[m] = mv[sel, :2].mean(axis=0)

    pv, pvn, pfaces = read_obj(os.path.join(mesh_dir, f"cf2_{PROPELLERS}.obj"))
    centre = pv[[f[0] for f in pfaces], :2]
    owner = np.argmin([[np.hypot(*(c - axes[m])) for m in (1, 2, 3, 4)] for c in centre], axis=1) + 1
    for m in (1, 2, 3, 4):
        rotor = UsdGeom.Xform.Define(stage, f"/cf21/rotor_{m}")
        rotor.AddTranslateOp().Set(Gf.Vec3d(axes[m][0], axes[m][1], 0.0))
        rotor.AddRotateZOp().Set(0.0)
        faces = [f for f, o in zip(pfaces, owner) if o == m]
        define_mesh(stage, f"/cf21/rotor_{m}/propeller", pv, pvn, faces, mats["prop_black"],
                    offset=(axes[m][0], axes[m][1], 0.0))

    # LEDs on top of the PCB arms, 8.5 mm inboard of the motor axes
    fv, _, _ = read_obj(os.path.join(mesh_dir, "cf2_5.obj"))
    radial = np.hypot(fv[:, 0], fv[:, 1])
    arm_top = float(fv[radial > radial.max() - 0.012, 2].max())
    for m in (1, 2, 3, 4):
        d = axes[m] / np.linalg.norm(axes[m])
        p = d * (np.linalg.norm(axes[m]) - 0.0085)
        led = UsdGeom.Sphere.Define(stage, f"/cf21/led_{m}")
        led.CreateRadiusAttr(0.0011)
        led.AddTranslateOp().Set(Gf.Vec3d(p[0], p[1], arm_top + 0.0004))
        UsdShade.MaterialBindingAPI.Apply(led.GetPrim()).Bind(mats[LED_MATERIAL[m]])
    stage.GetRootLayer().Save()
    return axes, arm_top


def build_body(mesh_dir, out_path):
    mount, _, _ = read_obj(os.path.join(mesh_dir, "cf2_6.obj"))
    foot = float(mount[:, 2].min())
    if os.path.exists(out_path):
        os.remove(out_path)
    stage = Usd.Stage.CreateNew(out_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    body = UsdGeom.Xform.Define(stage, "/cf21_body")
    stage.SetDefaultPrim(body.GetPrim())
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
    mass.CreateMassAttr(MASS)
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*INERTIA))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(0.0, 0.0, 0.0))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    box = UsdGeom.Cube.Define(stage, "/cf21_body/collider")
    box.CreateSizeAttr(1.0)
    height = 0.018
    box.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, foot + height / 2))
    box.AddScaleOp().Set(Gf.Vec3f(0.09, 0.09, height))
    box.CreatePurposeAttr(UsdGeom.Tokens.guide)
    UsdPhysics.CollisionAPI.Apply(box.GetPrim())
    stage.GetRootLayer().Save()
    return foot


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--meshes", default=os.path.join(here, "..", "assets", "crazyflie_2"),
                    help="directory with the Menagerie cf2_0.obj ... cf2_6.obj meshes")
    ap.add_argument("--out", default=os.path.join(here, "..", "assets", "cf21"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    axes, arm_top = build_visual(args.meshes, os.path.join(args.out, "cf21_visual.usd"))
    foot = build_body(args.meshes, os.path.join(args.out, "cf21_body.usda"))
    print("motor axes", {m: np.round(a, 4).tolist() for m, a in axes.items()}, "arm top", round(arm_top, 4),
          "foot", round(foot, 4))

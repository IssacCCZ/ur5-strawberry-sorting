"""Build USD scene geometry from the project's SDF and local meshes."""
import functools

import numpy as np
import trimesh
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from .assets import SOURCE
from .kinematics import transform

COLORS = {"Gazebo/Grey": (0.48, 0.51, 0.49), "Gazebo/DarkGrey": (0.12, 0.15, 0.16),
          "Gazebo/Red": (0.48, 0.035, 0.025), "Gazebo/Blue": (0.035, 0.12, 0.34),
          "Gazebo/Green": (0.10, 0.40, 0.08), "vrc/mud": (0.25, 0.17, 0.10)}
COLORS.update({"galvanized": (0.46, 0.49, 0.48), "concrete": (0.29, 0.29, 0.26),
               "coco": (0.075, 0.035, 0.016), "irrigation": (0.012, 0.018, 0.016),
               "leaf": (0.055, 0.22, 0.025), "leaf_light": (0.085, 0.28, 0.035),
               "stem": (0.15, 0.26, 0.035), "frosted": (0.68, 0.76, 0.71),
               "planter": (0.33, 0.26, 0.18), "rubber": (0.024, 0.028, 0.025)})
TEXTURES = {
    "strawberry": "strawberry/materials/textures/strawberry.jpg",
    "strawberry_yellow": "strawberry_maturity/materials/textures/strawberry_yellow.jpg",
    "strawberry_green": "strawberry_maturity/materials/textures/strawberry_green.jpg",
    "vrc/mud": "mud_box/materials/textures/mud_soft_leaves.png",
}


def set_transform(prim, matrix):
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    # Gf matrices use row vectors; numpy kinematics use column vectors.
    xform.AddTransformOp().Set(Gf.Matrix4d(np.asarray(matrix).T.tolist()))


def world_transform(prim):
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def xform(stage, path, pose):
    prim = UsdGeom.Xform.Define(stage, path).GetPrim()
    set_transform(prim, transform(pose[:3], pose[3:]))
    return prim


def material(stage, name):
    path = "/World/Looks/" + name.replace("/", "_")
    existing = stage.GetPrimAtPath(path)
    if existing:
        return UsdShade.Material(existing)
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + "/Surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(COLORS.get(name, (0.7, 0.7, 0.7)))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.75)
    if name in {"Gazebo/Grey", "Gazebo/DarkGrey", "galvanized"}:
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.72)
        shader.GetInput("roughness").Set(0.36)
    if name in {"Gazebo/Red", "Gazebo/Blue", "leaf", "leaf_light"}:
        shader.GetInput("roughness").Set(0.48)
    shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    if name in TEXTURES:
        texture_path = (SOURCE / "models" / TEXTURES[name]).resolve()
        if not texture_path.is_file():
            raise FileNotFoundError(texture_path)
        reader = UsdShade.Shader.Define(stage, path + "/UV")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        texture = UsdShade.Shader.Define(stage, path + "/Texture")
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(str(texture_path))
        texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.GetInput("diffuseColor").ConnectToSource(texture.ConnectableAPI(), "rgb")
    return mat


@functools.lru_cache(maxsize=32)
def load_mesh(path):
    # trimesh applies COLLADA scene-node transforms and retains UV coordinates.
    loaded = trimesh.load(path, force="scene", process=False)
    meshes = []
    for node in loaded.graph.nodes_geometry:
        matrix, geometry = loaded.graph[node]
        mesh = loaded.geometry[geometry].copy()
        mesh.apply_transform(matrix)
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"No geometry in {path}")
    return meshes


def geometry(stage, parent, record, collision=False):
    root = xform(stage, parent + "/" + record["name"], record["pose"])
    path = str(root.GetPath())
    if record["type"] == "box":
        mesh = trimesh.creation.box(extents=record["size"])
        # Planar UVs are only used for the soil box; other boxes are solid.
        mesh.visual = trimesh.visual.TextureVisuals(uv=mesh.vertices[:, :2] * 2.)
        meshes = [mesh]
    elif record["type"] == "cylinder":
        meshes = [trimesh.creation.cylinder(radius=record["radius"], height=record["length"], sections=48)]
    else:
        meshes = load_mesh(record["path"])
    for index, mesh in enumerate(meshes):
        usd_mesh = UsdGeom.Mesh.Define(stage, f"{path}/mesh_{index}")
        scale = np.array(record.get("scale", [1, 1, 1]))
        usd_mesh.CreatePointsAttr((mesh.vertices * scale).astype(np.float32).tolist())
        usd_mesh.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
        usd_mesh.CreateFaceVertexIndicesAttr(mesh.faces.flatten().tolist())
        usd_mesh.CreateSubdivisionSchemeAttr("none")
        usd_mesh.CreateDoubleSidedAttr(True)
        prim = usd_mesh.GetPrim()
        if collision:
            UsdPhysics.CollisionAPI.Apply(prim)
            # Static shelf triangle meshes preserve the open gaps.
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr("none")
            usd_mesh.CreateVisibilityAttr("invisible")
        else:
            uv = getattr(mesh.visual, "uv", None)
            if uv is not None:
                UsdGeom.PrimvarsAPI(prim).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                      UsdGeom.Tokens.vertex).Set(np.asarray(uv).tolist())
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material(stage, record["material"]))


def build_scene(stage, records):
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    fruit_paths = {}
    for record in records:
        path = "/World/" + record["name"]
        model = xform(stage, path, record["pose"])
        if record["name"].startswith("strawberry"):
            UsdPhysics.RigidBodyAPI.Apply(model)
            mass = UsdPhysics.MassAPI.Apply(model)
            mass.CreateMassAttr(0.01)
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-5))
            mass.CreateCenterOfMassAttr(Gf.Vec3f(0))
            fruit_paths[record["name"]] = path
        for link in record["links"]:
            link_path = path + "/" + link["name"]
            xform(stage, link_path, link["pose"])
            for item in link["visuals"]:
                geometry(stage, link_path, item)
            for item in link["collisions"]:
                geometry(stage, link_path, item, collision=True)
    from .environment import build_greenhouse
    build_greenhouse(stage, records)
    return fruit_paths

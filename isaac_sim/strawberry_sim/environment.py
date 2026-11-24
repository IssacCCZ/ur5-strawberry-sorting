"""Procedural greenhouse dressing in metres, outside the arm's working envelope.

Decorative vegetation, pipework and building frames are visual geometry only.
The original task fixtures and ground retain their collision geometry.
"""
import math
from itertools import pairwise

import numpy as np
from pxr import Gf, UsdGeom, UsdLux, UsdShade

from .scene import material


def bind(prim, stage, name):
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(material(stage, name))


def box(stage, path, position, size, surface):
    shape = UsdGeom.Cube.Define(stage, path)
    shape.CreateSizeAttr(1.)
    shape.AddTranslateOp().Set(Gf.Vec3d(*position))
    shape.AddScaleOp().Set(Gf.Vec3f(*size))
    bind(shape.GetPrim(), stage, surface)
    return shape


def pipe(stage, path, start, end, radius, surface="galvanized"):
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    direction = end - start
    shape = UsdGeom.Cylinder.Define(stage, path)
    shape.CreateRadiusAttr(radius)
    shape.CreateHeightAttr(float(np.linalg.norm(direction)))
    shape.CreateAxisAttr("Z")
    shape.AddTranslateOp().Set(Gf.Vec3d(*((start + end) / 2)))
    rotation = Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*(direction / np.linalg.norm(direction))))
    shape.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(rotation.GetQuat())
    bind(shape.GetPrim(), stage, surface)
    return shape


def leaf(stage, path, origin, angle, length, width, lift, surface):
    """Curved, finely serrated leaf blade with a pale central vein."""
    forward = np.array([math.cos(angle), math.sin(angle), 0.])
    side = np.array([-math.sin(angle), math.cos(angle), 0.])
    origin = np.asarray(origin)
    points = []
    segments = 20
    for i in range(segments + 1):
        u = i / segments
        half_width = width / 2 * math.sin(math.pi * u) ** 0.8 * (0.90 if i % 2 else 1.)
        center = origin + forward * length * u + [0, 0, lift*u + 0.012*math.sin(math.pi*u)]
        for v in (-1, 0, 1):
            points.append(center + side * half_width * v + [0, 0, -0.004*abs(v)*math.sin(math.pi*u)])
    faces = []
    for i in range(segments):
        for j in range(2):
            a = i*3+j
            faces.extend([(a, a+3, a+1), (a+1, a+3, a+4)])
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(np.array(points, dtype=np.float32).tolist())
    mesh.CreateFaceVertexCountsAttr([3]*len(faces))
    mesh.CreateFaceVertexIndicesAttr(np.array(faces).ravel().tolist())
    mesh.CreateSubdivisionSchemeAttr("none")
    mesh.CreateDoubleSidedAttr(True)
    # Vertex normals retain the blade curvature instead of faceted highlights.
    normals = np.zeros_like(points)
    for a, b, c in faces:
        normal = np.cross(points[b]-points[a], points[c]-points[a])
        for index in (a, b, c):
            normals[index] += normal
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    mesh.CreateNormalsAttr(normals.astype(np.float32).tolist())
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    bind(mesh.GetPrim(), stage, surface)
    # A narrow raised central vein, following the leaf's curve.
    for i in range(4):
        a, b = i/4, (i+1)/4
        def center(u):
            return origin + forward*length*u + [0, 0, lift*u+0.012*math.sin(math.pi*u)+0.0004]
        pipe(stage, path+f"_vein_{i}", center(a), center(b), 0.00045, "stem")


def plant(stage, path, origin, phase=0., scale=1.):
    UsdGeom.Xform.Define(stage, path)
    origin = np.asarray(origin)
    for branch in range(4):
        angle = phase + branch*math.pi/2
        junction = origin + scale*np.array([0.036*math.cos(angle), 0.036*math.sin(angle), 0.04+(branch%2)*0.012])
        pipe(stage, f"{path}/petiole_{branch}", origin, junction, 0.0013*scale, "stem")
        for leaflet, spread in enumerate((-0.75, 0, 0.75)):
            leaf(stage, f"{path}/leaf_{branch}_{leaflet}", junction, angle+spread,
                 scale*(0.070 if leaflet==1 else 0.053), scale*0.044,
                 scale*(-0.020 if branch%2 else 0.008), "leaf" if branch%2 else "leaf_light")


def grow_bench(stage, path, x, y, length, plant_count):
    """Background strawberry growing trough; foliage only, no detection targets."""
    UsdGeom.Xform.Define(stage, path)
    box(stage, path+"/trough", [x, y, 0.92], [0.32, length, 0.15], "planter")
    box(stage, path+"/substrate", [x, y, 0.999], [0.29, length-0.035, 0.012], "coco")
    for offset in (-length/2+0.12, length/2-0.12):
        for dx in (-0.115, 0.115):
            pipe(stage, path+f"/leg_{offset:.2f}_{dx:.2f}".replace("-", "m").replace(".", "p"),
                 [x+dx, y+offset, 0.015], [x+dx, y+offset, 0.86], 0.015)
        pipe(stage, path+f"/cross_{offset:.2f}".replace("-", "m").replace(".", "p"),
             [x-0.115, y+offset, 0.32], [x+0.115, y+offset, 0.32], 0.012)
    pipe(stage, path+"/drip", [x+0.10, y-length/2, 1.009], [x+0.10, y+length/2, 1.009], 0.005, "irrigation")
    for i, py in enumerate(np.linspace(y-length/2+0.14, y+length/2-0.14, plant_count)):
        plant(stage, path+f"/plant_{i}", [x, py, 1.008], phase=i*1.37, scale=1.25)


def build_greenhouse(stage, records):
    root = "/World/Environment"
    UsdGeom.Xform.Define(stage, root)
    # Thin working slabs are flush with soil, rather than a floating platform.
    for i in range(3):
        for j in range(3):
            box(stage, f"{root}/slab_{i}_{j}", [-0.45+i*0.50, -0.40+j*0.50, -0.019],
                [0.495, 0.495, 0.040], "concrete")
    # Primary planting insert, stems and foliage sit behind/above the three fruit targets.
    box(stage, root+"/main_substrate", [0.60, 0.045, 0.925], [0.102, 0.25, 0.020], "coco")
    for i, py in enumerate((-0.045, 0.045, 0.135)):
        plant(stage, root+f"/main_plant_{i}", [0.61, py, 0.939], phase=i*1.1, scale=0.60)
    for record in records:
        if record["name"].startswith("strawberry"):
            tip = np.array(record["pose"][:3]) + [0, 0, 0.020]
            crown = np.array([0.603, tip[1], 0.951])
            middle = (tip+crown)/2 + [0, 0, 0.023]
            for i in range(6):
                def curve(t, tip=tip, middle=middle, crown=crown):
                    return (1-t)**2*tip + 2*(1-t)*t*middle + t*t*crown
                pipe(stage, root+f"/{record['name']}_stem_{i}", curve(i/6), curve((i+1)/6), 0.0011, "stem")
    pipe(stage, root+"/main_drip", [0.65, -0.09, 0.947], [0.65, 0.185, 0.947], 0.0035, "irrigation")
    pipe(stage, root+"/supply", [0.68, 0.20, 0.02], [0.68, 0.20, 0.947], 0.004, "irrigation")
    pipe(stage, root+"/drip_elbow", [0.68, 0.20, 0.947], [0.65, 0.185, 0.947], 0.0035, "irrigation")
    # Fasteners / mounting flange give the robot pedestal a mechanical scale.
    pipe(stage, root+"/base_flange", [0, 0, 0.336], [0, 0, 0.351], 0.185)
    pipe(stage, root+"/base_neck", [0, 0, 0.351], [0, 0, 0.360], 0.095)
    for i in range(6):
        a = i*math.pi/3
        x, y = 0.151*math.cos(a), 0.151*math.sin(a)
        pipe(stage, root+f"/anchor_{i}", [x, y, 0.351], [x, y, 0.359], 0.008, "Gazebo/DarkGrey")
    # Remote rows stay outside the task's original collision envelope.
    for i, x in enumerate((1.65, 2.55)):
        grow_bench(stage, root+f"/row_{i}", x, 0., 5.8, 22)
    grow_bench(stage, root+"/rear_row", -1.9, -1.5, 2.6, 10)
    # A six-metre greenhouse, with diffuse wall panels and round steel hoops.
    for i, y in enumerate((-4., -2., 0., 2., 4.)):
        points = [[3.*math.cos(a), y, 1.6+1.8*math.sin(a)] for a in np.linspace(0, math.pi, 17)]
        points = [[3., y, 0.], *points, [-3., y, 0.]]
        for j, (a, b) in enumerate(pairwise(points)):
            pipe(stage, root+f"/hoop_{i}_{j}", a, b, 0.025)
    for i, x in enumerate((-3., 3.)):
        box(stage, root+f"/wall_{i}", [x, 0, 1.1], [0.012, 8., 2.2], "frosted")
        for k, height in enumerate((0.12, 1.55, 2.12)):
            pipe(stage, root+f"/rail_{i}_{k}", [x, -4, height], [x, 4, height], 0.018)
    pipe(stage, root+"/ridge", [0, -4, 3.4], [0, 4, 3.4], 0.021)
    # Soft skylight plus a broad sun produces contact shadows and readable materials.
    dome = UsdLux.DomeLight.Define(stage, "/World/Light")
    dome.CreateIntensityAttr(420.)
    dome.CreateColorAttr(Gf.Vec3f(0.78, 0.86, 1.0))
    sun = UsdLux.DistantLight.Define(stage, root+"/Sun")
    sun.CreateIntensityAttr(1800.)
    sun.CreateAngleAttr(8.)
    sun.CreateColorAttr(Gf.Vec3f(1., 0.91, 0.77))
    sun.AddRotateXYZOp().Set(Gf.Vec3f(-28., -35., -25.))
    fill = UsdLux.RectLight.Define(stage, root+"/RoofFill")
    fill.CreateIntensityAttr(180.)
    fill.CreateWidthAttr(5.)
    fill.CreateHeightAttr(7.)
    fill.AddTranslateOp().Set(Gf.Vec3d(0., 0., 3.3))

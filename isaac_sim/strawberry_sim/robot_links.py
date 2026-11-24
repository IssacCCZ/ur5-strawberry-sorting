"""Resolve robot frames without confusing them with importer geometry containers."""
from pxr import Usd, UsdGeom, UsdPhysics


def find_robot_link(robot_root, name, *, require_rigid_body=False):
    if not robot_root:
        raise RuntimeError("Cannot find robot links: imported robot root is invalid")
    candidates = [prim for prim in Usd.PrimRange(robot_root)
                  if prim.GetName() == name and prim.IsA(UsdGeom.Xform)]
    bodies = [prim for prim in candidates if prim.HasAPI(UsdPhysics.RigidBodyAPI)]
    # Importers also create colliders/<link> and visuals/<link> auxiliary prims.
    # These Xforms are geometry containers, not the simulated link frames.
    found = bodies if bodies or require_rigid_body else candidates
    if len(found) != 1:
        details = ", ".join(f"{prim.GetPath()} (rigid_body={prim.HasAPI(UsdPhysics.RigidBodyAPI)})"
                            for prim in candidates) or "none"
        raise RuntimeError(
            f"Expected one robot link {name} under {robot_root.GetPath()}, got {len(found)}; "
            f"require_rigid_body={require_rigid_body}; candidates: {details}")
    return found[0]

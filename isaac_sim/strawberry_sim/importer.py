"""Robot import across the Isaac Sim 5.1 and 6.1 importer APIs."""
import omni.kit.commands
from isaacsim.core.utils.extensions import enable_extension
from pxr import Usd, UsdGeom, UsdPhysics


def import_robot(stage, urdf):
    enable_extension("isaacsim.asset.importer.urdf")
    import isaacsim.asset.importer.urdf as importer

    if hasattr(importer, "URDFImporterConfig"):
        config = importer.URDFImporterConfig(
            urdf_path=str(urdf), usd_path=str(urdf.parent / "robot_usd"),
            merge_fixed_joints=False, merge_mesh=False, collision_from_visuals=False,
            collision_type="Convex Hull", allow_self_collision=False, fix_base=True,
            joint_drive_type="force", joint_target_type="position",
            override_joint_stiffness=20000., override_joint_damping=1000.)
        usd_path = importer.URDFImporter(config).import_urdf()
        if not usd_path:
            raise RuntimeError(f"URDF import failed: {urdf}")
        root = UsdGeom.Xform.Define(stage, "/World/ur5").GetPrim()
        root.GetReferences().AddReference(str(usd_path))
        # 6.1 exports several physics backends without an authored default selection.
        root.GetVariantSets().GetVariantSet("Physics").SetVariantSelection("physx")
        return root
    else:
        from isaacsim.asset.importer.urdf import _urdf

        ok, config = omni.kit.commands.execute("URDFCreateImportConfig")
        if not ok:
            raise RuntimeError("Isaac Sim URDF importer did not initialize")
        config.merge_fixed_joints = False
        config.fix_base = True
        config.make_default_prim = False
        config.import_inertia_tensor = True
        config.self_collision = False
        config.collision_from_visuals = False
        config.parse_mimic = False
        config.default_drive_strength = 20000.0
        config.default_position_drive_damping = 1000.0
        config.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
        ok, robot_path = omni.kit.commands.execute(
            "URDFParseAndImportFile", urdf_path=str(urdf), import_config=config,
            get_articulation_root=False)
        if not ok or not robot_path:
            raise RuntimeError(f"URDF import failed: {urdf}")
        root = stage.GetPrimAtPath(robot_path)
        if not root:
            raise RuntimeError(f"URDF importer returned a missing robot prim: {robot_path}")
        # 5.1 defaults to acceleration drives; runtime gains are tuned for force
        # drives (as explicitly selected in 6.1). Set this before PhysX reset.
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdPhysics.RevoluteJoint):
                drive = UsdPhysics.DriveAPI(prim, "angular")
                if drive:
                    drive.CreateTypeAttr("force")
        return root

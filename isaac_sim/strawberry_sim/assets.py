"""Resolve the original ROS assets without a ROS installation or ROS master."""
from __future__ import annotations

import copy
import json
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/code"
ROBOT_PACKAGE = SOURCE / "ur_eye_in_hand/ur5-gazebo-grasping/ur5_single_arm_tufts"
DEFAULT_OUTPUT = ROOT / "isaac_sim/generated"
XACRO_NS = "http://ros.org/wiki/xacro"


def packages(source=SOURCE):
    result = {}
    for manifest in sorted(source.rglob("package.xml")):
        name = ET.parse(manifest).getroot().findtext("name")
        if name in result:
            raise ValueError(f"Duplicate ROS package: {name}")
        result[name] = manifest.parent.resolve()
    return result


def resolve_uri(uri, package_map=None):
    if uri.startswith("package://"):
        name, relative = uri[10:].split("/", 1)
        path = (package_map or packages())[name] / relative
    elif uri.startswith("model://"):
        relative = uri[8:]
        candidates = [SOURCE / "models" / relative, ROBOT_PACKAGE / "models" / relative]
        path = next((p for p in candidates if p.exists()), candidates[0])
    else:
        path = Path(uri.removeprefix("file://"))
    if not path.exists():
        raise FileNotFoundError(f"Unresolved asset {uri}: {path}")
    return path.resolve()


def prepare_robot(output=DEFAULT_OUTPUT):
    """Expand the launch file's Xacro, never the obsolete checked-in URDF.

    Stage includes and replace $(find ...) locally. This also upgrades the
    old unprefixed <property> syntax, without modifying the ROS 1 sources.
    """
    import xacro

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    package_map = packages()
    with tempfile.TemporaryDirectory(prefix="strawberry-xacro-") as temporary:
        staged = {}

        def stage_file(path):
            path = path.resolve()
            if path in staged:
                return staged[path]
            target = Path(temporary) / f"{len(staged)}_{path.name}"
            staged[path] = target
            text = path.read_text().replace("http://wiki.ros.org/xacro", XACRO_NS)
            text = re.sub(r"\$\(find\s+([^\s)]+)\)",
                          lambda m: str(package_map[m[1]]), text)
            tree = ET.fromstring(text)
            for node in tree.iter():
                if node.tag in {"property", "macro", "insert_block", "include", "if", "unless"}:
                    node.tag = f"{{{XACRO_NS}}}{node.tag}"
                if node.tag == f"{{{XACRO_NS}}}include":
                    include = Path(node.attrib["filename"])
                    if not include.is_absolute():
                        include = path.parent / include
                    node.set("filename", str(stage_file(include)))
            ET.register_namespace("xacro", XACRO_NS)
            ET.ElementTree(tree).write(target, encoding="unicode")
            return target

        input_file = stage_file(ROBOT_PACKAGE / "urdf/ur5_single_arm.urdf.xacro")
        robot = ET.fromstring(xacro.process_file(str(input_file)).toxml())

    for child in list(robot):
        if child.tag not in {"link", "joint", "material"}:
            robot.remove(child)
    mimic = {}
    for joint in robot.findall("joint"):
        node = joint.find("mimic")
        if node is not None:
            mimic[joint.attrib["name"]] = {
                "joint": node.attrib["joint"],
                "multiplier": float(node.get("multiplier", "1")),
                "offset": float(node.get("offset", "0")),
            }
            # Explicitly drive coupled fingers. Avoid importer-version-specific
            # mimic constraints and accidental duplicate drives.
            joint.remove(node)
    for mesh in robot.iter("mesh"):
        mesh.set("filename", str(resolve_uri(mesh.attrib["filename"], package_map)))
    compact_camera_mount(robot)
    ET.indent(robot)
    destination = output / "ur5_robotiq.urdf"
    ET.ElementTree(robot).write(destination, encoding="utf-8", xml_declaration=True)
    (output / "mimic.json").write_text(json.dumps(mimic, indent=2) + "\n")
    return destination


def compact_camera_mount(robot):
    """Replace the protruding legacy mesh with a 73 mm aluminium L bracket.

    Preserve the camera joint / optical calibration. The horizontal plate ends
    at the camera's y=-70 mm attachment. The short return supports the rear
    housing and stops behind the optical plane (z=-20 mm, looking down -Z).
    """
    holder = robot.find("link[@name='camera_holder_link']")
    for child in list(holder):
        holder.remove(child)
    inertial = ET.SubElement(holder, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 -0.0405 -0.003", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="0.028")
    ET.SubElement(inertial, "inertia", ixx="0.000021", iyy="0.000006", izz="0.000025",
                  ixy="0", ixz="0", iyz="0")
    for name, xyz, size in (
        ("mount_plate", "0 -0.035 -0.0015", "0.040 0.073 0.003"),
        ("camera_return", "0 -0.070 -0.010", "0.040 0.003 0.014"),
    ):
        for tag in ("visual", "collision"):
            node = ET.SubElement(holder, tag, name=name)
            ET.SubElement(node, "origin", xyz=xyz, rpy="0 0 0")
            ET.SubElement(ET.SubElement(node, "geometry"), "box", size=size)
            if tag == "visual":
                mat = ET.SubElement(node, "material", name="mount_aluminium")
                ET.SubElement(mat, "color", rgba="0.22 0.25 0.27 1")


def numbers(text, default):
    return [float(x) for x in (text or default).split()]


def read_world():
    """Read the finite subset of SDF used by strawberry_grasp.world.

    Includes are expanded in source order; link and geometry transforms stay
    separate so their rotations compose correctly in USD.
    """
    world = ET.parse(ROBOT_PACKAGE / "worlds/strawberry_grasp.world").getroot().find("world")
    result = []
    for element in world:
        if element.tag == "include":
            model = copy.deepcopy(ET.parse(resolve_uri(element.findtext("uri") + "/model.sdf")).getroot().find("model"))
            model.set("name", element.findtext("name", model.get("name")))
            old = model.find("pose")
            if old is not None:
                model.remove(old)
            model.append(copy.deepcopy(element.find("pose")))
        elif element.tag == "model":
            model = element
        else:
            continue
        record = {"name": model.get("name"), "pose": numbers(model.findtext("pose"), "0 0 0 0 0 0"), "links": []}
        for link in model.findall("link"):
            item = {"name": link.get("name"), "pose": numbers(link.findtext("pose"), "0 0 0 0 0 0"), "visuals": [], "collisions": []}
            for tag, key in [("visual", "visuals"), ("collision", "collisions")]:
                for node in link.findall(tag):
                    geometry = node.find("geometry")
                    shape = next(iter(geometry))
                    data = {"name": node.get("name"), "pose": numbers(node.findtext("pose"), "0 0 0 0 0 0"), "type": shape.tag,
                            "material": node.findtext("material/script/name", "Gazebo/Grey")}
                    if shape.tag == "mesh":
                        data.update(path=str(resolve_uri(shape.findtext("uri"))), scale=numbers(shape.findtext("scale"), "1 1 1"))
                    elif shape.tag == "box":
                        data["size"] = numbers(shape.findtext("size"), "1 1 1")
                    elif shape.tag == "cylinder":
                        data.update(radius=float(shape.findtext("radius")), length=float(shape.findtext("length")))
                    else:
                        raise ValueError(f"Unsupported SDF geometry: {shape.tag}")
                    item[key].append(data)
            record["links"].append(item)
        result.append(record)
    return result


def prepare(output=DEFAULT_OUTPUT):
    output = Path(output)
    robot = prepare_robot(output)
    scene = read_world()
    (output / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    return robot, scene

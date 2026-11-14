"""CMU mocap (ASF/AMC) -> Unitree G1 29-DoF, written as mjlab / BeyondMimic CSV.

CSV row (no header): base xyz [m], base quat xyzw, 29 joint angles [rad] in MuJoCo order. 30 fps.
IK is GMR's GeneralMotionRetargeting; we feed it CMU bone poses under GMR's LAFAN1 body names.

Usage: .venv/bin/python retarget.py [trial ...]
"""
import os
import json
import sys
import types
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).parent
G1_XML = os.environ.get("G1_MENAGERIE", "mujoco_menagerie/unitree_g1") + "/g1.xml"  # MuJoCo Menagerie checkout
OUT_FPS = 30
ASF_TO_M = 0.0254 / 0.45  # CMU FAQ: (1/0.45) * 2.54 / 100
# CMU axes (x left, y up, z forward) -> robot axes (x forward, y left, z up)
P = np.array([[0.0, 0, 1], [1, 0, 0], [0, 1, 0]])

# trial, CMU capture fps, clip name, description from the CMU index (mocap.cs.cmu.edu)
CLIPS = [
    ("13_17", 120, "boxing_13_17", "boxing"),
    ("13_18", 120, "boxing_13_18", "boxing"),
    ("14_01", 120, "boxing_14_01", "boxing"),
    ("15_13", 120, "boxing_15_13", "boxing"),
    ("17_10", 120, "boxing_17_10", "boxing"),
    ("79_08", 60, "boxing_79_08", "boxing"),
    ("80_10", 60, "boxing_80_10", "boxing"),
    ("02_05", 120, "punch_strike_02_05", "punch/strike"),
    ("143_23", 120, "punching_143_23", "Punching"),
    ("144_20", 120, "punch_sequence_144_20", "Punch_Sequence"),
    ("144_13", 120, "left_punch_sequence_144_13", "Left_Punch_Sequence001"),
    ("135_09", 120, "oizuki_lunge_punch_135_09", "Oiduki"),
    ("76_01", 120, "backstep_then_punch_76_01", "walk backwards then attack with a punch"),
    ("141_14", 120, "punch_and_kick_141_14", "Punch and Kick"),
    ("143_24", 120, "kicking_143_24", "Kicking"),
    ("144_05", 120, "front_kicking_144_05", "Front_Kicking"),
    ("144_09", 120, "left_front_kicking_144_09", "Left_Front_Kicking"),
    ("135_04", 120, "front_kick_135_04", "Front Kick"),
    ("135_07", 120, "roundhouse_mawashigeri_135_07", "Mawashigeri"),
    ("135_11", 120, "side_kick_yokogeri_135_11", "Yokogeri"),
    ("74_03", 60, "kick_74_03", "kick"),
    ("144_26", 120, "right_blocks_144_26", "Right_Blocks"),
    ("76_04", 120, "guard_pose_76_04", "defensive guard pose"),
    ("78_29", 120, "footwork_side_to_side_78_29", "DefensiveMoveSideToSide CleanedGRS"),
    ("78_28", 120, "footwork_zigzag_78_28", "DefensiveMoveZigZag CleanedGRS"),
]

# GMR (LAFAN1) body name -> (CMU bone whose start joint gives the position, bone giving the orientation, G1 link)
BODIES = {
    "Hips": ("root", "root", "pelvis"),
    "Spine2": ("thorax", "thorax", "torso_link"),
    "LeftUpLeg": ("lfemur", "lfemur", "left_hip_yaw_link"),
    "LeftLeg": ("ltibia", "ltibia", "left_knee_link"),
    "LeftFootMod": ("lfoot", "lfoot", "left_ankle_roll_link"),
    "RightUpLeg": ("rfemur", "rfemur", "right_hip_yaw_link"),
    "RightLeg": ("rtibia", "rtibia", "right_knee_link"),
    "RightFootMod": ("rfoot", "rfoot", "right_ankle_roll_link"),
    "LeftArm": ("lhumerus", "lhumerus", "left_shoulder_yaw_link"),
    "LeftForeArm": ("lradius", "lradius", "left_elbow_link"),
    "LeftHand": ("lwrist", "lradius", "left_wrist_yaw_link"),
    "RightArm": ("rhumerus", "rhumerus", "right_shoulder_yaw_link"),
    "RightForeArm": ("rradius", "rradius", "right_elbow_link"),
    "RightHand": ("rwrist", "rradius", "right_wrist_yaw_link"),
}


def euler(axes, deg):  # ASF convention: R = Rz Ry Rx (fixed axes), for whichever axes are present
    return R.from_euler(axes, deg, degrees=True)


def parse_asf(path):
    text = Path(path).read_text()
    bones = {"root": dict(dir=np.zeros(3), len=0.0, axis=R.identity(), dof="xyz")}
    for block in text.split(":bonedata")[1].split(":hierarchy")[0].split("begin")[1:]:
        f = {l.split()[0]: l.split()[1:] for l in block.splitlines() if l.split()}
        bones[f["name"][0]] = dict(
            dir=np.array(f["direction"], float),
            len=float(f["length"][0]),
            axis=euler("xyz", [float(v) for v in f["axis"][:3]]),
            dof="".join(d[1] for d in f.get("dof", [])),
        )
    parent = {}
    for line in text.split(":hierarchy")[1].splitlines():
        t = line.split()
        if len(t) > 1 and t[0] not in ("begin", "end"):
            parent.update({c: t[0] for c in t[1:]})
    return bones, parent  # parent is in top-down order


def parse_amc(path):
    data = {}
    for line in Path(path).read_text().splitlines():
        t = line.split()
        if t and t[0][0].isalpha():
            data.setdefault(t[0], []).append([float(v) for v in t[1:]])
    return {k: np.array(v) for k, v in data.items()}


def fk(bones, parent, amc):
    """Per bone: start position [m] and world rotation relative to the ASF rest pose, in robot axes."""
    n = len(amc["root"])
    rot = {"root": euler("xyz", amc["root"][:, 3:])}
    start = {"root": amc["root"][:, :3]}
    end = dict(start)
    for b, p in parent.items():
        B = bones[b]
        local = B["axis"] * euler(B["dof"], amc[b]) * B["axis"].inv() if B["dof"] else R.identity(n)
        rot[b] = rot[p] * local
        start[b] = end[p]
        end[b] = start[b] + rot[b].apply(B["dir"] * B["len"])
    Pr = R.from_matrix(P)
    return {b: (start[b] @ P.T * ASF_TO_M, Pr * rot[b] * Pr.inv()) for b in rot}


def min_rot(u, v):
    axis = np.cross(u, v)
    s = np.linalg.norm(axis)
    return R.from_rotvec(axis / s * np.arctan2(s, u @ v)) if s > 1e-9 else R.identity()


def rest_offsets(bones, model, data):
    """Orientation each G1 link would have if the robot stood in the CMU rest (T) pose."""
    link0 = lambda name: R.from_matrix(data.body(name).xmat.reshape(3, 3))  # at zero qpos
    down = np.array([0, 0, -1.0])
    off = {}
    for gmr, (_, bone, link) in BODIES.items():
        d = P @ bones[bone]["dir"]
        if "Up" in gmr or "Leg" in gmr or gmr.endswith("Arm") and "Fore" not in gmr:
            off[gmr] = min_rot(down, d) * link0(link)  # thigh / shin / upper arm hang along -z at zero qpos
        elif "ForeArm" in gmr or "Hand" in gmr:
            # G1 forearm lies along link +x and +q about link +y extends the elbow; CMU +rx flexes it.
            ext = -(P @ bones[bone]["axis"].apply([1, 0, 0]))
            off[gmr] = R.from_matrix(np.column_stack([d, ext, np.cross(d, ext)]))
        else:
            off[gmr] = link0(link)  # pelvis, torso, feet: rest pose == robot zero pose
    return off


def load_gmr(scales, data):
    """Import GMR's retargeter without its package __init__ (which pulls torch/smplx), with a CMU config."""
    pkg = types.ModuleType("general_motion_retargeting")
    pkg.__path__ = [str(HERE / "_gmr" / "general_motion_retargeting")]
    sys.modules["general_motion_retargeting"] = pkg
    from general_motion_retargeting import motion_retarget, params

    cfg = json.loads((params.IK_CONFIG_ROOT / "bvh_lafan1_to_g1.json").read_text())
    for table in (cfg["ik_match_table1"], cfg["ik_match_table2"]):
        for link, entry in table.items():
            entry[4] = [1, 0, 0, 0]  # rest-pose offsets are already applied in rest_offsets()
            if "_yaw_link" in link and "wrist" not in link:
                # CMU gives the hip / shoulder joint centre, but the G1 yaw link origin sits 10-15 cm down the limb
                joint = data.body(link.replace("yaw", "pitch"))
                entry[3] = (data.body(link).xmat.reshape(3, 3).T @ (data.body(link).xpos - joint.xpos)).tolist()
            if "knee" in link:
                entry[1] = 0  # G1 thigh:shin ratio differs from a human's; a knee position target drags the feet off theirs
    cfg["human_scale_table"] = scales
    cfg_path = HERE / "cmu_to_g1.json"
    cfg_path.write_text(json.dumps(cfg, indent=1))
    params.IK_CONFIG_DICT["cmu"] = {"unitree_g1": cfg_path}
    params.ROBOT_XML_DICT["unitree_g1"] = G1_XML
    motion_retarget.print = lambda *a, **k: None
    return motion_retarget.GeneralMotionRetargeting("cmu", "unitree_g1", verbose=False)


def retarget(trial, src_fps):
    subject = trial.split("_")[0]
    bones, parent = parse_asf(HERE / "cmu_raw" / f"{subject}.asf")
    amc = parse_amc(HERE / "cmu_raw" / f"{trial}.amc")
    human = fk(bones, parent, amc)

    model = mujoco.MjModel.from_xml_path(G1_XML)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    off = rest_offsets(bones, model, data)

    # limb-length scales (robot / human), applied by GMR about the root
    dist = lambda a, b: np.linalg.norm(data.body(a).xpos - data.body(b).xpos)
    s_leg = dist("left_hip_pitch_link", "left_ankle_roll_link") / ((bones["lfemur"]["len"] + bones["ltibia"]["len"]) * ASF_TO_M)
    s_arm = (dist("left_shoulder_roll_link", "left_elbow_link") + dist("left_elbow_link", "left_wrist_yaw_link")) / (
        (bones["lhumerus"]["len"] + bones["lradius"]["len"]) * ASF_TO_M
    )
    scales = {k: float(s_arm if "Arm" in k or "Hand" in k else s_leg) for k in BODIES}
    gmr = load_gmr(scales, data)

    # start every clip at the origin facing +x
    fwd = human["root"][1][0].apply([1, 0, 0])
    yaw = R.from_euler("z", -np.arctan2(fwd[1], fwd[0]))
    origin = human["root"][0][0] * [1, 1, 0]

    frames = range(0, len(amc["root"]), src_fps // OUT_FPS)
    qpos = []
    for i, f in enumerate(frames):
        targets = {}
        for name, (pos_bone, rot_bone, _) in BODIES.items():
            pos = yaw.apply(human[pos_bone][0][f] - origin)
            rot = yaw * human[rot_bone][1][f] * off[name]
            if "Foot" in name:  # CMU ankle roll is unreliable (soles tilt 10-20 deg in a plain stance): keep yaw + pitch only
                x = rot.apply([1, 0, 0])
                y = np.cross([0, 0, 1], x) / np.linalg.norm(np.cross([0, 0, 1], x))
                rot = R.from_matrix(np.column_stack([x, y, np.cross(x, y)]))
            targets[name] = [pos, rot.as_quat(scalar_first=True)]
        for _ in range(30 if i == 0 else 1):  # let the first frame converge from the zero pose
            q = gmr.retarget(dict(targets))
        qpos.append(q)
    qpos = np.array(qpos)

    # GMR's ground handling is LAFAN-specific. Shift the whole clip in z so the lower ankle, at its 5th
    # percentile over the clip, sits at the G1's flat-foot ankle height (CMU's floor is flat at y=0).
    soles = [g for g in range(model.ngeom) if model.geom_contype[g] and "ankle_roll" in model.body(model.geom_bodyid[g]).name]
    flat_ankle = -(model.geom_pos[soles, 2] - model.geom_size[soles, 0]).min()
    ankle = []
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_kinematics(model, data)
        ankle.append(min(data.body("left_ankle_roll_link").xpos[2], data.body("right_ankle_roll_link").xpos[2]))
    qpos[:, 2] += flat_ankle - np.percentile(ankle, 5)

    return np.hstack([qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]]), s_leg, s_arm


if __name__ == "__main__":
    out = HERE / "g1_csv"
    out.mkdir(exist_ok=True)
    only = sys.argv[1:]
    manifest_path = out / "manifest.json"
    manifest = {m["name"]: m for m in json.loads(manifest_path.read_text())} if manifest_path.exists() else {}
    for trial, src_fps, name, desc in CLIPS:
        if only and trial not in only:
            continue
        rows, s_leg, s_arm = retarget(trial, src_fps)
        np.savetxt(out / f"{name}.csv", rows, delimiter=",", fmt="%.6f")
        manifest[name] = dict(
            name=name, file=f"{name}.csv", cmu_subject=int(trial.split("_")[0]), cmu_trial=trial,
            description=desc, frames=len(rows), fps=OUT_FPS, duration_s=round((len(rows) - 1) / OUT_FPS, 2),
            source_fps=src_fps, source=f"http://mocap.cs.cmu.edu/subjects/{trial.split('_')[0]}/{trial}.amc",
        )
        print(f"{name}: {len(rows)} frames, leg scale {s_leg:.2f}, arm scale {s_arm:.2f}", flush=True)
    manifest_path.write_text(json.dumps([manifest[c[2]] for c in CLIPS if c[2] in manifest], indent=1))

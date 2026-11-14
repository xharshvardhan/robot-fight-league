"""Play each g1_csv/*.csv kinematically on the G1 and report foot/joint/root sanity numbers.

Usage: .venv/bin/python verify.py                      # table for every clip -> g1_csv/verify.json
       .venv/bin/python verify.py name:t0:t1 ...       # also render contact_sheet.png (6 frames per clip, t in s)
"""
import os
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).parent
SCENE = os.environ.get("G1_MENAGERIE", "mujoco_menagerie/unitree_g1") + "/scene.xml"  # g1.xml + floor
FPS = 30

model = mujoco.MjModel.from_xml_path(SCENE)
data = mujoco.MjData(model)
soles = [g for g in range(model.ngeom) if model.geom_contype[g] and "ankle_roll" in model.body(model.geom_bodyid[g]).name]
left = [g for g in soles if "left" in model.body(model.geom_bodyid[g]).name]
lo, hi = model.jnt_range[1:].T  # joint 0 is the free base


def set_frame(row):
    data.qpos[:3] = row[:3]
    data.qpos[3:7] = row[[6, 3, 4, 5]]  # CSV is xyzw, MuJoCo is wxyz
    data.qpos[7:] = row[7:]
    mujoco.mj_forward(model, data)


def check(rows):
    foot = []  # per frame, per foot: lowest sole point z, sole centre xy
    for row in rows:
        set_frame(row)
        z = data.geom_xpos[soles, 2] - model.geom_size[soles, 0]
        isl = np.isin(soles, left)
        foot.append([[z[m].min(), *data.geom_xpos[np.array(soles)[m], :2].mean(0)] for m in (isl, ~isl)])
    foot = np.array(foot)  # (T, 2, 3)
    q = rows[:, 7:]
    excess = np.maximum(lo - q, q - hi).max(0)
    speed = np.linalg.norm(np.diff(foot[:, :, 1:], axis=0), axis=2) * FPS
    planted = foot[1:, :, 0] < 0.02
    return dict(
        seconds=round((len(rows) - 1) / FPS, 1),
        min_foot_z=round(float(foot[:, :, 0].min()), 3),
        max_penetration=round(float(max(0, -foot[:, :, 0].min())), 3),
        frames_penetrating_gt_1cm=int((foot[:, :, 0].min(1) < -0.01).sum()),
        frames_airborne_gt_3cm=int((foot[:, :, 0].min(1) > 0.03).sum()),
        joints_over_range=[model.joint(j + 1).name for j in np.where(excess > 1e-4)[0]],
        max_range_excess_rad=round(float(max(0, excess.max())), 4),
        root_z=[round(float(rows[:, 2].min()), 2), round(float(rows[:, 2].max()), 2)],
        planted_foot_slide_m_per_s=round(float(speed[planted].mean()), 3) if planted.any() else None,
        max_joint_speed_rad_s=round(float(np.abs(np.diff(q, axis=0)).max() * FPS), 1),
    )


def contact_sheet(specs, path, n=6, w=320, h=240, azimuth=110):
    from PIL import Image

    renderer = mujoco.Renderer(model, h, w)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 2.0, -10, azimuth  # side-on, slightly in front of a robot that starts facing +x
    sheet = Image.new("RGB", (w * n, h * len(specs)))
    for r, spec in enumerate(specs):
        name, t0, t1 = spec.split(":")
        rows = np.loadtxt(HERE / "g1_csv" / f"{name}.csv", delimiter=",")
        for c, t in enumerate(np.linspace(float(t0), float(t1), n)):
            set_frame(rows[min(int(round(t * FPS)), len(rows) - 1)])
            cam.lookat[:] = [*data.qpos[:2], 0.7]
            renderer.update_scene(data, cam)
            sheet.paste(Image.fromarray(renderer.render()), (c * w, r * h))
    sheet.save(path)
    print("wrote", path)


if __name__ == "__main__":
    report = {}
    for f in sorted((HERE / "g1_csv").glob("*.csv")):
        rows = np.loadtxt(f, delimiter=",")
        assert rows.shape[1] == 36, f"{f.name}: expected 36 columns"
        assert np.allclose(np.linalg.norm(rows[:, 3:7], axis=1), 1, atol=1e-4), f"{f.name}: quaternion not unit"
        report[f.stem] = r = check(rows)
        print(f"{f.stem:32s} {r['seconds']:5.1f}s foot_z_min {r['min_foot_z']:+.3f} pen>1cm {r['frames_penetrating_gt_1cm']:4d}f "
              f"air {r['frames_airborne_gt_3cm']:4d}f root_z {r['root_z']} slide {r['planted_foot_slide_m_per_s']} "
              f"qvel_max {r['max_joint_speed_rad_s']} over_range {r['joints_over_range'] or '-'}")
    (HERE / "g1_csv" / "verify.json").write_text(json.dumps(report, indent=1))
    if sys.argv[1:]:
        contact_sheet(sys.argv[1:], HERE / "contact_sheet.png")

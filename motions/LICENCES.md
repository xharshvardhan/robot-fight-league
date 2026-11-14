# Licences relied on (checked 11 Nov 2026)

## Motion data: CMU Graphics Lab Motion Capture Database
- Source: ASF/AMC files downloaded directly from http://mocap.cs.cmu.edu/subjects/<subject>/ (kept in `cmu_raw/`).
  No AMASS, LAFAN1, Bandai Namco or cgspeed BVH data was used.
- Terms, quoted from the site (saved copies in `_ref/cmu_home.html`, `_ref/cmu_faq.html`):
  - Home page: "This dataset of motions is free for all uses."
  - Home page: "This data is free for use in research projects. You may include this data in commercially-sold
    products, but you may not resell this data directly, even in converted form."
  - FAQ: "The motion capture data may be copied, modified, or redistributed without permission."
- What that means here: using the motions in monetised videos is allowed. Do **not** sell the clips themselves
  (the CSVs in `g1_csv/` are the CMU data "in converted form").
- Requested credit: "The data used in this project was obtained from mocap.cs.cmu.edu. The database was created
  with funding from NSF EIA-0196217."

## Retargeting code
- GMR (github.com/YanjieZe/GMR, commit 25f56db, pruned copy in `_gmr/`): MIT. Only its IK solver class and the
  LAFAN1->G1 weight table are used; none of its bundled motion data, SMPL-X body models or robot assets.
- mink 1.4.0: Apache-2.0. MuJoCo 4.0.0: Apache-2.0. daqp 0.11.2: MIT. qpsolvers 4.13.0: LGPL-3.0 (used unmodified
  as a library). numpy, scipy: BSD. Pillow: MIT-CMU.

## Robot model
- Unitree G1 from MuJoCo Menagerie (`mujoco-armwrestle/menagerie/unitree_g1`, read only): BSD-3-Clause,
  copyright Unitree Robotics.

## Output format reference (no code copied)
- mjlab `docs/source/training/motion_imitation.rst` and `src/mjlab/scripts/csv_to_npz.py`; unitree_rl_mjlab README.

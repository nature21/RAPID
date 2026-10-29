---
name: repair-real2sim-scene
description: Align a generated Real2Sim MuJoCo scene to its real RGB/RGB-D image by correcting object geometry, meshes, poses, and missing or spurious objects, with physical validation. Use for scene repair or visual alignment of a supplied Real2Sim bundle.
---

# Repair Real2Sim Scene

Make the supplied scene match the real camera image as closely as the evidence
supports, while remaining physically healthy. You may edit `scene.xml`, modify
or replace mesh files, add missed objects, and remove spurious or duplicate
objects. Physical validation is required, but passing it does not establish
visual alignment.

## Inspect the evidence

1. Read `physics_preflight.json` and locate every object-object contact and each
   object that drifted, dropped, rotated, or retained velocity during the
   settling rollout, not only the first failing pair.
2. Inspect the attached original image, `input.png`, and the corresponding
   object masks under `masks/`. Use left-to-right ordering, visible silhouette,
   support contact, occlusion, and known object proportions as constraints.
3. Read `scene.xml`. Treat each direct `<worldbody><body>` as one reconstructed
   object. Its `pos` and `quat` define world pose; geoms use body-local poses.
   Box `size` is half-extents. The initial object inventory and masks can be
   wrong; resolve contradictions against the real image and valid metric depth.
4. Use `depth_camera.npz` to constrain metric size, depth, and support placement. It contains
   metric intrinsics, calibrated world/camera transforms, depth, and point maps.
5. Render the current XML from the source camera with the launcher-provided
   `python -m real2sim.scene_repair render` command. Inspect that PNG against
   `input.png` before deciding on edits.

## Correct the scene

- Move or rotate the object body when its entire mesh and collision belong at a
  different world pose.
- Change a collision box's local `pos`, `quat`, or `size` when the box axes or
  half-extents do not represent the visible object.
- When correcting visual size, apply a consistent `scale` to every `<mesh>`
  asset used by that object and update its collision box accordingly. Do not
  shrink only the collision geometry to silence preflight.
- Edit mesh vertices/topology, crop reconstruction artifacts, or replace poor
  geometry when pose/scale changes cannot reproduce the visible silhouette.
  Keep collision geometry consistent with the resulting visible object; use
  compound collision parts when a single box or convex mesh is inadequate.
- Add an object when image evidence shows a missed object. Remove duplicates
  and false detections based on image/depth evidence. Occlusion alone does not
  prove an object is absent. Explain the evidence for every addition/deletion.
- Keep names for retained objects. Give additions descriptive unique body names
  and an active primary geom named `<body_name>_collision`; additional collision
  geoms are allowed. Objects must remain direct worldbody bodies without nested
  bodies. Use a freejoint for new movable objects and static bodies for fixed
  scene structure supported by the image.
- Edit existing scene asset files in place if needed. Put new mesh files and
  their material/texture dependencies under `meshes/`, using bundle-relative
  references. New XML mesh/material/texture assets are allowed. Preserve existing
  shared non-mesh assets; use a new object-specific material for appearance edits.
- Keep objects resting just above the support surface. Preserve the visible
  left-to-right order and avoid introducing new occlusion or implausible gaps.
- Remove the reset-state penetrations that inject separating impulses. Passing
  a permissive penetration threshold is insufficient if the settling rollout
  moves or topples the objects.
- Keep static dividers fixed and dynamic packages free unless the source
  evidence specifically requires a pose or size correction.
- If collision size changes materially, keep the existing mass and update the
  inertial pose/diagonal inertia consistently with the corrected box.

After correcting collision geometry, perform a separate visual alignment pass.
A collision-only edit can pass physics while leaving a shifted, tilted, or
oversized visual mesh. An unchanged source-camera render is not evidence that
the visual matches the corrected collision. Render `--geometry overlay` and
inspect every edited object's visual/collision agreement and real RGB silhouette.
Apply a common transform to every visual color/texture submesh of an object;
independently fitting each colored fragment destroys the object's appearance.

When the corrected collision is a trustworthy box for the real object, use the
launcher-provided `align-visuals --objects BODY_NAME ...` command. It fits the
entire visual assembly to the box, preserving collision poses and physics.
Inspect the result against RGB before accepting it. Use manual mesh edits for
compound collisions and shapes for which box fitting is inappropriate. Do not
move or shrink a correct collision merely to accommodate a bad visual mesh.

Preserve the calibrated camera, lights, world frame, floor, global physics
settings, and retained objects' joints and contact settings. Do not freeze
movable objects or disable collisions to pass validation. Keep raw camera
images, depth, masks, repository source, and repair backups unchanged. The
launcher synchronizes the final object inventory in metadata and restores XML
and asset files on failure; do not edit metadata yourself.

## Validate and iterate

Run the exact validation command supplied in the task. It enforces the allowed
edit scope, compiles MJCF, checks reconstructed-object and support-floor
penetration, then simulates two unforced seconds. Every dynamic object must stay
near its reconstructed pose without toppling and finish near rest. If it fails,
use the per-object drift/rotation/velocity metrics, contact pairs, normals, and
source-camera renders to refine the pose/size.

Validation also compares visual and collision envelopes in the collision frame.
It rejects face discrepancies exceeding the larger of 1 cm or 15% of that
collision extent. Read the per-object metrics and correct the visual geometry
when the collision already matches the real object. This geometric check does
not replace inspecting RGB; two mutually consistent shapes can both be wrong.

Render the result at reset and after the settling rollout from the calibrated
source camera. Inspect both. Do not finish merely because compilation or the
reset-state contact gate succeeds: the settled scene must preserve the object
silhouettes, ordering, upright orientation, scale, and support placement seen
in the original image.

In the final response, summarize mesh/pose changes and object additions/removals
with their image evidence, the final maximum penetration, settling metrics, and
both calibrated render paths. State remaining visual mismatches and uncertainty
about occluded geometry rather than claiming alignment from physics alone.

# Generated-image reconstruction: three reusable stages

This branch adds two independent entry points. Original SfM, chart alignment,
Gaussian model, renderer, mesh extraction, and existing training entry points
are unchanged. Two pointmap-loading fixes associate points by filename and use
bounded sampling indices (the original two-view case could index out of range). It requires the original repository's installed environment,
CUDA extensions, MASt3R/retrieval weights, and DepthAnythingV2 weights. Cloning
this branch does not install those dependencies or download model weights.

Run commands from the repository root in the installed MAtCha environment.
Paths below are examples; no script fixes a scene name, image count, chart count,
or 1280x960 size. Full input resolution is used during Gaussian training.

## Inputs

```text
data/SCENE/
  images/          # All generated input images, unique filenames/stems
  keyframes.txt    # Selected exact filenames, one per line, including extension
```

Example `keyframes.txt`:

```text
target_000001.png
target_000021.png
target_000047.png
```

Blank lines and lines starting with `#` are allowed. Paths, integer indices,
duplicates, and names absent from SfM are rejected. Names are sorted into the
same lexical order used by the upstream camera loader. The file selects charts;
it does not limit dense RGB supervision. Select at least two charts. If selection was already performed,
copy its `frames.txt` to `keyframes.txt`. For the existing 398-image task, use
`data/robo_lounge_generated_398/keyframes_30/frames.txt` directly if present.

## Stage 1: all-image SfM (unchanged)

```bash
python train.py \
  -s data/SCENE/images \
  -o output/SCENE \
  --sfm_config unposed \
  --sfm_only
```

Do not pass `--image_idx`/`--n_images`: all input images must receive cameras.
MASt3R output is `output/SCENE/mast3r_sfm`. SfM GPU requirements remain upstream.

## Stage 2: selected-chart preparation and alignment

```bash
python scripts/align_selected_charts.py \
  --sfm_scene output/SCENE/mast3r_sfm \
  --keyframes data/SCENE/keyframes.txt \
  --output_path output/SCENE/charts \
  --config default
```

`--config strong` invokes upstream stronger deformation regularization;
it does not change the training-stage chart schedule. The default encoder is
`vitl`; override `--depthanything_encoder` and
`--depthanythingv2_checkpoint_dir` if needed.

This entry validates inputs, prepares consistent subsets and calls the original
`scripts/align_charts.py` in a subprocess using the current Python interpreter.
Alignment failure is propagated as a nonzero exit code. Output:

```text
output/SCENE/charts/
  scene/                    # Selected chart images, cameras, pointmaps, points
  dense_scene/              # ALL SfM cameras/images/points, not just keyframes
  pipeline_manifest.json    # Sorted names and camera-source identity
  charts_data.npz           # Original alignment output
```

All camera/point coordinates and IDs are retained. Image observations and point
tracks are kept consistent, correcting raw-pixel track indices from the upstream
MASt3R exporter to compact observation indices. No point-budget sampling occurs.
This reader intentionally supports this repo's **single-track MASt3R binary
export**, not arbitrary COLMAP/Pi3 models; incompatible tracks are rejected.
Point files/PLY are streamed in chunks to bound temporary CPU memory.

For preparation without GPU alignment, add `--prepare_only`. Rerunning without
that option reuses matching prepared subsets and starts alignment. Existing
`charts_data.npz` is never overwritten; use a new output directory for another
alignment configuration. Failed partial preparation requires a fresh directory.

Image/pointmap links are relative. Transfer the complete experiment tree and
source images while preserving relative layout, or rerun preparation on the
server. A directory containing symlinks alone is not a self-contained dataset.

## Stage 3: all-image dense Gaussian training

```bash
python scripts/train_dense_gaussians.py \
  --sfm_scene output/SCENE/mast3r_sfm \
  --chart_scene output/SCENE/charts \
  --output_path output/SCENE/free_gaussians \
  --config configs/free_gaussians_refinement/generated_dense.yaml
```

The YAML contains training parameters only. Scene paths come from CLI arguments.
The trainer verifies the source camera identity, all-image dense membership,
chart membership/order and geometry shapes. All images use the generated-image
loss policy, including when their filenames have no `target_` prefix.
This entry does not implement a separate real/generated mixture policy.

Defaults:

| Setting | Value |
| --- | --- |
| Iterations | 30000 |
| Image resolution/storage | Full source resolution / GPU |
| SH degree | 3 |
| Chart schedule | Every 20 steps throughout training |
| Dense schedule | All other steps; shuffled without replacement per cycle |
| RGB weight / DSSIM | 1 / 0.2 |
| Geometry | Aligned chart depth, normals, curvature and anisotropy |
| Extra DAV2 depth-order supervision | Disabled for both chart and dense views |
| Normal consistency | 0.05 after step 7000 |
| Distortion loss | 0 |
| Depth ratio / mip filter | 0.5 / enabled |
| Densification | 500–15000, every 100, gradient threshold 0.0003 |
| Opacity reset | Every 3000 steps |
| Initialization | All valid chart faces + all dense SfM points, no point budget |
| PLY | 7000, 15000, 30000 and final step |
| Recoverable checkpoint | Every 5000 steps through 30000, and final step |

Chart images also remain eligible for dense RGB steps. A dense step samples one
image, not a batch containing every image. As in the original pipeline, chart
alignment normalizes its working coordinates and saves `scale_factor`; training
divides chart points and depths by that factor to recover the original SfM scale.
Dense points and camera translations retain their original SfM coordinates.
No additional training-time normalization is applied. Chart masks exclude
nonfinite geometry and nonpositive depths, with the same rule used by validation
and training. Depths above `100` remain valid: `Camera.zfar` is not a hard far
visibility cutoff in the upstream CUDA rasterizer. Visibility is handled by the
renderer, not by the data validator. Use a consistent image/pointmap shape across a scene, as required by upstream
parallel chart alignment. RGB sources need not be 1280x960:
no resizing is introduced and no fixed image/pointmap dimensions are assumed.
All-view initialization can still exceed GPU capacity; this branch does not
promise that arbitrary scene sizes fit any particular GPU.

Use `--data_device cpu` to store RGB on CPU. Training remains CUDA-based.
Use `--validate_only` to check prepared data/configuration before model loading.
Use `--smoke-test --output_path output/SCENE/free_gaussians_smoke` for a 40-step
run including chart updates at 20/40, final PLY and a checkpoint. This does not
reduce initialization geometry or guarantee low memory for a large scene.

Resume with the same data/configuration and add:

```bash
--resume output/SCENE/free_gaussians/chkpnt10000.pth
```

Checkpoints include Adam, mip, SH, sampling queues and RNG state. A hash validates
configuration, prepared manifests and chart NPZ. CPU/GPU image residency and
output location may change on resume; training loss/optimizer changes are
rejected. Custom checkpoints cannot be passed to the upstream trainer.
`--iterations` overrides the stopping step, not the position-LR decay horizon
(`--position_lr_max_steps`, default 30000).

PLY uses upstream field names, including mip, and can be read by compatible
viewers. Existing mesh extraction entry points are not changed or automatically
run by this pipeline.

## Validation

```bash
python -m unittest discover -s tests -p test_dense_generate.py -v
```

CPU tests cover arbitrary view counts, selected chart export, retained camera
and point coordinates, compact COLMAP tracks, rejection of bad inputs, portable
camera identity, chart masks, fixed scheduling after step 3000, exact sampler
restore, and streamed PLY SH/mip fields.

Integration validation for this branch also exercised two actual charts through
1000 alignment iterations and then 40 GPU training steps with the resulting
NPZ (538306 initial Gaussians). A separate small fixture exercised densification
from 3883 to 6154 Gaussians at step 600, then resumed step 601 with updated
parameters and Adam step 601, including CUDA-to-CPU image residency switching.
These small tests do not establish memory usage or quality for a full 398-view
run; no full reconstruction is launched when installing/checking out the branch.

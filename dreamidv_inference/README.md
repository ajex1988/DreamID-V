# DreamID-V Unified Inference Module

This directory contains a standalone inference-oriented package that merges the four original entry points:

- `generate.py`
- `generate_dreamidv.py`
- `generate_dreamidv_dwpose.py`
- `generate_dreamidv_faster.py`

The new package keeps the original model implementations in `dreamidv_wan` and `dreamidv_wan_faster`, but centralizes all orchestration logic into a single YAML-driven interface with shared chunking.

## What This Package Solves

The original repository had four separate scripts with duplicated logic:

- distributed/runtime setup
- reference asset handling
- preprocessing and caching
- save-path creation
- model invocation

The new module consolidates that logic and adds a single chunking implementation at the top level, based on the corrected behavior from `generate_dreamidv_faster_chunked.py`.

That means all four pipelines now share the same chunk planner:

- sequence-level preprocessing
- chunk splitting
- `4n+1` frame correction
- pad-then-trim generation
- chunk concat and final save

This avoids the previous class of frame-loss bugs where invalid chunk sizes were silently truncated by DreamID-V.

## Directory Layout

```text
dreamidv_inference/
  __init__.py
  backends.py
  chunking.py
  cli.py
  config.py
  preprocessors.py
  runner.py
  run_all_pipelines.py
  configs/
    manual.yaml
    express.yaml
    dwpose.yaml
    faster.yaml
  tests/
    test_planning.py
  README.md
```

## Unified Pipeline Modes

The `pipeline.name` field determines which legacy behavior is reproduced.

### `manual`

Equivalent to `generate.py`.

Use this when you already have:

- the reference source video
- a face mask video
- a pose video
- the reference image

The backend is the standard `dreamidv_wan.DreamIDV` model and the inference call receives four conditioning paths:

1. source video
2. face mask video
3. reference image
4. pose video

### `express`

Equivalent to `generate_dreamidv.py`.

This mode auto-generates mask and pose videos with the `express_adaption` pipeline before inference.

The backend is still the standard `dreamidv_wan.DreamIDV` model. The difference is only how pose/mask assets are produced.

### `dwpose`

Equivalent to `generate_dreamidv_dwpose.py`.

This mode auto-generates pose and mask videos with `pose.extract.process_dwpose`.

The backend is still the standard `dreamidv_wan.DreamIDV` model.

### `faster`

Equivalent to `generate_dreamidv_faster.py`.

This mode uses DWPose preprocessing but switches the backend to `dreamidv_wan_faster.DreamIDV`.

This backend consumes:

1. source video
2. face mask video
3. reference image

It does not consume pose embeddings.

## YAML Config Structure

Every run is configured through one YAML file with five sections:

```yaml
pipeline:
  name: dwpose
  prompt: chang face
  temp_dir: null

model:
  task: swapface
  size: 1280*720
  frame_num: 81
  sample_fps: 24
  ckpt_dir: /path/to/wan/checkpoints
  dreamidv_ckpt: /path/to/dreamidv.ckpt
  offload_model: null
  ulysses_size: 1
  ring_size: 1
  t5_fsdp: false
  t5_cpu: false
  dit_fsdp: false
  sample_solver: unipc
  sample_steps: 12
  sample_shift: 5.0
  sample_guide_scale_img: 4.0
  base_seed: -1

inputs:
  ref_image: ./assets/test_case/ref_image/an_1.jpg
  ref_video: ./assets/test_case/ref_video/a_girl.mp4
  ref_video_facemask: null
  ref_video_pose: null

chunk:
  enabled: true
  chunk_size: 81
  chunk_dir: null
  reuse_preprocessed: true
  reuse_chunk_files: false

output:
  save_file: null
  output_dir: ./results_unified
```

## Chunking Behavior

Chunking now sits above every backend, not only the faster one.

### Key Rules

- `chunk.chunk_size` is treated as the requested maximum chunk size.
- The runner converts it to the largest valid DreamID-V length of the form `4n+1`.
- If the last chunk is shorter than a valid DreamID-V length, it is padded by repeating the last frame.
- The generated tensor is trimmed back to the original chunk length before concatenation.

### Why This Matters

DreamID-V internally truncates invalid frame counts to `4n+1`.

For example:

- requested chunk size = `80`
- DreamID-V valid size = `77`

If you feed raw 80-frame chunks directly into the model, each chunk loses 3 frames. The new unified runner prevents that by normalizing chunk sizes before generation.

## Runtime Flow

For all pipeline modes, the flow is:

1. Load YAML config.
2. Initialize distributed/runtime settings.
3. Prepare or generate conditioning assets.
4. Probe source video length.
5. Build chunk plan.
6. For each chunk:
   - slice source/conditioning videos
   - pad chunk to a valid `4n+1` length
   - run the backend on the padded chunk
   - trim output back to the original chunk length
7. Concatenate all chunks.
8. Save one final output video.

## Main Entry Points

### Single run

```bash
python -m dreamidv_inference.cli --config dreamidv_inference/configs/dwpose.yaml
```

### Run multiple pipeline variants with one interface

```bash
python -m dreamidv_inference.run_all_pipelines \
  --config dreamidv_inference/configs/manual.yaml \
  --pipelines manual express dwpose faster
```

This script loads one base config, swaps only `pipeline.name`, and runs each pipeline through the same unified runner.

For `manual`, your config must provide `inputs.ref_video_facemask` and `inputs.ref_video_pose`.

For `express`, `dwpose`, and `faster`, those fields are ignored because preprocessing is generated automatically.

## Practical Recommendations

### When to use `manual`

Use `manual` when you already trust your own pose and face-mask videos and do not want runtime preprocessing.

### When to use `express`

Use `express` when you want preprocessing aligned with the original `express_adaption` path.

### When to use `dwpose`

Use `dwpose` when you want the standard DreamID-V backend but with DWPose-generated assets.

### When to use `faster`

Use `faster` when throughput matters most and you are willing to use the reduced conditioning path.

## Dependency Notes

The runtime environment still needs the original project dependencies because this package calls:

- `dreamidv_wan`
- `dreamidv_wan_faster`
- `pose.extract`
- `express_adaption`

YAML loading additionally requires `PyYAML`.

If `PyYAML` is not installed:

```bash
pip install pyyaml
```

## Validation

The package includes a lightweight structural test:

```bash
python -m unittest dreamidv_inference.tests.test_planning
```

This does not run full model inference. It verifies the corrected chunk-planning logic that protects frame counts.

## Migration Notes

If you previously called the old scripts directly:

- `generate.py` -> `pipeline.name: manual`
- `generate_dreamidv.py` -> `pipeline.name: express`
- `generate_dreamidv_dwpose.py` -> `pipeline.name: dwpose`
- `generate_dreamidv_faster.py` -> `pipeline.name: faster`

The unified interface is intended to be imported as a module or called from `python -m dreamidv_inference.cli`.

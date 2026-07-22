# Deterministic Color Masks

This folder contains an independent color-mask runtime. It does not import the
existing vision review tools, browser app, HNL rules, semantic color helpers, or
run-local color ID code.

Runtime dependencies are `numpy` and `Pillow`.

The runtime contract is deliberately small:

```text
source JPEG frame
  -> PIL.Image.open(path).convert("RGB")
  -> pack each pixel as 0xRRGGBB
  -> color_lut_v1.npz lookup
  -> uint8 mask bitfield
```

## Files

```text
src/color_masks/
  mask_frames.py              Runtime frame-to-mask script
  artifacts/color_lut_v1.npz  Dense RGB LUT generated once
  input_frames/               Default drop folder for JPEG frames
  output/                     Default mask output folder
```

The LUT generator lives outside `src`:

```text
accessory_scripts/color_masks/build_color_lut.py
```

## LUT Format

`artifacts/color_lut_v1.npz` stores:

```text
lut: uint8[16777216]
metadata_json: JSON string
```

The LUT index is the exact decoded RGB key:

```text
rgb_key = (R << 16) | (G << 8) | B
```

The LUT value is a bitfield:

```text
bit 0 = layer 001
bit 1 = layer 002
bit 2 = layer 003
bit 3 = layer 007
```

Overlaps are preserved. If a color belongs to layers `002` and `007`, its value
contains both bits.

## Build The LUT

Generate or refresh the LUT with:

```bash
python accessory_scripts/color_masks/build_color_lut.py
```

The generator reads existing artifact JSON only. It does not import current
project code. By default it reads:

```text
assets/color_layers/layer_sets.json
```

And writes:

```text
src/color_masks/artifacts/color_lut_v1.npz
```

## Run Batch Masking

Process the default drop folder:

```bash
python src/color_masks/mask_frames.py
```

Process an explicit frame folder:

```bash
python src/color_masks/mask_frames.py \
  --frames-dir /path/to/jpeg_frames \
  --output /path/to/mask_output
```

Process a frame manifest:

```bash
python src/color_masks/mask_frames.py \
  --manifest assets/frame_runs/run-20260720T023225Z/hnl_decoded_manifest.json \
  --output /path/to/mask_output
```

Process a live/drop folder until interrupted:

```bash
python src/color_masks/mask_frames.py \
  --frames-dir src/color_masks/input_frames \
  --watch
```

## Output

The script writes:

```text
output/
  mask_manifest.json
  masks/
    frame_000000_0.maskbits.u8.bin
    frame_000001_1.maskbits.u8.bin
```

Each `.maskbits.u8.bin` file is a raw `uint8` array in row-major order with
shape `height * width`.

`mask_manifest.json` records the source frame, mask file, dimensions, SHA-1,
selected pixel count, overlap count, and per-layer pixel counts for every frame.

## Exactness Boundary

This system is deterministic for the same source image bytes decoded through:

```text
PIL.Image.open(path).convert("RGB")
```

Do not expect exact byte parity if frames are resized, transcoded, recompressed,
decoded through browser canvas, or decoded through a different JPEG library.

The runtime intentionally has no fallback color semantics. A pixel either has an
exact decoded RGB value in the LUT or it maps to zero.

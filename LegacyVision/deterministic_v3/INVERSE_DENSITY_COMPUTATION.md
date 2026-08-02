# Inverse Density Computation Reference

This document traces the implementation that produces the **Final Inverse Density** view in `index.html`, including the offline Python preprocessing, the live browser calculation, the function call order, and the calibrated settings reported on 2026-07-31.

## Important implementation boundary

The calibrated Final Inverse Density is **not calculated by Python**. It is calculated live by JavaScript embedded in `index.html`.

Python is used offline to:

1. segment source frames into a binary gate mask;
2. crop and resize each gate object into a 120 x 120 mask tile;
3. package those mask tiles into an atlas loaded by the UI; and
4. optionally package per-color-layer membership into a second atlas.

The UI reads the Python-produced binary mask, then recomputes density with the current controls. In particular, the Python builder constants `NEIGHBOR_RADIUS = 3` and `DENSITY_GAMMA = 1.5` are **not** the calibrated UI values of radius 12 and inverse gamma 1.65.

The Python atlas also contains precomputed `density`, `inverse_density`, and support images, but the current Final Inverse Density renderer does not read those images. `decodeGateRow()` reads only the atlas `source` and `mask` tiles, and the UI computes `finalInverseDensity` from the decoded mask.

## Reported calibrated profile

| UI control | DOM ID | Runtime setting | Reported value |
|---|---|---|---:|
| Neighbor radius | `gateRadius` | `settings.radius` | 12 px |
| Density gamma | `gateDensityGamma` | `settings.densityGamma` | 1.00 |
| Inverse gamma | `gateInverseDensityGamma` | `settings.inverseDensityGamma` | 1.65 |
| Normalization | `input[name="gateDensityMode"]` | `settings.mode` | `mean` (mean-relative) |
| Ridge centroid | `gateRidgeCentroidToggle` | `settings.ridge` | treated as enabled for the calibrated formula below |
| Ridge radius | `gateRidgeRadius` | `settings.ridgeRadius` | 10 px |
| Ridge gamma | `gateRidgeGamma` | `settings.ridgeGamma` | 4.15 |

Two inputs that affect exact reproduction were not included in the reported calibration:

- `relative cap` (`gateRelativeCap` / `settings.relativeCap`) is required by mean-relative normalization. Its source default is **2.00**. The formulas below use the symbol `C`; substitute `C = 2.00` if the control was unchanged.
- `inverse uses layer confidence` (`gateInverseDensityLayerConfidenceToggle`) changes the neighborhood mass. Its source default is **unchecked**, which means binary mask mass. Both branches are documented below.

The cleanup toggles and point-centroid toggle were also not reported. Their source defaults are unchecked. Cleanup changes the input mask before density is computed; point-centroid weighting adds another multiplier after inverse density is computed.

## Complete execution order

The relevant live UI call path is:

```text
index.html loads Python-generated collection and layer-sidecar JavaScript
  -> setActiveTab("gates")
  -> renderGateRows()
  -> loadGateAtlas()
  -> loadGateLayerAtlas()
  -> decodeGateRowsInBatches()
  -> decodeGateRow()
       -> threshold Python atlas mask RGB at > 127 into sourceMask
       -> decodeLayerMembership() for optional confidence mass
       -> drawComputedGateRow(row, gateSettings())
            -> computeGateTile(cache, settings)
                 -> runRefinementPipeline(...): sourceMask -> finalMask
                 -> computeDensityFromMask(...): separate normal-density branch
                 -> computeInverseDensityFromMask(...): base inverse density
                      -> fillDensityMass(...)
                           -> layerConfidenceForPixel(...) when enabled
                      -> computeNormalizedDensityFromMass(...)
                           -> integralFrom(...)
                           -> integralRect(...) for each neighborhood
                           -> normalizedSupportInput(...)
                                -> meanRelativeSupport(...) in calibrated mode
                                -> clamp(...)
                      -> reciprocal-gamma inverse transform
                 -> applyGateCentroidWeights(...baseDensity...): density branch
                 -> applyGateCentroidWeights(...baseInverseDensity...): inverse branch
                      -> integralFrom(...) three times for ridge weighting
                      -> integralRect(...) for local mass and x/y moments
                      -> distanceFalloff(...)
                 -> remaining local/directional computations
            -> paintHeatCanvas(...finalInverseDensity...)
```

When a relevant UI control changes, `render()` calls `scheduleGateRecompute()`, which debounces for 40 ms and calls `recomputeGateRows()`. That function snapshots `gateSettings()` once and calls `drawComputedGateRow()` for each decoded row in batches.

## Exact live inverse-density math

The following describes `computeInverseDensityFromMask()` and the inverse call to `applyGateCentroidWeights()`.

### 1. Input mask

Let `F(p)` be `finalMask[p]`, either 0 or 1. `runRefinementPipeline()` begins by copying `sourceMask` to `finalMask`, then applies enabled cleanup stages in their selected order:

1. `applyGateIslandRemoval()`
2. `applyGateVoidFill()`
3. `applyGateProtrusionTrim()`
4. `applyGateIntrusionFill()`

Only enabled stages run. With all cleanup toggles at their source defaults, `F` is an unchanged copy of the Python-produced mask tile.

### 2. Per-pixel density mass

`fillDensityMass()` creates `inverseDensityMass`:

```text
M(q) = 0,                                      when F(q) = 0
M(q) = 1,                                      when F(q) = 1 and confidence is off
M(q) = layerConfidenceForPixel(q),             when F(q) = 1 and confidence is on
```

`layerConfidenceForPixel()` returns the maximum confidence among active membership layers at that pixel. The UI layer order is:

1. `Gate9316main`
2. `Gate9132branding`
3. `Gate9158-9214-9316support`

The source-default confidences are 1.00, 0.80, and 0.80 respectively. If layer membership is unavailable, the function returns 1, so the behavior falls back to binary mass.

### 3. Square-neighborhood density

`computeNormalizedDensityFromMass()` first constructs a summed-area table with `integralFrom()`. For each foreground pixel `p = (x, y)`, it clips a square neighborhood to the tile bounds:

```text
x0 = max(0, x - radius)
y0 = max(0, y - radius)
x1 = min(width  - 1, x + radius)
y1 = min(height - 1, y + radius)
```

With the calibrated radius, `radius = 12`, so an interior neighborhood is 25 x 25 pixels. Border neighborhoods are smaller because they are clipped, and their denominator is reduced accordingly.

`integralRect()` returns the neighborhood mass. The raw local density is:

```text
A(p) = (x1 - x0 + 1) * (y1 - y0 + 1)
d(p) = sum(q in N12(p), M(q)) / A(p)
```

Density is computed only for foreground pixels. Background output is set to zero.

### 4. Mean-relative normalization

The function computes the mean of the raw local densities over foreground pixels only:

```text
mu = sum(p where F(p)=1, d(p)) / count(p where F(p)=1)
```

For `settings.mode === "mean"`, `normalizedSupportInput()` calls `meanRelativeSupport()`:

```text
n(p) = clamp((d(p) / max(mu, 1e-6)) / max(C, 1e-6), 0, 1)
```

Here `C` is `settings.relativeCap`. If the control retained its default value of 2.00:

```text
n(p) = clamp(d(p) / (2 * mu), 0, 1)
```

The percentile floor and ceiling controls are ignored in mean-relative mode.

### 5. Reciprocal inverse-gamma transform

`computeInverseDensityFromMask()` uses the reciprocal of inverse gamma:

```text
e = 1 / max(1e-6, inverseDensityGamma)
b(p) = F(p) * pow(1 - clamp(n(p), 0, 1), e)
```

For the calibrated inverse gamma of 1.65:

```text
e = 1 / 1.65 = 0.606060606060...
b(p) = F(p) * pow(1 - n(p), 0.606060606060...)
```

`b(p)` is stored in `baseInverseDensity`. Because the exponent is below 1, values between 0 and 1 are lifted relative to a linear inverse.

The reported density gamma of 1.00 does not enter this inverse branch. It is used by the separate `computeDensityFromMask()` call that produces `baseDensity`.

### 6. Ridge-centroid weighting

The second `applyGateCentroidWeights()` call transforms `baseInverseDensity` into `finalInverseDensity`.

When ridge weighting is enabled, the function builds three summed-area tables over `b(q)`:

```text
W(q)  = b(q)
WX(q) = b(q) * q.x
WY(q) = b(q) * q.y
```

For each foreground pixel `p`, it clips a square window of radius 10 to the tile, then calculates the inverse-density-weighted local centroid:

```text
mass = sum(q in N10(p), b(q))
cx   = sum(q in N10(p), b(q) * q.x) / mass
cy   = sum(q in N10(p), b(q) * q.y) / mass
delta(p) = hypot(p.x - cx, p.y - cy)
```

`distanceFalloff()` computes the calibrated ridge multiplier:

```text
t(p) = 1 - clamp(delta(p) / 10, 0, 1)
ridgeMultiplier(p) = pow(t(p), 4.15)
```

The displayed result is:

```text
finalInverseDensity(p) = b(p) * ridgeMultiplier(p)
```

If the local mass is zero, the multiplier is zero. If ridge weighting is disabled, its multiplier is 1 and the ridge radius/gamma controls have no effect. If point-centroid weighting is also enabled, its `distanceFalloff()` multiplier is multiplied with the ridge multiplier before the result is written.

### 7. Rendering

`paintHeatCanvas()` clamps each `finalInverseDensity` value to [0, 1] and interpolates between:

- trough RGB: `[37, 99, 235]`
- peak RGB: `[255, 59, 68]`
- background RGB: `[5, 6, 7]`

Rendering changes only the color representation; it does not change the computed float values.

## Python modules and functions involved

### `build_gate_collection.py`

Relevant imports and their use:

| Module | Functions/classes used in this path |
|---|---|
| `argparse` | `ArgumentParser` for builder inputs |
| `json` | reads color-layer definitions and writes collection metadata |
| `math` | `ceil()` for atlas sizing |
| `re` | extracts layer JSON from its JavaScript assignment |
| `pathlib` | `Path` for input/output locations |
| `numpy` (`np`) | arrays, color math, masks, connected components fallback data, summed-area density, powers, and image conversion |
| Pillow (`PIL.Image`) | opens frames, crops/resizes mask tiles, creates atlas images, and saves PNGs |
| OpenCV (`cv2`, optional) | `connectedComponents(..., connectivity=4)`; `connected_labels()` has a pure-Python fallback |

Offline call order relevant to the mask consumed by Final Inverse Density:

```text
main()
  -> parse_args()
  -> load_color_layers()
  -> frame_path_for()
  -> Image.open(...).convert("RGB")
  -> np.array(...)
  -> build_composite()
       -> srgb_to_oklab()
       -> layer_contains()
            -> bezier_polygon() -> quadratic_point(), when a Bezier layer is used
            -> points_in_polygon(), when a Bezier layer is used
       -> prune_without_main() -> connected_labels()
       -> prune_small_groups() -> connected_labels()
  -> standard_components() -> connected_labels()
  -> build_tiles()
       -> centered_tile_from_crop(..., Image.Resampling.NEAREST, ...)
       -> density_from_mask()                    # offline preview only
       -> inverse_density = mask * (1 - density) # offline preview only
       -> inverse_support = mask * inverse_density ** 1.5
       -> heat_rgb() / support_rgb()             # offline preview colors
  -> write_outputs()
```

The mask tile from `build_tiles()` is the Python result used by the live calculation. `centered_tile_from_crop()` preserves aspect ratio, scales to fit 120 x 120, centers the result, and pads with zero. Mask resizing uses nearest-neighbor sampling.

The builder's offline preview formula is:

```text
density_from_mask = square-neighborhood binary mean with radius 3
inverse_density   = mask * (1 - density_from_mask)
inverse_support   = mask * pow(inverse_density, 1.5)
```

This preview formula differs from the calibrated live path in normalization, radius, reciprocal gamma, and ridge weighting.

### `build_gate_layer_sidecar.py`

This module is relevant only when an operation uses per-layer confidence. It imports shared segmentation functions and constants from `build_gate_collection.py`.

Its external modules are `argparse`, `json`, `math`, `collections.defaultdict`, `pathlib.Path`, `numpy`, and `PIL.Image`.

Relevant call order:

```text
main()
  -> parse_args()
  -> load_color_layers()                         # imported from collection builder
  -> frame_source_path()
  -> Image.open(...).convert("RGB")
  -> layer_membership_for_frame()
       -> srgb_to_oklab()                        # imported
       -> layer_contains()                       # imported
       -> prune_without_main()                   # imported
       -> prune_small_groups()                   # imported
  -> membership_tile()
       -> centered_tile_from_crop(...NEAREST...) # imported
  -> write_sidecar()
```

`membership_tile()` stores binary membership for the three layers in the RGB channels. In the browser, `decodeLayerMembership()` thresholds each channel at greater than 127. When inverse layer confidence is enabled, `layerConfidenceForPixel()` uses the maximum confidence of the active channels.

## Source locations

- `index.html:708-746` — density, inverse, normalization, and centroid controls
- `index.html:1061-1080` — clamping and settings readers
- `index.html:1214-1223` — inverse-density procedure-stage metadata
- `index.html:1536-1572` — normalization helper functions
- `index.html:1671-1674` — ridge/point distance falloff
- `index.html:2139-2189` — atlas mask decoding and runtime arrays
- `index.html:2190-2267` — `gateSettings()`
- `index.html:2269-2290` — summed-area-table helpers
- `index.html:2300-2349` — `applyGateCentroidWeights()`
- `index.html:2582-2603` — cleanup pipeline
- `index.html:2607-2672` — density and inverse-density computation
- `index.html:2761-2774` — per-tile computation order
- `index.html:3424-3437` — Final Inverse Density rendering
- `index.html:3457-3487` — recompute scheduling and row order
- `build_gate_collection.py:1-50` — imports and builder constants
- `build_gate_collection.py:93-278` — segmentation and component construction
- `build_gate_collection.py:280-349` — tile, density, and inverse preview construction
- `build_gate_collection.py:352-411` — atlas and metadata output
- `build_gate_collection.py:414-479` — builder execution order
- `build_gate_layer_sidecar.py:1-41` — imports and layer encoding
- `build_gate_layer_sidecar.py:53-154` — membership construction and output
- `build_gate_layer_sidecar.py:157-193` — sidecar execution order

## Exact reproduction checklist

To reproduce the same numeric Final Inverse Density outside the UI, capture or confirm all of the following:

1. the Python-generated 120 x 120 source mask tile;
2. all enabled cleanup stages and their values;
3. radius = 12;
4. mode = `mean`;
5. the exact relative-cap value (`2.00` only if unchanged);
6. inverse gamma = 1.65;
7. whether inverse layer confidence is enabled, plus the three confidence values if it is;
8. whether ridge centroid is enabled;
9. ridge radius = 10 and ridge gamma = 4.15;
10. whether point centroid is enabled and, if so, its radius and gamma.

Density gamma = 1.00 should also be recorded as part of the overall profile, but it does not enter the Final Inverse Density branch directly.

# Camera and Gate Geometry

## Gate boundaries

- Outer boundary: 2700 mm wide × 2700 mm high.
- Gate depth: 260 mm.
- Inner aperture boundary: 1500 mm wide × 1500 mm high.
- Gate-face width from the outer wall to the inner wall: 600 mm on each side.

## PnP quadrilateral

Fit the PnP quadrilateral to the centerline of the gate face between its outer and inner boundaries. This square is approximately 2100 mm × 2100 mm:

```text
(2700 mm + 1500 mm) / 2 = 2100 mm
```

This 2100 mm centerline square is distinct from the physical 1500 mm × 1500 mm inner aperture.

## Camera intrinsics

- Image resolution: 640 px × 360 px.
- Principal point `[cx, cy]`: `[320 px, 180 px]`.
- Focal lengths `[fx, fy]`: `[320 px, 320 px]`.
- Lens distortion: none.
- Preserve the incoming image orientation. Do not mirror, rotate, flip, or otherwise transform it before applying these intrinsics.

```text
K = [[320,   0, 320],
     [  0, 320, 180],
     [  0,   0,   1]]
```

Use zero distortion coefficients for PnP.

## Included mask asset

`color_lut_v1.npz` is an unchanged copy of the deterministic-v3 color lookup table used to construct the LUT mask.

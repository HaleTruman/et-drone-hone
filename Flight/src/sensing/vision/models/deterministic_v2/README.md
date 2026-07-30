# Vision Runtime

`vision/` is the minimal production lane for the gate projection pipeline.
It is intentionally independent from the prior experimental workbench.

The runtime decodes each source frame once, runs the promoted production
profiles `A`, `B`, and `C` independently, then performs one full-run global gate
mapping pass before writing final frame JSON. The final public payload is
`vision_results.gates`, a list of `VisionGateResult` records.

## Run

```bash
python vision/main.py run --source-dir <jpeg_dir> --output-root <output_dir> --debug
python vision/main.py run --single-frame <jpg> --output-root <output_dir>
```

## Contract

External systems should import `VisionGateResult` and `VisionResults` from
`vision/src/schema.py`. The production pipeline writes one compact JSON
frame result per source image, plus `latest.json`, `status.json`, and
`run_manifest.json`.

The runtime does not import from the prior workbench modules.

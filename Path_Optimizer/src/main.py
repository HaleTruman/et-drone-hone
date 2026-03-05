from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _ensure_src_on_path() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


def _parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    project_root = here.parent
    default_scenario = project_root / "course_model" / "targets-SimBlank-20260216_194648.json"
    default_out = project_root / "artifacts" / "last_result.json"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default=str(default_scenario),
        help="Path to scenario JSON.",
    )
    parser.add_argument("--v-max", type=float, default=6.0)
    parser.add_argument("--a-fwd-max", type=float, default=2.0)
    parser.add_argument("--a-brake-max", type=float, default=3.0)
    parser.add_argument("--a-lat-max", type=float, default=4.0)
    parser.add_argument("--r-min-m", type=float, default=None, help="Optional hard turning radius limit (meters).")
    parser.add_argument(
        "--kappa-max-1pm",
        type=float,
        default=None,
        help="Optional hard curvature limit (1/m). Specify only one of --r-min-m or --kappa-max-1pm.",
    )
    parser.add_argument("--use-theta-max", action="store_true", help="Derive a_lat_max from theta_max_deg.")
    parser.add_argument("--theta-max-deg", type=float, default=35.0, help="Max tilt angle in degrees (for coupling).")
    parser.add_argument("--heading-constrained", action="store_true", help="Enable yaw-rate speed cap.")
    parser.add_argument("--yaw-rate-max-dps", type=float, default=120.0, help="Yaw-rate max in deg/s (for yaw cap).")
    parser.add_argument("--lambda-min", type=float, default=0.2)
    parser.add_argument("--lambda-max", type=float, default=2.0)
    parser.add_argument("--lambda-steps", type=int, default=25)
    parser.add_argument("--samples-per-segment", type=int, default=50)
    parser.add_argument(
        "--out",
        default=str(default_out),
        help="Output JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    _ensure_src_on_path()

    from opt_engine.optimize import optimize_lambda_grid
    from opt_engine.scenario_io import load_scenario
    from opt_engine.types import Constraints, SamplingConfig

    args = _parse_args()
    scenario = load_scenario(Path(args.scenario))

    constraints = Constraints(
        v_max=args.v_max,
        a_fwd_max=args.a_fwd_max,
        a_brake_max=args.a_brake_max,
        a_lat_max=args.a_lat_max,
        r_min_m=args.r_min_m,
        kappa_max_1pm=args.kappa_max_1pm,
        use_theta_max=bool(args.use_theta_max),
        theta_max_deg=float(args.theta_max_deg),
        heading_constrained=bool(args.heading_constrained),
        yaw_rate_max_rps=float(args.yaw_rate_max_dps) * math.pi / 180.0,
    )
    sampling = SamplingConfig(samples_per_segment=args.samples_per_segment)

    result = optimize_lambda_grid(
        scenario=scenario,
        constraints=constraints,
        sampling=sampling,
        lambda_min=args.lambda_min,
        lambda_max=args.lambda_max,
        lambda_steps=args.lambda_steps,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result.to_json(), indent=2))
    print(f"Wrote {out_path}")
    print(f"best_lambda={result.best_lambda:.6g}  time_s={result.time_s:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

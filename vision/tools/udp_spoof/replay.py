from __future__ import annotations

import argparse
from pathlib import Path

from vision.src.io.udp_protocol import DEFAULT_CHUNK_PAYLOAD_BYTES, DEFAULT_FPS, DEFAULT_HOST, DEFAULT_PORT
from vision.tools.udp_spoof.udp_shim import ShimConfig, send_sample_frames


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay local review/sample frames as UDP vision packets.")
    parser.add_argument("frames_dir", type=Path)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_PAYLOAD_BYTES)
    parser.add_argument("--max-frames", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = send_sample_frames(
        ShimConfig(
            frames_dir=args.frames_dir,
            host=args.host,
            port=args.port,
            fps=args.fps,
            chunk_payload_bytes=args.chunk_size,
            max_frames=args.max_frames,
        )
    )
    print(
        f"sent frames={stats.frames_sent} packets={stats.packets_sent} "
        f"bytes={stats.bytes_sent} udp={args.host}:{args.port}"
    )


if __name__ == "__main__":
    main()

import json
import mimetypes
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


SRC_DIR = Path(__file__).resolve().parent
ROOT = SRC_DIR.parent
UI_DIR = SRC_DIR / "ui"
DATASETS_DIR = ROOT / "datasets"
ARTIFACTS_DIR = ROOT / "artifacts"
VALIDATION_DIR = ROOT / "validation"
HOST = "127.0.0.1"
PORT = int(os.environ.get("ML_VIEWER_PORT", "8765"))


def safe_child(base, *parts):
    path = base.joinpath(*parts).resolve()
    if base.resolve() not in path.parents and path != base.resolve():
        raise ValueError("Path escapes base directory")
    return path


class ViewerHandler(SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"{self.address_string()} - {format % args}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        try:
            if path == "/api/datasets":
                self.send_json(self.list_datasets())
                return

            if path == "/api/evaluations":
                self.send_json(self.list_evaluations())
                return

            if path == "/api/validations":
                self.send_json(self.list_validations())
                return

            if path.startswith("/api/datasets/") and path.endswith("/metadata"):
                _, _, _, dataset_name, _, run_name, _ = path.split("/", 6)
                self.send_json(self.load_metadata(dataset_name, run_name))
                return

            if path.startswith("/api/evaluations/") and path.endswith("/predictions"):
                artifact_name = path.split("/")[3]
                self.send_json(self.load_predictions(artifact_name))
                return

            if path.startswith("/api/evaluations/") and path.endswith("/metrics"):
                artifact_name = path.split("/")[3]
                self.send_json(self.load_metrics(artifact_name))
                return

            if path.startswith("/api/validations/") and path.endswith("/predictions"):
                run_name = path.split("/")[3]
                self.send_json(self.load_validation_predictions(run_name))
                return

            if path.startswith("/api/validations/") and path.endswith("/metrics"):
                run_name = path.split("/")[3]
                self.send_json(self.load_validation_metrics(run_name))
                return

            if path.startswith("/datasets/"):
                self.serve_dataset_file(path)
                return

            if path.startswith("/artifacts/"):
                self.serve_artifact_file(path)
                return

            if path.startswith("/validation/"):
                self.serve_validation_file(path)
                return

            self.serve_ui(path)
        except Exception as exc:
            self.send_error(500, str(exc))

    def list_datasets(self):
        if not DATASETS_DIR.exists():
            return []

        runs = []
        for dataset_dir in sorted(DATASETS_DIR.iterdir(), key=lambda p: p.name):
            if not dataset_dir.is_dir():
                continue
            runs_dir = dataset_dir / "runs"
            if not runs_dir.exists():
                continue
            for run_dir in sorted(runs_dir.iterdir(), key=lambda p: p.name):
                if not run_dir.is_dir():
                    continue
                metadata_path = run_dir / "metadata.json"
                frame_dir = run_dir / "frames"
                frame_count = len(list(frame_dir.glob("*.png"))) if frame_dir.exists() else 0
                runs.append(
                    {
                        "name": f"{dataset_dir.name}/{run_dir.name}",
                        "dataset_name": dataset_dir.name,
                        "run_name": run_dir.name,
                        "has_metadata": metadata_path.exists(),
                        "frame_count": frame_count,
                        "modified": run_dir.stat().st_mtime,
                    }
                )
        return runs

    def list_evaluations(self):
        if not ARTIFACTS_DIR.exists():
            return []

        evaluations = []
        for artifact_dir in sorted(ARTIFACTS_DIR.iterdir(), key=lambda p: p.name):
            if not artifact_dir.is_dir():
                continue
            evaluation_dir = artifact_dir / "evaluation"
            predictions_path = evaluation_dir / "predictions.jsonl"
            metrics_path = evaluation_dir / "metrics.json"
            if not predictions_path.exists():
                continue
            frame_count = 0
            with predictions_path.open("r", encoding="utf-8") as predictions_file:
                for line in predictions_file:
                    if line.strip():
                        frame_count += 1
            evaluations.append(
                {
                    "name": artifact_dir.name,
                    "has_metrics": metrics_path.exists(),
                    "frame_count": frame_count,
                    "modified": predictions_path.stat().st_mtime,
                }
            )
        return evaluations

    def list_validations(self):
        if not VALIDATION_DIR.exists():
            return []

        validations = []
        for run_dir in sorted(VALIDATION_DIR.iterdir(), key=lambda p: p.name):
            if not run_dir.is_dir():
                continue
            predictions_path = run_dir / "predictions.jsonl"
            metrics_path = run_dir / "metrics.json"
            if not predictions_path.exists():
                continue
            frame_count = 0
            with predictions_path.open("r", encoding="utf-8") as predictions_file:
                for line in predictions_file:
                    if line.strip():
                        frame_count += 1
            validations.append(
                {
                    "name": run_dir.name,
                    "has_metrics": metrics_path.exists(),
                    "frame_count": frame_count,
                    "modified": predictions_path.stat().st_mtime,
                }
            )
        return validations

    def load_metadata(self, dataset_name, run_name):
        metadata_path = safe_child(
            DATASETS_DIR,
            dataset_name,
            "runs",
            run_name,
            "metadata.json",
        )
        if not metadata_path.exists():
            return []
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            return json.load(metadata_file)

    def load_predictions(self, artifact_name):
        predictions_path = safe_child(
            ARTIFACTS_DIR,
            artifact_name,
            "evaluation",
            "predictions.jsonl",
        )
        if not predictions_path.exists():
            return []
        predictions = []
        with predictions_path.open("r", encoding="utf-8") as predictions_file:
            for line in predictions_file:
                if line.strip():
                    predictions.append(json.loads(line))
        return predictions

    def load_metrics(self, artifact_name):
        metrics_path = safe_child(
            ARTIFACTS_DIR,
            artifact_name,
            "evaluation",
            "metrics.json",
        )
        if not metrics_path.exists():
            return {}
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)

    def load_validation_predictions(self, run_name):
        predictions_path = safe_child(VALIDATION_DIR, run_name, "predictions.jsonl")
        if not predictions_path.exists():
            return []
        predictions = []
        with predictions_path.open("r", encoding="utf-8") as predictions_file:
            for line in predictions_file:
                if line.strip():
                    predictions.append(json.loads(line))
        return predictions

    def load_validation_metrics(self, run_name):
        metrics_path = safe_child(VALIDATION_DIR, run_name, "metrics.json")
        if not metrics_path.exists():
            return {}
        with metrics_path.open("r", encoding="utf-8") as metrics_file:
            return json.load(metrics_file)

    def serve_dataset_file(self, request_path):
        relative = request_path.removeprefix("/datasets/").split("/")
        file_path = safe_child(DATASETS_DIR, *relative)
        self.serve_file(file_path)

    def serve_artifact_file(self, request_path):
        relative = request_path.removeprefix("/artifacts/").split("/")
        file_path = safe_child(ARTIFACTS_DIR, *relative)
        self.serve_file(file_path)

    def serve_validation_file(self, request_path):
        relative = request_path.removeprefix("/validation/").split("/")
        file_path = safe_child(VALIDATION_DIR, *relative)
        self.serve_file(file_path)

    def serve_ui(self, request_path):
        if request_path in ("", "/"):
            request_path = "/index.html"
        file_path = safe_child(UI_DIR, request_path.lstrip("/"))
        if not file_path.exists() and file_path.suffix == "":
            file_path = UI_DIR / "index.html"
        self.serve_file(file_path)

    def serve_file(self, file_path):
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "File not found")
            return

        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    os.chdir(ROOT)
    try:
        server = ThreadingHTTPServer((HOST, PORT), ViewerHandler)
    except OSError as exc:
        print(f"Could not start viewer on http://{HOST}:{PORT}: {exc}", flush=True)
        print("If another viewer is already running, open that URL or stop the old process.", flush=True)
        sys.exit(1)

    print(f"Viewer running at http://{HOST}:{PORT}", flush=True)
    print("Leave this terminal open while using the viewer. Press Ctrl+C to stop.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped.", flush=True)

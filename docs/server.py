"""Local server for slide 3 of the presentation deck (docs/index.html).

Slide 3 has two modes:
  - static:  the "Example" tabs — canned data from
             docs/assets/js/dashboard-scenarios.js, no backend involved.
  - dynamic: the "Try your own prompt" box — this server's `POST /api/route`,
             which wraps the real `PEQRouter.run()` with every device live
             (build_executors(live_phone=True, live_pc=True, live_cloud=True)).
             A prompt that routes to a device you haven't configured (see
             below) fails with a clear error instead of returning fake text —
             dynamic mode never silently falls back to a simulated response.

dashboard-static/ is untouched by this — this server exists specifically to
back the deck's embedded copy of that dashboard on slide 3.

Stdlib only, matching this project's dependency-free convention (see
pyproject.toml: `dependencies = []`).

Usage:
    python docs/server.py [--port 8090]

Then open http://localhost:8090/#slide-3 — same origin as the API, so no
CORS setup is needed. The default port is 8090, not 8000, on purpose: 8000 is
where the local X-Elite NPU server listens (see "pc" below), and this server
calling itself instead of that server is a real failure mode, not a
theoretical one. Live execution needs, per device (same as `peqrouter run`'s
--live-* flags, see peqrouter/cli.py):
  - phone: PHONE_SERVER_ENDPOINT and PHONE_SERVER_TOKEN env vars.
  - pc:    the local X-Elite NPU server (peqrouter/executors.py's
           XEliteExecutor defaults to http://localhost:8000 if
           XELITE_SERVER_ENDPOINT isn't set).
  - cloud: Cirrascale credentials (see peqrouter/executors.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from demo import calibrated_telemetry  # noqa: E402
from peqrouter.estimator import CalibratedEstimator, EstimatorUnavailableError  # noqa: E402
from peqrouter.executors import build_executors  # noqa: E402
from peqrouter.models import (  # noqa: E402
    Device,
    PEQRouterError,
    OptimizationProfile,
    RouteRequest,
    default_device_configs,
)
from peqrouter.router import PEQRouter  # noqa: E402
from peqrouter.scenarios import built_in_scenarios  # noqa: E402

# Same convention as cli.py/demo.py: resolved from this file's location, not
# the working directory.
HEADS_DIR = REPO_ROOT / "benchmarks" / "calibration" / "heads"
STATIC_DIR = Path(__file__).resolve().parent

MAX_PROMPT_CHARS = 4000
MAX_BODY_BYTES = 16_384


def build_router():
    estimator = None
    try:
        estimator = CalibratedEstimator(HEADS_DIR)
    except EstimatorUnavailableError as error:
        print(f"! routing without the calibrated estimator: {error}", file=sys.stderr)

    # Name the models that actually answered during calibration, same as
    # demo.py, so a live routing result doesn't claim less than it can prove.
    device_configs = dict(default_device_configs())
    if estimator is not None:
        for device, config in device_configs.items():
            observed = estimator.observed_model_id(device)
            if observed:
                device_configs[device] = replace(config, model_id=observed)

    router = PEQRouter(device_configs=device_configs, estimator=estimator)

    telemetry = built_in_scenarios()["healthy"]
    if estimator is not None:
        telemetry = calibrated_telemetry(telemetry, estimator)

    # Dynamic mode means true execution: every device is live. A route that
    # lands on an unconfigured device raises a *ConfigurationError (caught
    # below as an PEQRouterError) instead of quietly returning simulated text.
    executors = build_executors(live_phone=True, live_pc=True, live_cloud=True)
    return router, telemetry, executors


class DeckHandler(SimpleHTTPRequestHandler):
    router: PEQRouter
    telemetry: dict
    executors: dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format, *args):  # noqa: A002 - matches base signature
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def do_POST(self):
        if self.path != "/api/route":
            self.send_error(404, "no such endpoint")
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json_error(400, "missing or oversized request body")
            return
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._json_error(400, "body must be JSON")
            return

        prompt = payload.get("prompt")
        origin = payload.get("origin")
        profile = payload.get("profile", "balanced")
        if not isinstance(prompt, str) or not prompt.strip():
            self._json_error(400, "prompt must be a non-empty string")
            return
        if len(prompt) > MAX_PROMPT_CHARS:
            self._json_error(400, f"prompt must be under {MAX_PROMPT_CHARS} characters")
            return

        try:
            request = RouteRequest(
                prompt=prompt,
                origin=Device(origin),
                telemetry=self.telemetry,
                profile=OptimizationProfile(profile),
            )
            result = self.router.run(request, self.executors)
        except ValueError as error:
            # Device(origin) / OptimizationProfile(profile) raise plain
            # ValueError on a value outside the enum.
            self._json_error(400, str(error))
            return
        except PEQRouterError as error:
            # Domain-level rejection (e.g. origin=cloud, no eligible device) —
            # a well-formed request the router still can't honor.
            self._json_error(422, str(error))
            return

        self._json_response(200, result.to_dict())

    def _json_response(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, status: int, message: str) -> None:
        self._json_response(status, {"error": message})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()

    router, telemetry, executors = build_router()
    DeckHandler.router = router
    DeckHandler.telemetry = telemetry
    DeckHandler.executors = executors

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), DeckHandler)
    print(f"PEQRouter deck: http://127.0.0.1:{args.port}/#slide-3")
    print("Dynamic mode ('Try your own prompt') executes live on whichever device wins routing:")
    phone_ready = bool(os.environ.get("PHONE_SERVER_ENDPOINT")) and bool(os.environ.get("PHONE_SERVER_TOKEN"))
    cloud_ready = bool(os.environ.get("INFERENCE_CLOUD_API_KEY")) and bool(os.environ.get("INFERENCE_CLOUD_ENDPOINT"))
    print(f"  phone : {'configured' if phone_ready else 'NOT configured (PHONE_SERVER_ENDPOINT / PHONE_SERVER_TOKEN)'}")
    print(f"  pc    : defaults to {os.environ.get('XELITE_SERVER_ENDPOINT', 'http://localhost:8000')} — must actually be running")
    print(f"  cloud : {'configured' if cloud_ready else 'NOT configured (Cirrascale credentials)'}")
    print("A prompt routed to an unconfigured/unreachable device returns a clear error, not fake text.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

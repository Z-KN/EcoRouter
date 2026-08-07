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
    python docs/server.py [--port 8090] [--no-debug]

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

Debugging
---------
The UI shows a destination and a wall-clock number and nothing about how
either was arrived at, which makes "why did this go to cloud" and "why did
this take five seconds" unanswerable from the browser. Debug mode (on by
default; `--no-debug` turns it off) answers both:

  1. Is each model alive?  Probed for real, not inferred from env vars: phone
     and PC over their `GET /health` endpoints, cloud over the Cirrascale
     model catalog — at startup, and on demand at `GET /api/debug/health`.
     Each report also compares the model the device is actually serving
     against the model_id routing scored it as.

  2. What did the router score?  Every candidate's penalties, the profile
     weights applied to them, the weighted contribution of each term, and the
     final score — printed as a table to stderr and attached to the response
     as `debug.scores`. The privacy verdict and the calibrated estimator's
     P(pass)/confidence/nearest calibration prompts print alongside, because
     those decide eligibility before any score is compared.

  3. Which endpoint did it go to?  The resolved URL of the selected device and
     the executor class that dispatched to it, plus a route/execute/total
     timing split — the fastest way to tell a slow *router* from a slow
     *model*, which the UI's single latency number cannot.

`POST /api/route` responses carry all of it under a `debug` key. The dashboard
ignores unknown keys, so this is additive; `--no-debug` omits it entirely.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from demo import calibrated_telemetry  # noqa: E402
from peqrouter.estimator import CalibratedEstimator, EstimatorUnavailableError  # noqa: E402
from peqrouter.executors import ObservedExecutor, build_executors, phone_health  # noqa: E402
from peqrouter.models import (  # noqa: E402
    Device,
    PEQRouterError,
    OptimizationProfile,
    RouteRequest,
    default_device_configs,
)
# _PROFILE_WEIGHTS is read here for the debug view only. A candidate's score is
# already weight-normalized, so showing *why* one term dominated needs the same
# weight table the router scored with, not a second copy of it that can drift.
from peqrouter.router import _PROFILE_WEIGHTS, PEQRouter  # noqa: E402
from peqrouter.scenarios import built_in_scenarios  # noqa: E402

# Same convention as cli.py/demo.py: resolved from this file's location, not
# the working directory.
HEADS_DIR = REPO_ROOT / "benchmarks" / "calibration" / "heads"
STATIC_DIR = Path(__file__).resolve().parent

MAX_PROMPT_CHARS = 4000
MAX_BODY_BYTES = 16_384

# Long enough for a phone whose Wi-Fi radio is asleep to answer /health (the
# Android app's InferenceService documents multi-second TTFB from 802.11
# power-save), short enough that three dead devices don't stall startup.
HEALTH_TIMEOUT_SECONDS = 8.0

_DEFAULT_XELITE_ENDPOINT = "http://localhost:8000"


# ---------------------------------------------------------------- liveness --


def resolved_endpoint(device: Device) -> str | None:
    """Where a live executor for ``device`` would actually send this request.

    Mirrors how the executors in peqrouter/executors.py resolve their own
    endpoints. ``None`` means nothing is configured at all, which is a
    different failure from "configured but unreachable" and worth saying so.
    """

    if device is Device.PHONE:
        return (os.environ.get("PHONE_SERVER_ENDPOINT") or "").rstrip("/") or None
    if device is Device.PC:
        return (os.environ.get("XELITE_SERVER_ENDPOINT") or _DEFAULT_XELITE_ENDPOINT).rstrip("/")
    return (os.environ.get("INFERENCE_CLOUD_ENDPOINT") or "").rstrip("/") or None


def _probe_phone() -> tuple[bool, str, dict | None]:
    if resolved_endpoint(Device.PHONE) is None:
        return False, "PHONE_SERVER_ENDPOINT is not set", None
    try:
        payload = phone_health(timeout_seconds=HEALTH_TIMEOUT_SECONDS)
    except PEQRouterError as error:
        return False, str(error), None
    except Exception as error:  # noqa: BLE001 - a probe must never raise
        return False, f"{type(error).__name__}: {error}", None

    status = payload.get("status")
    # /health is unauthenticated but /v1/chat/completions is not, so a healthy
    # phone with no token is not a device this server can actually route to.
    if not os.environ.get("PHONE_SERVER_TOKEN"):
        return False, f"/health says {status!r} but PHONE_SERVER_TOKEN is not set", payload
    if status != "healthy":
        return False, f"/health says {status!r} — no model loaded in the app", payload
    return True, f"serving {payload.get('model')!r}", payload


def _probe_pc() -> tuple[bool, str, dict | None]:
    endpoint = resolved_endpoint(Device.PC)
    try:
        with urllib.request.urlopen(f"{endpoint}/health", timeout=HEALTH_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        # serve_qwen_vl.py answers 503 {"status": "loading"} while weights load
        # — reachable but not yet usable, which is worth reporting distinctly.
        try:
            payload = json.loads(error.read())
        except Exception:  # noqa: BLE001
            payload = None
        status = (payload or {}).get("status", f"HTTP {error.code}")
        return False, f"{endpoint}/health says {status!r}", payload
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return False, f"could not reach {endpoint} ({error})", None
    except Exception as error:  # noqa: BLE001
        return False, f"{type(error).__name__}: {error}", None

    status = payload.get("status") if isinstance(payload, dict) else None
    if status != "healthy":
        return False, f"{endpoint}/health says {status!r}", payload
    return True, f"serving {payload.get('model')!r}", payload


def _probe_cloud(cloud_executor, configured_model_id: str) -> tuple[bool, str, dict | None]:
    if resolved_endpoint(Device.CLOUD) is None:
        return False, "INFERENCE_CLOUD_ENDPOINT is not set", None
    if not os.environ.get("INFERENCE_CLOUD_API_KEY"):
        return False, "INFERENCE_CLOUD_API_KEY is not set", None
    if not hasattr(cloud_executor, "list_models"):
        return False, f"cloud executor is {type(cloud_executor).__name__}, not live", None
    # CirrascaleExecutor caches the catalog for the process lifetime, which is
    # right for routing and wrong for a liveness probe -- a cached hit answers
    # in 0.0ms and would report a cloud that has since gone away as alive.
    cloud_executor._available_models = None  # noqa: SLF001
    try:
        models = cloud_executor.list_models()
    except PEQRouterError as error:
        return False, str(error), None
    except Exception as error:  # noqa: BLE001
        return False, f"{type(error).__name__}: {error}", None

    if configured_model_id not in models:
        return (
            False,
            f"{configured_model_id!r} is not in the catalog of {len(models)} LLMs",
            {"models": list(models)},
        )
    return (
        True,
        f"{configured_model_id!r} in a catalog of {len(models)} LLMs",
        {"model": configured_model_id, "models": list(models)},
    )


def probe_devices(router: PEQRouter, executors: dict) -> dict:
    """Liveness for all three devices, probed concurrently.

    Answers debug question (1). Deliberately a real network call per device
    rather than an env-var check: "credentials are set" and "the model will
    answer" are different claims, and only the second predicts whether a route
    to that device succeeds.
    """

    def one(device: Device):
        configured = router.device_configs[device].model_id
        started = time.perf_counter()
        if device is Device.PHONE:
            alive, detail, payload = _probe_phone()
        elif device is Device.PC:
            alive, detail, payload = _probe_pc()
        else:
            alive, detail, payload = _probe_cloud(executors.get(Device.CLOUD), configured)
        elapsed_ms = (time.perf_counter() - started) * 1000

        served = (payload or {}).get("model")
        return device, {
            "device": device.value,
            "alive": alive,
            "endpoint": resolved_endpoint(device),
            "configured_model_id": configured,
            "served_model_id": served if isinstance(served, str) else None,
            # A mismatch means routing scored one model while a different one
            # would answer — the measured latency/energy constants behind that
            # score belong to the model that isn't running.
            "model_matches": None if not isinstance(served, str) else served == configured,
            "probe_latency_ms": round(elapsed_ms, 1),
            "detail": detail,
        }

    with ThreadPoolExecutor(max_workers=len(Device)) as pool:
        reports = dict(pool.map(one, list(Device)))
    return {device.value: reports[device] for device in Device}


def print_health(health: dict) -> None:
    for device in Device:
        report = health[device.value]
        mark = "OK  " if report["alive"] else "DOWN"
        print(f"  [{mark}] {device.value:<5} {report['endpoint'] or '(no endpoint configured)'}")
        print(
            f"         model={report['configured_model_id']}  {report['detail']}  "
            f"({report['probe_latency_ms']}ms)"
        )
        if report["model_matches"] is False:
            print(
                f"         ! serving {report['served_model_id']!r} but routing scores "
                f"{report['configured_model_id']!r} — the latency/energy constants "
                "belong to the other model"
            )


# ------------------------------------------------------- request recording --


class RecordingEstimator:
    """Pass-through around CalibratedEstimator that remembers its last estimate.

    ``PEQRouter.route()`` does not surface the ``PromptEstimate`` it routed on.
    Re-running ``estimate()`` afterwards for the debug view would report a
    second, independently computed number rather than the one that actually
    decided the route, so the estimate is captured on the way through instead.
    Thread-local, because this backs a ThreadingHTTPServer.
    """

    def __init__(self, inner: CalibratedEstimator) -> None:
        self._inner = inner
        self._local = threading.local()

    def estimate(self, prompt: str, intent: str | None = None):
        estimate = self._inner.estimate(prompt, intent=intent)
        self._local.estimate = estimate
        return estimate

    def take(self):
        estimate = getattr(self._local, "estimate", None)
        self._local.estimate = None
        return estimate

    def __getattr__(self, name):
        # object.__getattribute__ rather than self._inner: were _inner ever
        # missing, plain attribute access here would recurse forever.
        return getattr(object.__getattribute__(self, "_inner"), name)


class ExecutionClock:
    """Records how long the executor call itself took, per request thread."""

    def __init__(self) -> None:
        self._local = threading.local()

    def record(self, elapsed_ms: float) -> None:
        self._local.elapsed_ms = elapsed_ms

    def take(self) -> float | None:
        elapsed_ms = getattr(self._local, "elapsed_ms", None)
        self._local.elapsed_ms = None
        return elapsed_ms


class _TimedExecutor:
    """Wraps an executor to time its call without changing what it returns.

    Splitting route time from execute time is debug question (3)'s real
    payload: the UI reports one wall-clock number, inside which a slow router
    and a slow model look identical.
    """

    def __init__(self, inner, clock: ExecutionClock) -> None:
        self._inner = inner
        self._clock = clock

    @property
    def inner(self):
        return self._inner

    def execute(self, prompt: str, decision) -> str:
        started = time.perf_counter()
        try:
            return self._inner.execute(prompt, decision)
        finally:
            self._clock.record((time.perf_counter() - started) * 1000)


class _TimedObservedExecutor(_TimedExecutor):
    def execute_observed(self, prompt: str, decision):
        started = time.perf_counter()
        try:
            return self._inner.execute_observed(prompt, decision)
        finally:
            self._clock.record((time.perf_counter() - started) * 1000)


def timed_executors(executors: dict, clock: ExecutionClock) -> dict:
    """Wrap each executor, preserving whether it satisfies ObservedExecutor.

    ``PEQRouter.run()`` branches on ``isinstance(executor, ObservedExecutor)``
    to decide whether live metrics are collected, so a wrapper that always
    exposed ``execute_observed`` would silently promote SimulatedExecutor, and
    one that never did would silently drop every live measurement.
    """

    wrapped = {}
    for device, executor in executors.items():
        cls = _TimedObservedExecutor if isinstance(executor, ObservedExecutor) else _TimedExecutor
        wrapped[device] = cls(executor, clock)
    return wrapped


# ------------------------------------------------------------- debug views --


def _score_rows(decision) -> list[dict]:
    weights = _PROFILE_WEIGHTS[decision.profile]
    rows = []
    for item in decision.candidates:
        # The router divides by the applicable weight, so reproduce that here
        # rather than showing raw weight x penalty products, which would not
        # sum to the score the router actually compared.
        applicable = sum(weights[name] for name in item.penalties) or 1.0
        rows.append(
            {
                "device": item.device.value,
                "model_id": item.model_id,
                "eligible": item.eligible,
                "exclusion_reasons": list(item.exclusion_reasons),
                "quality_sufficient": item.quality_sufficient,
                "predicted_latency_ms": item.predicted_latency_ms,
                "predicted_energy_joules": item.predicted_energy_joules,
                "penalties": {name: round(value, 6) for name, value in item.penalties.items()},
                "weighted_contributions": {
                    name: round(weights[name] * value / applicable, 6)
                    for name, value in item.penalties.items()
                },
                "score": item.score,
                "selected": item.device == decision.selected_device,
            }
        )
    rows.sort(
        key=lambda row: (row["score"] is None, row["score"] if row["score"] is not None else 0.0)
    )
    return rows


def _estimator_debug(estimate, quality_floor: float | None) -> dict:
    if estimate is None:
        return {
            "available": False,
            "note": "no calibrated estimate; quality gated on the static capability_score",
        }
    return {
        "available": True,
        "confidence": estimate.confidence,
        "trusted": estimate.trusted,
        "quality_floor": quality_floor,
        "mean_distance": round(estimate.mean_distance, 4),
        # None is not zero: it means no labelled neighbour voted for that
        # device, so the router fell back to the static capability rule.
        "p_pass": {
            device.value: (None if value is None else round(value, 4))
            for device, value in estimate.p_pass.items()
        },
        "length_p50": {device.value: value for device, value in estimate.length_p50.items()},
        "length_p90": {device.value: value for device, value in estimate.length_p90.items()},
        "nearest": [
            {"id": prompt_id, "similarity": round(similarity, 4), "prompt": text[:120]}
            for prompt_id, text, similarity in estimate.neighbours[:5]
        ],
    }


def build_route_debug(
    *,
    request: RouteRequest,
    decision,
    estimate,
    quality_floor: float | None,
    executors: dict,
    route_ms: float,
    execute_ms: float | None,
    total_ms: float,
    health: dict | None,
    error: str | None = None,
) -> dict:
    debug: dict = {
        "timing_ms": {
            "route": round(route_ms, 2),
            "execute": None if execute_ms is None else round(execute_ms, 2),
            "total": round(total_ms, 2),
        },
        "request": {
            "origin": request.origin.value,
            "profile": request.profile.value,
            "prompt_chars": len(request.prompt),
        },
        "estimator": _estimator_debug(estimate, quality_floor),
        "health": health,
    }
    if error is not None:
        debug["error"] = error
    if decision is None:
        return debug

    analysis = decision.analysis
    selected = executors.get(decision.selected_device)
    debug.update(
        {
            "privacy": {
                "sensitive": analysis.sensitive,
                "pii_categories": list(analysis.pii_categories),
                # The one hard constraint that keeps a prompt off the cloud
                # (router.py's _evaluate). When this is False, no score was
                # ever consulted about privacy.
                "cloud_blocked": analysis.sensitive,
            },
            "analysis": analysis.to_dict(),
            "profile_weights": dict(_PROFILE_WEIGHTS[decision.profile]),
            "scores": _score_rows(decision),
            "selected": {
                "device": decision.selected_device.value,
                "model_id": decision.model_id,
                "endpoint": resolved_endpoint(decision.selected_device),
                "executor": type(getattr(selected, "inner", selected)).__name__,
                "quality_degraded": decision.quality_degraded,
                "explanation": decision.explanation,
            },
        }
    )
    return debug


def print_route_debug(debug: dict, prompt: str) -> None:
    """Print the debug block to stderr as a table. Answers (1)-(3) at a glance."""

    out = sys.stderr
    write = out.write
    condensed = " ".join(prompt.split())
    if len(condensed) > 90:
        condensed = condensed[:87] + "..."
    request = debug["request"]

    write("\n" + "-" * 108 + "\n")
    write(f"[route] prompt    : {condensed!r}\n")
    write(
        f"[route] request   : origin={request['origin']} profile={request['profile']} "
        f"chars={request['prompt_chars']}\n"
    )

    if "privacy" in debug:
        privacy = debug["privacy"]
        categories = ", ".join(privacy["pii_categories"]) or "none"
        verdict = "cloud BLOCKED by privacy policy" if privacy["cloud_blocked"] else "cloud allowed"
        write(f"[route] privacy   : sensitive={privacy['sensitive']} pii=[{categories}] -> {verdict}\n")

    if "analysis" in debug:
        analysis = debug["analysis"]
        write(
            f"[route] analysis  : intent={analysis['intent']} complexity={analysis['complexity']} "
            f"required_quality={analysis['required_quality']} "
            f"tokens in={analysis['estimated_input_tokens']} out={analysis['estimated_output_tokens']}\n"
        )

    estimator = debug["estimator"]
    if estimator["available"]:
        write(
            f"[route] estimator : confidence={estimator['confidence']} trusted={estimator['trusted']} "
            f"floor={estimator['quality_floor']} mean_distance={estimator['mean_distance']}\n"
        )
        p_pass = "  ".join(f"{name}={value}" for name, value in estimator["p_pass"].items())
        write(f"[route]   p_pass  : {p_pass}   (None = no labelled neighbour; static capability used)\n")
        nearest = " | ".join(
            f"{item['id']} ({item['similarity']:.2f})" for item in estimator["nearest"][:3]
        )
        write(f"[route]   nearest : {nearest}\n")
    else:
        write(f"[route] estimator : unavailable — {estimator['note']}\n")

    if "scores" in debug:
        weight_text = " ".join(f"{name}={value:g}" for name, value in debug["profile_weights"].items())
        write(f"[route] scores    : lower is better; weights {weight_text}\n")
        write(
            f"[route]     {'device':<6} {'model':<34} {'elig':<5} {'qual':<5} "
            f"{'lat_ms':>9} {'joules':>8}  {'weighted lat/energy/qual':<26} {'score':>9}\n"
        )
        for row in debug["scores"]:
            marker = "->" if row["selected"] else "  "
            model = row["model_id"] if len(row["model_id"]) <= 34 else row["model_id"][:31] + "..."
            latency = "-" if row["predicted_latency_ms"] is None else f"{row['predicted_latency_ms']:.1f}"
            energy = "-" if row["predicted_energy_joules"] is None else f"{row['predicted_energy_joules']:.1f}"
            contributions = row["weighted_contributions"]
            breakdown = "/".join(
                f"{contributions.get(name, 0.0):.4f}" for name in ("latency", "energy", "quality")
            )
            score = "-" if row["score"] is None else f"{row['score']:.6f}"
            write(
                f"[route]   {marker}{row['device']:<6} {model:<34} "
                f"{'yes' if row['eligible'] else 'NO':<5} {'yes' if row['quality_sufficient'] else 'NO':<5} "
                f"{latency:>9} {energy:>8}  {breakdown:<26} {score:>9}\n"
            )
            if row["exclusion_reasons"]:
                write(f"[route]     {'':<6} excluded: {', '.join(row['exclusion_reasons'])}\n")

    if "selected" in debug:
        selected = debug["selected"]
        write(
            f"[route] endpoint  : {selected['device']} -> "
            f"{selected['endpoint'] or '(none configured)'} via {selected['executor']}\n"
        )
        write(f"[route] reason    : {selected['explanation']}\n")
        if selected["quality_degraded"]:
            write("[route]           ! quality_degraded — no eligible device met the requested quality\n")

    health = debug.get("health")
    if health:
        summary = "  ".join(
            f"{name}={'alive' if report['alive'] else 'DOWN'}" for name, report in health.items()
        )
        write(f"[route] liveness  : {summary}   (last probe; refresh at /api/debug/health)\n")

    timing = debug["timing_ms"]
    execute = "-" if timing["execute"] is None else f"{timing['execute']}ms"
    write(f"[route] timing    : route={timing['route']}ms execute={execute} total={timing['total']}ms\n")
    if "error" in debug:
        write(f"[route] FAILED    : {debug['error']}\n")
    write("-" * 108 + "\n")
    out.flush()


# ------------------------------------------------------------------ server --


def build_router(debug: bool = True):
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

    telemetry = built_in_scenarios()["healthy"]
    if estimator is not None:
        telemetry = calibrated_telemetry(telemetry, estimator)
        if debug:
            estimator = RecordingEstimator(estimator)

    router = PEQRouter(device_configs=device_configs, estimator=estimator)

    # Dynamic mode means true execution: every device is live. A route that
    # lands on an unconfigured device raises a *ConfigurationError (caught
    # below as an PEQRouterError) instead of quietly returning simulated text.
    executors = build_executors(live_phone=True, live_pc=True, live_cloud=True)
    return router, telemetry, executors


class DeckHandler(SimpleHTTPRequestHandler):
    router: PEQRouter
    telemetry: dict
    executors: dict
    raw_executors: dict
    clock: ExecutionClock
    debug: bool
    health: dict

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def log_message(self, format, *args):  # noqa: A002 - matches base signature
        sys.stderr.write(f"{self.address_string()} - {format % args}\n")

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/debug/health":
            # Re-probed per call: "is it alive" is only ever true as of now,
            # and a cached answer is the thing that made this hard to debug.
            health = probe_devices(self.router, self.raw_executors)
            type(self).health = health
            self._json_response(200, {"devices": health})
            return
        if path == "/api/debug/config":
            self._json_response(200, self._config_payload())
            return
        super().do_GET()

    def _config_payload(self) -> dict:
        return {
            "debug": self.debug,
            "profile_weights": {
                profile.value: dict(weights) for profile, weights in _PROFILE_WEIGHTS.items()
            },
            "devices": {
                device.value: {
                    "model_id": self.router.device_configs[device].model_id,
                    "capability_score": self.router.device_configs[device].capability_score,
                    "endpoint": resolved_endpoint(device),
                    "executor": type(self.raw_executors[device]).__name__,
                    "telemetry": {
                        "available": self.telemetry[device].available,
                        "network_latency_ms": self.telemetry[device].network_latency_ms,
                        "throughput_tokens_per_second": round(
                            self.telemetry[device].throughput_tokens_per_second, 3
                        ),
                        "energy_joules_per_token": round(
                            self.telemetry[device].energy_joules_per_token, 6
                        ),
                    },
                }
                for device in Device
            },
            "estimator": {
                "available": self.router.estimator is not None,
                "quality_floor": getattr(self.router.estimator, "quality_floor", None),
            },
        }

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
        except ValueError as error:
            # Device(origin) / OptimizationProfile(profile) raise plain
            # ValueError on a value outside the enum.
            self._json_error(400, str(error))
            return

        self.clock.take()  # discard any stale reading left on this thread
        started = time.perf_counter()
        try:
            result = self.router.run(request, self.executors)
        except ValueError as error:
            self._json_error(400, str(error))
            return
        except PEQRouterError as error:
            # Domain-level rejection (e.g. no eligible device) or a live device
            # that would not answer — a well-formed request the router still
            # can't honor. Worth a debug block: this is where "which endpoint"
            # and "is it alive" matter most.
            total_ms = (time.perf_counter() - started) * 1000
            self._emit_debug(request, None, total_ms, error=str(error))
            self._json_error(422, str(error))
            return

        total_ms = (time.perf_counter() - started) * 1000
        body = result.to_dict()
        debug = self._emit_debug(request, result.decision, total_ms)
        if debug is not None:
            body["debug"] = debug
        self._json_response(200, body)

    def _emit_debug(self, request, decision, total_ms, error=None):
        """Assemble, print, and return the debug block (None when disabled)."""

        if not self.debug:
            return None
        execute_ms = self.clock.take()
        route_ms = total_ms - execute_ms if execute_ms is not None else total_ms
        estimate = None
        if hasattr(self.router.estimator, "take"):
            estimate = self.router.estimator.take()
        if decision is None and error is not None:
            # Routing is deterministic given this telemetry, so re-running it
            # reconstructs exactly the decision that just failed to execute —
            # the one worth seeing when a live device won't answer.
            try:
                decision = self.router.route(request)
                if hasattr(self.router.estimator, "take"):
                    estimate = self.router.estimator.take() or estimate
            except PEQRouterError:
                decision = None
        try:
            debug = build_route_debug(
                request=request,
                decision=decision,
                estimate=estimate,
                quality_floor=getattr(self.router.estimator, "quality_floor", None),
                executors=self.executors,
                route_ms=route_ms,
                execute_ms=execute_ms,
                total_ms=total_ms,
                health=self.health,
                error=error,
            )
            print_route_debug(debug, request.prompt)
        except Exception as report_error:  # noqa: BLE001 - debug must never break routing
            sys.stderr.write(f"[route] debug reporting failed: {report_error}\n")
            return None
        return debug

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
    parser.add_argument(
        "--no-debug",
        dest="debug",
        action="store_false",
        help="omit the startup probe, the per-request stderr table, and the response's debug key",
    )
    parser.set_defaults(debug=True)
    args = parser.parse_args()

    router, telemetry, executors = build_router(debug=args.debug)
    clock = ExecutionClock()
    DeckHandler.router = router
    DeckHandler.telemetry = telemetry
    DeckHandler.raw_executors = executors
    DeckHandler.executors = timed_executors(executors, clock) if args.debug else executors
    DeckHandler.clock = clock
    DeckHandler.debug = args.debug
    DeckHandler.health = {}

    print(f"PEQRouter deck: http://127.0.0.1:{args.port}/#slide-3")
    print("Dynamic mode ('Try your own prompt') executes live on whichever device wins routing.")
    if args.debug:
        print("Probing each device — a real request per model, not an env-var check:")
        DeckHandler.health = probe_devices(router, executors)
        print_health(DeckHandler.health)
        print("Debug on: per-request score tables go to stderr and ride along in the JSON `debug` key.")
        print(f"  liveness : http://127.0.0.1:{args.port}/api/debug/health")
        print(f"  config   : http://127.0.0.1:{args.port}/api/debug/config")
    else:
        print("Debug off (--no-debug): no startup probe, no score tables, no `debug` key.")
    print("A prompt routed to an unconfigured/unreachable device returns a clear error, not fake text.")

    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), DeckHandler)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

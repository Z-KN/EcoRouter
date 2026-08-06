"""Command-line interface for EcoRouter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .analyzer import HeuristicPromptAnalyzer
from .executors import (
    CirrascaleExecutor,
    build_executors,
    phone_health,
)
from .models import (
    Device,
    ExecutionError,
    ExecutionMetrics,
    NoRouteError,
    OptimizationProfile,
    PrivacyError,
    RouteDecision,
    RouteRequest,
    ValidationError,
)
from .router import EcoRouter
from .scenarios import built_in_scenarios, load_device_configs, load_telemetry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ecorouter",
        description="Route a text prompt to a predeployed phone, PC, or cloud model.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("route", "Select a destination without invoking it."),
        ("run", "Select a destination and invoke its configured executor."),
    ):
        subparser = subparsers.add_parser(command, help=help_text)
        prompt_group = subparser.add_mutually_exclusive_group()
        prompt_group.add_argument("--prompt", help="Prompt text. Prefer stdin or --prompt-file for PII.")
        prompt_group.add_argument("--prompt-file", type=Path, help="UTF-8 file containing the prompt.")
        subparser.add_argument("--origin", choices=("phone", "pc"), required=True)
        telemetry_group = subparser.add_mutually_exclusive_group()
        telemetry_group.add_argument("--telemetry", type=Path, help="Telemetry JSON snapshot.")
        telemetry_group.add_argument(
            "--scenario",
            choices=tuple(built_in_scenarios()),
            default="healthy",
            help="Built-in telemetry scenario (default: healthy).",
        )
        subparser.add_argument("--config", type=Path, help="Optional model ID/capability JSON.")
        subparser.add_argument(
            "--profile",
            choices=tuple(profile.value for profile in OptimizationProfile),
            default=OptimizationProfile.BALANCED.value,
        )
        subparser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
        subparser.add_argument(
            "--no-presidio",
            action="store_true",
            help=(
                "Use the dependency-free regex-based HeuristicPromptAnalyzer instead of "
                "Presidio/spaCy. Detects fewer PII categories (no NER-based PERSON/NRP "
                "detection) -- for environments where Presidio cannot be installed, not "
                "for production use."
            ),
        )
        if command == "run":
            subparser.add_argument(
                "--live-cloud",
                action="store_true",
                help="Invoke Cirrascale when cloud is selected; phone and PC remain simulated.",
            )
            subparser.add_argument(
                "--live-pc",
                action="store_true",
                help=(
                    "Invoke the local X-Elite NPU server (XELITE_SERVER_ENDPOINT, default "
                    "http://localhost:8000) when PC is selected; phone and cloud remain as configured."
                ),
            )
            subparser.add_argument(
                "--live-phone",
                action="store_true",
                help=(
                    "Invoke the phone's in-app inference server (PHONE_SERVER_ENDPOINT, "
                    "PHONE_SERVER_TOKEN) when phone is selected; PC and cloud remain as configured."
                ),
            )
    cloud_models = subparsers.add_parser(
        "cloud-models", help="List LLMs available from the configured Cirrascale account."
    )
    cloud_models.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    phone_health_parser = subparsers.add_parser(
        "phone-health", help="Check the phone's in-app inference server (PHONE_SERVER_ENDPOINT)."
    )
    phone_health_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return parser


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return args.prompt_file.read_text(encoding="utf-8")
    return sys.stdin.read()


def _human_decision(decision: RouteDecision) -> str:
    analysis = decision.analysis
    categories = ", ".join(analysis.pii_categories) if analysis.pii_categories else "none"
    lines = [
        f"Selected: {decision.selected_device.value} / {decision.model_id}",
        f"Profile: {decision.profile.value}",
        (
            f"Prompt analysis: intent={analysis.intent.value}, complexity={analysis.complexity:.2f}, "
            f"sensitive={str(analysis.sensitive).lower()}, PII categories={categories}"
        ),
        f"Quality degraded: {str(decision.quality_degraded).lower()}",
        f"Why: {decision.explanation}",
        "Candidates:",
    ]
    for candidate in decision.candidates:
        if candidate.eligible:
            lines.append(
                f"  - {candidate.device.value}/{candidate.model_id}: score={candidate.score:.4f}, "
                f"quality_ok={str(candidate.quality_sufficient).lower()}, "
                f"latency={candidate.predicted_latency_ms:.1f}ms, "
                f"energy={candidate.predicted_energy_joules:.3f}J, "
                f"cost=${candidate.predicted_cloud_cost_usd:.6f}"
            )
        else:
            lines.append(
                f"  - {candidate.device.value}/{candidate.model_id}: excluded "
                f"({'; '.join(candidate.exclusion_reasons)})"
            )
    return "\n".join(lines)


def _human_metrics(metrics: ExecutionMetrics) -> str:
    latency = (
        f"{metrics.api_turnaround_latency_ms:.3f} ms"
        if metrics.api_turnaround_latency_ms is not None
        else "unavailable"
    )
    estimated_energy = (
        f"{metrics.estimated_energy_joules:.6f} J"
        if metrics.estimated_energy_joules is not None
        else "unavailable"
    )
    measured_energy = (
        f"{metrics.measured_energy_joules:.6f} J" if metrics.measured_energy_joules is not None else "unavailable"
    )
    lines = [
        "Live observations:",
        f"  API turnaround latency: {latency}",
        (
            "  SDK tokens: "
            f"prompt={metrics.prompt_tokens}, completion={metrics.completion_tokens}, "
            f"total={metrics.total_tokens}"
        ),
        f"  Measured energy: {measured_energy}",
        (
            f"  Estimated energy: {estimated_energy} "
            f"({metrics.energy_joules_per_token:.6f} J/token; {metrics.confidence})"
        ),
        f"  Estimate scope: {metrics.energy_scope}",
    ]
    if metrics.ttft_ms is not None:
        lines.append(f"  TTFT: {metrics.ttft_ms:.2f} ms")
    if metrics.prefill_speed_tokens_per_second is not None:
        lines.append(f"  Prefill speed: {metrics.prefill_speed_tokens_per_second:.2f} tok/s")
    if metrics.decode_speed_tokens_per_second is not None:
        lines.append(f"  Decode speed: {metrics.decode_speed_tokens_per_second:.2f} tok/s")
    if metrics.tokens_per_joule is not None:
        lines.append(f"  Efficiency: {metrics.tokens_per_joule:.2f} tok/J")
    if metrics.compute_unit is not None:
        lines.append(f"  Compute unit: {metrics.compute_unit}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "cloud-models":
            models = CirrascaleExecutor().list_models()
            if args.json:
                payload = {"count": len(models), "models": list(models)}
                print(json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(f"Available Cirrascale LLMs: {len(models)}")
                for model in models:
                    print(f"  - {model}")
            return 0

        if args.command == "phone-health":
            health = phone_health()
            if args.json:
                print(json.dumps(health, indent=2, sort_keys=True))
            else:
                print(f"Phone server status: {health.get('status')}")
                print(f"  Model: {health.get('model')}")
                print(f"  Uptime: {health.get('uptime_s')}s")
                print(f"  Requests served: {health.get('requests_served')}")
            return 0

        prompt = _read_prompt(args)
        scenarios = built_in_scenarios()
        telemetry = load_telemetry(args.telemetry) if args.telemetry else scenarios[args.scenario]
        configs = load_device_configs(args.config) if args.config else None
        request = RouteRequest(
            prompt=prompt,
            origin=Device(args.origin),
            telemetry=telemetry,
            profile=OptimizationProfile(args.profile),
        )
        analyzer = HeuristicPromptAnalyzer() if args.no_presidio else None
        router = EcoRouter(configs, analyzer=analyzer)
        if args.command == "run":
            executors = build_executors(
                live_phone=args.live_phone,
                live_pc=args.live_pc,
                live_cloud=args.live_cloud,
            )
            result = router.run(request, executors)
            if args.json:
                print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            else:
                print(_human_decision(result.decision))
                print("Response: " + result.response)
                if result.metrics is not None:
                    print(_human_metrics(result.metrics))
        else:
            decision = router.route(request)
            if args.json:
                print(json.dumps(decision.to_dict(), indent=2, sort_keys=True))
            else:
                print(_human_decision(decision))
        return 0
    except NoRouteError as error:
        print(f"routing error: {error}", file=sys.stderr)
        return 3
    except ExecutionError as error:
        print(f"execution error: {error}", file=sys.stderr)
        return 4
    except PrivacyError as error:
        print(f"privacy error: {error}", file=sys.stderr)
        return 5
    except (ValidationError, OSError, json.JSONDecodeError, ValueError) as error:
        print(f"input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

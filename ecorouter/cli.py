"""Command-line interface for EcoRouter."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .executors import default_simulated_executors
from .models import (
    Device,
    EcoRouterError,
    ExecutionError,
    NoRouteError,
    OptimizationProfile,
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
        ("run", "Select a destination and invoke its simulated executor."),
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
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
        router = EcoRouter(configs)
        if args.command == "run":
            result = router.run(request, default_simulated_executors())
            if args.json:
                print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            else:
                print(_human_decision(result.decision))
                print("Response: " + result.response)
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
    except (ValidationError, OSError, json.JSONDecodeError, ValueError) as error:
        print(f"input error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Render baseline-comparison figures for the blind one-device baselines.

This script answers a very specific question: if every prompt is blindly
routed to exactly one destination, how do phone/PC/cloud compare on the
existing calibration sweep data?

It intentionally avoids third-party plotting dependencies so the figures can
be regenerated in a clean checkout without extra installation steps. The
output is SVG, which is publication-friendly and easy to diff.

Default inputs:
    benchmarks/calibration/runs/sweep.jsonl
    benchmarks/calibration/runs/sweep_phone.jsonl
    benchmarks/calibration/runs/sweep_pc_only.jsonl
    benchmarks/calibration/runs/sweep_cloud_llama70b.jsonl

Default outputs:
    benchmarks/calibration/figures/blind_baseline_overall.svg
    benchmarks/calibration/figures/blind_baseline_by_category.svg
    benchmarks/calibration/figures/blind_baseline_summary.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Iterable


DEFAULT_INPUTS = [
    Path(__file__).parent / "runs" / "sweep.jsonl",
    Path(__file__).parent / "runs" / "sweep_phone.jsonl",
    Path(__file__).parent / "runs" / "sweep_pc_only.jsonl",
    Path(__file__).parent / "runs" / "sweep_cloud_llama70b.jsonl",
]
DEFAULT_OUT_DIR = Path(__file__).parent / "figures"
DEVICES = ("phone", "pc", "cloud")
DEVICE_LABELS = {"phone": "Phone", "pc": "PC", "cloud": "Cloud"}
DEVICE_COLORS = {"phone": "#1f6feb", "pc": "#2da44e", "cloud": "#f59e0b"}
PALETTE = ["#1f6feb", "#2da44e", "#f59e0b", "#d29922", "#8b949e"]
CLOUD_ASSUMED_POWER_W = 75.0
CANVAS_BG = "#ffffff"
PANEL_BG = "#ffffff"
PANEL_BORDER = "#dbe3ee"
GRID_COLOR = "#e8eef5"
TEXT_DARK = "#0f172a"


@dataclass(frozen=True)
class DeviceSummary:
    label: str
    total: int
    passed: int
    accuracy: float | None
    median_wall_ms: float | None
    median_energy_j: float | None
    median_power_w: float | None
    median_completion_tokens: float | None


def _read_rows(paths: Iterable[Path]) -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("error") is not None:
                continue
            prompt_id = str(row.get("id"))
            device = str(row.get("device"))
            rows[(prompt_id, device)] = row
    return rows


def _summarize(rows: dict[tuple[str, str], dict]) -> dict[str, DeviceSummary]:
    summaries: dict[str, DeviceSummary] = {}
    for device in DEVICES:
        device_rows = [row for row in rows.values() if row.get("device") == device]
        graded = [row for row in device_rows if row.get("passed") is not None]
        pass_count = sum(1 for row in graded if row.get("passed") is True)
        wall = [float(row["wall_latency_ms"]) for row in device_rows if row.get("wall_latency_ms") is not None]
        energy = [float(row["measured_energy_joules"]) for row in device_rows if row.get("measured_energy_joules") is not None]
        power = [
            float(row["measured_energy_joules"]) / (float(row["wall_latency_ms"]) / 1000.0)
            for row in device_rows
            if row.get("measured_energy_joules") is not None and row.get("wall_latency_ms") is not None and float(row["wall_latency_ms"]) > 0
        ]
        completion_tokens = [float(row["completion_tokens"]) for row in device_rows if row.get("completion_tokens") is not None]
        median_power_w = median(power) if power else (CLOUD_ASSUMED_POWER_W if device == "cloud" else None)
        summaries[device] = DeviceSummary(
            label=DEVICE_LABELS[device],
            total=len(device_rows),
            passed=pass_count,
            accuracy=(pass_count / len(graded)) if graded else None,
            median_wall_ms=median(wall) if wall else None,
            median_energy_j=median(energy) if energy else None,
            median_power_w=median_power_w,
            median_completion_tokens=median(completion_tokens) if completion_tokens else None,
        )
    return summaries


def _category_table(rows: dict[tuple[str, str], dict]) -> dict[str, dict[str, tuple[int, int]]]:
    categories: dict[str, dict[str, list[int]]] = {}
    for row in rows.values():
        if row.get("passed") is None:
            continue
        category = str(row.get("category", "other"))
        device = str(row.get("device"))
        categories.setdefault(category, {}).setdefault(device, [0, 0])
        categories[category][device][1] += 1
        if row.get("passed") is True:
            categories[category][device][0] += 1
    return {category: {device: (vals[0], vals[1]) for device, vals in device_map.items()} for category, device_map in categories.items()}


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "  .title { font: 700 23px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #0f172a; letter-spacing: -0.02em; }",
        "  .subtitle { font: 400 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #475569; }",
        "  .axis { font: 500 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #334155; }",
        "  .value { font: 700 12px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #0f172a; }",
        "  .panel-title { font: 700 14px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #0f172a; }",
        "  .panel-subtitle { font: 400 11px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #64748b; }",
        "  .small { font: 400 10px -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; fill: #64748b; }",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="{CANVAS_BG}"/>',
        f'<text x="32" y="38" class="title">{title}</text>',
    ]


def _format_value(value: float | None, *, percent: bool = False, digits: int = 1) -> str:
    if value is None:
        return "n/a"
    if percent:
        return f"{value * 100:.0f}%"
    return f"{value:.{digits}f}"


def _bar_chart_panel(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float | None],
    colors: list[str],
    value_suffix: str = "",
    percent: bool = False,
    max_value: float | None = None,
) -> list[str]:
    pad_top = 52
    pad_right = 16
    pad_bottom = 38
    pad_left = 42
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    chart_x = x + pad_left
    chart_y = y + pad_top
    lines = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{PANEL_BG}" stroke="{PANEL_BORDER}"/>',
        f'<text x="{x + 18}" y="{y + 24}" class="panel-title">{title}</text>',
        f'<text x="{x + 18}" y="{y + 40}" class="panel-subtitle">{subtitle}</text>',
    ]

    if max_value is None:
        usable_values = [value for value in values if value is not None]
        max_value = max(usable_values) if usable_values else 1.0
    if percent:
        max_value = 1.0
    if max_value <= 0:
        max_value = 1.0

    # Axis line and tick marks.
    lines.append(f'<line x1="{chart_x}" y1="{chart_y + plot_h}" x2="{chart_x + plot_w}" y2="{chart_y + plot_h}" stroke="{PANEL_BORDER}"/>')
    for tick in range(5):
        tick_value = max_value * tick / 4
        tick_y = chart_y + plot_h - (plot_h * tick / 4)
        lines.append(f'<line x1="{chart_x - 4}" y1="{tick_y}" x2="{chart_x + plot_w}" y2="{tick_y}" stroke="{GRID_COLOR}" stroke-dasharray="2 4"/>')
        label = _format_value(tick_value, percent=percent, digits=0 if percent else 1)
        lines.append(f'<text x="{chart_x - 8}" y="{tick_y + 4}" text-anchor="end" class="small">{label}{value_suffix}</text>')

    bar_gap = 16
    bar_w = (plot_w - bar_gap * (len(labels) - 1)) / max(len(labels), 1)
    for idx, (label, value, color) in enumerate(zip(labels, values, colors, strict=False)):
        bar_x = chart_x + idx * (bar_w + bar_gap)
        lines.append(f'<text x="{bar_x + bar_w / 2:.1f}" y="{y + height - 10}" text-anchor="middle" class="axis">{label}</text>')
        if value is None:
            lines.append(f'<text x="{bar_x + bar_w / 2:.1f}" y="{chart_y + plot_h / 2:.1f}" text-anchor="middle" class="value">n/a</text>')
            continue
        bar_h = plot_h * max(0.0, min(value, max_value)) / max_value
        top = chart_y + plot_h - bar_h
        lines.append(f'<rect x="{bar_x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="8" fill="{color}"/>')
        label_text = f'{_format_value(value, percent=percent, digits=1)}{value_suffix}'
        label_x = bar_x + bar_w / 2
        label_y = top - 8
        if label_y < chart_y + 12 or bar_h < 26:
            label_y = top + (bar_h / 2)
            label_fill = "#ffffff"
            label_baseline = "middle"
        else:
            label_fill = TEXT_DARK
            label_baseline = "auto"
        lines.append(
            f'<text x="{label_x:.1f}" y="{label_y:.1f}" text-anchor="middle" dominant-baseline="{label_baseline}" class="value" fill="{label_fill}">{label_text}</text>'
        )
    return lines


def _category_chart(
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    categories: dict[str, dict[str, tuple[int, int]]],
) -> list[str]:
    pad_top = 52
    pad_right = 16
    pad_bottom = 48
    pad_left = 42
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    chart_x = x + pad_left
    chart_y = y + pad_top
    ordered_categories = sorted(categories)
    bar_gap = 22
    cluster_gap = 28
    cluster_w = (plot_w - cluster_gap * (len(ordered_categories) - 1)) / max(len(ordered_categories), 1)
    bar_w = (cluster_w - bar_gap * (len(DEVICES) - 1)) / max(len(DEVICES), 1)
    lines = [
        f'<rect x="{x}" y="{y}" width="{width}" height="{height}" rx="18" fill="{PANEL_BG}" stroke="{PANEL_BORDER}"/>',
        f'<text x="{x + 18}" y="{y + 24}" class="panel-title">Blind-baseline pass rate by category</text>',
        f'<text x="{x + 18}" y="{y + 34}" class="panel-subtitle">Each group shows the fraction of graded prompts answered correctly when all prompts are forced to one device.</text>',
    ]

    lines.append(f'<line x1="{chart_x}" y1="{chart_y + plot_h}" x2="{chart_x + plot_w}" y2="{chart_y + plot_h}" stroke="{PANEL_BORDER}"/>')
    for tick in range(5):
        tick_value = tick / 4
        tick_y = chart_y + plot_h - (plot_h * tick / 4)
        lines.append(f'<line x1="{chart_x - 4}" y1="{tick_y}" x2="{chart_x + plot_w}" y2="{tick_y}" stroke="{GRID_COLOR}" stroke-dasharray="2 4"/>')
        lines.append(f'<text x="{chart_x - 8}" y="{tick_y + 4}" text-anchor="end" class="small">{tick_value:.0%}</text>')

    for cat_idx, category in enumerate(ordered_categories):
        cluster_x = chart_x + cat_idx * (cluster_w + cluster_gap)
        lines.append(f'<text x="{cluster_x + cluster_w / 2:.1f}" y="{y + height - 12}" text-anchor="middle" class="axis">{category}</text>')
        stats = categories[category]
        for dev_idx, device in enumerate(DEVICES):
            passed, total = stats.get(device, (0, 0))
            value = (passed / total) if total else None
            bar_x = cluster_x + dev_idx * (bar_w + bar_gap)
            bar_h = plot_h * (value if value is not None else 0.0)
            top = chart_y + plot_h - bar_h
            lines.append(f'<rect x="{bar_x:.1f}" y="{top:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" rx="6" fill="{DEVICE_COLORS[device]}"/>')
            if value is None:
                lines.append(f'<text x="{bar_x + bar_w / 2:.1f}" y="{top - 6:.1f}" text-anchor="middle" class="value">n/a</text>')
            else:
                lines.append(f'<text x="{bar_x + bar_w / 2:.1f}" y="{top - 6:.1f}" text-anchor="middle" class="value">{value:.0%}</text>')
    return lines


def _legend(x: int, y: int) -> list[str]:
    items = []
    cursor = x
    for device in DEVICES:
        items.append(f'<rect x="{cursor}" y="{y}" width="12" height="12" rx="3" fill="{DEVICE_COLORS[device]}"/>')
        items.append(f'<text x="{cursor + 18}" y="{y + 11}" class="axis">{DEVICE_LABELS[device]}</text>')
        cursor += 100
    return items


def _build_overall_svg(summaries: dict[str, DeviceSummary]) -> str:
    width, height = 980, 780
    lines = _svg_header(width, height, "Blind baseline comparison")
    lines.append('<text x="32" y="58" class="subtitle">Forced single-device baselines from offline calibration runs. White background preserved.</text>')
    lines.extend(_legend(32, 78))

    labels = [DEVICE_LABELS[device] for device in DEVICES]
    colors = [DEVICE_COLORS[device] for device in DEVICES]
    lines.extend(
        _bar_chart_panel(
            x=24,
            y=98,
            width=932,
            height=192,
            title="Pass rate",
            subtitle="Exact-match / mechanical grading on labeled prompts only.",
            labels=labels,
            values=[summaries[device].accuracy for device in DEVICES],
            colors=colors,
            percent=True,
        )
    )
    lines.extend(
        _bar_chart_panel(
            x=24,
            y=312,
            width=932,
            height=192,
            title="Median wall latency",
            subtitle="End-to-end request time measured in the calibration sweep.",
            labels=labels,
            values=[summaries[device].median_wall_ms for device in DEVICES],
            colors=colors,
            value_suffix=" ms",
            max_value=max((summaries[device].median_wall_ms or 0.0) for device in DEVICES),
        )
    )
    lines.extend(
        _bar_chart_panel(
            x=24,
            y=526,
            width=932,
            height=192,
            title="Median average power",
            subtitle="Phone and PC are derived from recorded energy divided by wall time; cloud uses a fixed 75 W assumption.",
            labels=labels,
            values=[summaries[device].median_power_w for device in DEVICES],
            colors=colors,
            value_suffix=" W",
            max_value=max((summaries[device].median_power_w or 0.0) for device in DEVICES),
        )
    )
    lines.append('</svg>')
    return "\n".join(lines)


def _build_category_svg(categories: dict[str, dict[str, tuple[int, int]]]) -> str:
    width, height = 980, 590
    lines = _svg_header(width, height, "Blind baseline comparison by category")
    lines.append('<text x="32" y="58" class="subtitle">Per-category blind routing accuracy, still on a white canvas.</text>')
    lines.extend(_legend(32, 78))
    lines.extend(_category_chart(x=24, y=98, width=932, height=450, categories=categories))
    lines.append('</svg>')
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="*", default=DEFAULT_INPUTS, help="Calibration sweep JSONL files.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for generated figures.")
    args = parser.parse_args()

    rows = _read_rows(args.input)
    summaries = _summarize(rows)
    categories = _category_table(rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    overall_svg = _build_overall_svg(summaries)
    category_svg = _build_category_svg(categories)

    overall_path = args.out_dir / "blind_baseline_overall.svg"
    category_path = args.out_dir / "blind_baseline_by_category.svg"
    summary_path = args.out_dir / "blind_baseline_summary.json"

    overall_path.write_text(overall_svg, encoding="utf-8")
    category_path.write_text(category_svg, encoding="utf-8")

    summary_payload = {
        "inputs": [str(path) for path in args.input],
        "devices": {
            device: {
                "label": summaries[device].label,
                "total": summaries[device].total,
                "passed": summaries[device].passed,
                "accuracy": summaries[device].accuracy,
                "median_wall_latency_ms": summaries[device].median_wall_ms,
                "median_energy_joules": summaries[device].median_energy_j,
                "median_power_watts": summaries[device].median_power_w,
                "median_completion_tokens": summaries[device].median_completion_tokens,
            }
            for device in DEVICES
        },
        "categories": {
            category: {
                device: {"passed": passed, "graded": graded, "accuracy": (passed / graded) if graded else None}
                for device, (passed, graded) in device_map.items()
            }
            for category, device_map in sorted(categories.items())
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Wrote {overall_path}")
    print(f"Wrote {category_path}")
    print(f"Wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
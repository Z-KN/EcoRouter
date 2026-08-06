"""Telemetry parsing and built-in demonstration scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .models import (
    CLOUD_AI_100_TDP_WATTS,
    Device,
    DeviceConfig,
    DeviceTelemetry,
    ValidationError,
    default_device_configs,
)

# Measured medians from the 106-prompt calibration sweep
# (benchmarks/calibration/heads/heads.json -> device_constants). Energy: 106
# samples each for phone/PC. Decode throughput: same source, also 106 samples
# each -- phone's model is much smaller (Qwen3-0.6B) and much faster than
# PC's (Qwen3-VL-4B-Instruct on the X-Elite NPU), so PC is both slower AND
# more energy-hungry per token, not a wash.
_PHONE_JOULES_PER_TOKEN = 0.0412
_PHONE_TOKENS_PER_SECOND = 94.22
_PC_JOULES_PER_TOKEN = 0.4447
_PC_TOKENS_PER_SECOND = 20.19

# Cloud has no decode-only throughput or per-token energy constant the way
# phone/PC do (heads.json reports both null for cloud -- CirrascaleExecutor
# never captured ttft/decode-speed, only whole-call latency and token
# counts). What *is* measured, from benchmarks/calibration/runs/
# sweep_cloud_llama70b.jsonl (100 rows with both fields): the median
# end-to-end rate of (prompt_tokens + completion_tokens) / api_latency_ms
# across the sweep, folding network + queueing + generation into one number
# since they were never captured separately for cloud. Because this rate
# already includes network time, cloud's network_latency_ms below is 0 --
# adding a separate network hop on top would double-count it. This is the
# same "wall-clock, not compute-only" tradeoff CLOUD_AI_100_TDP_WATTS makes
# for cloud's energy, for the same reason.
_CLOUD_TOKENS_PER_SECOND = 33.37
_CLOUD_JOULES_PER_TOKEN = CLOUD_AI_100_TDP_WATTS / _CLOUD_TOKENS_PER_SECOND


def _healthy() -> dict[Device, DeviceTelemetry]:
    return {
        # network_latency_ms for phone/PC (8/18) is still an illustrative,
        # unmeasured LAN-hop placeholder -- nothing in this repo measures
        # cross-device network latency. Left as-is rather than invented.
        Device.PHONE: DeviceTelemetry(
            True, 8, _PHONE_TOKENS_PER_SECOND, _PHONE_JOULES_PER_TOKEN, 0.08, 0.08, 95, 0
        ),
        Device.PC: DeviceTelemetry(
            True, 18, _PC_TOKENS_PER_SECOND, _PC_JOULES_PER_TOKEN, 0.18, 0.15, 85, 0
        ),
        Device.CLOUD: DeviceTelemetry(
            True, 0, _CLOUD_TOKENS_PER_SECOND, _CLOUD_JOULES_PER_TOKEN, 0.20, 0.12, None, 0.03
        ),
    }


def built_in_scenarios() -> dict[str, dict[Device, DeviceTelemetry]]:
    healthy = _healthy()
    phone_low_battery = dict(healthy)
    phone_low_battery[Device.PHONE] = DeviceTelemetry(
        True, 8, _PHONE_TOKENS_PER_SECOND, _PHONE_JOULES_PER_TOKEN, 0.08, 0.08, 4, 0
    )
    pc_congested = dict(healthy)
    pc_congested[Device.PC] = DeviceTelemetry(
        # Congestion is modeled as half the healthy throughput and ~2x the
        # healthy energy draw -- same ratios as before the healthy baseline
        # was corrected to the measured values.
        True, 18, _PC_TOKENS_PER_SECOND / 2, _PC_JOULES_PER_TOKEN * 2, 0.92, 0.90, 75, 0
    )
    cloud_offline = dict(healthy)
    cloud_offline[Device.CLOUD] = DeviceTelemetry(
        False, 0, 0, _CLOUD_JOULES_PER_TOKEN, 0.0, 0.0, None, 0.03
    )
    return {
        "healthy": healthy,
        "phone-low-battery": phone_low_battery,
        "pc-congested": pc_congested,
        "cloud-offline": cloud_offline,
    }


def telemetry_from_mapping(data: Mapping[str, Any]) -> dict[Device, DeviceTelemetry]:
    expected = {device.value for device in Device}
    if set(data) != expected:
        raise ValidationError("telemetry JSON must contain exactly phone, pc, and cloud")
    result: dict[Device, DeviceTelemetry] = {}
    for device in Device:
        values = data[device.value]
        if not isinstance(values, Mapping):
            raise ValidationError(f"telemetry for {device.value} must be a JSON object")
        try:
            result[device] = DeviceTelemetry(**values)
        except TypeError as error:
            raise ValidationError(f"invalid telemetry fields for {device.value}: {error}") from error
    return result


def load_telemetry(path: str | Path) -> dict[Device, DeviceTelemetry]:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ValidationError("telemetry JSON root must be an object")
    return telemetry_from_mapping(data)


def device_configs_from_mapping(data: Mapping[str, Any]) -> dict[Device, DeviceConfig]:
    device_data = data.get("devices")
    if not isinstance(device_data, Mapping):
        raise ValidationError("model configuration must contain a devices object")
    expected = {device.value for device in Device}
    if set(device_data) != expected:
        raise ValidationError("model configuration must contain exactly phone, pc, and cloud")
    configs = default_device_configs()
    for device in Device:
        values = device_data[device.value]
        if not isinstance(values, Mapping):
            raise ValidationError(f"configuration for {device.value} must be a JSON object")
        try:
            configs[device] = DeviceConfig(**values)
        except TypeError as error:
            raise ValidationError(f"invalid configuration fields for {device.value}: {error}") from error
    return configs


def load_device_configs(path: str | Path) -> dict[Device, DeviceConfig]:
    with Path(path).open(encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, Mapping):
        raise ValidationError("model configuration JSON root must be an object")
    return device_configs_from_mapping(data)

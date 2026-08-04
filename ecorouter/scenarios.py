"""Telemetry parsing and built-in demonstration scenarios."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .models import Device, DeviceConfig, DeviceTelemetry, ValidationError, default_device_configs


def _healthy() -> dict[Device, DeviceTelemetry]:
    return {
        Device.PHONE: DeviceTelemetry(True, 8, 100, 0.010, 0.08, 0.08, 95, 0),
        Device.PC: DeviceTelemetry(True, 18, 120, 0.025, 0.18, 0.15, 85, 0),
        Device.CLOUD: DeviceTelemetry(True, 55, 70, 0.040, 0.20, 0.12, None, 0.03),
    }


def built_in_scenarios() -> dict[str, dict[Device, DeviceTelemetry]]:
    healthy = _healthy()
    phone_low_battery = dict(healthy)
    phone_low_battery[Device.PHONE] = DeviceTelemetry(
        True, 8, 100, 0.010, 0.08, 0.08, 4, 0
    )
    pc_congested = dict(healthy)
    pc_congested[Device.PC] = DeviceTelemetry(
        True, 18, 35, 0.050, 0.92, 0.90, 75, 0
    )
    cloud_offline = dict(healthy)
    cloud_offline[Device.CLOUD] = DeviceTelemetry(
        False, 55, 0, 0.040, 0.0, 0.0, None, 0.03
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

// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.router

/**
 * Kotlin port of `ecorouter/scenarios.py`'s built-in telemetry scenarios.
 *
 * Live telemetry collection (battery/thermal/network probes on-device) is
 * not implemented here yet, same as it isn't on the PC side -- see
 * TODO.md's "Connect live telemetry collectors" item. Until then, routing
 * from the phone picks one of these canned snapshots the same way the PC
 * CLI's `--scenario` flag does.
 */
object Scenarios {

    // Measured medians from the 106-prompt calibration sweep
    // (benchmarks/calibration/heads/heads.json -> device_constants) -- kept
    // numerically identical to scenarios.py so on-device and PC routing agree
    // on a given scenario.
    private const val PHONE_JOULES_PER_TOKEN = 0.0412
    private const val PHONE_TOKENS_PER_SECOND = 94.22
    private const val PC_JOULES_PER_TOKEN = 0.4447
    private const val PC_TOKENS_PER_SECOND = 20.19
    private const val CLOUD_TOKENS_PER_SECOND = 33.37
    private val CLOUD_JOULES_PER_TOKEN = CLOUD_AI_100_TDP_WATTS / CLOUD_TOKENS_PER_SECOND

    private fun healthy(): Map<Device, DeviceTelemetry> = mapOf(
        Device.PHONE to DeviceTelemetry(
            available = true,
            networkLatencyMs = 8.0,
            throughputTokensPerSecond = PHONE_TOKENS_PER_SECOND,
            energyJoulesPerToken = PHONE_JOULES_PER_TOKEN,
            utilization = 0.08,
            thermalPressure = 0.08,
            batteryPercent = 95.0,
            cloudCostPer1kTokensUsd = 0.0,
        ),
        Device.PC to DeviceTelemetry(
            available = true,
            networkLatencyMs = 18.0,
            throughputTokensPerSecond = PC_TOKENS_PER_SECOND,
            energyJoulesPerToken = PC_JOULES_PER_TOKEN,
            utilization = 0.18,
            thermalPressure = 0.15,
            batteryPercent = 85.0,
            cloudCostPer1kTokensUsd = 0.0,
        ),
        Device.CLOUD to DeviceTelemetry(
            available = true,
            networkLatencyMs = 0.0,
            throughputTokensPerSecond = CLOUD_TOKENS_PER_SECOND,
            energyJoulesPerToken = CLOUD_JOULES_PER_TOKEN,
            utilization = 0.20,
            thermalPressure = 0.12,
            batteryPercent = null,
            cloudCostPer1kTokensUsd = 0.03,
        ),
    )

    fun builtInScenarios(): Map<String, Map<Device, DeviceTelemetry>> {
        val healthy = healthy()

        val phoneLowBattery = healthy.toMutableMap()
        phoneLowBattery[Device.PHONE] = healthy.getValue(Device.PHONE).copy(batteryPercent = 4.0)

        val pcCongested = healthy.toMutableMap()
        // Congestion modeled as half the healthy throughput and ~2x the
        // healthy energy draw, same ratios as scenarios.py.
        pcCongested[Device.PC] = healthy.getValue(Device.PC).copy(
            throughputTokensPerSecond = PC_TOKENS_PER_SECOND / 2,
            energyJoulesPerToken = PC_JOULES_PER_TOKEN * 2,
            utilization = 0.92,
            thermalPressure = 0.90,
            batteryPercent = 75.0,
        )

        val cloudOffline = healthy.toMutableMap()
        cloudOffline[Device.CLOUD] = healthy.getValue(Device.CLOUD).copy(
            available = false,
            networkLatencyMs = 0.0,
            throughputTokensPerSecond = 0.0,
            utilization = 0.0,
            thermalPressure = 0.0,
        )

        return mapOf(
            "healthy" to healthy,
            "phone-low-battery" to phoneLowBattery,
            "pc-congested" to pcCongested,
            "cloud-offline" to cloudOffline,
        )
    }
}

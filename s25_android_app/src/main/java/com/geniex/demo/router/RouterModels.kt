// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.router

/**
 * Kotlin port of `peqrouter/models.py`'s typed contracts, scoped to what the
 * on-device router needs: no calibrated-estimator fields, since
 * [PEQRouter] runs the static-capability path only (see its class doc).
 */

open class PEQRouterError(message: String) : Exception(message)

class ValidationError(message: String) : PEQRouterError(message)

class NoRouteError(message: String) : PEQRouterError(message)

private fun requireValid(condition: Boolean, message: () -> String) {
    if (!condition) throw ValidationError(message())
}

enum class Device(val value: String) {
    PHONE("phone"),
    PC("pc"),
    CLOUD("cloud"),
}

enum class OptimizationProfile(val value: String) {
    BALANCED("balanced"),
    LOW_LATENCY("low-latency"),
    LOW_ENERGY("low-energy"),
    HIGH_QUALITY("high-quality"),
}

enum class Intent(val value: String) {
    LOOKUP("lookup"),
    SUMMARIZATION("summarization"),
    CODING("coding"),
    REASONING("reasoning"),
    CREATIVE("creative"),
    GENERAL("general"),
}

data class DeviceTelemetry(
    val available: Boolean,
    val networkLatencyMs: Double,
    val throughputTokensPerSecond: Double,
    val energyJoulesPerToken: Double,
    val utilization: Double,
    val thermalPressure: Double,
    val batteryPercent: Double? = null,
    val cloudCostPer1kTokensUsd: Double = 0.0,
) {
    init {
        requireValid(networkLatencyMs >= 0) { "network_latency_ms must be non-negative" }
        requireValid(throughputTokensPerSecond >= 0) { "throughput_tokens_per_second must be non-negative" }
        if (available) {
            requireValid(throughputTokensPerSecond != 0.0) { "available devices need positive throughput" }
        }
        requireValid(energyJoulesPerToken >= 0) { "energy_joules_per_token must be non-negative" }
        requireValid(utilization in 0.0..1.0) { "utilization must be between 0.0 and 1.0" }
        requireValid(thermalPressure in 0.0..1.0) { "thermal_pressure must be between 0.0 and 1.0" }
        batteryPercent?.let { requireValid(it in 0.0..100.0) { "battery_percent must be between 0.0 and 100.0" } }
        requireValid(cloudCostPer1kTokensUsd >= 0) { "cloud_cost_per_1k_tokens_usd must be non-negative" }
    }
}

data class DeviceConfig(
    val modelId: String,
    val capabilityScore: Double,
) {
    init {
        requireValid(modelId.isNotBlank()) { "model_id must not be empty" }
        requireValid(capabilityScore in 0.0..1.0) { "capability_score must be between 0.0 and 1.0" }
    }
}

fun defaultDeviceConfigs(): Map<Device, DeviceConfig> = mapOf(
    Device.PHONE to DeviceConfig("phone-model", 0.60),
    Device.PC to DeviceConfig("pc-model", 0.80),
    Device.CLOUD to DeviceConfig("Llama-3.3-70B", 0.95),
)

// Qualcomm Cloud AI 100 accelerator rated TDP -- same constant and same
// "wall-clock upper-bound estimate, not a calibrated measurement" caveat as
// peqrouter/models.py's CLOUD_AI_100_TDP_WATTS.
const val CLOUD_AI_100_TDP_WATTS = 75.0

data class RouteRequest(
    val prompt: String,
    val origin: Device,
    val telemetry: Map<Device, DeviceTelemetry>,
    val profile: OptimizationProfile = OptimizationProfile.BALANCED,
) {
    init {
        requireValid(prompt.isNotBlank()) { "prompt must not be empty" }
        requireValid(origin == Device.PHONE || origin == Device.PC) { "origin must be phone or pc" }
        val missing = Device.entries.toSet() - telemetry.keys
        val extra = telemetry.keys - Device.entries.toSet()
        requireValid(missing.isEmpty() && extra.isEmpty()) {
            "telemetry must describe phone, pc, and cloud"
        }
    }
}

data class PromptAnalysis(
    val intent: Intent,
    val complexity: Double,
    val sensitive: Boolean,
    val piiCategories: List<String>,
    val estimatedInputTokens: Int,
    val estimatedOutputTokens: Int,
    val requiredQuality: Double,
)

data class CandidateEvaluation(
    val device: Device,
    val modelId: String,
    val eligible: Boolean,
    val exclusionReasons: List<String>,
    val qualitySufficient: Boolean,
    val predictedLatencyMs: Double?,
    val predictedEnergyJoules: Double?,
    val predictedCloudCostUsd: Double?,
    val score: Double?,
    val penalties: Map<String, Double> = emptyMap(),
)

data class RouteDecision(
    val selectedDevice: Device,
    val modelId: String,
    val profile: OptimizationProfile,
    val analysis: PromptAnalysis,
    val qualityDegraded: Boolean,
    val explanation: String,
    val candidates: List<CandidateEvaluation>,
) {
    val selectedCandidate: CandidateEvaluation
        get() = candidates.first { it.device == selectedDevice }
}

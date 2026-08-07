// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.router

import kotlin.math.max
import kotlin.math.min

/**
 * Kotlin port of `peqrouter/router.py`'s [PEQRouter] policy engine, scoped to
 * its static-capability path -- equivalent to running the Python CLI with
 * `--no-estimator`. There is no on-device MiniLM/torch calibrated estimator
 * (`peqrouter/estimator.py`) yet; quality is judged purely by comparing each
 * device's configured `capabilityScore` against the analyzer's
 * `requiredQuality`, and output length is the analyzer's heuristic guess.
 * Everything else -- eligibility gating, latency/energy/cost prediction,
 * profile-weighted scoring, and the degraded-quality fallback -- mirrors
 * router.py exactly so a decision made here agrees with one made on the PC
 * for the same prompt, telemetry, and profile.
 */
class PEQRouter(
    private val deviceConfigs: Map<Device, DeviceConfig> = defaultDeviceConfigs(),
) {
    init {
        if (deviceConfigs.keys != Device.entries.toSet()) {
            throw ValidationError("device configuration must describe phone, pc, and cloud")
        }
    }

    companion object {
        private val PROFILE_WEIGHTS: Map<OptimizationProfile, Map<String, Double>> = mapOf(
            OptimizationProfile.BALANCED to mapOf("latency" to 0.500, "energy" to 0.333, "quality" to 0.167),
            OptimizationProfile.LOW_LATENCY to mapOf("latency" to 0.733, "energy" to 0.133, "quality" to 0.134),
            OptimizationProfile.LOW_ENERGY to mapOf("latency" to 0.250, "energy" to 0.667, "quality" to 0.083),
            OptimizationProfile.HIGH_QUALITY to mapOf("latency" to 0.200, "energy" to 0.133, "quality" to 0.667),
        )

        private val DEVICE_ORDER: Map<Device, Int> = mapOf(Device.PHONE to 0, Device.PC to 1, Device.CLOUD to 2)

        private fun clamp(value: Double): Double = max(0.0, min(1.0, value))
    }

    fun route(request: RouteRequest): RouteDecision {
        val analysis = HeuristicPromptAnalyzer.analyze(request.prompt)
        val candidates = Device.entries.map { evaluate(it, request, analysis) }
        val eligible = candidates.filter { it.eligible }
        if (eligible.isEmpty()) {
            val reasons = candidates.joinToString("; ") { "${it.device.value}: ${it.exclusionReasons.joinToString(", ")}" }
            throw NoRouteError("no eligible destination; $reasons")
        }

        val sufficient = eligible.filter { it.qualitySufficient }
        val qualityDegraded: Boolean
        val selected: CandidateEvaluation
        val explanation: String
        if (sufficient.isNotEmpty()) {
            qualityDegraded = false
            selected = sufficient.minWithOrNull(scoreSortKeyComparator())!!
            explanation = "Selected ${selected.device.value}/${selected.modelId}: it had the lowest " +
                "${request.profile.value} score among privacy-safe, healthy models meeting required quality."
        } else {
            qualityDegraded = true
            selected = eligible.minWithOrNull(
                compareByDescending<CandidateEvaluation> { deviceConfigs.getValue(it.device).capabilityScore }
                    .then(scoreSortKeyComparator()),
            )!!
            explanation = "Selected ${selected.device.value}/${selected.modelId} as the highest-capability eligible " +
                "model; no privacy-safe, healthy destination met the requested quality."
        }

        return RouteDecision(
            selectedDevice = selected.device,
            modelId = selected.modelId,
            profile = request.profile,
            analysis = analysis,
            qualityDegraded = qualityDegraded,
            explanation = explanation,
            candidates = candidates,
        )
    }

    private fun scoreSortKeyComparator(): Comparator<CandidateEvaluation> =
        compareBy<CandidateEvaluation> { it.score ?: Double.POSITIVE_INFINITY }
            .thenBy { it.predictedLatencyMs ?: Double.POSITIVE_INFINITY }
            .thenBy { DEVICE_ORDER.getValue(it.device) }

    private fun evaluate(device: Device, request: RouteRequest, analysis: PromptAnalysis): CandidateEvaluation {
        val telemetry = request.telemetry.getValue(device)
        val config = deviceConfigs.getValue(device)
        val exclusions = mutableListOf<String>()
        if (!telemetry.available) exclusions.add("unavailable")
        if (device == Device.CLOUD && analysis.sensitive) exclusions.add("cloud blocked by privacy policy")

        val outputTokens = analysis.estimatedOutputTokens
        val totalTokens = analysis.estimatedInputTokens + outputTokens
        val canEstimate = telemetry.throughputTokensPerSecond > 0
        val networkLatency = if (device == request.origin) 0.0 else telemetry.networkLatencyMs
        val latency = if (canEstimate) {
            networkLatency + totalTokens / telemetry.throughputTokensPerSecond * 1000
        } else null
        val energy = if (canEstimate) totalTokens * telemetry.energyJoulesPerToken else null
        val cloudCost = if (canEstimate) totalTokens / 1000.0 * telemetry.cloudCostPer1kTokensUsd else null

        var score: Double? = null
        val penalties = mutableMapOf<String, Double>()
        if (canEstimate && latency != null && energy != null) {
            penalties["latency"] = clamp(latency / 10_000)
            penalties["energy"] = clamp(energy / 50)
            // No calibrated estimator on-device -- quality is always scored
            // as a penalty here (unlike router.py's estimator branch, which
            // omits it once a trusted per-prompt estimate exists).
            penalties["quality"] = 1.0 - config.capabilityScore
            val weights = PROFILE_WEIGHTS.getValue(request.profile)
            val applicableWeight = penalties.keys.sumOf { weights.getValue(it) }
            score = penalties.entries.sumOf { (name, value) -> weights.getValue(name) * value } / applicableWeight
        }

        return CandidateEvaluation(
            device = device,
            modelId = config.modelId,
            eligible = exclusions.isEmpty(),
            exclusionReasons = exclusions,
            qualitySufficient = config.capabilityScore >= analysis.requiredQuality,
            predictedLatencyMs = latency,
            predictedEnergyJoules = energy,
            predictedCloudCostUsd = cloudCost,
            score = score,
            penalties = penalties,
        )
    }
}

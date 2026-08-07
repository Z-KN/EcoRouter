// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.router

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

/**
 * Parity checks against tests/test_router.py's static-capability-path cases
 * (equivalent to the Python CLI's `--no-estimator`) -- there is no
 * calibrated-estimator port on-device, so the estimator-specific tests
 * (untrusted-estimate fallback, trusted p_pass override, etc.) don't apply
 * here.
 */
class EcoRouterTest {

    private fun requestFor(
        prompt: String,
        origin: Device = Device.PHONE,
        scenario: String = "healthy",
        profile: OptimizationProfile = OptimizationProfile.BALANCED,
    ): RouteRequest = RouteRequest(prompt, origin, Scenarios.builtInScenarios().getValue(scenario), profile)

    @Test
    fun `short lookup in balanced profile picks phone`() {
        val decision = EcoRouter().route(requestFor("What's the weather tomorrow?"))

        assertEquals(Device.PHONE, decision.selectedDevice)
        assertFalse(decision.qualityDegraded)
    }

    @Test
    fun `complex reasoning uses cloud`() {
        val prompt = "First analyze this distributed architecture, then compare every trade-off step by step. " +
            "Include equations and detailed reasoning. ".repeat(20) +
            "Which design wins? What could fail?"

        val decision = EcoRouter().route(requestFor(prompt))

        assertEquals(Device.CLOUD, decision.selectedDevice)
    }

    @Test
    fun `high quality profile can change the winner`() {
        val balanced = EcoRouter().route(requestFor("What's the weather tomorrow?"))
        val highQuality = EcoRouter().route(
            requestFor("What's the weather tomorrow?", profile = OptimizationProfile.HIGH_QUALITY),
        )

        assertEquals(Device.PHONE, balanced.selectedDevice)
        assertEquals(Device.CLOUD, highQuality.selectedDevice)
    }

    @Test
    fun `PII blocks cloud`() {
        val decision = EcoRouter().route(requestFor("Summarize records for alice@example.com"))
        val cloud = decision.candidates.first { it.device == Device.CLOUD }

        assertFalse(cloud.eligible)
        assertTrue(cloud.exclusionReasons.contains("cloud blocked by privacy policy"))
    }

    @Test
    fun `sensitive complex request stays local and flags degradation`() {
        val prompt = "Analyze records for alice@example.com. First compare every option step by step. " +
            "Detailed architecture equations and optimization requirements. ".repeat(25) +
            "What should change? What could fail?"

        val decision = EcoRouter().route(requestFor(prompt))

        assertEquals(Device.PC, decision.selectedDevice)
        assertTrue(decision.qualityDegraded)
    }

    @Test
    fun `no route when all destinations are blocked`() {
        val telemetry = mapOf(
            Device.PHONE to DeviceTelemetry(false, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0),
            Device.PC to DeviceTelemetry(false, 0.0, 0.0, 0.0, 0.0, 0.0, 50.0, 0.0),
            Device.CLOUD to DeviceTelemetry(true, 30.0, 100.0, 0.1, 0.1, 0.1, null, 0.01),
        )

        try {
            EcoRouter().route(RouteRequest("Email alice@example.com", Device.PHONE, telemetry))
            fail("expected NoRouteError")
        } catch (expected: NoRouteError) {
            // expected
        }
    }

    @Test
    fun `origin sets local network latency to zero`() {
        val telemetry = Device.entries.associateWith { DeviceTelemetry(true, 100.0, 100.0, 0.01, 0.1, 0.1, null, 0.0) }
        val configs = Device.entries.associateWith { DeviceConfig("${it.value}-model", 0.60) } +
            mapOf(Device.CLOUD to DeviceConfig("cloud-model", 0.50))

        val decision = EcoRouter(configs).route(RouteRequest("Hello", Device.PC, telemetry))

        assertEquals(Device.PC, decision.selectedDevice)
    }

    @Test
    fun `exact tie uses stable device order`() {
        val telemetry = Device.entries.associateWith { DeviceTelemetry(true, 0.0, 100.0, 0.01, 0.1, 0.1, null, 0.0) }
        val configs = Device.entries.associateWith { DeviceConfig("${it.value}-model", 0.60) }

        val decision = EcoRouter(configs).route(RouteRequest("Contact alice@example.com", Device.PHONE, telemetry))

        assertEquals(Device.PHONE, decision.selectedDevice)
    }

    @Test
    fun `request requires all three telemetry entries`() {
        try {
            RouteRequest(
                "Hello",
                Device.PHONE,
                mapOf(Device.PHONE to Scenarios.builtInScenarios().getValue("healthy").getValue(Device.PHONE)),
            )
            fail("expected ValidationError")
        } catch (expected: ValidationError) {
            // expected
        }
    }

    @Test
    fun `available device requires positive throughput`() {
        try {
            DeviceTelemetry(true, 10.0, 0.0, 0.01, 0.1, 0.1, 80.0, 0.0)
            fail("expected ValidationError")
        } catch (expected: ValidationError) {
            // expected
        }
    }

    @Test
    fun `low battery and thermal pressure no longer affect routing`() {
        val telemetry = Scenarios.builtInScenarios().getValue("phone-low-battery").toMutableMap()
        telemetry[Device.PC] = DeviceTelemetry(true, 18.0, 120.0, 0.025, 0.18, 0.95, 85.0, 0.0)

        val decision = EcoRouter().route(RouteRequest("Hello", Device.PHONE, telemetry))

        assertTrue(decision.candidates.first { it.device == Device.PHONE }.eligible)
        assertTrue(decision.candidates.first { it.device == Device.PC }.eligible)
        assertEquals(
            setOf("latency", "energy", "quality"),
            decision.candidates.first { it.device == Device.PHONE }.penalties.keys,
        )
    }

    @Test
    fun `cloud offline scenario never selects cloud`() {
        val decision = EcoRouter().route(requestFor("What's the weather tomorrow?", scenario = "cloud-offline"))

        assertNotEquals(Device.CLOUD, decision.selectedDevice)
    }
}

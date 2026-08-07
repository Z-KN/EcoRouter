// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.router

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

/** Parity checks against tests/test_analyzer.py's cases for HeuristicPromptAnalyzer. */
class PromptAnalyzerTest {

    @Test
    fun `detects PII categories without retaining values`() {
        val analysis = HeuristicPromptAnalyzer.analyze(
            "Email alice@example.com or call 217-555-0199; password=top-secret",
        )

        assertTrue(analysis.sensitive)
        assertEquals(listOf("email", "phone_number", "secret"), analysis.piiCategories)
    }

    @Test
    fun `classifies intent using documented precedence`() {
        assertEquals(Intent.CODING, HeuristicPromptAnalyzer.analyze("Debug this Python function").intent)
        assertEquals(Intent.SUMMARIZATION, HeuristicPromptAnalyzer.analyze("Summarize these notes").intent)
        assertEquals(Intent.LOOKUP, HeuristicPromptAnalyzer.analyze("What's the weather?").intent)
    }

    @Test
    fun `multi-step reasoning is more complex than lookup`() {
        val lookup = HeuristicPromptAnalyzer.analyze("What's the weather tomorrow?")
        val reasoning = HeuristicPromptAnalyzer.analyze(
            "First analyze the architecture, then compare the trade-offs step by step. " +
                "What is fastest? What is most efficient?",
        )

        assertTrue(reasoning.complexity > lookup.complexity)
        assertTrue(reasoning.requiredQuality > lookup.requiredQuality)
        assertTrue(reasoning.estimatedOutputTokens > lookup.estimatedOutputTokens)
    }

    @Test
    fun `detects person and address`() {
        var analysis = HeuristicPromptAnalyzer.analyze(
            "Patient Maria Garcia, date of birth 1984-03-12, needs a discharge summary.",
        )
        assertTrue(analysis.sensitive)
        assertEquals(listOf("person"), analysis.piiCategories)

        analysis = HeuristicPromptAnalyzer.analyze(
            "Employee Robert Chen, who lives at 42 Oak Street, needs a performance review.",
        )
        assertTrue(analysis.sensitive)
        assertEquals(listOf("person", "address"), analysis.piiCategories)
    }

    @Test
    fun `plain prompt is not sensitive`() {
        val analysis = HeuristicPromptAnalyzer.analyze("Write a haiku about the ocean")
        assertFalse(analysis.sensitive)
        assertTrue(analysis.piiCategories.isEmpty())
    }
}

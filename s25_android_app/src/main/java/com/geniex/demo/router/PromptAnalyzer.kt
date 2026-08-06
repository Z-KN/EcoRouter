// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.router

import kotlin.math.ceil
import kotlin.math.min

/**
 * Kotlin port of `ecorouter/analyzer.py`'s `HeuristicPromptAnalyzer`: the
 * same dependency-free, regex-only PII/intent/complexity heuristic, so a
 * prompt accepted on the phone gets the identical privacy-sensitivity and
 * routing-feature verdict it would get from the PC's Python router. Detection
 * is deliberately blunt and biased toward over-matching (see analyzer.py) --
 * over-blocking cloud is the safe failure direction here too.
 */
object HeuristicPromptAnalyzer {

    private val SECRET_PATTERN = Regex(
        """\b(?:api[_ -]?key|access[_ -]?token|password|private[_ -]?key)\s*(?:is|=|:)\s*[^\s,;]+""",
        RegexOption.IGNORE_CASE,
    )

    private const val ADDRESS_SUFFIX =
        "Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|" +
            "Court|Ct|Place|Pl|Way|Circle|Cir|Highway|Hwy|Parkway|Pkwy"

    private val PII_PATTERNS: List<Pair<String, Regex>> = listOf(
        "email" to Regex("""\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b""", RegexOption.IGNORE_CASE),
        "ssn" to Regex("""(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"""),
        "payment_card" to Regex("""(?<!\d)(?:\d[ -]*?){13,19}(?!\d)"""),
        "phone_number" to Regex("""(?<!\w)(?:\+?1[ .-]?)?(?:\(?\d{3}\)?[ .-]?)\d{3}[ .-]\d{4}(?!\w)"""),
        "secret" to SECRET_PATTERN,
        // No NLP on-device: names/addresses use the same blunt instrument as
        // analyzer.py -- consecutive Title-Case words, and a house-number-plus-
        // street-suffix pattern. Both over-match on purpose.
        "person" to Regex("""\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b"""),
        "address" to Regex("""\b\d{1,6}\s+(?:[A-Z][a-zA-Z]*\s+){1,4}(?:$ADDRESS_SUFFIX)\.?\b"""),
    )

    private val INTENT_PATTERNS: List<Pair<Intent, Regex>> = listOf(
        Intent.CODING to Regex("""\b(?:code|debug|function|class|python|javascript|sql|algorithm|compile)\b""", RegexOption.IGNORE_CASE),
        Intent.SUMMARIZATION to Regex("""\b(?:summari[sz]e|summary|condense|key points|tl;dr)\b""", RegexOption.IGNORE_CASE),
        Intent.CREATIVE to Regex("""\b(?:write a story|poem|brainstorm|creative|slogan|screenplay)\b""", RegexOption.IGNORE_CASE),
        Intent.REASONING to Regex("""\b(?:analy[sz]e|compare|derive|prove|reason|evaluate|trade-?offs?|step by step)\b""", RegexOption.IGNORE_CASE),
        Intent.LOOKUP to Regex("""\b(?:what|when|where|who|weather|define|lookup|find)\b""", RegexOption.IGNORE_CASE),
    )

    private val MULTI_STEP = Regex(
        """\b(?:first|second|third|then|next|finally|step\s+\d+|and then|checklist|action plan)\b""",
        RegexOption.IGNORE_CASE,
    )
    private val COMPLEX_REASONING = Regex(
        """\b(?:analy[sz]e|compare|derive|prove|reason|algorithm|debug|optimi[sz]e|architecture|equation|calculate)\b""",
        RegexOption.IGNORE_CASE,
    )

    private val OUTPUT_TOKENS: Map<Intent, Int> = mapOf(
        Intent.LOOKUP to 96,
        Intent.SUMMARIZATION to 256,
        Intent.CODING to 512,
        Intent.REASONING to 384,
        Intent.CREATIVE to 384,
        Intent.GENERAL to 192,
    )

    fun analyze(prompt: String): PromptAnalysis {
        val categories = PII_PATTERNS.filter { (_, pattern) -> pattern.containsMatchIn(prompt) }.map { it.first }

        val intent = intent(prompt)
        val inputTokens = maxOf(1, ceil(prompt.length / 4.0).toInt())

        var complexity = 0.15
        if (inputTokens > 80) complexity += 0.20
        if (inputTokens > 200) complexity += 0.20
        if (MULTI_STEP.containsMatchIn(prompt)) complexity += 0.20
        if (COMPLEX_REASONING.containsMatchIn(prompt)) complexity += 0.20
        if (prompt.count { it == '?' } > 1) complexity += 0.10
        complexity = min(1.0, complexity)

        return PromptAnalysis(
            intent = intent,
            complexity = complexity,
            sensitive = categories.isNotEmpty(),
            piiCategories = categories,
            estimatedInputTokens = inputTokens,
            estimatedOutputTokens = OUTPUT_TOKENS.getValue(intent),
            requiredQuality = min(0.95, 0.45 + 0.50 * complexity),
        )
    }

    private fun intent(prompt: String): Intent {
        for ((intent, pattern) in INTENT_PATTERNS) {
            if (pattern.containsMatchIn(prompt)) return intent
        }
        return Intent.GENERAL
    }
}

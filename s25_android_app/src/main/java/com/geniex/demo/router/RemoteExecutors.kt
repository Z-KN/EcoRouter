// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.router

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import org.json.JSONObject
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStreamReader
import java.io.OutputStreamWriter
import java.net.HttpURLConnection
import java.net.URL
import java.nio.charset.StandardCharsets

/** Raised when the local X-Elite PC server is unreachable or misconfigured. */
class PcConfigurationError(message: String) : EcoRouterError(message)

/** Raised when the local X-Elite PC server responds but not usefully. */
class PcExecutionError(message: String) : EcoRouterError(message)

/**
 * Result of dispatching a routed request to a remote executor, trimmed to
 * the fields [RouterActivity] displays. Mirrors the shape
 * `XEliteExecutor.execute_observed` returns in executors.py, just read from
 * this side of the same wire protocol instead of the other.
 */
data class RemoteExecutionResult(
    val text: String,
    val modelId: String?,
    val apiTurnaroundLatencyMs: Double,
    val promptTokens: Int?,
    val completionTokens: Int?,
    val totalTokens: Int?,
    val ttftMs: Double?,
    val prefillSpeedTokensPerSecond: Double?,
    val decodeSpeedTokensPerSecond: Double?,
    val computeUnit: String?,
    val backend: String?,
)

/**
 * Kotlin counterpart of `XEliteExecutor` (executors.py): calls the same
 * OpenAI-compatible `/v1/chat/completions` endpoint exposed by
 * `x_elite_laptop_server/serve_qwen_vl.py`, but dialed from the phone over
 * the LAN instead of from the PC process itself -- this is the new hop that
 * lets a routing decision made *on the phone* actually reach the PC.
 *
 * There is no environment-variable default here the way the Python
 * executor has `XELITE_SERVER_ENDPOINT` (defaulting to `localhost:8000`,
 * which means "the PC" only when the router itself runs on the PC): from
 * the phone, "the PC" is some LAN address the user has to supply, so
 * [RouterActivity] takes it as an explicit field instead.
 */
object XEliteRemoteExecutor {

    private const val TIMEOUT_MS = 120_000

    suspend fun execute(endpoint: String, prompt: String, maxTokens: Int): RemoteExecutionResult =
        withContext(Dispatchers.IO) {
            val base = endpoint.trim().trimEnd('/')
            if (base.isEmpty()) {
                throw PcConfigurationError("PC endpoint is not set -- enter the address shown by x_elite_laptop_server")
            }

            val payload = JSONObject()
                .put(
                    "messages",
                    org.json.JSONArray().put(JSONObject().put("role", "user").put("content", prompt)),
                )
                .put("max_tokens", maxTokens)
                .put("stream", false)

            val started = System.nanoTime()
            val body = try {
                postJson("$base/v1/chat/completions", payload, headers = emptyMap())
            } catch (error: IOException) {
                throw PcConfigurationError("could not reach the PC server at $base: ${error.message}")
            }
            val latencyMs = (System.nanoTime() - started) / 1_000_000.0

            val content = try {
                body.getJSONArray("choices").getJSONObject(0).getJSONObject("message").getString("content")
            } catch (error: Exception) {
                throw PcExecutionError("PC server returned an unexpected response shape.")
            }
            if (content.isBlank()) throw PcExecutionError("PC server returned an empty response.")

            val usage = body.optJSONObject("usage")
            val profile = body.optJSONObject("quad_profile")

            RemoteExecutionResult(
                text = content,
                modelId = body.optStringOrNull("model"),
                apiTurnaroundLatencyMs = latencyMs,
                promptTokens = usage?.optInt("prompt_tokens")?.takeIf { usage.has("prompt_tokens") },
                completionTokens = usage?.optInt("completion_tokens")?.takeIf { usage.has("completion_tokens") },
                totalTokens = usage?.optInt("total_tokens")?.takeIf { usage.has("total_tokens") },
                ttftMs = profile?.optDouble("ttft_ms")?.takeIf { profile.has("ttft_ms") },
                prefillSpeedTokensPerSecond = profile?.optDouble("prefill_speed_tok_s")?.takeIf { profile.has("prefill_speed_tok_s") },
                decodeSpeedTokensPerSecond = profile?.optDouble("decode_speed_tok_s")?.takeIf { profile.has("decode_speed_tok_s") },
                computeUnit = profile?.optStringOrNull("device"),
                backend = profile?.optStringOrNull("backend"),
            )
        }

    private fun JSONObject.optStringOrNull(key: String): String? = if (has(key)) getString(key) else null

    private fun postJson(url: String, payload: JSONObject, headers: Map<String, String>): JSONObject {
        val connection = URL(url).openConnection() as HttpURLConnection
        try {
            connection.requestMethod = "POST"
            connection.doOutput = true
            connection.connectTimeout = TIMEOUT_MS
            connection.readTimeout = TIMEOUT_MS
            connection.setRequestProperty("Content-Type", "application/json")
            headers.forEach { (key, value) -> connection.setRequestProperty(key, value) }

            OutputStreamWriter(connection.outputStream, StandardCharsets.UTF_8).use { it.write(payload.toString()) }

            val status = connection.responseCode
            val stream = if (status in 200..299) connection.inputStream else connection.errorStream
            val text = BufferedReader(InputStreamReader(stream, StandardCharsets.UTF_8)).use { it.readText() }
            if (status !in 200..299) {
                throw IOException("HTTP $status: $text")
            }
            return JSONObject(text)
        } finally {
            connection.disconnect()
        }
    }
}

/**
 * Cloud dispatch from the phone is deliberately not wired up yet: unlike PC
 * (a plain LAN endpoint) and phone (no network hop at all), Cirrascale
 * requires an API key -- README.md is explicit that credential must live in
 * an environment variable, never source, command line, or a committed file,
 * and a personal phone is a materially different trust boundary than the PC
 * process that currently holds it. Surfacing that as an explicit error
 * keeps the omission visible instead of a silent no-op.
 */
class CloudDispatchNotSupportedError(message: String) : EcoRouterError(message)

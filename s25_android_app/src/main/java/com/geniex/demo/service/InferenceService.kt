// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import androidx.core.app.NotificationCompat
import com.geniex.demo.GenerationConfigSample
import com.geniex.demo.R
import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.VlmWrapper
import com.geniex.sdk.bean.ChatMessage
import com.geniex.sdk.bean.LlmStreamResult
import com.geniex.sdk.bean.VlmChatMessage
import com.geniex.sdk.bean.VlmContent
import fi.iki.elonen.NanoHTTPD
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.Inet4Address
import java.net.NetworkInterface
import java.security.SecureRandom
import java.util.UUID

/**
 * Owns the loaded GenieX model and a NanoHTTPD server exposing it on the LAN
 * at an OpenAI-compatible `/v1/chat/completions`, so EcoRouter can route to
 * this phone the same way it routes to the PC (`x_elite_laptop_server`) and
 * the cloud.
 *
 * Runs as a foreground service so serving continues while the app is
 * backgrounded or the screen is off -- required for the wireless power
 * measurement workflow: the phone must stay unplugged (see
 * [measuredNpuPowerMw]), so requests have to keep flowing without a charging
 * cable to wake the screen.
 *
 * [MainActivity] binds to this service and stores the loaded model here
 * (rather than in its own fields) so that both the UI's chat flow and this
 * server's request flow share one native model handle -- concurrent
 * `generate()` calls on that handle crash the app, so both paths must
 * serialize through [mutex].
 */
class InferenceService : Service() {

    inner class LocalBinder : Binder() {
        fun getService(): InferenceService = this@InferenceService
    }

    private val binder = LocalBinder()

    override fun onBind(intent: Intent?): IBinder = binder

    /** Serializes every call into the native model handle -- shared with MainActivity's UI chat path. */
    val mutex = Mutex()

    @Volatile var llmWrapper: LlmWrapper? = null
    @Volatile var vlmWrapper: VlmWrapper? = null
    @Volatile var isLoadLlmModel = false
    @Volatile var isLoadVlmModel = false

    /** Catalog id (model_list.json `id`) of the currently loaded model, e.g. "Qwen3-0.6B-GGUF". */
    @Volatile var loadedModelId: String = ""

    /** Whether the loaded model is dispatching to the NPU -- gates the measured-energy fields, see [measuredNpuPowerMw]. */
    @Volatile var lastComputeUnitIsNpu = true

    private var httpServer: RouterHttpServer? = null
    private val startedAtMs = System.currentTimeMillis()
    private var requestCount = 0L
    private var totalDecodeTokens = 0L
    private var totalDecodeTimeS = 0.0

    class ServiceBusyException : Exception("a generation is already in progress")

    override fun onCreate() {
        super.onCreate()
        createNotificationChannel()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_NOT_STICKY
    }

    override fun onDestroy() {
        stopHttpServer()
        super.onDestroy()
    }

    fun isModelLoaded(): Boolean = isLoadLlmModel || isLoadVlmModel

    fun setLoadedLlm(wrapper: LlmWrapper, modelId: String, computeUnitIsNpu: Boolean) {
        llmWrapper = wrapper
        vlmWrapper = null
        isLoadLlmModel = true
        isLoadVlmModel = false
        loadedModelId = modelId
        lastComputeUnitIsNpu = computeUnitIsNpu
    }

    fun setLoadedVlm(wrapper: VlmWrapper, modelId: String, computeUnitIsNpu: Boolean) {
        vlmWrapper = wrapper
        llmWrapper = null
        isLoadVlmModel = true
        isLoadLlmModel = false
        loadedModelId = modelId
        lastComputeUnitIsNpu = computeUnitIsNpu
    }

    fun clearLoaded() {
        llmWrapper = null
        vlmWrapper = null
        isLoadLlmModel = false
        isLoadVlmModel = false
        loadedModelId = ""
    }

    /**
     * Incremental NPU power (mW) measured on-device: whole-phone battery
     * discharge current during sustained back-to-back decode, minus an idle
     * baseline (~796 mW), phone unplugged. Not a per-app isolation, but a
     * solid steady-state figure for this exact model+bundle on this exact
     * hardware. Only valid when the model is actually running on NPU (gated
     * by [lastComputeUnitIsNpu]) and the phone is not currently charging.
     */
    fun measuredNpuPowerMw(modelId: String): Double? =
        when (modelId) {
            "Qwen3-0.6B-GGUF" -> 2538.1 // llama.cpp/ggml-hexagon path
            "Qwen3.5-0.8B-GGUF" -> 3678.9 // VLM, llama.cpp/ggml-hexagon path
            "Qwen3-0.6B" -> 3345.5 // sideloaded QAIRT bundle; observed range 3345-5486 mW,
            // using the conservative (outlier-excluded) estimate
            else -> null
        }

    // ---- LAN server lifecycle ----------------------------------------

    fun isServerRunning(): Boolean = httpServer != null

    fun serverPort(): Int = httpServer?.listeningPort ?: DEFAULT_PORT

    fun serverToken(): String {
        val prefs = getSharedPreferences(PREFS_FILE, Context.MODE_PRIVATE)
        var token = prefs.getString(KEY_SERVER_TOKEN, null)
        if (token == null) {
            token = ByteArray(24).also { SecureRandom().nextBytes(it) }
                .joinToString("") { "%02x".format(it) }
            prefs.edit().putString(KEY_SERVER_TOKEN, token).apply()
        }
        return token
    }

    /** First non-loopback IPv4 address -- the LAN address EcoRouter should target. */
    fun localAddress(): String? =
        try {
            NetworkInterface.getNetworkInterfaces().toList()
                .flatMap { it.inetAddresses.toList() }
                .firstOrNull { !it.isLoopbackAddress && it is Inet4Address }
                ?.hostAddress
        } catch (e: Exception) {
            Log.e(TAG, "localAddress lookup failed", e)
            null
        }

    fun startHttpServer(port: Int = DEFAULT_PORT): Result<Int> {
        val existing = httpServer
        if (existing != null) return Result.success(existing.listeningPort)
        return try {
            val server = RouterHttpServer(port)
            server.start(NanoHTTPD.SOCKET_READ_TIMEOUT, false)
            httpServer = server
            // Promote to a foreground service only while actually listening --
            // this is what keeps routed requests flowing while the app is
            // backgrounded/screen off, without showing a notification during
            // ordinary local chat use.
            startForeground(NOTIFICATION_ID, buildNotification())
            Result.success(server.listeningPort)
        } catch (e: IOException) {
            Result.failure(e)
        }
    }

    fun stopHttpServer() {
        httpServer?.stop()
        httpServer = null
        stopForeground(STOP_FOREGROUND_REMOVE)
    }

    // ---- generation ----------------------------------------------------

    private data class GenResult(
        val text: String,
        val ttftMs: Double,
        val promptTokens: Int,
        val prefillSpeedTokS: Double,
        val generatedTokens: Int,
        val decodeSpeedTokS: Double,
        val promptTimeMs: Double,
        val decodeTimeMs: Double,
    )

    /**
     * Single-turn, non-streaming generation for a routed request. Deliberately
     * does not touch [MainActivity]'s visible chat history -- a request routed
     * here from EcoRouter is a one-shot completion, not a UI conversation turn.
     */
    suspend fun generateOnce(prompt: String, maxTokens: Int): PhoneGenerationResult {
        if (!isModelLoaded()) throw IllegalStateException("no model loaded")
        if (!mutex.tryLock()) throw ServiceBusyException()
        try {
            val config = GenerationConfigSample(maxTokens = maxTokens).toGenerationConfig()
            val result = if (isLoadVlmModel) {
                generateVlmOnce(prompt, config)
            } else {
                generateLlmOnce(prompt, config)
            }
            requestCount += 1
            totalDecodeTokens += result.generatedTokens
            totalDecodeTimeS += result.decodeTimeMs / 1000.0
            return PhoneGenerationResult(
                text = result.text,
                modelId = loadedModelId,
                computeUnitIsNpu = lastComputeUnitIsNpu,
                ttftMs = result.ttftMs,
                promptTokens = result.promptTokens,
                prefillSpeedTokS = result.prefillSpeedTokS,
                generatedTokens = result.generatedTokens,
                decodeSpeedTokS = result.decodeSpeedTokS,
                latencyMs = result.promptTimeMs + result.decodeTimeMs,
            )
        } finally {
            mutex.unlock()
        }
    }

    private suspend fun generateLlmOnce(prompt: String, config: com.geniex.sdk.bean.GenerationConfig): GenResult {
        val wrapper = llmWrapper ?: throw IllegalStateException("no LLM model loaded")
        // Clears any KV-cache/session state left by a prior generateOnce() or interactive
        // chat turn -- otherwise the SDK's prompt-prefix cache can match an identical
        // routed prompt against its own previous response and return zero new tokens.
        wrapper.reset()
        var formattedText: String? = null
        var templateError: Throwable? = null
        wrapper.applyChatTemplate(arrayOf(ChatMessage(role = "user", prompt)), null, false)
            .onSuccess { formattedText = it.formattedText }
            .onFailure { templateError = it }
        val text = formattedText ?: throw (templateError ?: IllegalStateException("chat template failed"))

        val sb = StringBuilder()
        var completed: GenResult? = null
        wrapper.generateStreamFlow(text, config).collect { streamResult ->
            when (streamResult) {
                is LlmStreamResult.Token -> sb.append(streamResult.text)
                is LlmStreamResult.Completed -> {
                    val p = streamResult.profile
                    completed = GenResult(
                        text = sb.toString(),
                        ttftMs = p.ttftMs,
                        promptTokens = p.promptTokens.toInt(),
                        prefillSpeedTokS = p.prefillSpeed,
                        generatedTokens = p.generatedTokens.toInt(),
                        decodeSpeedTokS = p.decodingSpeed,
                        promptTimeMs = p.promptTimeMs,
                        decodeTimeMs = p.decodeTimeMs,
                    )
                }
                is LlmStreamResult.Error -> throw streamResult.throwable
            }
        }
        return completed ?: throw IllegalStateException("generation did not complete")
    }

    private suspend fun generateVlmOnce(prompt: String, config: com.geniex.sdk.bean.GenerationConfig): GenResult {
        val wrapper = vlmWrapper ?: throw IllegalStateException("no VLM model loaded")
        // See generateLlmOnce: clears leftover KV-cache/session state so a repeated
        // routed prompt is generated fresh instead of matching a cached prior response.
        wrapper.reset()
        val message = VlmChatMessage(role = "user", contents = listOf(VlmContent("text", prompt)))
        var formattedText: String? = null
        var templateError: Throwable? = null
        wrapper.applyChatTemplate(arrayOf(message), null, false)
            .onSuccess { formattedText = it.formattedText }
            .onFailure { templateError = it }
        val text = formattedText ?: throw (templateError ?: IllegalStateException("chat template failed"))

        val sb = StringBuilder()
        var completed: GenResult? = null
        wrapper.generateStreamFlow(text, config).collect { streamResult ->
            when (streamResult) {
                is LlmStreamResult.Token -> sb.append(streamResult.text)
                is LlmStreamResult.Completed -> {
                    val p = streamResult.profile
                    completed = GenResult(
                        text = sb.toString(),
                        ttftMs = p.ttftMs,
                        promptTokens = p.promptTokens.toInt(),
                        prefillSpeedTokS = p.prefillSpeed,
                        generatedTokens = p.generatedTokens.toInt(),
                        decodeSpeedTokS = p.decodingSpeed,
                        promptTimeMs = p.promptTimeMs,
                        decodeTimeMs = p.decodeTimeMs,
                    )
                }
                is LlmStreamResult.Error -> throw streamResult.throwable
            }
        }
        return completed ?: throw IllegalStateException("generation did not complete")
    }

    data class PhoneGenerationResult(
        val text: String,
        val modelId: String,
        val computeUnitIsNpu: Boolean,
        val ttftMs: Double,
        val promptTokens: Int,
        val prefillSpeedTokS: Double,
        val generatedTokens: Int,
        val decodeSpeedTokS: Double,
        val latencyMs: Double,
    )

    private data class BatterySnapshot(val charging: Boolean, val batteryPercent: Int?)

    private fun batterySnapshot(): BatterySnapshot {
        val intent = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val status = intent?.getIntExtra(BatteryManager.EXTRA_STATUS, -1) ?: -1
        val charging = status == BatteryManager.BATTERY_STATUS_CHARGING || status == BatteryManager.BATTERY_STATUS_FULL
        val level = intent?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
        val scale = intent?.getIntExtra(BatteryManager.EXTRA_SCALE, -1) ?: -1
        val percent = if (level >= 0 && scale > 0) (level * 100) / scale else null
        return BatterySnapshot(charging, percent)
    }

    // ---- HTTP server -----------------------------------------------------

    private inner class RouterHttpServer(port: Int) : NanoHTTPD(port) {
        override fun serve(session: IHTTPSession): Response =
            try {
                when {
                    session.method == Method.GET && session.uri == "/health" -> handleHealth()
                    session.method == Method.GET && session.uri == "/metrics" -> handleMetrics()
                    session.method == Method.POST && session.uri == "/v1/chat/completions" -> handleChatCompletions(session)
                    else -> jsonResponse(Response.Status.NOT_FOUND, JSONObject().put("error", "not found"))
                }
            } catch (e: Exception) {
                Log.e(TAG, "request failed", e)
                jsonResponse(Response.Status.INTERNAL_ERROR, JSONObject().put("error", e.message ?: "internal error"))
            }
    }

    private object TooManyRequestsStatus : NanoHTTPD.Response.IStatus {
        override fun getRequestStatus(): Int = 429
        override fun getDescription(): String = "429 Too Many Requests"
    }

    private fun jsonResponse(status: NanoHTTPD.Response.IStatus, json: JSONObject): NanoHTTPD.Response =
        NanoHTTPD.newFixedLengthResponse(status, "application/json", json.toString())

    private fun handleHealth(): NanoHTTPD.Response {
        val json = JSONObject()
            .put("status", if (isModelLoaded()) "healthy" else "idle")
            .put("model", if (isModelLoaded()) loadedModelId else JSONObject.NULL)
            .put("uptime_s", (System.currentTimeMillis() - startedAtMs) / 1000.0)
            .put("requests_served", requestCount)
        return jsonResponse(NanoHTTPD.Response.Status.OK, json)
    }

    private fun handleMetrics(): NanoHTTPD.Response {
        val json = JSONObject()
            .put("requests_served", requestCount)
            .put("total_decode_tokens", totalDecodeTokens)
            .put(
                "avg_decode_speed_tok_s",
                if (totalDecodeTimeS > 0) totalDecodeTokens / totalDecodeTimeS else JSONObject.NULL,
            )
        return jsonResponse(NanoHTTPD.Response.Status.OK, json)
    }

    private fun handleChatCompletions(session: NanoHTTPD.IHTTPSession): NanoHTTPD.Response {
        val authHeader = session.headers["authorization"]
        if (authHeader != "Bearer ${serverToken()}") {
            return jsonResponse(NanoHTTPD.Response.Status.UNAUTHORIZED, JSONObject().put("error", "invalid or missing bearer token"))
        }
        if (!isModelLoaded()) {
            return jsonResponse(NanoHTTPD.Response.Status.SERVICE_UNAVAILABLE, JSONObject().put("error", "no model loaded"))
        }

        val body = HashMap<String, String>()
        session.parseBody(body)
        val requestJson = JSONObject(body["postData"] ?: "{}")

        if (requestJson.optBoolean("stream", false)) {
            return jsonResponse(NanoHTTPD.Response.Status.BAD_REQUEST, JSONObject().put("error", "streaming is not supported"))
        }
        val prompt = extractPromptText(requestJson.optJSONArray("messages"))
            ?: return jsonResponse(NanoHTTPD.Response.Status.BAD_REQUEST, JSONObject().put("error", "messages must include user text content"))
        val maxTokens = requestJson.optInt("max_tokens", 512)

        val result = try {
            runBlocking { generateOnce(prompt, maxTokens) }
        } catch (e: ServiceBusyException) {
            return jsonResponse(TooManyRequestsStatus, JSONObject().put("error", e.message))
        } catch (e: Exception) {
            Log.e(TAG, "generation failed", e)
            return jsonResponse(NanoHTTPD.Response.Status.INTERNAL_ERROR, JSONObject().put("error", e.message ?: "generation failed"))
        }
        return jsonResponse(NanoHTTPD.Response.Status.OK, buildChatCompletionJson(result))
    }

    private fun extractPromptText(messages: JSONArray?): String? {
        if (messages == null || messages.length() == 0) return null
        val last = messages.getJSONObject(messages.length() - 1)
        val content = last.opt("content")
        return (content as? String)?.takeIf { it.isNotBlank() }
    }

    private fun buildChatCompletionJson(result: PhoneGenerationResult): JSONObject {
        val usage = JSONObject()
            .put("prompt_tokens", result.promptTokens)
            .put("completion_tokens", result.generatedTokens)
            .put("total_tokens", result.promptTokens + result.generatedTokens)

        val message = JSONObject().put("role", "assistant").put("content", result.text)
        val choice = JSONObject()
            .put("index", 0)
            .put("message", message)
            .put("finish_reason", "stop")
        val choices = JSONArray().put(choice)

        val battery = batterySnapshot()
        val powerMw = measuredNpuPowerMw(result.modelId)
        val energyAvailable = powerMw != null && result.computeUnitIsNpu && result.latencyMs > 0 && !battery.charging
        val energyMj = if (energyAvailable) powerMw!! * (result.latencyMs / 1000.0) else null
        val tokPerJoule = if (energyMj != null && energyMj > 0) result.generatedTokens / (energyMj / 1000.0) else null

        val phoneProfile = JSONObject()
            .put("ttft_ms", result.ttftMs)
            .put("prompt_tokens", result.promptTokens)
            .put("prefill_speed_tok_s", result.prefillSpeedTokS)
            .put("generated_tokens", result.generatedTokens)
            .put("decode_speed_tok_s", result.decodeSpeedTokS)
            .put("latency_ms", result.latencyMs)
            .put("compute_unit", if (result.computeUnitIsNpu) "npu" else "non-npu")
            .put("measured_power_mw", powerMw ?: JSONObject.NULL)
            .put("measured_energy_mj", energyMj ?: JSONObject.NULL)
            .put("tokens_per_joule", tokPerJoule ?: JSONObject.NULL)
            .put("energy_available", energyAvailable)
            .put("charging", battery.charging)
            .put("battery_percent", battery.batteryPercent ?: JSONObject.NULL)

        return JSONObject()
            .put("id", "chatcmpl-" + UUID.randomUUID().toString().replace("-", "").take(24))
            .put("object", "chat.completion")
            .put("created", System.currentTimeMillis() / 1000)
            .put("model", result.modelId)
            .put("choices", choices)
            .put("usage", usage)
            .put("phone_profile", phoneProfile)
    }

    // ---- foreground notification ------------------------------------

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return
        val channel = NotificationChannel(
            CHANNEL_ID,
            "EcoRouter phone server",
            NotificationManager.IMPORTANCE_LOW,
        ).apply {
            description = "Keeps the in-app inference server running for routed requests"
        }
        (getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager).createNotificationChannel(channel)
    }

    private fun buildNotification(): Notification =
        NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("EcoRouter phone server")
            .setContentText("Ready to serve routed requests on the LAN")
            .setSmallIcon(R.mipmap.ic_launcher)
            .setOngoing(true)
            .build()

    companion object {
        private const val TAG = "InferenceService"
        private const val CHANNEL_ID = "ecorouter_inference_service"
        private const val NOTIFICATION_ID = 42
        private const val PREFS_FILE = "ecorouter_server"
        private const val KEY_SERVER_TOKEN = "server_token"
        const val DEFAULT_PORT = 8080
    }
}

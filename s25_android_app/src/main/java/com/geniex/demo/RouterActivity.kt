// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo

import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.ServiceConnection
import android.os.Bundle
import android.os.IBinder
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Toast
import androidx.fragment.app.FragmentActivity
import com.geniex.demo.databinding.ActivityRouterBinding
import com.geniex.demo.router.CandidateEvaluation
import com.geniex.demo.router.CloudDispatchNotSupportedError
import com.geniex.demo.router.Device
import com.geniex.demo.router.PEQRouter
import com.geniex.demo.router.PEQRouterError
import com.geniex.demo.router.OptimizationProfile
import com.geniex.demo.router.RouteDecision
import com.geniex.demo.router.RouteRequest
import com.geniex.demo.router.Scenarios
import com.geniex.demo.router.XEliteRemoteExecutor
import com.geniex.demo.service.InferenceService
import com.geniex.demo.utils.inflate
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import java.util.Locale

/**
 * Accepts a prompt on the phone (the extension discussed in TODO.md: routing
 * with the origin *at* the phone, not just the phone as a dispatch target)
 * and runs [PEQRouter] locally to decide phone/PC/cloud, then dispatches to
 * whichever device wins:
 *  - phone: calls [InferenceService.generateOnce] directly, no network hop.
 *  - PC: POSTs to the X-Elite server address the user supplies (see
 *    [XEliteRemoteExecutor] for why this can't default the way the Python
 *    executor's `XELITE_SERVER_ENDPOINT` does).
 *  - cloud: deliberately not wired up yet -- see [CloudDispatchNotSupportedError].
 *
 * There is no on-device calibrated estimator (`peqrouter/estimator.py`'s
 * MiniLM/torch heads), so [PEQRouter] here always runs the static-capability
 * path -- same as the PC CLI's `--no-estimator`. Live telemetry collection
 * isn't implemented either (TODO.md), so telemetry is one of the same
 * built-in scenarios the PC CLI's `--scenario` flag offers.
 */
class RouterActivity : FragmentActivity() {

    private val binding by inflate<ActivityRouterBinding>()
    private val activityScope = CoroutineScope(Dispatchers.IO)

    private var inferenceService: InferenceService? = null
    private var serviceBound = false
    private val serviceConnection = object : ServiceConnection {
        override fun onServiceConnected(name: ComponentName?, binder: IBinder?) {
            inferenceService = (binder as InferenceService.LocalBinder).getService()
        }

        override fun onServiceDisconnected(name: ComponentName?) {
            inferenceService = null
        }
    }

    private var lastDecision: RouteDecision? = null
    private var lastPrompt: String = ""

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        bindService(Intent(this, InferenceService::class.java), serviceConnection, Context.BIND_AUTO_CREATE)
        serviceBound = true

        binding.spProfile.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            OptimizationProfile.entries.map { it.value },
        )
        val scenarioNames = Scenarios.builtInScenarios().keys.toList()
        binding.spScenario.adapter = ArrayAdapter(
            this,
            android.R.layout.simple_spinner_dropdown_item,
            scenarioNames,
        )

        binding.btnBack.setOnClickListener { finish() }
        binding.btnRoute.setOnClickListener { onRouteClicked() }
        binding.btnExecute.setOnClickListener { onExecuteClicked() }
    }

    override fun onDestroy() {
        if (serviceBound) {
            unbindService(serviceConnection)
            serviceBound = false
        }
        activityScope.coroutineContext[Job]?.cancel()
        super.onDestroy()
    }

    private fun onRouteClicked() {
        val prompt = binding.etPrompt.text?.toString()?.trim().orEmpty()
        if (prompt.isEmpty()) {
            Toast.makeText(this, "enter a prompt first", Toast.LENGTH_SHORT).show()
            return
        }
        val scenarioName = binding.spScenario.selectedItem as? String ?: "healthy"
        val profile = OptimizationProfile.entries[binding.spProfile.selectedItemPosition]
        val telemetry = Scenarios.builtInScenarios().getValue(scenarioName)

        binding.tvResult.text = ""
        binding.btnExecute.isEnabled = false
        lastDecision = null

        try {
            val request = RouteRequest(prompt, Device.PHONE, telemetry, profile)
            val decision = PEQRouter().route(request)
            lastDecision = decision
            lastPrompt = prompt
            binding.tvDecision.text = renderDecision(decision)
            binding.btnExecute.isEnabled = true
        } catch (error: PEQRouterError) {
            binding.tvDecision.text = "routing error: ${error.message}"
        }
    }

    private fun renderDecision(decision: RouteDecision): String {
        val analysis = decision.analysis
        val categories = if (analysis.piiCategories.isEmpty()) "none" else analysis.piiCategories.joinToString(", ")
        val sb = StringBuilder()
        sb.appendLine("Selected: ${decision.selectedDevice.value} / ${decision.modelId}")
        sb.appendLine("Profile: ${decision.profile.value}")
        sb.appendLine(
            "Prompt analysis: intent=${analysis.intent.value}, complexity=${"%.2f".format(Locale.US, analysis.complexity)}, " +
                "sensitive=${analysis.sensitive}, PII categories=$categories",
        )
        sb.appendLine("Quality degraded: ${decision.qualityDegraded}")
        sb.appendLine("Why: ${decision.explanation}")
        sb.appendLine("Candidates:")
        decision.candidates.forEach { candidate: CandidateEvaluation ->
            if (candidate.eligible) {
                sb.appendLine(
                    "  - ${candidate.device.value}/${candidate.modelId}: score=${"%.4f".format(Locale.US, candidate.score ?: 0.0)}, " +
                        "quality_ok=${candidate.qualitySufficient}, " +
                        "latency=${"%.1f".format(Locale.US, candidate.predictedLatencyMs ?: 0.0)}ms, " +
                        "energy=${"%.3f".format(Locale.US, candidate.predictedEnergyJoules ?: 0.0)}J",
                )
            } else {
                sb.appendLine("  - ${candidate.device.value}/${candidate.modelId}: excluded (${candidate.exclusionReasons.joinToString("; ")})")
            }
        }
        return sb.toString()
    }

    private fun onExecuteClicked() {
        val decision = lastDecision ?: return
        binding.btnExecute.isEnabled = false
        binding.pbExecuting.visibility = View.VISIBLE
        binding.tvResult.text = ""

        activityScope.launch {
            val outcome = try {
                when (decision.selectedDevice) {
                    Device.PHONE -> executeOnPhone(decision)
                    Device.PC -> executeOnPc(decision)
                    Device.CLOUD -> throw CloudDispatchNotSupportedError(
                        "live cloud dispatch from the phone is not implemented: it would require storing a " +
                            "Cirrascale API key on-device, which this app deliberately does not do. Run this " +
                            "prompt through the PC CLI's --live-cloud instead, or pick a scenario/profile that " +
                            "keeps the decision local.",
                    )
                }
            } catch (error: Exception) {
                "execution error: ${error.message}"
            }
            runOnUiThread {
                binding.pbExecuting.visibility = View.GONE
                binding.btnExecute.isEnabled = true
                binding.tvResult.text = outcome
            }
        }
    }

    private suspend fun executeOnPhone(decision: RouteDecision): String {
        val service = inferenceService ?: return "execution error: inference service not bound"
        if (!service.isModelLoaded()) {
            return "execution error: no model loaded on this phone -- load one from the main screen first"
        }
        return try {
            val result = service.generateOnce(lastPrompt, decision.analysis.estimatedOutputTokens)
            "Response (${result.modelId}, ${"%.1f".format(Locale.US, result.latencyMs)} ms, " +
                "${result.generatedTokens} tokens):\n${result.text}"
        } catch (busy: InferenceService.ServiceBusyException) {
            "execution error: phone is busy with another generation"
        } catch (error: Exception) {
            "execution error: ${error.message}"
        }
    }

    private suspend fun executeOnPc(decision: RouteDecision): String {
        val endpoint = binding.etPcEndpoint.text?.toString()?.trim().orEmpty()
        val result = XEliteRemoteExecutor.execute(endpoint, lastPrompt, decision.analysis.estimatedOutputTokens)
        return "Response (${result.modelId ?: decision.modelId}, ${"%.1f".format(Locale.US, result.apiTurnaroundLatencyMs)} ms):\n${result.text}"
    }
}

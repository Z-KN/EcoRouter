// ---------------------------------------------------------------------
// Copyright (c) 2026 Qualcomm Technologies, Inc. and/or its subsidiaries.
// SPDX-License-Identifier: BSD-3-Clause
// ---------------------------------------------------------------------
package com.geniex.demo

import android.Manifest
import android.app.Activity
import android.content.ComponentName
import android.content.Context
import android.content.DialogInterface
import android.content.Intent
import android.content.ServiceConnection
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.os.IBinder
import android.os.PowerManager
import android.provider.MediaStore
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.inputmethod.InputMethodManager
import android.widget.AdapterView
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.ImageButton
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.PopupWindow
import android.widget.ProgressBar
import android.widget.SimpleAdapter
import android.widget.Spinner
import android.widget.Switch
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import androidx.core.widget.doAfterTextChanged
import androidx.fragment.app.FragmentActivity
import androidx.recyclerview.widget.RecyclerView
import com.geniex.demo.bean.ModelData
import com.geniex.demo.bean.getSupportPluginIds
import com.geniex.demo.bean.isNpuModel
import com.geniex.demo.databinding.ActivityMainBinding
import com.geniex.demo.databinding.DialogSelectPluginIdBinding
import com.geniex.demo.listeners.CustomDialogInterface
import com.geniex.demo.service.InferenceService
import com.geniex.demo.utils.ExecShell
import com.geniex.demo.utils.ImgUtil
import com.geniex.demo.utils.inflate
import com.geniex.sdk.GenieXSdk
import com.geniex.sdk.LlmWrapper
import com.geniex.sdk.ModelManagerWrapper
import com.geniex.sdk.VlmWrapper
import com.geniex.sdk.bean.ChatMessage
import com.geniex.sdk.bean.ComputeUnitValue
import com.geniex.sdk.bean.HubSource
import com.geniex.sdk.bean.LlmCreateInput
import com.geniex.sdk.bean.LlmStreamResult
import com.geniex.sdk.bean.ModelConfig
import com.geniex.sdk.bean.ModelPullInput
import com.geniex.sdk.bean.VlmChatMessage
import com.geniex.sdk.bean.VlmContent
import com.geniex.sdk.bean.VlmCreateInput
import com.gyf.immersionbar.ktx.immersionBar
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.launch
import kotlinx.serialization.json.Json
import java.io.File
import java.io.FileNotFoundException
import java.io.FileOutputStream
import java.util.Locale

class MainActivity : FragmentActivity() {
    private val binding: ActivityMainBinding by inflate()
    private var downloadJob: Job? = null
    private var downloadingModelData: ModelData? = null
    private lateinit var llDownloading: LinearLayout
    private lateinit var tvDownloadProgress: TextView
    private lateinit var pbDownloading: ProgressBar
    private lateinit var spModelList: Spinner
    private lateinit var btnDownload: Button
    private lateinit var btnLoadModel: Button
    private lateinit var btnUnloadModel: Button
    private lateinit var btnStop: Button
    private lateinit var etInput: EditText
    private lateinit var btnSend: Button
    private lateinit var btnClearHistory: Button
    private lateinit var btnAddImage: Button

    private lateinit var recyclerView: RecyclerView
    private lateinit var adapter: ChatAdapter

    private lateinit var scrollImages: HorizontalScrollView
    private lateinit var topScrollContainer: LinearLayout
    private lateinit var llLoading: LinearLayout
    private lateinit var vTip: View

    private val modelScope = CoroutineScope(Dispatchers.IO)

    private val chatList = arrayListOf<ChatMessage>()
    private val vlmChatList = arrayListOf<VlmChatMessage>()
    private lateinit var modelList: List<ModelData>
    private var selectModelId = ""

    // The loaded model, load flags, and generation mutex all live in
    // InferenceService (not here) so that routed requests arriving over the
    // LAN server and this Activity's own chat both serialize through the
    // same native model handle -- see InferenceService's class doc.
    private var inferenceService: InferenceService? = null
    private var serviceBound = false
    private val serviceConnection =
        object : ServiceConnection {
            override fun onServiceConnected(
                name: ComponentName?,
                binder: IBinder?,
            ) {
                inferenceService = (binder as InferenceService.LocalBinder).getService()
                refreshServerStatusUi()
            }

            override fun onServiceDisconnected(name: ComponentName?) {
                inferenceService = null
            }
        }

    private lateinit var swServer: Switch
    private lateinit var tvServerStatus: TextView

    private var enableThinking = false
    private var isGenerating = false

    private val savedImageFiles = mutableListOf<File>()
    private val messages = arrayListOf<Message>()
    private var loadingMessageIndex: Int = -1

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        immersionBar {
            statusBarColorInt(Color.WHITE)
            statusBarDarkFont(true)
        }
        bindService(Intent(this, InferenceService::class.java), serviceConnection, Context.BIND_AUTO_CREATE)
        serviceBound = true
        initData()
        initView()
        setListeners()
    }

    override fun onDestroy() {
        if (serviceBound) {
            unbindService(serviceConnection)
            serviceBound = false
        }
        super.onDestroy()
    }

    private fun initView() {
        adapter = ChatAdapter(messages)
        binding.rvChat.adapter = adapter

        llDownloading = findViewById(R.id.ll_downloading)
        tvDownloadProgress = findViewById(R.id.tv_download_progress)
        pbDownloading = findViewById(R.id.pb_downloading)
        spModelList = findViewById(R.id.sp_model_list)
        spModelList.adapter =
            object : SimpleAdapter(
                this,
                modelList.map {
                    val map = mutableMapOf<String, String>()
                    map["displayName"] = it.displayName
                    map
                },
                R.layout.item_model,
                arrayOf("displayName"),
                intArrayOf(R.id.tv_model_id),
            ) {
            }
        spModelList.onItemSelectedListener =
            object : AdapterView.OnItemSelectedListener {
                override fun onItemSelected(
                    parent: AdapterView<*>?,
                    view: View?,
                    position: Int,
                    id: Long,
                ) {
                    selectModelId = modelList[position].id

                    messages.clear()
                    adapter.notifyDataSetChanged()
                    binding.rvChat.scrollTo(0, 0)
                }

                override fun onNothingSelected(parent: AdapterView<*>?) {
                    selectModelId = ""
                }
            }
        btnDownload = findViewById(R.id.btn_download)
        btnLoadModel = findViewById(R.id.btn_load_model)
        btnUnloadModel = findViewById(R.id.btn_unload_model)
        btnStop = findViewById(R.id.btn_stop)
        etInput = findViewById(R.id.et_input)
        btnAddImage = findViewById(R.id.btn_add_image)

        btnSend = findViewById(R.id.btn_send)
        btnSend.isEnabled = false
        etInput.doAfterTextChanged { refreshSendButtonState() }
        btnClearHistory = findViewById(R.id.btn_clear_history)
        scrollImages = findViewById(R.id.scroll_images)
        topScrollContainer = findViewById(R.id.ll_images_container)
        llLoading = findViewById(R.id.ll_loading)
        vTip = findViewById<View>(R.id.v_tip)
        swServer = findViewById(R.id.sw_server)
        tvServerStatus = findViewById(R.id.tv_server_status)

        findViewById<Button>(R.id.btn_test).setOnClickListener {
            Thread {
                val exeFile = File(filesDir, "geniex_test_llm")
                val chmodProcess = Runtime.getRuntime().exec("chmod 755 " + exeFile.absolutePath)
                chmodProcess.waitFor()
                Log.d(TAG, "exeFile exe? ${exeFile.canExecute()}")
                Log.d(TAG, "Exe Thread:${Thread.currentThread().name}")
                ExecShell()
                    .executeCommand(
                        arrayOf(
                            "cat",
                            "/sys/devices/soc0/sku",
                        ),
                    ).forEach {
                        Log.d(TAG, "cmd:$it")
                    }
            }.start()
        }

        findViewById<View>(R.id.v_tip).setOnClickListener {
            Toast.makeText(this, "please unload model first", Toast.LENGTH_SHORT).show()
        }
    }

    private fun parseModelList() {
        try {
            val baseJson = assets.open("model_list.json").bufferedReader().use { it.readText() }
            val json = Json { ignoreUnknownKeys = true }
            modelList = json.decodeFromString<List<ModelData>>(baseJson)
        } catch (e: Exception) {
            Log.e(TAG, "parseModelList: $e")
        }
    }

    /**
     * Step 0. Parse the model list and initialise the SDK. Model presence
     * is queried from the Rust model manager, not tracked client-side.
     */
    private fun initData() {
        parseModelList()
        initGenieXSdk()
    }

    /**
     * Step 1. initGenieXSdk environment
     */
    private fun initGenieXSdk() {
        GenieXSdk.getInstance().init(
            this,
            object : GenieXSdk.InitCallback {
                override fun onSuccess() {
                }

                override fun onFailure(reason: String) {
                    Log.e(TAG, "GenieXSdk init failed: $reason")
                }
            },
        )
    }

    private fun onLoadModelSuccess(tip: String) {
        runOnUiThread {
            Toast
                .makeText(
                    this@MainActivity,
                    tip,
                    Toast.LENGTH_SHORT,
                ).show()
            // change UI
            btnAddImage.visibility = View.INVISIBLE
            if (inferenceService?.isLoadVlmModel == true) {
                btnAddImage.visibility = View.VISIBLE
            }
            btnUnloadModel.visibility = View.VISIBLE
            llLoading.visibility = View.INVISIBLE
            btnStop.visibility = View.VISIBLE
            refreshSendButtonState()
        }
    }

    private fun onLoadModelFailed(tip: String) {
        runOnUiThread {
            vTip.visibility = View.GONE
            Toast.makeText(this@MainActivity, tip, Toast.LENGTH_SHORT).show()
            // change UI
            btnAddImage.visibility = View.INVISIBLE
            btnUnloadModel.visibility = View.GONE
            llLoading.visibility = View.INVISIBLE
        }
    }

    private fun hasLoadedModel(): Boolean = inferenceService?.isModelLoaded() ?: false

    /** Reflects the LAN server's running state (address, port, token) in the toggle row. */
    private fun refreshServerStatusUi() {
        val service = inferenceService ?: return
        runOnUiThread {
            swServer.isChecked = service.isServerRunning()
            tvServerStatus.text =
                if (service.isServerRunning()) {
                    val address = service.localAddress() ?: "unknown-ip"
                    "http://$address:${service.serverPort()}  token: ${service.serverToken()}"
                } else {
                    "Server off"
                }
        }
    }

    /**
     * Send is enabled only when (a) a model is loaded, (b) no inference
     * is in flight, and (c) there is something to send — text or an
     * attached image (VLM only).
     */
    private fun refreshSendButtonState() {
        runOnUiThread {
            val hasText = etInput.text?.isNotBlank() == true
            val hasAttachment = savedImageFiles.isNotEmpty()
            btnSend.isEnabled = hasLoadedModel() && !isGenerating && (hasText || hasAttachment)
        }
    }

    /**
     * Checks the Rust model manager's cache for [modelData]. Uses
     * `getPaths`, which canonicalises the name (so `ai-hub-models/<repo>`
     * and `qualcomm/<repo>` map to the same on-disk entry) and returns
     * null while the pull is still in `.inflight/`.
     */
    private suspend fun isModelDownloaded(modelData: ModelData): Boolean = ModelManagerWrapper.getPaths(modelData.modelName) != null

    private fun loadModel(
        selectModelData: ModelData,
        modelDataPluginId: String,
        nGpuLayers: Int,
        deviceId: String? = null,
    ) {
        // null deviceId only reaches here via the single-choice (qairt/NPU-only)
        // path -- see getSupportPluginIds() -- so absence of an explicit choice
        // means NPU, not "unknown".
        val computeUnitIsNpu = (deviceId ?: ComputeUnitValue.NPU.value) == ComputeUnitValue.NPU.value
        modelScope.launch {
            inferenceService?.clearLoaded()
            val paths = ModelManagerWrapper.getPaths(selectModelData.modelName)
            if (paths == null) {
                onLoadModelFailed("model paths unavailable — pull it first")
                return@launch
            }
            // Manifest-written runtime_id wins when present; fall back to
            // the user's UI selection for GGUF models that skip the manifest.
            val pluginId = paths.runtime_id.ifEmpty { modelDataPluginId }
            val resolvedDeviceId = deviceId
            when (selectModelData.type) {
                "chat", "llm" -> {
                    // QAIRT rejects non-zero n_ctx / n_gpu_layers (both fixed at compile
                    // time in the AI Hub bundle) — and the Kotlin ModelConfig defaults
                    // are non-zero, so zero them explicitly for the qairt path.
                    val isQairt = pluginId == "qairt"
                    val conf =
                        if (isQairt) {
                            ModelConfig(nCtx = 0, nGpuLayers = 0, enable_thinking = enableThinking)
                        } else {
                            ModelConfig(
                                nCtx = 1024,
                                nGpuLayers = nGpuLayers,
                                enable_thinking = enableThinking,
                            )
                        }
                    LlmWrapper
                        .builder()
                        .llmCreateInput(
                            LlmCreateInput(
                                model_name = paths.model_name,
                                model_path = paths.model_path,
                                tokenizer_path = paths.tokenizer_path,
                                config = conf,
                                runtime_id = pluginId,
                                compute_unit = resolvedDeviceId ?: ComputeUnitValue.NPU.value,
                            ),
                        ).build()
                        .onSuccess { wrapper ->
                            inferenceService?.setLoadedLlm(wrapper, selectModelData.id, computeUnitIsNpu)
                            onLoadModelSuccess("llm model loaded")
                        }.onFailure { error ->
                            onLoadModelFailed(error.message.toString())
                        }
                }

                "multimodal", "vlm" -> {
                    val isNpuVlm = pluginId == "qairt"
                    val config =
                        if (isNpuVlm) {
                            // QAIRT rejects non-zero n_ctx / n_gpu_layers for VLM too.
                            ModelConfig(nCtx = 0, nGpuLayers = 0, nThreads = 8, enable_thinking = enableThinking)
                        } else {
                            ModelConfig(
                                nCtx = 1024,
                                nThreads = 4,
                                nBatch = 1,
                                nUBatch = 1,
                                nGpuLayers = nGpuLayers,
                                enable_thinking = enableThinking,
                            )
                        }
                    VlmWrapper
                        .builder()
                        .vlmCreateInput(
                            VlmCreateInput(
                                model_name = paths.model_name,
                                model_path = paths.model_path,
                                mmproj_path = paths.mmproj_path,
                                config = config,
                                runtime_id = pluginId,
                                compute_unit = resolvedDeviceId ?: "HTP0",
                            ),
                        ).build()
                        .onSuccess {
                            inferenceService?.setLoadedVlm(it, selectModelData.id, computeUnitIsNpu)
                            onLoadModelSuccess("vlm model loaded")
                        }.onFailure { error ->
                            onLoadModelFailed(error.message.toString())
                        }
                }

                else -> {
                    onLoadModelFailed("model type error")
                }
            }
        }
    }

    private fun downloadModel(selectModelData: ModelData) {
        if (hasLoadedModel()) {
            Toast.makeText(this@MainActivity, "unload the current model first", Toast.LENGTH_SHORT).show()
            return
        }
        if (downloadJob?.isActive == true) {
            Toast
                .makeText(
                    this@MainActivity,
                    "${downloadingModelData?.displayName ?: "a model"} is already downloading",
                    Toast.LENGTH_SHORT,
                ).show()
            return
        }

        downloadingModelData = selectModelData
        llDownloading.visibility = View.VISIBLE
        tvDownloadProgress.text = "0%"

        val hub =
            runCatching { HubSource.valueOf(selectModelData.hub ?: "AUTO") }
                .getOrDefault(HubSource.AUTO)
        // AI Hub pulls route through chipset-matched assets. The Rust side
        // can auto-detect the host only on Windows-on-Snapdragon, so on
        // Android we must pass an explicit chipset for anything that ends
        // up on the AI Hub path — whether hub is AIHUB or AUTO + ai-hub-models/*
        // (or its canonical alias qualcomm/*).
        val name = selectModelData.modelName
        val isAiHubName =
            name.startsWith("ai-hub-models/", ignoreCase = true) ||
                name.startsWith("qualcomm/", ignoreCase = true)
        val willUseAiHub =
            hub == HubSource.AIHUB ||
                (hub == HubSource.AUTO && isAiHubName)
        if (willUseAiHub && selectModelData.chipset.isNullOrBlank()) {
            llDownloading.visibility = View.GONE
            Toast.makeText(this@MainActivity, "AI Hub models require a chipset. Update model_list.json.", Toast.LENGTH_SHORT).show()
            return
        }
        val input =
            ModelPullInput(
                model_name = selectModelData.modelName,
                precision = selectModelData.quant,
                hub = hub,
                local_path = selectModelData.localPath,
                chipset = selectModelData.chipset,
                display_name = selectModelData.aiHubDisplayName,
            )

        val wakeLock =
            (getSystemService(Context.POWER_SERVICE) as PowerManager)
                .newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "geniex:model_download")
        wakeLock.acquire()
        downloadJob =
            modelScope.launch {
                try {
                    // Short-circuit if already cached — the manager filters .inflight/
                    // models out of list(), so this only matches a complete pull.
                    if (isModelDownloaded(selectModelData)) {
                        runOnUiThread {
                            llDownloading.visibility = View.GONE
                            Toast.makeText(this@MainActivity, "model already downloaded", Toast.LENGTH_SHORT).show()
                        }
                        return@launch
                    }

                    ModelManagerWrapper.pullFlow(input).collect { event ->
                        when (event) {
                            is ModelManagerWrapper.PullEvent.Progress -> {
                                val total = event.files.sumOf { if (it.total_bytes > 0) it.total_bytes else 0L }
                                val done = event.files.sumOf { it.downloaded_bytes }
                                val percent = if (total > 0) ((done * 100) / total).toInt() else 0
                                runOnUiThread { tvDownloadProgress.text = "$percent%" }
                            }

                            is ModelManagerWrapper.PullEvent.Completed -> {
                                runOnUiThread {
                                    llDownloading.visibility = View.GONE
                                    Toast
                                        .makeText(
                                            this@MainActivity,
                                            "${selectModelData.displayName} downloaded",
                                            Toast.LENGTH_SHORT,
                                        ).show()
                                }
                            }

                            is ModelManagerWrapper.PullEvent.Error -> {
                                Log.e(TAG, "pull failed rc=${event.code}: ${event.message}")
                                runOnUiThread {
                                    llDownloading.visibility = View.GONE
                                    Toast
                                        .makeText(
                                            this@MainActivity,
                                            "Download failed. Please check your network connection and try again.",
                                            Toast.LENGTH_LONG,
                                        ).show()
                                }
                            }
                        }
                    }
                } finally {
                    if (wakeLock.isHeld) wakeLock.release()
                }
            }
    }

    private fun setListeners() {
        swServer.setOnCheckedChangeListener { _, checked ->
            val service = inferenceService
            if (service == null) {
                Toast.makeText(this@MainActivity, "inference service not bound", Toast.LENGTH_SHORT).show()
                swServer.isChecked = false
                return@setOnCheckedChangeListener
            }
            if (checked) {
                service.startHttpServer().onFailure { error ->
                    Toast.makeText(this@MainActivity, "could not start server: ${error.message}", Toast.LENGTH_LONG).show()
                    swServer.isChecked = false
                }
            } else {
                service.stopHttpServer()
            }
            refreshServerStatusUi()
        }

        btnAddImage.setOnClickListener {
            openGallery()
        }

        btnClearHistory.setOnClickListener {
            clearHistory()
        }
        /*
         * Step 3. download model. Cancelling the coroutine triggers the
         * flow's awaitClose which flips the Rust progress callback to
         * return false — partial files stay on disk for a resumed pull.
         * Use the Retry button to kick off a fresh pull that resumes.
         */
        binding.btnCancelDownload.setOnClickListener {
            downloadJob?.cancel()
            downloadJob = null
            tvDownloadProgress.text = "0%"
            binding.llDownloading.visibility = View.GONE
        }
        binding.btnRetryDownload.setOnClickListener {
            downloadJob?.cancel()
            downloadJob = null
            downloadingModelData?.let { downloadModel(it) }
        }
        btnDownload.setOnClickListener {
            if (downloadJob?.isActive == true) {
                if (downloadingModelData?.id == selectModelId) {
                    binding.llDownloading.visibility = View.VISIBLE
                } else {
                    Toast
                        .makeText(
                            this@MainActivity,
                            "${downloadingModelData?.displayName} is currently downloading.",
                            Toast.LENGTH_SHORT,
                        ).show()
                }
                return@setOnClickListener
            }
            val selectModelData = modelList.first { it.id == selectModelId }
            downloadModel(selectModelData)
        }
        /*
         * Step 4. load model
         */
        btnLoadModel.setOnClickListener {
            val selectModelData = modelList.first { it.id == selectModelId }
            Log.d(TAG, "current select model data:$selectModelData")
            if (hasLoadedModel()) {
                Toast.makeText(this@MainActivity, "please unload first", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            // Availability is checked against the manager's cache — a pull
            // that was cancelled mid-flight is not listed until it completes.
            modelScope.launch {
                if (!isModelDownloaded(selectModelData)) {
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "Model not downloaded — tap Download first.", Toast.LENGTH_LONG).show()
                    }
                    return@launch
                }
                runOnUiThread { startLoadModel(selectModelData) }
            }
        }

        /*
         * Step 5. send message
         */
        btnSend.setOnClickListener {
            if (!hasLoadedModel()) {
                Toast
                    .makeText(this@MainActivity, "please load model first", Toast.LENGTH_SHORT)
                    .show()
                return@setOnClickListener
            }
            // Guard against re-entry: a second click while a previous
            // generate() is still running would race on the native handle
            // and crash the app.
            if (isGenerating) return@setOnClickListener
            isGenerating = true
            refreshSendButtonState()

            if (savedImageFiles.isNotEmpty()) {
                messages.add(Message("", MessageType.IMAGES, savedImageFiles.map { it }))
                reloadRecycleView()
            }

            val inputString = etInput.text.trim().toString()
            etInput.setText("")
            etInput.clearFocus()
            val imm = getSystemService(Context.INPUT_METHOD_SERVICE) as InputMethodManager
            imm.hideSoftInputFromWindow(etInput.windowToken, 0)

            if (inputString.isNotEmpty()) {
                messages.add(Message(inputString, MessageType.USER))
                reloadRecycleView()
            }

            showLoadingIndicator()

            val supportFunctionCall = false
            var tools: String? = null
            if (supportFunctionCall) {
                // if this model support 'function call'
                tools =
                    "[{\"type\":\"function\",\"function\":{\"name\": \"campaign_investigation\",\"description\": \"Check campaign limits and determine appropriate action. If customer has reached limit, return a message (hardcoded or generated by model). If limit not reached, contact support.\",\"parameters\": {\"type\": \"object\", \"properties\":{\"campaign_name\":{\"type\": \"string\",\"description\": \"The name of the campaign to investigate\"}}, \"required\":[\"campaign_name\"]}}}]"
            }

            if (!hasLoadedModel()) {
                Toast.makeText(this@MainActivity, "model not loaded", Toast.LENGTH_SHORT).show()
                isGenerating = false
                refreshSendButtonState()
                return@setOnClickListener
            }

            modelScope.launch {
                val service = inferenceService
                if (service == null) {
                    runOnUiThread { Toast.makeText(this@MainActivity, "inference service not bound", Toast.LENGTH_SHORT).show() }
                    removeLoadingIndicator()
                    isGenerating = false
                    refreshSendButtonState()
                    return@launch
                }
                // Serialize with any in-flight routed request -- both paths
                // drive the same native model handle, see InferenceService.
                if (!service.mutex.tryLock()) {
                    runOnUiThread { Toast.makeText(this@MainActivity, "device is busy serving a routed request", Toast.LENGTH_SHORT).show() }
                    removeLoadingIndicator()
                    isGenerating = false
                    refreshSendButtonState()
                    return@launch
                }
                try {
                    val selectModelData = modelList.first { it.id == selectModelId }
                    val isNpu = ModelManagerWrapper.getPaths(selectModelData.modelName)?.runtime_id == "qairt"
                    Log.d(TAG, "isNpu: $isNpu")

                    val sb = StringBuilder()
                    if (service.isLoadVlmModel) {
                        val contents =
                            savedImageFiles
                                .map {
                                    VlmContent("image", it.absolutePath)
                                }.toMutableList()
                        contents.add(VlmContent("text", inputString))
                        clearImages()
                        val sendMsg = VlmChatMessage(role = "user", contents = contents)
                        vlmChatList.add(sendMsg)

                        Log.d(TAG, "before apply chat template:$vlmChatList")
                        service.vlmWrapper!!
                            .applyChatTemplate(vlmChatList.toTypedArray(), tools, enableThinking)
                            .onSuccess { result ->
                                Log.d(TAG, "vlm chat template:${result.formattedText}")
                                val baseConfig =
                                    GenerationConfigSample().toGenerationConfig()
                                // Only inject the current turn's media: SDK tokenizes
                                // incrementally, so re-passing history bitmaps breaks
                                // mtmd_tokenize (markers/bitmaps mismatch).
                                val configWithMedia =
                                    service.vlmWrapper!!.injectMediaPathsToConfig(
                                        arrayOf(sendMsg),
                                        baseConfig,
                                    )

                                Log.d(TAG, "Config has ${configWithMedia.imageCount} images")

                                service.vlmWrapper!!
                                    .generateStreamFlow(
                                        result.formattedText,
                                        configWithMedia,
                                    ).collect { handleResult(sb, it) }
                            }.onFailure {
                                runOnUiThread {
                                    Toast
                                        .makeText(
                                            this@MainActivity,
                                            it.message,
                                            Toast.LENGTH_SHORT,
                                        ).show()
                                }
                            }
                    } else {
                        chatList.add(ChatMessage(role = "user", inputString))
                        // Apply chat template and generate
                        service.llmWrapper!!
                            .applyChatTemplate(
                                chatList.toTypedArray(),
                                tools,
                                enableThinking,
                            ).onSuccess { templateOutput ->
                                Log.d(TAG, "chat template:${templateOutput.formattedText}")
                                service.llmWrapper!!
                                    .generateStreamFlow(
                                        templateOutput.formattedText,
                                        GenerationConfigSample().toGenerationConfig(),
                                    ).collect { streamResult ->
                                        handleResult(sb, streamResult)
                                    }
                            }.onFailure { error ->
                                runOnUiThread {
                                    Toast
                                        .makeText(
                                            this@MainActivity,
                                            error.message,
                                            Toast.LENGTH_SHORT,
                                        ).show()
                                }
                            }
                    }

                    clearImages()
                } finally {
                    service.mutex.unlock()
                    removeLoadingIndicator()
                    isGenerating = false
                    refreshSendButtonState()
                }
            }
        }

        /*
         * Step 6. others
         */
        btnUnloadModel.setOnClickListener {
            if (!hasLoadedModel()) {
                Toast.makeText(this@MainActivity, "model not loaded", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }
            // Unload model and cleanup
            val handleUnloadResult = fun(result: Int) {
                inferenceService?.clearLoaded()
                chatList.clear()
                vlmChatList.clear()
                runOnUiThread {
                    vTip.visibility = View.GONE
                    btnUnloadModel.visibility = View.GONE
                    btnStop.visibility = View.GONE
                    btnAddImage.visibility = View.INVISIBLE
                    messages.clear()
                    clearImages()
                    reloadRecycleView()
                    Toast
                        .makeText(
                            this@MainActivity,
                            if (result == 0) {
                                "unload success"
                            } else {
                                "unload failed and error code: $result"
                            },
                            Toast.LENGTH_SHORT,
                        ).show()
                    refreshSendButtonState()
                }
            }
            modelScope.launch {
                val service = inferenceService
                if (service == null) {
                    handleUnloadResult(0)
                    return@launch
                }
                // Wait for any in-flight generation (UI or routed) to finish
                // before tearing down the native handle it's using.
                service.mutex.lock()
                try {
                    if (service.isLoadVlmModel) {
                        service.vlmWrapper?.stopStream()
                        service.vlmWrapper?.destroy()
                        vlmChatList.clear()
                    } else if (service.isLoadLlmModel) {
                        service.llmWrapper?.stopStream()
                        service.llmWrapper?.destroy()
                        chatList.clear()
                    }
                } finally {
                    service.mutex.unlock()
                }
                handleUnloadResult(0)
            }
        }
        btnStop.setOnClickListener {
            if (!hasLoadedModel()) {
                Toast
                    .makeText(
                        this@MainActivity,
                        "model not loaded",
                        Toast.LENGTH_SHORT,
                    ).show()
                return@setOnClickListener
            }
            // Stop streaming
            modelScope.launch {
                val service = inferenceService ?: return@launch
                if (service.isLoadVlmModel) {
                    service.vlmWrapper?.stopStream()
                } else if (service.isLoadLlmModel) {
                    service.llmWrapper?.stopStream()
                }
            }
        }
    }

    private fun startLoadModel(selectModelData: ModelData) {
        vTip.visibility = View.VISIBLE
        llLoading.visibility = View.VISIBLE

        val supportPluginIds = selectModelData.getSupportPluginIds()
        Log.d(TAG, "support plugin_id:$supportPluginIds")
        var modelDataPluginId = "llama_cpp"
        var nGpuLayers = 0
        if (supportPluginIds.size > 1) {
            val dialogBinding = DialogSelectPluginIdBinding.inflate(layoutInflater)
            val isGgufLlmModel =
                !selectModelData.isNpuModel() &&
                    (selectModelData.type == "chat" || selectModelData.type == "llm")
            supportPluginIds.forEach {
                when (it) {
                    "cpu" -> {
                        dialogBinding.rbCpu.visibility = View.VISIBLE
                        dialogBinding.rbCpu.isChecked = true
                    }

                    "gpu" -> {
                        dialogBinding.rbGpu.visibility = View.VISIBLE
                    }

                    "npu" -> {
                        dialogBinding.rbNpu.visibility = View.VISIBLE
                        dialogBinding.rbNpu.isChecked = true
                    }
                }
            }
            if (isGgufLlmModel) {
                dialogBinding.rbNpu.visibility = View.VISIBLE
            }
            dialogBinding.rgSelectPluginId.setOnCheckedChangeListener { _, checkedId ->
                dialogBinding.llGpuLayers.visibility =
                    if (checkedId == R.id.rb_gpu) View.VISIBLE else View.GONE
            }

            val dialogOnClickListener =
                object : CustomDialogInterface.OnClickListener() {
                    override fun onClick(
                        dialog: DialogInterface?,
                        which: Int,
                    ) {
                        nGpuLayers = 0
                        var ggufLlmDeviceId: String? = null
                        val checkedId = dialogBinding.rgSelectPluginId.checkedRadioButtonId
                        if (checkedId == R.id.rb_gpu) {
                            if (dialogBinding.llGpuLayers.visibility == View.VISIBLE) {
                                nGpuLayers =
                                    dialogBinding.etGpuLayers.text
                                        .toString()
                                        .toInt()
                                if (nGpuLayers == 0) {
                                    Toast
                                        .makeText(
                                            this@MainActivity,
                                            "nGpuLayers min value is 1",
                                            Toast.LENGTH_SHORT,
                                        ).show()
                                    return
                                }
                            }
                            ggufLlmDeviceId = ComputeUnitValue.GPU.value
                        } else if (checkedId == R.id.rb_npu) {
                            nGpuLayers = 999
                            ggufLlmDeviceId = ComputeUnitValue.NPU.value
                        } else if (checkedId == R.id.rb_cpu) {
                            ggufLlmDeviceId = ComputeUnitValue.CPU.value
                        }
                        when (which) {
                            DialogInterface.BUTTON_POSITIVE -> {
                                dialog?.dismiss()
                                loadModel(selectModelData, modelDataPluginId, nGpuLayers, ggufLlmDeviceId)
                            }

                            DialogInterface.BUTTON_NEGATIVE -> {
                                llLoading.visibility = View.INVISIBLE
                                vTip.visibility = View.GONE
                            }
                        }
                    }
                }
            val alertDialog =
                AlertDialog
                    .Builder(this)
                    .setView(dialogBinding.root)
                    .setNegativeButton("Cancel", dialogOnClickListener)
                    .setPositiveButton("OK", dialogOnClickListener)
                    .setCancelable(false)
                    .create()
            alertDialog.show()
            dialogOnClickListener.resetPositiveButton(alertDialog)
        } else {
            if ("npu" == supportPluginIds[0]) {
                modelDataPluginId = "npu"
            }
            loadModel(selectModelData, modelDataPluginId, nGpuLayers)
        }
    }

    fun handleResult(
        sb: StringBuilder,
        streamResult: LlmStreamResult,
    ) {
        when (streamResult) {
            is LlmStreamResult.Token -> {
                removeLoadingIndicator()
                runOnUiThread {
                    sb.append(streamResult.text)
                    Message(sb.toString(), MessageType.ASSISTANT).let { lastMsg ->
                        val size = messages.size
                        messages[size - 1].let { msg ->
                            if (msg.type != MessageType.ASSISTANT) {
                                messages.add(lastMsg)
                            } else {
                                messages[size - 1] = lastMsg
                            }
                        }
                    }
                    adapter.notifyDataSetChanged()
                }
                Log.d(TAG, "Token: ${streamResult.text}")
            }

            is LlmStreamResult.Completed -> {
                removeLoadingIndicator()
                if (inferenceService?.isLoadVlmModel == true) {
                    vlmChatList.add(
                        VlmChatMessage(
                            "assistant",
                            listOf(VlmContent("text", sb.toString())),
                        ),
                    )
                } else {
                    chatList.add(ChatMessage("assistant", sb.toString()))
                }

                runOnUiThread {
                    val content = sb.toString()
                    val size = messages.size
                    messages[size - 1] = Message(content, MessageType.ASSISTANT)

                    val ttft = String.format(Locale.US, "%.2f", streamResult.profile.ttftMs)
                    val promptTokens = streamResult.profile.promptTokens
                    val prefillSpeed =
                        String.format(Locale.US, "%.2f", streamResult.profile.prefillSpeed)

                    val generatedTokens = streamResult.profile.generatedTokens
                    val decodingSpeed =
                        String.format(Locale.US, "%.2f", streamResult.profile.decodingSpeed)

                    // Latency = full response time (prefill + decode), matching what the
                    // user actually waited for. Energy = measured incremental NPU power x
                    // that latency. tok/J = generated tokens per joule of that energy --
                    // an efficiency figure that's comparable across models even though
                    // their power draw and speed both differ.
                    val latencyMs = streamResult.profile.promptTimeMs + streamResult.profile.decodeTimeMs
                    val service = inferenceService
                    val energyLine =
                        service
                            ?.let { it.measuredNpuPowerMw(it.loadedModelId) }
                            ?.takeIf { service?.lastComputeUnitIsNpu == true && latencyMs > 0 }
                            ?.let { powerMw ->
                                val energyMj = powerMw * (latencyMs / 1000.0)
                                val tokPerJoule = if (energyMj > 0) generatedTokens / (energyMj / 1000.0) else 0.0
                                "\nLatency: ${String.format(Locale.US, "%.1f", latencyMs)} ms; " +
                                    "Energy: ${String.format(Locale.US, "%.1f", energyMj)} mJ; " +
                                    "Efficiency: ${String.format(Locale.US, "%.1f", tokPerJoule)} tok/J"
                            } ?: ""

                    val profileData =
                        "TTFT: $ttft ms; Prompt Tokens: $promptTokens; \nPrefilling Speed: $prefillSpeed tok/s\nGenerated Tokens: $generatedTokens; Decoding Speed: $decodingSpeed tok/s$energyLine"
                    messages.add(
                        Message(
                            profileData,
                            MessageType.PROFILE,
                        ),
                    )
                    reloadRecycleView()
                }
                Log.d(TAG, "Completed: ${streamResult.profile}")
            }

            is LlmStreamResult.Error -> {
                removeLoadingIndicator()
                runOnUiThread {
                    val reason = streamResult.throwable.message ?: streamResult.throwable.toString()
                    messages.add(Message("Error: $reason", MessageType.PROFILE))
                    reloadRecycleView()
                }
                Log.d(TAG, "Error: $streamResult")
            }
        }
    }

    private fun openGallery() {
        val intent = Intent(Intent.ACTION_PICK, null)
        intent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*")
        startActivityForResult(intent, 1)
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray,
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)

        if (requestCode == 0) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openGallery()
            } else {
                Toast.makeText(this, "Not allow", Toast.LENGTH_SHORT).show()
            }
        } else if (requestCode == 2001) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            } else {
                Toast.makeText(this, "Camera not allow", Toast.LENGTH_SHORT).show()
            }
        }
    }

    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?,
    ) {
        super.onActivityResult(requestCode, resultCode, data)

        var bitmap: Bitmap? = null
        if (requestCode == 1) {
            if (resultCode == Activity.RESULT_OK && data != null) {
                val inputStream = contentResolver.openInputStream(data.data!!)
                bitmap = BitmapFactory.decodeStream(inputStream)
            }
        } else if (requestCode == 1001 && resultCode == Activity.RESULT_OK) {
            photoFile?.let {
                bitmap = BitmapFactory.decodeFile(it.absolutePath)
            }
        }

        bitmap?.let {
            try {
                val file = File(filesDir, "chat_${System.currentTimeMillis()}.jpg")
                val success = saveBitmapToFile(it, file)
                if (success) {
                    Log.d(TAG, "Save success: ${file.absolutePath}")
                    savedImageFiles.add(file)
                    refreshTopScrollContainer()
                } else {
                    Toast.makeText(this, "Save Image failed", Toast.LENGTH_SHORT).show()
                }
            } catch (e: FileNotFoundException) {
                Log.e(TAG, "save image failed", e)
            }
        }
    }

    private fun saveBitmapToFile(
        bitmap: Bitmap,
        file: File,
    ): Boolean =
        try {
            val tempDir = File(this.filesDir, "tmp").apply { if (!exists()) mkdirs() }

            val tempFile =
                File(
                    tempDir,
                    "tmp_${System.currentTimeMillis()}.jpg",
                )
            FileOutputStream(tempFile).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out)
            }

            val outFile =
                File(
                    tempDir,
                    "out_${System.currentTimeMillis()}.jpg",
                )
            ImgUtil.squareCrop(
                ImgUtil.downscaleAndSave(
                    imageFile = tempFile,
                    outFile = outFile,
                    maxSize = 448,
                    format = Bitmap.CompressFormat.JPEG,
                    quality = 90,
                ),
                file,
                448,
            )
            true
        } catch (e: Exception) {
            Log.e(TAG, "saveBitmapToFile failed", e)
            false
        }

    private fun clearHistory() {
        val service = inferenceService
        if (service?.isLoadLlmModel == true) {
            chatList.clear()
            modelScope.launch {
                service.llmWrapper?.reset()
            }
        }
        if (service?.isLoadVlmModel == true) {
            vlmChatList.clear()
            modelScope.launch {
                service.vlmWrapper?.reset()
            }
        }
        messages.clear()
        clearImages()
        reloadRecycleView()
    }

    private var popupWindow: PopupWindow? = null

    private fun showPopupMenu(anchorView: View) {
        if (popupWindow?.isShowing == true) {
            popupWindow?.dismiss()
            return
        }

        val popupView = LayoutInflater.from(this).inflate(R.layout.menu_layout, null)

        popupWindow =
            PopupWindow(
                popupView,
                anchorView.width * 2,
                android.view.ViewGroup.LayoutParams.WRAP_CONTENT,
                true,
            )

        popupWindow?.isOutsideTouchable = true
        popupWindow?.elevation = 10f

        val btnCamera = popupView.findViewById<Button>(R.id.btn_camera)
        val btnPhoto = popupView.findViewById<Button>(R.id.btn_photo)

        btnCamera.setOnClickListener {
            popupWindow?.dismiss()
            checkAndOpenCamera()
        }
        btnPhoto.setOnClickListener {
            popupWindow?.dismiss()
            openGallery()
        }

        popupView.measure(
            View.MeasureSpec.UNSPECIFIED,
            View.MeasureSpec.UNSPECIFIED,
        )
        val popupHeight = popupView.measuredHeight
        popupWindow?.showAsDropDown(anchorView, 0, -anchorView.height - popupHeight)
    }

    private var photoUri: Uri? = null
    private var photoFile: File? = null

    private fun checkAndOpenCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                2001,
            )
        } else {
            openCamera()
        }
    }

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        photoFile =
            File(
                getExternalFilesDir(Environment.DIRECTORY_PICTURES),
                "photo_${System.currentTimeMillis()}.jpg",
            )
        photoUri =
            FileProvider.getUriForFile(
                this,
                "${applicationContext.packageName}.fileprovider",
                photoFile!!,
            )

        intent.putExtra(MediaStore.EXTRA_OUTPUT, photoUri)
        intent.addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        startActivityForResult(intent, 1001)
    }

    private fun clearImages() {
        savedImageFiles.clear()
        refreshTopScrollContainer()
    }

    private fun refreshTopScrollContainer() {
        refreshSendButtonState()
        runOnUiThread {
            topScrollContainer.removeAllViews()
            if (savedImageFiles.isEmpty()) {
                scrollImages.visibility = View.GONE
                return@runOnUiThread
            }

            scrollImages.visibility = View.VISIBLE

            for (file in savedImageFiles) {
                val itemView =
                    LayoutInflater
                        .from(this)
                        .inflate(R.layout.item_image_scroll, topScrollContainer, false)
                val ivImage = itemView.findViewById<ImageView>(R.id.iv_image)
                val btnRemove = itemView.findViewById<ImageButton>(R.id.btn_remove)

                ivImage.setImageURI(Uri.fromFile(file))

                btnRemove.setOnClickListener {
                    savedImageFiles.remove(file)
                    refreshTopScrollContainer()
                }
                topScrollContainer.addView(itemView)
            }
        }
    }

    private fun reloadRecycleView() {
        adapter.notifyDataSetChanged()
        binding.rvChat.scrollToPosition(messages.size - 1)
    }

    private fun showLoadingIndicator() {
        runOnUiThread {
            if (loadingMessageIndex >= 0) return@runOnUiThread
            messages.add(Message("", MessageType.LOADING))
            loadingMessageIndex = messages.size - 1
            reloadRecycleView()
        }
    }

    private fun removeLoadingIndicator() {
        runOnUiThread {
            val idx = loadingMessageIndex
            if (idx < 0 || idx >= messages.size) {
                loadingMessageIndex = -1
                return@runOnUiThread
            }
            if (messages[idx].type == MessageType.LOADING) {
                messages.removeAt(idx)
                adapter.notifyItemRemoved(idx)
            }
            loadingMessageIndex = -1
        }
    }

    companion object {
        private const val TAG = "GenieXDemo"
    }
}

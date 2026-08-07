/**
 * Static example data for the EcoRouter dashboard.
 *
 * Each entry has the exact shape EcoRouter's Python API already produces:
 * `ExecutionResult.to_dict()` from ecorouter/models.py (decision + response +
 * metrics). These are hand-authored, not captured from a live run, but every
 * number was derived using the same formulas as ecorouter/router.py against
 * the real "healthy" telemetry constants in ecorouter/scenarios.py (phone
 * 94.22 tok/s @ 0.0412 J/token, PC 20.19 tok/s @ 0.4447 J/token, cloud
 * 33.37 tok/s @ 2.2475 J/token, $0.03 / 1k cloud tokens) — so the ranking
 * and scores are self-consistent with the router's actual policy.
 *
 * When live routing is wired up, replace SCENARIOS with the parsed JSON body
 * of a POST to whatever endpoint wraps `EcoRouter.run()` (see dashboard/README.md).
 */

const SCENARIOS = [
  {
    id: "lookup",
    tabLabel: "Simple lookup",
    prompt: "What is 15 percent of 240?",
    origin: "phone",
    scenario: "healthy",
    result: {
      decision: {
        selected_device: "phone",
        model_id: "Qwen3-0.6B",
        profile: "balanced",
        analysis: {
          intent: "reasoning",
          complexity: 0.12,
          sensitive: false,
          pii_categories: [],
          estimated_input_tokens: 9,
          estimated_output_tokens: 40,
          required_quality: 0.45,
        },
        quality_degraded: false,
        explanation:
          "Selected phone/Qwen3-0.6B: it had the lowest balanced score among privacy-safe, healthy models meeting required quality.",
        predicted: { latency_ms: 520.06, energy_joules: 2.019, cloud_cost_usd: 0.0 },
        candidates: [
          {
            device: "phone",
            model_id: "Qwen3-0.6B",
            eligible: true,
            exclusion_reasons: [],
            quality_sufficient: true,
            predicted_latency_ms: 520.06,
            predicted_energy_joules: 2.019,
            predicted_cloud_cost_usd: 0.0,
            score: 0.093763,
            penalties: { latency: 0.052006, energy: 0.002884, quality: 0.4 },
          },
          {
            device: "pc",
            model_id: "Qwen3-VL-4B-Instruct",
            eligible: true,
            exclusion_reasons: [],
            quality_sufficient: true,
            predicted_latency_ms: 2444.95,
            predicted_energy_joules: 21.79,
            predicted_cloud_cost_usd: 0.0,
            score: 0.166013,
            penalties: { latency: 0.244495, energy: 0.031129, quality: 0.2 },
          },
          {
            device: "cloud",
            model_id: "Llama-3.3-70B",
            eligible: true,
            exclusion_reasons: [],
            quality_sufficient: true,
            predicted_latency_ms: 1468.4,
            predicted_energy_joules: 110.13,
            predicted_cloud_cost_usd: 0.00147,
            score: 0.13416,
            penalties: { latency: 0.14684, energy: 0.157328, quality: 0.05 },
          },
        ],
      },
      response: "15% of 240 is 36.",
      metrics: {
        api_turnaround_latency_ms: 612.4,
        prompt_tokens: 9,
        completion_tokens: 14,
        total_tokens: 23,
        measured_energy_joules: 0.87,
        estimated_energy_joules: 0.948,
        energy_joules_per_token: 0.0412,
        energy_estimate_method: "measured_npu_power_x_latency_minus_idle_baseline",
        energy_scope:
          "measured whole-device battery discharge during decode, NPU-only, per-model calibration; excludes Wi-Fi radio energy for the request itself; only valid while the phone is unplugged",
        confidence: "measured",
        ttft_ms: 180.2,
        prefill_speed_tokens_per_second: 120.5,
        decode_speed_tokens_per_second: 95.3,
        tokens_per_joule: 26.4,
        compute_unit: "NPU",
        backend: null,
      },
    },
  },

  {
    id: "reasoning",
    tabLabel: "Multi-step reasoning",
    prompt:
      "A train leaves at 2:15 PM travelling 80 km/h. A second train leaves the same station at 3:00 PM travelling 100 km/h in the same direction. At what time does the second train catch the first?",
    origin: "pc",
    scenario: "healthy",
    result: {
      decision: {
        selected_device: "cloud",
        model_id: "Llama-3.3-70B",
        profile: "balanced",
        analysis: {
          intent: "reasoning",
          complexity: 0.62,
          sensitive: false,
          pii_categories: [],
          estimated_input_tokens: 55,
          estimated_output_tokens: 180,
          required_quality: 0.85,
        },
        quality_degraded: false,
        explanation:
          "Selected cloud/Llama-3.3-70B: it had the lowest balanced score among privacy-safe, healthy models meeting required quality.",
        predicted: { latency_ms: 7042.6, energy_joules: 528.2, cloud_cost_usd: 0.00705 },
        candidates: [
          {
            device: "phone",
            model_id: "Qwen3-0.6B",
            eligible: true,
            exclusion_reasons: [],
            quality_sufficient: false,
            predicted_latency_ms: 2502.3,
            predicted_energy_joules: 9.682,
            predicted_cloud_cost_usd: 0.0,
            score: 0.196521,
            penalties: { latency: 0.25023, energy: 0.013831, quality: 0.4 },
          },
          {
            device: "pc",
            model_id: "Qwen3-VL-4B-Instruct",
            eligible: true,
            exclusion_reasons: [],
            quality_sufficient: false,
            predicted_latency_ms: 11639.4,
            predicted_energy_joules: 104.505,
            predicted_cloud_cost_usd: 0.0,
            score: 0.583117,
            penalties: { latency: 1.0, energy: 0.149292, quality: 0.2 },
          },
          {
            device: "cloud",
            model_id: "Llama-3.3-70B",
            eligible: true,
            exclusion_reasons: [],
            quality_sufficient: true,
            predicted_latency_ms: 7042.6,
            predicted_energy_joules: 528.2,
            predicted_cloud_cost_usd: 0.00705,
            score: 0.611752,
            penalties: { latency: 0.70426, energy: 0.754571, quality: 0.05 },
          },
        ],
      },
      response:
        "Train A has a 45-minute head start, covering 60 km before Train B departs. Train B closes the gap at 20 km/h (100 - 80). Closing 60 km at 20 km/h takes 3 hours, so Train B catches Train A at 3:00 PM + 3h = 6:00 PM.",
      metrics: {
        api_turnaround_latency_ms: 6890.4,
        prompt_tokens: 55,
        completion_tokens: 162,
        total_tokens: 217,
        measured_energy_joules: 516.78,
        estimated_energy_joules: 487.71,
        energy_joules_per_token: 2.2475,
        energy_estimate_method: "cloud_accelerator_tdp_x_latency",
        energy_scope:
          "Qualcomm Cloud AI 100 rated TDP (75 W) x wall-clock request latency; not an on-device power measurement -- includes network and queueing time and does not account for multi-tenant sharing of the accelerator",
        confidence: "measured",
        ttft_ms: null,
        prefill_speed_tokens_per_second: null,
        decode_speed_tokens_per_second: null,
        tokens_per_joule: null,
        compute_unit: null,
        backend: null,
      },
    },
  },

  {
    id: "privacy",
    tabLabel: "Privacy-sensitive",
    prompt: "Summarize the medical history for John Smith, SSN 123-45-6789.",
    origin: "pc",
    scenario: "healthy",
    result: {
      decision: {
        selected_device: "pc",
        model_id: "Qwen3-VL-4B-Instruct",
        profile: "high-quality",
        analysis: {
          intent: "summarization",
          complexity: 0.35,
          sensitive: true,
          pii_categories: ["PERSON", "US_SSN"],
          estimated_input_tokens: 14,
          estimated_output_tokens: 120,
          required_quality: 0.55,
        },
        quality_degraded: false,
        explanation:
          "Selected pc/Qwen3-VL-4B-Instruct: it had the lowest high-quality score among privacy-safe, healthy models meeting required quality.",
        predicted: { latency_ms: 6636.0, energy_joules: 59.59, cloud_cost_usd: 0.0 },
        candidates: [
          {
            device: "phone",
            model_id: "Qwen3-0.6B",
            eligible: true,
            exclusion_reasons: [],
            quality_sufficient: true,
            predicted_latency_ms: 1430.2,
            predicted_energy_joules: 5.521,
            predicted_cloud_cost_usd: 0.0,
            score: 0.335091,
            penalties: { latency: 0.14302, energy: 0.007887, quality: 0.4 },
          },
          {
            device: "pc",
            model_id: "Qwen3-VL-4B-Instruct",
            eligible: true,
            exclusion_reasons: [],
            quality_sufficient: true,
            predicted_latency_ms: 6636.0,
            predicted_energy_joules: 59.59,
            predicted_cloud_cost_usd: 0.0,
            score: 0.234873,
            penalties: { latency: 0.6636, energy: 0.085129, quality: 0.2 },
          },
          {
            device: "cloud",
            model_id: "Llama-3.3-70B",
            eligible: false,
            exclusion_reasons: ["cloud blocked by privacy policy"],
            quality_sufficient: true,
            predicted_latency_ms: 4016.2,
            predicted_energy_joules: 301.22,
            predicted_cloud_cost_usd: 0.00402,
            score: 0.123193,
            penalties: { latency: 0.40162, energy: 0.430314, quality: 0.05 },
          },
        ],
      },
      response:
        "Summary (SSN omitted for privacy): patient John Smith presented for a routine follow-up. Vitals stable, no new diagnoses; continue the current medication regimen and schedule a follow-up in 6 months.",
      metrics: {
        api_turnaround_latency_ms: 7120.5,
        prompt_tokens: 14,
        completion_tokens: 98,
        total_tokens: 112,
        measured_energy_joules: 42.7,
        estimated_energy_joules: 49.81,
        energy_joules_per_token: 0.4447,
        energy_estimate_method: "measured_npu_power_x_latency_minus_idle_baseline",
        energy_scope:
          "measured whole-laptop battery discharge during sustained NPU-serving load minus an idle baseline, per-model calibration; covers prefill + decode; excludes display/background baseline drift; only valid while the laptop is unplugged",
        confidence: "measured",
        ttft_ms: 310.4,
        prefill_speed_tokens_per_second: 48.2,
        decode_speed_tokens_per_second: 18.6,
        tokens_per_joule: 2.62,
        compute_unit: "NPU",
        backend: "QNN",
      },
    },
  },
];

# EcoRouter Backlog

This is the canonical list of deliberate MVP simplifications and deferred work. Each item has
a priority, target version, dependency, and measurable completion condition. Features completed
in the current implementation are documented in the README instead of duplicated here.

## Known MVP simplifications

- [ ] **P1 | v2 | Replace text-only prompt analysis.** Dependency: select multimodal input
  contracts. Complete when the router accepts typed image and audio inputs without breaking
  text requests.
- [ ] **P1 | v1.5 | Replace caller-supplied telemetry.** Dependency: device-agent protocols.
  Complete when phone, PC, and cloud snapshots can be collected and timestamped automatically.
- [ ] **P1 | v1.5 | Calibrate static capabilities and scoring weights.** Dependency: benchmark
  corpus and target hardware. Per-device energy/throughput constants and the profile weights are
  now derived from the 106-prompt calibration sweep (README's
  [Energy honesty](README.md#energy-honesty) and
  [Profile weights and calibration results](README.md#profile-weights-and-calibration-results));
  remaining: the static per-device `capability_score` (0.60/0.80/0.95) is still an illustrative
  fallback, only superseded per-prompt when the calibrated estimator is trusted. Complete when
  that fallback number is also derived from repeatable measurements rather than a guess.
- [ ] **P1 | v1.5 | Replace approximate token, latency, energy, and cost predictions.**
  Dependency: runtime observations. Complete when predictions are validated against held-out
  measurements and error metrics are reported.

## Next version

- [ ] **P1 | v1.4 | Evaluate and calibrate the regex privacy heuristic.** Dependency:
  representative privacy test corpus. Complete when person/address/secret precision and recall
  are measured against it and known false positives/negatives are documented. An NLP-backed
  analyzer (e.g. Presidio) remains an option for a future version if regex recall proves
  insufficient once real usage data exists.
- [ ] **P2 | v1.5 | Support configurable and multilingual NLP models.** Dependency: target
  language requirements and model benchmarks. Complete when model/language selection is
  configuration-driven and each supported language has integration coverage.
- [ ] **P0 | v1.4 | Connect live telemetry collectors.** Dependency: phone, PC, and cloud device
  agents. Complete when stale/unreachable states and current performance metrics flow into a
  `RouteRequest` without hand-authored JSON.
- [ ] **P1 | v1.4 | Add Cirrascale streaming and cancellation.** Dependency: stable Imagine SDK
  streaming behavior. Complete when live cloud runs can stream, cancel an in-flight request,
  and pass offline adapter tests for interrupted streams.
- [ ] **P0 | v1.4 | Add provider-side energy telemetry.** Dependency: provider or controlled
  server access to NVML, CPU RAPL, and node power data. Complete when request-attributed measured
  joules, idle-baseline treatment, concurrent-work attribution, and measurement scope are emitted
  separately from estimates.
- [ ] **P1 | v1.5 | Calibrate separate prefill and decode energy coefficients.** Dependency:
  benchmark runs on identified hardware. Phone/PC now have a real, measured single J/token
  constant each (see README's [Energy honesty](README.md#energy-honesty)); complete when separate
  input-token/prefill and output-token/decode coefficients replace that single figure and
  held-out estimation error is reported.
- [ ] **P1 | v1.5 | Benchmark and calibrate the policy.** Dependency: telemetry and real
  executors. Complete when thresholds and all four profiles are backed by a versioned benchmark
  report.
- [ ] **P1 | v1.4 | Add OCR and image PII detection.** Dependency: selected on-device OCR
  runtime. Complete when sensitive text in reference images blocks unsafe cloud routing.
- [ ] **P1 | v1.5 | Add FastAPI and WebSocket endpoints.** Dependency: stable device-agent
  protocol. Complete when clients can submit requests, stream routing events, and receive
  structured results through authenticated endpoints.

## Later versions

- [ ] **P2 | v2 | Add audio and general multimodal ingestion.** Dependency: transcription and
  media preprocessing adapters. Complete when mixed text/image/audio requests share one typed
  decomposition pipeline.
- [ ] **P2 | v2 | Add privacy-preserving redaction before cloud routing.** Dependency: evaluated
  redaction policy. Complete when users can opt in and tests prove detected values do not leave
  local devices.
- [ ] **P2 | v2 | Learn latency and energy predictions.** Dependency: consented observation
  logging and sufficient benchmark data. Complete when an MLP or GBDT beats the heuristic
  baseline on held-out data and has a safe fallback.
- [ ] **P2 | v2 | Add a quality-sufficiency classifier and feedback loop.** Dependency: labeled
  results and feedback policy. Complete when calibration and false-sufficiency rates are
  measured and monitored.
- [ ] **P1 | v2 | Build the real-time PC dashboard.** Dependency: streaming API. Complete when
  it visualizes candidate gates, scores, selected routes, latency, energy, battery impact, and
  cloud cost for live requests.
- [ ] **P0 | production | Add production security and operations.** Dependency: deployment
  architecture and threat model. Complete when authentication, encrypted transport, access
  controls, privacy-safe audit records, monitoring, and deployment packaging pass a security
  review.

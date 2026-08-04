# EcoRouter Backlog

This is the canonical list of deliberate MVP simplifications and deferred work. Each item has
a priority, target version, dependency, and measurable completion condition.

## Known MVP simplifications

- [ ] **P1 · v1.1 — Replace text-only prompt analysis.** Dependency: select multimodal input
  contracts. Complete when the router accepts typed image and audio inputs without breaking
  text requests.
- [ ] **P1 · v1.1 — Replace caller-supplied telemetry.** Dependency: device-agent protocols.
  Complete when phone, PC, and cloud snapshots can be collected and timestamped automatically.
- [ ] **P1 · v1.1 — Calibrate static capabilities and scoring weights.** Dependency: benchmark
  corpus and target hardware. Complete when configuration values are derived from repeatable
  measurements rather than illustrative defaults.
- [ ] **P1 · v1.1 — Replace approximate token, latency, energy, and cost estimates.** Dependency:
  runtime observations. Complete when estimates are validated against held-out measurements
  and error metrics are reported.
- [ ] **P0 · v1.1 — Replace simulated execution.** Dependency: runtime-specific model adapters.
  Complete when each destination can execute a routed request and return a real response.

## Next version

- [ ] **P0 · v1.1 — Connect live telemetry collectors.** Dependency: phone, PC, and cloud device
  agents. Complete when stale/unreachable states and current performance metrics flow into a
  `RouteRequest` without hand-authored JSON.
- [ ] **P0 · v1.1 — Add real model-runtime adapters.** Dependency: deployed model endpoints and
  credentials. Complete when executors support timeouts, cancellation, and sanitized errors on
  all three destinations.
- [ ] **P1 · v1.1 — Benchmark and calibrate the policy.** Dependency: telemetry and real
  executors. Complete when thresholds and all four profiles are backed by a versioned benchmark
  report.
- [ ] **P1 · v1.1 — Add OCR and image PII detection.** Dependency: selected on-device OCR
  runtime. Complete when sensitive text in reference images blocks unsafe cloud routing.
- [ ] **P1 · v1.1 — Add FastAPI and WebSocket endpoints.** Dependency: stable device-agent
  protocol. Complete when clients can submit requests, stream routing events, and receive
  structured results through authenticated endpoints.

## Later versions

- [ ] **P2 · v2 — Add audio and general multimodal ingestion.** Dependency: transcription and
  media preprocessing adapters. Complete when mixed text/image/audio requests share one typed
  decomposition pipeline.
- [ ] **P2 · v2 — Add privacy-preserving redaction before cloud routing.** Dependency: evaluated
  redaction policy. Complete when users can opt in and tests prove detected values do not leave
  local devices.
- [ ] **P2 · v2 — Learn latency and energy predictions.** Dependency: consented observation
  logging and sufficient benchmark data. Complete when an MLP or GBDT beats the heuristic
  baseline on held-out data and has a safe fallback.
- [ ] **P2 · v2 — Add a quality-sufficiency classifier and feedback loop.** Dependency: labeled
  results and feedback policy. Complete when calibration and false-sufficiency rates are
  measured and monitored.
- [ ] **P1 · v2 — Build the real-time PC dashboard.** Dependency: streaming API. Complete when
  it visualizes candidate gates, scores, selected routes, latency, energy, battery impact, and
  cloud cost for live requests.
- [ ] **P0 · production — Add production security and operations.** Dependency: deployment
  architecture and threat model. Complete when authentication, encrypted transport, access
  controls, privacy-safe audit records, monitoring, and deployment packaging pass a security
  review.

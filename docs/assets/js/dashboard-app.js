/**
 * Renders SCENARIOS (see dashboard-scenarios.js) into the dashboard shell
 * embedded in slide 3 of docs/index.html.
 *
 * Adapted from dashboard-static/app.js: identical rendering logic, minus the
 * standalone page's light/dark theme toggle (this deck is dark-only), and
 * the scenario-tab click handler now preserves the deck's #slide-N hash
 * instead of overwriting it.
 *
 * Everything here reads from one object shaped like
 * `ExecutionResult.to_dict()` (peqrouter/models.py). To wire up live routing
 * later: replace `renderScenario(SCENARIOS[i])` calls with
 * `renderScenario(await postToPEQRouter(promptText))`.
 */

const DEVICE_LABELS = { phone: "Phone", pc: "PC", cloud: "Cloud" };
const DEVICE_ORDER = ["phone", "pc", "cloud"];

const requestedScenario = parseInt(new URLSearchParams(location.search).get("scenario"), 10);
let activeIndex = SCENARIOS[requestedScenario] ? requestedScenario : 0;

function fmtNumber(value, digits = 0) {
  if (value === null || value === undefined) return "—";
  return Number(value).toLocaleString(undefined, {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function fmtMs(value) {
  if (value === null || value === undefined) return "—";
  return value >= 1000 ? `${fmtNumber(value / 1000, 2)} s` : `${fmtNumber(value, 0)} ms`;
}

function fmtJoules(value) {
  if (value === null || value === undefined) return "—";
  return `${fmtNumber(value, value < 10 ? 3 : 1)} J`;
}

function fmtUsd(value) {
  if (value === null || value === undefined) return "—";
  return `$${Number(value).toFixed(value < 0.01 ? 6 : 4)}`;
}

function capitalize(word) {
  return word.charAt(0).toUpperCase() + word.slice(1);
}

function statTile(label, value, sub) {
  const tile = document.createElement("div");
  tile.className = "stat-tile";
  tile.innerHTML = `
    <div class="stat-label">${label}</div>
    <div class="stat-value">${value}</div>
    ${sub ? `<div class="stat-sub">${sub}</div>` : ""}
  `;
  return tile;
}

function meterTile(label, fraction, valueText, deviceVar = "--series-phone") {
  const pct = Math.round(Math.max(0, Math.min(1, fraction)) * 100);
  const tile = document.createElement("div");
  tile.className = "stat-tile";
  tile.innerHTML = `
    <div class="stat-label">${label}</div>
    <div class="stat-value">${valueText}</div>
    <div class="meter-track"><div class="meter-fill" style="width:${pct}%; background: var(${deviceVar});"></div></div>
  `;
  return tile;
}

function statusBadgeHtml(kind, label) {
  return `<span class="status-badge status-${kind}"><span class="dot"></span>${label}</span>`;
}

function statusTile(label, kind, badgeLabel) {
  const tile = document.createElement("div");
  tile.className = "stat-tile";
  tile.innerHTML = `
    <div class="stat-label">${label}</div>
    <div class="stat-value">${statusBadgeHtml(kind, badgeLabel)}</div>
  `;
  return tile;
}

function renderScenarioTabs() {
  const wrap = document.getElementById("scenarioTabs");
  wrap.innerHTML = "";
  SCENARIOS.forEach((scenario, index) => {
    const btn = document.createElement("button");
    btn.className = "scenario-tab";
    btn.type = "button";
    btn.role = "tab";
    btn.setAttribute("aria-selected", index === activeIndex ? "true" : "false");
    btn.textContent = scenario.tabLabel;
    btn.addEventListener("click", () => {
      activeIndex = index;
      history.replaceState(null, "", `?scenario=${index}${location.hash}`);
      renderScenarioTabs();
      renderScenario(SCENARIOS[activeIndex]);
    });
    wrap.appendChild(btn);
  });
}

function renderPrompt(scenario) {
  const originChip = document.getElementById("originChip");
  originChip.textContent = `Origin: ${DEVICE_LABELS[scenario.origin]}`;
  originChip.dataset.device = scenario.origin;

  document.getElementById("profileChip").textContent = `Profile: ${scenario.result.decision.profile}`;
  document.getElementById("promptText").textContent = scenario.prompt;
}

function renderFlow(scenario) {
  const decision = scenario.result.decision;
  const selected = decision.selected_device;
  const svg = document.getElementById("flowSvg");

  document.getElementById("hubProfile").textContent = decision.profile;

  DEVICE_ORDER.forEach((device) => {
    const candidate = decision.candidates.find((c) => c.device === device);
    const modelLabel = document.getElementById(`${device}ModelLabel`);
    modelLabel.textContent = candidate.model_id;

    const path = svg.querySelector(`.flow-path[data-device="${device}"]`);
    const node = svg.querySelector(`.flow-node-device[data-device="${device}"]`);
    const isSelected = device === selected;
    const isBlocked = !candidate.eligible;

    path.classList.toggle("active", isSelected);
    path.classList.toggle("blocked", isBlocked && !isSelected);
    node.classList.toggle("selected", isSelected);
    node.classList.toggle("blocked", isBlocked && !isSelected);
  });

  document.getElementById("pathIn").classList.add("active");
  document.querySelector(".flow-node-hub").classList.add("active");

  // Traveling "packet" dots along the input path and the selected output path,
  // using SVG's own <animateMotion>/<mpath> so no animation loop is needed in JS.
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const packetIn = document.getElementById("packetIn");
  const packetOut = document.getElementById("packetOut");
  packetIn.innerHTML = "";
  packetOut.innerHTML = "";
  if (!reduceMotion) {
    packetIn.innerHTML = `
      <circle class="packet packet-in">
        <animateMotion dur="1.4s" repeatCount="indefinite">
          <mpath href="#pathIn"></mpath>
        </animateMotion>
      </circle>`;
    packetOut.innerHTML = `
      <circle class="packet packet-${selected}">
        <animateMotion dur="1.4s" repeatCount="indefinite" begin="0.2s">
          <mpath href="#path${capitalize(selected)}"></mpath>
        </animateMotion>
      </circle>`;
  }
}

function renderParams(scenario) {
  const analysis = scenario.result.decision.analysis;
  const decision = scenario.result.decision;
  const grid = document.getElementById("statGrid");
  grid.innerHTML = "";

  grid.appendChild(statTile("Intent", capitalize(analysis.intent)));
  grid.appendChild(
    meterTile("Complexity", analysis.complexity, analysis.complexity.toFixed(2), `--series-${decision.selected_device}`)
  );
  grid.appendChild(
    meterTile(
      "Required quality",
      analysis.required_quality,
      analysis.required_quality.toFixed(2),
      `--series-${decision.selected_device}`
    )
  );
  grid.appendChild(
    analysis.sensitive
      ? statusTile("Privacy", "critical", "Sensitive — cloud excluded")
      : statusTile("Privacy", "good", "Not sensitive")
  );
  grid.appendChild(statTile("Est. input tokens", fmtNumber(analysis.estimated_input_tokens)));
  grid.appendChild(statTile("Est. output tokens", fmtNumber(analysis.estimated_output_tokens)));
  grid.appendChild(
    decision.quality_degraded
      ? statusTile("Quality gate", "warning", "Degraded — no device met the floor")
      : statusTile("Quality gate", "good", "Requirement met")
  );

  const piiTile = document.createElement("div");
  piiTile.className = "stat-tile";
  piiTile.innerHTML = `
    <div class="stat-label">PII categories</div>
    <div class="pii-chips">
      ${
        analysis.pii_categories.length
          ? analysis.pii_categories.map((cat) => `<span class="pii-chip">${cat}</span>`).join("")
          : `<span class="stat-sub">None detected</span>`
      }
    </div>
  `;
  grid.appendChild(piiTile);

  document.getElementById("explanationText").textContent = decision.explanation;

  const body = document.getElementById("candidatesBody");
  body.innerHTML = "";
  DEVICE_ORDER.forEach((device) => {
    const candidate = decision.candidates.find((c) => c.device === device);
    const isSelected = device === decision.selected_device;
    const tr = document.createElement("tr");
    tr.className = [!candidate.eligible ? "ineligible" : "", isSelected ? "selected" : ""].join(" ").trim();
    if (isSelected) tr.style.setProperty("--row-accent", `var(--series-${device})`);

    let statusHtml;
    if (!candidate.eligible) {
      statusHtml = `Blocked<span class="reason-note">${candidate.exclusion_reasons.join(", ")}</span>`;
    } else if (!candidate.quality_sufficient) {
      statusHtml = `<span class="reason-note">Below quality floor</span>`;
    } else {
      statusHtml = isSelected ? "Selected" : "Eligible";
    }

    tr.innerHTML = `
      <td><span class="device-cell"><span class="device-dot" style="background: var(--series-${device});"></span>${DEVICE_LABELS[device]}</span></td>
      <td>${candidate.model_id}</td>
      <td>${statusHtml}</td>
      <td>${candidate.score === null ? "—" : candidate.score.toFixed(4)}</td>
      <td>${fmtMs(candidate.predicted_latency_ms)}</td>
      <td>${fmtJoules(candidate.predicted_energy_joules)}</td>
      <td>${candidate.predicted_cloud_cost_usd ? fmtUsd(candidate.predicted_cloud_cost_usd) : "—"}</td>
    `;
    body.appendChild(tr);
  });
}

function renderResponse(scenario) {
  const decision = scenario.result.decision;
  const header = document.getElementById("responseHeader");
  header.innerHTML = `
    <span class="device-dot" style="display:inline-block;width:10px;height:10px;border-radius:50%;background:var(--series-${decision.selected_device});"></span>
    ${DEVICE_LABELS[decision.selected_device]} / ${decision.model_id}
  `;
  document.getElementById("responseText").textContent = scenario.result.response;
}

function renderMetrics(scenario) {
  const decision = scenario.result.decision;
  const metrics = scenario.result.metrics;
  const primaryGrid = document.getElementById("primaryMetricsGrid");
  const grid = document.getElementById("metricsGrid");
  primaryGrid.innerHTML = "";
  grid.innerHTML = "";

  if (!metrics) {
    primaryGrid.appendChild(statTile("Latency", "—", "Awaiting live result"));
    primaryGrid.appendChild(statTile("Energy", "—", "Awaiting live result"));
    return;
  }

  primaryGrid.appendChild(
    statTile("Latency", fmtMs(metrics.api_turnaround_latency_ms))
  );
  const energyValue = metrics.measured_energy_joules ?? metrics.estimated_energy_joules;
  primaryGrid.appendChild(statTile("Energy", fmtJoules(energyValue)));
  grid.appendChild(
    statTile("Tokens", fmtNumber(metrics.total_tokens), `${fmtNumber(metrics.prompt_tokens)} prompt · ${fmtNumber(metrics.completion_tokens)} completion`)
  );
  grid.appendChild(
    metrics.confidence === "measured"
      ? statusTile("Confidence", "good", "Measured")
      : statusTile("Confidence", "warning", "Uncalibrated estimate")
  );
  if (decision.predicted.cloud_cost_usd) {
    grid.appendChild(statTile("Cloud cost", fmtUsd(decision.predicted.cloud_cost_usd)));
  }
  if (metrics.tokens_per_joule !== null) {
    grid.appendChild(statTile("Efficiency", `${fmtNumber(metrics.tokens_per_joule, 1)} tok/J`));
  }
  if (metrics.ttft_ms !== null) {
    grid.appendChild(statTile("Time to first token", fmtMs(metrics.ttft_ms)));
  }
  if (metrics.decode_speed_tokens_per_second !== null) {
    grid.appendChild(statTile("Decode speed", `${fmtNumber(metrics.decode_speed_tokens_per_second, 1)} tok/s`));
  }
  if (metrics.compute_unit) {
    grid.appendChild(statTile("Compute unit", metrics.compute_unit, metrics.backend || undefined));
  }
}

function renderScenario(scenario) {
  renderPrompt(scenario);
  renderFlow(scenario);
  renderParams(scenario);
  renderResponse(scenario);
  renderMetrics(scenario);
}

// Wires the "Try your own prompt" box to POST /api/route (see docs/server.py)
// and render the real response through the same renderScenario() pipeline
// the canned SCENARIOS use — the shape is identical because both are
// ExecutionResult.to_dict() (peqrouter/models.py). Fails gracefully when no
// local server is reachable (e.g. the hosted GitHub Pages copy of this deck).
function setupLiveInput() {
  const input = document.getElementById("liveInput");
  const button = document.getElementById("liveRouteBtn");
  const originSelect = document.getElementById("liveOrigin");
  const profileSelect = document.getElementById("liveProfile");
  const badge = document.getElementById("routingLinkStatus");

  function setLinkStatus(linked) {
    badge.className = `status-badge ${linked ? "status-good" : "status-warning"}`;
    badge.innerHTML = `<span class="dot"></span>Routing ${linked ? "linked" : "not linked"}`;
  }

  // Same-path probe as a real route call, but with an empty body so it never
  // reaches router.run() — the real backend always answers with JSON (a 400
  // here), while a static host (e.g. GitHub Pages) 404s with an HTML page.
  async function checkLink() {
    try {
      const response = await fetch("/api/route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
      });
      const contentType = response.headers.get("content-type") || "";
      setLinkStatus(contentType.includes("application/json"));
    } catch (error) {
      setLinkStatus(false);
    }
  }

  async function submit() {
    const prompt = input.value.trim();
    if (!prompt) return;

    input.disabled = true;
    button.disabled = true;
    button.textContent = "Routing…";

    const origin = originSelect.value;
    const profile = profileSelect.value;

    try {
      const response = await fetch("/api/route", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, origin, profile }),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error((body && body.error) || `HTTP ${response.status}`);
      }
      setLinkStatus(true);
      activeIndex = -1;
      renderScenarioTabs();
      renderScenario({ prompt, origin, result: body });
    } catch (error) {
      setLinkStatus(false);
    } finally {
      input.disabled = false;
      button.disabled = false;
      button.textContent = "Route";
    }
  }

  button.addEventListener("click", submit);
  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") submit();
  });

  checkLink();
}

renderScenarioTabs();
renderScenario(SCENARIOS[activeIndex]);
setupLiveInput();

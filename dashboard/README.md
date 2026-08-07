# EcoRouter dashboard (static)

A plain HTML/CSS/JS visualization of one EcoRouter routing decision:
prompt → EcoRouter → phone/PC/cloud, the analysis parameters behind the pick,
the response, and its execution metrics.

No build step. Open `index.html` directly in a browser, or serve the folder:

```
python3 -m http.server 8000 --directory dashboard
```

## Files

- `index.html` — page structure (prompt panel, flow diagram, parameters,
  response, metrics).
- `styles.css` — palette (light/dark), layout, and the flow animation.
- `scenarios.js` — the example data, shaped exactly like
  `ExecutionResult.to_dict()` in `ecorouter/models.py`. Three hand-authored
  scenarios (each derived using the real formulas in `ecorouter/router.py`
  against the "healthy" telemetry constants in `ecorouter/scenarios.py`) show
  phone, PC, and cloud each winning once, including a privacy-blocked case.
- `app.js` — reads one scenario object and renders every panel from it. No
  panel hardcodes a value that's in the JSON — this is what makes it easy to
  swap in live data later.

## Making it live

The scenario picker in the header is the seam. Right now it does:

```js
activeIndex = index;
renderScenario(SCENARIOS[activeIndex]);
```

To go live, add a small backend that wraps `EcoRouter.run()` and returns
`result.to_dict()` as JSON — e.g. a FastAPI/Flask endpoint:

```python
from ecorouter.router import EcoRouter
from ecorouter.models import RouteRequest, Device, OptimizationProfile
from ecorouter.executors import default_simulated_executors
from ecorouter.scenarios import built_in_scenarios

@app.post("/api/route")
def route(prompt: str, origin: str, profile: str = "balanced"):
    request = RouteRequest(
        prompt=prompt,
        origin=Device(origin),
        telemetry=built_in_scenarios()["healthy"],
        profile=OptimizationProfile(profile),
    )
    result = EcoRouter().run(request, default_simulated_executors())
    return result.to_dict()
```

Then in `app.js`, replace the disabled `#liveInput`/`Route` button with a
handler that calls it and feeds the response straight into the existing
`renderScenario()`:

```js
const body = await fetch("/api/route", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt, origin, profile }),
}).then((r) => r.json());

renderScenario({ prompt, origin, result: body });
```

Nothing else in `app.js`, `styles.css`, or `index.html` needs to change —
`renderScenario()` only ever reads the `decision`/`response`/`metrics` shape,
which is exactly what `to_dict()` already produces. The same wiring works for
`--live-phone`/`--live-pc`/`--live-cloud`, since `metrics` is `null` for a
simulated leg and populated for a live one; `renderMetrics()` already handles
both.

Image-prompt input has a placeholder slot in the prompt panel — enabling it
depends on the multimodal input work tracked in the repo's `TODO.md`.

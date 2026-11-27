"use strict";

// Gateway page: models with prices and live health, read-only settings, route preview, and demo
// traffic controls. All data text goes in through textContent or createTextNode.

const REFRESH_MS = 5000;
const DEMO_POLL_MS = 1000;
const SCENARIO_PLANS = {
  smoke: "smoke: 0.5 requests/s for 40 s",
  normal: "normal: 2/s for 60 s",
  spike: "spike: 2/s for 20 s, then 10/s for 20 s, then 2/s for 20 s",
};
const HEALTH_CLASSES = { Healthy: "healthy", Degraded: "degraded", "No recent traffic": "idle" };

const state = {
  minutes: 60,
  live: false,
  liveTimer: null,
  busy: false,
  config: null,
  demoTimer: null,
  demoStatus: null,
};

const byId = (id) => document.getElementById(id);

// ---------- helpers ----------

function appendChildren(node, children) {
  for (const child of [].concat(children)) {
    if (child === null || child === undefined) continue;
    node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return node;
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return appendChildren(node, children);
}

const cell = (text) => el("td", {}, [text]);
const isMissing = (value) => value === null || value === undefined || Number.isNaN(value);

function formatMs(value) {
  if (isMissing(value)) return "-";
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}

const formatMoney = (value) => (isMissing(value) ? "-" : `$${value.toFixed(6)}`);
const formatPrice = (value) => (isMissing(value) ? "-" : `$${value}`);
const formatPercent = (value) => (isMissing(value) ? "-" : `${value.toFixed(1)}%`);
const formatCount = (value) => (isMissing(value) ? "-" : String(value));
const formatTokens = (value) => (isMissing(value) ? "-" : value.toFixed(0));

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { Accept: "application/json", "Content-Type": "application/json" }, ...options });
  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    body = null;
  }
  if (!response.ok) {
    let type = "error";
    let message = response.statusText;
    if (body && body.error) {
      type = body.error.type || type;
      message = body.error.message || message;
    } else if (body && body.detail) {
      type = "validation_error";
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
    const error = new Error(`HTTP ${response.status} | ${type}: ${message}`);
    error.status = response.status;
    throw error;
  }
  return body;
}

function showError(message) {
  const box = byId("error");
  box.textContent = message || "";
  box.hidden = !message;
}

function definitionList(target, rows) {
  target.replaceChildren(...rows.flatMap(([label, value]) => [el("dt", {}, [label]), el("dd", {}, [String(value)])]));
}

// ---------- models ----------

function healthCell(word) {
  const dotClass = HEALTH_CLASSES[word] || "idle";
  return el("td", {}, [el("span", { class: `dot ${dotClass}`, "aria-hidden": "true" }), word]);
}

function renderModels(health) {
  const prices = new Map((state.config ? state.config.models : []).map((model) => [model.name, model]));
  const rows = health.models.map((entry) => {
    const model = prices.get(entry.name) || {};
    return el("tr", {}, [
      el("th", { scope: "row" }, [entry.name]),
      cell(model.provider || "-"), cell(model.model_id || "-"),
      cell(formatPrice(model.input_price_per_million_tokens)),
      cell(formatPrice(model.output_price_per_million_tokens)),
      cell(formatCount(entry.attempts)), cell(entry.attempts ? formatPercent(entry.error_rate_percent) : "-"),
      cell(formatCount(entry.rate_limit_errors)), cell(formatMs(entry.p50_ms)), cell(formatMs(entry.p95_ms)),
      cell(formatTokens(entry.avg_output_tokens)), cell(formatMoney(entry.cost_usd)), healthCell(entry.health),
    ]);
  });
  byId("table-models").tBodies[0].replaceChildren(...rows);
}

async function refreshModels() {
  if (state.busy) return;
  state.busy = true;
  try {
    const health = await api(`/api/gateway/models?minutes=${state.minutes}`);
    showError("");
    renderModels(health);
    byId("updated").textContent = `Updated ${new Date().toLocaleTimeString([], { hour12: false })}`;
  } catch (error) {
    showError(error.message);
  } finally {
    state.busy = false;
  }
}

function setLive(on) {
  state.live = on;
  const button = byId("live");
  button.setAttribute("aria-pressed", String(on));
  button.textContent = on ? "Live: on" : "Live: off";
  window.clearInterval(state.liveTimer);
  state.liveTimer = on ? window.setInterval(refreshModels, REFRESH_MS) : null;
}

// ---------- settings ----------

function ruleRows(routing) {
  return [
    ["long_question", `more than ${routing.medium_words} words: +1; more than ${routing.long_words} words: +2`],
    ["keywords", `contains any of: ${routing.keywords.join(", ")}: +${routing.keywords_points} (counted once)`],
    ["many_numbers", `${routing.min_numbers} or more numbers: +${routing.many_numbers_points}`],
    ["multiple_questions", `${routing.min_question_marks} or more question marks: +${routing.multiple_questions_points}`],
  ];
}

function renderSettings(config) {
  definitionList(byId("settings-gateway"), [
    ["Mode", config.mode],
    ["Default model", config.default_model_name],
    ["Max output tokens", config.max_output_tokens],
    ["Temperature", config.temperature],
    ["Timeout", `${config.request_timeout_seconds} s`],
    ["Database file", config.database_file],
  ]);
  const routing = config.routing;
  definitionList(byId("settings-routing"), [
    ["Small model", routing.small_model],
    ["Big model", routing.big_model],
    ["Threshold", `score ${routing.threshold} or more goes to ${routing.big_model}`],
    ["Fallback", routing.fallback_enabled ? "on" : "off"],
    ["Fallback model", routing.fallback_model],
    ["Fallback on cut-off", routing.fallback_on_truncated ? "yes" : "no"],
  ]);
  byId("table-rules").tBodies[0].replaceChildren(...ruleRows(routing).map(([name, words]) =>
    el("tr", {}, [el("th", { scope: "row" }, [name]), cell(words)])));
}

// ---------- route preview ----------

function renderPreview(result) {
  const hits = result.hits.length
    ? el("div", { class: "table-wrap" }, [el("table", {}, [
      el("thead", {}, [el("tr", {}, ["Rule", "Points", "Detail"].map((label) => el("th", { scope: "col" }, [label])))]),
      el("tbody", {}, result.hits.map((hit) => el("tr", {}, [cell(hit.rule), cell(`+${hit.points}`), cell(hit.detail || "-")]))),
    ])])
    : el("p", { class: "muted" }, ["No rule fired."]);
  const notFired = result.not_fired.length ? result.not_fired.join(", ") : "none";
  byId("preview-result").replaceChildren(
    el("p", { class: "chosen" }, [`Goes to: ${result.model}`]),
    el("p", {}, [`score ${result.score} (threshold ${result.threshold})`]),
    hits,
    el("p", { class: "muted" }, [`Rules that did not fire: ${notFired}`]),
  );
}

async function onPreview(event) {
  event.preventDefault();
  const question = byId("preview-input").value;
  const button = byId("preview-button");
  button.disabled = true;
  try {
    renderPreview(await api("/api/gateway/route-preview", { method: "POST", body: JSON.stringify({ question }) }));
  } catch (error) {
    byId("preview-result").replaceChildren(el("p", { class: "error" }, [error.message]));
  } finally {
    button.disabled = false;
  }
}

// ---------- demo traffic ----------

function fillSelect(select, items, selected) {
  select.replaceChildren(...items.map(([value, label]) => el("option", { value }, [label])));
  if (items.some(([value]) => value === selected)) select.value = selected;
}

function showDemoMessage(message) {
  const box = byId("demo-message");
  box.textContent = message || "";
  box.hidden = !message;
}

function renderDemoStatus(status) {
  state.demoStatus = status;
  const running = status.state === "running";
  byId("demo-availability").textContent = status.available
    ? "Demo traffic is available: the gateway is in fake mode and uses a 'fake' database."
    : `Demo traffic is not available. ${status.reason}`;
  for (const id of ["demo-scenario", "demo-model", "demo-questions", "demo-start"]) {
    byId(id).disabled = !status.available || running;
  }
  byId("demo-stop").disabled = !status.available || !running;
  const bar = byId("demo-bar");
  bar.max = Math.max(status.planned, 1);
  bar.value = status.completed;
  const plan = status.scenario ? `${status.scenario}, model ${status.model || "(gateway default)"}, ${status.questions_file}` : "no run yet";
  byId("demo-counts").textContent = `State: ${status.state} (${plan}) | planned ${status.planned} | sent ${status.sent} | `
    + `completed ${status.completed} | ok ${status.ok} | errors ${status.errors} | ${status.elapsed_seconds.toFixed(1)} s`;
  if (running && !state.demoTimer) state.demoTimer = window.setInterval(pollDemo, DEMO_POLL_MS);
  if (!running && state.demoTimer) {
    window.clearInterval(state.demoTimer);
    state.demoTimer = null;
  }
}

async function pollDemo() {
  try {
    renderDemoStatus(await api("/api/demo/status"));
  } catch (error) {
    showDemoMessage(error.message);
  }
}

async function loadDemo() {
  try {
    const options = await api("/api/demo/options");
    fillSelect(byId("demo-scenario"), options.scenarios.map((name) => [name, SCENARIO_PLANS[name] || name]), "smoke");
    fillSelect(byId("demo-model"), [["", "(gateway default)"], ...options.models.map((name) => [name, name])], "");
    fillSelect(byId("demo-questions"), options.questions_files.map((name) => [name, name]), "load_questions.txt");
    byId("demo-note").textContent = `Requests are written to the database the gateway is using (${options.database_file}). `
      + "Open Observability with Live on to watch them.";
    renderDemoStatus(await api("/api/demo/status"));
  } catch (error) {
    showDemoMessage(error.message);
  }
}

async function onDemoStart(event) {
  event.preventDefault();
  showDemoMessage("");
  const body = { scenario: byId("demo-scenario").value, questions_file: byId("demo-questions").value || null };
  if (byId("demo-model").value) body.model = byId("demo-model").value;
  try {
    renderDemoStatus(await api("/api/demo/start", { method: "POST", body: JSON.stringify(body) }));
  } catch (error) {
    showDemoMessage(error.message);
    pollDemo();
  }
}

async function onDemoStop() {
  showDemoMessage("");
  try {
    renderDemoStatus(await api("/api/demo/stop", { method: "POST" }));
  } catch (error) {
    showDemoMessage(error.message);
  }
}

// ---------- start ----------

function markRange() {
  for (const button of document.querySelectorAll("[data-minutes]")) {
    button.setAttribute("aria-pressed", String(Number(button.dataset.minutes) === state.minutes));
  }
}

function wireControls() {
  for (const button of document.querySelectorAll("[data-minutes]")) {
    button.addEventListener("click", () => {
      state.minutes = Number(button.dataset.minutes);
      markRange();
      refreshModels();
    });
  }
  byId("live").addEventListener("click", () => setLive(!state.live));
  byId("preview-form").addEventListener("submit", onPreview);
  byId("demo-form").addEventListener("submit", onDemoStart);
  byId("demo-stop").addEventListener("click", onDemoStop);
}

async function init() {
  markRange();
  wireControls();
  try {
    state.config = await api("/api/gateway/config");
    renderSettings(state.config);
  } catch (error) {
    showError(error.message);
  }
  await refreshModels();
  await loadDemo();
}

init();

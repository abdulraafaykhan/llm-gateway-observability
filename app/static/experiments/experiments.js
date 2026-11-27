"use strict";

// Experiments page: saved eval runs, a quality-vs-cost chart, offline replays, and run
// comparison. It only reads; all data text goes in through textContent or createTextNode.

const SVG_NS = "http://www.w3.org/2000/svg";
const CHART = { width: 680, height: 380, left: 64, right: 24, top: 24, bottom: 52 };
const BASELINE_LABELS = new Set(["all big", "all small"]);

const state = {
  split: "all",
  evalSet: null,
  runs: [],
  skipped: [],
  replay: null,
  replayPair: null,
  comparePair: null,
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

function svg(tag, attrs = {}, children = []) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return appendChildren(node, children);
}

const cell = (text) => el("td", {}, [text]);
const isMissing = (value) => value === null || value === undefined || Number.isNaN(value);
const formatMoney = (value) => (isMissing(value) ? "-" : `$${value.toFixed(6)}`);
const formatPercent = (value) => (isMissing(value) ? "-" : `${value.toFixed(1)}%`);
const formatCount = (value) => (isMissing(value) ? "-" : String(value));
const normalizeSet = (path) => (path || "").replaceAll("\\", "/");

function formatMs(value) {
  if (isMissing(value)) return "-";
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}

function table(headings, rows) {
  return el("div", { class: "table-wrap" }, [el("table", {}, [
    el("thead", {}, [el("tr", {}, headings.map((label) => el("th", { scope: "col" }, [label])))]),
    el("tbody", {}, rows),
  ])]);
}

async function api(path) {
  const response = await fetch(path, { headers: { Accept: "application/json" } });
  let body = null;
  try {
    body = await response.json();
  } catch (error) {
    body = null;
  }
  if (!response.ok) {
    let message = response.statusText;
    let type = "error";
    if (body && body.error) {
      type = body.error.type || type;
      message = body.error.message || message;
    } else if (body && body.detail) {
      type = "validation_error";
      message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    }
    throw new Error(`HTTP ${response.status} | ${type}: ${message}`);
  }
  return body;
}

function showError(message) {
  const box = byId("error");
  box.textContent = message || "";
  box.hidden = !message;
}

function query(params) {
  return new URLSearchParams({ split: state.split, ...params }).toString();
}

// ---------- runs ----------

function setRuns() {
  return state.runs.filter((run) => normalizeSet(run.eval_set_path) === state.evalSet);
}

function evalSetChoices() {
  const counts = new Map();
  for (const run of state.runs) {
    const key = normalizeSet(run.eval_set_path);
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  return [...counts.entries()].sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
}

function renderEvalSets() {
  const choices = evalSetChoices();
  const select = byId("eval-set");
  select.replaceChildren(...choices.map(([path, count]) => el("option", { value: path }, [`${path} (${count} runs)`])));
  if (!choices.some(([path]) => path === state.evalSet)) state.evalSet = choices.length ? choices[0][0] : null;
  if (state.evalSet) select.value = state.evalSet;
}

function renderRunsTable() {
  const rows = setRuns().map((run) => el("tr", {}, [
    cell(run.file), cell(run.label), cell(run.model_requested), cell(new Date(run.created_at).toLocaleString()),
    cell(formatCount(run.questions)), cell(formatCount(run.passed)), cell(formatPercent(run.score_percent)),
    cell(formatCount(run.errors)), cell(formatCount(run.truncated)), cell(formatMoney(run.total_cost_usd)),
    cell(formatMs(run.latency_p95_ms)),
  ]));
  byId("table-runs").tBodies[0].replaceChildren(...rows);
  const skipped = byId("skipped");
  skipped.hidden = state.skipped.length === 0;
  skipped.textContent = state.skipped.length
    ? `Skipped files: ${state.skipped.map((item) => `${item.file} (${item.reason})`).join("; ")}`
    : "";
}

function fillRunSelect(select, runs, preferredModel, fallbackIndex) {
  const previous = select.value;
  select.replaceChildren(...runs.map((run) => el("option", { value: run.file }, [`${run.label} (${run.file})`])));
  if (runs.some((run) => run.file === previous)) {
    select.value = previous;
    return;
  }
  const preferred = runs.find((run) => run.model_requested === preferredModel);
  const pick = preferred || runs[Math.min(fallbackIndex, runs.length - 1)];
  if (pick) select.value = pick.file;
}

function renderPairPickers() {
  const runs = setRuns();
  const enough = runs.length >= 2;
  byId("replay-empty").hidden = enough;
  byId("replay-form").hidden = !enough;
  byId("compare-form").hidden = !enough;
  fillRunSelect(byId("replay-small"), runs, "small", 0);
  fillRunSelect(byId("replay-big"), runs, "big", 1);
  fillRunSelect(byId("compare-a"), runs, "small", 0);
  fillRunSelect(byId("compare-b"), runs, "big", 1);
  if (!enough) {
    byId("replay-result").replaceChildren();
    byId("compare-result").replaceChildren();
  }
}

// ---------- chart ----------

function niceMax(value) {
  if (!(value > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (value <= step * power) return step * power;
  }
  return 10 * power;
}

function shortLabel(label) {
  return label.replace(/^threshold /, "thr ").replace(/^tokens > /, "tokens>");
}

function chartPoints() {
  const measured = setRuns().map((run) => ({
    label: run.label, score: run.score_percent, cost: run.total_cost_usd, hollow: false,
    source: `measured run ${run.file}`,
  }));
  if (!state.replay) return measured;
  const source = `offline replay of ${state.replay.small_file} and ${state.replay.big_file}`;
  const router = state.replay.router_rows.map((row) => ({ ...row, kind: "router replay" }));
  const cascade = state.replay.cascade_rows
    .filter((row) => !BASELINE_LABELS.has(row.label))
    .map((row) => ({ ...row, kind: "cascade replay" }));
  const replayed = [...router, ...cascade].map((row) => ({
    label: shortLabel(row.label), score: row.score_percent, cost: row.cost_usd, hollow: true,
    source: `${row.kind}, ${source}`,
  }));
  return [...measured, ...replayed];
}

function yRange(points) {
  if (!byId("zoom").checked || !points.length) return [0, 100];
  const scores = points.map((point) => point.score);
  return [Math.max(0, Math.floor(Math.min(...scores) - 5)), Math.min(100, Math.ceil(Math.max(...scores) + 5))];
}

function drawAxes(root, xMax, yMin, yMax) {
  const plotW = CHART.width - CHART.left - CHART.right;
  const plotH = CHART.height - CHART.top - CHART.bottom;
  for (let step = 0; step <= 4; step += 1) {
    const yValue = yMin + ((yMax - yMin) * step) / 4;
    const y = CHART.top + plotH - (plotH * step) / 4;
    root.appendChild(svg("line", { x1: CHART.left, x2: CHART.left + plotW, y1: y, y2: y, class: "grid" }));
    root.appendChild(svg("text", { x: CHART.left - 6, y: y + 4, "text-anchor": "end", class: "axis-label" }, [`${yValue.toFixed(0)}%`]));
    const x = CHART.left + (plotW * step) / 4;
    root.appendChild(svg("text", { x, y: CHART.top + plotH + 18, "text-anchor": "middle", class: "axis-label" }, [`$${((xMax * step) / 4).toFixed(4)}`]));
  }
  root.appendChild(svg("text", { x: CHART.left + plotW / 2, y: CHART.height - 8, "text-anchor": "middle", class: "axis-title" }, ["Total cost (USD)"]));
  const yTitle = byId("zoom").checked ? `Score (%), zoomed ${yMin} to ${yMax}` : "Score (%), 0 to 100";
  root.appendChild(svg("text", {
    x: 14, y: CHART.top + plotH / 2, "text-anchor": "middle", class: "axis-title",
    transform: `rotate(-90 14 ${CHART.top + plotH / 2})`,
  }, [yTitle]));
}

function renderChart() {
  const box = byId("chart");
  const points = chartPoints();
  if (!points.length) {
    box.replaceChildren(el("p", { class: "empty" }, ["No runs to plot for this eval set."]));
    return;
  }
  const xMax = niceMax(Math.max(...points.map((point) => point.cost)));
  const [yMin, yMax] = yRange(points);
  const plotW = CHART.width - CHART.left - CHART.right;
  const plotH = CHART.height - CHART.top - CHART.bottom;
  const measured = points.filter((point) => !point.hollow).length;
  const summary = `Quality vs cost: ${measured} measured runs and ${points.length - measured} replay settings; `
    + points.map((point) => `${point.label} ${point.score.toFixed(1)}% at $${point.cost.toFixed(6)}`).join(", ");
  const root = svg("svg", { viewBox: `0 0 ${CHART.width} ${CHART.height}`, role: "img", "aria-label": summary, class: "chart" });
  root.appendChild(svg("title", {}, ["Quality vs cost"]));
  drawAxes(root, xMax, yMin, yMax);
  for (const point of points) {
    const x = CHART.left + (plotW * point.cost) / xMax;
    const clamped = Math.min(Math.max(point.score, yMin), yMax);
    const y = CHART.top + plotH - (plotH * (clamped - yMin)) / Math.max(yMax - yMin, 1);
    const tooltip = `${point.label}: score ${point.score.toFixed(1)}%, cost $${point.cost.toFixed(6)} (${point.source})`;
    root.appendChild(svg("circle", { cx: x, cy: y, r: point.hollow ? 5 : 6, class: point.hollow ? "hollow" : "filled" }, [
      svg("title", {}, [tooltip]),
    ]));
    root.appendChild(svg("text", { x: x + 8, y: y - 6, class: point.hollow ? "point-label hollow-label" : "point-label" }, [point.label]));
  }
  const legend = el("p", { class: "legend" }, [
    el("span", { class: "legend-filled", "aria-hidden": "true" }), "measured run   ",
    el("span", { class: "legend-hollow", "aria-hidden": "true" }), "offline replay setting",
  ]);
  box.replaceChildren(root, legend);
}

// ---------- replay ----------

function routerRows(rows) {
  return rows.map((row) => el("tr", {}, [
    el("th", { scope: "row" }, [row.label]), cell(formatCount(row.to_small)), cell(formatCount(row.to_big)),
    cell(formatCount(row.fallbacks)), cell(formatCount(row.passed)), cell(formatPercent(row.score_percent)),
    cell(formatMoney(row.cost_usd)),
  ]));
}

function cascadeRows(rows) {
  return rows.map((row) => el("tr", {}, [
    el("th", { scope: "row" }, [row.label]), cell(formatCount(row.kept_small)), cell(formatCount(row.escalated)),
    cell(formatCount(row.passed)), cell(formatPercent(row.score_percent)), cell(formatCount(row.rescued)),
    cell(formatCount(row.lost)), cell(formatMoney(row.cost_usd)),
    cell(isMissing(row.cost_vs_all_big_percent) ? "n/a" : `${Math.round(row.cost_vs_all_big_percent)}%`),
  ]));
}

function renderReplay(result) {
  const tokens = result.small_tokens;
  const tokenLine = tokens
    ? `Small's output tokens (answered questions): p50 ${tokens.p50.toFixed(0)} | p75 ${tokens.p75.toFixed(0)} | p90 ${tokens.p90.toFixed(0)} | max ${tokens.max.toFixed(0)}`
    : "Small's output tokens: none answered.";
  byId("replay-result").replaceChildren(
    el("p", {}, [`small: ${result.small_label} | big: ${result.big_label} | split: ${result.split} | questions: ${result.questions} | fallback: on`]),
    el("h3", {}, ["Router replay"]),
    table(["Setting", "To small", "To big", "Fallbacks", "Passed", "Score", "Cost"], routerRows(result.router_rows)),
    el("h3", {}, ["Cascade replay"]),
    table(["Setting", "Kept small", "Escalated", "Passed", "Score", "Rescued", "Lost", "Cost", "Cost vs all big"], cascadeRows(result.cascade_rows)),
    el("p", { class: "muted" }, [tokenLine]),
  );
}

async function loadReplay(small, big) {
  state.replayPair = [small, big];
  byId("replay-result").replaceChildren(el("p", { class: "muted" }, ["Loading..."]));
  try {
    state.replay = await api(`/api/experiments/replay?${query({ small, big })}`);
    showError("");
    renderReplay(state.replay);
  } catch (error) {
    state.replay = null;
    byId("replay-result").replaceChildren(el("p", { class: "error" }, [error.message]));
  }
  renderChart();
}

// ---------- compare ----------

function groupCard(label, group) {
  return el("div", { class: "card" }, [
    el("div", { class: "card-label" }, [label]),
    el("div", { class: "card-value" }, [String(group.count)]),
    el("div", { class: "card-note" }, [`easy ${group.easy}, hard ${group.hard}`]),
  ]);
}

function questionTable(title, questions) {
  if (!questions.length) return [el("h3", {}, [title]), el("p", { class: "muted" }, ["None."])];
  return [el("h3", {}, [title]), table(["Id", "Difficulty", "Question", "Expected"], questions.map((item) => el("tr", {}, [
    cell(item.id), cell(item.difficulty), el("td", { class: "wrap" }, [item.question]), cell(item.expected.join(" / ")),
  ])))];
}

function renderCompare(result) {
  byId("compare-result").replaceChildren(
    el("p", {}, [`A: ${result.a_label} | B: ${result.b_label} | split: ${result.split} | compared: ${result.compared} questions`
      + ` | score A ${formatPercent(result.score_a_percent)} | score B ${formatPercent(result.score_b_percent)}`]),
    el("div", { class: "cards" }, [
      groupCard("Both passed", result.both_passed),
      groupCard(`Only B (${result.b_label}) passed`, result.only_b_passed),
      groupCard(`Only A (${result.a_label}) passed`, result.only_a_passed),
      groupCard("Neither passed", result.both_not_passed),
    ]),
    ...questionTable(`Only B (${result.b_label}) passed`, result.only_b_passed_questions),
    ...questionTable("Neither passed (check if the answer key is wrong)", result.both_not_passed_questions),
  );
}

async function loadCompare(a, b) {
  state.comparePair = [a, b];
  byId("compare-result").replaceChildren(el("p", { class: "muted" }, ["Loading..."]));
  try {
    renderCompare(await api(`/api/experiments/compare?${query({ a, b })}`));
    showError("");
  } catch (error) {
    byId("compare-result").replaceChildren(el("p", { class: "error" }, [error.message]));
  }
}

// ---------- loading ----------

function inSet(files) {
  const names = new Set(setRuns().map((run) => run.file));
  return files && files.every((file) => names.has(file));
}

async function refreshPairs() {
  if (state.replayPair && inSet(state.replayPair)) await loadReplay(...state.replayPair);
  else {
    state.replay = null;
    byId("replay-result").replaceChildren();
  }
  if (state.comparePair && inSet(state.comparePair)) await loadCompare(...state.comparePair);
  else byId("compare-result").replaceChildren();
}

function renderAll() {
  const empty = state.runs.length === 0;
  byId("no-runs").hidden = !empty;
  byId("content").hidden = empty;
  renderEvalSets();
  renderRunsTable();
  renderPairPickers();
  renderChart();
}

async function loadRuns() {
  byId("status").textContent = "Loading...";
  try {
    const data = await api(`/api/experiments/runs?${query({})}`);
    state.runs = data.runs;
    state.skipped = data.skipped;
    showError("");
    renderAll();
    byId("status").textContent = `${state.runs.length} runs (split: ${state.split})`;
  } catch (error) {
    showError(error.message);
    byId("status").textContent = "Could not load runs.";
  }
}

function wireControls() {
  byId("split").addEventListener("change", async (event) => {
    state.split = event.target.value;
    await loadRuns();
    await refreshPairs();
  });
  byId("eval-set").addEventListener("change", async (event) => {
    state.evalSet = event.target.value;
    renderAll();
    await refreshPairs();
  });
  byId("zoom").addEventListener("change", renderChart);
  byId("replay-form").addEventListener("submit", (event) => {
    event.preventDefault();
    loadReplay(byId("replay-small").value, byId("replay-big").value);
  });
  byId("compare-form").addEventListener("submit", (event) => {
    event.preventDefault();
    loadCompare(byId("compare-a").value, byId("compare-b").value);
  });
}

// Links like ?small=...&big=... or ?a=...&b=... open a replay or comparison directly.
async function openFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const small = params.get("small");
  const big = params.get("big");
  const a = params.get("a");
  const b = params.get("b");
  const linked = state.runs.find((run) => run.file === (small || a));
  if (linked && normalizeSet(linked.eval_set_path) !== state.evalSet) {
    state.evalSet = normalizeSet(linked.eval_set_path);
    renderAll();
  }
  if (small && big) await loadReplay(small, big);
  if (a && b) await loadCompare(a, b);
}

async function init() {
  const requestedSplit = new URLSearchParams(window.location.search).get("split");
  if (["all", "tune", "test"].includes(requestedSplit)) state.split = requestedSplit;
  byId("split").value = state.split;
  wireControls();
  await loadRuns();
  await openFromUrl();
}

init();

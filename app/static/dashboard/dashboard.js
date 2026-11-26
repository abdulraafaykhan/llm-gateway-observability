"use strict";

// Read-only dashboard. Every piece of database text is inserted with textContent or
// createTextNode only, because logged questions and answers are untrusted.

const SVG_NS = "http://www.w3.org/2000/svg";
const RANGES = { "15m": 15, "1h": 60, "24h": 1440, all: null };
const FIXED_MODEL_COLORS = { small: "var(--model-small)", big: "var(--model-big)" };
const EXTRA_MODEL_COLORS = ["var(--model-extra-1)", "var(--model-extra-2)", "var(--model-extra-3)"];
const CARD_LABELS = [
  "Requests", "Answered", "Errors", "Truncated", "Fallbacks", "Total cost",
  "Cost per 1,000 answered", "Latency p50", "Latency p95", "Latency p99",
];
const REFRESH_MS = 5000;
const RECENT_LIMIT = 50;
const CHART = { width: 640, height: 260, left: 64, right: 16, top: 28, bottom: 48 };

const state = {
  source: null, range: "all", live: false, timer: null,
  busy: false, pending: false, lastSummary: null, openIds: new Set(),
};

const byId = (id) => document.getElementById(id);

// ---------- DOM helpers ----------

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

const svgTitle = (text) => svg("title", {}, [text]);

// ---------- formatting ----------

function isMissing(value) {
  return value === null || value === undefined || Number.isNaN(value);
}

function formatMs(value) {
  if (isMissing(value)) return "-";
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}

const formatMoney = (value) => (isMissing(value) ? "-" : `$${value.toFixed(6)}`);
const formatPercent = (value) => (isMissing(value) ? "-" : `${value.toFixed(1)}%`);
const formatCount = (value) => (isMissing(value) ? "-" : String(value));

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatClock(iso, withSeconds) {
  const options = { hour: "2-digit", minute: "2-digit", hour12: false };
  if (withSeconds) options.second = "2-digit";
  return new Date(iso).toLocaleTimeString([], options);
}

const formatLocalDateTime = (iso) => new Date(iso).toLocaleString();

function modelColors(names) {
  const colors = {};
  let extra = 0;
  for (const name of [...names].sort()) {
    colors[name] = FIXED_MODEL_COLORS[name] || EXTRA_MODEL_COLORS[extra++ % EXTRA_MODEL_COLORS.length];
  }
  return colors;
}

// ---------- API ----------

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
    if (body && body.error && body.error.message) message = body.error.message;
    else if (body && body.detail) message = JSON.stringify(body.detail);
    throw new Error(`Request failed (HTTP ${response.status}): ${message}`);
  }
  return body;
}

function queryString(extra = {}) {
  const params = new URLSearchParams();
  if (state.source) params.set("source", state.source);
  const minutes = RANGES[state.range];
  if (minutes) params.set("minutes", String(minutes));
  for (const [key, value] of Object.entries(extra)) params.set(key, String(value));
  return params.toString();
}

function showError(message) {
  const box = byId("error");
  box.textContent = message || "";
  box.hidden = !message;
}

// ---------- chart helpers ----------

function niceMax(value) {
  if (!(value > 0)) return 1;
  const power = 10 ** Math.floor(Math.log10(value));
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (value <= step * power) return step * power;
  }
  return 10 * power;
}

const plotHeight = () => CHART.height - CHART.top - CHART.bottom;
const plotWidth = () => CHART.width - CHART.left - CHART.right;
const yPos = (value, max) => CHART.top + plotHeight() - (plotHeight() * value) / max;

function chartFrame(title, summary) {
  const root = svg("svg", {
    viewBox: `0 0 ${CHART.width} ${CHART.height}`, role: "img", "aria-label": summary, class: "chart",
  });
  root.appendChild(svgTitle(title));
  return root;
}

function drawYAxis(root, max, format) {
  for (let step = 0; step <= 4; step += 1) {
    const value = (max * step) / 4;
    const y = yPos(value, max);
    root.appendChild(svg("line", { x1: CHART.left, x2: CHART.width - CHART.right, y1: y, y2: y, class: "grid" }));
    root.appendChild(svg("text", { x: CHART.left - 6, y: y + 4, "text-anchor": "end", class: "axis-label" }, [format(value)]));
  }
}

function drawLegend(root, items) {
  let x = CHART.left;
  for (const [label, color] of items) {
    root.appendChild(svg("rect", { x, y: 6, width: 12, height: 12, style: `fill:${color}` }));
    root.appendChild(svg("text", { x: x + 16, y: 17 }, [label]));
    x += 24 + label.length * 7.5;
  }
}

function tickIndexes(count) {
  if (count <= 1) return [0];
  const ticks = Math.min(5, count);
  const indexes = new Set();
  for (let i = 0; i < ticks; i += 1) indexes.add(Math.round((i * (count - 1)) / (ticks - 1)));
  return [...indexes];
}

function drawTimeTicks(root, buckets, withSeconds) {
  const slot = plotWidth() / buckets.length;
  for (const index of tickIndexes(buckets.length)) {
    const x = CHART.left + index * slot + slot / 2;
    root.appendChild(svg("text", {
      x, y: CHART.height - CHART.bottom + 18, "text-anchor": "middle", class: "axis-label",
    }, [formatClock(buckets[index].start, withSeconds)]));
  }
}

// groups: [{ label, values: [number|null, ...] }], one bar per series in each group.
function groupedBarChart(container, title, groups, seriesNames, colors, format) {
  const values = groups.flatMap((group) => group.values).filter((value) => !isMissing(value));
  const max = niceMax(Math.max(0, ...values));
  const described = groups.map((group) =>
    `${group.label}: ${seriesNames.map((name, i) => `${name} ${format(group.values[i])}`).join(", ")}`);
  const root = chartFrame(title, `${title}. ${described.join("; ")}.`);
  drawYAxis(root, max, format);
  const groupWidth = plotWidth() / Math.max(groups.length, 1);
  const barWidth = Math.min(56, (groupWidth * 0.8) / seriesNames.length);
  groups.forEach((group, groupIndex) => {
    const groupStart = CHART.left + groupIndex * groupWidth + (groupWidth - barWidth * seriesNames.length) / 2;
    group.values.forEach((value, seriesIndex) => {
      const x = groupStart + seriesIndex * barWidth;
      const y = yPos(isMissing(value) ? 0 : value, max);
      const label = `${group.label} ${seriesNames[seriesIndex]}: ${format(value)}`;
      root.appendChild(svg("rect", {
        x: x + 2, y, width: Math.max(barWidth - 4, 1), height: yPos(0, max) - y,
        style: `fill:${colors[group.label]}`, class: `series-${seriesIndex}`,
      }, [svgTitle(label)]));
      root.appendChild(svg("text", { x: x + barWidth / 2, y: y - 5, "text-anchor": "middle", class: "value-label" }, [format(value)]));
      if (seriesNames.length > 1) {
        root.appendChild(svg("text", {
          x: x + barWidth / 2, y: CHART.height - CHART.bottom + 16, "text-anchor": "middle", class: "axis-label",
        }, [seriesNames[seriesIndex]]));
      }
    });
    root.appendChild(svg("text", {
      x: CHART.left + groupIndex * groupWidth + groupWidth / 2, y: CHART.height - CHART.bottom + 34,
      "text-anchor": "middle", class: "group-label",
    }, [group.label]));
  });
  container.replaceChildren(root);
}

function barChart(container, title, items, colors, format) {
  const groups = items.map((item) => ({ label: item.label, values: [item.value] }));
  groupedBarChart(container, title, groups, ["value"], colors, format);
}

function stackedBarChart(container, title, buckets, withSeconds) {
  const total = buckets.reduce((sum, bucket) => sum + bucket.requests, 0);
  const errors = buckets.reduce((sum, bucket) => sum + bucket.errors, 0);
  const max = Math.max(4, niceMax(Math.max(0, ...buckets.map((bucket) => bucket.requests))));
  const summary = `${title}: ${total} requests including ${errors} errors across ${buckets.length} time buckets.`;
  const root = chartFrame(title, summary);
  drawYAxis(root, max, (value) => String(Math.round(value)));
  drawLegend(root, [["Without errors", "var(--ok)"], ["Errors", "var(--error)"]]);
  const slot = plotWidth() / buckets.length;
  const barWidth = Math.max(slot * 0.8, 1);
  const base = yPos(0, max);
  buckets.forEach((bucket, index) => {
    const x = CHART.left + index * slot + (slot - barWidth) / 2;
    const label = `${formatClock(bucket.start, withSeconds)}: ${bucket.requests} requests, ${bucket.errors} errors`;
    const yGood = yPos(bucket.requests - bucket.errors, max);
    if (bucket.requests - bucket.errors > 0) {
      root.appendChild(svg("rect", { x, y: yGood, width: barWidth, height: base - yGood, style: "fill:var(--ok)" }, [svgTitle(label)]));
    }
    if (bucket.errors > 0) {
      const yTop = yPos(bucket.requests, max);
      root.appendChild(svg("rect", { x, y: yTop, width: barWidth, height: yGood - yTop, style: "fill:var(--error)" }, [svgTitle(label)]));
    }
  });
  drawTimeTicks(root, buckets, withSeconds);
  container.replaceChildren(root);
}

function lineChart(container, title, buckets, withSeconds) {
  const values = buckets.map((bucket) => bucket.p95_ms).filter((value) => !isMissing(value));
  if (!values.length) {
    container.replaceChildren(el("p", { class: "empty" }, ["No answered requests in this range."]));
    return;
  }
  const max = niceMax(Math.max(...values));
  const summary = `${title}: from ${formatMs(Math.min(...values))} to ${formatMs(Math.max(...values))} across ${values.length} time buckets with answers.`;
  const root = chartFrame(title, summary);
  drawYAxis(root, max, formatMs);
  const slot = plotWidth() / buckets.length;
  let segment = [];
  const flush = () => {
    if (segment.length > 1) root.appendChild(svg("polyline", { points: segment.join(" "), class: "line" }));
    segment = [];
  };
  buckets.forEach((bucket, index) => {
    if (isMissing(bucket.p95_ms)) {
      flush();
      return;
    }
    const x = CHART.left + index * slot + slot / 2;
    const y = yPos(bucket.p95_ms, max);
    segment.push(`${x},${y}`);
    root.appendChild(svg("circle", { cx: x, cy: y, r: 3, class: "point" }, [
      svgTitle(`${formatClock(bucket.start, withSeconds)}: p95 ${formatMs(bucket.p95_ms)}`),
    ]));
  });
  flush();
  drawTimeTicks(root, buckets, withSeconds);
  container.replaceChildren(root);
}

// ---------- page sections ----------

function renderCards(overall) {
  let items;
  if (!overall) {
    items = CARD_LABELS.map((label) => [label, "Loading...", null]);
  } else {
    const latency = overall.latency || {};
    items = [
      ["Requests", formatCount(overall.requests)],
      ["Answered", formatCount(overall.answered)],
      ["Errors", formatCount(overall.errors), `${formatPercent(overall.error_rate_percent)} error rate`],
      ["Truncated", formatCount(overall.truncated)],
      ["Fallbacks", formatCount(overall.fallbacks)],
      ["Total cost", formatMoney(overall.total_cost_usd)],
      ["Cost per 1,000 answered", formatMoney(overall.cost_per_1000_answered_usd)],
      ["Latency p50", formatMs(latency.p50_ms)],
      ["Latency p95", formatMs(latency.p95_ms)],
      ["Latency p99", formatMs(latency.p99_ms)],
    ];
  }
  byId("cards").replaceChildren(...items.map(([label, value, note]) => el("div", { class: "card" }, [
    el("div", { class: "card-label" }, [label]),
    el("div", { class: "card-value" }, [value]),
    note ? el("div", { class: "card-note" }, [note]) : null,
  ])));
}

function renderCharts(summary, series) {
  const entries = Object.entries(summary.by_model);
  const colors = modelColors(entries.map(([name]) => name));
  const latencyGroups = entries.map(([name, group]) => {
    const latency = group.latency || {};
    return { label: name, values: [latency.p50_ms ?? null, latency.p95_ms ?? null, latency.p99_ms ?? null] };
  });
  groupedBarChart(byId("chart-latency"), "Latency by model", latencyGroups, ["p50", "p95", "p99"], colors, formatMs);
  const withSeconds = series.bucket_seconds < 60;
  if (series.buckets.length) {
    stackedBarChart(byId("chart-requests"), "Requests over time", series.buckets, withSeconds);
    lineChart(byId("chart-p95"), "p95 latency over time", series.buckets, withSeconds);
  }
  const costs = entries.map(([name, group]) => ({ label: name, value: group.total_cost_usd }));
  barChart(byId("chart-cost"), "Cost by model", costs, colors, formatMoney);
}

const cell = (text) => el("td", {}, [text]);

function renderModelTable(byModel) {
  const rows = Object.entries(byModel).map(([name, group]) => {
    const latency = group.latency || {};
    return el("tr", {}, [
      el("th", { scope: "row" }, [name]),
      cell(formatCount(group.requests)), cell(formatCount(group.answered)),
      cell(formatCount(group.truncated)), cell(formatCount(group.errors)),
      cell(formatPercent(group.error_rate_percent)), cell(formatCount(group.input_tokens)),
      cell(formatCount(group.output_tokens)), cell(formatMoney(group.total_cost_usd)),
      cell(formatMoney(group.cost_per_1000_answered_usd)), cell(formatMs(latency.p50_ms)),
      cell(formatMs(latency.p95_ms)), cell(formatMs(latency.p99_ms)),
    ]);
  });
  byId("table-models").tBodies[0].replaceChildren(...rows);
}

function detailBlock(item) {
  const failed = item.status === "error";
  return el("dl", {}, [
    el("dt", {}, ["Question"]), el("dd", {}, [el("pre", {}, [item.question || "-"])]),
    el("dt", {}, [failed ? "Error" : "Answer"]),
    el("dd", {}, [el("pre", {}, [(failed ? item.error_message : item.answer) || "-"])]),
    el("dt", {}, ["Route reason"]), el("dd", {}, [el("pre", {}, [item.route_reason || "-"])]),
  ]);
}

function renderRecent(items) {
  const rows = [];
  for (const item of items) {
    const open = state.openIds.has(item.id);
    const row = el("tr", { tabindex: "0", class: "clickable", "aria-expanded": String(open) }, [
      cell(formatLocalDateTime(item.created_at)), cell(item.model_name), cell(item.requested_model || "-"),
      cell(item.status), cell(item.route_reason || "-"), cell(item.fallback_used ? "yes" : "no"),
      cell(formatCount(item.input_tokens)), cell(formatCount(item.output_tokens)),
      cell(formatMs(item.latency_ms)), cell(formatMoney(item.cost_usd)),
    ]);
    const detail = el("tr", { class: "detail" }, [el("td", { colspan: "10" }, [detailBlock(item)])]);
    detail.hidden = !open;
    const toggle = () => {
      const opening = detail.hidden;
      detail.hidden = !opening;
      row.setAttribute("aria-expanded", String(opening));
      if (opening) state.openIds.add(item.id);
      else state.openIds.delete(item.id);
    };
    row.addEventListener("click", toggle);
    row.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        toggle();
      }
    });
    rows.push(row, detail);
  }
  byId("table-recent").tBodies[0].replaceChildren(...rows);
}

function render(summary, series, recent) {
  state.lastSummary = summary;
  byId("banner").hidden = !summary.source.toLowerCase().includes("fake");
  renderCards(summary.overall);
  const empty = summary.overall.requests === 0;
  byId("empty").hidden = !empty;
  byId("charts").hidden = empty;
  if (!empty) renderCharts(summary, series);
  renderModelTable(summary.by_model);
  renderRecent(recent.items);
}

// ---------- loading ----------

async function refresh(force = false) {
  if (state.busy) {
    if (force) state.pending = true;
    return;
  }
  state.busy = true;
  if (force) renderCards(null);
  try {
    const [summary, series, recent] = await Promise.all([
      api(`/api/dashboard/summary?${queryString()}`),
      api(`/api/dashboard/timeseries?${queryString()}`),
      api(`/api/dashboard/requests?${queryString({ limit: RECENT_LIMIT })}`),
    ]);
    showError("");
    render(summary, series, recent);
    byId("updated").textContent = `Updated ${new Date().toLocaleTimeString([], { hour12: false })}`;
  } catch (error) {
    showError(error.message);
    if (state.lastSummary) renderCards(state.lastSummary.overall);
  } finally {
    state.busy = false;
    if (state.pending) {
      state.pending = false;
      refresh(true);
    }
  }
}

async function loadSources() {
  const data = await api("/api/dashboard/sources");
  const select = byId("source");
  select.replaceChildren(...data.sources.map((source) =>
    el("option", { value: source.name }, [`${source.name} (${formatBytes(source.size_bytes)})`])));
  if (!data.sources.length) {
    byId("no-sources").hidden = false;
    byId("content").hidden = true;
    return false;
  }
  const requested = new URLSearchParams(window.location.search).get("source");
  state.source = requested || data.default;
  if (data.sources.some((source) => source.name === state.source)) select.value = state.source;
  return true;
}

async function loadMode() {
  try {
    const health = await api("/health");
    byId("mode").textContent = `Gateway mode: ${health.mode}`;
  } catch (error) {
    byId("mode").textContent = "Gateway mode: unknown";
  }
}

function saveToUrl() {
  const params = new URLSearchParams();
  if (state.source) params.set("source", state.source);
  params.set("range", state.range);
  window.history.replaceState(null, "", `${window.location.pathname}?${params.toString()}`);
}

function markRange() {
  for (const button of document.querySelectorAll("[data-range]")) {
    button.setAttribute("aria-pressed", String(button.dataset.range === state.range));
  }
}

function setLive(on) {
  state.live = on;
  const button = byId("live");
  button.setAttribute("aria-pressed", String(on));
  button.textContent = on ? "Live: on" : "Live: off";
  window.clearInterval(state.timer);
  state.timer = on ? window.setInterval(() => refresh(false), REFRESH_MS) : null;
}

function wireControls() {
  byId("source").addEventListener("change", (event) => {
    state.source = event.target.value;
    state.openIds.clear();
    saveToUrl();
    refresh(true);
  });
  for (const button of document.querySelectorAll("[data-range]")) {
    button.addEventListener("click", () => {
      state.range = button.dataset.range;
      markRange();
      saveToUrl();
      refresh(true);
    });
  }
  byId("live").addEventListener("click", () => setLive(!state.live));
}

async function init() {
  const requestedRange = new URLSearchParams(window.location.search).get("range");
  if (requestedRange && requestedRange in RANGES) state.range = requestedRange;
  markRange();
  wireControls();
  renderCards(null);
  loadMode();
  try {
    if (await loadSources()) await refresh(true);
  } catch (error) {
    showError(error.message);
  }
}

init();

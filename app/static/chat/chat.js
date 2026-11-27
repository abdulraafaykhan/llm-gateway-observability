"use strict";

// Chat page. All text from the user, the models, and the API goes in through textContent or
// createTextNode only, because it is untrusted.

const AUTO_MODEL = "auto";
const TIMEOUT_MS = 150000;

const state = {
  history: [],
  models: [],
  busy: false,
  generation: 0,
  controllers: new Set(),
  selectedNode: null,
};

const byId = (id) => document.getElementById(id);
const conversationBox = () => byId("conversation");

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

function scrollToEnd() {
  const box = conversationBox();
  box.scrollTop = box.scrollHeight;
}

// ---------- formatting ----------

function isMissing(value) {
  return value === null || value === undefined || Number.isNaN(value);
}

function formatMs(value) {
  if (isMissing(value)) return "-";
  return value < 1000 ? `${Math.round(value)} ms` : `${(value / 1000).toFixed(2)} s`;
}

const formatMoney = (value) => (isMissing(value) ? "-" : `$${value.toFixed(6)}`);
const formatCount = (value) => (isMissing(value) ? "-" : String(value));

function errorText(failure) {
  const status = failure.status ? `HTTP ${failure.status}` : "No response";
  return `${status} | ${failure.type}: ${failure.message}`;
}

// ---------- API ----------

function describeError(status, body, statusText) {
  if (body && body.error) {
    return { status, type: body.error.type || "error", message: body.error.message || statusText };
  }
  if (body && body.detail) {
    const message = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    return { status, type: "validation_error", message };
  }
  return { status, type: "error", message: statusText || "Request failed" };
}

async function postChat(messages, model) {
  const controller = new AbortController();
  state.controllers.add(controller);
  const timer = window.setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const response = await fetch("/v1/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ messages, model }),
      signal: controller.signal,
    });
    let body = null;
    try {
      body = await response.json();
    } catch (error) {
      body = null;
    }
    if (response.ok && body) return { ok: true, data: body };
    return { ok: false, ...describeError(response.status, body, response.statusText) };
  } catch (error) {
    if (error.name === "AbortError") {
      const seconds = TIMEOUT_MS / 1000;
      return { ok: false, status: null, type: "timeout", message: `No answer after ${seconds} seconds, so the request was stopped.` };
    }
    return { ok: false, status: null, type: "network", message: `Could not reach the gateway (${error.message}).` };
  } finally {
    window.clearTimeout(timer);
    state.controllers.delete(controller);
  }
}

function startWaiting(node) {
  const started = performance.now();
  const update = () => {
    node.textContent = `Waiting... ${((performance.now() - started) / 1000).toFixed(1)} s`;
  };
  update();
  const timer = window.setInterval(update, 100);
  return () => {
    window.clearInterval(timer);
    node.textContent = "";
  };
}

// ---------- inspector ----------

const cutOffLabel = () => el("p", { class: "tag tag-cut" }, ["Answer was cut off"]);

function renderInspector(reply) {
  const box = byId("inspector");
  if (!reply) {
    box.replaceChildren(el("p", { class: "muted" }, ["Send a message to see how it was handled."]));
    return;
  }
  const rows = [
    ["Model that answered", reply.model || "-"],
    ["Model asked for", reply.requested_model || "-"],
    ["Status", reply.truncated ? "cut off" : "answered"],
    ["Route reason", reply.route_reason || "-"],
    ["Fallback used", reply.fallback_used ? "yes" : "no"],
    ["Tokens in", formatCount(reply.input_tokens)],
    ["Tokens out", formatCount(reply.output_tokens)],
    ["Latency", formatMs(reply.latency_ms)],
    ["Cost", formatMoney(reply.cost_usd)],
  ];
  const list = el("dl", {}, rows.flatMap(([label, value]) => [el("dt", {}, [label]), el("dd", {}, [value])]));
  box.replaceChildren(...(reply.truncated ? [cutOffLabel()] : []), list);
}

function selectReply(node, reply) {
  if (state.selectedNode) state.selectedNode.classList.remove("selected");
  state.selectedNode = node;
  if (node) node.classList.add("selected");
  renderInspector(reply);
}

function makeSelectable(node, reply) {
  node.setAttribute("tabindex", "0");
  node.setAttribute("title", "Show in Inspector");
  node.addEventListener("click", () => selectReply(node, reply));
  node.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectReply(node, reply);
    }
  });
}

// ---------- conversation ----------

function addUserBubble(text) {
  const bubble = el("div", { class: "bubble user" }, [el("p", { class: "bubble-text" }, [text])]);
  conversationBox().appendChild(bubble);
  scrollToEnd();
  return bubble;
}

function addAssistantBubble(reply) {
  const caption = `${reply.model} | ${formatMs(reply.latency_ms)} | ${formatMoney(reply.cost_usd)}`;
  const bubble = el("div", { class: "bubble assistant" }, [
    el("p", { class: "bubble-text" }, [reply.answer]),
    reply.truncated ? cutOffLabel() : null,
    el("p", { class: "caption" }, [caption]),
  ]);
  makeSelectable(bubble, reply);
  conversationBox().appendChild(bubble);
  scrollToEnd();
  return bubble;
}

function addErrorBubble(failure) {
  const bubble = el("div", { class: "bubble error-bubble" }, [el("p", { class: "bubble-text" }, [errorText(failure)])]);
  conversationBox().appendChild(bubble);
  scrollToEnd();
}

async function sendNormal(text, generation) {
  const userBubble = addUserBubble(text);
  const messages = [...state.history, { role: "user", content: text }];
  const result = await postChat(messages, byId("model").value);
  if (generation !== state.generation) return;
  if (!result.ok) {
    userBubble.classList.add("unanswered");
    userBubble.appendChild(el("p", { class: "tag tag-error" }, ["Not answered"]));
    addErrorBubble(result);
    return;
  }
  state.history = [...messages, { role: "assistant", content: result.data.answer }];
  selectReply(addAssistantBubble(result.data), result.data);
}

// ---------- compare mode ----------

function fillColumn(column, reply) {
  const tokens = `${formatCount(reply.input_tokens)} in / ${formatCount(reply.output_tokens)} out tokens`;
  appendChildren(column, [
    el("p", { class: "bubble-text" }, [reply.answer]),
    reply.truncated ? cutOffLabel() : null,
    el("p", { class: "caption" }, [`${formatMs(reply.latency_ms)} | ${tokens} | ${formatMoney(reply.cost_usd)}`]),
  ]);
}

async function runColumn(column, model, text, generation, onAnswer) {
  const status = el("p", { class: "muted" });
  column.appendChild(status);
  const stopWaiting = startWaiting(status);
  const result = await postChat([{ role: "user", content: text }], model);
  stopWaiting();
  status.remove();
  if (generation !== state.generation) return;
  if (!result.ok) {
    column.appendChild(el("p", { class: "bubble-text error-text" }, [errorText(result)]));
    return;
  }
  fillColumn(column, result.data);
  makeSelectable(column, result.data);
  onAnswer(column, result.data);
}

async function sendCompare(text, generation) {
  const models = [byId("left-model").value, byId("right-model").value];
  const columns = el("div", { class: "compare-columns" });
  const card = el("div", { class: "compare-card" }, [el("p", { class: "compare-question" }, [text]), columns]);
  conversationBox().appendChild(card);
  scrollToEnd();
  let selected = false;
  const selectFirst = (column, reply) => {
    if (selected) return;
    selected = true;
    selectReply(column, reply);
  };
  const runs = models.map((model) => {
    const column = el("div", { class: "compare-column" }, [el("h3", {}, [model])]);
    columns.appendChild(column);
    return runColumn(column, model, text, generation, selectFirst);
  });
  await Promise.all(runs);
}

// ---------- controls ----------

function setBusy(on) {
  state.busy = on;
  byId("send").disabled = on;
}

async function onSend(event) {
  event.preventDefault();
  const input = byId("input");
  const text = input.value.trim();
  if (!text || state.busy) return;
  input.value = "";
  setBusy(true);
  const generation = state.generation;
  const stopWaiting = startWaiting(byId("waiting"));
  try {
    if (byId("compare").checked) await sendCompare(text, generation);
    else await sendNormal(text, generation);
  } finally {
    stopWaiting();
    setBusy(false);
  }
}

function newChat() {
  state.generation += 1;
  for (const controller of state.controllers) controller.abort();
  state.history = [];
  state.selectedNode = null;
  conversationBox().replaceChildren();
  renderInspector(null);
  byId("input").focus();
}

function onCompareToggle() {
  const on = byId("compare").checked;
  byId("model").disabled = on;
  byId("compare-pickers").hidden = !on;
  byId("compare-note").hidden = !on;
}

function fillSelect(select, names, selected) {
  select.replaceChildren(...names.map((name) => el("option", { value: name }, [name])));
  if (names.includes(selected)) select.value = selected;
}

function showPageError(message) {
  const box = byId("page-error");
  box.textContent = message;
  box.hidden = false;
}

async function loadHealth() {
  try {
    const response = await fetch("/health", { headers: { Accept: "application/json" } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const health = await response.json();
    state.models = Array.isArray(health.models) ? health.models : [];
    byId("fake-banner").hidden = health.mode !== "fake";
  } catch (error) {
    showPageError(`Could not read /health (${error.message}), so the model lists are empty.`);
  }
  fillSelect(byId("model"), [AUTO_MODEL, ...state.models], AUTO_MODEL);
  fillSelect(byId("left-model"), state.models, state.models[0]);
  fillSelect(byId("right-model"), state.models, state.models[1] ?? state.models[0]);
}

function wireControls() {
  byId("composer").addEventListener("submit", onSend);
  byId("input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      byId("composer").requestSubmit();
    }
  });
  byId("compare").addEventListener("change", onCompareToggle);
  byId("new-chat").addEventListener("click", newChat);
}

function init() {
  wireControls();
  onCompareToggle();
  renderInspector(null);
  loadHealth();
}

init();

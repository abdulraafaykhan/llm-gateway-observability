"use strict";

// Chat page. Text from the user, the models, and the API goes in through textContent,
// createTextNode, or renderMarkdown (which only builds fixed elements around text nodes).

const AUTO_MODEL = "auto";
const TIMEOUT_MS = 150000;
const CONCISE_INSTRUCTION = "Answer in plain text. Be brief unless I ask for detail. Do not use tables or LaTeX.";
const STATUS_WORDS = { ok: "answered", truncated: "cut off", error: "failed" };

const state = {
  history: [],
  models: [],
  busy: false,
  generation: 0,
  controllers: new Set(),
  selectedNode: null,
  answers: [],
  showRaw: false,
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

const cell = (text) => el("td", {}, [text]);

function scrollToEnd() {
  const box = conversationBox();
  box.scrollTop = box.scrollHeight;
}

// Show the top of a new answer or compare card, not the bottom.
function scrollToTop(node) {
  conversationBox().scrollTop = Math.max(0, node.offsetTop - 8);
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

function metricsText(reply) {
  const tokens = `${formatCount(reply.input_tokens)} in / ${formatCount(reply.output_tokens)} out tokens`;
  return `${formatMs(reply.latency_ms)} | ${tokens} | ${formatMoney(reply.cost_usd)}`;
}

// ---------- answers ----------

function renderAnswer(container, text) {
  if (state.showRaw) container.replaceChildren(el("p", { class: "bubble-text" }, [text]));
  else container.replaceChildren(window.renderMarkdown(text));
}

function answerNode(text) {
  const container = el("div", { class: "answer md" });
  renderAnswer(container, text);
  state.answers.push({ container, text });
  return container;
}

function rerenderAnswers() {
  state.showRaw = byId("raw").checked;
  for (const { container, text } of state.answers) renderAnswer(container, text);
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

function requestMessages(messages, concise) {
  return concise ? [{ role: "system", content: CONCISE_INSTRUCTION }, ...messages] : messages;
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

function attemptsTable(reply) {
  const attempts = reply.attempts || [];
  const labels = ["#", "Model", "Status", "Tokens in", "Tokens out", "Time", "Cost"];
  const head = el("thead", {}, [el("tr", {}, labels.map((label) => el("th", { scope: "col" }, [label])))]);
  const rows = [];
  for (const attempt of attempts) {
    rows.push(el("tr", {}, [
      cell(String(attempt.attempt)), cell(attempt.model), cell(STATUS_WORDS[attempt.status] || attempt.status),
      cell(formatCount(attempt.input_tokens)), cell(formatCount(attempt.output_tokens)),
      cell(formatMs(attempt.latency_ms)), cell(formatMoney(attempt.cost_usd)),
    ]));
    if (attempt.error_message) {
      rows.push(el("tr", { class: "attempt-error" }, [el("td", { colspan: "7" }, [`Error: ${attempt.error_message}`])]));
    }
  }
  const sum = (key) => attempts.reduce((total, attempt) => total + (attempt[key] || 0), 0);
  rows.push(el("tr", { class: "attempt-total" }, [
    el("th", { scope: "row", colspan: "3" }, ["Total"]),
    cell(formatCount(sum("input_tokens"))), cell(formatCount(sum("output_tokens"))),
    cell(formatMs(reply.latency_ms)), cell(formatMoney(sum("cost_usd"))),
  ]));
  return el("div", { class: "table-wrap" }, [el("table", { class: "attempts" }, [head, el("tbody", {}, rows)])]);
}

function renderInspector(entry) {
  const box = byId("inspector");
  if (!entry) {
    box.replaceChildren(el("p", { class: "muted" }, ["Send a message to see how it was handled."]));
    return;
  }
  const { reply, concise } = entry;
  const rows = [
    ["Model that answered", reply.model || "-"],
    ["Model asked for", reply.requested_model || "-"],
    ["Status", reply.truncated ? "cut off" : "answered"],
    ["Route reason", reply.route_reason || "-"],
    ["Fallback used", reply.fallback_used ? "yes" : "no"],
    ["Concise mode", concise ? "on" : "off"],
    ["Tokens in", formatCount(reply.input_tokens)],
    ["Tokens out", formatCount(reply.output_tokens)],
    ["Latency", formatMs(reply.latency_ms)],
    ["Cost", formatMoney(reply.cost_usd)],
  ];
  const list = el("dl", {}, rows.flatMap(([label, value]) => [el("dt", {}, [label]), el("dd", {}, [value])]));
  const attempts = (reply.attempts || []).length ? [el("h3", {}, ["Attempts"]), attemptsTable(reply)] : [];
  box.replaceChildren(...(reply.truncated ? [cutOffLabel()] : []), list, ...attempts);
}

function selectReply(node, entry) {
  if (state.selectedNode) state.selectedNode.classList.remove("selected");
  state.selectedNode = node;
  if (node) node.classList.add("selected");
  renderInspector(entry);
}

function makeSelectable(node, entry) {
  node.setAttribute("tabindex", "0");
  node.setAttribute("title", "Show in Inspector");
  node.addEventListener("click", () => selectReply(node, entry));
  node.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      selectReply(node, entry);
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

function addAssistantBubble(entry) {
  const { reply } = entry;
  const caption = `${reply.model} | ${formatMs(reply.latency_ms)} | ${formatMoney(reply.cost_usd)}`;
  const bubble = el("div", { class: "bubble assistant" }, [
    answerNode(reply.answer),
    reply.truncated ? cutOffLabel() : null,
    el("p", { class: "caption" }, [caption]),
  ]);
  makeSelectable(bubble, entry);
  conversationBox().appendChild(bubble);
  scrollToTop(bubble);
  return bubble;
}

function addErrorBubble(failure) {
  const bubble = el("div", { class: "bubble error-bubble" }, [el("p", { class: "bubble-text" }, [errorText(failure)])]);
  conversationBox().appendChild(bubble);
  scrollToEnd();
}

async function sendNormal(text, generation, concise) {
  const userBubble = addUserBubble(text);
  const messages = [...state.history, { role: "user", content: text }];
  const result = await postChat(requestMessages(messages, concise), byId("model").value);
  if (generation !== state.generation) return;
  if (!result.ok) {
    userBubble.classList.add("unanswered");
    userBubble.appendChild(el("p", { class: "tag tag-error" }, ["Not answered"]));
    addErrorBubble(result);
    return;
  }
  state.history = [...messages, { role: "assistant", content: result.data.answer }];
  const entry = { reply: result.data, concise };
  selectReply(addAssistantBubble(entry), entry);
}

// ---------- compare mode ----------

function pickMore(a, b, value) {
  return value(a) >= value(b) ? [a, b] : [b, a];
}

function compareSentences(outcomes) {
  const failed = outcomes.filter((outcome) => !outcome.result.ok);
  if (failed.length) return failed.map((outcome) => `${outcome.label} failed: ${outcome.result.message}`);
  const [a, b] = outcomes.map((outcome) => ({ label: outcome.label, reply: outcome.result.data }));
  const sentences = [];
  if (a.reply.cost_usd > 0 && b.reply.cost_usd > 0) {
    const [high, low] = pickMore(a, b, (side) => side.reply.cost_usd);
    sentences.push(`${high.label} cost ${(high.reply.cost_usd / low.reply.cost_usd).toFixed(1)}x ${low.label}.`);
  } else {
    sentences.push("No cost ratio, because one of the costs was 0.");
  }
  const [slow, fast] = pickMore(a, b, (side) => side.reply.latency_ms);
  const seconds = (slow.reply.latency_ms - fast.reply.latency_ms) / 1000;
  sentences.push(seconds < 0.05 ? "Both took about the same time." : `${slow.label} was ${seconds.toFixed(1)} s slower than ${fast.label}.`);
  const cutOff = [a, b].filter((side) => side.reply.truncated);
  if (cutOff.length) {
    for (const side of cutOff) sentences.push(`${side.label} was cut off, so its length is not comparable.`);
  } else {
    const [more, fewer] = pickMore(a, b, (side) => side.reply.output_tokens);
    const difference = more.reply.output_tokens - fewer.reply.output_tokens;
    sentences.push(difference === 0
      ? "Both wrote the same number of output tokens."
      : `${more.label} wrote ${difference} more output tokens than ${fewer.label} (${more.reply.output_tokens} vs ${fewer.reply.output_tokens}).`);
  }
  return sentences;
}

async function runColumn(column, metrics, model, text, generation, concise, onAnswer) {
  const stopWaiting = startWaiting(metrics);
  const result = await postChat(requestMessages([{ role: "user", content: text }], concise), model);
  stopWaiting();
  if (generation !== state.generation) return result;
  if (!result.ok) {
    metrics.textContent = "Failed";
    column.appendChild(el("p", { class: "bubble-text error-text" }, [errorText(result)]));
    return result;
  }
  metrics.textContent = metricsText(result.data);
  appendChildren(column, [answerNode(result.data.answer), result.data.truncated ? cutOffLabel() : null]);
  const entry = { reply: result.data, concise };
  makeSelectable(column, entry);
  onAnswer(column, entry);
  return result;
}

async function sendCompare(text, generation, concise) {
  const models = [byId("left-model").value, byId("right-model").value];
  const labels = models[0] === models[1] ? [`left ${models[0]}`, `right ${models[1]}`] : models;
  const columns = el("div", { class: "compare-columns" });
  const card = el("div", { class: "compare-card" }, [el("p", { class: "compare-question" }, [text]), columns]);
  conversationBox().appendChild(card);
  scrollToTop(card);
  let selected = false;
  const selectFirst = (column, entry) => {
    if (selected) return;
    selected = true;
    selectReply(column, entry);
  };
  const runs = models.map((model, index) => {
    const metrics = el("p", { class: "caption metrics" });
    const column = el("div", { class: "compare-column" }, [el("h3", {}, [model]), metrics]);
    columns.appendChild(column);
    return runColumn(column, metrics, model, text, generation, concise, selectFirst)
      .then((result) => ({ label: labels[index], result }));
  });
  const outcomes = await Promise.all(runs);
  if (generation !== state.generation) return;
  const summary = el("div", { class: "compare-summary" }, compareSentences(outcomes).map((line) => el("p", {}, [line])));
  card.insertBefore(summary, columns);
  scrollToTop(card);
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
  const concise = byId("concise").checked;
  const stopWaiting = startWaiting(byId("waiting"));
  try {
    if (byId("compare").checked) await sendCompare(text, generation, concise);
    else await sendNormal(text, generation, concise);
  } finally {
    stopWaiting();
    setBusy(false);
  }
}

function newChat() {
  state.generation += 1;
  for (const controller of state.controllers) controller.abort();
  state.history = [];
  state.answers = [];
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
  byId("raw").addEventListener("change", rerenderAnswers);
  byId("new-chat").addEventListener("click", newChat);
}

function init() {
  byId("concise-text").textContent = CONCISE_INSTRUCTION;
  wireControls();
  onCompareToggle();
  state.showRaw = byId("raw").checked;
  renderInspector(null);
  loadHealth();
}

init();

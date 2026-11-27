"use strict";

// Shared top bar: page links plus the gateway mode read once from /health.
(function buildSiteNav() {
  const LINKS = [
    ["Chat", "/chat"],
    ["Observability", "/dashboard"],
    ["Gateway", "/gateway"],
    ["Experiments", "/experiments"],
  ];

  function isCurrent(href) {
    const path = window.location.pathname;
    return path === href || path.startsWith(`${href}/`);
  }

  function makeLink(label, href) {
    const item = document.createElement("li");
    const link = document.createElement("a");
    link.href = href;
    link.textContent = label;
    if (isCurrent(href)) link.setAttribute("aria-current", "page");
    item.appendChild(link);
    return item;
  }

  async function loadMode(chip) {
    try {
      const response = await fetch("/health", { headers: { Accept: "application/json" } });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const health = await response.json();
      chip.textContent = `Gateway mode: ${health.mode || "unknown"}`;
    } catch (error) {
      chip.textContent = "Gateway mode: unknown";
    }
  }

  const nav = document.getElementById("site-nav");
  if (!nav) return;
  nav.setAttribute("aria-label", "Site");

  const brand = document.createElement("span");
  brand.className = "site-nav-brand";
  brand.textContent = "LLM Gateway";

  const list = document.createElement("ul");
  list.className = "site-nav-links";
  for (const [label, href] of LINKS) list.appendChild(makeLink(label, href));

  const chip = document.createElement("span");
  chip.className = "site-nav-mode";
  chip.textContent = "Gateway mode: -";

  nav.replaceChildren(brand, list, chip);
  loadMode(chip);
})();

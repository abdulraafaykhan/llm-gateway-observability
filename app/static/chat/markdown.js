"use strict";

// A small, safe markdown subset for model answers. It only ever creates the fixed elements
// below (p, br, strong, code, pre, h3-h6, ul, ol, li, hr, table parts, div) and puts all input
// text in through text nodes, so nothing in an answer can become page markup.
(function defineRenderMarkdown() {
  const HEADING = /^(#{1,6})\s+(.*)$/;
  const BULLET = /^\s*[-*\u2022]\s+(.*)$/;
  const NUMBERED = /^\s*(\d+)\.\s+(.*)$/;
  const FENCE = /^\s*```/;
  const RULE = /^\s*---\s*$/;
  const INLINE = /\*\*([^*]+?)\*\*|`([^`]+)`/g;
  const LATEX_DELIMITERS = /\\\(|\\\)|\\\[|\\\]/g;
  const SEPARATOR_CELL = /^:?-+:?$/;
  // Heading levels map to a fixed list, so all of them stay smaller than the page title.
  const HEADING_TAGS = ["h3", "h3", "h4", "h5", "h6", "h6"];

  function make(tag, className) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }

  function appendText(parent, text) {
    const clean = text.replace(LATEX_DELIMITERS, "");
    if (clean) parent.appendChild(document.createTextNode(clean));
  }

  // Bold and inline code inside one line of text.
  function appendInline(parent, text) {
    let last = 0;
    for (const match of text.matchAll(INLINE)) {
      appendText(parent, text.slice(last, match.index));
      if (match[1] !== undefined) {
        const strong = make("strong");
        appendText(strong, match[1]);
        parent.appendChild(strong);
      } else {
        const code = make("code");
        code.textContent = match[2];
        parent.appendChild(code);
      }
      last = match.index + match[0].length;
    }
    appendText(parent, text.slice(last));
  }

  function splitCells(line) {
    let text = line.trim();
    if (text.startsWith("|")) text = text.slice(1);
    if (text.endsWith("|")) text = text.slice(0, -1);
    return text.split("|").map((cell) => cell.trim());
  }

  function isSeparator(line) {
    if (!line || !line.includes("|") || !line.includes("-")) return false;
    return splitCells(line).every((cell) => SEPARATOR_CELL.test(cell));
  }

  function isTableStart(lines, index) {
    return lines[index].includes("|") && index + 1 < lines.length && isSeparator(lines[index + 1]);
  }

  function startsBlock(lines, index) {
    const line = lines[index];
    return FENCE.test(line) || RULE.test(line) || HEADING.test(line) || BULLET.test(line)
      || NUMBERED.test(line) || isTableStart(lines, index);
  }

  function readCode(lines, index, target) {
    const body = [];
    let cursor = index + 1;
    while (cursor < lines.length && !FENCE.test(lines[cursor])) {
      body.push(lines[cursor]);
      cursor += 1;
    }
    const pre = make("pre", "md-code");
    const code = make("code");
    code.textContent = body.join("\n");
    pre.appendChild(code);
    target.appendChild(pre);
    return cursor + 1;
  }

  function tableRow(cells, cellTag, width) {
    const row = make("tr");
    for (let column = 0; column < width; column += 1) {
      const cell = make(cellTag);
      if (cellTag === "th") cell.setAttribute("scope", "col");
      appendInline(cell, cells[column] || "");
      row.appendChild(cell);
    }
    return row;
  }

  function readTable(lines, index, target) {
    const header = splitCells(lines[index]);
    const head = make("thead");
    head.appendChild(tableRow(header, "th", header.length));
    const body = make("tbody");
    let cursor = index + 2;
    while (cursor < lines.length && lines[cursor].includes("|") && lines[cursor].trim()) {
      body.appendChild(tableRow(splitCells(lines[cursor]), "td", header.length));
      cursor += 1;
    }
    const table = make("table");
    table.append(head, body);
    const wrap = make("div", "md-table-wrap");
    wrap.appendChild(table);
    target.appendChild(wrap);
    return cursor;
  }

  function readList(lines, index, target, pattern, tag) {
    const list = make(tag);
    const first = pattern.exec(lines[index]);
    if (tag === "ol" && first && first[1] !== "1") list.setAttribute("start", String(Number(first[1])));
    let cursor = index;
    while (cursor < lines.length) {
      const match = pattern.exec(lines[cursor]);
      if (!match) break;
      const item = make("li");
      appendInline(item, match[match.length - 1]);
      list.appendChild(item);
      cursor += 1;
    }
    target.appendChild(list);
    return cursor;
  }

  function readParagraph(lines, index, target) {
    const paragraph = make("p");
    let cursor = index;
    while (cursor < lines.length && lines[cursor].trim() && (cursor === index || !startsBlock(lines, cursor))) {
      if (cursor > index) paragraph.appendChild(make("br"));
      appendInline(paragraph, lines[cursor]);
      cursor += 1;
    }
    target.appendChild(paragraph);
    return cursor;
  }

  function readBlock(lines, index, target) {
    const line = lines[index];
    if (FENCE.test(line)) return readCode(lines, index, target);
    if (RULE.test(line)) {
      target.appendChild(make("hr"));
      return index + 1;
    }
    const heading = HEADING.exec(line);
    if (heading) {
      const node = make(HEADING_TAGS[heading[1].length - 1], "md-heading");
      appendInline(node, heading[2]);
      target.appendChild(node);
      return index + 1;
    }
    if (isTableStart(lines, index)) return readTable(lines, index, target);
    if (BULLET.test(line)) return readList(lines, index, target, BULLET, "ul");
    if (NUMBERED.test(line)) return readList(lines, index, target, NUMBERED, "ol");
    return readParagraph(lines, index, target);
  }

  function renderMarkdown(text) {
    const fragment = document.createDocumentFragment();
    const lines = String(text ?? "").replace(/\r\n?/g, "\n").split("\n");
    let index = 0;
    while (index < lines.length) {
      if (!lines[index].trim()) {
        index += 1;
        continue;
      }
      index = readBlock(lines, index, fragment);
    }
    return fragment;
  }

  window.renderMarkdown = renderMarkdown;
})();

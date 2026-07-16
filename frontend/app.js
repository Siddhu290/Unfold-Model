/* Model X-Ray frontend. Talks only to the JSON API — all tensor data arrives
   pre-summarized (stats, histograms, downsampled heatmaps). */
"use strict";

const $ = (id) => document.getElementById(id);

const S = {
  session: null, arch: null, flat: {},
  trace: null, backward: null, step: null,
  selectedPath: null, playIdx: 0, timer: null,
  expandedGroups: new Set(), collapsed: new Set(),
  rowByPath: {}, tab: "forward",
  topo: null,               // {edges, calls} — true dataflow from real execution
  playDir: "forward",       // shared playback direction (tree + graph)
  graphMode: "structure",   // structure | activation | gradient | update
  graphDirty: true,
};

/* ---------------- API ---------------- */

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return res.json();
}
const post = (path, body) => api(path, {
  method: "POST", headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/* ---------------- banner ---------------- */

function banner(msg, kind = "info") {
  const el = $("banner");
  if (!msg) { el.classList.add("hidden"); return; }
  el.textContent = msg;
  el.className = kind;
}

/* ---------------- formatting ---------------- */

function fmtNum(x) {
  if (x === null || x === undefined) return "–";
  if (x === 0) return "0";
  const a = Math.abs(x);
  if (a >= 1e5 || a < 1e-4) return x.toExponential(2);
  return +x.toPrecision(4) + "";
}
function fmtCount(n) {
  if (n === null || n === undefined) return "–";
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return "" + n;
}
const fmtShape = (s) => (s ? "[" + s.join("×") + "]" : "?");
const esc = (t) => String(t).replace(/[&<>"]/g, (c) =>
  ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function classChip(cls) {
  const c = cls.toLowerCase();
  let k = "c-other";
  if (/linear|conv1d$|lazylinear/.test(c) && !/conv1d\d/.test(c)) k = "c-linear";
  if (/^conv|pool/.test(c)) k = "c-conv";
  if (/norm/.test(c)) k = "c-norm";
  if (/relu|gelu|sigmoid|tanh|softmax|activation|silu|elu|mish/.test(c)) k = "c-act";
  if (/attention|attn/.test(c)) k = "c-attn";
  if (/embed/.test(c)) k = "c-emb";
  if (/sequential|modulelist|moduledict|encoder|decoder|model|block/.test(c)) k = "c-container";
  return `<span class="tclass ${k}">${esc(cls)}</span>`;
}

/* ---------------- canvas renderers ---------------- */

function divergingColor(v, maxAbs) {
  const t = maxAbs > 0 ? Math.max(-1, Math.min(1, v / maxAbs)) : 0;
  // blue (neg) -> near-black (0) -> red/orange (pos)
  if (t >= 0) {
    const u = t;
    return [30 + 225 * u, 40 + 60 * u, 50 + 30 * u];
  }
  const u = -t;
  return [30 + 30 * u, 40 + 100 * u, 50 + 205 * u];
}

function renderHeatmap(container, hm, caption) {
  if (!hm) return;
  const { rows, cols, data } = hm;
  let maxAbs = 0;
  for (const row of data) for (const v of row) maxAbs = Math.max(maxAbs, Math.abs(v));
  const cv = document.createElement("canvas");
  cv.className = "heatmap";
  cv.width = cols; cv.height = rows;
  const scale = Math.max(2, Math.min(6, Math.floor(360 / cols)));
  cv.style.width = cols * scale + "px";
  cv.style.height = rows * scale + "px";
  const ctx = cv.getContext("2d");
  const img = ctx.createImageData(cols, rows);
  for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
    const [r, g, b] = divergingColor(data[i][j], maxAbs);
    const o = (i * cols + j) * 4;
    img.data[o] = r; img.data[o + 1] = g; img.data[o + 2] = b; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  container.appendChild(cv);
  const cap = document.createElement("div");
  cap.className = "heat-caption";
  cap.textContent = (caption || "") +
    `  ${rows}×${cols} view of ${fmtShape(hm.source_shape)} · blue −${fmtNum(maxAbs)} … red +${fmtNum(maxAbs)}`;
  container.appendChild(cap);
}

function renderHistogram(container, hist, caption) {
  if (!hist) return;
  const cv = document.createElement("canvas");
  cv.className = "hist"; cv.width = 330; cv.height = 84;
  const ctx = cv.getContext("2d");
  const counts = hist.counts, n = counts.length;
  const maxC = Math.max(...counts, 1);
  const bw = cv.width / n;
  for (let i = 0; i < n; i++) {
    const h = (counts[i] / maxC) * (cv.height - 14);
    ctx.fillStyle = "#58a6ff";
    ctx.fillRect(i * bw + 1, cv.height - 12 - h, bw - 2, h);
  }
  ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace";
  ctx.fillText(fmtNum(hist.min), 2, cv.height - 2);
  const maxTxt = fmtNum(hist.max);
  ctx.fillText(maxTxt, cv.width - ctx.measureText(maxTxt).width - 2, cv.height - 2);
  container.appendChild(cv);
  if (caption) {
    const cap = document.createElement("div");
    cap.className = "heat-caption"; cap.textContent = caption;
    container.appendChild(cap);
  }
}

function statsGrid(stats) {
  if (!stats) return "";
  const items = [["mean", stats.mean], ["std", stats.std], ["min", stats.min],
    ["max", stats.max], ["‖·‖₂", stats.l2_norm], ["zeros", stats.zero_frac != null ?
    (stats.zero_frac * 100).toFixed(1) + "%" : "–"]];
  return `<div class="statgrid">` + items.map(([k, v]) =>
    `<div><span>${k}</span>${typeof v === "string" ? v : fmtNum(v)}</div>`).join("") + `</div>`;
}

/* ---------------- loading models ---------------- */

async function loadDemos() {
  const demos = await api("/api/demos");
  $("demo-select").innerHTML = Object.entries(demos)
    .map(([k, v]) => `<option value="${k}">${esc(v)}</option>`).join("");
}

function busy(btn, on) { btn.disabled = on; }

async function onLoaded(resp) {
  S.session = resp.session; S.arch = resp.arch;
  S.trace = S.backward = S.step = null;
  S.topo = resp.topology || null;
  S.profile = null;
  S.playDir = "forward"; S.graphMode = "structure"; S.graphDirty = true;
  $("edit-compare").innerHTML = ""; $("mdiff-result").innerHTML = "";
  $("profile-result").innerHTML = ""; renderEditHistory([]);
  syncGraphChips();
  S.selectedPath = null; S.expandedGroups.clear(); S.collapsed.clear();
  $("welcome").classList.add("hidden");
  $("layout").classList.remove("hidden");
  resetResultStrip();
  const info = S.session;
  $("session-info").textContent =
    `${info.source} · ${info.root_class} · ${fmtCount(info.total_params)} params`;
  banner(info.warnings && info.warnings.length ? info.warnings.join(" ") : null,
    "warn");
  if (!info.runnable) {
    banner("Weights-only file: architecture inferred from parameter names. " +
      "Forward/backward execution needs a full model or a HuggingFace ID.", "info");
  }
  // input controls
  const isText = info.has_tokenizer;
  $("input-text-ctl").classList.toggle("hidden", !isText);
  $("input-tensor-ctl").classList.toggle("hidden", isText);
  const shape = (info.meta || {}).input_shape;
  if (shape) $("fw-shape").value = shape.join(",");
  $("fw-result").classList.add("hidden");
  $("bw-result").classList.add("hidden");
  $("step-result").classList.add("hidden");
  buildFlat(); renderTree(); renderDetail();
  $("arch-summary").textContent =
    `${fmtCount(S.arch.total_params)} params · ${S.arch.num_modules || "?"} modules`;
}

function buildFlat() {
  S.flat = {};
  (function visit(n) { S.flat[n.path] = n; n.children.forEach(visit); })(S.arch.tree);
}

/* ---------------- architecture tree ---------------- */

function renderTree() {
  const root = $("arch-tree");
  root.innerHTML = ""; S.rowByPath = {};
  root.appendChild(renderNode(S.arch.tree, true));
  // graph shares the expand/collapse state — keep it in lockstep
  S.graphDirty = true;
  if (S.tab === "graph") renderGraph();
}

function renderNode(node, isRoot) {
  const wrap = document.createElement("div");
  wrap.className = "tnode";
  const row = document.createElement("div");
  row.className = "trow";
  const hasKids = node.children.length > 0;
  const isCollapsed = S.collapsed.has(node.path) && !isRoot;
  const name = isRoot ? (S.arch.root_class || "model") :
    node.path.split(".").pop();
  row.innerHTML =
    `<span class="caret">${hasKids ? (isCollapsed ? "▸" : "▾") : "·"}</span>` +
    `<span class="tname">${esc(name)}</span>` + classChip(node.class) +
    (node.out_shape ? `<span class="tshape">→${fmtShape(node.out_shape)}</span>` : "") +
    (node.n_calls > 1 ? `<span class="tshape">×${node.n_calls} calls</span>` : "") +
    (node.own_param_count || node.total_param_count ?
      `<span class="tparams">${fmtCount(node.total_param_count)}</span>` : "");
  row.onclick = (e) => {
    if (e.target.classList.contains("caret") && hasKids) {
      S.collapsed.has(node.path) ? S.collapsed.delete(node.path) : S.collapsed.add(node.path);
      renderTree(); return;
    }
    selectNode(node.path);
  };
  S.rowByPath[node.path] = row;
  wrap.appendChild(row);

  if (hasKids && !isCollapsed) {
    const kids = document.createElement("div");
    kids.className = "tchildren";
    const groups = node.repeat_groups || [];
    let i = 0;
    while (i < node.children.length) {
      const g = groups.find((g) => g.start === i);
      const gkey = node.path + "#" + i;
      if (g && !S.expandedGroups.has(gkey)) {
        // collapsed repeat group: representative + ×N badge
        const rep = renderNode(node.children[i], false);
        const badge = document.createElement("span");
        badge.className = "repeat-badge";
        badge.textContent = `×${g.count} identical — expand`;
        badge.onclick = (e) => {
          e.stopPropagation(); S.expandedGroups.add(gkey); renderTree();
        };
        rep.querySelector(".trow").appendChild(badge);
        kids.appendChild(rep);
        i += g.count;
      } else {
        if (g) {
          const collapseBtn = document.createElement("div");
          collapseBtn.className = "repeat-badge"; collapseBtn.style.margin = "2px 0 2px 18px";
          collapseBtn.style.display = "inline-block";
          collapseBtn.textContent = `▾ collapse ×${g.count} group`;
          collapseBtn.onclick = () => { S.expandedGroups.delete(gkey); renderTree(); };
          kids.appendChild(collapseBtn);
          for (let j = i; j < i + g.count; j++) kids.appendChild(renderNode(node.children[j], false));
          i += g.count; continue;
        }
        kids.appendChild(renderNode(node.children[i], false));
        i++;
      }
    }
    wrap.appendChild(kids);
  }
  return wrap;
}

/* ---------------- selection + detail panel ---------------- */

async function selectNode(path) {
  S.selectedPath = path;
  document.querySelectorAll(".trow.selected").forEach((r) => r.classList.remove("selected"));
  const row = S.rowByPath[path];
  if (row) { row.classList.add("selected"); row.scrollIntoView({ block: "nearest" }); }
  renderDetail();
}

async function renderDetail() {
  const path = S.selectedPath;
  const empty = $("detail-empty"), body = $("detail-body");
  if (path === null || !S.flat[path]) {
    empty.classList.remove("hidden"); body.classList.add("hidden"); return;
  }
  empty.classList.add("hidden"); body.classList.remove("hidden");
  const node = S.flat[path];
  $("detail-title").textContent = path || S.arch.root_class || "model";
  $("detail-meta").innerHTML = `<div class="kv">
    <span class="k">class</span><span>${esc(node.class)} ${node.extra_repr ?
      `<span class="dim">(${esc(node.extra_repr)})</span>` : ""}</span>
    <span class="k">shape</span><span>${fmtShape(node.in_shape)} → ${fmtShape(node.out_shape)}</span>
    <span class="k">params</span><span>${fmtCount(node.own_param_count)} own · ${fmtCount(node.total_param_count)} incl. children</span>
    ${node.call_order != null ? `<span class="k">exec order</span><span>#${node.call_order}${node.n_calls > 1 ? ` (${node.n_calls} calls)` : ""}</span>` : ""}
  </div>`;

  // weights
  const wsec = $("weights-section"), psel = $("param-select");
  if (node.params.length) {
    wsec.classList.remove("hidden");
    psel.innerHTML = node.params.map((p) =>
      `<option value="${esc(p.name)}">${esc(p.name)} ${fmtShape(p.shape)}</option>`).join("");
    psel.onchange = () => loadWeight(path, psel.value);
    loadWeight(path, node.params[0].name);
  } else {
    wsec.classList.add("hidden");
  }

  // activation from last trace
  const asec = $("activation-section");
  const rec = S.trace && S.trace.records.find((r) => r.path === path);
  if (rec) {
    asec.classList.remove("hidden");
    const av = $("activation-view");
    av.innerHTML = `<div class="kv"><span class="k">out</span><span>${fmtShape(rec.out_shape)}</span></div>` +
      statsGrid(rec.output && rec.output.stats);
    if (rec.retained) {
      try {
        const d = await api(`/api/session/${S.session.session_id}/activation/${rec.call_index}`);
        renderHeatmap(av, d.heatmap, "activation");
        renderHistogram(av, d.histogram, "distribution of activation values");
      } catch (e) {}
    }
  } else asec.classList.add("hidden");

  // gradient for this node's params
  const gsec = $("grad-section");
  const gnames = S.backward ? node.params
    .map((p) => (path ? path + "." + p.name : p.name))
    .filter((n) => S.backward.param_grads[n]) : [];
  if (gnames.length) {
    gsec.classList.remove("hidden");
    const gv = $("grad-view"); gv.innerHTML = "";
    for (const n of gnames.slice(0, 2)) {
      const light = S.backward.param_grads[n];
      gv.insertAdjacentHTML("beforeend",
        `<div class="kv"><span class="k">∇${esc(n.split(".").pop())}</span>` +
        `<span>‖∇‖ = ${fmtNum(light.stats && light.stats.l2_norm)}</span></div>` +
        statsGrid(light.stats));
      try {
        const d = await api(`/api/session/${S.session.session_id}/grad?name=${encodeURIComponent(n)}`);
        renderHeatmap(gv, d.heatmap, `∂L/∂ ${n}`);
      } catch (e) {}
    }
  } else gsec.classList.add("hidden");

  lensChipsFor(path, $("detail-meta"));
  showTheory(node.class);

  // phase D/E/F/L/M integrations for the current selection
  $("edit-target").textContent = path || "(root — pick a child layer)";
  $("steer-layer").textContent = path || "(pick a block)";
  $("sae-layer").textContent = path || "(pick a block)";
  $("analysis-view").innerHTML = "";
  $("btn-ablate").classList.toggle("hidden", !/attn|attention/i.test(node.class));
}

async function loadWeight(path, param) {
  const wv = $("weight-view");
  wv.innerHTML = `<span class="dim">loading…</span>`;
  try {
    const d = await api(`/api/session/${S.session.session_id}/weight?path=${encodeURIComponent(path)}&param=${encodeURIComponent(param)}`);
    wv.innerHTML = statsGrid(d.stats);
    renderHeatmap(wv, d.heatmap, `${path ? path + "." : ""}${param}`);
    renderHistogram(wv, d.histogram, "weight value distribution");
    if (d.values) {
      wv.insertAdjacentHTML("beforeend",
        `<div class="heat-caption">values: [${d.values.map(fmtNum).join(", ")}]</div>`);
    }
  } catch (e) { wv.innerHTML = `<span class="dim">${esc(e.message)}</span>`; }
}

async function showTheory(className) {
  try {
    const t = await api(`/api/theory/${encodeURIComponent(className)}`);
    $("theory-view").innerHTML = theoryHTML(t, className);
  } catch (e) {}
}

function theoryHTML(t, requested) {
  return `<div class="theory-block">
    <h3>${esc(t.title)}${requested && t.key === "_generic" ?
      ` <span class="dim small">(${esc(requested)})</span>` : ""}</h3>
    <div class="theory-formula">${esc(t.formula)}</div>
    <h4>What it does</h4><p>${esc(t.what)}</p>
    <h4>Why it exists</h4><p>${esc(t.why)}</p>
    <h4>The gradient</h4><p>${esc(t.gradient)}</p>
  </div>`;
}

/* ---------------- forward pass ---------------- */

function inputSpec() {
  if (S.session.has_tokenizer) return { kind: "text", text: $("fw-text").value };
  const shape = $("fw-shape").value.split(",").map((s) => parseInt(s.trim(), 10));
  if (shape.some(isNaN)) throw new Error("Enter an input shape like 1,1,28,28");
  return { kind: "tensor", shape, fill: $("fw-fill").value };
}

async function runForward() {
  const btn = $("btn-forward"); busy(btn, true); banner(null);
  try {
    const resp = await post(`/api/session/${S.session.session_id}/forward`,
      { input: inputSpec(), topk: 5 });
    S.trace = resp;
    if (resp.edges) {
      S.topo = {
        edges: resp.edges,
        calls: resp.records.map((r) => ({
          call_index: r.call_index, path: r.path, class: r.class })),
      };
    }
    S.graphDirty = true; syncGraphChips();
    if (S.graphMode === "structure") setGraphMode("activation");
    const arch = await api(`/api/session/${S.session.session_id}/arch`);
    S.arch = arch; buildFlat(); renderTree();
    $("fw-result").classList.remove("hidden");
    const n = resp.records.length;
    $("pb-slider").max = n - 1; $("pb-slider").value = 0;
    $("pb2-slider").max = n - 1; $("pb2-slider").value = 0;
    S.playDir = "forward"; $("pb2-dir").value = "forward";
    setPlayIdx(0);
    renderOutputCard(resp);
    setupAttention(resp);
    updateResultStrip(resp);
  } catch (e) { banner("Forward failed: " + e.message, "error"); }
  busy(btn, false);
}

function renderOutputCard(resp) {
  const el = $("fw-output-card");
  let html = `<div class="card"><h3>Final output</h3>`;
  if (resp.output) {
    html += `<div class="kv"><span class="k">shape</span><span>${fmtShape(resp.output.shape)}</span></div>` +
      statsGrid(resp.output.stats);
  }
  if (resp.llm && resp.llm.topk) {
    html += `<h4 class="dim small" style="margin:8px 0 4px">TOP-5 ${resp.output && resp.output.shape.length === 3 ? "NEXT TOKEN" : "CLASS"} PROBABILITIES (softmax of final logits)</h4>`;
    const maxP = resp.llm.topk[0].prob || 1;
    for (const e2 of resp.llm.topk) {
      html += `<div class="tokbar"><span class="lbl">${esc(JSON.stringify(e2.label))}</span>
        <span class="bar" style="width:${Math.max(2, (e2.prob / maxP) * 240)}px"></span>
        <span class="pct">${(e2.prob * 100).toFixed(2)}%</span></div>`;
    }
  }
  if (resp.input && resp.input.tokens) {
    html += `<div class="heat-caption">input tokens: ${resp.input.tokens.map((t) => esc(t)).join(" | ")}</div>`;
  }
  el.innerHTML = html + `</div>`;
}

function setupAttention(resp) {
  const card = $("attn-card");
  if (!resp.attention) { card.classList.add("hidden"); return; }
  card.classList.remove("hidden");
  const L = resp.attention.num_layers, H = resp.attention.num_heads;
  $("attn-layer").innerHTML = Array.from({ length: L }, (_, i) => `<option>${i}</option>`).join("");
  $("attn-head").innerHTML = Array.from({ length: H }, (_, i) => `<option>${i}</option>`).join("");
  const draw = async () => {
    const d = await api(`/api/session/${S.session.session_id}/attention?layer=${$("attn-layer").value}&head=${$("attn-head").value}`);
    const cv = $("attn-canvas"), ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    const { rows, cols, data } = d.heatmap;
    const cw = cv.width / cols, ch = cv.height / rows;
    let maxV = 0;
    for (const r of data) for (const v of r) maxV = Math.max(maxV, v);
    for (let i = 0; i < rows; i++) for (let j = 0; j < cols; j++) {
      const u = maxV > 0 ? data[i][j] / maxV : 0;
      ctx.fillStyle = `rgb(${20 + 235 * u},${30 + 130 * u},${40 + 20 * u})`;
      ctx.fillRect(j * cw, i * ch, Math.ceil(cw), Math.ceil(ch));
    }
    $("attn-tokens").textContent = d.tokens ?
      "tokens (query=rows, key=cols): " + d.tokens.join(" · ") : "";
  };
  $("attn-layer").onchange = draw; $("attn-head").onchange = draw;
  draw().catch(() => {});
}

/* playback — one shared player drives the tree AND the graph */
function setPlayIdx(i) {
  if (!S.trace) return;
  const recs = S.trace.records;
  S.playIdx = Math.max(0, Math.min(recs.length - 1, i));
  const back = S.playDir === "backward";
  $("pb-slider").value = S.playIdx;
  $("pb-label").textContent = `${S.playIdx + 1} / ${recs.length}`;
  $("pb2-slider").value = S.playIdx;
  $("pb2-label").textContent = `${S.playIdx + 1} / ${recs.length}${back ? " ◀" : ""}`;
  const rec = recs[S.playIdx];
  // tree highlight (amber when replaying the backward direction)
  Object.values(S.rowByPath).forEach((r) =>
    r.classList.remove("exec-active", "exec-done", "bw"));
  const done = (j) => (back ? j > S.playIdx : j < S.playIdx);
  for (let j = 0; j < recs.length; j++) {
    const r = S.rowByPath[recs[j].path];
    if (!r) continue;
    if (j === S.playIdx) { r.classList.add("exec-active"); if (back) r.classList.add("bw"); }
    else if (done(j)) r.classList.add("exec-done");
  }
  const active = S.rowByPath[rec.path];
  if (active) active.scrollIntoView({ block: "nearest" });
  // step card
  $("fw-step-card").innerHTML = `<div class="card">
    <h3>step ${rec.call_index}: <code>${esc(rec.path || "(model output)")}</code> ${classChip(rec.class)}</h3>
    <div class="kv">
      <span class="k">input</span><span>${fmtShape(rec.in_shape)}</span>
      <span class="k">output</span><span>${fmtShape(rec.out_shape)}</span>
    </div>${statsGrid(rec.output && rec.output.stats)}
    <button onclick="window.__inspect(${rec.call_index})">🔍 inspect this activation</button>
  </div>`;
  if (S.tab === "forward") showTheory(rec.class);
  graphOnStep(S.playIdx, back);
}
window.__inspect = (callIdx) => {
  const rec = S.trace.records[callIdx];
  selectNode(rec.path);
};

function togglePlay() {
  const btns = [$("pb-play"), $("pb2-play")];
  if (S.timer) {
    clearInterval(S.timer); S.timer = null;
    btns.forEach((b) => b.textContent = "▶ play");
    return;
  }
  btns.forEach((b) => b.textContent = "⏸ pause");
  const back = S.playDir === "backward";
  const last = S.trace.records.length - 1;
  if (back ? S.playIdx <= 0 : S.playIdx >= last) setPlayIdx(back ? last : 0);
  S.timer = setInterval(() => {
    const next = S.playIdx + (back ? -1 : 1);
    if (next < 0 || next > last) togglePlay();
    else setPlayIdx(next);
  }, 320);
}

/* ---------------- backward pass ---------------- */

function targetSpec() {
  const kind = $("bw-target-kind").value;
  const val = $("bw-target-value").value;
  if (kind === "class") return { kind: "class", index: parseInt(val || "0", 10) };
  if (kind === "token") return { kind: "token", text: val || " " };
  return { kind: "argmax" };
}

async function runBackward() {
  const btn = $("btn-backward"); busy(btn, true); banner(null);
  try {
    const resp = await post(`/api/session/${S.session.session_id}/backward`,
      { input: inputSpec(), target: targetSpec() });
    S.backward = resp;
    syncGraphChips(); setGraphMode("gradient");
    $("pb2-dir").querySelector('[value="backward"]').disabled = false;
    $("bw-result").classList.remove("hidden");
    const ld = resp.loss_desc;
    $("bw-loss-card").innerHTML = `<div class="card">
      <h3>Loss = ${fmtNum(resp.loss)}</h3>
      <div class="kv">
        <span class="k">loss fn</span><span>${esc(ld.loss_fn)}</span>
        <span class="k">target</span><span>${esc(ld.target_label)} <span class="dim">(${esc(ld.target_kind)})</span></span>
      </div>
      <p class="dim small">loss.backward() has run: every parameter now holds its real
      ∂L/∂θ in .grad — these exact tensors feed the optimizer step in the Δ Update tab.</p>
    </div>`;
    renderGradBars(resp);
    showTheory("_backprop");
    if (S.selectedPath !== null) renderDetail();
  } catch (e) { banner("Backward failed: " + e.message, "error"); }
  busy(btn, false);
}

function renderGradBars(resp) {
  const el = $("grad-bars"); el.innerHTML = "";
  const entries = Object.entries(resp.layer_grad_norms);
  // order by execution order (reverse = backward flow direction) when known
  entries.sort((a, b) => {
    const na = S.flat[a[0]], nb = S.flat[b[0]];
    const oa = na && na.call_order != null ? na.call_order : 1e9;
    const ob = nb && nb.call_order != null ? nb.call_order : 1e9;
    return ob - oa;
  });
  const max = Math.max(...entries.map(([, v]) => v), 1e-12);
  const min = Math.min(...entries.filter(([, v]) => v > 0).map(([, v]) => v));
  for (const [name, v] of entries) {
    const w = Math.max(2, (v / max) * 420);
    const row = document.createElement("div");
    row.className = "gradbar-row";
    row.innerHTML = `<span class="gname" title="${esc(name)}">${esc(name)}</span>
      <span class="gbar" style="width:${w}px"></span>
      <span class="gval">${fmtNum(v)}</span>`;
    row.onclick = () => selectNode(S.flat[name] ? name : name);
    el.appendChild(row);
  }
  const ratio = max / (min || max);
  let diag = `Largest ‖∇‖ is ${fmtNum(ratio)}× the smallest. `;
  if (ratio > 1e4) diag += "That spread is large — layers at the small end are barely learning (vanishing-gradient territory). Common causes: saturating activations, depth without residual connections, or a tiny learning signal at that depth.";
  else if (ratio < 50) diag += "Gradients are well balanced across depth — a healthy learning signal at every layer (residual connections and normalization usually deserve the credit).";
  else diag += "A moderate spread — typical for this kind of architecture. Early layers naturally receive somewhat smaller gradients as the signal is chain-ruled through more factors.";
  $("grad-diagnosis").textContent = diag;
}

/* ---------------- optimizer step ---------------- */

async function runStep() {
  const btn = $("btn-step"); busy(btn, true); banner(null);
  try {
    const resp = await post(`/api/session/${S.session.session_id}/step`,
      { optimizer: $("opt-name").value, lr: parseFloat($("opt-lr").value) });
    S.step = resp;
    syncGraphChips(); setGraphMode("update");
    pulseUpdatedNodes();
    $("step-result").classList.remove("hidden");
    $("step-summary").innerHTML = `<div class="card">
      <h3>Applied one real ${esc(resp.optimizer.toUpperCase())} step (lr=${resp.lr})</h3>
      <p class="dim small">θ ← θ − lr·(update). The model's weights HAVE changed —
      run the forward pass again to see the new predictions, or Undo to restore.</p></div>`;
    renderStepTable(resp);
    showTheory("_gradient_descent");
  } catch (e) { banner("Step failed: " + e.message, "error"); }
  busy(btn, false);
}

function renderStepTable(resp) {
  const rows = Object.entries(resp.param_diffs)
    .sort((a, b) => b[1].update_norm - a[1].update_norm);
  const maxU = Math.max(...rows.map(([, d]) => d.update_norm), 1e-12);
  let html = `<table class="difftable"><tr><th>parameter</th><th>shape</th>
    <th>‖∇‖</th><th>‖Δw‖</th><th>Δ/‖w‖</th><th></th></tr>`;
  for (const [name, d] of rows.slice(0, 40)) {
    html += `<tr data-name="${esc(name)}"><td>${esc(name)}</td><td>${fmtShape(d.shape)}</td>
      <td>${fmtNum(d.grad_norm)}</td><td>${fmtNum(d.update_norm)}</td>
      <td>${d.relative_update != null ? (d.relative_update * 100).toFixed(3) + "%" : "–"}</td>
      <td><span class="deltabar" style="width:${Math.max(2, (d.update_norm / maxU) * 120)}px"></span></td></tr>`;
  }
  html += `</table>`;
  if (rows.length > 40) html += `<div class="heat-caption">showing top 40 of ${rows.length} parameters by update size</div>`;
  $("step-table").innerHTML = html;
  document.querySelectorAll("#step-table tr[data-name]").forEach((tr) => {
    tr.onclick = () => showDiff(tr.dataset.name);
  });
}

async function showDiff(name) {
  const el = $("diff-view");
  el.classList.remove("hidden");
  el.innerHTML = `<div class="card"><h3>Δ ${esc(name)}</h3><span class="dim">loading…</span></div>`;
  try {
    const d = await api(`/api/session/${S.session.session_id}/diff?name=${encodeURIComponent(name)}`);
    const card = document.createElement("div"); card.className = "card";
    card.innerHTML = `<h3>${esc(name)} — before / after / change</h3>
      <p class="dim small">The Δ heatmap is −lr × the optimizer's update: compare its
      pattern with this parameter's gradient heatmap (select the layer, Gradient
      section) — with SGD they are identical up to scale.</p>`;
    const trio = document.createElement("div"); trio.className = "heatmap-trio";
    for (const [key, label] of [["before", "w before"], ["after", "w after"], ["delta", "Δw (after − before)"]]) {
      const cell = document.createElement("div");
      renderHeatmap(cell, d[key].heatmap, label);
      trio.appendChild(cell);
    }
    card.appendChild(trio);
    card.insertAdjacentHTML("beforeend", statsGrid(d.delta.stats));
    el.innerHTML = ""; el.appendChild(card);
  } catch (e) { el.innerHTML = `<div class="card dim">${esc(e.message)}</div>`; }
}

async function runUndo() {
  try {
    await post(`/api/session/${S.session.session_id}/undo`, {});
    banner("Weights restored to their pre-step values.", "info");
    $("step-result").classList.add("hidden");
  } catch (e) { banner(e.message, "error"); }
}

/* ---------------- tabs + wiring ---------------- */

function switchTab(name) {
  S.tab = name;
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  for (const t of ["forward", "backward", "update", "graph", "patch", "edit",
                   "steer", "sae", "profile"]) {
    $("tab-" + t).classList.toggle("hidden", t !== name);
  }
  if (name === "backward") showTheory("_backprop");
  if (name === "update") showTheory("_gradient_descent");
  if (name === "graph") renderGraph();
  if (name === "patch" && !$("patch-clean").value) {
    $("patch-clean").value = $("fw-text").value;
  }
}

/* ---------------- result strip + streaming generation (phase A) -------- */

let genAbort = null;

function isLM() {
  return !!(S.session && S.session.has_tokenizer);
}

function resetResultStrip() {
  $("result-strip").classList.remove("hidden");
  $("rs-gen-controls").classList.toggle("hidden", !isLM());
  $("btn-lens").classList.add("hidden");
  $("btn-dist").classList.add("hidden");
  $("rs-verdict").innerHTML =
    `<span class="dim">run a forward pass to see the model's answer</span>`;
  $("rs-tokens").innerHTML = "";
  $("dist-panel").classList.add("hidden");
  $("lens-panel").classList.add("hidden");
  $("attr-panel").classList.add("hidden");
  $("attr-view").innerHTML = "";
  $("btn-attr").classList.add("hidden");
  $("btn-report").classList.remove("hidden");
  $("circuit-result").classList.add("hidden");
  $("train-result").classList.add("hidden");
  $("train-corpus-wrap").classList.toggle("hidden", !isLM());
  S.lensByPath = null;
  S.circuit = null; S.circuitData = null;
  $("steer-run").classList.add("hidden");
  $("steer-result").innerHTML = ""; $("steer-batch-result").innerHTML = "";
  $("steer-dir-info").textContent = "";
  $("sae-result").classList.add("hidden");
  $("sae-features").innerHTML = ""; $("sae-curve").innerHTML = "";
}

function updateResultStrip(resp) {
  $("btn-dist").classList.remove("hidden");
  $("btn-attr").classList.remove("hidden");
  const v = $("rs-verdict");
  if (!resp.llm || !resp.llm.topk.length) return;
  const t = resp.llm.topk[0];
  if (resp.output && resp.output.shape.length === 3) {   // language model
    v.innerHTML = `<span class="dim">model's next token:</span> ` +
      `<b>${esc(JSON.stringify(t.label))}</b> ${(t.prob * 100).toFixed(1)}%`;
    $("btn-lens").classList.remove("hidden");
  } else {                                               // classifier
    v.innerHTML = `<span class="dim">prediction:</span> <b>${esc(t.label)}</b>` +
      `<span class="conf-bar" style="width:${Math.max(4, t.prob * 140)}px"></span>` +
      `${(t.prob * 100).toFixed(1)}% confidence`;
  }
}

async function runGenerate() {
  if (genAbort) {   // acting as a pause button mid-stream
    genAbort.abort(); genAbort = null;
    $("btn-generate").textContent = "▶ generate";
    return;
  }
  const prompt = $("fw-text").value;
  S.genTokens = []; S.genPrompt = prompt;
  $("rs-tokens").innerHTML = "";
  $("rs-verdict").innerHTML = `<span class="dim">generating from</span> ` +
    `<code>${esc(prompt.length > 42 ? "…" + prompt.slice(-40) : prompt)}</code>`;
  genAbort = new AbortController();
  $("btn-generate").textContent = "⏸ pause";
  try {
    const res = await fetch(`/api/session/${S.session.session_id}/generate`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      signal: genAbort.signal,
      body: JSON.stringify({
        input: { kind: "text", text: prompt },
        max_new_tokens: parseInt($("gen-n").value, 10) || 12,
        mode: $("gen-mode").value,
        temperature: parseFloat($("gen-temp").value) || 1.0,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (line) handleGenEvent(JSON.parse(line));
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") banner("Generation failed: " + e.message, "error");
  }
  genAbort = null;
  $("btn-generate").textContent = "▶ generate";
}

function handleGenEvent(ev) {
  if (ev.event === "token") {
    const k = S.genTokens.length;
    S.genTokens.push(ev);
    const chip = document.createElement("span");
    chip.className = "tok-chip";
    chip.textContent = ev.token;
    chip.title = `p = ${(ev.prob * 100).toFixed(2)}% · click to inspect the ` +
      `forward pass that produced this token`;
    chip.style.background = `rgba(63,185,80,${(0.08 + 0.5 * Math.min(1, ev.prob)).toFixed(3)})`;
    chip.onclick = () => inspectAtToken(k);
    $("rs-tokens").appendChild(chip);
  } else if (ev.event === "error") {
    banner(ev.detail, "error");
  }
}

function inspectAtToken(k) {
  if (genAbort) { genAbort.abort(); genAbort = null; $("btn-generate").textContent = "▶ generate"; }
  const prefix = S.genPrompt + S.genTokens.slice(0, k).map((t) => t.token).join("");
  $("fw-text").value = prefix;
  switchTab("forward");
  runForward();
  banner(`Inspecting the forward pass that chose token ` +
    `${JSON.stringify(S.genTokens[k].token)} (generation step ${k + 1}). ` +
    `The graph/tree/inspector now show exactly that computation.`, "info");
}

/* --- full distribution: windowed (virtualized) list over the whole vocab --- */

const DIST = { total: 0, rows: [], rowH: 22, page: 1000, pending: new Set() };

async function toggleDist() {
  const panel = $("dist-panel");
  if (!panel.classList.contains("hidden")) { panel.classList.add("hidden"); return; }
  panel.classList.remove("hidden");
  DIST.total = 0; DIST.rows = []; DIST.pending.clear();
  $("dist-rows").innerHTML = "";
  await distFetch(0);
  $("dist-total").textContent = DIST.total.toLocaleString();
  $("dist-spacer").style.height = DIST.total * DIST.rowH + "px";
  $("dist-scroll").scrollTop = 0;
  distRender();
}

async function distFetch(page) {
  if (DIST.pending.has(page)) return;
  DIST.pending.add(page);
  try {
    const d = await api(`/api/session/${S.session.session_id}/distribution` +
      `?offset=${page * DIST.page}&limit=${DIST.page}`);
    DIST.total = d.total;
    for (const e of d.entries) DIST.rows[e.rank] = e;
  } catch (e) { banner(e.message, "error"); }
}

async function distRender() {
  const sc = $("dist-scroll");
  const start = Math.max(0, Math.floor(sc.scrollTop / DIST.rowH) - 5);
  const end = Math.min(DIST.total, start + Math.ceil(sc.clientHeight / DIST.rowH) + 10);
  for (let p = Math.floor(start / DIST.page); p <= Math.floor(end / DIST.page); p++) {
    if (DIST.rows[p * DIST.page] === undefined) await distFetch(p);
  }
  const maxP = DIST.rows[0] ? DIST.rows[0].prob : 1;
  const holder = $("dist-rows");
  holder.style.transform = `translateY(${start * DIST.rowH}px)`;
  let html = "";
  for (let i = start; i < end; i++) {
    const r = DIST.rows[i];
    if (!r) continue;
    html += `<div class="dist-row"><span class="dr-rank">#${r.rank + 1}</span>` +
      `<span class="dr-label">${esc(JSON.stringify(r.label))}</span>` +
      `<span class="dr-bar" style="width:${Math.max(1, (r.prob / maxP) * 260)}px"></span>` +
      `<span class="dr-prob">${(r.prob * 100).toPrecision(3)}%</span></div>`;
  }
  holder.innerHTML = html;
}

/* ---------------- logit lens (phase B) ---------------- */

async function runLens() {
  const btn = $("btn-lens"); busy(btn, true);
  try {
    const d = await api(`/api/session/${S.session.session_id}/logit_lens?k=3`,
      { method: "POST" });
    renderLens(d);
  } catch (e) { banner("Logit lens failed: " + e.message, "error"); }
  busy(btn, false);
}

function renderLens(d) {
  $("lens-panel").classList.remove("hidden");
  S.lensByPath = {};
  const strip = $("lens-strip");
  strip.innerHTML = "";
  const cells = d.rows.map((r) => ({ ...r, name: r.stage === "embedding" ? "embed" : r.path }));
  cells.forEach((r) => {
    S.lensByPath[r.path] = r;
    const top = r.topk[0];
    const cell = document.createElement("div");
    cell.className = "lens-cell";
    cell.style.background = `rgba(63,185,80,${(0.04 + 0.5 * top.prob).toFixed(3)})`;
    cell.innerHTML = `<div class="lc-layer">${esc(r.name)}</div>` +
      `<div class="lc-tok">${esc(JSON.stringify(top.label))}</div>` +
      `<div class="lc-prob">${(top.prob * 100).toFixed(1)}%</div>`;
    cell.title = r.topk.map((t) => `${JSON.stringify(t.label)} ${(t.prob * 100).toFixed(1)}%`).join("\n");
    cell.onclick = () => selectNode(r.path);
    strip.appendChild(cell);
    strip.insertAdjacentHTML("beforeend", `<span class="lens-arrow">→</span>`);
  });
  if (d.final && d.final.length) {
    const f = d.final[0];
    const cell = document.createElement("div");
    cell.className = "lens-cell final";
    cell.innerHTML = `<div class="lc-layer">real output</div>` +
      `<div class="lc-tok">${esc(JSON.stringify(f.label))}</div>` +
      `<div class="lc-prob">${(f.prob * 100).toFixed(1)}%</div>`;
    strip.appendChild(cell);
  }
}

async function lensChipsFor(path, container) {
  // inspector integration: lens prediction for the selected layer
  if (!isLM() || !S.trace) return;
  const rec = S.trace.records.find((r) => r.path === path && r.retained);
  if (!rec || !rec.out_shape || rec.out_shape.length !== 3) return;
  try {
    const d = await api(`/api/session/${S.session.session_id}/logit_lens_one` +
      `?call_index=${rec.call_index}&k=5`);
    const div = document.createElement("div");
    div.style.padding = "6px 12px";
    div.innerHTML = `<span class="dim small">🔬 logit lens — this layer would predict: </span>` +
      d.topk.map((t) =>
        `<span class="lens-chip">${esc(JSON.stringify(t.label))} ${(t.prob * 100).toFixed(1)}%</span>`
      ).join("");
    container.appendChild(div);
  } catch (e) { /* layer isn't a (B, T, d_model) hidden state — no lens */ }
}

/* ---------------- activation patching (phase C) ---------------- */

function patchInputsOk() {
  if (!$("patch-clean").value.trim() || !$("patch-corr").value.trim()) {
    banner("Fill in BOTH prompt boxes — the grey text in an empty box is just " +
      "an example. The two prompts should differ in one subject and tokenize " +
      "to the same length (e.g. “…Paris is located in…” vs “…Rome is located in…”).",
      "warn");
    return false;
  }
  return true;
}

async function runPatch() {
  if (!patchInputsOk()) return;
  const btn = $("btn-patch"); busy(btn, true); banner(null);
  try {
    const target = $("patch-target").value.trim();
    const d = await post(`/api/session/${S.session.session_id}/patch`, {
      clean: { kind: "text", text: $("patch-clean").value },
      corrupted: { kind: "text", text: $("patch-corr").value },
      target: target ? { kind: "token", text: target } : {},
      positions: $("patch-pos").value,
    });
    renderPatch(d);
  } catch (e) { banner("Patch failed: " + e.message, "error"); }
  busy(btn, false);
}

function renderPatch(d) {
  $("patch-result").classList.remove("hidden");
  const diffs = d.diff_tokens.map((t) =>
    `pos ${t.pos}: ${JSON.stringify(t.clean)} → ${JSON.stringify(t.corrupted)}`).join(" · ");
  $("patch-summary").innerHTML = `<div class="card">
    <h3>target ${esc(JSON.stringify(d.target.label))}</h3>
    <div class="kv">
      <span class="k">clean run</span><span>p(target) = ${(d.clean.p_target * 100).toFixed(2)}% · top-1 ${esc(JSON.stringify(d.clean.top1))}</span>
      <span class="k">corrupted run</span><span>p(target) = ${(d.corrupted.p_target * 100).toFixed(2)}% · top-1 ${esc(JSON.stringify(d.corrupted.top1))}</span>
      <span class="k">patched tokens</span><span>${esc(diffs) || "(all positions)"}</span>
    </div></div>`;
  const el = $("patch-bars"); el.innerHTML = "";
  const vals = d.results.filter((r) => r.restoration !== null && r.restoration !== undefined);
  const maxAbs = Math.max(...vals.map((r) => Math.abs(r.restoration)), 1);
  for (const r of d.results) {
    const row = document.createElement("div");
    row.className = "gradbar-row";
    if (r.error) {
      row.innerHTML = `<span class="gname">${esc(r.path)}</span><span class="dim small">${esc(r.error)}</span>`;
    } else {
      const w = Math.max(2, (Math.abs(r.restoration) / maxAbs) * 380);
      const col = r.restoration >= 0 ? "linear-gradient(90deg,#1f6feb,#79c0ff)"
                                     : "linear-gradient(90deg,#b62324,#f85149)";
      row.innerHTML = `<span class="gname" title="${esc(r.path)}">${esc(r.path)}</span>
        <span class="gbar" style="width:${w}px;background:${col}"></span>
        <span class="gval">${(r.restoration * 100).toFixed(1)}%${r.flipped_back ? " ✔ flips back" : ""}</span>`;
      row.onclick = () => selectNode(r.path);
    }
    el.appendChild(row);
  }
}

/* ---------------- input attribution (phase J) ---------------- */

async function runAttribution() {
  const btn = $("btn-attr-run"); busy(btn, true);
  const method = $("attr-method").value;
  $("attr-view").innerHTML = `<span class="dim">computing ${method === "ig" ?
    "16 interpolated forward+backward passes" : "one forward+backward pass"}…</span>`;
  try {
    const d = await post(`/api/session/${S.session.session_id}/attribution`,
      { input: inputSpec(), method, steps: 16,
        contrast: $("attr-contrast").value.trim() || null });
    renderAttribution(d);
  } catch (e) { $("attr-view").innerHTML = `<span class="dim">${esc(e.message)}</span>`; }
  busy(btn, false);
}

function renderAttribution(d) {
  const el = $("attr-view");
  const head = `<div class="dim small" style="margin-bottom:6px">
    ${d.method === "ig" ? `integrated gradients (${d.steps} steps, baseline: ${esc(d.baseline || "zeros")})` :
      "saliency (gradient × input at the embedding)"} · target
    <b>${esc(JSON.stringify(d.target.label))}</b>${d.contrast ?
      ` vs <b>${esc(JSON.stringify(d.contrast))}</b>` : ""}
    ${d.embedding_layer ? ` · measured at <code>${esc(d.embedding_layer)}</code>` : ""}</div>`;
  if (d.kind === "text") {
    const maxAbs = Math.max(...d.scores.map((s) => Math.abs(s.score)), 1e-12);
    el.innerHTML = head + d.scores.map((s) => {
      const t = s.score / maxAbs;
      const bg = t >= 0 ? `rgba(63,185,80,${(0.10 + 0.65 * t).toFixed(3)})`
                        : `rgba(248,81,73,${(0.10 + 0.65 * -t).toFixed(3)})`;
      return `<span class="attr-tok" style="background:${bg}"
        title="score ${fmtNum(s.score)} · ${(s.frac * 100).toFixed(1)}% of total mass">${esc(s.token)}</span>`;
    }).join("") +
    `<div class="heat-caption" style="margin-top:6px">green pushes the answer up, red pushes it down · hover for exact share</div>` +
    (d.completeness ? `<div class="dim small">completeness: Σattr = ${fmtNum(d.completeness.sum_attributions)}
      vs Δlogit = ${fmtNum(d.completeness.difference)}</div>` : "");
  } else {
    el.innerHTML = head;
    renderHeatmap(el, d.map.heatmap, `${d.method} over the input`);
    if (d.completeness) {
      el.insertAdjacentHTML("beforeend", `<div class="dim small">completeness:
        Σattr = ${fmtNum(d.completeness.sum_attributions)} vs Δlogit =
        ${fmtNum(d.completeness.difference)}</div>`);
    }
  }
}

/* ---------------- circuit discovery (phase H) ---------------- */

let circuitAbort = null;

async function runCircuit() {
  if (circuitAbort) { circuitAbort.abort(); circuitAbort = null;
    $("btn-circuit").textContent = "Discover circuit (layer × head sweep)"; return; }
  if (!patchInputsOk()) return;
  banner(null);
  circuitAbort = new AbortController();
  $("btn-circuit").textContent = "⏸ cancel sweep";
  const prog = $("circuit-progress");
  prog.textContent = "starting…";
  try {
    const res = await fetch(`/api/session/${S.session.session_id}/circuit`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      signal: circuitAbort.signal,
      body: JSON.stringify({
        clean: { kind: "text", text: $("patch-clean").value },
        corrupted: { kind: "text", text: $("patch-corr").value },
        target: $("patch-target").value.trim() ?
          { kind: "token", text: $("patch-target").value } : {},
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "", meta = null, doneCount = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.event === "start") {
          meta = ev;
          prog.textContent = `${ev.n_runs} patched runs, ~${ev.estimate_s}s estimated…`;
        } else if (ev.event === "layer" || ev.event === "head") {
          doneCount++;
          if (meta) prog.textContent =
            `${doneCount}/${meta.n_runs} runs · ${ev.event} ` +
            `${ev.path.split(".").pop()}${ev.event === "head" ? " H" + ev.head : ""} → ` +
            `${ev.restoration != null ? (ev.restoration * 100).toFixed(0) + "%" : "–"}`;
        } else if (ev.event === "done") {
          S.circuitData = ev;
          prog.textContent = `done in ${ev.elapsed_s}s`;
          renderCircuitGrid(ev);
        } else if (ev.event === "error") {
          throw new Error(ev.detail);
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") banner("Circuit sweep failed: " + e.message, "error");
  }
  circuitAbort = null;
  $("btn-circuit").textContent = "Discover circuit (layer × head sweep)";
}

function renderCircuitGrid(d) {
  $("circuit-result").classList.remove("hidden");
  const cv = $("circuit-grid");
  const L = d.layers.length, H = d.n_heads;
  const cell = Math.max(10, Math.min(22, Math.floor(640 / (H + 3))));
  cv.width = cell * 2 + 6 + H * cell;
  cv.height = L * cell;
  const ctx = cv.getContext("2d");
  const all = [...d.layer_curve, ...d.matrix.flat()].filter((v) => v != null);
  const maxAbs = Math.max(...all.map(Math.abs), 1e-9);
  const color = (v) => {
    if (v == null) return "#3a3a3a";
    const t = Math.max(-1, Math.min(1, v / maxAbs));
    return t >= 0 ? `rgb(${30 + 225 * t},${40 + 60 * t},${50 + 30 * t})`
                  : `rgb(${30 - 30 * t},${40 - 100 * t},${50 - 205 * t})`;
  };
  for (let i = 0; i < L; i++) {
    ctx.fillStyle = color(d.layer_curve[i]);
    ctx.fillRect(0, i * cell, cell * 2, cell - 1);        // whole-layer column
    for (let h = 0; h < H; h++) {
      ctx.fillStyle = color(d.matrix[i][h]);
      ctx.fillRect(cell * 2 + 6 + h * cell, i * cell, cell - 1, cell - 1);
    }
  }
  cv.title = "left block: whole-layer patch · grid: per-head patches";
  updateCircuitSelection();
}

function circuitSelection(thresh) {
  const d = S.circuitData;
  const layers = [], heads = {};
  d.layers.forEach((p, i) => {
    if (d.layer_curve[i] != null && d.layer_curve[i] >= thresh) layers.push(p);
    const hs = (d.matrix[i] || []).map((v, h) => [v, h])
      .filter(([v]) => v != null && v >= thresh).map(([, h]) => h);
    if (hs.length) heads[p] = hs;
  });
  return { layers, heads };
}

function updateCircuitSelection() {
  if (!S.circuitData) return;
  const thresh = (+$("circuit-thresh").value) / 100;
  $("circuit-thresh-label").textContent = $("circuit-thresh").value + "%";
  const sel = circuitSelection(thresh);
  const parts = [];
  for (const p of sel.layers) {
    const hs = sel.heads[p];
    parts.push(`${p}${hs ? ` [heads ${hs.join(",")}]` : ""}`);
  }
  for (const [p, hs] of Object.entries(sel.heads)) {
    if (!sel.layers.includes(p)) parts.push(`${p} [heads ${hs.join(",")} only]`);
  }
  $("circuit-members").textContent = parts.length ?
    `circuit at ≥${$("circuit-thresh").value}% restoration: ` + parts.join(" · ") :
    "no layers/heads above this threshold";
  if (S.circuit) applyCircuitOverlay();     // live-update an active overlay
}

function applyCircuitOverlay() {
  if (!G.built) return;
  G.nodeEls.forEach((el) => el.classList.remove("in-circuit"));
  G.edgeEls.forEach((el) => el.classList.remove("in-circuit"));
  if (!S.circuit || !S.circuitData) return;
  const thresh = (+$("circuit-thresh").value) / 100;
  const sel = circuitSelection(thresh);
  const { map } = buildVisibleMap();
  const marked = new Set();
  for (const p of [...sel.layers, ...Object.keys(sel.heads)]) {
    const vid = map[p];
    if (vid !== undefined && G.nodeEls.has(vid)) {
      G.nodeEls.get(vid).classList.add("in-circuit");
      marked.add(vid);
    }
  }
  G.edgeEls.forEach((el, key) => {
    const [a, b] = key.split("→");
    if (marked.has(a) && marked.has(b)) el.classList.add("in-circuit");
  });
}

/* ---------------- training loop (phase I) ---------------- */

let trainAbort = null;

async function runTrain() {
  if (trainAbort) {
    trainAbort.abort(); trainAbort = null;
    $("btn-train").textContent = "▶ train";
    banner("Training paused — the weights are live at the last completed step; " +
      "use the Forward/Backward tabs to inspect this exact moment.", "info");
    return;
  }
  banner(null);
  trainAbort = new AbortController();
  $("btn-train").textContent = "⏸ pause";
  $("train-result").classList.remove("hidden");
  $("train-diff-view").innerHTML = "";
  const losses = [];
  const status = $("train-status");
  try {
    const source = isLM()
      ? { kind: "corpus", text: $("train-corpus").value || undefined }
      : { kind: "quadrant" };
    const res = await fetch(`/api/session/${S.session.session_id}/train`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      signal: trainAbort.signal,
      body: JSON.stringify({
        steps: parseInt($("train-steps").value, 10) || 50,
        optimizer: $("opt-name").value, lr: parseFloat($("opt-lr").value) || 0.01,
        checkpoint_every: parseInt($("train-ck").value, 10) || 10,
        source,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.event === "start") {
          status.textContent = `training on the ${ev.task} task · batch ${ev.batch_size}` +
            (ev.note ? ` · ${ev.note}` : "");
        } else if (ev.event === "step") {
          losses.push(ev.loss);
          status.textContent = `step ${ev.i} · loss ${fmtNum(ev.loss)}`;
          if (losses.length % 2 === 0 || losses.length < 5) drawLossCurve(losses);
        } else if (ev.event === "done") {
          drawLossCurve(losses);
          status.textContent = `done: loss ${fmtNum(ev.initial_loss)} → ${fmtNum(ev.final_loss)} ` +
            `over ${losses.length} real steps · checkpoints at [${ev.checkpoints.join(", ")}]`;
          setupScrubber(ev.checkpoints);
        } else if (ev.event === "error") {
          throw new Error(ev.detail);
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") banner("Training failed: " + e.message, "error");
  }
  trainAbort = null;
  $("btn-train").textContent = "▶ train";
  if (S.trace) runForward();   // refresh the strip/trace with trained weights
}

function drawLossCurve(losses) {
  const holder = $("train-curve");
  holder.innerHTML = "";
  linePlot(holder, losses.map((_, i) => i), losses,
    { label: `loss over ${losses.length} steps`, color: "#3fb950" });
}

function setupScrubber(checkpoints) {
  S.trainCheckpoints = checkpoints;
  const wrap = $("train-scrub-wrap");
  wrap.style.display = checkpoints.length > 1 ? "flex" : "none";
  const sc = $("train-scrub");
  sc.max = checkpoints.length - 1;
  sc.value = checkpoints.length - 1;
  $("train-scrub-label").textContent = `step ${checkpoints[checkpoints.length - 1]}`;
  sc.oninput = () => {
    $("train-scrub-label").textContent = `step ${checkpoints[+sc.value]}`;
  };
}

async function trainRestore() {
  const step = S.trainCheckpoints[+$("train-scrub").value];
  try {
    await post(`/api/session/${S.session.session_id}/train_restore?step=${step}`, {});
    S.graphDirty = true;
    if (S.trace) await runForward();
    banner(`Weights restored to training step ${step} — the forward pass and ` +
      `result strip now show that checkpoint's behavior.`, "info");
  } catch (e) { banner(e.message, "error"); }
}

async function trainDiff() {
  const step = S.trainCheckpoints[+$("train-scrub").value];
  try {
    const d = await post(`/api/session/${S.session.session_id}/train_diff?step=${step}`, {});
    const rows = Object.entries(d.param_diffs).sort((a, b) => b[1].update_norm - a[1].update_norm);
    const maxU = Math.max(...rows.map(([, v]) => v.update_norm), 1e-12);
    let html = `<div class="card"><h3>current weights vs ${esc(d.other)}</h3></div>
      <table class="difftable"><tr><th>parameter</th><th>shape</th><th>‖Δ‖</th><th>Δ/‖w‖</th><th></th></tr>`;
    for (const [name, v] of rows.slice(0, 25)) {
      html += `<tr data-name="${esc(name)}"><td>${esc(name)}</td><td>${fmtShape(v.shape)}</td>
        <td>${fmtNum(v.update_norm)}</td>
        <td>${v.relative_update != null ? (v.relative_update * 100).toFixed(3) + "%" : "–"}</td>
        <td><span class="deltabar" style="width:${Math.max(2, (v.update_norm / maxU) * 120)}px"></span></td></tr>`;
    }
    html += `</table><div id="train-diff-detail"></div>`;
    $("train-diff-view").innerHTML = html;
    document.querySelectorAll("#train-diff-view tr[data-name]").forEach((tr) => {
      tr.onclick = async () => {
        const det = await api(`/api/session/${S.session.session_id}/diff_model_param?name=${encodeURIComponent(tr.dataset.name)}`);
        const card = document.createElement("div"); card.className = "card";
        card.innerHTML = `<h3>${esc(tr.dataset.name)} — current / checkpoint / Δ</h3>`;
        const trio = document.createElement("div"); trio.className = "heatmap-trio";
        for (const [key, label] of [["before", "current"], ["after", "checkpoint"], ["delta", "Δ"]]) {
          const cellEl = document.createElement("div");
          renderHeatmap(cellEl, det[key].heatmap, label);
          trio.appendChild(cellEl);
        }
        card.appendChild(trio);
        const dd = $("train-diff-detail"); dd.innerHTML = ""; dd.appendChild(card);
      };
    });
  } catch (e) { banner(e.message, "error"); }
}

/* ---------------- activation steering (phase L) ---------------- */

let steerTimer = null;

async function buildSteerDirection() {
  if (!S.selectedPath) { banner("Select a block layer first (tree or graph).", "warn"); return; }
  const btn = $("btn-steer-dir"); busy(btn, true); banner(null);
  try {
    const d = await post(`/api/session/${S.session.session_id}/steer_direction`, {
      prompt_a: $("steer-a").value, prompt_b: $("steer-b").value,
      layer: S.selectedPath,
    });
    $("steer-dir-info").textContent =
      `‖direction‖ = ${fmtNum(d.norm)} in ${d.dim}-dim residual stream at ${d.layer_path}`;
    $("steer-run").classList.remove("hidden");
    $("steer-alpha").value = 0; $("steer-alpha-label").textContent = "0";
    runSteer();
  } catch (e) { banner("Direction failed: " + e.message, "error"); }
  busy(btn, false);
}

function watchTokens() {
  // " great, terrible" -> [" great", " terrible"] (leading space = GPT-2 word)
  return $("steer-watch").value.split(",").map((t) => t.trim())
    .filter(Boolean).map((t) => " " + t);
}

async function runSteer() {
  const alpha = parseFloat($("steer-alpha").value);
  try {
    const d = await post(`/api/session/${S.session.session_id}/steer`, {
      input: { kind: "text", text: $("steer-prompt").value },
      alpha, watch: watchTokens(),
    });
    const maxP = Math.max(...d.topk.map((t) => Math.max(t.prob, t.prob_base)), 1e-9);
    let html = `<div class="card"><h3>α = ${alpha} · top-1
      ${esc(JSON.stringify(d.top1_base))} → ${esc(JSON.stringify(d.top1_steered))}
      ${d.top1_changed ? "⚠ SHIFTED" : ""} · KL ${fmtNum(d.kl_from_base)}</h3>`;
    for (const t of d.topk) {
      html += `<div class="tokbar"><span class="lbl">${esc(JSON.stringify(t.label))}</span>
        <span class="bar" style="width:${Math.max(2, (t.prob_base / maxP) * 130)}px;background:#30363d"></span>
        <span class="bar" style="width:${Math.max(2, (t.prob / maxP) * 130)}px"></span>
        <span class="pct">${(t.prob_base * 100).toFixed(1)}% → ${(t.prob * 100).toFixed(1)}%</span></div>`;
    }
    if (d.watch && d.watch.length) {
      html += `<div class="statgrid">` + d.watch.map((w) =>
        `<div><span>p(${esc(JSON.stringify(w.token))})</span>
         <span style="color:${w.delta > 0 ? "var(--good)" : w.delta < 0 ? "var(--bad)" : "inherit"}">
         ${(w.p_base * 100).toFixed(2)}% → ${(w.p_steered * 100).toFixed(2)}%</span></div>`).join("") + `</div>`;
    }
    $("steer-result").innerHTML = html + `</div>`;
  } catch (e) { $("steer-result").innerHTML = `<div class="card dim">${esc(e.message)}</div>`; }
}

async function runSteerBatch() {
  const btn = $("btn-steer-batch"); busy(btn, true);
  const prompts = $("steer-batch-prompts").value.split("\n")
    .map((l) => l.trim()).filter(Boolean);
  if (!prompts.length) {
    banner("Enter test prompts, one per line — unrelated to the direction pair.", "warn");
    busy(btn, false); return;
  }
  try {
    const d = await post(`/api/session/${S.session.session_id}/steer_batch`, {
      prompts, alpha: parseFloat($("steer-alpha").value), watch: watchTokens(),
    });
    let html = `<table class="difftable"><tr><th>prompt</th><th>top-1 before</th>
      <th>top-1 after</th><th>KL</th><th>watched Δ</th></tr>`;
    for (const r of d.results) {
      const wd = (r.watch || []).map((w) =>
        `${esc(JSON.stringify(w.token))} ${w.delta >= 0 ? "+" : ""}${(w.delta * 100).toFixed(2)}pp`).join(" · ");
      html += `<tr><td>${esc(r.prompt)}</td><td>${esc(JSON.stringify(r.top1_base))}</td>
        <td${r.top1_changed ? ' style="color:var(--warn)"' : ""}>${esc(JSON.stringify(r.top1_steered))}</td>
        <td>${fmtNum(r.kl_from_base)}</td><td>${wd || "–"}</td></tr>`;
    }
    $("steer-batch-result").innerHTML = html + `</table>
      <div class="heat-caption">the same direction, applied to prompts it was never built from —
      consistent shifts = a real concept direction; scattered shifts = a prompt-specific artifact</div>`;
  } catch (e) { banner(e.message, "error"); }
  busy(btn, false);
}

/* ---------------- sparse autoencoder (phase M) ---------------- */

async function runSAETrain() {
  if (!S.selectedPath) { banner("Select a block layer first.", "warn"); return; }
  const btn = $("btn-sae-train"); busy(btn, true); banner(null);
  $("sae-result").classList.remove("hidden");
  $("sae-features").innerHTML = "";
  const losses = [];
  try {
    const res = await fetch(`/api/session/${S.session.session_id}/sae_train`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        layer: S.selectedPath,
        expansion: parseInt($("sae-exp").value, 10) || 2,
        l1: parseFloat($("sae-l1").value) || 0.005,
        steps: parseInt($("sae-steps").value, 10) || 500,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let nl;
      while ((nl = buf.indexOf("\n")) >= 0) {
        const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
        if (!line) continue;
        const ev = JSON.parse(line);
        if (ev.event === "start") {
          $("sae-status").textContent = `training ${ev.features} features on ` +
            `${ev.rows} activation rows (d=${ev.d}) from ${ev.layer}…`;
        } else if (ev.event === "step") {
          losses.push(ev.loss);
          $("sae-status").textContent =
            `step ${ev.i} · loss ${fmtNum(ev.loss)} · L0 ≈ ${ev.l0.toFixed(1)} active features`;
          const holder = $("sae-curve"); holder.innerHTML = "";
          linePlot(holder, losses.map((_, i) => i), losses,
            { label: "SAE loss (recon + λ·L1)", color: "#d2a8ff" });
        } else if (ev.event === "done") {
          $("sae-status").textContent = `done in ${ev.elapsed_s}s: ${ev.alive}/${ev.features} ` +
            `features alive · mean L0 ${ev.mean_l0.toFixed(1)} per input`;
        } else if (ev.event === "error") throw new Error(ev.detail);
      }
    }
  } catch (e) { banner("SAE training failed: " + e.message, "error"); }
  busy(btn, false);
}

async function runSAEDecompose() {
  const btn = $("btn-sae-decompose"); busy(btn, true);
  try {
    const d = await post(`/api/session/${S.session.session_id}/sae_decompose`,
      { input: inputSpec() });
    let html = `<div class="card"><h3>${esc(d.input)} — ${d.l0} of
      ${d.total_features} features active at ${esc(d.layer)}
      <span class="dim small">(recon R² ${d.recon_r2.toFixed(2)})</span></h3></div>`;
    const maxS = Math.max(...d.active.map((f) => f.strength), 1e-9);
    for (const f of d.active) {
      const ex = f.examples.map((e2) =>
        `<span class="lens-chip" title="strength ${fmtNum(e2.strength)}">${esc(e2.token != null ?
          `${e2.label} @ ${JSON.stringify(e2.token)}` : e2.label)}</span>`).join("");
      html += `<div class="gradbar-row" style="cursor:default">
        <span class="gname">feature ${f.feature}</span>
        <span class="gbar" style="width:${Math.max(2, (f.strength / maxS) * 160)}px;background:linear-gradient(90deg,#6e40aa,#d2a8ff)"></span>
        <span class="gval">${fmtNum(f.strength)}</span></div>
        <div style="margin:0 0 8px 230px">${ex ||
          '<span class="dim small">no strong training examples</span>'}</div>`;
    }
    $("sae-features").innerHTML = html + `<div class="heat-caption">
      each feature's chips are the training inputs (and token positions) that
      activate it most — eyeball them to guess what the feature represents</div>`;
  } catch (e) { banner(e.message, "error"); }
  busy(btn, false);
}

/* ---------------- dataset-scale + robustness (phases N/O) ---------------- */

async function streamNDJSON(url, body, onEvent) {
  const res = await fetch(url, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1);
      if (!line) continue;
      const ev = JSON.parse(line);
      if (ev.event === "error") throw new Error(ev.detail);
      onEvent(ev);
    }
  }
}

async function runAttrBatch() {
  const btn = $("btn-attr-batch"); busy(btn, true);
  const view = $("attr-view");
  view.innerHTML = `<span class="dim">running attribution across a prompt batch…</span>`;
  try {
    let doneEv = null;
    await streamNDJSON(`/api/session/${S.session.session_id}/aggregate`,
      { analysis: "attribution" }, (ev) => {
        if (ev.event === "prompt") {
          view.innerHTML = `<span class="dim">prompt ${ev.i + 1}: ` +
            `${esc(ev.prompt)} → top token ${esc(JSON.stringify(ev.top_token))} ` +
            `(${(ev.top_frac * 100).toFixed(0)}%)…</span>`;
        } else if (ev.event === "done") doneEv = ev;
      });
    let html = `<div class="card"><h3>attribution concentration across
      ${doneEv.n_prompts} prompts</h3>
      <p class="dim small">the top token holds ${(doneEv.mean_top_frac * 100).toFixed(0)}%
      of attribution mass on average (range ${(doneEv.min_top_frac * 100).toFixed(0)}–${(doneEv.max_top_frac * 100).toFixed(0)}%).
      A distribution over many prompts is evidence; a single example is an anecdote.</p></div>
      <table class="difftable"><tr><th>prompt</th><th>predicts</th><th>top token</th><th>share</th></tr>`;
    for (const r of doneEv.rows) {
      html += `<tr><td>${esc(r.prompt)}</td><td>${esc(JSON.stringify(r.target))}</td>
        <td>${esc(JSON.stringify(r.top_token))}</td><td>${(r.top_frac * 100).toFixed(0)}%</td></tr>`;
    }
    view.innerHTML = html + `</table>`;
  } catch (e) { view.innerHTML = `<span class="dim">${esc(e.message)}</span>`; }
  busy(btn, false);
}

async function runAggregateHeads(layer) {
  const view = $("analysis-view");
  view.innerHTML = `<span class="dim">ablating every head across a prompt batch…</span>`;
  try {
    let doneEv = null;
    await streamNDJSON(`/api/session/${S.session.session_id}/aggregate`,
      { analysis: "head_ablation", layer }, (ev) => {
        if (ev.event === "start") {
          view.innerHTML = `<span class="dim">${ev.n_prompts} prompts × ` +
            `${ev.n_heads} heads ≈ ${ev.estimate_s}s…</span>`;
        } else if (ev.event === "done") doneEv = ev;
      });
    const maxD = Math.max(...doneEv.heads.map((h) => Math.abs(h.mean_delta)), 1e-12);
    view.innerHTML = `<div class="dim small" style="margin-bottom:4px">
      mean importance over ${doneEv.n_prompts} prompts — “consistently important
      across N prompts” is evidence; one example is an anecdote:</div>` +
      doneEv.heads.map((h) => `<div class="gradbar-row" style="cursor:default">
        <span class="gname">head ${h.head}${h.consistent ? " ★" : ""}</span>
        <span class="gbar" style="width:${Math.max(2, (Math.abs(h.mean_delta) / maxD) * 200)}px"></span>
        <span class="gval">${(h.mean_delta * 100).toFixed(2)}pp mean · top-3 in ${(h.top3_frac * 100).toFixed(0)}% of prompts</span>
      </div>`).join("") +
      `<div class="heat-caption">★ = in the per-prompt top-3 at least half the time</div>`;
  } catch (e) { view.innerHTML = `<span class="dim">${esc(e.message)}</span>`; }
}

async function runFragility() {
  const btn = $("btn-fragility"); busy(btn, true);
  const view = $("attr-view");
  try {
    if (isLM()) {
      view.innerHTML = `<span class="dim">starting substitution search…</span>`;
      let doneEv = null;
      await streamNDJSON(`/api/session/${S.session.session_id}/robustness`,
        { input: inputSpec() }, (ev) => {
          if (ev.event === "start") {
            view.innerHTML = `<span class="dim">${ev.n_runs} single-token swaps ` +
              `≈ ${ev.estimate_s}s (baseline ${esc(JSON.stringify(ev.baseline_top1))} ` +
              `${(ev.p_top1 * 100).toFixed(1)}%)…</span>`;
          } else if (ev.event === "position") {
            view.insertAdjacentHTML("beforeend",
              `<div class="dim small">pos ${ev.pos} ${esc(JSON.stringify(ev.token))}: ` +
              `${ev.flips ? "⚠ flips" : "holds"} (best swap → ` +
              `${esc(JSON.stringify(ev.best_swap.token))})</div>`);
          } else if (ev.event === "done") doneEv = ev;
        });
      let html = `<div class="card"><h3>fragility of ${esc(JSON.stringify(doneEv.baseline_top1))}
        (${(doneEv.p_top1 * 100).toFixed(1)}%) on ${esc(doneEv.prompt)}</h3></div>
        <table class="difftable"><tr><th>token</th><th>best swap</th><th>Δp(top-1)</th><th>flips?</th></tr>`;
      for (const p of doneEv.positions) {
        html += `<tr><td>${esc(JSON.stringify(p.token))}</td>
          <td>${esc(JSON.stringify(p.best_swap.token))}
            <span class="dim small">(cos ${p.best_swap.similarity.toFixed(2)})</span></td>
          <td>−${(p.fragility * 100).toFixed(1)}pp</td>
          <td>${p.flips ? `⚠ → ${esc(JSON.stringify(p.best_swap.new_top1))}` : ""}</td></tr>`;
      }
      html += `</table>`;
      if (doneEv.cross_check) {
        html += `<div class="note">${esc(doneEv.cross_check.note)}.</div>`;
      } else {
        html += `<div class="heat-caption">run attribution on this exact prompt first
          to get the built-in attribution ↔ fragility cross-check</div>`;
      }
      view.innerHTML = html;
    } else {
      view.innerHTML = `<span class="dim">FGSM sweep…</span>`;
      const d = await post(`/api/session/${S.session.session_id}/fgsm`,
        { input: inputSpec() });
      view.innerHTML = `<div class="card"><h3>FGSM: x + ε·sign(∇ loss) —
        p(${esc(d.baseline_top1)}) starts at ${(d.p_top1 * 100).toFixed(1)}%</h3></div>`;
      linePlot(view, d.curve.map((c) => c.epsilon), d.curve.map((c) => c.p_top1),
        { label: "p(top-1) vs perturbation size ε", color: "#f85149" });
      view.insertAdjacentHTML("beforeend", d.curve.map((c) =>
        `<div class="dist-row"><span class="dr-rank">ε=${c.epsilon}</span>
         <span class="dr-label">p=${(c.p_top1 * 100).toFixed(1)}%</span>
         <span class="dr-prob">${c.flipped ? "⚠ flips to " + esc(c.new_top1) : "holds"}</span></div>`).join(""));
    }
  } catch (e) { view.innerHTML = `<span class="dim">${esc(e.message)}</span>`; }
  busy(btn, false);
}

/* ---------------- architecture editing (phase D) ---------------- */

async function runEdit(op, extra = {}) {
  if (!S.selectedPath) { banner("Select a layer in the tree or graph first.", "warn"); return; }
  banner(null);
  try {
    const d = await post(`/api/session/${S.session.session_id}/edit`,
      { op, path: S.selectedPath, ...extra });
    applyEditResponse(d);
    banner((d.warnings && d.warnings.length) ? d.warnings.join(" ")
      : `Edit applied: ${d.desc}. Re-running the input through both models…`,
      (d.warnings && d.warnings.length) ? "warn" : "info");
  } catch (e) { banner("Edit failed: " + e.message, "error"); }
}

async function runEditUndo() {
  try {
    const d = await post(`/api/session/${S.session.session_id}/edit_undo`, {});
    applyEditResponse(d);
    banner(`Undone: ${d.undone}`, "info");
    if (!d.history.length) $("edit-compare").innerHTML = "";
  } catch (e) { banner(e.message, "error"); }
}

function applyEditResponse(d) {
  if (d.arch) { S.arch = d.arch; buildFlat(); }
  if (d.topology) S.topo = d.topology;
  S.graphDirty = true;
  renderTree();
  renderEditHistory(d.history || []);
  if (d.compare) renderCompare(d.compare);
  // immediate causal feedback: re-run the hooked forward on the edited model
  if (S.trace) runForward();
}

function renderEditHistory(hist) {
  $("edit-history").classList.toggle("hidden", !hist.length);
  $("edit-history-list").innerHTML = hist.map((h, i) =>
    `<div>${i + 1}. ${esc(h)}</div>`).join("");
}

function renderCompare(c) {
  const maxP = Math.max(...c.rows.map((r) => Math.max(r.p_original, r.p_edited)), 1e-9);
  let rows = "";
  for (const r of c.rows.slice(0, 8)) {
    const d = r.delta;
    rows += `<div class="tokbar"><span class="lbl">${esc(JSON.stringify(r.label))}</span>
      <span class="bar" style="width:${Math.max(2, (r.p_original / maxP) * 150)}px;background:#1f6feb"></span>
      <span class="bar" style="width:${Math.max(2, (r.p_edited / maxP) * 150)}px;background:#3fb950"></span>
      <span class="pct" style="color:${d > 0 ? "var(--good)" : d < 0 ? "var(--bad)" : "var(--dim)"}">
        ${d >= 0 ? "+" : ""}${(d * 100).toFixed(2)}pp</span></div>`;
  }
  $("edit-compare").innerHTML = `<div class="card">
    <h3>original <span style="color:#79c0ff">■</span> vs edited <span style="color:#3fb950">■</span>
      — same real input</h3>
    <div class="kv">
      <span class="k">top-1</span><span>${esc(JSON.stringify(c.top1_original))} →
        ${esc(JSON.stringify(c.top1_edited))}${c.top1_changed ? " ⚠ CHANGED" : " (unchanged)"}</span>
      <span class="k">KL(edited‖orig)</span><span>${fmtNum(c.kl_divergence)}</span>
    </div>${rows}</div>`;
}

/* --- model diffing (phase F) --- */

async function runModelDiff() {
  const btn = $("btn-mdiff"); busy(btn, true); banner(null);
  try {
    const d = await post(`/api/session/${S.session.session_id}/diff_model`,
      { ref: $("mdiff-ref").value.trim(), allow_pickle: $("mdiff-pickle").checked });
    const rows = Object.entries(d.param_diffs).sort((a, b) => b[1].update_norm - a[1].update_norm);
    const maxU = Math.max(...rows.map(([, v]) => v.update_norm), 1e-12);
    let html = `<div class="card"><h3>vs ${esc(d.other)}</h3>
      <p class="dim small">${d.n_params_compared} tensors compared · ${d.n_identical} identical ·
      ${d.shape_mismatch.length} shape mismatches · ${d.missing_in_other.length} missing in other</p></div>
      <table class="difftable"><tr><th>parameter</th><th>shape</th><th>‖Δ‖</th><th>Δ/‖w‖</th><th></th></tr>`;
    for (const [name, v] of rows.slice(0, 40)) {
      html += `<tr data-name="${esc(name)}"><td>${esc(name)}</td><td>${fmtShape(v.shape)}</td>
        <td>${fmtNum(v.update_norm)}</td>
        <td>${v.relative_update != null ? (v.relative_update * 100).toFixed(3) + "%" : "–"}</td>
        <td><span class="deltabar" style="width:${Math.max(2, (v.update_norm / maxU) * 120)}px"></span></td></tr>`;
    }
    html += `</table><div id="mdiff-detail"></div>`;
    $("mdiff-result").innerHTML = html;
    document.querySelectorAll("#mdiff-result tr[data-name]").forEach((tr) => {
      tr.onclick = async () => {
        const det = await api(`/api/session/${S.session.session_id}/diff_model_param?name=${encodeURIComponent(tr.dataset.name)}`);
        const card = document.createElement("div"); card.className = "card";
        card.innerHTML = `<h3>${esc(tr.dataset.name)} — this model / other checkpoint / Δ</h3>`;
        const trio = document.createElement("div"); trio.className = "heatmap-trio";
        for (const [key, label] of [["before", "this model"], ["after", "other"], ["delta", "other − this"]]) {
          const cell = document.createElement("div");
          renderHeatmap(cell, det[key].heatmap, label);
          trio.appendChild(cell);
        }
        card.appendChild(trio);
        const dd = $("mdiff-detail"); dd.innerHTML = ""; dd.appendChild(card);
      };
    });
  } catch (e) { banner("Model diff failed: " + e.message, "error"); }
  busy(btn, false);
}

/* ---------------- profiling (phase G) ---------------- */

async function runProfile() {
  const btn = $("btn-profile"); busy(btn, true); banner(null);
  try {
    const d = await api(`/api/session/${S.session.session_id}/profile`);
    S.profile = d;
    syncGraphChips();
    renderProfile(d, "ms");
  } catch (e) { banner("Profile failed: " + e.message, "error"); }
  busy(btn, false);
}

function renderProfile(d, sortKey) {
  const rows = d.rows.filter((r) => r.is_leaf)
    .sort((a, b) => (b[sortKey] || 0) - (a[sortKey] || 0));
  const cols = [["path", "layer"], ["class", "type"], ["calls", "calls"],
    ["ms", "ms ▾"], ["pct_of_total", "% total"], ["params", "params"],
    ["param_bytes", "memory"], ["flops", "FLOPs*"]];
  let html = `<div class="card"><h3>forward pass: ${d.total_ms.toFixed(2)} ms total
    <span class="dim small">(${d.leaf_ms.toFixed(2)} ms in leaf modules)</span></h3>
    <p class="dim small">${esc(d.note)}</p></div>
    <table class="difftable"><tr>` + cols.map(([k, label]) =>
    `<th data-sort="${k}" style="cursor:pointer">${k === sortKey ? label.replace(" ▾", "") + " ▾" : label.replace(" ▾", "")}</th>`).join("") + `</tr>`;
  for (const r of rows.slice(0, 60)) {
    html += `<tr data-path="${esc(r.path)}"><td>${esc(r.path)}</td><td>${esc(r.class)}</td>
      <td>${r.calls}</td><td>${r.ms.toFixed(3)}</td><td>${r.pct_of_total.toFixed(1)}%</td>
      <td>${fmtCount(r.params)}</td><td>${r.param_bytes ? fmtCount(r.param_bytes) + "B" : "–"}</td>
      <td>${r.flops != null ? fmtCount(r.flops) : "–"}</td></tr>`;
  }
  html += `</table>`;
  if (rows.length > 60) html += `<div class="heat-caption">top 60 of ${rows.length} leaf modules</div>`;
  $("profile-result").innerHTML = html;
  document.querySelectorAll("#profile-result th[data-sort]").forEach((th) => {
    th.onclick = () => renderProfile(d, th.dataset.sort);
  });
  document.querySelectorAll("#profile-result tr[data-path]").forEach((tr) => {
    tr.onclick = () => selectNode(tr.dataset.path === "(model)" ? "" : tr.dataset.path);
  });
}

/* ---------------- inspector analysis panel (phases E/F) ---------------- */

function linePlot(container, xs, ys, { label = "", logY = false, color = "#58a6ff" } = {}) {
  const cv = document.createElement("canvas");
  cv.className = "hist"; cv.width = 330; cv.height = 110;
  const ctx = cv.getContext("2d");
  const vals = logY ? ys.map((y) => Math.log10(Math.max(y, 1e-12))) : ys;
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const px = (i) => 8 + (i / Math.max(1, xs.length - 1)) * (cv.width - 16);
  const py = (v) => cv.height - 16 - ((v - lo) / Math.max(1e-12, hi - lo)) * (cv.height - 30);
  ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
  vals.forEach((v, i) => i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v)));
  ctx.stroke();
  ctx.fillStyle = "#8b949e"; ctx.font = "9px monospace";
  ctx.fillText(label + (logY ? " (log scale)" : ""), 8, 10);
  container.appendChild(cv);
}

function analysisBusy(msg) {
  $("analysis-view").innerHTML = `<span class="dim">${esc(msg)}</span>`;
}

async function runSVD() {
  const param = $("param-select").value || "weight";
  analysisBusy("computing singular values…");
  try {
    const d = await api(`/api/session/${S.session.session_id}/svd?path=${encodeURIComponent(S.selectedPath)}&param=${encodeURIComponent(param)}`);
    const el = $("analysis-view");
    el.innerHTML = `<div class="kv">
      <span class="k">matrix</span><span>${fmtShape(d.matrix_shape)} (full rank ${d.full_rank}${d.approximate ? ", top-512 approx" : ""})</span>
      <span class="k">effective rank</span><span><b>${d.effective_rank.toFixed(1)}</b> (spectral entropy)</span>
      <span class="k">svals &gt; 1% max</span><span>${d.rank_1pct}</span>
      <span class="k">90% energy in</span><span>${d.rank_90pct_energy} directions</span></div>`;
    linePlot(el, d.singular_values.map((_, i) => i), d.singular_values,
      { label: `singular value spectrum — ${S.selectedPath}.${param}`, logY: true });
    if (d.rank_1pct < d.full_rank / 4) {
      el.insertAdjacentHTML("beforeend", `<div class="note">This matrix is nominally rank ${d.full_rank}
        but effectively ~rank ${d.rank_1pct} — most of its capacity is redundant (compressible).</div>`);
    }
  } catch (e) { analysisBusy(e.message); }
}

async function runDead() {
  analysisBusy("running a real input batch through the model…");
  try {
    const d = await api(`/api/session/${S.session.session_id}/dead_neurons?path=${encodeURIComponent(S.selectedPath)}`,
      { method: "POST" });
    $("analysis-view").innerHTML = `<div class="kv">
      <span class="k">probe</span><span>${d.n_inputs} real inputs</span>
      <span class="k">units</span><span>${d.total_units}</span>
      <span class="k">never fired</span><span><b>${d.dead_count}</b> (${(d.dead_frac * 100).toFixed(1)}%)</span>
      ${d.dead_count ? `<span class="k">dead indices</span><span>${d.dead_indices.join(", ")}${d.dead_count > 64 ? "…" : ""}</span>` : ""}
    </div>`;
  } catch (e) { analysisBusy(e.message); }
}

async function runQuant(bits) {
  analysisBusy(`quantizing to int${bits}, re-running, restoring…`);
  try {
    const d = await api(`/api/session/${S.session.session_id}/quantize_sim?path=${encodeURIComponent(S.selectedPath)}&bits=${bits}`,
      { method: "POST" });
    $("analysis-view").innerHTML = `<div class="kv">
      <span class="k">int${bits} on</span><span>${esc(d.path)} (weights restored after)</span>
      <span class="k">KL drift</span><span>${fmtNum(d.kl_divergence)}</span>
      <span class="k">top-1</span><span>${esc(JSON.stringify(d.top1_before))} → ${esc(JSON.stringify(d.top1_after))}
        ${d.top1_changed ? "⚠ CHANGED" : "(survives)"}</span>
      <span class="k">p(top-1)</span><span>${(d.p_top1_before * 100).toFixed(2)}% → ${(d.p_top1_after * 100).toFixed(2)}%</span>
      <span class="k">max prob shift</span><span>${fmtNum(d.max_prob_shift)}</span></div>`;
  } catch (e) { analysisBusy(e.message); }
}

async function runPrune() {
  analysisBusy("prune sweep: 10/25/50/75/90% smallest weights…");
  try {
    const d = await api(`/api/session/${S.session.session_id}/prune_sim?path=${encodeURIComponent(S.selectedPath)}`,
      { method: "POST" });
    const el = $("analysis-view");
    el.innerHTML = `<div class="kv"><span class="k">baseline</span>
      <span>p(${esc(JSON.stringify(d.baseline_top1))}) = ${(d.p_top1_baseline * 100).toFixed(2)}%</span></div>`;
    linePlot(el, d.curve.map((c) => c.fraction), d.curve.map((c) => c.p_top1),
      { label: "p(top-1) as smallest X% of weights are zeroed", color: "#f85149" });
    el.insertAdjacentHTML("beforeend", d.curve.map((c) =>
      `<div class="dist-row"><span class="dr-rank">${(c.fraction * 100).toFixed(0)}%</span>
       <span class="dr-label">p=${(c.p_top1 * 100).toFixed(2)}%</span>
       <span class="dr-prob">KL ${fmtNum(c.kl_divergence)}${c.top1_changed ? " · ⚠ top-1 flips" : ""}</span></div>`).join(""));
  } catch (e) { analysisBusy(e.message); }
}

async function runMaxAct() {
  analysisBusy("running candidate inputs…");
  try {
    const d = await post(`/api/session/${S.session.session_id}/max_activating`,
      { path: S.selectedPath });
    const maxS = Math.max(...d.results.map((r) => r.score), 1e-12);
    $("analysis-view").innerHTML = `<div class="dim small" style="margin-bottom:4px">
      inputs ranked by ${esc(d.metric)} at ${esc(d.path)}</div>` +
      d.results.map((r) => `<div class="gradbar-row"><span class="gname">${esc(r.input)}</span>
        <span class="gbar" style="width:${Math.max(2, (r.score / maxS) * 220)}px"></span>
        <span class="gval">${fmtNum(r.score)}</span></div>`).join("");
  } catch (e) { analysisBusy(e.message); }
}

async function runAblate() {
  analysisBusy("ablating heads one at a time (one real re-run each)…");
  try {
    const d = await post(`/api/session/${S.session.session_id}/ablate_heads`,
      { input: inputSpec(), layer: S.selectedPath });
    const maxD = Math.max(...d.heads.map((h) => Math.abs(h.delta)), 1e-12);
    $("analysis-view").innerHTML = `<div class="dim small" style="margin-bottom:4px">
      baseline p(${esc(JSON.stringify(d.baseline_top1))}) = ${(d.p_top1_baseline * 100).toFixed(2)}% —
      Δ when each head is zeroed
      <button onclick="runAggregateHeads('${esc(d.layer)}')"
        title="one prompt is an anecdote — average over a batch">Σ across N prompts</button>:</div>` +
      d.heads.map((h) => `<div class="gradbar-row"><span class="gname">head ${h.head}</span>
        <span class="gbar" style="width:${Math.max(2, (Math.abs(h.delta) / maxD) * 220)};
          width:${Math.max(2, (Math.abs(h.delta) / maxD) * 220)}px;
          background:${h.delta >= 0 ? "linear-gradient(90deg,#9e6a03,#e3b341)" : "linear-gradient(90deg,#1f6feb,#79c0ff)"}"></span>
        <span class="gval">${h.delta >= 0 ? "−" : "+"}${(Math.abs(h.delta) * 100).toFixed(2)}pp${h.top1_changed ? " ⚠ flips top-1" : ""}</span></div>`).join("");
  } catch (e) { analysisBusy(e.message); }
}

/* ---------------- graph view ----------------
   Renders the TRUE dataflow captured by the backend (autograd-derived edge
   topology from the last real execution). Shares expand/collapse state and
   the playback player with the tree; pulls activation/gradient/update
   numbers from the exact same responses that drive the other panels. */

const G = {
  built: false, nodes: null, nodeEls: new Map(), edgeEls: new Map(),
  callNode: new Map(), nodeCalls: new Map(), rootG: null, svg: null,
  view: { x: 20, y: 10, k: 1 },
};

function classColor(cls) {
  const c = (cls || "").toLowerCase();
  if (c === "input" || c === "output") return "#58a6ff";
  if (c === "group") return "#d2a8ff";
  if (/attention|attn/.test(c)) return "#ff9eb2";
  if (/embed/.test(c)) return "#76e3e3";
  if (/norm/.test(c)) return "#e3b341";
  if (/relu|gelu|sigmoid|tanh|softmax|activation|silu|elu|mish|identity/.test(c)) return "#e2a8f0";
  if (/^conv|pool/.test(c)) return "#56d364";
  if (/linear|conv1d$/.test(c)) return "#79c0ff";
  if (/sequential|modulelist|block|model|encoder|decoder/.test(c)) return "#8b949e";
  return "#c9d1d9";
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}

function buildVisibleMap() {
  const map = {}, groupInfo = {}, memberOf = {};
  function childForced(parent, myId, forcedId) {
    if (forcedId) return forcedId;
    if (parent.path !== "" && S.collapsed.has(parent.path)) return myId;
    return null;
  }
  function visit(node, forcedId, memberKey) {
    const myId = forcedId || node.path;
    map[node.path] = myId;
    if (memberKey) memberOf[node.path] = memberKey;
    const groups = node.repeat_groups || [];
    const kids = node.children;
    let i = 0;
    while (i < kids.length) {
      const g = groups.find((x) => x.start === i);
      const gkey = node.path + "#" + i;
      if (g && !forcedId && !childForced(node, myId, forcedId) && !S.expandedGroups.has(gkey)) {
        const gid = "group:" + gkey;
        groupInfo[gid] = {
          label: `${g.class} ×${g.count}`, gkey,
          params: kids.slice(i, i + g.count).reduce((a, c) => a + c.total_param_count, 0),
          path: kids[i].path, cls: "group",
        };
        for (let j = i; j < i + g.count; j++) visit(kids[j], gid, memberKey);
        i += g.count;
      } else {
        const childMember = (g && !forcedId) ? gkey : memberKey;
        const upto = g && !forcedId ? i + g.count : i + 1;
        for (let j = i; j < upto; j++) {
          visit(kids[j], childForced(node, myId, forcedId), childMember);
        }
        i = upto;
      }
    }
  }
  visit(S.arch.tree, null, null);
  return { map, groupInfo, memberOf };
}

function computeGraph() {
  const { map, groupInfo, memberOf } = buildVisibleMap();
  const callPath = {};
  S.topo.calls.forEach((c) => { callPath[c.call_index] = c.path; });
  const nodeOf = (ci) => ci === -1 ? "<in>" : ci === -2 ? "<out>" : map[callPath[ci]];

  const nodes = new Map();
  function ensure(id, first) {
    if (id === undefined) return;
    if (!nodes.has(id)) {
      let info;
      if (id === "<in>") info = { label: "INPUT", cls: "input", params: 0, virtual: true };
      else if (id === "<out>") info = { label: "OUTPUT", cls: "output", params: 0, virtual: true };
      else if (groupInfo[id]) info = { ...groupInfo[id], isGroup: true };
      else {
        const n = S.flat[id];
        const segs = id.split(".");
        // "head.2" reads better than a bare "2" for Sequential children
        const label = id === "" ? (S.arch.root_class || "model")
          : /^\d+$/.test(segs[segs.length - 1]) && segs.length > 1
            ? segs.slice(-2).join(".") : segs[segs.length - 1];
        info = {
          label, cls: n ? n.class : "?", params: n ? n.total_param_count : 0,
          path: id, memberKey: memberOf[id],
        };
      }
      nodes.set(id, { id, first, ...info });
    } else {
      nodes.get(id).first = Math.min(nodes.get(id).first, first);
    }
  }

  const edgeMap = new Map();
  for (const e of S.topo.edges) {
    const a = nodeOf(e.src), b = nodeOf(e.dst);
    if (a === undefined || b === undefined || a === b) continue;
    ensure(a, e.src === -1 ? -1 : e.src);
    ensure(b, e.dst === -2 ? Number.MAX_SAFE_INTEGER : e.dst);
    const key = a + "→" + b;
    if (!edgeMap.has(key)) edgeMap.set(key, { a, b, key });
  }

  // call -> visible node, and node -> its call list (drives playback + colors)
  G.callNode = new Map([[-1, "<in>"], [-2, "<out>"]]);
  G.nodeCalls = new Map();
  for (const c of S.topo.calls) {
    const id = map[c.path];
    G.callNode.set(c.call_index, id);
    if (nodes.has(id)) {
      if (!G.nodeCalls.has(id)) G.nodeCalls.set(id, []);
      G.nodeCalls.get(id).push(c.call_index);
    }
  }

  // layering: longest path over forward edges (back edges = module reuse)
  const order = [...nodes.values()].sort((x, y) => x.first - y.first);
  const incoming = new Map();
  for (const e of edgeMap.values()) {
    e.back = nodes.get(e.b).first < nodes.get(e.a).first;
    if (!e.back) {
      if (!incoming.has(e.b)) incoming.set(e.b, []);
      incoming.get(e.b).push(e.a);
    }
  }
  let maxLayer = 0;
  for (const n of order) {
    let L = n.id === "<in>" ? 0 : 1;
    for (const src of incoming.get(n.id) || []) L = Math.max(L, nodes.get(src).layer + 1);
    n.layer = L; maxLayer = Math.max(maxLayer, L);
  }

  const V = 78, H = 176;
  const byLayer = {};
  order.forEach((n) => { (byLayer[n.layer] ||= []).push(n); });
  let maxRow = 1;
  Object.values(byLayer).forEach((row) => {
    row.sort((a, b) => a.first - b.first);
    maxRow = Math.max(maxRow, row.length);
  });
  const width = Math.max(760, maxRow * H + 200);
  Object.entries(byLayer).forEach(([L, row]) => row.forEach((n, i) => {
    n.w = Math.max(96, Math.min(200, n.label.length * 7.2 + 26));
    n.h = 36;
    n.x = width / 2 + (i - (row.length - 1) / 2) * H;
    n.y = 44 + (+L) * V;
  }));
  return { nodes, edges: [...edgeMap.values()], width, height: 90 + maxLayer * V };
}

function edgePath(e, nodes) {
  const a = nodes.get(e.a), b = nodes.get(e.b);
  const x1 = a.x, y1 = a.y + a.h / 2, x2 = b.x, y2 = b.y - b.h / 2;
  const span = Math.abs(b.layer - a.layer);
  if (e.back) {                       // module re-entered: dashed bow left
    const bx = Math.min(x1, x2) - 70 - span * 10;
    return `M ${x1 - a.w / 4} ${a.y} C ${bx} ${a.y}, ${bx} ${b.y}, ${x2 - b.w / 4} ${b.y}`;
  }
  if (span <= 1) {
    return `M ${x1} ${y1} C ${x1} ${y1 + 26}, ${x2} ${y2 - 26}, ${x2} ${y2}`;
  }
  // skip connection: visibly separate bow around the main path
  const lane = (e.laneIdx || 0) * 16;
  const bx = Math.max(x1, x2) + 64 + span * 9 + lane;
  return `M ${x1} ${y1} C ${bx} ${y1 + 30}, ${bx} ${y2 - 30}, ${x2} ${y2}`;
}

function renderGraph() {
  const wrap = $("graph-wrap"), emptyEl = $("graph-empty");
  if (!S.topo || !S.arch) {
    wrap.classList.add("hidden"); emptyEl.classList.remove("hidden");
    G.built = false; return;
  }
  emptyEl.classList.add("hidden"); wrap.classList.remove("hidden");
  if (!S.graphDirty && G.built) return;

  const { nodes, edges, width, height } = computeGraph();
  G.nodes = nodes;
  const svg = $("graph-svg");
  svg.innerHTML = "";
  G.svg = svg;
  const root = svgEl("g", { id: "graph-root" });
  svg.appendChild(root);
  G.rootG = root;

  // assign lanes so parallel skip edges don't overlap
  let lane = 0;
  edges.forEach((e) => {
    const span = Math.abs(nodes.get(e.b).layer - nodes.get(e.a).layer);
    if (!e.back && span > 1) e.laneIdx = (lane++ % 4);
  });

  G.edgeEls = new Map();
  for (const e of edges) {
    const p = svgEl("path", { d: edgePath(e, nodes), class: "gedge" + (e.back ? " back" : "") });
    root.appendChild(p);
    G.edgeEls.set(e.key, p);
  }

  G.nodeEls = new Map();
  for (const n of nodes.values()) {
    const g = svgEl("g", {
      class: "gnode" + (n.virtual ? " virtual" : "") + (n.isGroup ? " group" : ""),
      transform: `translate(${n.x - n.w / 2}, ${n.y - n.h / 2})`,
    });
    g.appendChild(svgEl("rect", { width: n.w, height: n.h, rx: 7, stroke: classColor(n.cls) }));
    const label = svgEl("text", { x: n.w / 2, y: n.params ? 15 : 22, "text-anchor": "middle" });
    label.textContent = n.label;
    g.appendChild(label);
    if (n.params) {
      const sub = svgEl("text", { x: n.w / 2, y: 28, "text-anchor": "middle", class: "gsub" });
      sub.textContent = fmtCount(n.params) + (n.isGroup ? " · click to expand" : "");
      g.appendChild(sub);
    }
    g.onclick = () => {
      if (n.isGroup) { S.expandedGroups.add(n.gkey); renderTree(); return; }
      if (n.virtual) return;
      selectNode(n.path);
    };
    root.appendChild(g);
    G.nodeEls.set(n.id, g);
  }

  // one ⊖ per expanded group, on its top-most node, to re-collapse
  const byMember = new Map();
  for (const n of nodes.values()) {
    if (!n.memberKey) continue;
    const cur = byMember.get(n.memberKey);
    if (!cur || n.y < cur.y || (n.y === cur.y && n.x < cur.x)) byMember.set(n.memberKey, n);
  }
  byMember.forEach((n, gkey) => {
    const g = G.nodeEls.get(n.id);
    const c = svgEl("circle", { cx: n.w - 2, cy: 2, r: 8, class: "collapse-btn" });
    const t = svgEl("text", { x: n.w - 2, y: 6, "text-anchor": "middle" });
    t.textContent = "⊖";
    const collapse = (ev) => {
      ev.stopPropagation();
      S.expandedGroups.delete(gkey);
      renderTree();
    };
    c.onclick = t.onclick = collapse;
    g.appendChild(c); g.appendChild(t);
  });

  // fit + pan/zoom
  const cw = svg.clientWidth || 800;
  G.view = { x: 12, y: 8, k: Math.min(1, (cw - 24) / width) };
  applyView();
  svg.onwheel = (ev) => {
    ev.preventDefault();
    const f = Math.exp(-ev.deltaY * 0.0012);
    const r = svg.getBoundingClientRect();
    const mx = ev.clientX - r.left, my = ev.clientY - r.top;
    G.view.x = mx - (mx - G.view.x) * f;
    G.view.y = my - (my - G.view.y) * f;
    G.view.k = Math.max(0.08, Math.min(4, G.view.k * f));
    applyView();
  };
  let drag = null;
  svg.onmousedown = (ev) => { drag = { x: ev.clientX - G.view.x, y: ev.clientY - G.view.y }; };
  svg.onmousemove = (ev) => {
    if (!drag) return;
    G.view.x = ev.clientX - drag.x; G.view.y = ev.clientY - drag.y; applyView();
  };
  svg.onmouseup = svg.onmouseleave = () => { drag = null; };

  G.built = true;
  S.graphDirty = false;
  applyGraphMode();
  if (S.trace) graphOnStep(S.playIdx, S.playDir === "backward");
  if (S.circuit) applyCircuitOverlay();
}

function applyView() {
  G.rootG.setAttribute("transform",
    `translate(${G.view.x},${G.view.y}) scale(${G.view.k})`);
}

/* --- mode coloring: same numbers as the existing panels, no recomputation --- */

function nodeIntensities(mode) {
  const raw = new Map();
  const add = (id, v) => {
    if (id === undefined || !G.nodes.has(id) || !(v > 0)) return;
    raw.set(id, Math.max(raw.get(id) || 0, v));
  };
  if (mode === "activation" && S.trace) {
    G.nodeCalls.forEach((calls, id) => {
      for (const ci of calls) {
        const st = S.trace.records[ci] && S.trace.records[ci].output &&
          S.trace.records[ci].output.stats;
        if (st) add(id, st.abs_mean);
      }
    });
  } else if (mode === "gradient" && S.backward) {
    const { map } = buildVisibleMap();
    for (const [k, v] of Object.entries(S.backward.layer_grad_norms)) add(map[k], v);
  } else if (mode === "update" && S.step) {
    const { map } = buildVisibleMap();
    for (const [name, d] of Object.entries(S.step.param_diffs)) {
      add(map[name.split(".").slice(0, -1).join(".")], d.update_norm);
    }
  } else if (mode === "time" && S.profile) {
    const { map } = buildVisibleMap();
    for (const r of S.profile.rows) {
      if (r.is_leaf && r.path !== "(model)") add(map[r.path], r.ms);
    }
  }
  // log-normalize (magnitudes span decades)
  const vals = [...raw.values()];
  if (!vals.length) return raw;
  const lmin = Math.log(Math.min(...vals)), lmax = Math.log(Math.max(...vals));
  const out = new Map();
  raw.forEach((v, id) => out.set(id,
    lmax > lmin ? (Math.log(v) - lmin) / (lmax - lmin) : 1));
  return out;
}

const MODE_RGB = { activation: "63,185,80", gradient: "227,179,65",
  update: "248,81,73", time: "88,166,255" };

function applyGraphMode() {
  if (!G.built) return;
  const mode = S.graphMode;
  const inten = mode === "structure" ? new Map() : nodeIntensities(mode);
  G.nodeEls.forEach((el, id) => {
    const rect = el.querySelector("rect");
    const t = inten.get(id);
    rect.style.fill = (t !== undefined && MODE_RGB[mode])
      ? `rgba(${MODE_RGB[mode]},${(0.10 + 0.6 * t).toFixed(3)})` : "";
  });
}

function syncGraphChips() {
  const avail = { structure: true, activation: !!S.trace, gradient: !!S.backward,
    update: !!S.step, time: !!S.profile };
  document.querySelectorAll("#graph-modes .chip").forEach((c) => {
    c.disabled = !avail[c.dataset.mode];
    c.classList.toggle("active", c.dataset.mode === S.graphMode);
  });
  const bw = $("pb2-dir") && $("pb2-dir").querySelector('[value="backward"]');
  if (bw) bw.disabled = !S.backward;
}

function setGraphMode(mode) {
  const chip = document.querySelector(`#graph-modes [data-mode="${mode}"]`);
  if (!chip || chip.disabled) return;
  S.graphMode = mode;
  syncGraphChips();
  applyGraphMode();
}

/* --- playback + pulses --- */

function graphOnStep(idx, back) {
  if (!G.built || !S.trace) return;
  G.nodeEls.forEach((el) => el.classList.remove("g-active", "bw", "g-done"));
  G.edgeEls.forEach((el) => el.classList.remove("g-lit", "bw"));
  G.nodeCalls.forEach((calls, id) => {
    if (calls.some((ci) => (back ? ci > idx : ci < idx)))
      G.nodeEls.get(id).classList.add("g-done");
  });
  const isLast = idx === S.trace.records.length - 1;
  let nid = G.callNode.get(idx);
  if ((nid === undefined || !G.nodeEls.has(nid)) && isLast) nid = "<out>";
  if (nid !== undefined && G.nodeEls.has(nid)) {
    const el = G.nodeEls.get(nid);
    el.classList.add("g-active");
    if (back) el.classList.add("bw");
  }
  const color = back ? "#e3b341" : "#3fb950";
  for (const e of S.topo.edges) {
    const hit = (back ? e.src === idx : e.dst === idx) || (isLast && e.dst === -2);
    if (!hit) continue;
    const a = G.callNode.get(e.src), b = G.callNode.get(e.dst);
    if (a === undefined || b === undefined || a === b) continue;
    const el = G.edgeEls.get(a + "→" + b);
    if (!el) continue;
    el.classList.add("g-lit");
    if (back) el.classList.add("bw");
    if (S.tab === "graph") pulseAlong(el, color, back);
  }
}

function pulseAlong(pathEl, color, reverse, dur = 280) {
  const len = pathEl.getTotalLength();
  if (!len) return;
  const dot = svgEl("circle", { r: 4.5, fill: color, class: "pulse" });
  G.rootG.appendChild(dot);
  const t0 = performance.now();
  (function frame(t) {
    let u = Math.min(1, (t - t0) / dur);
    if (reverse) u = 1 - u;
    const p = pathEl.getPointAtLength(u * len);
    dot.setAttribute("cx", p.x); dot.setAttribute("cy", p.y);
    if (t - t0 < dur) requestAnimationFrame(frame); else dot.remove();
  })(t0);
}

function pulseUpdatedNodes() {
  if (!G.built || !S.step) return;
  const inten = nodeIntensities("update");
  let delay = 0;
  [...inten.entries()].sort((a, b) => b[1] - a[1]).forEach(([id, t]) => {
    if (t < 0.03) return;
    const el = G.nodeEls.get(id);
    if (!el) return;
    setTimeout(() => {
      el.classList.add("upd-pulse");
      setTimeout(() => el.classList.remove("upd-pulse"), 950);
    }, delay);
    delay += 45;
  });
}

function wire() {
  $("btn-load-demo").onclick = async () => {
    const b = $("btn-load-demo"); busy(b, true); banner(null);
    try { onLoaded(await post("/api/load", { demo: $("demo-select").value })); }
    catch (e) { banner(e.message, "error"); }
    busy(b, false);
  };
  $("btn-load-hf").onclick = async () => {
    const b = $("btn-load-hf"); busy(b, true);
    banner("Downloading model from HuggingFace — this can take a while…", "info");
    try {
      onLoaded(await post("/api/load", { hf_id: $("hf-id").value.trim() }));
      banner(null);
    } catch (e) { banner(e.message, "error"); }
    busy(b, false);
  };
  $("btn-upload").onclick = async () => {
    const f = $("file-input").files[0];
    if (!f) { banner("Choose a file first.", "warn"); return; }
    const b = $("btn-upload"); busy(b, true); banner("Uploading…", "info");
    const fd = new FormData();
    fd.append("file", f);
    fd.append("allow_pickle", $("allow-pickle").checked);
    try {
      const res = await fetch("/api/upload", { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
      onLoaded(await res.json()); banner(null);
    } catch (e) { banner(e.message, "error"); }
    busy(b, false);
  };
  $("btn-forward").onclick = runForward;
  $("btn-backward").onclick = runBackward;
  $("btn-step").onclick = runStep;
  $("btn-undo").onclick = runUndo;
  $("pb-first").onclick = $("pb2-first").onclick = () => setPlayIdx(0);
  $("pb-prev").onclick = $("pb2-prev").onclick = () => setPlayIdx(S.playIdx - 1);
  $("pb-next").onclick = $("pb2-next").onclick = () => setPlayIdx(S.playIdx + 1);
  $("pb-last").onclick = $("pb2-last").onclick = () => setPlayIdx(S.trace.records.length - 1);
  $("pb-play").onclick = $("pb2-play").onclick = togglePlay;
  $("pb-slider").oninput = (e) => setPlayIdx(+e.target.value);
  $("pb2-slider").oninput = (e) => setPlayIdx(+e.target.value);
  $("pb2-dir").onchange = (e) => {
    S.playDir = e.target.value;
    if (S.trace) setPlayIdx(S.playDir === "backward" ? S.trace.records.length - 1 : 0);
  };
  document.querySelectorAll("#graph-modes .chip").forEach((c) =>
    c.onclick = () => setGraphMode(c.dataset.mode));
  $("btn-generate").onclick = runGenerate;
  $("btn-dist").onclick = toggleDist;
  $("dist-close").onclick = () => $("dist-panel").classList.add("hidden");
  $("dist-scroll").onscroll = () => distRender();
  $("btn-lens").onclick = runLens;
  $("lens-close").onclick = () => $("lens-panel").classList.add("hidden");
  $("btn-patch").onclick = runPatch;
  $("btn-edit-remove").onclick = () => runEdit("remove");
  $("btn-edit-swap").onclick = () => runEdit("swap_activation", { to: $("edit-act").value });
  $("btn-edit-dup").onclick = () => runEdit("duplicate", {
    init: document.querySelector('input[name="dup-init"]:checked').value });
  $("btn-edit-up").onclick = () => runEdit("reorder", { direction: "up" });
  $("btn-edit-down").onclick = () => runEdit("reorder", { direction: "down" });
  $("btn-edit-undo").onclick = runEditUndo;
  $("btn-mdiff").onclick = runModelDiff;
  $("btn-profile").onclick = runProfile;
  $("btn-attr").onclick = () => {
    $("attr-panel").classList.toggle("hidden");
    if (!$("attr-panel").classList.contains("hidden") && !$("attr-view").innerHTML) {
      runAttribution();
    }
  };
  $("btn-attr-run").onclick = runAttribution;
  $("attr-close").onclick = () => $("attr-panel").classList.add("hidden");
  $("btn-report").onclick = () => {
    window.location = `/api/session/${S.session.session_id}/report.md`;
  };
  $("btn-circuit").onclick = runCircuit;
  $("circuit-thresh").oninput = updateCircuitSelection;
  $("btn-circuit-overlay").onclick = () => {
    S.circuit = true;
    applyCircuitOverlay();
    switchTab("graph");
    banner("Circuit overlay active (teal) — it persists across coloring modes. " +
      "Adjust the threshold in 🧪 Patch to widen/narrow it.", "info");
  };
  $("btn-circuit-clear").onclick = () => {
    S.circuit = null;
    applyCircuitOverlay();
  };
  $("btn-train").onclick = runTrain;
  $("btn-train-restore").onclick = trainRestore;
  $("btn-train-diff").onclick = trainDiff;
  $("btn-steer-dir").onclick = buildSteerDirection;
  $("steer-alpha").oninput = () => {
    $("steer-alpha-label").textContent = $("steer-alpha").value;
    clearTimeout(steerTimer);
    steerTimer = setTimeout(runSteer, 180);   // debounced live slider
  };
  $("steer-prompt").onchange = $("steer-watch").onchange = () => runSteer();
  $("btn-steer-batch").onclick = runSteerBatch;
  $("btn-sae-train").onclick = runSAETrain;
  $("btn-sae-decompose").onclick = runSAEDecompose;
  $("btn-attr-batch").onclick = runAttrBatch;
  $("btn-fragility").onclick = runFragility;
  $("btn-svd").onclick = runSVD;
  $("btn-dead").onclick = runDead;
  $("btn-quant8").onclick = () => runQuant(8);
  $("btn-quant4").onclick = () => runQuant(4);
  $("btn-prune").onclick = runPrune;
  $("btn-maxact").onclick = runMaxAct;
  $("btn-ablate").onclick = runAblate;
  $("bw-target-kind").onchange = () =>
    $("bw-target-value").classList.toggle("hidden", $("bw-target-kind").value === "argmax");
  document.querySelectorAll(".tab").forEach((t) =>
    t.onclick = () => switchTab(t.dataset.tab));
  loadDemos().catch((e) => banner("API unreachable: " + e.message, "error"));
}

wire();

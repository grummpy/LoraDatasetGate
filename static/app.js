const $ = (id) => document.getElementById(id);
let report = null;

function esc(value) {
  return String(value).replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function options() {
  return {
    path: $("dataset-path").value.trim(),
    repeats: Number($("repeats").value),
    reg_per_subject_min: Number($("reg-min").value),
    reg_per_subject_max: Number($("reg-max").value),
    phash_threshold: Number($("phash").value),
    dhash_threshold: Number($("dhash").value),
    target_per_cell: Number($("target-cell").value),
  };
}

function setStatus(text, kind) {
  const node = $("status");
  node.textContent = text;
  node.className = "status" + (kind ? " " + kind : "");
}

async function api(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = data.detail || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function mediaURL(rel) {
  const params = new URLSearchParams({ dataset: report.dataset, rel });
  return "/api/media?" + params.toString();
}

function showTab(name) {
  document.querySelectorAll(".tab").forEach((button) => {
    const on = button.dataset.tab === name;
    button.setAttribute("aria-selected", on ? "true" : "false");
  });
  document.querySelectorAll("main section").forEach((section) => {
    section.hidden = section.id !== "panel-" + name;
  });
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => showTab(button.dataset.tab));
});

$("use-samples").addEventListener("click", async () => {
  try {
    const data = await fetch("/api/samples").then((response) => response.json());
    $("dataset-path").value = data.path;
    setStatus("Sample folder ready. Synthetic shapes only — no photographs.");
  } catch (error) {
    setStatus(error.message, "bad");
  }
});

$("run-all").addEventListener("click", async () => {
  const body = options();
  if (!body.path) {
    setStatus("Choose a dataset folder first.", "bad");
    return;
  }
  const button = $("run-all");
  button.disabled = true;
  setStatus("Running checks…");
  try {
    report = await api("/api/analyze", body);
    render();
    setStatus("Checks finished. Nothing was deleted and nothing left this machine.");
    showTab("report");
  } catch (error) {
    setStatus(error.message, "bad");
  } finally {
    button.disabled = false;
  }
});

function render() {
  renderDataset();
  renderDuplicates();
  renderBalance();
  renderCoverage();
  renderGate();
  renderReport();
}

function renderDataset() {
  const counts = report.counts;
  const thumbs = report.duplicates.groups
    .flatMap((group) => group.images)
    .slice(0, 18)
    .map((image) => `
      <figure class="thumb">
        <img src="${mediaURL(image.rel)}" alt="Synthetic sample ${esc(image.rel)}">
        <p class="name" title="${esc(image.rel)}">${esc(image.rel.split("/").pop())}</p>
        <p>${esc(image.split)} · ${image.width}×${image.height}</p>
      </figure>`)
    .join("");
  $("dataset-body").innerHTML = `
    <div class="stats">
      <div class="stat"><b>${counts.subject}</b> subject</div>
      <div class="stat"><b>${counts.reg}</b> regularization</div>
      <div class="stat"><b>${counts.captioned}</b> captions</div>
      <div class="stat"><b>${report.duplicates.multi_group_count}</b> dupe groups</div>
      <div class="stat"><b>${report.coverage.gaps.length}</b> coverage gaps</div>
      <div class="stat"><b>${report.gate.pending_count}</b> crops pending</div>
    </div>
    <div class="panel"><h2>Images</h2><div class="thumbs">${thumbs}</div></div>
    ${report.errors.length ? `<div class="callout bad">${report.errors.map(esc).join("<br>")}</div>` : ""}`;
}

function renderDuplicates() {
  const groups = report.duplicates.multi_groups;
  const drops = groups.flatMap((group) => group.images.filter((image) => image.role === "drop").map((image) => image.rel));
  const cards = groups.map((group) => {
    const frames = group.images.map((image) => `
      <figure class="frame-card ${image.role}">
        <img src="${mediaURL(image.rel)}" alt="${esc(image.rel)}">
        <p><span class="pill ${image.role}">${image.role}</span></p>
        <p class="name" title="${esc(image.rel)}">${esc(image.rel.split("/").pop())}</p>
        <p>${image.width}×${image.height} · sharpness ${image.sharpness}</p>
      </figure>`).join("");
    return `<article class="panel">
      <h2>Group ${group.id} · keep ${esc(group.suggested_keep.split("/").pop())}</h2>
      <p class="meta">${esc(group.keep_reason)}</p>
      <div class="film">${frames}</div>
    </article>`;
  }).join("");
  $("duplicates-body").innerHTML = `
    <div class="panel">
      <p>${report.duplicates.multi_group_count} near-duplicate group(s), ${report.duplicates.unique_count} unique images. ${esc(report.duplicates.similarity)}</p>
      <div class="actions">
        <button type="button" id="download-csv" class="ghost">Download keep/drop CSV</button>
        <button type="button" id="move-drops" class="bad" ${drops.length ? "" : "disabled"}>Move ${drops.length} suggested drops to _rejected/</button>
      </div>
      <p class="hint">Move renames files into _rejected/. It does not delete them.</p>
    </div>
    ${cards || `<div class="panel">No near-duplicate groups at these thresholds.</div>`}`;
  $("download-csv").addEventListener("click", downloadCsv);
  const move = $("move-drops");
  if (move) {
    move.addEventListener("click", () => moveDrops(drops));
  }
}

async function downloadCsv() {
  try {
    const response = await fetch("/api/duplicates/csv", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options()),
    });
    if (!response.ok) throw new Error("Could not build the CSV");
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "keep-drop.csv";
    link.click();
    URL.revokeObjectURL(link.href);
  } catch (error) {
    setStatus(error.message, "bad");
  }
}

async function moveDrops(files) {
  if (!files.length) return;
  const ok = window.confirm(`Move ${files.length} files into _rejected/? Nothing is deleted; they are renamed into that folder.`);
  if (!ok) return;
  try {
    await api("/api/duplicates/move", { path: report.dataset, files, confirm: true });
    report = await api("/api/analyze", options());
    render();
    setStatus(`Moved ${files.length} files into _rejected/.`);
    showTab("duplicates");
  } catch (error) {
    setStatus(error.message, "bad");
  }
}

function renderBalance() {
  const balance = report.balance;
  const tone = balance.action === "ok" ? "good" : balance.action === "none" ? "" : "warn";
  const rows = balance.concepts.map((row) => `
    <tr>
      <td>${esc(row.concept)}</td>
      <td>${esc(row.class_token || "—")}</td>
      <td>${row.subject_images}</td>
      <td>${row.reg_images}</td>
      <td>${esc(row.action)} ${row.count || ""}</td>
    </tr>`).join("");
  const ratio = balance.reg_per_subject == null ? "—" : balance.reg_per_subject.toFixed(2);
  $("balance-body").innerHTML = `
    <div class="stats">
      <div class="stat"><b>${balance.subject_count}</b> subject</div>
      <div class="stat"><b>${balance.reg_count}</b> reg</div>
      <div class="stat"><b>${ratio}</b> reg / subject</div>
      <div class="stat"><b>${balance.target_min}–${balance.target_max}</b> target reg</div>
    </div>
    <div class="callout ${tone}">${esc(balance.message)}</div>
    <div class="panel">
      <h2>Per concept</h2>
      <table>
        <thead><tr><th>Concept</th><th>Class</th><th>Subject</th><th>Reg</th><th>Advice</th></tr></thead>
        <tbody>${rows || `<tr><td colspan="5">No captions with a concept tag.</td></tr>`}</tbody>
      </table>
    </div>`;
}

function renderCoverage() {
  const coverage = report.coverage;
  const header = coverage.framings.map((framing) => `<div class="head">${esc(framing)}</div>`).join("");
  const rows = coverage.poses.map((pose) => {
    const cells = coverage.framings.map((framing) => {
      const cell = coverage.heatmap.find((item) => item.pose === pose && item.framing === framing);
      const level = cell.count <= 0 ? "gap" : cell.count === 1 ? "c1" : cell.count === 2 ? "c2" : "c3";
      const label = cell.gap ? "gap" : cell.count;
      return `<div class="cell ${level}" title="${esc(cell.files.join(", "))}"><b>${label}</b>${esc(pose.split("_").join(" "))}</div>`;
    }).join("");
    return `<div class="head">${esc(pose.split("_").join(" "))}</div>${cells}`;
  }).join("");
  const gaps = coverage.gaps.map((gap) => `<li>${esc(gap.suggestion)}</li>`).join("");
  const maxBucket = Math.max(1, ...coverage.buckets.sdxl.map((row) => row.count), ...coverage.buckets.sd15.map((row) => row.count));
  const bars = (model) => coverage.buckets[model].filter((row) => row.count).map((row) => `
    <div class="bar-row">
      <span>${esc(row.name)}</span>
      <div class="bar"><span style="width:${(row.count / maxBucket) * 100}%"></span></div>
      <span>${row.count}</span>
    </div>`).join("");
  const faces = coverage.faces.map((face) => `<li>${esc(face.rel)} · ${esc(face.pose)} · ${esc(face.framing)}</li>`).join("");
  $("coverage-body").innerHTML = `
    <div class="panel">
      <h2>Pose × framing</h2>
      <p class="legend"><span class="swatch" style="background:#2a2158"></span> 1
        <span class="swatch" style="background:#b4532a"></span> more
        <span class="swatch" style="background:repeating-linear-gradient(135deg,#2a1822,#2a1822 4px,#3a1c28 4px,#3a1c28 8px)"></span> gap</p>
      <div class="heat"><div></div>${header}${rows}</div>
    </div>
    <div class="panel"><h2>Add these</h2><ul>${gaps || "<li>No gaps at this target.</li>"}</ul></div>
    <div class="panel"><h2>SDXL buckets</h2><div class="bars">${bars("sdxl") || "<p>No images.</p>"}</div></div>
    <div class="panel"><h2>SD1.5 buckets</h2><div class="bars">${bars("sd15") || "<p>No images.</p>"}</div></div>
    <div class="panel"><h2>Detected faces</h2><ul>${faces || "<li>No faces.</li>"}</ul>
      <p class="hint">No face: ${coverage.no_face.map(esc).join(", ") || "none"}</p></div>`;
}

function renderGate() {
  const gate = report.gate;
  const cards = gate.crops.map((crop) => {
    const [x, y, w, h] = crop.crop_box;
    const box = `left:${(x / crop.image_width) * 100}%;top:${(y / crop.image_height) * 100}%;width:${(w / crop.image_width) * 100}%;height:${(h / crop.image_height) * 100}%`;
    return `<article class="crop-card">
      <div class="crop-frame">
        <img src="${mediaURL(crop.file)}" alt="Crop candidate ${esc(crop.file)}">
        <div class="crop-box" style="${box}"></div>
      </div>
      <div class="crop-meta">
        <p><span class="pill ${crop.status}">${esc(crop.status)}</span></p>
        <p class="name" title="${esc(crop.file)}">${esc(crop.file)}</p>
        <p>${esc(crop.pose)} · ${esc(crop.framing)} · box ${crop.crop_box.join(", ")}</p>
        <div class="actions">
          <button type="button" class="good" data-decision="approve" data-crop="${esc(crop.crop_id)}">Approve</button>
          <button type="button" class="bad" data-decision="reject" data-crop="${esc(crop.crop_id)}">Reject</button>
        </div>
        <form class="recrop" data-crop="${esc(crop.crop_id)}">
          <div class="options">
            <label>x <input name="x" type="number" value="${x}"></label>
            <label>y <input name="y" type="number" value="${y}"></label>
            <label>w <input name="w" type="number" value="${w}"></label>
            <label>h <input name="h" type="number" value="${h}"></label>
          </div>
          <button type="submit" class="ghost">Re-crop</button>
        </form>
      </div>
    </article>`;
  }).join("");
  const tail = gate.log.slice(-8).map((entry) => `${entry.timestamp}  ${entry.decision}  ${entry.file}  ${entry.crop_box.join(",")}`).join("\n");
  $("gate-body").innerHTML = `
    <div class="panel">
      <p>${esc(gate.block_reason || "Every crop is approved. Export writes only those crops.")}</p>
      <p class="meta">Approved ${gate.approved_count} · rejected ${gate.rejected_count} · pending ${gate.pending_count}</p>
      <div class="actions">
        <button type="button" id="export-crops" class="primary" ${gate.export_enabled ? "" : "disabled"}>Export approved crops</button>
      </div>
      <p class="hint">The export control stays off until every crop is approved. Rejected crops hold the gate shut until you re-crop and approve them. The log at ${esc(gate.log_path)} is append-only.</p>
    </div>
    ${cards || `<div class="panel">No faces to crop.</div>`}
    <div class="panel"><h2>Approval log</h2><div class="log">${esc(tail || "No decisions yet.")}</div></div>`;
  $("export-crops").addEventListener("click", exportCrops);
  $("gate-body").querySelectorAll("button[data-decision]").forEach((button) => {
    button.addEventListener("click", () => sendDecision(button.dataset.crop, button.dataset.decision));
  });
  $("gate-body").querySelectorAll("form.recrop").forEach((form) => {
    form.addEventListener("submit", (event) => {
      event.preventDefault();
      const data = new FormData(form);
      sendDecision(form.dataset.crop, "recrop", [
        Number(data.get("x")),
        Number(data.get("y")),
        Number(data.get("w")),
        Number(data.get("h")),
      ]);
    });
  });
}

async function sendDecision(cropId, decision, cropBox) {
  try {
    const body = { path: report.dataset, crop_id: cropId, decision };
    if (cropBox) body.crop_box = cropBox;
    const gate = await api("/api/gate/decision", body);
    report.gate = gate;
    renderGate();
    renderReport();
    setStatus(gate.block_reason || "Decision recorded. Export is unlocked.");
  } catch (error) {
    setStatus(error.message, "bad");
  }
}

async function exportCrops() {
  try {
    const result = await api("/api/gate/export", { path: report.dataset });
    setStatus(result.message);
    showTab("gate");
  } catch (error) {
    setStatus(error.message, "bad");
  }
}

function renderReport() {
  const gaps = report.coverage.gaps.map((gap) => `<li>${esc(gap.suggestion)}</li>`).join("");
  $("report-body").innerHTML = `
    <div class="panel">
      <h2>Report</h2>
      <p class="meta">${esc(report.generated_at)} · ${esc(report.dataset)}</p>
      <div class="callout">${esc(report.balance.message)}</div>
      <p>${report.duplicates.multi_group_count} near-duplicate groups. Suggested keeps are the highest resolution, then the sharpest. Nothing was deleted.</p>
      <h3>Coverage gaps</h3>
      <ul>${gaps || "<li>No gaps.</li>"}</ul>
      <p>${esc(report.gate.block_reason || "Every crop is approved.")}</p>
      <div class="actions">
        <button type="button" id="download-json" class="ghost">Download JSON</button>
        <button type="button" id="download-html" class="ghost">Download HTML</button>
      </div>
      <p class="hint">CPU only. Loopback only. No remote training jobs. The approval log is append-only.</p>
    </div>`;
  $("download-json").addEventListener("click", () => {
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "loradatasetgate-report.json";
    link.click();
    URL.revokeObjectURL(link.href);
  });
  $("download-html").addEventListener("click", async () => {
    const response = await fetch("/api/report/html", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(options()),
    });
    const blob = await response.blob();
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = "loradatasetgate-report.html";
    link.click();
    URL.revokeObjectURL(link.href);
  });
}

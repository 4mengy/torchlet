(function () {
  "use strict";

  const payload = window.TORCHLET_CODE;
  const core = window.TorchletCompareCore;
  const params = new URLSearchParams(window.location.search);

  const elements = {
    allFiles: document.querySelector("#allFiles"),
    baseVersion: document.querySelector("#baseVersion"),
    changedFilesOnly: document.querySelector("#changedFilesOnly"),
    changePosition: document.querySelector("#changePosition"),
    diffSummary: document.querySelector("#diffSummary"),
    diffView: document.querySelector("#diffView"),
    evolutionSummary: document.querySelector("#evolutionSummary"),
    expandAll: document.querySelector("#expandAll"),
    fileNav: document.querySelector("#fileNav"),
    filePathTitle: document.querySelector("#filePathTitle"),
    fileStatus: document.querySelector("#fileStatus"),
    fileSummary: document.querySelector("#fileSummary"),
    nextChange: document.querySelector("#nextChange"),
    nextPair: document.querySelector("#nextPair"),
    previousChange: document.querySelector("#previousChange"),
    previousPair: document.querySelector("#previousPair"),
    splitView: document.querySelector("#splitView"),
    targetVersion: document.querySelector("#targetVersion"),
    unifiedView: document.querySelector("#unifiedView"),
  };

  if (!payload || !core) {
    elements.diffSummary.textContent = "Data unavailable";
    elements.diffView.innerHTML =
      '<div class="empty-code">Comparison data did not load.</div>';
    return;
  }

  function readSavedView() {
    try {
      return window.localStorage.getItem("torchlet-diff-view");
    } catch (_error) {
      return null;
    }
  }

  function saveView(view) {
    try {
      window.localStorage.setItem("torchlet-diff-view", view);
    } catch (_error) {
      // Some browsers restrict storage for pages opened directly from disk.
    }
  }

  const requestedPair = core.resolveVersionPair(
    payload,
    params.get("base"),
    params.get("target")
  );
  const requestedHunk = Math.max(
    0,
    Number.parseInt(params.get("change") || "0", 10)
  );
  let state = {
    baseId: requestedPair.baseId,
    targetId: requestedPair.targetId,
    comparison: null,
    currentFile: null,
    currentHunk: requestedHunk,
    expandedFolds: new Set(),
    showAllFiles: false,
    view:
      params.get("view") === "unified" ||
      (!params.has("view") && readSavedView() === "unified")
        ? "unified"
        : "split",
  };
  let requestedFile = params.get("file");

  function escapeHtml(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function versionById(versionId) {
    return payload.versions.find((version) => version.id === versionId);
  }

  function versionLabel(version) {
    return `${version.id} · ${version.title}`;
  }

  function createOption(version) {
    const option = document.createElement("option");
    option.value = version.id;
    option.textContent = versionLabel(version);
    return option;
  }

  function replaceOptions(select, versions, selectedId) {
    select.replaceChildren(...versions.map(createOption));
    select.value = selectedId;
  }

  function renderVersionControls() {
    replaceOptions(
      elements.baseVersion,
      payload.versions.slice(0, -1),
      state.baseId
    );
    replaceOptions(
      elements.targetVersion,
      core.laterVersions(payload, state.baseId),
      state.targetId
    );

    const baseIndex = payload.versions.findIndex(
      (version) => version.id === state.baseId
    );
    const targetIndex = payload.versions.findIndex(
      (version) => version.id === state.targetId
    );
    const isAdjacent = targetIndex === baseIndex + 1;
    elements.previousPair.disabled = isAdjacent && baseIndex === 0;
    elements.nextPair.disabled = targetIndex === payload.versions.length - 1;
  }

  function renderEvolutionSummary() {
    const baseIndex = payload.versions.findIndex(
      (version) => version.id === state.baseId
    );
    const targetIndex = payload.versions.findIndex(
      (version) => version.id === state.targetId
    );
    const target = payload.versions[targetIndex];
    const stages = payload.versions.slice(baseIndex + 1, targetIndex + 1);
    const targetLink = `../versions/${encodeURIComponent(target.id)}/`;
    const rationale = target.rationale || target.theme;

    if (stages.length === 1) {
      elements.evolutionSummary.innerHTML = `
        <div>
          <span class="summary-label">Evolution summary</span>
          <strong>${escapeHtml(target.title)}</strong>
          <span class="summary-theme">${escapeHtml(target.theme)}</span>
          <p>${escapeHtml(rationale)}</p>
        </div>
        <a href="${targetLink}">Open Version notes</a>`;
      return;
    }

    const stageLinks = stages
      .map(
        (stage) =>
          `<li><a href="../versions/${encodeURIComponent(stage.id)}/">${escapeHtml(
            versionLabel(stage)
          )}</a><span>${escapeHtml(stage.theme)}</span></li>`
      )
      .join("");
    elements.evolutionSummary.innerHTML = `
      <details>
        <summary>${stages.length} implementation stages in this comparison</summary>
        <ol>${stageLinks}</ol>
      </details>
      <a href="${targetLink}">Open Target Version notes</a>`;
  }

  function statusLabel(status) {
    return {
      added: "Added",
      deleted: "Deleted",
      modified: "Modified",
      unchanged: "Unchanged",
    }[status];
  }

  function groupFiles(files) {
    const groups = new Map();
    for (const file of files) {
      const separator = file.path.lastIndexOf("/");
      const directory = separator === -1 ? "Root" : file.path.slice(0, separator);
      const name = separator === -1 ? file.path : file.path.slice(separator + 1);
      if (!groups.has(directory)) {
        groups.set(directory, []);
      }
      groups.get(directory).push({ ...file, name });
    }
    return groups;
  }

  function renderFileNav() {
    const allFiles = state.comparison.files;
    const visibleFiles = state.showAllFiles
      ? allFiles
      : allFiles.filter((file) => file.status !== "unchanged");
    const changedCount = allFiles.filter(
      (file) => file.status !== "unchanged"
    ).length;
    elements.fileSummary.textContent = `${changedCount} changed · ${allFiles.length} total`;
    elements.changedFilesOnly.setAttribute(
      "aria-pressed",
      String(!state.showAllFiles)
    );
    elements.allFiles.setAttribute("aria-pressed", String(state.showAllFiles));

    if (visibleFiles.length === 0) {
      elements.fileNav.innerHTML =
        '<p class="file-empty">No code changes between these Versions.</p>';
      return;
    }

    elements.fileNav.innerHTML = Array.from(groupFiles(visibleFiles))
      .map(([directory, files]) => {
        const items = files
          .map((file) => {
            const current = file.path === state.currentFile?.path;
            const statusCode = {
              added: "A",
              deleted: "D",
              modified: "M",
              unchanged: "·",
            }[file.status];
            return `<button class="file-item file-${file.status}" type="button"
              data-file="${escapeHtml(file.path)}"${
                current ? ' aria-current="true"' : ""
              }>
              <span class="file-status-code">${statusCode}</span>
              <span class="file-name">${escapeHtml(file.name)}</span>
              <span class="file-stats"><span class="add">+${
                file.added
              }</span><span class="delete">-${file.deleted}</span></span>
            </button>`;
          })
          .join("");
        return `<section class="file-group"><h3>${escapeHtml(
          directory
        )}</h3>${items}</section>`;
      })
      .join("");

    for (const button of elements.fileNav.querySelectorAll("[data-file]")) {
      button.addEventListener("click", () => {
        state.currentFile = state.comparison.files.find(
          (file) => file.path === button.dataset.file
        );
        state.currentHunk = 0;
        state.expandedFolds.clear();
        renderFileNav();
        renderDiff();
      });
    }
  }

  const pythonKeywords = new Set([
    "and",
    "as",
    "assert",
    "async",
    "await",
    "break",
    "class",
    "continue",
    "def",
    "del",
    "elif",
    "else",
    "except",
    "False",
    "finally",
    "for",
    "from",
    "global",
    "if",
    "import",
    "in",
    "is",
    "lambda",
    "None",
    "nonlocal",
    "not",
    "or",
    "pass",
    "raise",
    "return",
    "True",
    "try",
    "while",
    "with",
    "yield",
  ]);

  function syntaxClass(token) {
    if (/^\s+$/.test(token)) return "";
    if (/^#/.test(token)) return "syntax-comment";
    if (/^(?:[rubf]+)?["']/.test(token)) return "syntax-string";
    if (/^\d/.test(token)) return "syntax-number";
    if (pythonKeywords.has(token)) return "syntax-keyword";
    return "";
  }

  function syntaxHighlight(text) {
    const tokens =
      text.match(/#[^\n]*|(?:[rubf]+)?"(?:\\.|[^"\\])*"|(?:[rubf]+)?'(?:\\.|[^'\\])*'|\b[A-Za-z_]\w*\b|\b\d+(?:\.\d+)?\b|\s+|./gu) || [];
    return tokens
      .map((token) => {
        const className = syntaxClass(token);
        return className
          ? `<span class="${className}">${escapeHtml(token)}</span>`
          : escapeHtml(token);
      })
      .join("");
  }

  function renderSegments(segments, side, syntax) {
    return segments
      .map((segment) => {
        const code =
          syntax === "string"
            ? `<span class="syntax-string">${escapeHtml(segment.text)}</span>`
            : syntaxHighlight(segment.text);
        return segment.changed
          ? `<mark class="word-change ${side}">${code}</mark>`
          : code;
      })
      .join("");
  }

  function lineCode(row, side) {
    const text = side === "base" ? row.baseText : row.targetText;
    const segments = side === "base" ? row.baseSegments : row.targetSegments;
    const syntax = side === "base" ? row.baseSyntax : row.targetSyntax;
    if (segments) {
      return renderSegments(segments, side, syntax);
    }
    return syntax === "string"
      ? `<span class="syntax-string">${escapeHtml(text || "")}</span>`
      : syntaxHighlight(text || "");
  }

  function displayRows(file) {
    const rows = [];
    for (const row of file.displayRows) {
      if (row.kind !== "fold" || !state.expandedFolds.has(row.id)) {
        rows.push(row);
      } else {
        rows.push(...file.rows.slice(row.startRow, row.endRow + 1));
      }
    }
    return rows;
  }

  function hunkAttribute(row) {
    const file = state.currentFile;
    const hunk = file.hunks.find((item) => item.startRow === row.rowIndex);
    return hunk ? ` id="hunk-${hunk.index}" data-hunk="${hunk.index}"` : "";
  }

  function renderFoldRow(row, columnCount) {
    return `<tr class="diff-fold"><td colspan="${columnCount}"><button type="button" data-fold="${row.id}">Expand ${row.count} unchanged lines</button></td></tr>`;
  }

  function renderSplitRows(rows) {
    return rows
      .map((row) => {
        if (row.kind === "fold") return renderFoldRow(row, 6);
        const baseMarker = row.kind === "equal" || row.kind === "add" ? "" : "−";
        const targetMarker =
          row.kind === "equal" || row.kind === "delete" ? "" : "+";
        return `<tr class="diff-row diff-${row.kind}"${hunkAttribute(row)}>
          <td class="diff-line-no base">${row.baseNo || ""}</td>
          <td class="diff-marker base">${baseMarker}</td>
          <td class="diff-code base"><code>${lineCode(row, "base")}</code></td>
          <td class="diff-line-no target">${row.targetNo || ""}</td>
          <td class="diff-marker target">${targetMarker}</td>
          <td class="diff-code target"><code>${lineCode(row, "target")}</code></td>
        </tr>`;
      })
      .join("");
  }

  function unifiedRow(row, side, kind, marker, includeAnchor = true) {
    const lineNo = side === "base" ? row.baseNo : row.targetNo;
    return `<tr class="diff-row diff-${kind}"${
      includeAnchor ? hunkAttribute(row) : ""
    }>
      <td class="diff-line-no base">${side === "base" ? lineNo || "" : ""}</td>
      <td class="diff-line-no target">${side === "target" ? lineNo || "" : ""}</td>
      <td class="diff-marker ${side}">${marker}</td>
      <td class="diff-code ${side}"><code>${lineCode(row, side)}</code></td>
    </tr>`;
  }

  function renderUnifiedRows(rows) {
    return rows
      .map((row) => {
        if (row.kind === "fold") return renderFoldRow(row, 4);
        if (row.kind === "change") {
          return (
            unifiedRow(row, "base", "delete", "−") +
            unifiedRow(row, "target", "add", "+", false)
          );
        }
        if (row.kind === "delete") return unifiedRow(row, "base", "delete", "−");
        if (row.kind === "add") return unifiedRow(row, "target", "add", "+");
        return `<tr class="diff-row diff-equal">
          <td class="diff-line-no base">${row.baseNo || ""}</td>
          <td class="diff-line-no target">${row.targetNo || ""}</td>
          <td class="diff-marker"></td>
          <td class="diff-code target"><code>${lineCode(row, "target")}</code></td>
        </tr>`;
      })
      .join("");
  }

  function renderDiff() {
    const file = state.currentFile;
    if (!file) {
      elements.filePathTitle.textContent = "No changed files";
      elements.fileStatus.textContent = "";
      elements.diffSummary.textContent = "0 changes";
      elements.diffView.innerHTML =
        '<div class="empty-code">These Versions contain the same code.</div>';
      updateUrl();
      return;
    }

    elements.filePathTitle.textContent = file.path;
    elements.fileStatus.textContent = statusLabel(file.status);
    elements.fileStatus.className = `file-status file-${file.status}`;
    elements.diffSummary.innerHTML = `<span class="diff-count add">+${file.added}</span><span class="diff-count delete">-${file.deleted}</span>`;
    state.currentHunk = Math.min(
      state.currentHunk,
      Math.max(0, file.hunks.length - 1)
    );
    renderChangePosition();

    const rows = displayRows(file);
    const base = versionById(state.baseId);
    const target = versionById(state.targetId);
    const tableHeading =
      state.view === "split"
        ? `<tr><th colspan="3">${escapeHtml(
            versionLabel(base)
          )}</th><th colspan="3">${escapeHtml(versionLabel(target))}</th></tr>`
        : `<tr><th colspan="4">${escapeHtml(versionLabel(base))} → ${escapeHtml(
            versionLabel(target)
          )}</th></tr>`;
    const renderedRows =
      state.view === "split" ? renderSplitRows(rows) : renderUnifiedRows(rows);
    elements.diffView.innerHTML = `<table class="diff-table diff-${state.view}"><thead>${tableHeading}</thead><tbody>${renderedRows}</tbody></table>`;

    for (const button of elements.diffView.querySelectorAll("[data-fold]")) {
      button.addEventListener("click", () => {
        state.expandedFolds.add(button.dataset.fold);
        renderDiff();
      });
    }
    updateViewControls();
    updateUrl();
  }

  function renderChangePosition() {
    const hunkCount = state.currentFile ? state.currentFile.hunks.length : 0;
    elements.changePosition.textContent = hunkCount
      ? `${state.currentHunk + 1} / ${hunkCount} changes`
      : "No changes";
    elements.previousChange.disabled = hunkCount === 0 || state.currentHunk === 0;
    elements.nextChange.disabled =
      hunkCount === 0 || state.currentHunk === hunkCount - 1;
    const folds = state.currentFile
      ? state.currentFile.displayRows.filter((row) => row.kind === "fold")
      : [];
    elements.expandAll.disabled =
      folds.length === 0 ||
      folds.every((row) => state.expandedFolds.has(row.id));
  }

  function updateViewControls() {
    elements.splitView.setAttribute("aria-pressed", String(state.view === "split"));
    elements.unifiedView.setAttribute(
      "aria-pressed",
      String(state.view === "unified")
    );
  }

  function updateUrl() {
    const nextParams = new URLSearchParams({
      base: state.baseId,
      target: state.targetId,
      view: state.view,
      change: String(state.currentHunk),
    });
    if (state.currentFile) nextParams.set("file", state.currentFile.path);
    try {
      window.history.replaceState(null, "", `?${nextParams.toString()}`);
    } catch (_error) {
      // Keep offline file browsing functional if history mutation is restricted.
    }
  }

  function selectComparison(filePath, hunkIndex = 0, restoreScroll = false) {
    state.comparison = core.compareVersions(payload, state.baseId, state.targetId);
    const target = versionById(state.targetId);
    state.currentFile = core.chooseDefaultFile(
      state.comparison,
      target.important_file,
      filePath
    );
    if (filePath && state.currentFile?.status === "unchanged") {
      state.showAllFiles = true;
    }
    state.currentHunk = hunkIndex;
    state.expandedFolds.clear();
    renderVersionControls();
    renderEvolutionSummary();
    renderFileNav();
    renderDiff();
    if (restoreScroll && state.currentHunk > 0) {
      window.requestAnimationFrame(() => {
        document.querySelector(`#hunk-${state.currentHunk}`)?.scrollIntoView({
          block: "center",
        });
      });
    }
  }

  function selectPair(baseId, targetId) {
    state.baseId = baseId;
    state.targetId = targetId;
    requestedFile = null;
    selectComparison(null);
  }

  function moveToHunk(nextIndex) {
    if (!state.currentFile || nextIndex < 0 || nextIndex >= state.currentFile.hunks.length) {
      return;
    }
    state.currentHunk = nextIndex;
    renderChangePosition();
    updateUrl();
    document.querySelector(`#hunk-${state.currentHunk}`)?.scrollIntoView({
      block: "center",
      behavior: "smooth",
    });
  }

  elements.baseVersion.addEventListener("change", () => {
    const targets = core.laterVersions(payload, elements.baseVersion.value);
    selectPair(elements.baseVersion.value, targets[0].id);
  });
  elements.targetVersion.addEventListener("change", () => {
    selectPair(state.baseId, elements.targetVersion.value);
  });
  elements.changedFilesOnly.addEventListener("click", () => {
    state.showAllFiles = false;
    renderFileNav();
  });
  elements.allFiles.addEventListener("click", () => {
    state.showAllFiles = true;
    renderFileNav();
  });
  elements.splitView.addEventListener("click", () => {
    state.view = "split";
    saveView(state.view);
    renderDiff();
  });
  elements.unifiedView.addEventListener("click", () => {
    state.view = "unified";
    saveView(state.view);
    renderDiff();
  });
  elements.previousPair.addEventListener("click", () => {
    const baseIndex = payload.versions.findIndex(
      (version) => version.id === state.baseId
    );
    const targetIndex = payload.versions.findIndex(
      (version) => version.id === state.targetId
    );
    if (targetIndex > baseIndex + 1) {
      selectPair(
        payload.versions[targetIndex - 1].id,
        payload.versions[targetIndex].id
      );
    } else if (baseIndex > 0) {
      selectPair(payload.versions[baseIndex - 1].id, payload.versions[baseIndex].id);
    }
  });
  elements.nextPair.addEventListener("click", () => {
    const targetIndex = payload.versions.findIndex(
      (version) => version.id === state.targetId
    );
    if (targetIndex < payload.versions.length - 1) {
      selectPair(payload.versions[targetIndex].id, payload.versions[targetIndex + 1].id);
    }
  });
  elements.previousChange.addEventListener("click", () =>
    moveToHunk(state.currentHunk - 1)
  );
  elements.nextChange.addEventListener("click", () =>
    moveToHunk(state.currentHunk + 1)
  );
  elements.expandAll.addEventListener("click", () => {
    for (const row of state.currentFile.displayRows) {
      if (row.kind === "fold") state.expandedFolds.add(row.id);
    }
    renderDiff();
  });
  document.addEventListener("keydown", (event) => {
    if (event.target.matches("input, select, textarea, button")) return;
    if (event.key === "[") moveToHunk(state.currentHunk - 1);
    if (event.key === "]") moveToHunk(state.currentHunk + 1);
  });

  selectComparison(requestedFile, requestedHunk, true);
})();

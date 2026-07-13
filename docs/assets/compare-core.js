(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  }
  root.TorchletCompareCore = api;
})(typeof globalThis === "undefined" ? this : globalThis, function () {
  "use strict";

  function splitLines(code) {
    if (code === undefined || code === "") {
      return [];
    }
    return code.endsWith("\n") ? code.slice(0, -1).split("\n") : code.split("\n");
  }

  function alignSequences(baseItems, targetItems) {
    const commonLengths = Array.from({ length: baseItems.length + 1 }, () =>
      Array(targetItems.length + 1).fill(0)
    );

    for (let baseIndex = baseItems.length - 1; baseIndex >= 0; baseIndex -= 1) {
      for (
        let targetIndex = targetItems.length - 1;
        targetIndex >= 0;
        targetIndex -= 1
      ) {
        commonLengths[baseIndex][targetIndex] =
          baseItems[baseIndex] === targetItems[targetIndex]
            ? commonLengths[baseIndex + 1][targetIndex + 1] + 1
            : Math.max(
                commonLengths[baseIndex + 1][targetIndex],
                commonLengths[baseIndex][targetIndex + 1]
              );
      }
    }

    const operations = [];
    let baseIndex = 0;
    let targetIndex = 0;
    while (baseIndex < baseItems.length && targetIndex < targetItems.length) {
      if (baseItems[baseIndex] === targetItems[targetIndex]) {
        operations.push({
          kind: "equal",
          baseIndex,
          targetIndex,
          baseItem: baseItems[baseIndex],
          targetItem: targetItems[targetIndex],
        });
        baseIndex += 1;
        targetIndex += 1;
      } else if (
        commonLengths[baseIndex + 1][targetIndex] >=
        commonLengths[baseIndex][targetIndex + 1]
      ) {
        operations.push({
          kind: "delete",
          baseIndex,
          baseItem: baseItems[baseIndex],
        });
        baseIndex += 1;
      } else {
        operations.push({
          kind: "add",
          targetIndex,
          targetItem: targetItems[targetIndex],
        });
        targetIndex += 1;
      }
    }

    while (baseIndex < baseItems.length) {
      operations.push({
        kind: "delete",
        baseIndex,
        baseItem: baseItems[baseIndex],
      });
      baseIndex += 1;
    }
    while (targetIndex < targetItems.length) {
      operations.push({
        kind: "add",
        targetIndex,
        targetItem: targetItems[targetIndex],
      });
      targetIndex += 1;
    }
    return operations;
  }

  function buildOperations(baseLines, targetLines) {
    return alignSequences(baseLines, targetLines).map((operation) => ({
      kind: operation.kind,
      ...(operation.baseIndex === undefined
        ? {}
        : {
            baseNo: operation.baseIndex + 1,
            baseText: operation.baseItem,
          }),
      ...(operation.targetIndex === undefined
        ? {}
        : {
            targetNo: operation.targetIndex + 1,
            targetText: operation.targetItem,
          }),
    }));
  }

  function buildRows(baseCode, targetCode) {
    const operations = buildOperations(splitLines(baseCode), splitLines(targetCode));
    const rows = [];
    let added = 0;
    let deleted = 0;
    let index = 0;

    while (index < operations.length) {
      const operation = operations[index];
      if (operation.kind === "equal") {
        rows.push(operation);
        index += 1;
        continue;
      }

      const deletes = [];
      const additions = [];
      while (index < operations.length && operations[index].kind !== "equal") {
        if (operations[index].kind === "delete") {
          deletes.push(operations[index]);
        } else {
          additions.push(operations[index]);
        }
        index += 1;
      }

      deleted += deletes.length;
      added += additions.length;
      const changedRows = Math.max(deletes.length, additions.length);
      for (let offset = 0; offset < changedRows; offset += 1) {
        const base = deletes[offset];
        const target = additions[offset];
        if (base && target) {
          rows.push({
            kind: "change",
            baseNo: base.baseNo,
            targetNo: target.targetNo,
            baseText: base.baseText,
            targetText: target.targetText,
          });
        } else {
          rows.push(base || target);
        }
      }
    }
    return { rows, added, deleted };
  }

  function splitTokens(text) {
    return text.match(/\s+|[A-Za-z_]\w*|\d+(?:\.\d+)?|./gu) || [];
  }

  function mergeSegments(segments) {
    const merged = [];
    for (const segment of segments) {
      const previous = merged[merged.length - 1];
      if (previous && previous.changed === segment.changed) {
        previous.text += segment.text;
      } else {
        merged.push({ ...segment });
      }
    }
    return merged;
  }

  function diffSegments(baseText, targetText) {
    const baseTokens = splitTokens(baseText);
    const targetTokens = splitTokens(targetText);
    const baseSegments = [];
    const targetSegments = [];
    for (const operation of alignSequences(baseTokens, targetTokens)) {
      if (operation.kind === "equal") {
        baseSegments.push({ text: operation.baseItem, changed: false });
        targetSegments.push({ text: operation.targetItem, changed: false });
      } else if (operation.kind === "delete") {
        baseSegments.push({ text: operation.baseItem, changed: true });
      } else {
        targetSegments.push({ text: operation.targetItem, changed: true });
      }
    }
    return {
      baseSegments: mergeSegments(baseSegments),
      targetSegments: mergeSegments(targetSegments),
    };
  }

  function addHunks(rows) {
    const hunks = [];
    let activeHunk = null;
    rows.forEach((row, rowIndex) => {
      row.rowIndex = rowIndex;
      if (row.kind === "equal") {
        activeHunk = null;
        return;
      }
      if (!activeHunk) {
        activeHunk = {
          index: hunks.length,
          startRow: rowIndex,
          endRow: rowIndex,
        };
        hunks.push(activeHunk);
      } else {
        activeHunk.endRow = rowIndex;
      }
      row.hunkIndex = activeHunk.index;
    });
    return hunks;
  }

  function foldUnchangedRows(rows, contextLines, hunks) {
    if (hunks.length === 0) {
      return rows;
    }
    const visible = Array(rows.length).fill(false);
    for (const hunk of hunks) {
      const first = Math.max(0, hunk.startRow - contextLines);
      const last = Math.min(rows.length - 1, hunk.endRow + contextLines);
      for (let index = first; index <= last; index += 1) {
        visible[index] = true;
      }
    }

    const displayRows = [];
    let index = 0;
    while (index < rows.length) {
      if (visible[index]) {
        displayRows.push(rows[index]);
        index += 1;
        continue;
      }
      const start = index;
      while (index < rows.length && !visible[index]) {
        index += 1;
      }
      displayRows.push({
        kind: "fold",
        id: `fold-${start}-${index - 1}`,
        startRow: start,
        endRow: index - 1,
        count: index - start,
      });
    }
    return displayRows;
  }

  function annotatePythonStrings(rows, side) {
    const textKey = `${side}Text`;
    const syntaxKey = `${side}Syntax`;
    let openDelimiter = null;

    for (const row of rows) {
      const text = row[textKey];
      if (text === undefined) {
        continue;
      }
      if (openDelimiter) {
        row[syntaxKey] = "string";
        if (text.includes(openDelimiter)) {
          openDelimiter = null;
        }
        continue;
      }

      const doubleQuoteIndex = text.indexOf('\"\"\"');
      const singleQuoteIndex = text.indexOf("'''");
      const candidates = [
        { delimiter: '\"\"\"', index: doubleQuoteIndex },
        { delimiter: "'''", index: singleQuoteIndex },
      ].filter((candidate) => candidate.index !== -1);
      candidates.sort((left, right) => left.index - right.index);
      const opening = candidates[0];
      if (!opening) {
        continue;
      }

      row[syntaxKey] = "string";
      const closingIndex = text.indexOf(
        opening.delimiter,
        opening.index + opening.delimiter.length
      );
      if (closingIndex === -1) {
        openDelimiter = opening.delimiter;
      }
    }
  }

  function diffFile(baseCode, targetCode, options = {}) {
    const contextLines = options.contextLines ?? 3;
    const diff = buildRows(baseCode, targetCode);
    for (const row of diff.rows) {
      if (row.kind === "change") {
        Object.assign(row, diffSegments(row.baseText, row.targetText));
      }
    }
    annotatePythonStrings(diff.rows, "base");
    annotatePythonStrings(diff.rows, "target");
    const hunks = addHunks(diff.rows);
    return {
      ...diff,
      hunks,
      displayRows: foldUnchangedRows(diff.rows, contextLines, hunks),
    };
  }

  function compareFile(path, baseCode, targetCode) {
    const diff = diffFile(baseCode, targetCode);
    let status = "modified";
    if (baseCode === undefined) {
      status = "added";
    } else if (targetCode === undefined) {
      status = "deleted";
    } else if (baseCode === targetCode) {
      status = "unchanged";
    }
    return { path, status, ...diff };
  }

  function compareVersions(payload, baseId, targetId) {
    const versionIds = payload.versions.map((version) => version.id);
    const baseIndex = versionIds.indexOf(baseId);
    const targetIndex = versionIds.indexOf(targetId);
    if (baseIndex === -1 || targetIndex === -1 || targetIndex <= baseIndex) {
      throw new Error("Target Version must be later than Base Version");
    }

    const baseFiles = payload.code[baseId] || {};
    const targetFiles = payload.code[targetId] || {};
    const paths = Array.from(
      new Set([...Object.keys(baseFiles), ...Object.keys(targetFiles)])
    ).sort();
    const files = paths.map((path) =>
      compareFile(path, baseFiles[path], targetFiles[path])
    );
    const changedFiles = files.filter((file) => file.status !== "unchanged");
    return {
      baseId,
      targetId,
      files,
      summary: {
        changedFiles: changedFiles.length,
        added: files.reduce((total, file) => total + file.added, 0),
        deleted: files.reduce((total, file) => total + file.deleted, 0),
      },
    };
  }

  function chooseDefaultFile(comparison, preferredPath, requestedPath) {
    const requested = comparison.files.find((file) => file.path === requestedPath);
    if (requested) {
      return requested;
    }
    const preferred = comparison.files.find(
      (file) => file.path === preferredPath && file.status !== "unchanged"
    );
    if (preferred) {
      return preferred;
    }
    const changedFiles = comparison.files
      .filter((file) => file.status !== "unchanged")
      .sort(
        (left, right) =>
          right.added + right.deleted - (left.added + left.deleted) ||
          left.path.localeCompare(right.path)
      );
    return changedFiles[0] || null;
  }

  function laterVersions(payload, baseId) {
    const baseIndex = payload.versions.findIndex((version) => version.id === baseId);
    return baseIndex === -1 ? [] : payload.versions.slice(baseIndex + 1);
  }

  function resolveVersionPair(payload, requestedBaseId, requestedTargetId) {
    if (payload.versions.length < 2) {
      throw new Error("At least two implemented Versions are required");
    }
    const requestedTargets = laterVersions(payload, requestedBaseId);
    if (requestedTargets.some((version) => version.id === requestedTargetId)) {
      return { baseId: requestedBaseId, targetId: requestedTargetId };
    }
    return {
      baseId: payload.versions[0].id,
      targetId: payload.versions[1].id,
    };
  }

  return {
    buildRows,
    chooseDefaultFile,
    compareVersions,
    diffFile,
    laterVersions,
    resolveVersionPair,
  };
});

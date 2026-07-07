(function () {
  const payload = window.TORCHLET_CODE;
  const params = new URLSearchParams(window.location.search);

  const leftVersion = document.querySelector("#leftVersion");
  const rightVersion = document.querySelector("#rightVersion");
  const filePath = document.querySelector("#filePath");
  const leftTitle = document.querySelector("#leftTitle");
  const rightTitle = document.querySelector("#rightTitle");
  const leftMeta = document.querySelector("#leftMeta");
  const rightMeta = document.querySelector("#rightMeta");
  const diffSummary = document.querySelector("#diffSummary");
  const diffView = document.querySelector("#diffView");

  function hasVersion(id) {
    return payload.versions.some((version) => version.id === id);
  }

  function hasFile(path) {
    return payload.files.includes(path);
  }

  function option(value, label) {
    const item = document.createElement("option");
    item.value = value;
    item.textContent = label;
    return item;
  }

  for (const version of payload.versions) {
    leftVersion.appendChild(option(version.id, version.id));
    rightVersion.appendChild(option(version.id, version.id));
  }

  for (const path of payload.files) {
    filePath.appendChild(option(path, path));
  }

  const lastIndex = payload.versions.length - 1;
  const defaultLeft = payload.versions[Math.max(0, lastIndex - 1)].id;
  const defaultRight = payload.versions[lastIndex].id;
  const defaultFile = payload.files.includes("layer/gqa.py")
    ? "layer/gqa.py"
    : payload.files[0];

  const requestedLeft = params.get("left");
  const requestedRight = params.get("right");
  const requestedFile = params.get("file");

  leftVersion.value = hasVersion(requestedLeft) ? requestedLeft : defaultLeft;
  rightVersion.value = hasVersion(requestedRight) ? requestedRight : defaultRight;
  filePath.value = hasFile(requestedFile) ? requestedFile : defaultFile;

  function escapeHtml(value) {
    return value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function splitLines(code) {
    if (code === undefined) {
      return [];
    }
    if (code === "") {
      return [""];
    }
    return code.endsWith("\n") ? code.slice(0, -1).split("\n") : code.split("\n");
  }

  function buildRawDiff(leftLines, rightLines) {
    const leftLength = leftLines.length;
    const rightLength = rightLines.length;
    const dp = Array.from({ length: leftLength + 1 }, () =>
      Array(rightLength + 1).fill(0)
    );

    for (let i = leftLength - 1; i >= 0; i -= 1) {
      for (let j = rightLength - 1; j >= 0; j -= 1) {
        if (leftLines[i] === rightLines[j]) {
          dp[i][j] = dp[i + 1][j + 1] + 1;
        } else {
          dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
        }
      }
    }

    const ops = [];
    let i = 0;
    let j = 0;
    while (i < leftLength && j < rightLength) {
      if (leftLines[i] === rightLines[j]) {
        ops.push({
          kind: "equal",
          leftNo: i + 1,
          rightNo: j + 1,
          leftText: leftLines[i],
          rightText: rightLines[j],
        });
        i += 1;
        j += 1;
      } else if (dp[i + 1][j] >= dp[i][j + 1]) {
        ops.push({
          kind: "delete",
          leftNo: i + 1,
          leftText: leftLines[i],
        });
        i += 1;
      } else {
        ops.push({
          kind: "add",
          rightNo: j + 1,
          rightText: rightLines[j],
        });
        j += 1;
      }
    }

    while (i < leftLength) {
      ops.push({
        kind: "delete",
        leftNo: i + 1,
        leftText: leftLines[i],
      });
      i += 1;
    }

    while (j < rightLength) {
      ops.push({
        kind: "add",
        rightNo: j + 1,
        rightText: rightLines[j],
      });
      j += 1;
    }

    return ops;
  }

  function buildRows(leftLines, rightLines) {
    const ops = buildRawDiff(leftLines, rightLines);
    const rows = [];
    let added = 0;
    let deleted = 0;
    let index = 0;

    while (index < ops.length) {
      const op = ops[index];
      if (op.kind === "equal") {
        rows.push(op);
        index += 1;
        continue;
      }

      const deletes = [];
      const adds = [];
      while (index < ops.length && ops[index].kind !== "equal") {
        if (ops[index].kind === "delete") {
          deletes.push(ops[index]);
        } else {
          adds.push(ops[index]);
        }
        index += 1;
      }

      deleted += deletes.length;
      added += adds.length;
      const maxRows = Math.max(deletes.length, adds.length);
      for (let offset = 0; offset < maxRows; offset += 1) {
        const left = deletes[offset];
        const right = adds[offset];
        if (left && right) {
          rows.push({
            kind: "change",
            leftNo: left.leftNo,
            rightNo: right.rightNo,
            leftText: left.leftText,
            rightText: right.rightText,
          });
        } else if (left) {
          rows.push(left);
        } else if (right) {
          rows.push(right);
        }
      }
    }

    return { rows, added, deleted };
  }

  function renderLineNo(value) {
    return value === undefined ? "" : value;
  }

  function renderDiff(leftCode, rightCode) {
    const leftMissing = leftCode === undefined;
    const rightMissing = rightCode === undefined;
    if (leftMissing && rightMissing) {
      diffSummary.textContent = "file missing";
      diffView.innerHTML = '<div class="empty-code">This file does not exist.</div>';
      return;
    }

    const leftLines = splitLines(leftCode);
    const rightLines = splitLines(rightCode);
    const { rows, added, deleted } = buildRows(leftLines, rightLines);

    diffSummary.innerHTML =
      `<span class="diff-count add">+${added}</span>` +
      `<span class="diff-count delete">-${deleted}</span>`;

    const renderedRows = rows
      .map((row) => {
        const leftMarker = row.kind === "add" ? "" : row.kind === "equal" ? "" : "-";
        const rightMarker = row.kind === "delete" ? "" : row.kind === "equal" ? "" : "+";
        const leftText = row.kind === "add" ? "" : row.leftText;
        const rightText = row.kind === "delete" ? "" : row.rightText;
        return `<tr class="diff-row diff-${row.kind}">
          <td class="diff-line-no old">${renderLineNo(row.leftNo)}</td>
          <td class="diff-marker old">${leftMarker}</td>
          <td class="diff-code old">${escapeHtml(leftText || "")}</td>
          <td class="diff-line-no new">${renderLineNo(row.rightNo)}</td>
          <td class="diff-marker new">${rightMarker}</td>
          <td class="diff-code new">${escapeHtml(rightText || "")}</td>
        </tr>`;
      })
      .join("");

    diffView.innerHTML = `<table class="diff-table"><tbody>${renderedRows}</tbody></table>`;
  }

  function render() {
    const left = leftVersion.value;
    const right = rightVersion.value;
    const path = filePath.value;

    leftTitle.textContent = left;
    rightTitle.textContent = right;
    const leftCode = payload.code[left] && payload.code[left][path];
    const rightCode = payload.code[right] && payload.code[right][path];
    leftMeta.textContent = leftCode === undefined ? `${path} missing` : path;
    rightMeta.textContent = rightCode === undefined ? `${path} missing` : path;

    renderDiff(leftCode, rightCode);

    const nextParams = new URLSearchParams({ left, right, file: path });
    window.history.replaceState(null, "", `?${nextParams.toString()}`);
  }

  leftVersion.addEventListener("change", render);
  rightVersion.addEventListener("change", render);
  filePath.addEventListener("change", render);
  render();
})();

(function () {
  const payload = window.TORCHLET_CODE;
  const leftVersion = document.querySelector("#leftVersion");
  const rightVersion = document.querySelector("#rightVersion");
  const filePath = document.querySelector("#filePath");
  const leftTitle = document.querySelector("#leftTitle");
  const rightTitle = document.querySelector("#rightTitle");
  const leftMeta = document.querySelector("#leftMeta");
  const rightMeta = document.querySelector("#rightMeta");
  const leftCode = document.querySelector("#leftCode");
  const rightCode = document.querySelector("#rightCode");

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
  leftVersion.value = payload.versions[Math.max(0, lastIndex - 1)].id;
  rightVersion.value = payload.versions[lastIndex].id;
  filePath.value = payload.files.includes("layer/gqa.py")
    ? "layer/gqa.py"
    : payload.files[0];

  function escapeHtml(value) {
    return value
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function renderCode(target, code) {
    if (code === undefined) {
      target.innerHTML =
        '<div class="empty-code">This file does not exist in this version.</div>';
      return;
    }

    const lines = code.endsWith("\n") ? code.slice(0, -1).split("\n") : code.split("\n");
    const rows = lines
      .map(
        (line, index) =>
          `<tr><td class="line-no">${index + 1}</td><td>${escapeHtml(line)}</td></tr>`
      )
      .join("");
    target.innerHTML = `<table class="code-table"><tbody>${rows}</tbody></table>`;
  }

  function render() {
    const left = leftVersion.value;
    const right = rightVersion.value;
    const path = filePath.value;

    leftTitle.textContent = left;
    rightTitle.textContent = right;
    leftMeta.textContent = path;
    rightMeta.textContent = path;

    renderCode(leftCode, payload.code[left] && payload.code[left][path]);
    renderCode(rightCode, payload.code[right] && payload.code[right][path]);
  }

  leftVersion.addEventListener("change", render);
  rightVersion.addEventListener("change", render);
  filePath.addEventListener("change", render);
  render();
})();

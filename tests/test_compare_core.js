const test = require("node:test");
const assert = require("node:assert/strict");

const compareCore = require("../docs/assets/compare-core.js");


test("comparison reports the union of files with directed statuses and totals", () => {
  const payload = {
    versions: [
      { id: "v00", title: "Baseline" },
      { id: "v01", title: "Next" },
    ],
    code: {
      v00: {
        "a.py": "same\nold\n",
        "deleted.py": "gone\n",
        "stable.py": "unchanged\n",
      },
      v01: {
        "a.py": "same\nnew\nextra\n",
        "added.py": "arrived\n",
        "stable.py": "unchanged\n",
      },
    },
  };

  const comparison = compareCore.compareVersions(payload, "v00", "v01");
  const files = Object.fromEntries(
    comparison.files.map((file) => [file.path, file])
  );

  assert.deepEqual(
    comparison.files.map((file) => file.path),
    ["a.py", "added.py", "deleted.py", "stable.py"]
  );
  assert.equal(files["a.py"].status, "modified");
  assert.deepEqual(
    [files["a.py"].added, files["a.py"].deleted],
    [2, 1]
  );
  assert.equal(files["added.py"].status, "added");
  assert.equal(files["deleted.py"].status, "deleted");
  assert.equal(files["stable.py"].status, "unchanged");
  assert.deepEqual(comparison.summary, {
    changedFiles: 3,
    added: 3,
    deleted: 2,
  });
});


test("file diff exposes folded context, hunks, and word-level changes", () => {
  const baseLines = [
    "line 1",
    "line 2",
    "line 3",
    "line 4",
    "line 5",
    "value = old_name + 1",
    "line 7",
    "line 8",
    "line 9",
    "line 10",
    "line 11",
  ];
  const targetLines = [...baseLines];
  targetLines[5] = "value = new_name + 1";

  const diff = compareCore.diffFile(
    `${baseLines.join("\n")}\n`,
    `${targetLines.join("\n")}\n`,
    { contextLines: 1 }
  );

  assert.equal(diff.hunks.length, 1);
  assert.deepEqual(
    diff.displayRows.map((row) =>
      row.kind === "fold" ? ["fold", row.count] : [row.kind, row.baseNo]
    ),
    [
      ["fold", 4],
      ["equal", 5],
      ["change", 6],
      ["equal", 7],
      ["fold", 4],
    ]
  );
  const changedRow = diff.rows.find((row) => row.kind === "change");
  assert.deepEqual(
    changedRow.baseSegments.filter((segment) => segment.changed),
    [{ text: "old_name", changed: true }]
  );
  assert.deepEqual(
    changedRow.targetSegments.filter((segment) => segment.changed),
    [{ text: "new_name", changed: true }]
  );
});


test("default file honors a URL request, then the preferred changed file", () => {
  const comparison = {
    files: [
      { path: "large.py", status: "modified", added: 20, deleted: 10 },
      { path: "preferred.py", status: "modified", added: 1, deleted: 1 },
      { path: "stable.py", status: "unchanged", added: 0, deleted: 0 },
    ],
  };

  assert.equal(
    compareCore.chooseDefaultFile(comparison, "preferred.py", "stable.py").path,
    "stable.py"
  );
  assert.equal(
    compareCore.chooseDefaultFile(comparison, "preferred.py", "missing.py").path,
    "preferred.py"
  );
  assert.equal(
    compareCore.chooseDefaultFile(comparison, "stable.py", null).path,
    "large.py"
  );
});


test("version pair resolution defaults safely and only exposes later targets", () => {
  const payload = {
    versions: [
      { id: "v00" },
      { id: "v01" },
      { id: "v02" },
      { id: "v03" },
    ],
  };

  assert.deepEqual(compareCore.resolveVersionPair(payload, null, null), {
    baseId: "v00",
    targetId: "v01",
  });
  assert.deepEqual(compareCore.resolveVersionPair(payload, "v01", "v03"), {
    baseId: "v01",
    targetId: "v03",
  });
  assert.deepEqual(compareCore.resolveVersionPair(payload, "v03", "v01"), {
    baseId: "v00",
    targetId: "v01",
  });
  assert.deepEqual(
    compareCore.laterVersions(payload, "v01").map((version) => version.id),
    ["v02", "v03"]
  );
});


test("an empty file has zero added or deleted lines", () => {
  const added = compareCore.diffFile(undefined, "");
  const deleted = compareCore.diffFile("", undefined);

  assert.deepEqual([added.added, added.deleted, added.rows.length], [0, 0, 0]);
  assert.deepEqual(
    [deleted.added, deleted.deleted, deleted.rows.length],
    [0, 0, 0]
  );
});


test("a comparison with no changes has no implicit current file", () => {
  const comparison = {
    files: [
      { path: "stable.py", status: "unchanged", added: 0, deleted: 0 },
    ],
  };

  assert.equal(compareCore.chooseDefaultFile(comparison, "stable.py", null), null);
  assert.equal(
    compareCore.chooseDefaultFile(comparison, "stable.py", "stable.py").path,
    "stable.py"
  );
});


test("file diff marks every line in a Python triple-quoted string", () => {
  const code = [
    "class Example:",
    '    \"\"\"Summary',
    "    details continue here",
    '    \"\"\"',
    "    pass",
    "",
  ].join("\n");

  const diff = compareCore.diffFile(code, code);

  assert.deepEqual(
    diff.rows.map((row) => row.baseSyntax || null),
    [null, "string", "string", "string", null]
  );
  assert.deepEqual(
    diff.rows.map((row) => row.targetSyntax || null),
    [null, "string", "string", "string", null]
  );
});

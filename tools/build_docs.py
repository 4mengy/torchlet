from __future__ import annotations

import argparse
import html
import json
import re
import shutil
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CONTENT = DOCS / "content"
ASSETS = DOCS / "assets"
VERSION_CONTENT = CONTENT / "versions"
SRC = ROOT / "src" / "torchlet"

DEFAULT_COMPARE_FILES = {
    "v00_full_recompute": "layer/gqa.py",
    "v01_0_ragged_batch": "layer/gqa.py",
    "v01_1_split_gqa": "layer/gqa.py",
    "v02_kv_cache": "layer/gqa.py",
    "v03_request_states": "scheduler.py",
    "v04_continuous_batching": "scheduler.py",
    "v05_decode_slots": "scheduler.py",
    "v06_static_buffers": "scheduler.py",
    "v07_cuda_graph": "engine.py",
    "v08_paged_gqa_py": "kvcache.py",
    "v09_triton_basics": "layer/gqa.py",
    "v10_triton_paged_gqa": "layer/gqa.py",
    "v11_cuda_graph_triton_paged": "engine.py",
}


def slug_to_title(slug: str) -> str:
    return slug.replace("_", " ")


def render_inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: (
            f'<a href="{html.escape(match.group(2), quote=True)}">'
            f"{match.group(1)}</a>"
        ),
        escaped,
    )
    return escaped


def render_markdown(markdown: str) -> str:
    html_parts: list[str] = []
    paragraph: list[str] = []
    list_items: list[str] = []
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            html_parts.append(f"<p>{render_inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    def flush_list() -> None:
        if list_items:
            html_parts.append("<ul>")
            html_parts.extend(f"<li>{item}</li>" for item in list_items)
            html_parts.append("</ul>")
            list_items.clear()

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip()

        if line.startswith("```"):
            if in_code:
                html_parts.append(
                    "<pre><code>"
                    + html.escape("\n".join(code_lines))
                    + "</code></pre>"
                )
                code_lines.clear()
                in_code = False
            else:
                flush_paragraph()
                flush_list()
                in_code = True
            continue

        if in_code:
            code_lines.append(raw_line)
            continue

        if not line:
            flush_paragraph()
            flush_list()
            continue

        if line.startswith("#"):
            flush_paragraph()
            flush_list()
            level = min(len(line) - len(line.lstrip("#")), 4)
            title = line[level:].strip()
            html_parts.append(f"<h{level}>{render_inline(title)}</h{level}>")
            continue

        if line.startswith("- "):
            flush_paragraph()
            list_items.append(render_inline(line[2:].strip()))
            continue

        paragraph.append(line)

    flush_paragraph()
    flush_list()
    return "\n".join(html_parts)


def read_versions() -> list[dict[str, str]]:
    with (CONTENT / "versions.json").open() as file:
        return json.load(file)


def compare_href(prefix: str, left: str, right: str, file_path: str) -> str:
    return (
        f"{prefix}/compare/?left={quote(left)}&right={quote(right)}"
        f"&file={quote(file_path, safe='')}"
    )


def action_link(href: str, label: str, class_name: str = "action-link") -> str:
    return f'<a class="{class_name}" href="{href}">{html.escape(label)}</a>'


def version_actions(
    version: dict[str, str],
    previous: dict[str, str] | None,
    versions: list[dict[str, str]],
) -> str:
    links = []
    prefix = "../.."
    version_id = version["id"]
    file_path = DEFAULT_COMPARE_FILES.get(version_id, "layer/gqa.py")

    if (
        previous
        and version["status"] == "implemented"
        and previous["status"] == "implemented"
    ):
        links.append(
            action_link(
                compare_href(prefix, previous["id"], version_id, file_path),
                f"Compare {previous['id']} -> {version_id}",
            )
        )
    elif version["status"] == "implemented":
        links.append(
            action_link(
                compare_href(prefix, version_id, version_id, file_path),
                f"Open {version_id} in code compare",
            )
        )
    else:
        implemented = [item for item in versions if item["status"] == "implemented"]
        if len(implemented) >= 2:
            left = implemented[-2]["id"]
            right = implemented[-1]["id"]
            links.append(
                action_link(
                    compare_href(prefix, left, right, "layer/gqa.py"),
                    "Compare latest implemented versions",
                )
            )

    links.append(action_link(f"{prefix}/compare/", "Open compare tool", "secondary-link"))
    return '<div class="doc-actions">' + "".join(links) + "</div>"


def page_shell(
    *,
    title: str,
    body: str,
    versions: list[dict[str, str]],
    current: str | None = None,
    extra_class: str = "",
    extra_head: str = "",
    extra_body: str = "",
) -> str:
    nav_items = []
    for version in versions:
        href = f"versions/{version['id']}/"
        if current and current != "index":
            href = f"../{version['id']}/"
        if current == "compare":
            href = f"../versions/{version['id']}/"
        aria = ' aria-current="page"' if current == version["id"] else ""
        nav_items.append(
            f'<a class="version-link" href="{href}"{aria}>'
            f"{version['id']}<span class=\"status\">{version['status']}</span></a>"
        )

    root_prefix = "../.." if current and current not in {"index", "compare"} else ".."
    if current == "index":
        root_prefix = "."

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - Torchlet Docs</title>
  <link rel="stylesheet" href="{root_prefix}/assets/site.css">
  {extra_head}
</head>
<body>
  <header class="site-header">
    <a class="brand" href="{root_prefix}/">
      <span class="brand-mark">Torchlet</span>
      <span class="brand-subtitle">LLM inference walkthrough</span>
    </a>
    <nav class="top-nav" aria-label="Top navigation">
      <a href="{root_prefix}/">Guide</a>
      <a href="{root_prefix}/compare/">Compare code</a>
    </nav>
  </header>
  <div class="layout">
    <aside class="sidebar">
      <p class="sidebar-title">Versions</p>
      <nav class="version-nav" aria-label="Version navigation">
        {''.join(nav_items)}
      </nav>
    </aside>
    <main class="content">
      <article class="doc {extra_class}">
        {body}
      </article>
    </main>
  </div>
  {extra_body}
</body>
</html>
"""


def build_index(out_dir: Path, versions: list[dict[str, str]]) -> None:
    content = render_markdown((CONTENT / "index.md").read_text())
    cards = []
    for version in versions:
        cards.append(
            f'<a class="version-card" href="versions/{version["id"]}/">'
            f'<span class="badge {version["status"]}">{version["status"]}</span>'
            f"<h3>{version['id']}</h3>"
            f"<p>{html.escape(version['theme'])}</p>"
            "</a>"
        )
    intro_actions = (
        '<div class="doc-actions">'
        + action_link("versions/v00_full_recompute/", "Start walkthrough")
        + action_link("compare/", "Open code compare", "secondary-link")
        + "</div>"
    )
    body = (
        content
        + intro_actions
        + '<p class="lede">Each step keeps the previous implementation visible, '
        + "so the codebase can be read as a sequence of small design pressures.</p>"
        + '<section class="version-grid">'
        + "".join(cards)
        + "</section>"
    )
    (out_dir / "index.html").write_text(
        page_shell(title="Guide", body=body, versions=versions, current="index")
    )


def build_version_pages(out_dir: Path, versions: list[dict[str, str]]) -> None:
    for index, version in enumerate(versions):
        source = VERSION_CONTENT / f"{version['id']}.md"
        body = render_markdown(source.read_text())
        actions = version_actions(
            version,
            versions[index - 1] if index > 0 else None,
            versions,
        )
        body = body.replace("</h1>", f"</h1>{actions}", 1)
        body = (
            f'<span class="badge {version["status"]}">{version["status"]}</span>'
            + body
        )
        version_dir = out_dir / "versions" / version["id"]
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "index.html").write_text(
            page_shell(
                title=f"{version['id']} {version['title']}",
                body=body,
                versions=versions,
                current=version["id"],
            )
        )


def collect_code() -> dict[str, object]:
    version_dirs = sorted(
        path for path in SRC.iterdir() if path.is_dir() and path.name.startswith("v")
    )
    versions: list[dict[str, str]] = []
    all_files: set[str] = set()
    code: dict[str, dict[str, str]] = {}

    for version_dir in version_dirs:
        versions.append({"id": version_dir.name})
        code[version_dir.name] = {}
        for path in sorted(version_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(version_dir).as_posix()
            all_files.add(rel)
            code[version_dir.name][rel] = path.read_text()

    return {
        "versions": versions,
        "files": sorted(all_files),
        "code": code,
    }


def build_compare(out_dir: Path, versions: list[dict[str, str]]) -> None:
    compare_dir = out_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    body = """
<h1>Compare Code</h1>
<p class="lede">Choose two implemented versions and one file path. The viewer renders a split diff so adjacent design changes are easy to inspect.</p>
<div class="compare-toolbar">
  <label>Left version<select id="leftVersion"></select></label>
  <label>Right version<select id="rightVersion"></select></label>
  <label>File<select id="filePath"></select></label>
</div>
<section class="diff-panel">
  <div class="diff-panel-header">
    <div>
      <h2 id="leftTitle"></h2>
      <span id="leftMeta"></span>
    </div>
    <div class="diff-summary" id="diffSummary"></div>
    <div>
      <h2 id="rightTitle"></h2>
      <span id="rightMeta"></span>
    </div>
  </div>
  <div class="diff-scroll" id="diffView"></div>
</section>
"""
    (compare_dir / "index.html").write_text(
        page_shell(
            title="Compare Code",
            body=body,
            versions=versions,
            current="compare",
            extra_class="compare-page",
            extra_body=(
                '<script src="../data/code.js"></script>'
                '<script src="../assets/compare.js"></script>'
            ),
        )
    )


def build(out_dir: Path) -> None:
    versions = read_versions()

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    shutil.copytree(ASSETS, out_dir / "assets")

    data_dir = out_dir / "data"
    data_dir.mkdir()
    code_payload = collect_code()
    (data_dir / "code.json").write_text(json.dumps(code_payload))
    (data_dir / "code.js").write_text(
        "window.TORCHLET_CODE = " + json.dumps(code_payload) + ";\n"
    )

    build_index(out_dir, versions)
    build_version_pages(out_dir, versions)
    build_compare(out_dir, versions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Torchlet docs site.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DOCS / "_site",
        help="Output directory for the generated static site.",
    )
    args = parser.parse_args()
    build(args.out)


if __name__ == "__main__":
    main()

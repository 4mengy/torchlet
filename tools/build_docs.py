from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CONTENT = DOCS / "content"
ASSETS = DOCS / "assets"
VERSION_CONTENT = CONTENT / "versions"
ZH_CONTENT = CONTENT / "zh-CN"
ZH_VERSION_CONTENT = ZH_CONTENT / "versions"
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
    "v09_triton_basics": "layer/rope.py",
    "v10_triton_paged_gqa": "layer/gqa.py",
}

LOCALES = {
    "en": {
        "html_lang": "en",
        "docs_title": "Torchlet Docs",
        "content_dir": CONTENT,
        "version_content": VERSION_CONTENT,
        "why_heading": "Why Introduce It",
        "switch_label": "中文",
        "switch_lang": "zh-CN",
        "status_implemented": "implemented",
        "status_planned": "planned",
        "versions": "Versions",
        "version_navigation": "Version navigation",
        "files": "Files",
        "file_filter": "File filter",
        "changed": "Changed",
        "all": "All",
        "file_navigation": "File navigation",
        "on_this_page": "On this page",
        "brand_subtitle": "LLM inference implementation notes",
        "top_navigation": "Top navigation",
        "overview": "Overview",
        "compare_code": "Compare code",
        "compare_latest": "Compare latest implemented Versions",
        "compare_pair": "Compare {base} -> {target}",
        "open_compare": "Open compare tool",
        "open_version_notes": "Open version notes",
        "implementation_versions": "Implementation Versions",
        "version_route_lede": (
            "Each Version keeps the preceding implementation visible, so design "
            "pressures and structural changes can be examined directly."
        ),
        "guide_title": "Guide",
        "version_comparison": "Version comparison",
        "compare_changes": "Compare implementation changes",
        "compare_lede": (
            "Inspect the directed code changes from a Base Version to a later "
            "Target Version."
        ),
        "adjacent_comparisons": "Adjacent comparisons",
        "previous_pair": "Previous pair",
        "next_pair": "Next pair",
        "comparison_controls": "Comparison controls",
        "base_version": "Base Version",
        "target_version": "Target Version",
        "diff_view": "Diff view",
        "split": "Split",
        "unified": "Unified",
        "evolution_summary": "Evolution summary",
        "previous_change": "Previous change",
        "next_change": "Next change",
        "no_changes": "No changes",
        "expand_all": "Expand all",
        "compare_title": "Compare Code",
    },
    "zh": {
        "html_lang": "zh-CN",
        "docs_title": "Torchlet 文档",
        "content_dir": ZH_CONTENT,
        "version_content": ZH_VERSION_CONTENT,
        "why_heading": "为什么引入",
        "switch_label": "English",
        "switch_lang": "en",
        "status_implemented": "已实现",
        "status_planned": "计划中",
        "versions": "版本",
        "version_navigation": "版本导航",
        "files": "文件",
        "file_filter": "文件筛选",
        "changed": "有变更",
        "all": "全部",
        "file_navigation": "文件导航",
        "on_this_page": "本页目录",
        "brand_subtitle": "LLM 推理实现说明",
        "top_navigation": "顶部导航",
        "overview": "概览",
        "compare_code": "代码对比",
        "compare_latest": "对比最近两个已实现版本",
        "compare_pair": "对比 {base} → {target}",
        "open_compare": "打开代码对比工具",
        "open_version_notes": "打开版本说明",
        "implementation_versions": "实现版本",
        "version_route_lede": "每个版本都保留前一阶段的实现，便于直接观察设计压力和结构变化。",
        "guide_title": "指南",
        "version_comparison": "版本对比",
        "compare_changes": "对比实现变化",
        "compare_lede": "查看从基础版本到后续目标版本的代码变化。",
        "adjacent_comparisons": "相邻版本对比",
        "previous_pair": "上一组",
        "next_pair": "下一组",
        "comparison_controls": "对比控制",
        "base_version": "基础版本",
        "target_version": "目标版本",
        "diff_view": "差异视图",
        "split": "并排",
        "unified": "统一",
        "evolution_summary": "演进摘要",
        "previous_change": "上一处变更",
        "next_change": "下一处变更",
        "no_changes": "无变更",
        "expand_all": "展开全部",
        "compare_title": "代码对比",
    },
}

COMPARE_MESSAGES = {
    "zh": {
        "dataUnavailable": "数据不可用",
        "comparisonDataDidNotLoad": "未能加载对比数据。",
        "evolutionSummary": "演进摘要",
        "openVersionNotes": "打开版本说明",
        "stagesInComparison": "本次对比跨越 {count} 个实现阶段",
        "openTargetVersionNotes": "打开目标版本说明",
        "statusAdded": "新增",
        "statusDeleted": "删除",
        "statusModified": "修改",
        "statusUnchanged": "未变更",
        "root": "根目录",
        "fileSummary": "{changed} 个变更 · 共 {total} 个",
        "noCodeChanges": "这两个版本之间没有代码变化。",
        "expandUnchanged": "展开 {count} 行未变更内容",
        "noChangedFiles": "没有变更文件",
        "changesCount": "{count} 处变更",
        "sameCode": "这两个版本包含相同的代码。",
        "codeHorizontalScroll": "代码横向滚动",
        "changePosition": "第 {current} / {total} 处变更",
        "noChanges": "无变更",
    }
}


def slug_to_title(slug: str) -> str:
    return slug.replace("_", " ")


def render_inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        lambda match: (
            f'<a href="{html.escape(match.group(2), quote=True)}">{match.group(1)}</a>'
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
                    "<pre><code>" + html.escape("\n".join(code_lines)) + "</code></pre>"
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


def read_versions(locale: str = "en") -> list[dict[str, str]]:
    with (CONTENT / "versions.json").open() as file:
        versions = json.load(file)
    if locale == "en":
        return versions

    with (ZH_CONTENT / "versions.json").open() as file:
        translations = {item["id"]: item for item in json.load(file)}
    version_ids = {item["id"] for item in versions}
    if set(translations) != version_ids:
        raise ValueError(
            "Chinese Version metadata must match docs/content/versions.json"
        )
    return [{**item, **translations[item["id"]]} for item in versions]


def read_version_rationale(
    version_id: str,
    version_content: Path = VERSION_CONTENT,
    why_heading: str = "Why Introduce It",
) -> str:
    markdown = (version_content / f"{version_id}.md").read_text()
    match = re.search(
        rf"^## {re.escape(why_heading)}\s*$\n(.*?)(?=^## |\Z)",
        markdown,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        return ""
    paragraphs = [part.strip() for part in match.group(1).split("\n\n") if part.strip()]
    if not paragraphs:
        return ""
    rationale = " ".join(paragraphs[0].splitlines())
    rationale = re.sub(r"`([^`]+)`", r"\1", rationale)
    rationale = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", rationale)
    return rationale


def build_asset_version(code_payload: dict[str, object]) -> str:
    digest = hashlib.sha256()
    digest.update((ASSETS / "site.css").read_bytes())
    digest.update((ASSETS / "compare-core.js").read_bytes())
    digest.update((ASSETS / "compare.js").read_bytes())
    digest.update(json.dumps(code_payload, sort_keys=True).encode())
    return digest.hexdigest()[:12]


def versioned(path: str, asset_version: str) -> str:
    return f"{path}?v={asset_version}" if asset_version else path


def render_google_tag(measurement_id: str | None) -> str:
    if not measurement_id:
        return ""
    if not re.fullmatch(r"G-[A-Z0-9]+", measurement_id):
        raise ValueError("GA Measurement ID must match G-[A-Z0-9]+")
    return f"""<!-- Google tag (gtag.js) -->
  <script async src="https://www.googletagmanager.com/gtag/js?id={measurement_id}"></script>
  <script>
    window.dataLayer = window.dataLayer || [];
    function gtag(){{dataLayer.push(arguments);}}
    gtag("js", new Date());
    gtag("config", "{measurement_id}");
  </script>"""


def compare_href(prefix: str, base: str, target: str, file_path: str) -> str:
    return (
        f"{prefix}/compare/?base={quote(base)}&target={quote(target)}"
        f"&file={quote(file_path, safe='')}"
    )


def action_link(href: str, label: str, class_name: str = "action-link") -> str:
    return (
        f'<a class="{class_name}" href="{html.escape(href, quote=True)}">'
        f"{html.escape(label)}</a>"
    )


def status_label(status: str, text: dict[str, str]) -> str:
    return text[f"status_{status}"]


def version_actions(
    version: dict[str, str],
    versions: list[dict[str, str]],
    text: dict[str, str],
) -> str:
    links = []
    prefix = "../.."
    implemented = [item for item in versions if item["status"] == "implemented"]

    if version["status"] == "implemented" and version in implemented:
        implemented_index = implemented.index(version)
        if implemented_index > 0:
            base = implemented[implemented_index - 1]
            target = version
        elif len(implemented) > 1:
            base = version
            target = implemented[1]
        else:
            base = None
            target = None
        if base and target:
            file_path = DEFAULT_COMPARE_FILES.get(target["id"], "layer/gqa.py")
            links.append(
                action_link(
                    compare_href(prefix, base["id"], target["id"], file_path),
                    text["compare_pair"].format(base=base["id"], target=target["id"]),
                )
            )
    elif len(implemented) >= 2:
        base = implemented[-2]
        target = implemented[-1]
        file_path = DEFAULT_COMPARE_FILES.get(target["id"], "layer/gqa.py")
        links.append(
            action_link(
                compare_href(prefix, base["id"], target["id"], file_path),
                text["compare_latest"],
            )
        )

    links.append(
        action_link(f"{prefix}/compare/", text["open_compare"], "secondary-link")
    )
    return '<div class="doc-actions">' + "".join(links) + "</div>"


def page_shell(
    *,
    title: str,
    body: str,
    versions: list[dict[str, str]],
    locale: str,
    text: dict[str, str],
    current: str | None = None,
    extra_class: str = "",
    extra_head: str = "",
    extra_body: str = "",
    asset_version: str = "",
    table_of_contents: str = "",
) -> str:
    nav_items = []
    for version in versions:
        href = f"versions/{version['id']}/"
        if current and current != "index":
            href = f"../{version['id']}/"
        if current == "compare":
            href = f"../versions/{version['id']}/"
        aria = ' aria-current="page"' if current == version["id"] else ""
        version_label = f"{version['id']} · {version['title']}"
        nav_items.append(
            f'<a class="version-link" href="{href}" '
            f'title="{html.escape(version_label, quote=True)}"{aria}>'
            f'<span class="version-link-id">{version["id"]}</span>'
            f'<span class="version-link-title">{html.escape(version["title"])}</span>'
            f'<span class="status {version["status"]}">'
            f"{html.escape(status_label(version['status'], text))}</span>"
            "</a>"
        )

    root_prefix = "../.." if current and current not in {"index", "compare"} else ".."
    if current == "index":
        root_prefix = "."

    if current == "index":
        page_suffix = ""
    elif current == "compare":
        page_suffix = "compare/"
    else:
        page_suffix = f"versions/{current}/"
    if locale == "en":
        language_href = f"{root_prefix}/zh/{page_suffix}"
    else:
        language_href = f"{root_prefix}/../{page_suffix}"

    version_navigation = f"""
    <aside class="version-sidebar">
      <p class="sidebar-title">{text["versions"]}</p>
      <nav class="version-nav" aria-label="{text["version_navigation"]}">
        {"".join(nav_items)}
      </nav>
    </aside>"""

    if current == "index":
        page_class = "page-home"
        page_layout = f"""
  <div class="readme-layout">
    <main class="content">
      <article class="doc home-doc">{body}</article>
    </main>
  </div>"""
    elif current == "compare":
        page_class = "page-compare"
        page_layout = f"""
  <div class="compare-layout">
    <aside class="file-sidebar">
      <div class="file-sidebar-header">
        <div>
          <p class="sidebar-title">{text["files"]}</p>
          <span id="fileSummary" class="file-summary"></span>
        </div>
        <div class="file-filter" aria-label="{text["file_filter"]}">
          <button id="changedFilesOnly" type="button" aria-pressed="true">{text["changed"]}</button>
          <button id="allFiles" type="button" aria-pressed="false">{text["all"]}</button>
        </div>
      </div>
      <nav id="fileNav" aria-label="{text["file_navigation"]}"></nav>
    </aside>
    <main class="compare-main">
      <article class="doc {extra_class}">{body}</article>
    </main>
  </div>"""
    else:
        page_class = "page-version"
        page_layout = f"""
  <div class="docs-layout">
    {version_navigation}
    <main class="content">
      <article class="doc {extra_class}">{body}</article>
    </main>
    <aside class="toc-sidebar">
      <p class="sidebar-title">{text["on_this_page"]}</p>
      <nav class="toc-nav" aria-label="{text["on_this_page"]}">{table_of_contents}</nav>
    </aside>
  </div>"""

    return f"""<!doctype html>
<html lang="{text["html_lang"]}">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} - {text["docs_title"]}</title>
  <link rel="alternate" hreflang="{text["switch_lang"]}" href="{language_href}">
  <link rel="stylesheet"
        href="{versioned(f"{root_prefix}/assets/site.css", asset_version)}">
  {text.get("analytics_head", "")}
  {extra_head}
</head>
<body class="{page_class}">
  <header class="site-header">
    <a class="brand" href="{root_prefix}/">
      <span class="brand-mark">Torchlet</span>
      <span class="brand-subtitle">{text["brand_subtitle"]}</span>
    </a>
    <nav class="top-nav" aria-label="{text["top_navigation"]}">
      <a href="{root_prefix}/">{text["overview"]}</a>
      <a href="{root_prefix}/compare/">{text["compare_code"]}</a>
      <a id="languageSwitch" lang="{text["switch_lang"]}" hreflang="{text["switch_lang"]}"
         href="{language_href}">{text["switch_label"]}</a>
    </nav>
  </header>
  {page_layout}
  {extra_body}
  <script>
    const languageSwitch = document.querySelector("#languageSwitch");
    if (languageSwitch && window.location.search) {{
      const target = new URL(languageSwitch.href, window.location.href);
      target.search = window.location.search;
      languageSwitch.href = target.href;
    }}
  </script>
</body>
</html>
"""


def add_heading_anchors(body: str) -> tuple[str, str]:
    headings: list[tuple[str, str]] = []
    used_ids: dict[str, int] = {}

    def replace_heading(match: re.Match[str]) -> str:
        rendered_title = match.group(1)
        plain_title = html.unescape(re.sub(r"<[^>]+>", "", rendered_title))
        base_id = re.sub(r"[^a-z0-9]+", "-", plain_title.lower()).strip("-")
        base_id = base_id or "section"
        occurrence = used_ids.get(base_id, 0)
        used_ids[base_id] = occurrence + 1
        heading_id = base_id if occurrence == 0 else f"{base_id}-{occurrence + 1}"
        headings.append((heading_id, plain_title))
        return f'<h2 id="{heading_id}">{rendered_title}</h2>'

    anchored = re.sub(r"<h2>(.*?)</h2>", replace_heading, body)
    links = "".join(
        f'<a href="#{heading_id}">{html.escape(title)}</a>'
        for heading_id, title in headings
    )
    return anchored, links


def build_index(
    out_dir: Path,
    versions: list[dict[str, str]],
    asset_version: str,
    locale: str,
    text: dict[str, str],
    content_dir: Path,
) -> None:
    content = render_markdown((content_dir / "index.md").read_text())
    cards = []
    for version in versions:
        cards.append(
            f'<a class="version-card" href="versions/{version["id"]}/">'
            f'<span class="badge {version["status"]}">'
            f"{html.escape(status_label(version['status'], text))}</span>"
            f"<h3>{version['id']}</h3>"
            f"<p>{html.escape(version['theme'])}</p>"
            "</a>"
        )
    intro_actions = (
        '<div class="doc-actions">'
        + action_link("versions/v00_full_recompute/", text["open_version_notes"])
        + action_link("compare/", text["open_compare"], "secondary-link")
        + "</div>"
    )
    first_paragraph_end = content.find("</p>") + len("</p>")
    intro = content[:first_paragraph_end]
    details = content[first_paragraph_end:]
    body = (
        intro
        + intro_actions
        + f'<section class="version-route"><h2>{text["implementation_versions"]}</h2>'
        + f'<p class="lede">{text["version_route_lede"]}</p>'
        + '<section class="version-grid">'
        + "".join(cards)
        + "</section>"
        + "</section>"
        + details
    )
    (out_dir / "index.html").write_text(
        page_shell(
            title=text["guide_title"],
            body=body,
            versions=versions,
            locale=locale,
            text=text,
            current="index",
            asset_version=asset_version,
        )
    )


def build_version_pages(
    out_dir: Path,
    versions: list[dict[str, str]],
    asset_version: str,
    locale: str,
    text: dict[str, str],
    version_content: Path,
) -> None:
    for version in versions:
        source = version_content / f"{version['id']}.md"
        body = render_markdown(source.read_text())
        body, table_of_contents = add_heading_anchors(body)
        actions = version_actions(version, versions, text)
        body = body.replace("</h1>", f"</h1>{actions}", 1)
        body = (
            f'<span class="badge {version["status"]}">'
            f"{html.escape(status_label(version['status'], text))}</span>" + body
        )
        version_dir = out_dir / "versions" / version["id"]
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "index.html").write_text(
            page_shell(
                title=f"{version['id']} {version['title']}",
                body=body,
                versions=versions,
                locale=locale,
                text=text,
                current=version["id"],
                asset_version=asset_version,
                table_of_contents=table_of_contents,
            )
        )


def collect_code(
    version_metadata: list[dict[str, str]],
    version_content: Path = VERSION_CONTENT,
    why_heading: str = "Why Introduce It",
) -> dict[str, object]:
    implemented_versions: list[dict[str, str]] = []
    all_files: set[str] = set()
    code: dict[str, dict[str, str]] = {}

    for version in version_metadata:
        version_dir = SRC / version["id"]
        if version["status"] != "implemented" or not version_dir.is_dir():
            continue
        implemented_versions.append(
            {
                **version,
                "rationale": read_version_rationale(
                    version["id"], version_content, why_heading
                ),
                "important_file": DEFAULT_COMPARE_FILES.get(
                    version["id"], "layer/gqa.py"
                ),
            }
        )
        code[version["id"]] = {}
        for path in sorted(version_dir.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            rel = path.relative_to(version_dir).as_posix()
            all_files.add(rel)
            code[version["id"]][rel] = path.read_text()

    return {
        "versions": implemented_versions,
        "files": sorted(all_files),
        "code": code,
    }


def build_compare(
    out_dir: Path,
    versions: list[dict[str, str]],
    asset_version: str,
    locale: str,
    text: dict[str, str],
) -> None:
    compare_dir = out_dir / "compare"
    compare_dir.mkdir(parents=True, exist_ok=True)
    body = f"""
<header class="compare-heading">
  <div>
    <p class="eyebrow">{text["version_comparison"]}</p>
    <h1>{text["compare_changes"]}</h1>
    <p class="lede">{text["compare_lede"]}</p>
  </div>
  <div class="pair-navigation" aria-label="{text["adjacent_comparisons"]}">
    <button id="previousPair" type="button">{text["previous_pair"]}</button>
    <button id="nextPair" type="button">{text["next_pair"]}</button>
  </div>
</header>
<section class="compare-toolbar" aria-label="{text["comparison_controls"]}">
  <label>{text["base_version"]}<select id="baseVersion"></select></label>
  <span class="direction-arrow" aria-hidden="true">→</span>
  <label>{text["target_version"]}<select id="targetVersion"></select></label>
  <div class="view-toggle" aria-label="{text["diff_view"]}">
    <button id="splitView" type="button" aria-pressed="true">{text["split"]}</button>
    <button id="unifiedView" type="button" aria-pressed="false">{text["unified"]}</button>
  </div>
</section>
<section class="evolution-summary" id="evolutionSummary" aria-label="{text["evolution_summary"]}"></section>
<section class="diff-panel">
  <div class="diff-panel-header diff-toolbar">
    <div class="current-file">
      <span id="fileStatus" class="file-status"></span>
      <h2 id="filePathTitle"></h2>
    </div>
    <div class="diff-summary" id="diffSummary"></div>
    <div class="change-navigation">
      <button id="previousChange" type="button" aria-label="{text["previous_change"]}">↑</button>
      <span id="changePosition">{text["no_changes"]}</span>
      <button id="nextChange" type="button" aria-label="{text["next_change"]}">↓</button>
      <button id="expandAll" type="button">{text["expand_all"]}</button>
    </div>
  </div>
  <div class="diff-scroll" id="diffView"></div>
</section>
"""
    (compare_dir / "index.html").write_text(
        page_shell(
            title=text["compare_title"],
            body=body,
            versions=versions,
            locale=locale,
            text=text,
            current="compare",
            extra_class="compare-page",
            extra_body=(
                f'<script src="{versioned("../data/code.js", asset_version)}"></script>'
                "<script>window.TORCHLET_I18N = "
                + json.dumps(COMPARE_MESSAGES.get(locale, {}), ensure_ascii=False)
                + ";</script>"
                f'<script src="{versioned("../assets/compare-core.js", asset_version)}">'
                "</script>"
                f'<script src="{versioned("../assets/compare.js", asset_version)}">'
                "</script>"
            ),
            asset_version=asset_version,
        )
    )


def build(out_dir: Path, ga_measurement_id: str | None = None) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    build_locale(out_dir, "en", ga_measurement_id)
    build_locale(out_dir / "zh", "zh", ga_measurement_id)


def build_locale(
    out_dir: Path, locale: str, ga_measurement_id: str | None = None
) -> None:
    text = {
        **LOCALES[locale],
        "analytics_head": render_google_tag(ga_measurement_id),
    }
    versions = read_versions(locale)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ASSETS, out_dir / "assets")

    data_dir = out_dir / "data"
    data_dir.mkdir()
    code_payload = collect_code(
        versions,
        text["version_content"],
        text["why_heading"],
    )
    asset_version = build_asset_version(code_payload)
    (data_dir / "code.json").write_text(json.dumps(code_payload))
    (data_dir / "code.js").write_text(
        "window.TORCHLET_CODE = " + json.dumps(code_payload) + ";\n"
    )

    build_index(
        out_dir,
        versions,
        asset_version,
        locale,
        text,
        text["content_dir"],
    )
    build_version_pages(
        out_dir,
        versions,
        asset_version,
        locale,
        text,
        text["version_content"],
    )
    build_compare(out_dir, versions, asset_version, locale, text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Torchlet docs site.")
    parser.add_argument(
        "--out",
        type=Path,
        default=DOCS / "_site",
        help="Output directory for the generated static site.",
    )
    parser.add_argument(
        "--ga-measurement-id",
        default=os.environ.get("GA_MEASUREMENT_ID"),
        help="Optional GA4 Measurement ID, for example G-XXXXXXXXXX.",
    )
    args = parser.parse_args()
    build(args.out, args.ga_measurement_id)


if __name__ == "__main__":
    main()

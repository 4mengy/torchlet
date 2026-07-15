from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_build_docs_module():
    spec = importlib.util.spec_from_file_location(
        "build_docs", ROOT / "tools" / "build_docs.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DocsBuildTest(unittest.TestCase):
    def test_generated_pages_use_page_specific_documentation_layouts(self):
        build_docs = load_build_docs_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "site"
            build_docs.build(out_dir)

            index_html = (out_dir / "index.html").read_text()
            version_html = (
                out_dir / "versions" / "v02_kv_cache" / "index.html"
            ).read_text()
            compare_html = (out_dir / "compare" / "index.html").read_text()
            css = (out_dir / "assets" / "site.css").read_text()

        self.assertIn('class="readme-layout"', index_html)
        self.assertNotIn('aria-label="Version navigation"', index_html)
        self.assertIn('aria-label="Version navigation"', version_html)
        self.assertIn('aria-label="On this page"', version_html)
        self.assertIn('aria-label="File navigation"', compare_html)
        self.assertNotIn('aria-label="Version navigation"', compare_html)
        self.assertIn("--canvas-default: #f6f8fa;", css)
        self.assertIn("--accent-fg: #0969da;", css)

    def test_generated_site_uses_neutral_implementation_language(self):
        build_docs = load_build_docs_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "site"
            build_docs.build(out_dir)
            rendered_pages = "\n".join(
                path.read_text().lower() for path in out_dir.rglob("*.html")
            )

        for tutorial_term in ("walkthrough", "teaching", "teaches", "learns"):
            self.assertNotIn(tutorial_term, rendered_pages)
        self.assertIn("implementation notes", rendered_pages)
        self.assertIn("design pressure", rendered_pages)

    def test_compare_page_exposes_directed_workspace_and_version_metadata(self):
        build_docs = load_build_docs_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "site"
            build_docs.build(out_dir)
            compare_html = (out_dir / "compare" / "index.html").read_text()
            first_version_html = (
                out_dir / "versions" / "v00_full_recompute" / "index.html"
            ).read_text()
            payload = json.loads((out_dir / "data" / "code.json").read_text())

        self.assertIn('id="baseVersion"', compare_html)
        self.assertIn('id="targetVersion"', compare_html)
        self.assertIn('id="evolutionSummary"', compare_html)
        self.assertIn('id="changedFilesOnly"', compare_html)
        self.assertIn('id="splitView"', compare_html)
        self.assertIn('id="unifiedView"', compare_html)
        self.assertLess(
            compare_html.index("compare-core.js"), compare_html.index("compare.js")
        )
        self.assertIn(
            "?base=v00_full_recompute&amp;target=v01_0_ragged_batch",
            first_version_html,
        )
        self.assertEqual(payload["versions"][0]["title"], "Full Recompute")
        self.assertEqual(payload["versions"][0]["status"], "implemented")
        self.assertEqual(payload["versions"][1]["important_file"], "layer/gqa.py")
        self.assertIn("padding", payload["versions"][1]["rationale"].lower())
        self.assertIn("v09_triton_basics", payload["code"])
        self.assertIn("kernel/rope.py", payload["code"]["v09_triton_basics"])
        v09_metadata = next(
            item for item in payload["versions"] if item["id"] == "v09_triton_basics"
        )
        self.assertEqual(v09_metadata["important_file"], "layer/rope.py")

    def test_generated_site_provides_linked_english_and_chinese_pages(self):
        build_docs = load_build_docs_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "site"
            build_docs.build(out_dir)
            english_index = (out_dir / "index.html").read_text()
            chinese_index = (out_dir / "zh" / "index.html").read_text()
            english_version = (
                out_dir / "versions" / "v09_triton_basics" / "index.html"
            ).read_text()
            chinese_version = (
                out_dir / "zh" / "versions" / "v09_triton_basics" / "index.html"
            ).read_text()
            chinese_compare = (out_dir / "zh" / "compare" / "index.html").read_text()
            english_payload = json.loads((out_dir / "data" / "code.json").read_text())
            chinese_payload = json.loads(
                (out_dir / "zh" / "data" / "code.json").read_text()
            )

        self.assertIn('<html lang="en">', english_index)
        self.assertIn('<html lang="zh-CN">', chinese_index)
        self.assertIn('href="./zh/"', english_index)
        self.assertIn('href="./../"', chinese_index)
        self.assertIn('href="../../zh/versions/v09_triton_basics/"', english_version)
        self.assertIn('href="../../../versions/v09_triton_basics/"', chinese_version)
        self.assertIn("对比实现变化", chinese_compare)
        self.assertIn("window.TORCHLET_I18N", chinese_compare)
        self.assertIn("没有变更文件", chinese_compare)
        self.assertLess(
            chinese_compare.index("window.TORCHLET_I18N"),
            chinese_compare.index("compare.js"),
        )
        self.assertEqual(
            [item["id"] for item in english_payload["versions"]],
            [item["id"] for item in chinese_payload["versions"]],
        )
        self.assertEqual(chinese_payload["versions"][0]["title"], "完整重算")

    def test_readmes_link_to_their_language_counterparts(self):
        english_readme = (ROOT / "README.md").read_text()
        chinese_readme = (ROOT / "README.zh-CN.md").read_text()

        self.assertIn("[简体中文](README.zh-CN.md)", english_readme)
        self.assertIn("[English](README.md)", chinese_readme)

    def test_google_analytics_is_optional_and_applies_to_both_locales(self):
        build_docs = load_build_docs_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            untracked_dir = temp_root / "untracked"
            tracked_dir = temp_root / "tracked"
            build_docs.build(untracked_dir)
            build_docs.build(tracked_dir, "G-TEST123456")

            untracked_pages = [
                path.read_text() for path in untracked_dir.rglob("*.html")
            ]
            tracked_pages = [path.read_text() for path in tracked_dir.rglob("*.html")]

        self.assertTrue(untracked_pages)
        self.assertTrue(tracked_pages)
        self.assertTrue(
            all("googletagmanager.com/gtag/js" not in page for page in untracked_pages)
        )
        for page in tracked_pages:
            self.assertIn(
                "https://www.googletagmanager.com/gtag/js?id=G-TEST123456", page
            )
            self.assertIn('gtag("config", "G-TEST123456")', page)

        with self.assertRaisesRegex(ValueError, "GA Measurement ID"):
            build_docs.render_google_tag("not-a-measurement-id")

    def test_split_diff_keeps_gutters_fixed_while_code_scrolls(self):
        build_docs = load_build_docs_module()

        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "site"
            build_docs.build(out_dir)
            css = (out_dir / "assets" / "site.css").read_text()
            compare_js = (out_dir / "assets" / "compare.js").read_text()

        line_number_rule = css.split(".diff-line-no {", 1)[1].split("}", 1)[0]
        self.assertNotIn("position: sticky", line_number_rule)
        split_table_rule = css.split(".diff-table.diff-split {", 1)[1].split("}", 1)[0]
        self.assertIn("table-layout: fixed", split_table_rule)
        self.assertIn("setupSplitHorizontalScroll", compare_js)
        self.assertIn("diff-code-clip", compare_js)


if __name__ == "__main__":
    unittest.main()

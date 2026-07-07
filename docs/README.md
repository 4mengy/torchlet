# Torchlet Docs Architecture

The documentation site is a static walkthrough for Torchlet's versioned inference
implementations. It is designed to work on GitHub Pages without Node, Ruby, or
third-party Python packages.

## Structure

- `docs/content/index.md`: landing page copy.
- `docs/content/versions.json`: ordered version metadata used for navigation.
- `docs/content/versions/*.md`: one teaching page per roadmap version.
- `docs/assets/site.css`: the documentation visual system.
- `docs/assets/compare.js`: browser-side version/file selector for code compare.
- `tools/build_docs.py`: static site generator.
- `.github/workflows/docs.yml`: GitHub Pages deployment workflow.

## Version Page Contract

Each version page should answer the same questions:

- What did this version introduce?
- Why was it introduced after the previous version?
- What core principle does it teach?
- Which files should readers compare?
- What tradeoff is intentionally left visible?

This repeated structure is the main teaching device. It helps readers compare
conceptual changes, not just code changes.

## Code Compare Page

The build script scans `src/torchlet/v*/**/*.py`, stores the source in
`docs/_site/data/code.js`, and creates a static compare page at
`docs/_site/compare/index.html`.

The page is intentionally simple: it aligns one file path across two selected
versions. If a file exists in only one version, the missing side says so.

## Build Locally

```bash
python3 tools/build_docs.py
```

Then open `docs/_site/index.html` in a browser.

## Deploy On GitHub Pages

The included workflow builds the docs on pushes to `main` and deploys
`docs/_site` through GitHub Pages.

In the repository settings, set Pages to use GitHub Actions as the source.

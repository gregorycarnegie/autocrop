# Repository Guidelines

## Project Structure & Module Organization

- `main.py` is the GUI entry point. Core logic and pipelines live in `core/`, PyQt6 widgets and tabs in `ui/`, and custom text inputs in `line_edits/`. File type detection and signatures sit in `file_types/`, while shared assets (icons, styles) are under `resources/`. Rust-accelerated image ops are in `src/`, with build outputs landing in `build/`, `dist/`, or `target/`. Keep example media in `examples/` and test-only files small.

## Build, Test, and Development Commands

- Create a virtual env and activate it: `python -m venv .venv` then `.\.venv\Scripts\activate` (or `source .venv/bin/activate`).
- Install tooling and build the Rust extension: `pip install maturin` followed by `maturin develop` (rerun after Rust edits).
- Run the app locally: `python main.py`.
- Python lint/format: `ruff check .` and `ruff format .`.
- Type check: `pyright`.
- Rust quality gates: `cargo fmt`, `cargo clippy`, and `cargo test`.
- Package wheel (optional): `maturin build --release` (artifacts in `target/`).

## Coding Style & Naming Conventions

- Python: 4-space indents, 120-char lines (ruff). Use snake_case for functions/vars, PascalCase for classes, and UPPER_SNAKE_CASE for constants. Prefer type hints and concise docstrings for public APIs. Keep UI strings in `ui/`; push business logic into `core/` to ease testing.
- Rust: follow rustfmt defaults; prefer the `?` operator and avoid `unwrap` in library code. Keep unsafe blocks minimal and well-commented.

## Testing Guidelines

- Current coverage leans on linting and type checks plus Rust unit tests. Add focused Rust tests when touching `src/*.rs`; run `cargo test`.
- For Python changes, add minimal repro scripts in `examples/` or small fixtures alongside the affected module. Manually exercise key tabs (Photo, Folder, Mapping, Video) after changes to catch GUI regressions or performance slowdowns.

## Commit & Pull Request Guidelines

- Commit messages are short and imperative, matching history such as `cargo fmt`, `Update pyproject.toml`, or `removed unused imports`.
- PRs should include a brief summary, manual test notes (commands/tabs exercised), linked issues when relevant, and screenshots or GIFs for UI-affecting changes. Keep commits focused; avoid mixing formatting-only changes with feature work.

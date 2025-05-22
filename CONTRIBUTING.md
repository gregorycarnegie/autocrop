# Contributing

All contributions, bug reports, bug fixes, documentation improvements, enhancements, and ideas are welcome.

Please follow these steps:

* Fork this repository to your personal GitHub account and clone it locally
* Install the development setup (see section below)
* Branch off of `master` for every change you want to make
* Develop changes on your branch
* Test your changes (see section below)
* Modify the tests and documentation as necessary
* When your changes are ready, make a pull request to this repository

## Development Setup

This project works with [virtualenv](https://virtualenv.pypa.io/en/latest/) and requires Rust for building performance-critical components.

### Prerequisites

* Python 3.13+ (tested versions)
* Rust toolchain (install via [rustup.rs](https://rustup.rs/))

### Initial Setup

To start things off, run:

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

Then, install the build tools and dependencies:

```bash
pip install maturin
```

Build the Rust extension and install the package in development mode:

```bash
maturin develop
```

You can then run `autocrop` like so:

```bash
python main.py
```

As long as the virtual environment has been activated, this command will use the files in your local Git checkout. This makes it super easy to work on the code and test your changes.

To set up your virtual environment again in future, just run:

```bash
source env/bin/activate  # On Windows: env\Scripts\activate
```

### Rust Development

If you're modifying the Rust components in the project:

1. Make your changes to the Rust code
2. Rebuild the extension:

   ```bash
   maturin develop
   ```

3. Test your changes in the Python application

## Tests

Pull requests are tested using continuous integration (CI) which will green-light changes.

Specifically, we:

* Use [ruff](https://docs.astral.sh/ruff/) for Python code linting and formatting
* Run type checking with [pyright](https://github.com/microsoft/pyright)
* Test Rust components with `cargo test`
* Run integration tests with the GUI components

You can run the linting checks locally:

```bash
ruff check .
ruff format .
```

For Rust components:

```bash
cargo test
cargo clippy
```

## Project Structure

* `main.py` - Entry point for the application
* `core/` - Core business logic and processing
* `ui/` - PyQt6 user interface components
* `line_edits/` - Custom input widgets
* `file_types/` - File type detection and validation
* `src/` - Rust source code for performance-critical operations
* `resources/` - Application resources (icons, stylesheets, etc.)

## Code Style

* Python code should follow PEP 8 (enforced by ruff)
* Rust code should follow standard Rust conventions (enforced by clippy)
* Use type hints in Python where appropriate
* Write docstrings for public functions and classes
* Keep functions focused and reasonably sized

## Submitting Changes

1. Ensure your code passes all linting checks
2. Test your changes thoroughly with the GUI
3. Update documentation if you're adding new features
4. Write clear commit messages
5. Create a pull request with a description of what your changes do

## Contact

If you have any questions, please create an issue on GitHub or reach out to the maintainers.

# Autocrop changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Complete GUI rewrite using PyQt6
- Video processing capabilities for extracting and cropping frames
- Mapping-based batch processing with CSV/Excel support
- Real-time preview of crop results
- Rust-accelerated image processing with AVX2 optimizations
- Support for RAW image formats
- Multiple aspect ratio presets (Square, Golden Ratio, 2:3, 3:4, 4:5)
- Advanced face detection with confidence thresholds
- Multi-face detection and cropping
- Automatic tilt correction for better face alignment
- Exposure correction for under/over-exposed images
- Drag-and-drop file support
- Browser-style navigation interface
- Progress indicators with cancellation support

### Changed

- Completely redesigned architecture from command-line to GUI application
- Migrated from OpenCV Haar cascades to YuNet face detection model
- Replaced dlib shape predictor with OpenCV FacemarkLBF
- Enhanced file type validation with binary signature checking
- Improved error handling and user feedback
- Modern multi-threaded processing for better performance

### Technical Improvements

- Rust integration for CPU-intensive operations
- Memory-efficient image processing pipeline
- Enhanced security with path validation
- Thread-safe operations for GUI responsiveness
- Comprehensive type hints and documentation

---

## Legacy Versions (from original autocrop)

The following versions are from the original [autocrop library](https://github.com/leblancfg/autocrop) by François Leblanc:

### [1.3.1] - 2022-02-08

#### API Additions

- The `Cropper` class now accepts a `resize` arg, to determine whether to resize the image after cropping it or not.
- Similarly for the CLI, it can now be called with a `--no-resize` flag

#### Other changes

- The order of CLI args when calling `autocrop --help` changed to place flags first
- Start using type hints across the codebase

### [1.3.0] - 2022-01-25

#### Changes

- The `initial-setup` step in Makefile now also installs development packages.
- The `black` formatting package now gets installed with `requirements-dev.txt`

#### Deprecations

- Deprecate Python 3.6

### [1.2.0] - 2021-11-26

#### Changes

- Modify the `opencv-python` dependency over to `opencv-python-headless`, which doesn't have all the GUI baggage. Easier to download and package.

#### Documentation

- Created the `examples` directory and moved the example notebook to it.

### [1.1.1] - 2021-02-17

#### Deprecations

- Deprecate Python 3.5
- Deprecate OpenCV 3

#### API Additions

- User can now specify what file extension to save cropped files at the CLI

#### Security

- Update Pillow dependency in order to limit possible security issues

#### Other changes

- Updates to the developer setup tools and documentation
- Updates to the example notebook

### [1.1.0] - 2020-10-24

#### Added

- CLI now copies file by default

### [1.0.0] - 2020-03-24

#### Added

- Cropper class now available from Python API.
- Local multi-version testing for Python now available with `tox`.
- Extra regressions tests to defend against image warp and cropping outside the regions of interest.
- Support for Python 3.8

#### Bugfixes

- Specify encoding in `setup.py`, which was causing some errors on Windows.

#### Deprecated

- Support for padding argument — this is now solely handled by the `face_percent` parameter, and enforces the aspect ratio between `width` and `height`.
- Support for Python 2.7

### [0.3.2] - Earlier Release

#### Changes

- Autocrop now prints the filename of images where face detection failed
- Internal refactoring and more tests

### [0.3.1] - Earlier Release

#### Changes

- Add `-r`, `--reject` flag to specify directory where the images that autocrop *couldn't* find a face in are directed to.
- Instead of having the target files copied then cropped, they are instead cropped and saved to their respective target folder.

### [0.3.0] - Earlier Release

#### Changes

- Added support for padding (`padLeft`, etc.) in the CLI.

#### Bugfix

- Fixed warp on crop for `-w` and `-h` values

### [0.2.0] - Earlier Release

#### Changes

- Add `-o`, `--output` flag to specify directory where cropped images are to be dumped.
  - Error out if output folder set to current directory, i.e. `-o .`
  - If directory doesn't exist yet, create it.
  - If no face can be found in an image in batch, it is still copied over to `-o` folder.
  - If no output folder is added, ask for confirmation ([Y]/n), and destructively crop images in-place.

- Use `-i`, `--input` flags as synonyms for `-p` or `--path`: symmetrical in meaning to "output".
  - Is now standard nomenclature in documentation.
- `--input` or `--path` flag is now optional.
  - Standard behaviour without input folder is to non-recursively process all images in immediate folder, i.e. `-p .` as currently implemented.

#### Breaking Changes

- Removed all mentions of the hard-coded 'bkp' and 'crop' folders
- Calling autocrop without specifying an input path, i.e. `autocrop` does not look for the 'images' folder anymore.

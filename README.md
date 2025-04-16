# Autocrop

[![CI](https://github.com/leblancfg/autocrop/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/leblancfg/autocrop/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/leblancfg/autocrop/branch/master/graph/badge.svg)](https://codecov.io/gh/leblancfg/autocrop) [![Documentation](https://img.shields.io/badge/docs-passing-success.svg)](https://leblancfg.com/autocrop) [![PyPI version](https://badge.fury.io/py/autocrop.svg)](https://badge.fury.io/py/autocrop) [![Downloads](https://pepy.tech/badge/autocrop)](https://pepy.tech/project/autocrop)

<div align="center">
   <img title="obama_crop" src="https://cloud.githubusercontent.com/assets/15659410/10975709/3e38de48-83b6-11e5-8885-d95da758ca17.png">
</div>

`autocrop` is a user-friendly GUI application built with PyQt6 designed to automatically crop images and videos based on detected faces. It intelligently centers the output on the largest face found (or optionally detects multiple faces), making it ideal for processing profile pictures, creating ID card photos, or batch-processing large sets of images. Core calculations are accelerated using Rust with AVX2 optimizations for improved performance.

## Features

* **Graphical User Interface:** Easy-to-use interface built with PyQt6.
* **Multiple Cropping Modes**:
    * **Photo Crop:** Crop individual image files.
    * **Folder Crop:** Batch crop all supported images within a selected folder.
    * **Mapping Crop:** Batch crop images based on a mapping provided in a CSV or Excel file.
    * **Video Crop:** Extract and crop frames from video files.
* **Flexible Face Detection:**
    * Detects the largest face by default.
    * Option to detect and crop all faces found above a confidence threshold.
* **Image Adjustments**:
    * **Auto-Align:** Automatically correct head tilt.
    * **Exposure Correction:** Fix under/over-exposed images.
    * **Gamma Adjustment:** Fine-tune image brightness/contrast.
* **Customizable Output:**
    * Set specific output dimensions (width, height).
    * Apply aspect ratio presets (Square, Golden Ratio, 2:3, 3:4, 4:5).
    * Adjust margins around the detected face.
    * Choose output format (including keeping original, or converting to JPG, PNG, TIFF, BMP, WEBP).
* **Wide File Support**: Supports numerous image formats, RAW files (via `rawpy`), and video files (via `ffmpeg-python`).
* **Performance:** CPU-intensive calculations (like cropping coordinate calculation, gamma/exposure adjustments, rotation matrix generation) are implemented in Rust for significant speed improvements. Uses AVX2 instructions when available.
* **Live Preview:** See a preview of the crop result directly in the UI as you adjust settings.


## Screenshots

### Graphical User Interface

<div align="center">
  <img title="app_preview" src="https://github.com/gregorycarnegie/autocrop/blob/master/examples/app.jpg?raw=true" alt="App Preview">
</div>

### Cropped Images

<div align="center">
  <img title="face_crop_0" src="https://github.com/gregorycarnegie/autocrop/blob/master/examples/original_0.jpg?raw=true" alt="Face Crop 0">
</div>

<div align="center">
  <img title="face_crop_1" src="https://github.com/gregorycarnegie/autocrop/blob/master/examples/original_1.jpg?raw=true" alt="Face Crop 1">
</div>

<a href="https://stocksnap.io/photo/business-people-H6PSN9BPGZ">Photo</a> by <a href="https://stocksnap.io/author/directmedia">Direct Media</a> on <a href="https://stocksnap.io">StockSnap</a>

## Installation

**Prerequisites:**

* Python (Tested on 3.13+)
* Rust programming language toolchain (Install via [rustup.rs](https://rustup.rs/))

**Install from Source (Recommended for development)**

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/autocrop.git # Or the original repo
   cd autocrop
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

3. Install build tools and dependencies:
   ```bash
   uv pip install -e ".[dev]" # Install with development dependencies
   ```

4. Build the Rust extension and install in editable mode:
   ```bash
   maturin develop
   ```

## Usage

1. Activate your virtual environment if you installed from source.
2. Run the application:
   ```bash
   python main.py
   ```
3. Select a Tab based on your task:
   - **Photo Crop**: For single image files.
   - **Folder Crop**: For batch processing an entire folder.
   - **Mapping Crop**: For batch processing based on an Excel/CSV map.
   - **Video Crop**: For cropping video frames.
4. Select Input: Choose your input file, folder, or folder/table combination.
5. Select Destination: Choose where to save the cropped files.
6. Adjust Settings:
   - Set desired Width and Height.
   - Use Presets menu for common aspect ratios.
   - Check boxes for Exposure Fix, Multi-Face, Tilt Correction.
   - Use dials to adjust Sensitivity, Face %, Gamma, and Padding.
   - Select the desired Output Format.
7. Crop: Click the Crop button! Use the Cancel button to stop processing.

## Supported file types

The following file types are supported:

- EPS files (`.eps`)
- JPEG 2000 files (`.j2k`, `.j2p`, `.jp2`, `.jpx`)
- JPEG files (`.jfif`, `.jpeg`, `.jpg`, `.jpe`)
- LabEye IM files (`.im`)
- macOS ICNS files (`.icns`)
- Microsoft Paint bitmap files (`.msp`)
- PCX files (`.pcx`)
- Portable Network Graphics (`.png`)
- Portable Pixmap files (`.pbm`, `.pgm`, `.ppm`)
- SGI files (`.sgi`)
- SPIDER files (`.spi`)
- TGA files (`.tga`)
- TIFF files (`.tif`, `.tiff`)
- WebP (`.webp`)
- Windows bitmap files (`.bmp`, `.dib`)
- Windows ICO files (`.ico`)
- X bitmap files (`.xbm`)
- RAW files (`.dng`, `.arw`, `.cr2`, `.crw`, `.erf`, `.kdc`, `.nef`, `.nrw`, `.orf`, `.pef`, `.raf`, `.raw`, `.sr2`, `.srw`, `.x3f`)
- Video files (`.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`)

## Dependencies

Key dependencies include:

- PyQt6
- OpenCV (opencv-contrib-python)
- NumPy
- Polars (for mapping)
- fastexcel (for mapping)
- Rawpy (for RAW files)
- ffmpeg-python (for video)
- Maturin (for Rust build)
- psutil
- tifffile
- cachetools

## License

This project is licensed under the MIT License - see the LICENSE file for details.

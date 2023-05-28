# autocrop

[![CI](https://github.com/leblancfg/autocrop/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/leblancfg/autocrop/actions/workflows/ci.yml) [![codecov](https://codecov.io/gh/leblancfg/autocrop/branch/master/graph/badge.svg)](https://codecov.io/gh/leblancfg/autocrop) [![Documentation](https://img.shields.io/badge/docs-passing-success.svg)](https://leblancfg.com/autocrop) [![PyPI version](https://badge.fury.io/py/autocrop.svg)](https://badge.fury.io/py/autocrop) [![Downloads](https://pepy.tech/badge/autocrop)](https://pepy.tech/project/autocrop)

<p align="center"><img title="obama_crop" src="https://cloud.githubusercontent.com/assets/15659410/10975709/3e38de48-83b6-11e5-8885-d95da758ca17.png"></p>

Perfect for profile picture processing for your website or batch work for ID cards, autocrop will output images centered around the biggest face detected.

# Supported file types

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
- Video files (`.avi`, `.m4v`, `.mkv`, `.mov`, `.mp4`, `.wmv`)

# Misc
### Installing directly


Autocrop is [currently being tested on](https://github.com/leblancfg/autocrop/actions/workflows/ci.yml):

* Python 3.7 to 3.11
* OS:
    - Linux
    - macOS
    - Windows

# More Info
Check out:

* http://docs.opencv.org/master/d7/d8b/tutorial_py_face_detection.html#gsc.tab=0
* http://docs.opencv.org/master/d5/daf/tutorial_py_histogram_equalization.html#gsc.tab=0

Adapted from:

* http://photo.stackexchange.com/questions/60411/how-can-i-batch-crop-based-on-face-location

### Contributing

Although autocrop is essentially a CLI wrapper around a single OpenCV function, it is actively developed. It has active users throughout the world.

If you would like to contribute, please consult the [contribution docs](CONTRIBUTING.md).

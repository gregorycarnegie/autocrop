// src/face_detection.rs
use pyo3::prelude::*;

use crate::ImportablePyModuleBuilder;

#[pyclass]
#[derive(Debug, Clone)]
pub struct Rectangle {
    #[pyo3(get, set)]
    pub left: i32,
    #[pyo3(get, set)]
    pub top: i32,
    #[pyo3(get, set)]
    pub right: i32,
    #[pyo3(get, set)]
    pub bottom: i32,
    #[pyo3(get, set)]
    pub confidence: f64,
}

#[pymethods]
impl Rectangle {
    #[new]
    fn new(left: i32, top: i32, right: i32, bottom: i32, confidence: f64) -> PyResult<Self> {
        if left >= right || top >= bottom {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid rectangle coordinates",
            ));
        }
        if !(0.0..=1.0).contains(&confidence) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "confidence must be in [0, 1]",
            ));
        }

        Ok(Rectangle {
            left,
            top,
            right,
            bottom,
            confidence,
        })
    }

    #[getter]
    fn width(&self) -> i32 {
        self.right - self.left
    }

    #[getter]
    fn height(&self) -> i32 {
        self.bottom - self.top
    }

    #[getter]
    fn area(&self) -> i32 {
        self.width() * self.height()
    }

    /// Python: rect * n
    fn __mul__(&self, scale_factor: f64) -> (f64, f64, f64, f64) {
        // use your existing Rust Mul impl
        if scale_factor > 1.0 {
            (
                self.left as f64 * scale_factor,
                self.top as f64 * scale_factor,
                self.width() as f64 * scale_factor,
                self.height() as f64 * scale_factor,
            )
        } else {
            (
                self.left as f64,
                self.top as f64,
                self.width() as f64,
                self.height() as f64,
            )
        }
    }

    /// Python: n * rect
    fn __rmul__(&self, scale_factor: f64) -> (f64, f64, f64, f64) {
        self.__mul__(scale_factor)
    }
}

/// Fast scale factor determination for face detection optimization
#[pyfunction]
fn determine_scale_factor(width: i32, height: i32, face_scale_divisor: f64) -> PyResult<f64> {
    if face_scale_divisor <= 0.0 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "face_scale_divisor must be > 0",
        ));
    }
    let min_dim = std::cmp::min(width, height) as f64;
    // division in f64, then ensure at least 1.0
    Ok((min_dim / face_scale_divisor).max(1.0))
}

/// Find the face with highest confidence from a list of detections
#[pyfunction]
fn find_best_face(faces: Vec<Rectangle>) -> Option<Rectangle> {
    faces.into_iter().max_by(|a, b| {
        a.confidence
            .partial_cmp(&b.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let builder = ImportablePyModuleBuilder::from(m.clone())?;

    builder
        .add_class::<Rectangle>()?
        .add_function(wrap_pyfunction!(determine_scale_factor, m)?)?
        .add_function(wrap_pyfunction!(find_best_face, m)?)?;

    Ok(())
}

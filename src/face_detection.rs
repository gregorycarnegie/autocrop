// src/face_detection.rs
use pyo3::prelude::*;
use std::cmp::Ordering;

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
            return Err(pyo3::exceptions::PyValueError::new_err("Invalid rectangle coordinates"));
        }
        if !(0.0..=1.0).contains(&confidence) {
            return Err(pyo3::exceptions::PyValueError::new_err("confidence must be in [0, 1]"));
        }
        
        Ok(Rectangle { left, top, right, bottom, confidence })
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
}

/// Fast scale factor determination for face detection optimization
#[pyfunction]
fn determine_scale_factor(width: i32, height: i32, face_scale_divisor: i32) -> i32 {
    std::cmp::max(1, std::cmp::min(width, height) / face_scale_divisor)
}

/// Scale face coordinates based on scale factor used during detection
#[pyfunction]
fn scale_face_coordinates(face: &Rectangle, scale_factor: i32) -> (i32, i32, i32, i32) {
    if scale_factor > 1 {
        (
            face.left * scale_factor,
            face.top * scale_factor,
            face.width() * scale_factor,
            face.height() * scale_factor,
        )
    } else {
        (face.left, face.top, face.width(), face.height())
    }
}

/// Find the face with highest confidence from a list of detections
#[pyfunction]
fn find_best_face(faces: Vec<Rectangle>) -> Option<Rectangle> {
    faces.into_iter()
        .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap_or(Ordering::Equal))
}

pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let builder = ImportablePyModuleBuilder::from(m.clone())?;
    
    builder
        .add_class::<Rectangle>()?
        .add_function(wrap_pyfunction!(determine_scale_factor, m)?)?
        .add_function(wrap_pyfunction!(scale_face_coordinates, m)?)?
        .add_function(wrap_pyfunction!(find_best_face, m)?)?;
    
    Ok(())
}
// src/lib.rs
use pyo3::prelude::*;

mod dispatch_simd;
mod face_detection;
mod file_signatures;
mod file_types;
mod image_processing;
mod module_builder;
mod security;

use module_builder::ImportablePyModuleBuilder;

#[pymodule]
fn autocrop_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create submodules
    let file_types_module = ImportablePyModuleBuilder::new(py, "autocrop_rs.file_types")?;
    file_types::register_module(&file_types_module.as_module())?;
    let file_types = file_types_module.finish();

    let security_module = ImportablePyModuleBuilder::new(py, "autocrop_rs.security")?;
    security::register_module(&security_module.as_module())?;
    let security = security_module.finish();

    let image_processing_module =
        ImportablePyModuleBuilder::new(py, "autocrop_rs.image_processing")?;
    image_processing::register_module(&image_processing_module.as_module())?;
    let image_processing = image_processing_module.finish();

    let face_detection_module = ImportablePyModuleBuilder::new(py, "autocrop_rs.face_detection")?;
    face_detection::register_module(&face_detection_module.as_module())?;
    let face_detection = face_detection_module.finish();

    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_submodule(&file_types)?
        .add_submodule(&security)?
        .add_submodule(&image_processing)?
        .add_submodule(&face_detection)?
        .finish();

    Ok(())
}

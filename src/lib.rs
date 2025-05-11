// src/lib.rs
use pyo3::prelude::*;

mod file_types;
mod security;
mod image_processing;
mod dispatch_simd;
mod file_signatures;
mod module_builder;

use module_builder::ImportablePyModuleBuilder;

#[pymodule]
fn autocrop_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create submodules
    let file_types_module = ImportablePyModuleBuilder::new(py, "autocrop_rs.file_types")?
        .finish();
    file_types::file_types(&file_types_module)?;
    
    let security_module = ImportablePyModuleBuilder::new(py, "autocrop_rs.security")?
        .finish();
    security::security(&security_module)?;
    
    let image_processing_module = ImportablePyModuleBuilder::new(py, "autocrop_rs.image_processing")?
        .finish();
    image_processing::image_processing(&image_processing_module)?;
    
    // Use the builder to properly add submodules to the main module
    ImportablePyModuleBuilder::from(m.clone())
        .add_submodule(&file_types_module)?
        .add_submodule(&security_module)?
        .add_submodule(&image_processing_module)?;
    
    Ok(())
}
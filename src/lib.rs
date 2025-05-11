use pyo3::prelude::*;

mod file_types;
mod dispatch_simd;
mod file_signatures;
mod security;
mod image_processing;

/// Module definition
#[pymodule]
fn autocrop_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    image_processing::register_module(m)?;
    file_types::register_module(m)?;
    security::register_module(m)?;

    Ok(())
}

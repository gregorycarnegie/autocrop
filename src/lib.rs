// src/lib.rs
use pyo3::prelude::*;

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
    macro_rules! register_submodule {
        ($module:ident) => {{
            let builder =
                ImportablePyModuleBuilder::new(py, concat!("autocrop_rs.", stringify!($module)))?;
            $module::register_module(builder.as_module())?;
            builder.finish()
        }};
    }

    let file_types = register_submodule!(file_types);
    let security = register_submodule!(security);
    let image_processing = register_submodule!(image_processing);
    let face_detection = register_submodule!(face_detection);

    // Add submodules to main module
    ImportablePyModuleBuilder::from(m.clone())?
        .add_submodule(&file_types)?
        .add_submodule(&security)?
        .add_submodule(&image_processing)?
        .add_submodule(&face_detection)?
        .finish();

    Ok(())
}

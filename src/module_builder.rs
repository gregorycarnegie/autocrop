// https://github.com/biomancy/biobit/blob/c09082b2d0071689d9462d763f932a34f2b0f722/modules/core/py/src/bindings/utils/importable_py_module.rs
use pyo3::prelude::*;
use pyo3::types::{PyList, PyNone};
use pyo3::{ffi, PyClass};
use std::ffi::CString;

pub struct ImportablePyModuleBuilder<'py> {
    inner: Bound<'py, PyModule>,
}

impl<'py> ImportablePyModuleBuilder<'py> {
    pub fn new(py: Python<'py>, name: &str) -> PyResult<Self> {
        let module = unsafe {
            let ptr = ffi::PyImport_AddModule(CString::new(name)?.as_ptr());
            let bound = Bound::from_borrowed_ptr_or_err(py, ptr)?;
            bound.downcast_into::<PyModule>()?
        };

        module.setattr("__file__", PyNone::get(py))?;
        Ok(Self { inner: module })
    }

    pub fn add_submodule(self, module: &Bound<'_, PyModule>) -> PyResult<Self> {
        // Extract the string from the PyString object
        let fully_qualified_name: String = module.name()?.extract()?;
        let name = fully_qualified_name.rsplit('.').next()
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyImportError, _>(
                format!("Can't extract module name from {}", fully_qualified_name)
            ))?;
        
        self.inner.add(name, module)?;
        
        if !self.inner.hasattr("__path__")? {
            self.inner.setattr("__path__", PyList::empty(self.inner.py()))?;
        }
        Ok(self)
    }

    pub fn add_class<T: PyClass>(self) -> PyResult<Self> {
        self.inner.add_class::<T>()?;
        
        let type_object = T::lazy_type_object().get_or_init(self.inner.py());
        let __module__ = type_object.getattr("__module__")?.extract::<String>()?;
        
        if __module__ == "builtins" {
            type_object.setattr("__module__", self.inner.name()?)?;
        }
        Ok(self)
    }

    pub fn from(module: Bound<'py, PyModule>) -> Self {
        Self { inner: module }
    }

    pub fn finish(self) -> Bound<'py, PyModule> {
        self.inner
    }
}
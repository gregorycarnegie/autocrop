// File validation additions to lib.rs
use memmap2::MmapOptions;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};

use crate::dispatch_simd::compare_buffers;
use crate::file_signatures::{
    Signature, PHOTO_SIGNATURES_MAP, PNG_SIG, RAW_SIGNATURES_MAP, TABLE_SIGNATURES_MAP,
    TIFF_SIGNATURES_MAP, VIDEO_SIGNATURES_MAP,
};
use crate::ImportablePyModuleBuilder;

// File category enum matching Python's FileCategory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FileCategory {
    Photo = 0,
    Raw = 1,
    Tiff = 2,
    Video = 3,
    Table = 4,
    Unknown = 5,
}

fn get_signatures(path: &Path, category: FileCategory) -> Option<&'static [Signature]> {
    // Skip unknown category early
    if category == FileCategory::Unknown {
        return None;
    }

    // Get the extension in lowercase without leading dot
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    // Select the appropriate map based on category
    let map = match category {
        FileCategory::Photo => &PHOTO_SIGNATURES_MAP,
        FileCategory::Raw => &RAW_SIGNATURES_MAP,
        FileCategory::Tiff => &TIFF_SIGNATURES_MAP,
        FileCategory::Video => &VIDEO_SIGNATURES_MAP,
        FileCategory::Table => &TABLE_SIGNATURES_MAP,
        FileCategory::Unknown => return None,
    };

    // Look up signatures in the selected map
    map.get(extension.as_str()).copied()
}

// Helper function to validate a CSV file
fn validate_csv(path: &Path) -> bool {
    match File::open(path) {
        Ok(file) => {
            let mut reader = BufReader::with_capacity(2048, file);
            let mut buffer = String::with_capacity(1024);

            // Read a sample to detect dialect
            if reader.read_line(&mut buffer).is_err() || buffer.is_empty() {
                return false;
            }

            // More robust delimiter detection - allow any common delimiter
            let potential_delimiters = [',', '\t', ';', '|', ' '];

            // Check if ANY delimiter appears in the line
            for &delimiter in &potential_delimiters {
                if buffer.contains(delimiter) {
                    return true; // Found a delimiter, consider it valid
                }
            }

            // If no common delimiter is found but the line contains alphanumeric characters,
            // it might still be a simple single-column CSV
            if buffer.chars().any(|c| c.is_alphanumeric()) {
                return true;
            }

            false
        }
        Err(_) => false,
    }
}

/// SIMD accelerated signature checking using AVX2 or SSE2 instructions
/// Returns true if any signature matches
#[inline]
fn check_file_signatures(path: &Path, signatures: &[Signature]) -> bool {
    // Early return for empty signatures
    if signatures.is_empty() {
        return false;
    }

    // Calculate minimum read size needed
    let max_offset_plus_len = signatures
        .iter()
        .map(|(sig, offset)| offset + sig.len())
        .max()
        .unwrap_or(0);

    // OPTIMIZATION: Use memory-mapped I/O instead of reading the whole file
    if let Ok(file) = File::open(path) {
        if let Ok(mmap) = unsafe { MmapOptions::new().map(&file) } {
            // Check if the file is large enough
            if mmap.len() >= max_offset_plus_len {
                // Check each signature
                return signatures
                    .iter()
                    .any(|(sig, offset)| compare_buffers(&mmap, sig, *offset));
            }
        }
    }

    false // File couldn't be opened or read
}

/// Fast check for PNG file signature (8 bytes)
#[inline]
fn is_png_file(path: &Path) -> bool {
    if let Ok(mut file) = File::open(path) {
        let mut signature = [0u8; 8];
        if let Ok(bytes_read) = file.read(&mut signature) {
            if bytes_read == 8 {
                return compare_buffers(&signature, PNG_SIG, 0);
            }
        }
    }
    false
}

/// Fast check for JPEG file signature
#[inline]
fn is_jpeg_file(path: &Path) -> bool {
    if let Ok(mut file) = File::open(path) {
        let mut signature = [0u8; 3];
        if let Ok(bytes_read) = file.read(&mut signature) {
            if bytes_read == 3 {
                // JPEG has a 3-byte signature (using our new helper)
                return compare_buffers(&signature, &[0xFF, 0xD8, 0xFF], 0);
            }
        }
    }
    false
}

/// Optimized file validation with special handling for common formats
#[inline]
pub fn validate_file(path: &Path, category: FileCategory) -> bool {
    // Check if file exists and is readable
    if !path.exists() || !path.is_file() {
        return false;
    }

    // Get extension for fast path decisions
    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();

    // Fast path for common image formats
    match (&extension[..], category) {
        ("png", FileCategory::Photo) => return is_png_file(path),
        ("jpg" | "jpeg", FileCategory::Photo) => return is_jpeg_file(path),
        ("csv", FileCategory::Table) => return validate_csv(path),
        _ => {} // Continue with standard validation
    }

    // Get signatures for the file category and extension
    if let Some(signatures) = get_signatures(path, category) {
        return check_file_signatures(path, signatures);
    }

    false // No matching signatures
}

/// A faster version of validate_files that uses the optimized functions
#[pyfunction]
pub fn validate_files<'py>(
    py: Python<'py>,
    file_paths: Vec<String>,
    categories: Vec<u8>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    // Validate input lengths
    if file_paths.len() != categories.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "file_paths and categories must have the same length",
        ));
    }

    // Compute results in parallel and collect into a new vector
    let par_results: Vec<_> = file_paths
        .into_par_iter()
        .zip(categories.into_par_iter())
        .map(|(path, category)| {
            let file_category = match category {
                0 => FileCategory::Photo,
                1 => FileCategory::Raw,
                2 => FileCategory::Tiff,
                3 => FileCategory::Video,
                4 => FileCategory::Table,
                _ => FileCategory::Unknown,
            };

            validate_file(&PathBuf::from(&path), file_category)
        })
        .collect();

    // Convert to numpy array
    let array = Array1::from_vec(par_results);
    Ok(array.into_pyarray(py))
}

/// Validates a file by verifying its contents match its claimed category.
///
/// Args:
///     file_path: Path to the file to validate
///     category: Category code (0=Photo, 1=Raw, 2=Tiff, 3=Video, 4=Table, 5=Unknown)
///
/// Returns:
///     Boolean indicating if the file is valid for the specified category
#[pyfunction]
pub fn verify_file_type(file_path: String, category: u8) -> PyResult<bool> {
    // OPTIMIZATION: Use &Path instead of PathBuf where possible to avoid allocation
    let path = Path::new(&file_path);
    let file_category = match category {
        0 => FileCategory::Photo,
        1 => FileCategory::Raw,
        2 => FileCategory::Tiff,
        3 => FileCategory::Video,
        4 => FileCategory::Table,
        _ => FileCategory::Unknown,
    };

    // Special handling for table files
    if file_category == FileCategory::Table {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();

        // Enhanced validation for table files
        match extension.as_str() {
            "csv" => return Ok(validate_csv(path)),
            "xlsx" | "xlsm" | "xltx" | "xltm" => {
                // Excel files are ZIP files with specific contents
                // Just check if it's a valid ZIP file
                return Ok(is_zip_file(path));
            }
            "parquet" => {
                // Basic check for parquet files - they usually start with PAR1
                if let Ok(file) = File::open(path) {
                    if let Ok(mmap) = unsafe { MmapOptions::new().map(&file) } {
                        if mmap.len() >= 4 {
                            return Ok(&mmap[0..4] == [b'P', b'A', b'R', b'1']);
                        }
                    }
                }
                return Ok(false);
            }
            _ => {} // Fall through to standard validation
        }
    }

    // Standard validation for other file types
    Ok(validate_file(path, file_category))
}

// Helper function to check if a file is a valid ZIP file (for Excel)
fn is_zip_file(path: &Path) -> bool {
    if let Ok(file) = File::open(path) {
        if let Ok(mmap) = unsafe { MmapOptions::new().map(&file) } {
            if mmap.len() >= 4 {
                // Check for ZIP file signature "PK\x03\x04"
                return &mmap[0..4] == [0x50, 0x4B, 0x03, 0x04];
            }
        }
    }
    false
}

/// Register the functions with the Python module
// #[pymodule]
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let builder = ImportablePyModuleBuilder::from(m.clone())?;

    // Add functions to module
    builder
        .add_function(wrap_pyfunction!(validate_files, m)?)?
        .add_function(wrap_pyfunction!(verify_file_type, m)?)?;

    Ok(())
}

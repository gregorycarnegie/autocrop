// File validation additions to lib.rs
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::{Read, BufRead, BufReader};

use crate::ImportablePyModuleBuilder;
use crate::dispatch_simd::compare_buffers;
use crate::file_signatures::{PNG_SIG, RAW_SIGNATURES_MAP, PHOTO_SIGNATURES_MAP, TIFF_SIGNATURES_MAP, VIDEO_SIGNATURES_MAP, TABLE_SIGNATURES_MAP, Signature};

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
    let extension = path.extension()
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
            if reader.read_line(&mut buffer).is_err() {
                return false;
            }
            
            // More robust delimiter detection
            let potential_delimiters = [',', '\t', ';', '|'];
            let counts: Vec<_> = potential_delimiters.iter()
                .map(|&d| (d, buffer.chars().filter(|&c| c == d).count()))
                .collect();
                
            // Find most common delimiter with at least 1 occurrence
            if let Some(&(delimiter, count)) = counts.iter().max_by_key(|&&(_, c)| c) {
                if count > 0 {
                    // Validate consistency in next few lines
                    for _ in 0..2 {
                        buffer.clear();
                        if reader.read_line(&mut buffer).is_ok() && !buffer.is_empty() {
                            if buffer.chars().filter(|&c| c == delimiter).count() == 0 {
                                return false; // Inconsistent format
                            }
                        }
                    }
                    return true;
                }
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
    let max_offset_plus_len = signatures.iter()
        .map(|(sig, offset)| offset + sig.len())
        .max()
        .unwrap_or(0);
    
    // Small buffer for stack allocation
    const STACK_BUFFER_SIZE: usize = 64;
    
    // Open file with buffered IO
    if let Ok(mut file) = File::open(path) {
        // Use stack allocation for small reads, heap for larger
        if max_offset_plus_len <= STACK_BUFFER_SIZE {
            let mut buffer = [0u8; STACK_BUFFER_SIZE];
            
            if let Ok(bytes_read) = file.read(&mut buffer[..max_offset_plus_len]) {
                if bytes_read >= max_offset_plus_len {
                    // Check each signature
                    return signatures.iter().any(|(sig, offset)| {
                        compare_buffers(&buffer, sig, *offset)
                    });
                }
            }
        } else {
            // Use heap allocation for larger reads
            let mut buffer = vec![0u8; max_offset_plus_len];
            
            if let Ok(bytes_read) = file.read(&mut buffer) {
                if bytes_read >= max_offset_plus_len {
                    // Check each signature
                    return signatures.iter().any(|(sig, offset)| {
                        compare_buffers(&buffer, sig, *offset)
                    });
                }
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
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
        .unwrap_or_default();
        
    // Fast path for common image formats
    match (&extension[..], category) {
        ("png", FileCategory::Photo) => return is_png_file(path),
        ("jpg" | "jpeg", FileCategory::Photo) => return is_jpeg_file(path),
        ("csv", FileCategory::Table) => return validate_csv(path),
        _ => {}  // Continue with standard validation
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
            "file_paths and categories must have the same length"
        ));
    }
    
    let file_count = file_paths.len();
    
    // Create input tuples for processing
    let inputs: Vec<(usize, PathBuf, FileCategory)> = file_paths.into_iter()
        .zip(categories.into_iter().enumerate())
        .map(|(path, (idx, category))| {
            let file_category = match category {
                0 => FileCategory::Photo,
                1 => FileCategory::Raw,
                2 => FileCategory::Tiff,
                3 => FileCategory::Video,
                4 => FileCategory::Table,
                _ => FileCategory::Unknown,
            };
            
            (idx, PathBuf::from(path), file_category)
        })
        .collect();
    
    // Process in parallel using the optimized validator and collect results directly
    let results_with_indices: Vec<(usize, bool)> = inputs.into_par_iter()
        .map(|(idx, path, category)| {
            (idx, validate_file(&path, category))
        })
        .collect();
    
    // Initialize all results as false
    let mut results = vec![false; file_count];
    
    // Fill in the actual results based on the indices
    for (idx, valid) in results_with_indices {
        if idx < file_count {
            results[idx] = valid;
        }
    }
    
    // Convert to numpy array
    let array = Array1::from_vec(results);
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
    let path = PathBuf::from(file_path);
    let file_category = match category {
        0 => FileCategory::Photo,
        1 => FileCategory::Raw,
        2 => FileCategory::Tiff,
        3 => FileCategory::Video,
        4 => FileCategory::Table,
        _ => FileCategory::Unknown,
    };
    
    Ok(validate_file(&path, file_category))
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

// File validation additions to lib.rs
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::{Read, BufRead, BufReader};

use crate::ImportablePyModuleBuilder;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{
    __m128i, __m256i,
    _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8,
    _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8
};

use crate::dispatch_simd::dispatch_simd;
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
    const SAMPLE_LINES: usize = 3; // Check first few lines
    
    match File::open(path) {
        Ok(file) => {
            let mut reader = BufReader::with_capacity(2048, file);
            let mut buffer = String::with_capacity(1024);
            
            // Read the first few lines
            let mut line_count = 0;
            let mut delimiters = Vec::new();
            
            for _ in 0..SAMPLE_LINES {
                buffer.clear();
                match reader.read_line(&mut buffer) {
                    Ok(0) => break, // EOF
                    Ok(_) => {
                        if line_count == 0 {
                            // First line - determine possible delimiters
                            if buffer.contains(',') {
                                delimiters.push(',');
                            }
                            if buffer.contains('\t') {
                                delimiters.push('\t');
                            }
                            // Could add more potential delimiters here
                            
                            if delimiters.is_empty() {
                                return false; // No recognized delimiters
                            }
                        } else {
                            // Check that subsequent lines have the same structure
                            let mut has_delim = false;
                            for &delim in &delimiters {
                                if buffer.contains(delim) {
                                    has_delim = true;
                                    break;
                                }
                            }
                            if !has_delim {
                                return false; // Inconsistent format
                            }
                        }
                        line_count += 1;
                    },
                    Err(_) => return false,
                }
            }
            
            line_count > 0 && !delimiters.is_empty()
        },
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
                    // Use SIMD or scalar implementation based on architecture
                    return dispatch_simd(
                        (&buffer[..max_offset_plus_len], signatures),
                        |(buf, sigs)| unsafe { check_signatures_simd(buf, sigs) },
                        |(buf, sigs)| check_signatures_scalar(buf, sigs)
                    );
                }
            }
        } else {
            // Use heap allocation for larger reads
            let mut buffer = vec![0u8; max_offset_plus_len];
            
            if let Ok(bytes_read) = file.read(&mut buffer) {
                if bytes_read >= max_offset_plus_len {
                    // Use SIMD or scalar implementation based on architecture
                    return dispatch_simd(
                        (&buffer[..], signatures),
                        |(buf, sigs)| unsafe { check_signatures_simd(buf, sigs) },
                        |(buf, sigs)| check_signatures_scalar(buf, sigs)
                    );
                }
            }
        }
    }
    
    false // File couldn't be opened or read
}

/// Scalar implementation of signature checking
#[inline]
fn check_signatures_scalar(buffer: &[u8], signatures: &[Signature]) -> bool {
    signatures.iter().any(|(signature, offset)| {
        *offset + signature.len() <= buffer.len() &&
        &buffer[*offset..*offset + signature.len()] == *signature
    })
}

/// SIMD-accelerated signature checking using AVX2 when available
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn check_signatures_simd(buffer: &[u8], signatures: &[Signature]) -> bool {
    // For very short signatures (1-2 bytes), scalar code is faster
    if signatures.iter().all(|(sig, _)| sig.len() <= 2) {
        return check_signatures_scalar(buffer, signatures);
    }
    
    // Try each signature
    for (signature, offset) in signatures {
        let sig_len = signature.len();
        let buf_offset = *offset;
        
        // Ensure we have enough bytes
        if buf_offset + sig_len > buffer.len() {
            continue;
        }
        
        // Choose SIMD strategy based on signature length
        let matched = if sig_len >= 32 {
            // Use AVX2 for long signatures
            check_long_signature_avx2(buffer, buf_offset, signature)
        } else if sig_len >= 16 {
            // Use SSE2 for medium signatures
            check_medium_signature_sse2(buffer, buf_offset, signature)
        } else {
            // Use scalar comparison for short signatures
            &buffer[buf_offset..buf_offset + sig_len] == *signature
        };
        
        if matched {
            return true;
        }
    }
    
    false
}

/// Check a signature that's 32 bytes or longer using AVX2
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn check_long_signature_avx2(buffer: &[u8], offset: usize, signature: &[u8]) -> bool {
    let sig_len = signature.len();
    let chunks = sig_len / 32;
    let remainder = sig_len % 32;
    
    // Compare 32 bytes at a time
    for i in 0..chunks {
        let buf_ptr = buffer.as_ptr().add(offset + i * 32) as *const __m256i;
        let sig_ptr = signature.as_ptr().add(i * 32) as *const __m256i;
        
        let buf_chunk = _mm256_loadu_si256(buf_ptr);
        let sig_chunk = _mm256_loadu_si256(sig_ptr);
        
        // Compare equality (0xFF where equal, 0x00 where different)
        let comparison = _mm256_cmpeq_epi8(buf_chunk, sig_chunk);
        
        // Convert to bitmask (1 bit per byte)
        let mask = _mm256_movemask_epi8(comparison);
        
        // If all 32 bytes match, mask will be 0xFFFFFFFF
        // Fix: Use u32 explicitly
        if mask as u32 != 0xFFFF_FFFFu32 {
            return false;
        }
    }
    
    // Check remaining bytes if any
    if remainder > 0 {
        let start = chunks * 32;
        return &buffer[offset + start..offset + sig_len] == &signature[start..sig_len];
    }
    
    true
}

/// Check a signature that's 16-31 bytes long using SSE2
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn check_medium_signature_sse2(buffer: &[u8], offset: usize, signature: &[u8]) -> bool {
    let sig_len = signature.len();
    let chunks = sig_len / 16;
    let remainder = sig_len % 16;
    
    // Compare 16 bytes at a time
    for i in 0..chunks {
        let buf_ptr = buffer.as_ptr().add(offset + i * 16) as *const __m128i;
        let sig_ptr = signature.as_ptr().add(i * 16) as *const __m128i;
        
        let buf_chunk = _mm_loadu_si128(buf_ptr);
        let sig_chunk = _mm_loadu_si128(sig_ptr);
        
        // Compare equality (0xFF where equal, 0x00 where different)
        let comparison = _mm_cmpeq_epi8(buf_chunk, sig_chunk);
        
        // Convert to bitmask (1 bit per byte)
        let mask = _mm_movemask_epi8(comparison);
        
        // If all 16 bytes match, mask will be 0xFFFF
        // Fix: Use appropriate size and type
        if mask as u16 != 0xFFFFu16 {
            return false;
        }
    }
    
    // Check remaining bytes if any
    if remainder > 0 {
        let start = chunks * 16;
        return &buffer[offset + start..offset + sig_len] == &signature[start..sig_len];
    }
    
    true
}

/// Fast check for PNG file signature (8 bytes)
#[inline]
fn is_png_file(path: &Path) -> bool {
    if let Ok(mut file) = File::open(path) {
        let mut signature = [0u8; 8];
        if let Ok(bytes_read) = file.read(&mut signature) {
            if bytes_read == 8 {
                return dispatch_simd(
                    (&signature, PNG_SIG),
                    |(sig, png_sig)| unsafe { is_png_signature_simd(sig, png_sig) },
                    |(sig, png_sig)| sig == png_sig
                );
            }
        }
    }
    false
}

/// SIMD-optimized PNG signature check using 64-bit comparison
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
unsafe fn is_png_signature_simd(signature: &[u8; 8], png_sig: &[u8]) -> bool {
    // This is faster than AVX2 for just 8 bytes - use direct 64-bit comparison
    let sig_ptr = signature.as_ptr() as *const u64;
    let png_ptr = png_sig.as_ptr() as *const u64;
    
    std::ptr::read_unaligned(sig_ptr) == std::ptr::read_unaligned(png_ptr)
}

/// Fast check for JPEG file signature
#[inline]
fn is_jpeg_file(path: &Path) -> bool {
    if let Ok(mut file) = File::open(path) {
        let mut signature = [0u8; 3];
        if let Ok(bytes_read) = file.read(&mut signature) {
            if bytes_read == 3 {
                // JPEG has a 3-byte signature
                return signature == [0xFF, 0xD8, 0xFF];
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

// File validation additions to lib.rs
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use phf::{phf_map, Map};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::{Read, BufRead, BufReader};

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::{
    __m128i, __m256i,
    _mm_cmpeq_epi8, _mm_loadu_si128, _mm_movemask_epi8,
    _mm256_cmpeq_epi8, _mm256_loadu_si256, _mm256_movemask_epi8
};

use crate::dispatch_simd::dispatch_simd;

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

// Type aliases for better code readability
type Signature = (&'static [u8], usize);
type FileSignatureMap = Map<&'static str, &'static [Signature]>;

// Helper constants for common signature checks
const PNG_SIG: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

// Common file signatures as static byte slices
static JPG_SIGNATURES: &[Signature] = &[(&[0xFF, 0xD8, 0xFF], 0)];
static JP2_SIGNATURES: &[Signature] = &[(&[0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A], 0)];
static PNG_SIGNATURES: &[Signature] = &[(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], 0)];
static BMP_SIGNATURES: &[Signature] = &[(&[0x42, 0x4D], 0)];
static WEBP_SIGNATURES: &[Signature] = &[
    (&[0x52, 0x49, 0x46, 0x46], 0),  // "RIFF"
    (&[0x57, 0x45, 0x42, 0x50], 8)   // "WEBP"
];

// ASCII signatures can use byte literals
static PBM_SIGNATURES: &[Signature] = &[(&[b'P', b'4'], 0), (&[b'P', b'1'], 0)];
static PGM_SIGNATURES: &[Signature] = &[(&[b'P', b'5'], 0), (&[b'P', b'2'], 0)];
static PPM_SIGNATURES: &[Signature] = &[(&[b'P', b'6'], 0), (&[b'P', b'3'], 0)];

static PNM_SIGNATURES: &[Signature] = &[(&[b'P', b'7'], 0)];

// Portable FloatMap (32‑bit float HDR)
static PFM_SIGNATURES: &[Signature] = &[
    (&[b'P', b'F'], 0), // Colour
    (&[b'P', b'f'], 0)  // Greyscale
];

// Sun Raster signature
static SUN_RASTER_SIGNATURE: &[Signature] = &[(&[0x59, 0xA6, 0x6A, 0x95], 0)];

// RAW format signatures
static DNG_SIGNATURES: &[Signature] = &[(&[0x49, 0x49, 0x2A, 0x00], 0), (&[0x49, 0x49, 0x2A, 0x00], 0)]; // "II*\0" (little endian)
static CR2_SIGNATURE: &[Signature] = &[(&[0x49, 0x49, 0x2A, 0x00, 0x10, 0x00, 0x00, 0x00, 0x43, 0x52], 0)]; // "II*\0" (little endian)
static CRW_SIGNATURE: &[Signature] = &[(&[0x49, 0x49, 0x1A, 0x00, 0x00, 0x00, 0x48, 0x45, 0x41, 0x50, 0x43, 0x43, 0x44, 0x52, 0x02, 0x00], 0)]; // "II*\0" (little endian)
static EXR_SIGNATURE: &[Signature] = &[(&[0x76, 0x2F, 0x31, 0x01], 0)]; // "v/1\0"
static X3F_SIGNATURE: &[Signature] = &[(&[b'F', b'O', b'V', b'b'], 0)]; // "FOVb"
static ORF_SIGNATURES: &[Signature] = &[
    (&[b'I', b'I', b'R', b'O'], 0), // "IIR\0"
    (&[b'I', b'I'], 0)             // "II\0"
];
static TIFF_SIGNATURES: &[Signature] = &[
    (&[0x49, 0x49, 0x2A, 0x00], 0),  // "II*\0" (little endian)
    (&[0x4D, 0x4D, 0x00, 0x2A], 0)   // "MM\0*" (big endian)
];
static TIFF_LE_SIGNATURE: &[Signature] = &[(&[0x49, 0x49, 0x2A, 0x00], 0)];

// For longer signatures
static FUJI_RAF_SIGNATURE: &[Signature] = &[
    (&[b'F', b'U', b'J', b'I', b'F', b'I', b'L', b'M',
    b'C', b'C', b'D', b'-', b'R', b'A', b'W'], 0)
];

static RADIANCE_SIGNATURES: &[Signature] = &[
    (&[b'#', b'?', b'R', b'A', b'D', b'I', b'A', b'N', b'C', b'E'], 0), // HDR signature
    (&[b'#', b'?', b'R', b'G', b'B', b'E'], 0)                          // PIC signature
];

// Table format signatures
static ZIP_SIGNATURE: &[Signature] = &[(&[0x50, 0x4B, 0x03, 0x04], 0)]; // "PK\x03\x04"
static PARQUET_SIGNATURES: &[Signature] = &[
    (&[b'P', b'A', b'R', b'1'], 0), // "PAR1"
    (&[b'P', b'A', b'R', b'E'], 0)  // "PARE"
];

// Video format signatures
static AVI_SIGNATURES: &[Signature] = &[
    (&[b'R', b'I', b'F', b'F'], 0), // "RIFF"
    (&[b'A', b'V', b'I', b' '], 8)  // "AVI "
];
static MKV_SIGNATURE: &[Signature] = &[(&[0x1A, 0x45, 0xDF, 0xA3], 0)]; // Matroska signature
static MP4_SIGNATURES: &[Signature] = &[(&[b'f', b't', b'y', b'p'], 4)]; // "ftyp"

// Define a static PHF map for photo signatures
static PHOTO_SIGNATURES_MAP: FileSignatureMap = phf_map! {
    "jpg" => JPG_SIGNATURES,
    "jpeg" => JPG_SIGNATURES,
    "jfif" => JPG_SIGNATURES,
    "jpe" => JPG_SIGNATURES,
    "png" => PNG_SIGNATURES,
    "bmp" => BMP_SIGNATURES,
    "dib" => BMP_SIGNATURES,
    "webp" => WEBP_SIGNATURES,
    "jp2" => JP2_SIGNATURES,
    "pbm" => PBM_SIGNATURES,
    "pgm" => PGM_SIGNATURES,
    "ppm" => PPM_SIGNATURES,
    "pnm" => PNM_SIGNATURES,
    "pxm" => PNM_SIGNATURES,
    "pfm" => PFM_SIGNATURES,
    "sr" => SUN_RASTER_SIGNATURE,
    "ras" => SUN_RASTER_SIGNATURE,
    "hdr" => RADIANCE_SIGNATURES,
    "pic" => RADIANCE_SIGNATURES,
};

// Similar maps for other categories
static RAW_SIGNATURES_MAP: FileSignatureMap = phf_map! {
    "dng" => DNG_SIGNATURES,
    "arw" => TIFF_LE_SIGNATURE,
    "nef" => TIFF_SIGNATURES,
    "cr2" => CR2_SIGNATURE,
    "crw" => CRW_SIGNATURE,
    "raf" => FUJI_RAF_SIGNATURE,
    "x3f" => X3F_SIGNATURE,
    "orf" => ORF_SIGNATURES,
    "erf" => TIFF_SIGNATURES,
    "kdc" => TIFF_SIGNATURES,
    "nrw" => TIFF_LE_SIGNATURE,
    "pef" => TIFF_LE_SIGNATURE,
    "raw" => TIFF_SIGNATURES,
    "sr2" => TIFF_LE_SIGNATURE,
    "srw" => TIFF_LE_SIGNATURE,
    "exr" => EXR_SIGNATURE,
};

static TIFF_SIGNATURES_MAP: FileSignatureMap = phf_map! {
    "tiff" => TIFF_SIGNATURES,
    "tif" => TIFF_SIGNATURES,
};

static VIDEO_SIGNATURES_MAP: FileSignatureMap = phf_map! {
    "mp4" => MP4_SIGNATURES,
    "m4v" => MP4_SIGNATURES,
    "mov" => MP4_SIGNATURES,
    "avi" => AVI_SIGNATURES,
    "mkv" => MKV_SIGNATURE,
};

static TABLE_SIGNATURES_MAP: FileSignatureMap = phf_map! {
    "xlsx" => ZIP_SIGNATURE,
    "xlsm" => ZIP_SIGNATURE,
    "xltx" => ZIP_SIGNATURE,
    "xltm" => ZIP_SIGNATURE,
    "parquet" => PARQUET_SIGNATURES,
};

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
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(validate_files, m)?)?;
    m.add_function(wrap_pyfunction!(verify_file_type, m)?)?;
    
    Ok(())
}

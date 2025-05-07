// File validation additions to lib.rs
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::{Read, BufReader};
use std::collections::HashMap;
use std::sync::LazyLock;
use std::borrow::Cow;

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
type Signature = (Vec<u8>, usize);
type SignatureList = Vec<Signature>;
type ExtensionMap = HashMap<Cow<'static, str>, SignatureList>;

// Global registry initialized once for efficiency
static SIGNATURE_REGISTRY: LazyLock<SignatureRegistry> = LazyLock::new(|| SignatureRegistry::new());

// File signature registry
struct SignatureRegistry {
    signatures: HashMap<FileCategory, ExtensionMap>,
}

impl SignatureRegistry {
    fn new() -> Self {
        let mut registry = SignatureRegistry {
            signatures: HashMap::with_capacity(5),
        };
        
        // Initialize maps for each category
        let mut photo_signatures = ExtensionMap::new();
        let mut raw_signatures = ExtensionMap::new();
        let mut tiff_signatures = ExtensionMap::new();
        let mut video_signatures = ExtensionMap::new();
        let mut table_signatures = ExtensionMap::new();
        
        // Initialize photo signatures
        photo_signatures.insert(".jpg".into(), vec![(vec![0xFF, 0xD8, 0xFF], 0)]);
        photo_signatures.insert(".jpeg".into(), vec![(vec![0xFF, 0xD8, 0xFF], 0)]);
        photo_signatures.insert(".jfif".into(), vec![(vec![0xFF, 0xD8, 0xFF], 0)]);
        photo_signatures.insert(".jpe".into(), vec![(vec![0xFF, 0xD8, 0xFF], 0)]);
        photo_signatures.insert(".png".into(), vec![(vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], 0)]);
        photo_signatures.insert(".bmp".into(), vec![(vec![0x42, 0x4D], 0)]);
        photo_signatures.insert(".dib".into(), vec![(vec![0x42, 0x4D], 0)]);
        photo_signatures.insert(".webp".into(), vec![
            (vec![0x52, 0x49, 0x46, 0x46], 0), (vec![0x57, 0x45, 0x42, 0x50], 8)
        ]);
        photo_signatures.insert(".jp2".into(), vec![(vec![0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A], 0)]);
        
        // Netpbm family (ASCII vs. raw variants)
        photo_signatures.insert(".pbm".into(), vec![
            (vec![b'P', b'4'], 0),
            (vec![b'P', b'1'], 0)
        ]);
        photo_signatures.insert(".pgm".into(), vec![
            (vec![b'P', b'5'], 0),
            (vec![b'P', b'2'], 0)
        ]);
        photo_signatures.insert(".ppm".into(), vec![
            (vec![b'P', b'6'], 0),
            (vec![b'P', b'3'], 0)
        ]);
        photo_signatures.insert(".pnm".into(), vec![(vec![b'P', b'7'], 0)]);
        photo_signatures.insert(".pxm".into(), vec![(vec![b'P', b'7'], 0)]);
        
        // Portable FloatMap (32‑bit float HDR)
        photo_signatures.insert(".pfm".into(), vec![(vec![b'P', b'F'], 0), (vec![b'P', b'f'], 0)]);   // colour / greyscale
        
        // Sun Raster / SR files
        photo_signatures.insert(".sr".into(),  vec![(vec![0x59, 0xA6, 0x6A, 0x95], 0)]);
        photo_signatures.insert(".ras".into(), vec![(vec![0x59, 0xA6, 0x6A, 0x95], 0)]);

        // Radiance HDR / PIC: ASCII "#?RADIANCE" (occasionally "#?RGBE")
        photo_signatures.insert(".hdr".into(), vec![
            (vec![b'#', b'?', b'R', b'A', b'D', b'I', b'A', b'N', b'C', b'E'], 0),
            (vec![b'#', b'?', b'R', b'G', b'B', b'E'], 0)
        ]);
        photo_signatures.insert(".pic".into(), vec![
            (vec![b'#', b'?', b'R', b'A', b'D', b'I', b'A', b'N', b'C', b'E'], 0),
            (vec![b'#', b'?', b'R', b'G', b'B', b'E'], 0)
        ]);

        // Initialize raw signatures
        raw_signatures.insert(".dng".into(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0), (vec![0x49, 0x49, 0x00, 0x2A], 0)
        ]);
        raw_signatures.insert(".arw".into(), vec![(vec![0x49, 0x49, 0x2A, 0x00], 0)]);
        raw_signatures.insert(".nef".into(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0), (vec![0x4D, 0x4D, 0x00, 0x2A], 0)
        ]);
        raw_signatures.insert(".cr2".into(), vec![(vec![0x49, 0x49, 0x2A, 0x00, 0x10, 0x00, 0x00, 0x00, 0x43, 0x52], 0)]);
        raw_signatures.insert(".crw".into(), vec![(vec![0x49, 0x49, 0x1A, 0x00, 0x00, 0x00, 0x48, 0x45, 0x41, 0x50, 0x43, 0x43, 0x44, 0x52, 0x02, 0x00], 0)]);
        raw_signatures.insert(".raf".into(), vec![(vec![b'F', b'U', b'J', b'I', b'F', b'I', b'L', b'M', b'C', b'C', b'D', b'-', b'R', b'A', b'W'], 0)]);
        raw_signatures.insert(".x3f".into(), vec![(vec![b'F', b'O', b'V', b'b'], 0)]);
        raw_signatures.insert(".orf".into(), vec![
            (vec![b'I', b'I', b'R', b'O'], 0),
            (vec![b'I', b'I'], 0)
        ]);
        raw_signatures.insert(".erf".into(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0),
            (vec![0x4D, 0x4D, 0x00, 0x2A], 0)
        ]);
        raw_signatures.insert(".kdc".into(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0),
            (vec![0x4D, 0x4D, 0x00, 0x2A], 0)
        ]);
        raw_signatures.insert(".nrw".into(), vec![(vec![0x49, 0x49, 0x2A, 0x00], 0)]);
        raw_signatures.insert(".pef".into(), vec![(vec![0x49, 0x49, 0x2A, 0x00], 0)]);
        raw_signatures.insert(".raw".into(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0),
            (vec![0x4D, 0x4D, 0x00, 0x2A], 0)
        ]);
        raw_signatures.insert(".sr2".into(), vec![(vec![0x49, 0x49, 0x2A, 0x00], 0)]);
        raw_signatures.insert(".srw".into(), vec![(vec![0x49, 0x49, 0x2A, 0x00], 0)]);
        raw_signatures.insert(".exr".into(), vec![(vec![0x76, 0x2F, 0x31, 0x01], 0)]);
        
        // Initialize tiff signatures
        tiff_signatures.insert(".tiff".into(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0),  // Little-endian TIFF
            (vec![0x4D, 0x4D, 0x00, 0x2A], 0),  // Big-endian TIFF
        ]);
        tiff_signatures.insert(".tif".into(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0),  // Little-endian TIFF
            (vec![0x4D, 0x4D, 0x00, 0x2A], 0),  // Big-endian TIFF
        ]);

        // Initialize video signatures
        video_signatures.insert(".mp4".into(), vec![(vec![b'f', b't', b'y', b'p'], 4)]);
        video_signatures.insert(".m4v".into(), vec![(vec![b'f', b't', b'y', b'p'], 4)]);
        video_signatures.insert(".mov".into(), vec![(vec![b'f', b't', b'y', b'p'], 4)]);
        video_signatures.insert(".avi".into(), vec![
            (vec![b'R', b'I', b'F', b'F'], 0),
            (vec![b'A', b'V', b'I', b' '], 8),
        ]);
        video_signatures.insert(".mkv".into(), vec![(vec![0x1A, 0x45, 0xDF, 0xA3], 0)]);
        
        // Initialize table signatures
        table_signatures.insert(".xlsx".into(), vec![(vec![0x50, 0x4B, 0x03, 0x04], 0)]);
        table_signatures.insert(".xlsm".into(), vec![(vec![0x50, 0x4B, 0x03, 0x04], 0)]);
        table_signatures.insert(".xltx".into(), vec![(vec![0x50, 0x4B, 0x03, 0x04], 0)]);
        table_signatures.insert(".xltm".into(), vec![(vec![0x50, 0x4B, 0x03, 0x04], 0)]);
        table_signatures.insert(".parquet".into(), vec![
            (vec![b'P', b'A', b'R', b'1'], 0),
            (vec![b'P', b'A', b'R', b'E'], 0),
        ]);
        
        // Store all signature maps in the registry
        registry.signatures.insert(FileCategory::Photo, photo_signatures);
        registry.signatures.insert(FileCategory::Raw, raw_signatures);
        registry.signatures.insert(FileCategory::Tiff, tiff_signatures);
        registry.signatures.insert(FileCategory::Video, video_signatures);
        registry.signatures.insert(FileCategory::Table, table_signatures);
        
        registry
    }
    
    fn get_signatures(&self, path: &Path, category: FileCategory) -> Option<&SignatureList> {
        // Skip unknown category early
        if category == FileCategory::Unknown {
            return None;
        }
        
        // Get the extension in lowercase with a period
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .map(|s| format!(".{}", s.to_lowercase()))
            .unwrap_or_default();
            
        // Look up signatures for this category and extension
        self.signatures.get(&category).and_then(|map| map.get(&extension as &str))
    }
}

// Helper function to validate a CSV file
fn validate_csv(path: &Path) -> bool {
    match File::open(path) {
        Ok(file) => {
            let mut reader = BufReader::new(file);
            let mut buffer = [0u8; 1024]; // Reduced buffer size for efficiency
            
            match reader.read(&mut buffer) {
                Ok(0) => false, // Empty file
                Ok(bytes_read) => {
                    // Try to interpret as UTF-8 first
                    if let Ok(text) = std::str::from_utf8(&buffer[..bytes_read]) {
                        let first_line = text.lines().next().unwrap_or("");
                        first_line.contains(',') || first_line.contains('\t')
                    } else {
                        // Fallback to Latin-1 interpretation for non-UTF8 text
                        let first_line: String = buffer[..bytes_read.min(256)]
                            .iter()
                            .take_while(|&&b| b != b'\n' && b != b'\r')
                            .map(|&b| b as char)
                            .collect();
                        first_line.contains(',') || first_line.contains('\t')
                    }
                },
                Err(_) => false,
            }
        },
        Err(_) => false,
    }
}

// Validate a single file with improved memory usage and error handling
fn validate_file(path: &Path, category: FileCategory) -> bool {
    // Check if file exists and is readable
    if !path.exists() || !path.is_file() {
        return false;
    }
    
    // Special case for CSV files (Table category)
    if category == FileCategory::Table {
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
            
        if extension == "csv" {
            return validate_csv(path);
        }
    }
    
    // Get signatures for the file category and extension
    if let Some(signatures) = SIGNATURE_REGISTRY.get_signatures(path, category) {
        return check_file_signatures(path, signatures);
    }
    
    false
}

// Optimized helper function to check file signatures with smarter buffer management
fn check_file_signatures(path: &Path, signatures: &[Signature]) -> bool {
    // If there are no signatures to check, return false early
    if signatures.is_empty() {
        return false;
    }
    
    // Find the maximum read size needed
    let max_offset_plus_len = signatures.iter()
        .map(|(sig, offset)| offset + sig.len())
        .max()
        .unwrap_or(0);
    
    // Try to open the file
    match File::open(path) {
        Ok(mut file) => {
            // Read only as many bytes as we need
            let mut buffer = vec![0u8; max_offset_plus_len];
            match file.read(&mut buffer) {
                Ok(bytes_read) if bytes_read >= max_offset_plus_len => {
                    // Check each signature
                    signatures.iter().any(|(signature, offset)| {
                        *offset + signature.len() <= buffer.len() &&
                        &buffer[*offset..*offset + signature.len()] == signature
                    })
                },
                _ => false, // Not enough bytes or read error
            }
        },
        Err(_) => false,
    }
}

/// Validates multiple files in parallel based on their categories.
///
/// Args:
///     file_paths: List of string paths to files to validate
///     categories: List of category codes (0=Photo, 1=Raw, 2=Tiff, 3=Video, 4=Table, 5=Unknown)
///
/// Returns:
///     A boolean NumPy array with validation results (True = valid, False = invalid)
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
    
    // Convert to PathBuf and FileCategory in one pass
    let path_categories: Vec<(PathBuf, FileCategory)> = file_paths.into_iter()
        .zip(categories.into_iter())
        .map(|(path, category)| {
            let file_category = match category {
                0 => FileCategory::Photo,
                1 => FileCategory::Raw,
                2 => FileCategory::Tiff,
                3 => FileCategory::Video,
                4 => FileCategory::Table,
                _ => FileCategory::Unknown,
            };
            (PathBuf::from(path), file_category)
        })
        .collect();
    
    // Process files in parallel
    let results: Vec<bool> = path_categories.par_iter()
        .map(|(path, category)| validate_file(path, *category))
        .collect();
    
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
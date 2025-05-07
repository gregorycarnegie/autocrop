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

// File category enum matching Python's FileCategory
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FileCategory {
    Photo = 0,
    Raw = 1,
    Tiff = 2,
    Video = 3,
    Table = 4,
    Unknown = 5,
}

// Global registry initialized once for efficiency
// static SIGNATURE_REGISTRY: LazyLock<RwLock<SignatureRegistry>> = 
//     LazyLock::new(|| RwLock::new(SignatureRegistry::new()));
static SIGNATURE_REGISTRY: LazyLock<SignatureRegistry> = LazyLock::new(SignatureRegistry::new);

// File signature registry
struct SignatureRegistry {
    photo_signatures: HashMap<String, Vec<(Vec<u8>, usize)>>,
    raw_signatures: HashMap<String, Vec<(Vec<u8>, usize)>>,
    tiff_signatures: HashMap<String, Vec<(Vec<u8>, usize)>>,
    video_signatures: HashMap<String, Vec<(Vec<u8>, usize)>>,
    table_signatures: HashMap<String, Vec<(Vec<u8>, usize)>>,
}

impl SignatureRegistry {
    fn new() -> Self {
        let mut registry = SignatureRegistry {
            photo_signatures: HashMap::new(),
            raw_signatures: HashMap::new(),
            tiff_signatures: HashMap::new(),
            video_signatures: HashMap::new(),
            table_signatures: HashMap::new(),
        };
        
        // Initialize photo signatures
        registry.photo_signatures.insert(".jpg".to_string(), vec![(vec![0xFF, 0xD8, 0xFF], 0)]);
        registry.photo_signatures.insert(".jpeg".to_string(), vec![(vec![0xFF, 0xD8, 0xFF], 0)]);
        registry.photo_signatures.insert(".jfif".to_string(), vec![(vec![0xFF, 0xD8, 0xFF], 0)]);
        registry.photo_signatures.insert(".jpe".to_string(), vec![(vec![0xFF, 0xD8, 0xFF], 0)]);
        registry.photo_signatures.insert(".png".to_string(), vec![(vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A], 0)]);
        registry.photo_signatures.insert(".bmp".to_string(), vec![(vec![0x42, 0x4D], 0)]);
        registry.photo_signatures.insert(".dib".to_string(), vec![(vec![0x42, 0x4D], 0)]);
        registry.photo_signatures.insert(".webp".to_string(), vec![
            (vec![0x52, 0x49, 0x46, 0x46], 0), (vec![0x57, 0x45, 0x42, 0x50], 8)
        ]);
        registry.photo_signatures.insert(".jp2".to_string(), vec![(vec![0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A], 0)]);
        
        // Netpbm family (ASCII vs. raw variants)
        registry.photo_signatures.insert(".pbm".to_string(), vec![
            (vec![b'P', b'4'], 0),
            (vec![b'P', b'1'], 8)
        ]);
        registry.photo_signatures.insert(".pgm".to_string(), vec![
            (vec![b'P', b'5'], 0),
            (vec![b'P', b'2'], 8)
        ]);
        registry.photo_signatures.insert(".ppm".to_string(), vec![
            (vec![b'P', b'6'], 0),
            (vec![b'P', b'3'], 8)
        ]);
        registry.photo_signatures.insert(".pnm".to_string(), vec![(vec![b'P', b'7'], 0)]);
        registry.photo_signatures.insert(".pxm".to_string(), vec![(vec![b'P', b'7'], 0)]);
        
        // Portable FloatMap (32‑bit float HDR)
        registry.photo_signatures.insert(".pfm".to_string(), vec![(vec![b'P', b'F'], 0), (vec![b'P', b'f'], 0)]);   // colour / greyscale
        
        // Sun Raster / SR files
        registry.photo_signatures.insert(".sr".to_string(),  vec![(vec![0x59, 0xA6, 0x6A, 0x95], 0)]);
        registry.photo_signatures.insert( ".ras".to_string(), vec![(vec![0x59, 0xA6, 0x6A, 0x95], 0)]);

        // Radiance HDR / PIC: ASCII “#?RADIANCE” (occasionally “#?RGBE”)
        registry.photo_signatures.insert(".hdr".to_string(), vec![
            (vec![b'R',b'A',b'D',b'I',b'A',b'N',b'C',b'E'], 0),
            (vec![b'R',b'G',b'B',b'E'], 0)
        ]);
        registry.photo_signatures.insert(".pic".to_string(), vec![
            (vec![b'R',b'A',b'D',b'I',b'A',b'N',b'C',b'E'], 0),
            (vec![b'R',b'G',b'B',b'E'], 0)
        ]);

        // Initialize raw signatures (matching the ones in file_types/signature_checker.py)
        registry.raw_signatures.insert(".dng".to_string(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0), (vec![0x49, 0x49, 0x00, 0x2A], 0)
        ]);
        registry.raw_signatures.insert(".arw".to_string(), vec![(vec![0x49, 0x49, 0x2A, 0x00], 0)]);
        registry.raw_signatures.insert(".nef".to_string(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0), (vec![0x4D, 0x4D, 0x00, 0x2A], 0)
        ]);
        registry.raw_signatures.insert(".cr2".to_string(), vec![(vec![0x49, 0x49, 0x2A, 0x00, 0x10, 0x00, 0x00, 0x00, 0x43, 0x52], 0)]);
        registry.raw_signatures.insert(".crw".to_string(), vec![(vec![0x49, 0x49, 0x1A, 0x00, 0x00, 0x00, 0x48, 0x45, 0x41, 0x50, 0x43, 0x43, 0x44, 0x52, 0x02, 0x00], 0)]);
        registry.raw_signatures.insert(".raf".to_string(), vec![(vec![b'F', b'U', b'J', b'I', b'F', b'I', b'L', b'M', b'C', b'C', b'D', b'-', b'R', b'A', b'W'], 0)]);
        registry.raw_signatures.insert(".x3f".to_string(), vec![(vec![b'F', b'O', b'V', b'b'], 0)]);
        registry.raw_signatures.insert(".orf".to_string(), vec![
            (vec![b'I', b'I', b'R', b'O'], 0),
            (vec![b'I', b'I'], 0)
        ]);
        registry.raw_signatures.insert(".erf".to_string(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0),
            (vec![0x4D, 0x4D, 0x00, 0x2A], 0)
        ]);
        registry.raw_signatures.insert(".kdc".to_string(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0),
            (vec![0x4D, 0x4D, 0x00, 0x2A], 0)
        ]);
        registry.raw_signatures.insert(".nrw".to_string(), vec![(vec![0x49, 0x49, 0x2A, 0x00], 0)]);
        registry.raw_signatures.insert(".pef".to_string(), vec![(vec![0x49, 0x49, 0x2A, 0x00], 0)]);
        registry.raw_signatures.insert(".raw".to_string(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0),
            (vec![0x4D, 0x4D, 0x00, 0x2A], 0)
        ]);
        registry.raw_signatures.insert(".sr2".to_string(), vec![(vec![0x49, 0x49, 0x2A, 0x00], 0)]);
        registry.raw_signatures.insert(".srw".to_string(), vec![(vec![0x49, 0x49, 0x2A, 0x00], 0)]);
        registry.raw_signatures.insert(".exr".to_string(), vec![(vec![0x76, 0x2F, 0x31, 0x01], 0)]);
        
        // Initialize tiff signatures
        registry.tiff_signatures.insert(".tiff".to_string(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0),  // Little-endian TIFF
            (vec![0x4D, 0x4D, 0x00, 0x2A], 0),  // Big-endian TIFF
        ]);
        registry.tiff_signatures.insert(".tif".to_string(), vec![
            (vec![0x49, 0x49, 0x2A, 0x00], 0),  // Little-endian TIFF
            (vec![0x4D, 0x4D, 0x00, 0x2A], 0),  // Big-endian TIFF
        ]);

        // Initialize video signatures
        registry.video_signatures.insert(".mp4".to_string(), vec![(vec![b'f', b't', b'y', b'p'], 4)]);
        registry.video_signatures.insert(".m4v".to_string(), vec![(vec![b'f', b't', b'y', b'p'], 4)]);
        registry.video_signatures.insert(".mov".to_string(), vec![(vec![b'f', b't', b'y', b'p'], 4)]);
        registry.video_signatures.insert(".avi".to_string(), vec![
            (vec![b'R', b'I', b'F', b'F'], 0),
            (vec![b'A', b'V', b'I', b' '], 8),
        ]);
        registry.video_signatures.insert(".mkv".to_string(), vec![(vec![0x1A, 0x45, 0xDF, 0xA3], 0)]);
        
        // Initialize table signatures
        registry.table_signatures.insert(".xlsx".to_string(), vec![(vec![0x50, 0x4B, 0x03, 0x04], 0)]);
        registry.table_signatures.insert(".xlsm".to_string(), vec![(vec![0x50, 0x4B, 0x03, 0x04], 0)]);
        registry.table_signatures.insert(".xltx".to_string(), vec![(vec![0x50, 0x4B, 0x03, 0x04], 0)]);
        registry.table_signatures.insert(".xltm".to_string(), vec![(vec![0x50, 0x4B, 0x03, 0x04], 0)]);
        registry.table_signatures.insert(".parquet".to_string(), vec![
            (vec![b'P', b'A', b'R', b'1'], 0),
            (vec![b'P', b'A', b'R', b'E'], 0),
        ]);
        
        registry
    }
    
    fn get_signatures(&self, path: &Path, category: FileCategory) -> Option<&Vec<(Vec<u8>, usize)>> {
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .map(|s| format!(".{}", s.to_lowercase()))
            .unwrap_or_default();
            
        match category {
            FileCategory::Photo => self.photo_signatures.get(&extension),
            FileCategory::Raw => self.raw_signatures.get(&extension),
            FileCategory::Tiff => self.tiff_signatures.get(&extension),
            FileCategory::Video => self.video_signatures.get(&extension),
            FileCategory::Table => self.table_signatures.get(&extension),
            _ => None,
        }
    }
}

// Helper function to get the registry, initializing it if needed
fn get_registry() -> &'static SignatureRegistry {
    &SIGNATURE_REGISTRY
}

// Validate a single file
fn validate_file(path: &Path, category: FileCategory) -> bool {
    // Check if file exists and is readable
    if !path.exists() || !path.is_file() {
        return false;
    }
    
    // Special case for CSV files
    if category == FileCategory::Table {
        let extension = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
            
        if extension == "csv" {
            return validate_text_file(path);
        }
    }
    
    // Get the registry (now returns RwLockReadGuard<SignatureRegistry> directly)
    let registry = get_registry();
    
    // Get signatures for the file category and extension
    if let Some(signatures) = registry.get_signatures(path, category) {
        return check_file_signatures(path, signatures);
    }
    
    false
}

// Helper function to check file signatures
fn check_file_signatures(path: &Path, signatures: &[(Vec<u8>, usize)]) -> bool {
    if let Ok(mut file) = File::open(path) {
        // Find the maximum read size needed
        let max_offset_plus_len = signatures.iter()
            .map(|(sig, offset)| offset + sig.len())
            .max()
            .unwrap_or(0);
        
        // Read enough bytes
        let mut buffer = vec![0u8; max_offset_plus_len];
        if let Ok(_) = file.read_exact(&mut buffer) {
            // Check each signature
            for (signature, offset) in signatures {
                if *offset + signature.len() <= buffer.len() &&
                   &buffer[*offset..*offset + signature.len()] == signature {
                    return true;
                }
            }
        }
    }
    false
}

// Helper function to validate text files (CSV)
fn validate_text_file(path: &Path) -> bool {
    if let Ok(file) = File::open(path) {
        let mut reader = BufReader::new(file);
        let mut buffer = [0u8; 4096];
        
        if let Ok(bytes_read) = reader.read(&mut buffer) {
            if bytes_read == 0 {
                return false;
            }
            
            // Try to interpret as UTF-8
            if let Ok(text) = std::str::from_utf8(&buffer[..bytes_read]) {
                let lines = text.lines().take(2).collect::<Vec<_>>();
                if lines.is_empty() {
                    return false;
                }
                
                // Check for CSV structure
                return lines[0].contains(',') || lines[0].contains('\t');
            } else {
                // Try with Latin-1 encoding if UTF-8 fails
                // (simplified approach since Rust doesn't handle this directly)
                let text: String = buffer[..bytes_read]
                    .iter()
                    .map(|&b| b as char)
                    .collect();
                let first_line = text.lines().next().unwrap_or("");
                return first_line.contains(',') || first_line.contains('\t');
            }
        }
    }
    false
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
    // Convert paths to PathBuf
    let path_bufs: Vec<PathBuf> = file_paths.iter()
        .map(|p| PathBuf::from(p))
        .collect();
    
    // Convert categories to FileCategory enum
    let file_categories: Vec<FileCategory> = categories.iter()
        .map(|&c| match c {
            0 => FileCategory::Photo,
            1 => FileCategory::Raw,
            2 => FileCategory::Tiff,
            3 => FileCategory::Video,
            4 => FileCategory::Table,
            _ => FileCategory::Unknown,
        })
        .collect();
    
    // Validate files in parallel
    let results: Vec<bool> = path_bufs.par_iter()
        .zip(file_categories.par_iter())
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

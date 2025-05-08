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
type SignatureList = Vec<Signature>;
type ExtensionMap = HashMap<&'static str, SignatureList>;

// Global registry initialized once for efficiency
static SIGNATURE_REGISTRY: LazyLock<SignatureRegistry> = LazyLock::new(|| SignatureRegistry::new());

// File signature registry
struct SignatureRegistry {
    signatures: HashMap<FileCategory, ExtensionMap>,
}

// Common file signatures as static byte slices
static JPG_SIG: &[u8] = &[0xFF, 0xD8, 0xFF];
static JP2_SIG: &[u8] = &[0x00, 0x00, 0x00, 0x0C, 0x6A, 0x50, 0x20, 0x20, 0x0D, 0x0A, 0x87, 0x0A];
static PNG_SIG: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
static BMP_SIG: &[u8] = &[0x42, 0x4D];
static WEBP_SIG1: &[u8] = &[0x52, 0x49, 0x46, 0x46];  // "RIFF"
static WEBP_SIG2: &[u8] = &[0x57, 0x45, 0x42, 0x50];  // "WEBP"

// ASCII signatures can use byte literals
static PBM_RAW_SIG: &[u8] = &[b'P', b'4'];
static PBM_ASCII_SIG: &[u8] = &[b'P', b'1'];
static PGM_RAW_SIG: &[u8] = &[b'P', b'5'];
static PGM_ASCII_SIG: &[u8] = &[b'P', b'2'];
static PPM_RAW_SIG: &[u8] = &[b'P', b'6'];
static PPM_ASCII_SIG: &[u8] = &[b'P', b'3'];

static PNM_PXM_SIG: &[u8] = &[b'P', b'7'];

// Portable FloatMap (32‑bit float HDR)
static PFM_SIG1: &[u8] = &[b'P', b'F'];
static PFM_SIG2: &[u8] = &[b'P', b'f']; // Lowercase variant

// Sun Raster signature
static SUN_RASTER_SIG: &[u8] = &[0x59, 0xA6, 0x6A, 0x95]; // "Y\xA6j\x95"

// RAW format signatures
static DNG_SIG: &[u8] = &[0x49, 0x49, 0x2A, 0x00]; // "II*\0" (little endian)
static CR2_SIG: &[u8] = &[0x49, 0x49, 0x2A, 0x00, 0x10, 0x00, 0x00, 0x00, 0x43, 0x52]; // "II*\0" (little endian)
static CRW_SIG: &[u8] = &[0x49, 0x49, 0x1A, 0x00, 0x00, 0x00, 0x48, 0x45, 0x41, 0x50, 0x43, 0x43, 0x44, 0x52, 0x02, 0x00]; // "II*\0" (little endian)
static EXR_SIG: &[u8] = &[0x76, 0x2F, 0x31, 0x01]; // "v/1\0"
static X3F_SIG: &[u8] = &[b'F', b'O', b'V', b'b']; // "FOVb"
static ORF_SIG1: &[u8] = &[b'I', b'I', b'R', b'O']; // "IIR\0"
static ORF_SIG2: &[u8] = &[b'I', b'I'];
static TIFF_LE_SIG: &[u8] = &[0x49, 0x49, 0x2A, 0x00];  // "II*\0" (little endian)
static TIFF_BE_SIG: &[u8] = &[0x4D, 0x4D, 0x00, 0x2A];  // "MM\0*" (big endian)

// For longer signatures
static FUJI_RAF_SIG: &[u8] = &[
    b'F', b'U', b'J', b'I', b'F', b'I', b'L', b'M', 
    b'C', b'C', b'D', b'-', b'R', b'A', b'W'
];

static RADIANCE_HDR_SIG: &[u8] = &[
    b'#', b'?', b'R', b'A', b'D', b'I', b'A', b'N', b'C', b'E'
];
static RADIANCE_PIC_SIG: &[u8] = &[
    b'#', b'?', b'R', b'G', b'B', b'E'
];

// Table format signatures
static ZIP_SIG: &[u8] = &[0x50, 0x4B, 0x03, 0x04]; // "PK\x03\x04"
static PARQUET_SIG1: &[u8] = &[b'P', b'A', b'R', b'1']; // "PAR1"
static PARQUET_SIG2: &[u8] = &[b'P', b'A', b'R', b'E']; // "PARE"

// Video format signatures
static AVI_SIG1: &[u8] = &[b'R', b'I', b'F', b'F']; // "RIFF"
static AVI_SIG2: &[u8] = &[b'A', b'V', b'I', b' ']; // "AVI "
static MKV_SIG: &[u8] = &[0x1A, 0x45, 0xDF, 0xA3]; // Matroska signature
static MP4_SIG: &[u8] = &[b'f', b't', b'y', b'p']; // "ftyp"

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
        photo_signatures.insert(".jpg", vec![(JPG_SIG, 0)]);
        photo_signatures.insert(".jpeg", vec![(JPG_SIG, 0)]);
        photo_signatures.insert(".jfif", vec![(JPG_SIG, 0)]);
        photo_signatures.insert(".jpe", vec![(JPG_SIG, 0)]);
        photo_signatures.insert(".png", vec![(PNG_SIG, 0)]);
        photo_signatures.insert(".bmp", vec![(BMP_SIG, 0)]);
        photo_signatures.insert(".dib", vec![(BMP_SIG, 0)]);
        photo_signatures.insert(".webp", vec![(WEBP_SIG1, 0), (WEBP_SIG2, 8)]);
        photo_signatures.insert(".jp2", vec![(JP2_SIG, 0)]);
        
        // Netpbm family (ASCII vs. raw variants)
        photo_signatures.insert(".pbm", vec![(PBM_RAW_SIG, 0), (PBM_ASCII_SIG, 0)]);
        photo_signatures.insert(".pgm", vec![(PGM_RAW_SIG, 0), (PGM_ASCII_SIG, 0)]);
        photo_signatures.insert(".ppm", vec![(PPM_RAW_SIG, 0), (PPM_ASCII_SIG, 0)]);
        photo_signatures.insert(".pnm", vec![(PNM_PXM_SIG, 0)]);
        photo_signatures.insert(".pxm", vec![(PNM_PXM_SIG, 0)]);
        
        // Portable FloatMap (32‑bit float HDR)
        photo_signatures.insert(".pfm", vec![(PFM_SIG1, 0), (PFM_SIG2, 0)]);   // colour / greyscale
        
        // Sun Raster / SR files
        photo_signatures.insert(".sr",  vec![(SUN_RASTER_SIG, 0)]);
        photo_signatures.insert(".ras", vec![(SUN_RASTER_SIG, 0)]);

        // Radiance HDR / PIC: ASCII "#?RADIANCE" (occasionally "#?RGBE")
        photo_signatures.insert(".hdr", vec![(RADIANCE_HDR_SIG, 0), (RADIANCE_PIC_SIG, 0)]);
        photo_signatures.insert(".pic", vec![(RADIANCE_HDR_SIG, 0), (RADIANCE_PIC_SIG, 0)]);

        // Initialize raw signatures
        raw_signatures.insert(".dng", vec![(TIFF_LE_SIG, 0), (DNG_SIG, 0)]);
        raw_signatures.insert(".arw", vec![(TIFF_LE_SIG, 0)]);
        raw_signatures.insert(".nef", vec![(TIFF_LE_SIG, 0), (TIFF_BE_SIG, 0)]);
        raw_signatures.insert(".cr2", vec![(CR2_SIG, 0)]);
        raw_signatures.insert(".crw", vec![(CRW_SIG, 0)]);
        raw_signatures.insert(".raf", vec![(FUJI_RAF_SIG, 0)]);
        raw_signatures.insert(".x3f", vec![(X3F_SIG, 0)]);
        raw_signatures.insert(".orf", vec![(ORF_SIG1, 0), (ORF_SIG2, 0)]);
        raw_signatures.insert(".erf", vec![(TIFF_LE_SIG, 0), (TIFF_BE_SIG, 0)]);
        raw_signatures.insert(".kdc", vec![(TIFF_LE_SIG, 0), (TIFF_BE_SIG, 0)]);
        raw_signatures.insert(".nrw", vec![(TIFF_LE_SIG, 0)]);
        raw_signatures.insert(".pef", vec![(TIFF_LE_SIG, 0)]);
        raw_signatures.insert(".raw", vec![(TIFF_LE_SIG, 0), (TIFF_BE_SIG, 0)]);
        raw_signatures.insert(".sr2", vec![(TIFF_LE_SIG, 0)]);
        raw_signatures.insert(".srw", vec![(TIFF_LE_SIG, 0)]);
        raw_signatures.insert(".exr", vec![(EXR_SIG, 0)]);
        
        // Initialize tiff signatures
        tiff_signatures.insert(".tiff", vec![
            (TIFF_LE_SIG, 0),  // Little-endian TIFF
            (TIFF_BE_SIG, 0),  // Big-endian TIFF
        ]);
        tiff_signatures.insert(".tif", vec![
            (TIFF_LE_SIG, 0),  // Little-endian TIFF
            (TIFF_BE_SIG, 0),  // Big-endian TIFF
        ]);

        // Initialize video signatures
        video_signatures.insert(".mp4", vec![(MP4_SIG, 4)]);
        video_signatures.insert(".m4v", vec![(MP4_SIG, 4)]);
        video_signatures.insert(".mov", vec![(MP4_SIG, 4)]);
        video_signatures.insert(".avi", vec![(AVI_SIG1, 0), (AVI_SIG2, 8)]);
        video_signatures.insert(".mkv", vec![(MKV_SIG, 0)]);
        
        // Initialize table signatures
        table_signatures.insert(".xlsx", vec![(ZIP_SIG, 0)]);
        table_signatures.insert(".xlsm", vec![(ZIP_SIG, 0)]);
        table_signatures.insert(".xltx", vec![(ZIP_SIG, 0)]);
        table_signatures.insert(".xltm", vec![(ZIP_SIG, 0)]);
        table_signatures.insert(".parquet", vec![(PARQUET_SIG1, 0), (PARQUET_SIG2, 0)]);
        
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
        self.signatures.get(&category).and_then(|map| map.get(extension.as_str()))
    }

    fn prefilter_file_category(&self, first_bytes: &[u8], extension: &str) -> Option<Vec<FileCategory>> {
        if first_bytes.len() < 4 {
            return None;
        }
        
        let mut possible_categories = Vec::with_capacity(2);
        
        // JPEG detection
        if first_bytes.starts_with(JPG_SIG) {
            possible_categories.push(FileCategory::Photo);
        }
        // PNG detection
        else if first_bytes.len() >= 8 && first_bytes.starts_with(PNG_SIG) {
            possible_categories.push(FileCategory::Photo);
        }
        // TIFF or RAW detection
        else if first_bytes.starts_with(TIFF_LE_SIG) || first_bytes.starts_with(TIFF_BE_SIG) {
            // Check extension to narrow down
            let ext = extension.to_lowercase();
            if ext == ".tif" || ext == ".tiff" {
                possible_categories.push(FileCategory::Tiff);
            } else if ext == ".dng" || ext == ".arw" || ext == ".nef" || ext == ".cr2" {
                possible_categories.push(FileCategory::Raw);
            } else {
                // Could be either
                possible_categories.push(FileCategory::Tiff);
                possible_categories.push(FileCategory::Raw);
            }
        }
        // Video detection
        else if first_bytes.starts_with(AVI_SIG1) || 
                (first_bytes.len() >= 8 && first_bytes[4..8].eq(MP4_SIG)) ||  // Fixed comparison
                first_bytes.starts_with(MKV_SIG) {
            possible_categories.push(FileCategory::Video);
        }
        // Table detection
        else if first_bytes.starts_with(ZIP_SIG) || 
                first_bytes.starts_with(PARQUET_SIG1) ||
                first_bytes.starts_with(PARQUET_SIG2) {
            possible_categories.push(FileCategory::Table);
        }
        
        if possible_categories.is_empty() {
            None
        } else {
            Some(possible_categories)
        }
    }
}

// Helper function to validate a CSV file
fn validate_csv(path: &Path) -> bool {
    match File::open(path) {
        Ok(file) => {
            let mut reader = BufReader::with_capacity(512, file); // Smaller capacity
            let mut buffer = [0u8; 256]; // Even smaller read buffer
            
            match reader.read(&mut buffer) {
                Ok(0) => false, // Empty file
                Ok(bytes_read) => {
                    // Count delimiter characters directly
                    let comma_count = buffer[..bytes_read].iter().filter(|&&b| b == b',').count();
                    let tab_count = buffer[..bytes_read].iter().filter(|&&b| b == b'\t').count();
                    let newline_count = buffer[..bytes_read].iter().filter(|&&b| b == b'\n' || b == b'\r').count();
                    
                    // More specific CSV heuristic - need delimiters and at least one newline
                    (comma_count > 0 || tab_count > 0) && newline_count > 0
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
    
    // Get extension for prefiltering
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .map(|s| format!(".{}", s.to_lowercase()))
        .unwrap_or_default();
    
    // Read first 16 bytes for detection
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return false,
    };
    
    let mut reader = BufReader::new(file);
    let mut header = [0u8; 16];
    
    if let Ok(bytes_read) = reader.read(&mut header) {
        if bytes_read >= 4 {
            // First try direct MIME type detection
            if let Some(detected_category) = detect_mime_type(&header[..bytes_read]) {
                // If detected category matches requested category, we can be confident
                if detected_category == category {
                    return true;
                }
            }
            
            // If direct detection didn't provide a match, try prefiltering
            if let Some(possible_categories) = SIGNATURE_REGISTRY.prefilter_file_category(
                &header[..bytes_read], 
                &extension
            ) {
                // If our category isn't in the possible list, we can return false immediately
                if !possible_categories.contains(&category) {
                    return false;
                }
                
                // If our category is the only possibility, we can return true
                if possible_categories.len() == 1 && possible_categories[0] == category {
                    return true;
                }
                
                // If our category is one of multiple possibilities, continue to detailed checking
            }
        }
    }
    
    // Get signatures for the file category and extension
    if let Some(signatures) = SIGNATURE_REGISTRY.get_signatures(path, category) {
        return check_file_signatures(path, signatures);
    }
    
    false
}

/// Detect file type based on the first few bytes (magic numbers)
fn detect_mime_type(buffer: &[u8]) -> Option<FileCategory> {
    if buffer.len() < 4 {
        return None; // Not enough data
    }
    
    // JPEG signature
    if buffer.starts_with(JPG_SIG) {
        return Some(FileCategory::Photo);
    }
    
    // PNG signature 
    if buffer.len() >= 8 && buffer.starts_with(PNG_SIG) {
        return Some(FileCategory::Photo);
    }
    
    // TIFF signatures (could be TIFF or RAW)
    if buffer.starts_with(TIFF_LE_SIG) || buffer.starts_with(TIFF_BE_SIG) {
        return Some(FileCategory::Tiff); // Default to TIFF, would need to check specific RAW signatures
    }
    
    // WEBP detection
    if buffer.len() >= 12 && 
       buffer.starts_with(WEBP_SIG1) && 
       &buffer[8..12] == WEBP_SIG2 {
        return Some(FileCategory::Photo);
    }
    
    // BMP detection
    if buffer.starts_with(BMP_SIG) {
        return Some(FileCategory::Photo);
    }
    
    // MP4/QuickTime container detection
    if buffer.len() >= 8 && 
       buffer[4..8].eq(MP4_SIG) {  // Changed to use .eq() method
        return Some(FileCategory::Video);
    }
    
    // AVI detection
    if buffer.len() >= 12 && 
       buffer.starts_with(AVI_SIG1) && 
       &buffer[8..12] == AVI_SIG2 {
        return Some(FileCategory::Video);
    }
    
    // Matroska (MKV) detection
    if buffer.len() >= 4 && buffer.starts_with(MKV_SIG) {
        return Some(FileCategory::Video);
    }
    
    // ZIP-based formats (could be XLSX, etc.)
    if buffer.starts_with(ZIP_SIG) {
        return Some(FileCategory::Table);
    }
    
    // Parquet format detection
    if buffer.starts_with(PARQUET_SIG1) || buffer.starts_with(PARQUET_SIG2) {
        return Some(FileCategory::Table);
    }
    
    None // Unknown format
}

// Optimized helper function to check file signatures with static byte arrays
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
                    // Check signatures against the buffer
                    return signatures.iter().any(|(signature, offset)| {
                        *offset + signature.len() <= max_offset_plus_len &&
                        &buffer[*offset..*offset + signature.len()] == *signature
                    });
                }
            }
        } else {
            // Use heap allocation for larger reads
            let mut buffer = vec![0u8; max_offset_plus_len];
            
            if let Ok(bytes_read) = file.read(&mut buffer) {
                if bytes_read >= max_offset_plus_len {
                    // Check signatures against the buffer
                    return signatures.iter().any(|(signature, offset)| {
                        *offset + signature.len() <= buffer.len() &&
                        &buffer[*offset..*offset + signature.len()] == *signature
                    });
                }
            }
        }
    }
    
    false // File couldn't be opened or read
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
    
    let file_count = file_paths.len();
    
    // Create an index vector for chunking
    let indices: Vec<usize> = (0..file_count).collect();
    
    // Process in chunks for better cache locality
    const CHUNK_SIZE: usize = 64;
    
    // Process chunks in parallel and collect results
    let chunk_results: Vec<Vec<(usize, bool)>> = indices.par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            // Each thread creates its own results vector
            let mut local_results = Vec::with_capacity(chunk.len());
            
            for &i in chunk {
                let path = PathBuf::from(&file_paths[i]);
                let file_category = match categories[i] {
                    0 => FileCategory::Photo,
                    1 => FileCategory::Raw,
                    2 => FileCategory::Tiff,
                    3 => FileCategory::Video,
                    4 => FileCategory::Table,
                    _ => FileCategory::Unknown,
                };
                
                // Store result with index
                local_results.push((i, validate_file(&path, file_category)));
            }
            
            local_results
        })
        .collect();
    
    // Create the final results array
    let mut results = vec![false; file_count];
    
    // Combine all thread-local results
    for chunk in chunk_results {
        for (index, value) in chunk {
            results[index] = value;
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
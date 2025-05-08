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
type Signature = (&'static [u8], usize);
type SignatureList = Vec<Signature>;
type ExtensionMap = HashMap<Cow<'static, str>, SignatureList>;

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
        photo_signatures.insert(".jpg".into(), vec![(JPG_SIG, 0)]);
        photo_signatures.insert(".jpeg".into(), vec![(JPG_SIG, 0)]);
        photo_signatures.insert(".jfif".into(), vec![(JPG_SIG, 0)]);
        photo_signatures.insert(".jpe".into(), vec![(JPG_SIG, 0)]);
        photo_signatures.insert(".png".into(), vec![(PNG_SIG, 0)]);
        photo_signatures.insert(".bmp".into(), vec![(BMP_SIG, 0)]);
        photo_signatures.insert(".dib".into(), vec![(BMP_SIG, 0)]);
        photo_signatures.insert(".webp".into(), vec![(WEBP_SIG1, 0), (WEBP_SIG2, 8)]);
        photo_signatures.insert(".jp2".into(), vec![(JP2_SIG, 0)]);
        
        // Netpbm family (ASCII vs. raw variants)
        photo_signatures.insert(".pbm".into(), vec![(PBM_RAW_SIG, 0), (PBM_ASCII_SIG, 0)]);
        photo_signatures.insert(".pgm".into(), vec![(PGM_RAW_SIG, 0), (PGM_ASCII_SIG, 0)]);
        photo_signatures.insert(".ppm".into(), vec![(PPM_RAW_SIG, 0), (PPM_ASCII_SIG, 0)]);
        photo_signatures.insert(".pnm".into(), vec![(PNM_PXM_SIG, 0)]);
        photo_signatures.insert(".pxm".into(), vec![(PNM_PXM_SIG, 0)]);
        
        // Portable FloatMap (32‑bit float HDR)
        photo_signatures.insert(".pfm".into(), vec![(PFM_SIG1, 0), (PFM_SIG2, 0)]);   // colour / greyscale
        
        // Sun Raster / SR files
        photo_signatures.insert(".sr".into(),  vec![(SUN_RASTER_SIG, 0)]);
        photo_signatures.insert(".ras".into(), vec![(SUN_RASTER_SIG, 0)]);

        // Radiance HDR / PIC: ASCII "#?RADIANCE" (occasionally "#?RGBE")
        photo_signatures.insert(".hdr".into(), vec![(RADIANCE_HDR_SIG, 0), (RADIANCE_PIC_SIG, 0)]);
        photo_signatures.insert(".pic".into(), vec![(RADIANCE_HDR_SIG, 0), (RADIANCE_PIC_SIG, 0)]);

        // Initialize raw signatures
        raw_signatures.insert(".dng".into(), vec![(TIFF_LE_SIG, 0), (DNG_SIG, 0)]);
        raw_signatures.insert(".arw".into(), vec![(TIFF_LE_SIG, 0)]);
        raw_signatures.insert(".nef".into(), vec![(TIFF_LE_SIG, 0), (TIFF_BE_SIG, 0)]);
        raw_signatures.insert(".cr2".into(), vec![(CR2_SIG, 0)]);
        raw_signatures.insert(".crw".into(), vec![(CRW_SIG, 0)]);
        raw_signatures.insert(".raf".into(), vec![(FUJI_RAF_SIG, 0)]);
        raw_signatures.insert(".x3f".into(), vec![(X3F_SIG, 0)]);
        raw_signatures.insert(".orf".into(), vec![(ORF_SIG1, 0), (ORF_SIG2, 0)]);
        raw_signatures.insert(".erf".into(), vec![(TIFF_LE_SIG, 0), (TIFF_BE_SIG, 0)]);
        raw_signatures.insert(".kdc".into(), vec![(TIFF_LE_SIG, 0), (TIFF_BE_SIG, 0)]);
        raw_signatures.insert(".nrw".into(), vec![(TIFF_LE_SIG, 0)]);
        raw_signatures.insert(".pef".into(), vec![(TIFF_LE_SIG, 0)]);
        raw_signatures.insert(".raw".into(), vec![(TIFF_LE_SIG, 0), (TIFF_BE_SIG, 0)]);
        raw_signatures.insert(".sr2".into(), vec![(TIFF_LE_SIG, 0)]);
        raw_signatures.insert(".srw".into(), vec![(TIFF_LE_SIG, 0)]);
        raw_signatures.insert(".exr".into(), vec![(EXR_SIG, 0)]);
        
        // Initialize tiff signatures
        tiff_signatures.insert(".tiff".into(), vec![
            (TIFF_LE_SIG, 0),  // Little-endian TIFF
            (TIFF_BE_SIG, 0),  // Big-endian TIFF
        ]);
        tiff_signatures.insert(".tif".into(), vec![
            (TIFF_LE_SIG, 0),  // Little-endian TIFF
            (TIFF_BE_SIG, 0),  // Big-endian TIFF
        ]);

        // Initialize video signatures
        video_signatures.insert(".mp4".into(), vec![(MP4_SIG, 4)]);
        video_signatures.insert(".m4v".into(), vec![(MP4_SIG, 4)]);
        video_signatures.insert(".mov".into(), vec![(MP4_SIG, 4)]);
        video_signatures.insert(".avi".into(), vec![(AVI_SIG1, 0), (AVI_SIG2, 8)]);
        video_signatures.insert(".mkv".into(), vec![(MKV_SIG, 0)]);
        
        // Initialize table signatures
        table_signatures.insert(".xlsx".into(), vec![(ZIP_SIG, 0)]);
        table_signatures.insert(".xlsm".into(), vec![(ZIP_SIG, 0)]);
        table_signatures.insert(".xltx".into(), vec![(ZIP_SIG, 0)]);
        table_signatures.insert(".xltm".into(), vec![(ZIP_SIG, 0)]);
        table_signatures.insert(".parquet".into(), vec![(PARQUET_SIG1, 0), (PARQUET_SIG2, 0)]);
        
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

// Optimized helper function to check file signatures with static byte arrays
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
                        &buffer[*offset..*offset + signature.len()] == *signature
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
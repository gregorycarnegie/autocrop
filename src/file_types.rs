// File validation additions to lib.rs
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use phf::{phf_map, Map};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::{Read, BufRead, BufReader};

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
const JPEG_SIG: &[u8] = &[0xFF, 0xD8, 0xFF];
const PNG_SIG: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
const TIFF_LE_SIG: &[u8] = &[0x49, 0x49, 0x2A, 0x00];
const TIFF_BE_SIG: &[u8] = &[0x4D, 0x4D, 0x00, 0x2A];
const RIFF_SIG: &[u8] = &[0x52, 0x49, 0x46, 0x46]; // "RIFF"
const WEBP_SIG: &[u8] = &[0x57, 0x45, 0x42, 0x50]; // "WEBP"
const BMP_SIG: &[u8] = &[0x42, 0x4D];
const FTYP_SIG: &[u8] = &[b'f', b't', b'y', b'p']; // "ftyp"
const AVI_SIG: &[u8] = &[b'A', b'V', b'I', b' ']; // "AVI "
const MKV_SIG: &[u8] = &[0x1A, 0x45, 0xDF, 0xA3];
const ZIP_SIG: &[u8] = &[0x50, 0x4B, 0x03, 0x04]; // "PK\x03\x04"
const PARQUET_SIG1: &[u8] = &[b'P', b'A', b'R', b'1']; // "PAR1"
const PARQUET_SIG2: &[u8] = &[b'P', b'A', b'R', b'E']; // "PARE"

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

/// Validates a single file with improved memory usage and error handling
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
    
    // Get extension for analysis
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_ascii_lowercase())
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
            if let Some(possible_categories) = analyze_file_header(
                &header[..bytes_read], 
                &extension
            ) {
                // If our category is the only possibility, we can return true
                if possible_categories.len() == 1 && possible_categories[0] == category {
                    return true;
                }
                
                // If our category isn't in the possible list, we can return false immediately
                if !possible_categories.contains(&category) {
                    return false;
                }
                
                // If our category is one of multiple possibilities, continue to detailed checking
            }
        }
    }
    
    // Get signatures for the file category and extension
    if let Some(signatures) = get_signatures(path, category) {
        return check_file_signatures(path, signatures);
    }
    
    false
}

/// Analyze file header bytes to determine possible file categories
fn analyze_file_header(buffer: &[u8], extension: &str) -> Option<Vec<FileCategory>> {
    if buffer.len() < 4 {
        return None; // Not enough data
    }
    
    let mut possible_categories = Vec::with_capacity(2);
    
    // JPEG detection
    if buffer.starts_with(JPEG_SIG) {
        possible_categories.push(FileCategory::Photo);
    }
    // PNG detection
    else if buffer.len() >= 8 && buffer.starts_with(PNG_SIG) {
        possible_categories.push(FileCategory::Photo);
    }
    // TIFF or RAW detection
    else if buffer.starts_with(TIFF_LE_SIG) || buffer.starts_with(TIFF_BE_SIG) {
        // Check extension to disambiguate
        if extension == "tif" || extension == "tiff" {
            possible_categories.push(FileCategory::Tiff);
        } else if extension == "dng" || extension == "arw" || extension == "nef" || extension == "cr2" {
            possible_categories.push(FileCategory::Raw);
        } else {
            // Could be either
            possible_categories.push(FileCategory::Tiff);
            possible_categories.push(FileCategory::Raw);
        }
    }
    // WEBP detection
    else if buffer.len() >= 12 && 
        buffer.starts_with(RIFF_SIG) && 
        &buffer[8..12] == WEBP_SIG {
        possible_categories.push(FileCategory::Photo);
    }
    // BMP detection
    else if buffer.starts_with(BMP_SIG) {
        possible_categories.push(FileCategory::Photo);
    }
    // MP4/QuickTime container detection
    else if buffer.len() >= 8 && &buffer[4..8] == FTYP_SIG {
        possible_categories.push(FileCategory::Video);
    }
    // AVI detection
    else if buffer.len() >= 12 && 
        buffer.starts_with(RIFF_SIG) && 
        &buffer[8..12] == AVI_SIG {
        possible_categories.push(FileCategory::Video);
    }
    // Matroska (MKV) detection
    else if buffer.starts_with(MKV_SIG) {
        possible_categories.push(FileCategory::Video);
    }
    // ZIP-based formats (could be XLSX, etc.)
    else if buffer.starts_with(ZIP_SIG) {
        possible_categories.push(FileCategory::Table);
    }
    // Parquet format detection
    else if buffer.starts_with(PARQUET_SIG1) || 
            buffer.starts_with(PARQUET_SIG2) {
        possible_categories.push(FileCategory::Table);
    }
    
    if possible_categories.is_empty() {
        None
    } else {
        Some(possible_categories)
    }
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
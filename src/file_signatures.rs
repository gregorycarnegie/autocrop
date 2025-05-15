use phf::{phf_map, Map};

// Type aliases for better code readability
pub type Signature = (&'static [u8], usize);
type FileSignatureMap = Map<&'static str, &'static [Signature]>;

// Helper constants for common signature checks
pub const PNG_SIG: &[u8] = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

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
pub static PHOTO_SIGNATURES_MAP: FileSignatureMap = phf_map! {
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
pub static RAW_SIGNATURES_MAP: FileSignatureMap = phf_map! {
    "dng" => TIFF_LE_SIGNATURE,
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

pub static TIFF_SIGNATURES_MAP: FileSignatureMap = phf_map! {
    "tiff" => TIFF_SIGNATURES,
    "tif" => TIFF_SIGNATURES,
};

pub static VIDEO_SIGNATURES_MAP: FileSignatureMap = phf_map! {
    "mp4" => MP4_SIGNATURES,
    "m4v" => MP4_SIGNATURES,
    "mov" => MP4_SIGNATURES,
    "avi" => AVI_SIGNATURES,
    "mkv" => MKV_SIGNATURE,
};

pub static TABLE_SIGNATURES_MAP: FileSignatureMap = phf_map! {
    "xlsx" => ZIP_SIGNATURE,
    "xlsm" => ZIP_SIGNATURE,
    "xltx" => ZIP_SIGNATURE,
    "xltm" => ZIP_SIGNATURE,
    "parquet" => PARQUET_SIGNATURES,
};

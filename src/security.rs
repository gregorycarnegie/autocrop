// src/security.rs

use once_cell::sync::Lazy;
use pyo3::{exceptions::PyException, prelude::*};
use regex::Regex;
use std::collections::HashSet;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

use crate::ImportablePyModuleBuilder;

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;

#[pyclass(extends=PyException)]
#[derive(Debug)]
pub struct PathSecurityError {
    #[pyo3(get)]
    message: String,
}

#[pymethods]
impl PathSecurityError {
    #[new]
    fn new(message: String) -> Self {
        PathSecurityError { message }
    }

    fn __str__(&self) -> String {
        self.message.clone()
    }
}

static PATH_REGEX: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"([A-Z]:)?[/\\][^'"\s<>|?*\n]+"#).unwrap());

static ALLOWED_BASE_DIRS: OnceLock<Vec<PathBuf>> = OnceLock::new();

static DANGEROUS_NAMES: Lazy<Vec<&'static str>> = Lazy::new(|| {
    vec![
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
        "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ]
});

/// Sanitize a path string to prevent security vulnerabilities
#[pyfunction]
#[pyo3(signature = (path_str, allowed_operations=vec!["read".to_string(), "write".to_string()], max_path_length=4096, follow_symlinks=false))]
pub fn sanitize_path(
    path_str: &str,
    allowed_operations: Vec<String>,
    max_path_length: usize,
    follow_symlinks: bool,
) -> PyResult<String> {
    // Parse allowed operations
    let allowed_ops_set: HashSet<String> = allowed_operations.into_iter().collect();

    // 0. Pre-clean the path string - Remove quotation marks if they exist
    let cleaned_path_str = clean_path_string(path_str);

    // 1. Basic validation
    if cleaned_path_str.is_empty() {
        return Ok(String::new());
    }

    // 2. Length check
    if cleaned_path_str.len() > max_path_length {
        return Ok(String::new());
    }

    // 3. Remove null bytes and non-printable characters
    if cleaned_path_str.contains('\0') {
        return Err(PyErr::new::<PathSecurityError, _>(
            "Null byte detected in path",
        ));
    }

    let cleaned_path: String = cleaned_path_str
        .chars()
        .filter(|c| c.is_ascii_graphic() || c.is_whitespace())
        .collect();

    // 4. Create path and check for path traversal
    let path = Path::new(&cleaned_path);

    // Check for path traversal attempts
    for component in path.components() {
        if let Component::ParentDir = component {
            return Err(PyErr::new::<PathSecurityError, _>(
                "Path traversal attempted",
            ));
        }
    }

    // 5. Resolve the path
    let resolved_path = match follow_symlinks {
        true => resolve_path_following_symlinks(path)?,
        false => resolve_path_no_symlinks(path)?,
    };

    // 6. Validate against allowed base directories
    if !is_within_allowed_directories(&resolved_path, follow_symlinks)? {
        return Ok(String::new());
    }

    // 7. Check for dangerous path components (Windows-specific)
    if cfg!(target_os = "windows") && has_dangerous_windows_components(&resolved_path) {
        return Ok(String::new());
    }

    // 8. Check permissions if path exists
    if resolved_path.exists() {
        check_permissions(&resolved_path, &allowed_ops_set)?;
    }

    // 9. Additional security checks
    if !validate_path_components(&resolved_path)? {
        return Ok(String::new());
    }

    // 10. Convert to string and return
    match resolved_path.to_str() {
        Some(path_str) => Ok(path_str.to_string()),
        None => Ok(String::new()),
    }
}

/// Clean a path string by removing surrounding quotes and normalizing separators
/// This function is recursive to handle nested quotes
fn clean_path_string(path_str: &str) -> String {
    let mut result = path_str.to_string();
    const MAX_ITERATIONS: usize = 10; // Reasonable upper limit

    for _ in 0..MAX_ITERATIONS {
        if result.len() < 2 {
            break;
        }

        let first = result.chars().next().unwrap();
        let last = result.chars().last().unwrap();

        if (first == '\'' && last == '\'') ^ (first == '"' && last == '"') {
            result = result[1..result.len() - 1].to_string();
        } else {
            break;
        }
    }
    // Always normalize path separators
    result.replace('\\', "/")
}

/// Resolve path following symlinks
fn resolve_path_following_symlinks(path: &Path) -> PyResult<PathBuf> {
    match fs::canonicalize(path) {
        Ok(resolved) => Ok(resolved),
        Err(_) => {
            // If the path doesn't exist, just normalize it
            Ok(path.to_path_buf())
        }
    }
}

/// Resolve path without following symlinks
fn resolve_path_no_symlinks(path: &Path) -> PyResult<PathBuf> {
    // Check if the final path is a symlink
    if path.exists() && fs::symlink_metadata(path)?.file_type().is_symlink() {
        return Err(PyErr::new::<PathSecurityError, _>(
            "Symbolic links not allowed",
        ));
    }

    // For non-existent paths, we can't check for symlinks along the way
    // So we'll do a simple normalization
    let mut resolved = PathBuf::new();

    for component in path.components() {
        match component {
            Component::CurDir => continue,
            Component::ParentDir => {
                resolved.pop();
            }
            _ => resolved.push(component),
        }
    }

    // Check each existing component for symlinks
    let mut check_path = PathBuf::new();
    for component in resolved.components() {
        check_path.push(component);

        if check_path.exists()
            && let Ok(metadata) = fs::symlink_metadata(&check_path)
            && metadata.file_type().is_symlink()
        {
            return Err(PyErr::new::<PathSecurityError, _>(format!(
                "Symbolic link detected: {:?}",
                component
            )));
        }
    }

    Ok(resolved)
}

/// Check if path is within allowed directories
fn is_within_allowed_directories(path: &Path, follow_symlinks: bool) -> PyResult<bool> {
    let allowed_dirs =
        ALLOWED_BASE_DIRS.get_or_init(|| get_allowed_base_directories().unwrap_or_default());

    for allowed_dir in allowed_dirs {
        if is_safe_subpath(path, allowed_dir, follow_symlinks) {
            return Ok(true);
        }
    }

    Ok(false)
}

/// Get home directory using platform-specific methods
fn get_home_dir() -> Option<PathBuf> {
    #[cfg(unix)]
    {
        std::env::var_os("HOME").map(PathBuf::from)
    }

    #[cfg(windows)]
    {
        std::env::var_os("USERPROFILE")
            .map(PathBuf::from) // Convert OsString to PathBuf here
            .or_else(|| {
                // Fallback to HOMEDRIVE + HOMEPATH on Windows
                match (std::env::var_os("HOMEDRIVE"), std::env::var_os("HOMEPATH")) {
                    (Some(drive), Some(path)) => {
                        let mut home = PathBuf::from(drive);
                        home.push(path);
                        Some(home)
                    }
                    _ => None,
                }
            })
    }

    #[cfg(not(any(unix, windows)))]
    {
        None
    }
}

/// Get allowed base directories for the platform
fn get_allowed_base_directories() -> PyResult<Vec<PathBuf>> {
    let mut allowed_dirs = Vec::new();

    // Add home directory
    if let Some(home_dir) = get_home_dir() {
        allowed_dirs.push(home_dir);
    }

    // Platform-specific directories
    if cfg!(target_os = "windows") {
        // Add all drive roots
        for letter in b'A'..=b'Z' {
            let drive = format!("{}:", letter as char);
            let drive_path = PathBuf::from(drive);
            if drive_path.exists() {
                allowed_dirs.push(drive_path);
            }
        }
    } else {
        // Unix-like systems
        allowed_dirs.extend_from_slice(&[
            PathBuf::from("/mnt"),
            PathBuf::from("/media"),
            PathBuf::from("/tmp"),
            PathBuf::from("/var/tmp"),
        ]);
    }

    Ok(allowed_dirs)
}

/// Check if path is a safe subpath of base_path
fn is_safe_subpath(path: &Path, base_path: &Path, follow_symlinks: bool) -> bool {
    if follow_symlinks {
        match (path.canonicalize(), base_path.canonicalize()) {
            (Ok(abs_path), Ok(abs_base)) => abs_path.starts_with(abs_base),
            _ => false,
        }
    } else {
        // No symlink following - do manual path comparison
        let abs_path = if path.is_absolute() {
            path.to_path_buf()
        } else {
            match std::env::current_dir() {
                Ok(cwd) => cwd.join(path),
                Err(_) => return false,
            }
        };

        let abs_base = if base_path.is_absolute() {
            base_path.to_path_buf()
        } else {
            match std::env::current_dir() {
                Ok(cwd) => cwd.join(base_path),
                Err(_) => return false,
            }
        };

        abs_path.starts_with(abs_base)
    }
}

/// Check for dangerous Windows path components
fn has_dangerous_windows_components(path: &Path) -> bool {
    for component in path.components() {
        if let Component::Normal(name) = component
            && let Some(name_str) = name.to_str()
        {
            // Get the base name without extension
            let base_name = name_str.split('.').next().unwrap_or(name_str);

            // Check against dangerous names (case-insensitive)
            for &dangerous in DANGEROUS_NAMES.iter() {
                if base_name.eq_ignore_ascii_case(dangerous) {
                    return true;
                }
            }
        }
    }

    false
}

/// Check file permissions
fn check_permissions(path: &Path, allowed_operations: &HashSet<String>) -> PyResult<()> {
    // On Unix systems, check the mode
    #[cfg(unix)]
    {
        let metadata = fs::metadata(path)?;
        let permissions = metadata.permissions();
        let mode = permissions.mode();

        // Check for world-writable files (potential security risk)
        if mode & 0o002 != 0 {
            // Log warning but don't fail
            eprintln!("Warning: World-writable path detected: {:?}", path);
        }
    }

    // Check specific permissions based on allowed operations
    if allowed_operations.contains("read") && !can_read(path) {
        return Err(PyErr::new::<PathSecurityError, _>("No read permission"));
    }

    if allowed_operations.contains("write") && !can_write(path) {
        return Err(PyErr::new::<PathSecurityError, _>("No write permission"));
    }

    if allowed_operations.contains("execute") && !can_execute(path) {
        return Err(PyErr::new::<PathSecurityError, _>("No execute permission"));
    }

    Ok(())
}

/// Check if the current process can read from the path
fn can_read(path: &Path) -> bool {
    if path.is_dir() {
        // Can we enumerate entries?
        fs::read_dir(path).is_ok()
    } else {
        // Regular file open
        fs::File::open(path).is_ok()
    }
}

/// Check if the current process can write to the path
fn can_write(path: &Path) -> bool {
    if path.is_dir() {
        // Try creating (and then removing) a temp file in the directory
        let test_file = path.join(".perm_test");
        let result = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&test_file)
            .and_then(|f| {
                drop(f);
                fs::remove_file(&test_file)
            });
        result.is_ok()
    } else {
        // Normal files: can we open for write?
        fs::OpenOptions::new().write(true).open(path).is_ok()
    }
}

/// Check if the current process can execute the path
fn can_execute(path: &Path) -> bool {
    #[cfg(unix)]
    {
        if let Ok(metadata) = fs::metadata(path) {
            let mode = metadata.permissions().mode();
            // Check if any execute bit is set
            return mode & 0o111 != 0;
        }
    }

    // On Windows, check if it's an executable file
    #[cfg(windows)]
    {
        if let Some(ext) = path.extension() {
            let exe_extensions = ["exe", "bat", "cmd", "com"];
            return exe_extensions.iter().any(|&e| ext.eq_ignore_ascii_case(e));
        }
    }

    false
}

/// Validate individual path components
fn validate_path_components(path: &Path) -> PyResult<bool> {
    for component in path.components() {
        if let Component::Normal(name) = component
            && let Some(name_str) = name.to_str()
        {
            // Check for excessively long components
            if name_str.len() > 255 {
                return Ok(false);
            }

            // Check for suspicious characters
            let suspicious_chars = if cfg!(target_os = "windows") {
                "<>:\"|?*"
            } else {
                "<>|?*"
            };

            if name_str.chars().any(|c| suspicious_chars.contains(c)) {
                eprintln!(
                    "Warning: Suspicious character in path component: {}",
                    name_str
                );
                // Don't fail, just warn
            }
        }
    }

    Ok(true)
}

/// Helper function to get safe error messages without exposing paths
#[pyfunction]
pub fn get_safe_error_message(error_msg: &str) -> String {
    // Use cached regex to remove absolute paths from error messages
    PATH_REGEX.replace_all(error_msg, "<path>").to_string()
}

/// Module initialization
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let builder = ImportablePyModuleBuilder::from(m.clone())?;

    // Add everything to the module in a single chain
    builder
        .add_function(wrap_pyfunction!(sanitize_path, m)?)?
        .add_function(wrap_pyfunction!(get_safe_error_message, m)?)?
        .add_class::<PathSecurityError>()?;

    Ok(())
}

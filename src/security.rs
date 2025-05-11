// src/security.rs

use pyo3::{prelude::*, exceptions::PyException};
use std::path::{Path, PathBuf, Component};
use std::fs;
use std::collections::HashSet;
use regex;

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

/// Sanitize a path string to prevent security vulnerabilities
#[pyfunction]
#[pyo3(signature = (path_str, allowed_operations=None, max_path_length=4096, follow_symlinks=false))]
pub fn sanitize_path(
    path_str: &str,
    allowed_operations: Option<Vec<String>>,
    max_path_length: usize,
    follow_symlinks: bool,
) -> PyResult<Option<String>> {
    // Parse allowed operations
    let allowed_ops = allowed_operations.unwrap_or_else(|| vec!["read".to_string()]);
    let allowed_ops_set: HashSet<String> = allowed_ops.into_iter().collect();
    
    // 1. Basic validation
    if path_str.is_empty() {
        return Ok(None);
    }
    
    // 2. Length check
    if path_str.len() > max_path_length {
        return Ok(None);
    }
    
    // 3. Remove null bytes and non-printable characters
    if path_str.contains('\0') {
        return Err(PyErr::new::<PathSecurityError, _>("Null byte detected in path"));
    }
    
    let cleaned_path: String = path_str
        .chars()
        .filter(|c| c.is_ascii_graphic() || c.is_whitespace())
        .collect();
    
    // 4. Create path and check for path traversal
    let path = Path::new(&cleaned_path);
    
    // Check for path traversal attempts
    for component in path.components() {
        if let Component::ParentDir = component {
            return Err(PyErr::new::<PathSecurityError, _>("Path traversal attempted"));
        }
    }
    
    // 5. Resolve the path
    let resolved_path = match follow_symlinks {
        true => resolve_path_following_symlinks(path)?,
        false => resolve_path_no_symlinks(path)?,
    };
    
    // 6. Validate against allowed base directories
    if !is_within_allowed_directories(&resolved_path)? {
        return Ok(None);
    }
    
    // 7. Check for dangerous path components (Windows-specific)
    if cfg!(target_os = "windows") && has_dangerous_windows_components(&resolved_path) {
        return Ok(None);
    }
    
    // 8. Check permissions if path exists
    if resolved_path.exists() {
        check_permissions(&resolved_path, &allowed_ops_set)?;
    }
    
    // 9. Additional security checks
    if !validate_path_components(&resolved_path)? {
        return Ok(None);
    }
    
    // 10. Convert to string and return
    match resolved_path.to_str() {
        Some(path_str) => Ok(Some(path_str.to_string())),
        None => Ok(None),
    }
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
    // First check if it's a symlink
    if path.exists() && fs::symlink_metadata(path)?.file_type().is_symlink() {
        return Err(PyErr::new::<PathSecurityError, _>("Symbolic links not allowed"));
    }
    
    // Manually resolve the path component by component
    let mut resolved = PathBuf::new();
    
    // Handle absolute vs relative paths
    if path.is_absolute() {
        resolved.push("/");
    }
    
    for component in path.components() {
        match component {
            Component::Prefix(prefix) => resolved.push(prefix.as_os_str()),
            Component::RootDir => resolved.push("/"),
            Component::CurDir => continue,
            Component::ParentDir => {
                resolved.pop();
            }
            Component::Normal(name) => {
                resolved.push(name);
                // Check if this component is a symlink
                if resolved.exists() && fs::symlink_metadata(&resolved)?.file_type().is_symlink() {
                    return Err(PyErr::new::<PathSecurityError, _>(
                        format!("Symbolic link detected: {:?}", name)
                    ));
                }
            }
        }
    }
    
    Ok(resolved)
}

/// Check if path is within allowed directories
fn is_within_allowed_directories(path: &Path) -> PyResult<bool> {
    let allowed_dirs = get_allowed_base_directories()?;
    
    for allowed_dir in allowed_dirs {
        if is_safe_subpath(path, &allowed_dir) {
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
            .map(PathBuf::from)  // Convert OsString to PathBuf here
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
fn is_safe_subpath(path: &Path, base_path: &Path) -> bool {
    match (path.canonicalize(), base_path.canonicalize()) {
        (Ok(abs_path), Ok(abs_base)) => abs_path.starts_with(abs_base),
        _ => false,
    }
}

/// Check for dangerous Windows path components
fn has_dangerous_windows_components(path: &Path) -> bool {
    const DANGEROUS_NAMES: &[&str] = &[
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    
    for component in path.components() {
        if let Component::Normal(name) = component {
            if let Some(name_str) = name.to_str() {
                // Get the base name without extension
                let base_name = name_str.split('.').next().unwrap_or(name_str);
                
                // Check against dangerous names (case-insensitive)
                for &dangerous in DANGEROUS_NAMES {
                    if base_name.eq_ignore_ascii_case(dangerous) {
                        return true;
                    }
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
        fs::OpenOptions::new()
            .write(true)
            .open(path)
            .is_ok()
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
        if let Component::Normal(name) = component {
            if let Some(name_str) = name.to_str() {
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
                    eprintln!("Warning: Suspicious character in path component: {}", name_str);
                    // Don't fail, just warn
                }
            }
        }
    }
    
    Ok(true)
}

/// Helper function to get safe error messages without exposing paths
#[pyfunction]
pub fn get_safe_error_message(error_msg: &str) -> String {
    // Use regex to remove absolute paths from error messages
    let re = regex::Regex::new(r#"([A-Z]:)?[/\\][^'"\s<>|?*\n]+"#).unwrap();
    re.replace_all(error_msg, "<path>").to_string()
}

/// Module initialization
pub fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sanitize_path, m)?)?;
    m.add_function(wrap_pyfunction!(get_safe_error_message, m)?)?;
    m.add_class::<PathSecurityError>()?;
    
    Ok(())
}
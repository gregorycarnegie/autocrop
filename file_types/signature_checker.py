from pathlib import Path

from autocrop_rs import verify_file_type

from .file_category import FileCategory

class SignatureChecker:
    """
    A utility class for detecting file types based on binary signatures/magic numbers.
    Uses Rust-accelerated file verification for performance.
    """
    
    @classmethod
    def verify_file_type(cls, file_path: Path, expected_category: FileCategory) -> bool:
        """
        Comprehensive file type verification using Rust implementation.
        
        Args:
            file_path: Path to the file to verify
            expected_category: Expected file category
            
        Returns:
            bool: True if the file content matches the expected category
        """
        if not file_path.exists() or not file_path.is_file():
            return False
            
        # Convert FileCategory to numeric value for Rust
        category_value = {
            FileCategory.PHOTO: 0,
            FileCategory.RAW: 1,
            FileCategory.TIFF: 2,
            FileCategory.VIDEO: 3,
            FileCategory.TABLE: 4,
        }.get(expected_category, 5)
        
        # Call Rust implementation
        try:
            return verify_file_type(str(file_path), category_value)
        except Exception:
            # Fallback to always returning False on errors
            return False

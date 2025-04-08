"""
Utilities for file signature detection and validation.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

from .file_type_manager import FileCategory, FileTypeManager


class SignatureChecker:
    """
    A utility class for detecting file types based on binary signatures/magic numbers.
    Only used when more accurate type detection is needed beyond extensions.
    """
    # Common file signatures as (signature bytes, offset)
    _SIGNATURES: Dict[FileCategory, Dict[str, List[Tuple[bytes, int]]]] = {
        FileCategory.PHOTO: {
            '.jpg': [(b'\xFF\xD8\xFF', 0)],
            '.jpeg': [(b'\xFF\xD8\xFF', 0)],
            '.jfif': [(b'\xFF\xD8\xFF', 0)],
            '.jpe': [(b'\xFF\xD8\xFF', 0)],
            '.png': [(b'\x89PNG\r\n\x1A\n', 0)],
            '.bmp': [(b'BM', 0)],
            '.dib': [(b'BM', 0)],
            '.webp': [(b'RIFF', 0), (b'WEBP', 8)],
        },
        FileCategory.TIFF: {
            '.tiff': [(b'\x49\x49\x2A\x00', 0), (b'\x4D\x4D\x00\x2A', 0)],
            '.tif': [(b'\x49\x49\x2A\x00', 0), (b'\x4D\x4D\x00\x2A', 0)],
        },
        FileCategory.RAW: {
            '.dng': [(b'\x49\x49\x2A\x00', 0)],
            '.cr2': [(b'\x49\x49\x2A\x00\x10\x00\x00\x00\x43\x52', 0)],
            '.arw': [(b'\x49\x49\x2A\x00', 0)],
        },
        FileCategory.VIDEO: {
            '.mp4': [(b'ftyp', 4)],
            '.m4v': [(b'ftyp', 4)],
            '.mov': [(b'ftyp', 4), (b'moov', 4), (b'mdat', 4)],
            '.avi': [(b'RIFF', 0), (b'AVI ', 8)],
            '.mkv': [(b'\x1A\x45\xDF\xA3', 0)],
        },
    }
    
    # Extensions without reliable signatures (text-based formats)
    _TEXT_EXTENSIONS: Set[str] = {'.csv'}
    
    @classmethod
    def check_signature(cls, file_path: Path, expected_category: Optional[FileCategory] = None) -> bool:
        """
        Check if a file has a valid signature corresponding to its extension or expected category.
        
        Args:
            file_path: Path to the file to check
            expected_category: Optional expected file category to validate against
            
        Returns:
            bool: True if the file signature matches the expected type
        """
        if not file_path.exists() or not file_path.is_file():
            return False

        # Get file extension
        extension = file_path.suffix.lower()

        # Special case for text-based formats
        if extension in cls._TEXT_EXTENSIONS:
            return cls._validate_text_file(file_path)

        # Determine category to check
        category = expected_category
        if category is None:
            category = FileTypeManager.detect_category(file_path)

        if category == FileCategory.UNKNOWN:
            return False

        if signatures := cls._get_signatures(category, extension):
            # Check file against known signatures
            return cls._check_file_signatures(file_path, signatures)
        else:
            # No signatures defined, can't validate beyond extension
            return True
    
    @classmethod
    def _get_signatures(cls, category: FileCategory, extension: str) -> List[Tuple[bytes, int]]:
        """Get signatures for a given category and extension."""
        if category not in cls._SIGNATURES:
            return []
            
        category_sigs = cls._SIGNATURES[category]
        return category_sigs.get(extension, [])
    
    @staticmethod
    def _validate_text_file(file_path: Path) -> bool:
        """
        Simple validation for text-based files like CSV.
        Just checks that the file is readable as text.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to read a small amount to check if it's text
                f.read(1024)
            return True
        except UnicodeDecodeError:
            # Not a valid text file
            return False
    
    @staticmethod
    def _check_file_signatures(file_path: Path, signatures: List[Tuple[bytes, int]]) -> bool:
        """Check if file matches any of the provided signatures."""
        try:
            # Read enough bytes for signature checking
            max_offset = max(offset + len(sig) for sig, offset in signatures)
            with open(file_path, 'rb') as f:
                header = f.read(max_offset)

            return any(
                len(header) >= offset + len(signature)
                and header[offset : offset + len(signature)] == signature
                for signature, offset in signatures
            )
        except Exception:
            return False

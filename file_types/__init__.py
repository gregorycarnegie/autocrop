"""
File type handling package for the application.
Provides utilities for file type detection, validation, and filtering.
"""
from .file_category import FileCategory
from .file_type_manager import FileTypeManager, file_manager
from .signature_checker import SignatureChecker


__all__ = [
    'FileCategory',
    'FileTypeManager',
    'SignatureChecker',
    'file_manager',
]

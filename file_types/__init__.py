"""
File type handling package for the application.
Provides utilities for file type detection, validation, and filtering.
"""
from .file_type_manager import FileCategory, FileTypeManager, file_manager
from .signature_checker import SignatureChecker


__all__ = [
    'FileCategory',
    'FileTypeManager',
    'SignatureChecker',
    'file_manager',
]

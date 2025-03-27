from .registry import FileRegistry, FileTypeInfo
from .register import initialize_file_types

# Re-export the FileRegistry for easier access
registry = FileRegistry

__all__ = ['FileRegistry', 'FileTypeInfo', 'registry']

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Optional

@dataclass
class FileTypeInfo:
    """
    Stores metadata about a file type category.
    
    Attributes:
        extensions: Set of file extensions (with leading dot)
        default_dir: Default directory for this file type
        can_open: Whether files of this type can be opened
        can_save: Whether files of this type can be saved
        mime_types: Optional set of MIME types
        description: Human-readable description of the file type
    """
    extensions: Set[str]
    default_dir: Path
    can_open: bool = True
    can_save: bool = True
    mime_types: Optional[Set[str]] = None
    save_types: tuple[str] = ('.bmp', '.jpg', '.png', '.tiff', '.webp')
    description: str = ""

class FileRegistry:
    """
    Central registry for file type information.
    
    This class provides methods to register, access, and validate file types,
    as well as generate filter strings for file dialogs.
    """
    _types: Dict[str, FileTypeInfo] = {}
    
    @classmethod
    def register_type(cls, name: str, info: FileTypeInfo) -> None:
        """Register a new file type or update an existing one."""
        cls._types[name] = info
    
    @classmethod
    def get_type_info(cls, name: str) -> Optional[FileTypeInfo]:
        """Get information about a specific file type by name."""
        return cls._types.get(name)
    
    @classmethod
    def get_all_extensions(cls) -> Set[str]:
        """Get a set of all registered file extensions."""
        all_exts = set()
        for info in cls._types.values():
            all_exts.update(info.extensions)
        return all_exts
    
    @classmethod
    def get_extensions(cls, type_name: str) -> Set[str]:
        """Get all extensions for a specific file type."""
        return info.extensions if (info := cls._types.get(type_name)) else set()
    
    @classmethod
    def get_default_dir(cls, type_name: str) -> Path:
        """Get the default directory for a specific file type."""
        return info.default_dir if (info := cls._types.get(type_name)) else Path.home()
    
    @classmethod
    def is_valid_type(cls, path: Path, type_name: str) -> bool:
        """Check if a path matches a specific file type."""
        if not path.is_file() or type_name not in cls._types:
            return False
        return path.suffix.lower() in cls._types[type_name].extensions
    
    @classmethod
    def get_type_for_path(cls, path: Path) -> Optional[str]:
        """Determine the file type for a given path."""
        if not path.is_file():
            return None

        ext = path.suffix.lower()
        return next(
            (
                type_name
                for type_name, info in cls._types.items()
                if ext in info.extensions
            ),
            None,
        )
    
    @classmethod
    def get_filter_string(cls, type_name: Optional[str] = None) -> str:
        """
        Generate a filter string for file dialogs.
        
        If type_name is provided, returns a filter string for that type only.
        Otherwise, returns a combined filter string for all registered types.
        """
        if type_name and type_name in cls._types:
            info = cls._types[type_name]
            filters = [f'*{ext}' for ext in info.extensions]
            filter_text = f"{type_name.title()} Files ({' '.join(filters)})"
            return f"All Files (*);;{filter_text}"
        
        # Return all filters
        all_filters = ["All Files (*)"]
        for name, info in cls._types.items():
            filters = [f'*{ext}' for ext in info.extensions]
            all_filters.append(f"{name.title()} Files ({' '.join(filters)})")
        
        return ";;".join(all_filters)
    
    @classmethod
    def should_use_tiff_save(cls, path: Path) -> bool:
        """Determine if TIFF saving should be used based on the file extension."""
        return path.suffix.lower() in cls.get_extensions("tiff")

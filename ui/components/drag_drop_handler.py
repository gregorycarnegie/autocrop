import os
from pathlib import Path

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from PyQt6.QtWidgets import QStatusBar

from core.config import logger
from core.enums import FunctionType
from file_types import FileCategory, SignatureChecker, file_manager
from ui import utils as ut
from ui.components.tab_manager import TabManager


class DragDropHandler(QObject):
    """Handles drag and drop events for the main window"""

    # Signals
    file_dropped = pyqtSignal(Path, FunctionType)  # file_path, target_tab
    directory_dropped = pyqtSignal(Path, FunctionType)  # dir_path, target_tab

    def __init__(self, tab_manager: TabManager, status_bar: QStatusBar):
        super().__init__()
        self.tab_manager = tab_manager
        self.status_bar = status_bar

    def handle_drag_enter(self, event: QDragEnterEvent):
        """Handle drag enter events for browser-like drag and drop"""
        try:
            assert isinstance(event, QDragEnterEvent)
        except AssertionError:
            return

        ut.check_mime_data(event)
        # Show a status message
        self.status_bar.showMessage("Drop files here to open", 2000)

    def handle_drag_move(self, event: QDragMoveEvent):
        """Handle drag move events"""
        try:
            assert isinstance(event, QDragMoveEvent)
        except AssertionError:
            return

        ut.check_mime_data(event)

    def handle_drop(self, event: QDropEvent):
        """Handle drop events with enhanced security"""
        try:
            assert isinstance(event, QDropEvent)
        except AssertionError:
            return

        if (mime_data := event.mimeData()) is None:
            return

        if not mime_data.hasUrls():
            event.ignore()
            return

        event.setDropAction(Qt.DropAction.CopyAction)

        # Get the dropped URL and convert to a local file path
        try:
            url = mime_data.urls()[0]
            if not url.isLocalFile():
                ut.show_error_box("Only local files can be dropped")
                event.ignore()
                return
        except IndexError:
            ut.show_error_box("File may be corrupted or not accessible")
            event.ignore()
            return

        file_path_str = url.toLocalFile()

        # Validate the path with our improved sanitize_path function
        if not (safe_path_str := ut.sanitize_path(file_path_str)):
            event.ignore()
            return

        # Create a Path object from the sanitized path
        file_path = Path(safe_path_str).resolve()

        # Handle the file based on its type with proper error handling
        try:
            if file_path.is_dir():
                self._handle_dropped_directory(file_path)
            elif file_path.is_file():
                self._handle_dropped_file(file_path)
            else:
                ut.show_error_box("Dropped item is neither a file nor a directory")
                event.ignore()
                return

            event.accept()
            self.status_bar.showMessage(f"Opened {file_path.name}", 2000)

        except Exception as e:
            # Log error internally without exposing details
            logger.exception(f"Error processing dropped file: {e}")
            ut.show_error_box("An error occurred processing the dropped item")
            event.ignore()

    def _handle_dropped_directory(self, dir_path: Path):
        """Securely handle a dropped directory"""
        # Verify it's a valid directory
        if not dir_path.is_dir() or not os.access(dir_path, os.R_OK):
            ut.show_error_box("Directory is not accessible")
            return

        # Check directory contents to determine appropriate tab
        try:
            # Look for table files to determine if this is a mapping operation
            has_table_files = any(
                file_manager.is_valid_type(f, FileCategory.TABLE)
                for f in dir_path.iterdir()
                if f.is_file() and os.access(f, os.R_OK)
            )

            if has_table_files:
                # Handle as mapping tab
                self.directory_dropped.emit(dir_path, FunctionType.MAPPING)
            else:
                # Handle as folder tab
                self.directory_dropped.emit(dir_path, FunctionType.FOLDER)

        except OSError as e:
            # Log error internally without exposing details
            logger.exception(f"Error handling dropped directory: {e}")
            ut.show_error_box("An error occurred processing the directory")

    def _handle_dropped_file(self, file_path: Path):
        """Securely handle a dropped file"""
        # Verify it's a valid file
        if not file_path.is_file() or not os.access(file_path, os.R_OK):
            ut.show_error_box("File is not accessible")
            return

        try:
            # Determine the file type and handle accordingly
            if file_manager.is_valid_type(file_path, FileCategory.PHOTO):
                # Photo file - verify PHOTO signature
                if SignatureChecker.verify_file_type(file_path, FileCategory.PHOTO):
                    self.file_dropped.emit(file_path, FunctionType.PHOTO)
                else:
                    ut.show_error_box("File content doesn't match its extension (expected photo)")

            elif file_manager.is_valid_type(file_path, FileCategory.RAW):
                # RAW file - verify RAW signature
                if SignatureChecker.verify_file_type(file_path, FileCategory.RAW):
                    self.file_dropped.emit(file_path, FunctionType.PHOTO)
                else:
                    ut.show_error_box("File content doesn't match its extension (expected RAW)")

            elif file_manager.is_valid_type(file_path, FileCategory.TIFF):
                # TIFF file - verify TIFF signature
                if SignatureChecker.verify_file_type(file_path, FileCategory.TIFF):
                    self.file_dropped.emit(file_path, FunctionType.PHOTO)
                else:
                    ut.show_error_box("File content doesn't match its extension (expected TIFF)")

            elif file_manager.is_valid_type(file_path, FileCategory.VIDEO):
                # Video file - verify VIDEO signature
                if SignatureChecker.verify_file_type(file_path, FileCategory.VIDEO):
                    self.file_dropped.emit(file_path, FunctionType.VIDEO)
                else:
                    ut.show_error_box("File content doesn't match its extension (expected video)")

            elif file_manager.is_valid_type(file_path, FileCategory.TABLE):
                # Table file - verify TABLE signature
                if SignatureChecker.verify_file_type(file_path, FileCategory.TABLE):
                    self.file_dropped.emit(file_path, FunctionType.MAPPING)
                else:
                    ut.show_error_box("File content doesn't match its extension (expected table)")

            else:
                ut.show_error_box(f"Unsupported file type: {file_path.suffix}")

        except (OSError, ValueError, TypeError) as e:
            # Log error internally without exposing details
            logger.exception(f"Error handling dropped file: {e}")
            ut.show_error_box("An error occurred processing the file")

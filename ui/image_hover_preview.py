from pathlib import Path

import cv2
import cv2.typing as cvt
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap

from core import processing as prc
from core.face_tools import FaceToolPair
from core.job import Job
from file_types import FileCategory, file_manager


class ImageHoverPreview(QtWidgets.QLabel):
    """A tooltip-style widget that shows a preview image when hovering over files"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            QtCore.Qt.WindowType.Tool |
            QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.WindowStaysOnTopHint |
            QtCore.Qt.WindowType.WindowDoesNotAcceptFocus  # Add this to prevent focus issues
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        # Add this attribute to ensure proper rendering
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_PaintOnScreen, False)

        # Set a larger size for preview
        self.setFixedSize(250, 250)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 240);
                border: 2px solid #555;
                border-radius: 8px;
                padding: 5px;
            }
        """)

        self.setScaledContents(False)
        self.hide()

        # Timer to delay showing the preview - increased delay
        self.show_timer = QtCore.QTimer()
        self.show_timer.setSingleShot(True)
        self.show_timer.timeout.connect(self._actually_show)  # Changed to internal method

        # Cache for loaded images
        self.image_cache = {}
        self.current_path = None
        self.no_face_message = "No face detected"
        self.pending_mouse_pos = None

        print("ImageHoverPreview initialized")  # Debug

    def preview_file(self, file_path: str, mouse_pos: QtCore.QPoint,
                    face_tools: FaceToolPair, job: Job):
        """Load and show a preview of an image file"""
        print(f"preview_file called with: {file_path}")  # Debug

        if not file_path or file_path == self.current_path:
            print(f"Skipping preview - no path or same path: {file_path}")  # Debug
            return

        self.current_path = file_path
        self.pending_mouse_pos = mouse_pos

        # Check if image is in cache
        if file_path in self.image_cache:
            print(f"Using cached image for: {file_path}")  # Debug
            cached_item = self.image_cache[file_path]
            if cached_item == "no_face":
                self._show_no_face_message()
            elif cached_item == "error":
                self._show_error_message("Error loading image")
            else:
                self.setPixmap(cached_item)
            self._show_at_position(mouse_pos)
            return

        # Load image with a slight delay to avoid flickering
        print(f"Loading new image: {file_path}")  # Debug
        self.show_timer.start(200)  # Reduced delay for testing
        QtCore.QTimer.singleShot(0, lambda: self._load_image(file_path, mouse_pos, face_tools, job))

    def _actually_show(self):
        """Actually show the preview after timer expires"""
        if self.pending_mouse_pos and self.pixmap() and not self.pixmap().isNull():
            print("Timer expired, showing preview")  # Debug
            self._show_at_position(self.pending_mouse_pos)
        else:
            print("Timer expired but no valid preview to show")  # Debug

    def _show_at_position(self, mouse_pos: QtCore.QPoint):
        """Show the preview at the specified position"""
        print(f"_show_at_position called at: {mouse_pos}")  # Debug
        self.position_near_mouse(mouse_pos)
        self.show()
        self.raise_()  # Ensure it's on top
        self.activateWindow()  # Try to bring to front
        print(f"Preview should be visible now. isVisible: {self.isVisible()}")  # Debug

    def _load_image(self, file_path: str, mouse_pos: QtCore.QPoint,
                    face_tools: FaceToolPair, job: Job):
        """Load and process an image"""
        print(f"_load_image called for: {file_path}")  # Debug

        path = Path(file_path)
        if not path.exists():
            print(f"Path does not exist: {file_path}")  # Debug
            return

        try:
            # Load image at low resolution for performance
            if file_manager.is_valid_type(path, FileCategory.PHOTO) or \
               file_manager.is_valid_type(path, FileCategory.TIFF):
                image = cv2.imread(path.as_posix())
                print(f"Loaded image with cv2.imread: {image is not None}")  # Debug
            else:
                print(f"Unsupported file type: {file_path}")  # Debug
                return

            if image is None:
                print(f"Failed to load image: {file_path}")  # Debug
                return

            # Resize for preview (small size for performance)
            height, width = image.shape[:2]
            print(f"Original image size: {width}x{height}")  # Debug

            scale = min(300 / width, 300 / height, 1.0)
            if scale < 1.0:
                new_size = (int(width * scale), int(height * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
                print(f"Resized to: {new_size}")  # Debug

            # Create a preview job
            preview_job = self._create_preview_job(job)

            # Get cropped preview
            preview = prc.crop_single_face(image, preview_job, face_tools, video=False)
            print(f"Face detection result: {preview is not None}")  # Debug

            if preview is not None:
                # Convert to QPixmap and cache
                pixmap = self._cv2_to_pixmap(preview)
                self.image_cache[file_path] = pixmap
                print(f"Created pixmap: {not pixmap.isNull()}")  # Debug

                if self.current_path == file_path:
                    self.setPixmap(pixmap)
                    # Show immediately instead of using timer for testing
                    self._show_at_position(mouse_pos)
            else:
                # No face detected - cache this result and show message
                self.image_cache[file_path] = "no_face"
                if self.current_path == file_path:
                    self._show_no_face_message()
                    self._show_at_position(mouse_pos)

        except Exception as e:
            print(f"Error loading preview for {file_path}: {e}")  # Debug
            # Cache the error and show message
            self.image_cache[file_path] = "error"
            if self.current_path == file_path:
                self._show_error_message("Error loading image")
                self._show_at_position(mouse_pos)

    def _show_no_face_message(self):
        """Show a no face detected message"""
        print("Showing no face message")  # Debug
        self._create_message_pixmap(self.no_face_message, is_error=False)

    def _show_error_message(self, message: str):
        """Show an error message"""
        print(f"Showing error message: {message}")  # Debug
        self._create_message_pixmap(message, is_error=True)

    def _create_message_pixmap(self, message: str, is_error: bool = False):
        """Create a pixmap with a centered message"""
        pixmap = QPixmap(self.size())
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Set up the font
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)

        # Set colors based on message type
        if is_error:
            bg_color = QColor(220, 53, 69, 200)  # Error red
            text_color = QColor(255, 255, 255)
            border_color = QColor(220, 53, 69)
        else:
            bg_color = QColor(255, 193, 7, 200)  # Warning yellow
            text_color = QColor(0, 0, 0)
            border_color = QColor(255, 193, 7)

        # Calculate text rectangle
        text_rect = painter.fontMetrics().boundingRect(
            pixmap.rect(),
            QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
            message
        )
        text_rect.adjust(-15, -10, 15, 10)  # Add padding

        # Center the rectangle
        x = (pixmap.width() - text_rect.width()) // 2
        y = (pixmap.height() - text_rect.height()) // 2
        text_rect.moveTo(x, y)

        # Draw background
        painter.fillRect(text_rect, bg_color)

        # Draw border
        painter.setPen(QPen(border_color, 2))
        painter.drawRoundedRect(text_rect, 6, 6)

        # Draw text
        painter.setPen(text_color)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap, message)

        painter.end()
        self.setPixmap(pixmap)

    @staticmethod
    def _create_preview_job(job: Job) -> Job:
        """Create a job for preview maintaining the original aspect ratio"""
        # Use the original job dimensions if they're valid
        width = job.width if job.width > 0 else 200
        height = job.height if job.height > 0 else 200

        return Job(
            width=width,
            height=height,
            fix_exposure_job=job.fix_exposure_job,
            multi_face_job=False,  # Always a single face for preview
            auto_tilt_job=job.auto_tilt_job,
            sensitivity=job.sensitivity,
            face_percent=job.face_percent,
            gamma=job.gamma,
            top=job.top,
            bottom=job.bottom,
            left=job.left,
            right=job.right,
            radio_buttons=job.radio_buttons
        )

    def _cv2_to_pixmap(self, image: cvt.MatLike) -> QPixmap:
        """Convert OpenCV image to QPixmap and scale to fit the preview widget"""
        height, width, channels = image.shape
        bytes_per_line = channels * width

        # Convert BGR to RGB for Qt
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert to QImage
        q_image = QImage(bytes(rgb_image.data), width, height, bytes_per_line, QImage.Format.Format_BGR888)

        # Convert to QPixmap
        pixmap = QPixmap.fromImage(q_image)

        # Scale the pixmap to fit within the preview widget while maintaining aspect ratio
        return pixmap.scaled(
            self.width() - 10,  # Account for padding
            self.height() - 10,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )

    def position_near_mouse(self, mouse_pos: QtCore.QPoint):
        """Position the preview near the mouse cursor"""
        print(f"Positioning preview near mouse at: {mouse_pos}")  # Debug

        # Offset from cursor
        offset_x, offset_y = 20, 20

        # Calculate position
        x = mouse_pos.x() + offset_x
        y = mouse_pos.y() + offset_y

        if screen := QtWidgets.QApplication.screenAt(mouse_pos):
            screen_rect = screen.geometry()
            print(f"Screen geometry: {screen_rect}")  # Debug

            # Adjust if too close to the right edge
            if x + self.width() > screen_rect.right():
                x = mouse_pos.x() - self.width() - offset_x

            # Adjust if too close to the bottom edge
            if y + self.height() > screen_rect.bottom():
                y = mouse_pos.y() - self.height() - offset_y

        final_pos = QtCore.QPoint(x, y)
        print(f"Final preview position: {final_pos}")  # Debug
        self.move(final_pos)

    def hide_preview(self):
        """Hide the preview and clear the current path"""
        print("Hiding preview")  # Debug
        self.show_timer.stop()
        self.hide()
        self.current_path = None
        self.pending_mouse_pos = None

    def clear_cache(self):
        """Clear the image cache"""
        print("Clearing preview cache")  # Debug
        self.image_cache.clear()
        self.current_path = None

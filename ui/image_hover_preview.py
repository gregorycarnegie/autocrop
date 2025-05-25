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
            QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)

        # Set a larger size for preview
        self.setFixedSize(250, 250)
        self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                background-color: rgba(255, 255, 255, 240);
                border: 1px solid #555;
                border-radius: 4px;
                padding: 5px;
            }
        """)

        self.setScaledContents(False)
        self.hide()

        # Timer to delay showing the preview
        self.show_timer = QtCore.QTimer()
        self.show_timer.setSingleShot(True)
        self.show_timer.timeout.connect(self.show)

        # Cache for loaded images
        self.image_cache = {}
        self.current_path = None
        self.no_face_message = "No face detected"

    def preview_file(self, file_path: str, mouse_pos: QtCore.QPoint,
                    face_tools: FaceToolPair, job: Job):
        """Load and show a preview of an image file"""
        if not file_path or file_path == self.current_path:
            return

        self.current_path = file_path

        # Check if image is in cache
        if file_path in self.image_cache:
            cached_item = self.image_cache[file_path]
            if cached_item == "no_face":
                self._show_no_face_message()
            else:
                self.setPixmap(cached_item)
            self.position_near_mouse(mouse_pos)
            return

        # Load image in a separate thread
        QtCore.QTimer.singleShot(0, lambda: self._load_image(file_path, mouse_pos, face_tools, job))

    def _load_image(self, file_path: str, mouse_pos: QtCore.QPoint,
                    face_tools: FaceToolPair, job: Job):
        """Load and process an image in a separate thread"""
        path = Path(file_path)
        if not path.exists():
            return

        try:
            # Load image at low resolution for performance
            if file_manager.is_valid_type(path, FileCategory.PHOTO) or \
               file_manager.is_valid_type(path, FileCategory.TIFF):
                image = cv2.imread(path.as_posix())
            else:
                return

            if image is None:
                return

            # Resize for preview (small size for performance)
            height, width = image.shape[:2]
            scale = min(300 / width, 300 / height, 1.0)
            if scale < 1.0:
                new_size = (int(width * scale), int(height * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

            # Create a preview job
            preview_job = self._create_preview_job(job)

            # Get cropped preview
            preview = prc.crop_single_face(image, preview_job, face_tools, video=False)

            if preview is not None:
                # Convert to QPixmap and cache
                pixmap = self._cv2_to_pixmap(preview)
                self.image_cache[file_path] = pixmap

                if self.current_path == file_path:
                    self.setPixmap(pixmap)
                    self.position_near_mouse(mouse_pos)
            else:
                # No face detected - cache this result and show message
                self.image_cache[file_path] = "no_face"
                if self.current_path == file_path:
                    self._show_no_face_message()
                    self.position_near_mouse(mouse_pos)

        except Exception as e:
            print(f"Error loading preview for {file_path}: {e}")
            # Cache the error and show message
            self.image_cache[file_path] = "error"
            if self.current_path == file_path:
                self._show_error_message("Error loading image")
                self.position_near_mouse(mouse_pos)

    def _show_no_face_message(self):
        """Show a no face detected message"""
        self._create_message_pixmap(self.no_face_message, is_error=False)

    def _show_error_message(self, message: str):
        """Show an error message"""
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

        # Convert to QImage
        q_image = QImage(bytes(image.data), width, height, bytes_per_line, QImage.Format.Format_RGB888)

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
        # Offset from cursor
        offset_x, offset_y = 20, 20

        # Calculate position
        x = mouse_pos.x() + offset_x
        y = mouse_pos.y() + offset_y

        if screen := QtWidgets.QApplication.screenAt(mouse_pos):
            screen_rect = screen.geometry()

            # Adjust if too close to the right edge
            if x + self.width() > screen_rect.right():
                x = mouse_pos.x() - self.width() - offset_x

            # Adjust if too close to the bottom edge
            if y + self.height() > screen_rect.bottom():
                y = mouse_pos.y() - self.height() - offset_y

        self.move(x, y)

        # Delay showing to avoid flickering
        self.show_timer.start(300)  # 300ms delay

    def hide_preview(self):
        """Hide the preview and clear the current path"""
        self.show_timer.stop()
        self.hide()
        self.current_path = None

    def clear_cache(self):
        """Clear the image cache"""
        self.image_cache.clear()
        self.current_path = None

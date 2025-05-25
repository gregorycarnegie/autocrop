import logging
from pathlib import Path

import cv2
import cv2.typing as cvt
from PyQt6 import QtCore, QtWidgets
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap

from core import processing as prc
from core.config import Config
from core.face_tools import FaceToolPair
from core.job import Job
from file_types import FileCategory, file_manager

# Initialize module-level logger
logger = logging.getLogger(__name__)
if not Config.disable_logging:
    logger.setLevel(logging.CRITICAL + 1)


class ImageHoverPreview(QtWidgets.QLabel):
    """A tooltip-style widget that shows a preview image when hovering over files"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(
            QtCore.Qt.WindowType.Tool |
            QtCore.Qt.WindowType.FramelessWindowHint |
            QtCore.Qt.WindowType.WindowStaysOnTopHint |
            QtCore.Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_PaintOnScreen, False)

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

        self.show_timer = QtCore.QTimer()
        self.show_timer.setSingleShot(True)
        self.show_timer.timeout.connect(self._actually_show)

        self.image_cache = {}
        self.current_path = None
        self.no_face_message = "No face detected"
        self.pending_mouse_pos = None

        logger.debug("ImageHoverPreview initialized")

    def preview_file(self, file_path: str, mouse_pos: QtCore.QPoint,
                    face_tools: FaceToolPair, job: Job):
        logger.debug("preview_file called with: %s", file_path)

        if not file_path or file_path == self.current_path:
            logger.debug("Skipping preview - no path or same path: %s", file_path)
            return

        self.current_path = file_path
        self.pending_mouse_pos = mouse_pos

        if file_path in self.image_cache:
            logger.debug("Using cached image for: %s", file_path)
            cached_item = self.image_cache[file_path]
            if cached_item == "no_face":
                self._show_no_face_message()
            elif cached_item == "error":
                self._show_error_message("Error loading image")
            else:
                self.setPixmap(cached_item)
            self._show_at_position(mouse_pos)
            return

        logger.debug("Loading new image: %s", file_path)
        self.show_timer.start(200)
        QtCore.QTimer.singleShot(0, lambda: self._load_image(file_path, mouse_pos, face_tools, job))

    def _actually_show(self):
        if self.pending_mouse_pos and self.pixmap() and not self.pixmap().isNull():
            logger.debug("Timer expired, showing preview")
            self._show_at_position(self.pending_mouse_pos)
        else:
            logger.debug("Timer expired but no valid preview to show")

    def _show_at_position(self, mouse_pos: QtCore.QPoint):
        logger.debug("_show_at_position called at: %s", mouse_pos)
        self.position_near_mouse(mouse_pos)
        self.show()
        self.raise_()
        self.activateWindow()
        logger.debug("Preview should be visible now. isVisible: %s", self.isVisible())

    def _load_image(self, file_path: str, mouse_pos: QtCore.QPoint,
                    face_tools: FaceToolPair, job: Job):
        logger.debug("_load_image called for: %s", file_path)

        path = Path(file_path)
        if not path.exists():
            logger.warning("Path does not exist: %s", file_path)
            return

        try:
            if file_manager.is_valid_type(path, FileCategory.PHOTO) or \
               file_manager.is_valid_type(path, FileCategory.TIFF):
                image = cv2.imread(path.as_posix())
                logger.debug("Loaded image with cv2.imread: %s", image is not None)
            else:
                logger.warning("Unsupported file type: %s", file_path)
                return

            if image is None:
                logger.error("Failed to load image: %s", file_path)
                return

            height, width = image.shape[:2]
            logger.debug("Original image size: %dx%d", width, height)

            scale = min(300 / width, 300 / height, 1.0)
            if scale < 1.0:
                new_size = (int(width * scale), int(height * scale))
                image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
                logger.debug("Resized to: %s", new_size)

            preview_job = self._create_preview_job(job)
            preview = prc.crop_single_face(image, preview_job, face_tools, video=False)
            logger.debug("Face detection result: %s", preview is not None)

            if preview is not None:
                pixmap = self._cv2_to_pixmap(preview)
                self.image_cache[file_path] = pixmap
                logger.debug("Created pixmap: %s", not pixmap.isNull())

                if self.current_path == file_path:
                    self.setPixmap(pixmap)
                    self._show_at_position(mouse_pos)
            else:
                self.image_cache[file_path] = "no_face"
                if self.current_path == file_path:
                    self._show_no_face_message()
                    self._show_at_position(mouse_pos)

        except Exception as e:
            logger.exception("Error loading preview for %s: %s", file_path, e)
            self.image_cache[file_path] = "error"
            if self.current_path == file_path:
                self._show_error_message("Error loading image")
                self._show_at_position(mouse_pos)

    def _show_no_face_message(self):
        logger.debug("Showing no face message")
        self._create_message_pixmap(self.no_face_message, is_error=False)

    def _show_error_message(self, message: str):
        logger.debug("Showing error message: %s", message)
        self._create_message_pixmap(message, is_error=True)

    def _create_message_pixmap(self, message: str, is_error: bool = False):
        pixmap = QPixmap(self.size())
        pixmap.fill(QtCore.Qt.GlobalColor.transparent)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)

        if is_error:
            bg_color = QColor(220, 53, 69, 200)
            text_color = QColor(255, 255, 255)
            border_color = QColor(220, 53, 69)
        else:
            bg_color = QColor(255, 193, 7, 200)
            text_color = QColor(0, 0, 0)
            border_color = QColor(255, 193, 7)

        text_rect = painter.fontMetrics().boundingRect(
            pixmap.rect(),
            QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap,
            message
        )
        text_rect.adjust(-15, -10, 15, 10)

        x = (pixmap.width() - text_rect.width()) // 2
        y = (pixmap.height() - text_rect.height()) // 2
        text_rect.moveTo(x, y)

        painter.fillRect(text_rect, bg_color)
        painter.setPen(QPen(border_color, 2))
        painter.drawRoundedRect(text_rect, 6, 6)
        painter.setPen(text_color)
        painter.drawText(text_rect, QtCore.Qt.AlignmentFlag.AlignCenter | QtCore.Qt.TextFlag.TextWordWrap, message)
        painter.end()
        self.setPixmap(pixmap)

    @staticmethod
    def _create_preview_job(job: Job) -> Job:
        width = job.width if job.width > 0 else 200
        height = job.height if job.height > 0 else 200
        return Job(
            width=width,
            height=height,
            fix_exposure_job=job.fix_exposure_job,
            multi_face_job=False,
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
        height, width, channels = image.shape
        bytes_per_line = channels * width

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_image = QImage(bytes(rgb_image.data), width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_image)
        return pixmap.scaled(
            self.width() - 10,
            self.height() - 10,
            QtCore.Qt.AspectRatioMode.KeepAspectRatio,
            QtCore.Qt.TransformationMode.SmoothTransformation
        )

    def position_near_mouse(self, mouse_pos: QtCore.QPoint):
        logger.debug("Positioning preview near mouse at: %s", mouse_pos)
        offset_x, offset_y = 20, 20
        x = mouse_pos.x() + offset_x
        y = mouse_pos.y() + offset_y

        if screen := QtWidgets.QApplication.screenAt(mouse_pos):
            screen_rect = screen.geometry()
            logger.debug("Screen geometry: %s", screen_rect)
            if x + self.width() > screen_rect.right():
                x = mouse_pos.x() - self.width() - offset_x
            if y + self.height() > screen_rect.bottom():
                y = mouse_pos.y() - self.height() - offset_y

        final_pos = QtCore.QPoint(x, y)
        logger.debug("Final preview position: %s", final_pos)
        self.move(final_pos)

    def hide_preview(self):
        logger.debug("Hiding preview")
        self.show_timer.stop()
        self.hide()
        self.current_path = None
        self.pending_mouse_pos = None

    def clear_cache(self):
        logger.debug("Clearing preview cache")
        self.image_cache.clear()
        self.current_path = None

import contextlib
from collections.abc import Callable
from multiprocessing import Process
from pathlib import Path
from threading import Thread
from typing import Any

import numpy as np
from PyQt6.QtCore import QCoreApplication, QMetaObject, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import QIcon
from PyQt6.QtMultimedia import QAudioOutput, QMediaPlayer
from PyQt6.QtMultimediaWidgets import QVideoWidget
from PyQt6.QtWidgets import (
    QApplication,
    QFileDialog,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpacerItem,
    QToolBox,
    QWidget,
)

from core import Job
from core import processing as prc
from core.croppers import VideoCropper
from core.enums import FunctionType
from file_types import FileCategory, file_manager
from ui import utils as ut

from .crop_widget import UiCropWidget
from .enums import GuiIcon
from .media_controls import UiMediaControlWidget


class UiVideoTabWidget(UiCropWidget):
    """Video tab widget with enhanced inheritance from base crop_from_path widget"""

    PROGRESSBAR_STEPS: int = 1_000  # From UiCropBatchWidget

    def __init__(self, crop_worker: VideoCropper, object_name: str, parent: QWidget) -> None:
        """Initialize the video tab widget"""
        super().__init__(parent, object_name)

        # Path storage fields
        self.input_path = ""
        self.destination_path = ""
        self.crop_worker = crop_worker

        # Media player attributes
        self.vol_cache = 70
        self.rewind_timer = QTimer()
        self.default_directory = file_manager.get_default_directory(FileCategory.VIDEO).as_posix()
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.start_position, self.stop_position, self.step = .0, .0, 100
        self.speed = 0
        self.reverse = 0

        # Create additional UI elements
        self.progressBar = self.create_progress_bar("progressBar")
        self.progressBar_2 = self.create_progress_bar("progressBar_2")
        self.toolBox = QToolBox(self)
        self.toolBox.setObjectName("toolBox")

        # Create pages for the toolbox
        self.page_1 = QWidget()
        self.page_1.setObjectName("page_1")
        self.page_2 = QWidget()
        self.page_2.setObjectName("page_2")

        # Set up page layouts
        self.verticalLayout_200 = ut.setup_vbox("verticalLayout_200", self.page_1)
        self.verticalLayout_300 = ut.setup_vbox("verticalLayout_300", self.page_2)

        # Create video-specific widgets
        self.videoWidget = QVideoWidget()
        self.videoWidget.setObjectName("videoWidget")
        self.videoWidget.setStyleSheet("background: #1f2c33")

        self.muteButton_1 = QPushButton()
        self.muteButton_2 = QPushButton()
        self.volumeSlider_1 = QSlider()
        self.volumeSlider_2 = QSlider()
        self.positionLabel_1 = QLabel()
        self.positionLabel_2 = QLabel()
        self.timelineSlider_1 = QSlider()
        self.timelineSlider_2 = QSlider()
        self.durationLabel_1 = QLabel()
        self.durationLabel_2 = QLabel()

        self.frame_1 = self.create_main_frame("frame_1")
        self.frame_1.setParent(self.page_1)

        self.frame_2 = self.create_main_frame("frame_2")
        self.frame_2.setParent(self.page_2)
        # Create media control widgets
        self.mediacontrolWidget_1 = UiMediaControlWidget(self.frame_1, self.player, self.crop_worker)
        self.mediacontrolWidget_2 = UiMediaControlWidget(self.frame_2, self.player, self.crop_worker)

        self.previewButton = QPushButton("Update Preview")

        # Set up the main layout structure
        self.setup_layouts()

        self.connect_preview_updates()

        # Connect signals

        self.connect_signals()

        # Set initial UI text
        self.retranslateUi()

        # Set initial toolbox page
        self.toolBox.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(self)

    # Todo
    def connect_preview_updates(self):
        """Connect signals that should trigger a preview update"""
        # Connect to player position changes
        self.timelineSlider_1.sliderReleased.connect(self.display_crop_preview)
        self.timelineSlider_2.sliderReleased.connect(self.display_crop_preview)

        # Connect to settings changes
        self.mfaceCheckBox.stateChanged.connect(self.display_crop_preview)
        self.tiltCheckBox.stateChanged.connect(self.display_crop_preview)
        self.exposureCheckBox.stateChanged.connect(self.display_crop_preview)
        self.controlWidget.widthLineEdit.textChanged.connect(self.display_crop_preview)
        self.controlWidget.heightLineEdit.textChanged.connect(self.display_crop_preview)
        self.controlWidget.sensitivityDial.valueChanged.connect(self.display_crop_preview)
        self.controlWidget.fpctDial.valueChanged.connect(self.display_crop_preview)
        self.controlWidget.gammaDial.valueChanged.connect(self.display_crop_preview)
        self.controlWidget.topDial.valueChanged.connect(self.display_crop_preview)
        self.controlWidget.bottomDial.valueChanged.connect(self.display_crop_preview)
        self.controlWidget.leftDial.valueChanged.connect(self.display_crop_preview)
        self.controlWidget.rightDial.valueChanged.connect(self.display_crop_preview)

    def create_progress_bar(self, name: str, parent: QWidget | None = None) -> QProgressBar:
        """Create a progress bar with consistent styling"""
        progress_bar = QProgressBar() if parent is None else QProgressBar(parent)
        progress_bar.setObjectName(name)
        progress_bar.setMinimumSize(QSize(0, 15))
        progress_bar.setMaximumSize(QSize(16_777_215, 15))
        progress_bar.setRange(0, self.PROGRESSBAR_STEPS)
        progress_bar.setValue(0)

        # Apply styling
        progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: #f0f0f0;
                text-align: center;
                color: #505050;
            }

            QProgressBar::chunk {
                background-color: #4285f4;
                width: 10px;
                margin: 0.5px;
            }
        """)

        progress_bar.setTextVisible(False)
        return progress_bar

    def setup_layouts(self) -> None:
        """Set up the main layout structure"""
        # ---- Page 1: Video Player ----
        vertical_layout_9 = ut.setup_vbox("verticalLayout_9", self.frame_1)

        # Media controls layout
        media_control_layout = ut.setup_hbox("horizontalLayout_1")

        # Mute button
        self.muteButton_1.setObjectName("muteButton_1")
        self.muteButton_1.setMinimumSize(QSize(30, 30))
        self.muteButton_1.setMaximumSize(QSize(30, 30))
        self.muteButton_1.setBaseSize(QSize(30, 30))
        self.muteButton_1.setIcon(QIcon(GuiIcon.MULTIMEDIA_MUTE))
        media_control_layout.addWidget(self.muteButton_1)

        # Volume slider
        self.volumeSlider_1.setObjectName("volumeSlider_1")
        size_policy3 = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        size_policy3.setHorizontalStretch(0)
        size_policy3.setVerticalStretch(0)
        size_policy3.setHeightForWidth(self.volumeSlider_1.sizePolicy().hasHeightForWidth())
        self.volumeSlider_1.setSizePolicy(size_policy3)
        self.volumeSlider_1.setMinimumSize(QSize(0, 30))
        self.volumeSlider_1.setMaximumSize(QSize(16_777_215, 30))
        self.volumeSlider_1.setMinimum(-1)
        self.volumeSlider_1.setMaximum(100)
        self.volumeSlider_1.setValue(70)
        self.volumeSlider_1.setOrientation(Qt.Orientation.Horizontal)
        media_control_layout.addWidget(self.volumeSlider_1)

        # Position label
        self.positionLabel_1.setObjectName("positionLabel_1")
        size_policy4 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        size_policy4.setHorizontalStretch(0)
        size_policy4.setVerticalStretch(0)
        size_policy4.setHeightForWidth(self.positionLabel_1.sizePolicy().hasHeightForWidth())
        self.positionLabel_1.setSizePolicy(size_policy4)
        self.positionLabel_1.setMinimumSize(QSize(0, 30))
        self.positionLabel_1.setMaximumSize(QSize(16_777_215, 30))
        media_control_layout.addWidget(self.positionLabel_1)

        # Timeline slider
        self.timelineSlider_1.setObjectName("timelineSlider_1")
        size_policy5 = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        size_policy5.setHorizontalStretch(1)
        size_policy5.setVerticalStretch(0)
        size_policy5.setHeightForWidth(self.timelineSlider_1.sizePolicy().hasHeightForWidth())
        self.timelineSlider_1.setSizePolicy(size_policy5)
        self.timelineSlider_1.setMinimumSize(QSize(0, 30))
        self.timelineSlider_1.setMaximumSize(QSize(16_777_215, 30))
        self.timelineSlider_1.setOrientation(Qt.Orientation.Horizontal)
        media_control_layout.addWidget(self.timelineSlider_1)

        # Duration label
        self.durationLabel_1.setObjectName("durationLabel_1")
        size_policy4.setHeightForWidth(self.durationLabel_1.sizePolicy().hasHeightForWidth())
        self.durationLabel_1.setSizePolicy(size_policy4)
        self.durationLabel_1.setMinimumSize(QSize(0, 30))
        self.durationLabel_1.setMaximumSize(QSize(16_777_215, 30))
        media_control_layout.addWidget(self.durationLabel_1)

        # Horizontal spacer
        horizontal_spacer_1 = QSpacerItem(
            20, 20,
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Minimum
        )
        media_control_layout.addItem(horizontal_spacer_1)
        media_control_layout.setStretch(0, 1)
        media_control_layout.setStretch(1, 3)

        vertical_layout_9.addLayout(media_control_layout)

        # Video widget
        self.videoWidget.setParent(self.frame_1)
        self.size_policy_expand_expand.setHeightForWidth(self.videoWidget.sizePolicy().hasHeightForWidth())
        self.videoWidget.setSizePolicy(self.size_policy_expand_expand)
        self.videoWidget.setMinimumSize(QSize(200, 200))

        vertical_layout_9.addWidget(self.videoWidget)

        self.mediacontrolWidget_1.setObjectName("mediacontrolWidget_1")
        vertical_layout_9.addWidget(self.mediacontrolWidget_1)

        # Progress bar
        self.progressBar.setParent(self.frame_1)
        vertical_layout_9.addWidget(self.progressBar)

        vertical_layout_9.setStretch(1, 1)

        self.verticalLayout_200.addWidget(self.frame_1)

        # Destination selection
        self.verticalLayout_200.addLayout(self.horizontalLayout_3)

        # Add page to toolbox
        self.toolBox.addItem(self.page_1, "Video Player")

        # ---- Page 2: Crop View ----
        vertical_layout_10 = ut.setup_vbox("verticalLayout_10", self.frame_2)

        # Media controls layout for page 2
        media_control_layout2 = ut.setup_hbox("horizontalLayout_5")

        # Mute button
        self.muteButton_2.setObjectName("muteButton_2")
        self.muteButton_2.setMinimumSize(QSize(30, 30))
        self.muteButton_2.setMaximumSize(QSize(30, 30))
        self.muteButton_2.setBaseSize(QSize(30, 30))
        self.muteButton_2.setIcon(QIcon(GuiIcon.MULTIMEDIA_MUTE))
        media_control_layout2.addWidget(self.muteButton_2)

        # Volume slider
        self.volumeSlider_2.setObjectName("volumeSlider_2")
        size_policy3.setHeightForWidth(self.volumeSlider_2.sizePolicy().hasHeightForWidth())
        self.volumeSlider_2.setSizePolicy(size_policy3)
        self.volumeSlider_2.setMinimumSize(QSize(0, 30))
        self.volumeSlider_2.setMaximumSize(QSize(16_777_215, 30))
        self.volumeSlider_2.setMinimum(-1)
        self.volumeSlider_2.setMaximum(100)
        self.volumeSlider_2.setValue(70)
        self.volumeSlider_2.setOrientation(Qt.Orientation.Horizontal)
        media_control_layout2.addWidget(self.volumeSlider_2)

        # Position label
        self.positionLabel_2.setObjectName("positionLabel_2")
        size_policy4.setHeightForWidth(self.positionLabel_2.sizePolicy().hasHeightForWidth())
        self.positionLabel_2.setSizePolicy(size_policy4)
        self.positionLabel_2.setMinimumSize(QSize(0, 30))
        self.positionLabel_2.setMaximumSize(QSize(16_777_215, 30))
        media_control_layout2.addWidget(self.positionLabel_2)

        # Timeline slider
        self.timelineSlider_2.setObjectName("timelineSlider_2")
        size_policy5.setHeightForWidth(self.timelineSlider_2.sizePolicy().hasHeightForWidth())
        self.timelineSlider_2.setSizePolicy(size_policy5)
        self.timelineSlider_2.setMinimumSize(QSize(0, 30))
        self.timelineSlider_2.setMaximumSize(QSize(16_777_215, 30))
        self.timelineSlider_2.setOrientation(Qt.Orientation.Horizontal)
        media_control_layout2.addWidget(self.timelineSlider_2)

        # Duration label
        self.durationLabel_2.setObjectName("durationLabel_2")
        size_policy4.setHeightForWidth(self.durationLabel_2.sizePolicy().hasHeightForWidth())
        self.durationLabel_2.setSizePolicy(size_policy4)
        self.durationLabel_2.setMinimumSize(QSize(0, 30))
        self.durationLabel_2.setMaximumSize(QSize(16_777_215, 30))
        media_control_layout2.addWidget(self.durationLabel_2)

        media_control_layout2.setStretch(0, 1)
        media_control_layout2.setStretch(1, 3)

        vertical_layout_10.addLayout(media_control_layout2)

        # Checkbox section for page 2
        self.toggleCheckBox.setParent(self.frame_2)
        self.mfaceCheckBox.setParent(self.frame_2)
        self.tiltCheckBox.setParent(self.frame_2)
        self.exposureCheckBox.setParent(self.frame_2)

        checkbox_layout = ut.setup_hbox("horizontalLayout_4")
        checkbox_layout.addWidget(self.toggleCheckBox)

        # Add spacer
        horizontal_spacer_2 = QSpacerItem(
            40, 20,
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum
        )
        checkbox_layout.addItem(horizontal_spacer_2)

        # Add checkboxes
        checkbox_layout.addWidget(self.mfaceCheckBox)
        checkbox_layout.addWidget(self.tiltCheckBox)
        checkbox_layout.addWidget(self.exposureCheckBox)

        # Set stretch factor for spacer
        checkbox_layout.setStretch(1, 20)

        vertical_layout_10.addLayout(checkbox_layout)

        # Add preview button
        self.previewButton.setObjectName("previewButton")
        self.previewButton.setMinimumSize(QSize(0, 40))
        self.previewButton.setMaximumSize(QSize(16_777_215, 40))
        icon = ut.create_button_icon(GuiIcon.PICTURE)
        self.previewButton.setIcon(icon)
        self.previewButton.setIconSize(QSize(18, 18))
        vertical_layout_10.addWidget(self.previewButton)

        # Image widget (preview)
        self.imageWidget.setParent(self.frame_2)
        vertical_layout_10.addWidget(self.imageWidget)

        # Media control widget 2
        # self.mediacontrolWidget_2 = UiMediaControlWidget(frame_2, self.player, self.crop_worker)
        self.mediacontrolWidget_2.setObjectName("mediacontrolWidget_2")
        vertical_layout_10.addWidget(self.mediacontrolWidget_2)

        # Progress bar 2
        self.progressBar_2.setParent(self.frame_2)
        vertical_layout_10.addWidget(self.progressBar_2)

        # Add frame to page layout
        self.verticalLayout_300.addWidget(self.frame_2)

        # Add page to toolbox
        self.toolBox.addItem(self.page_2, "Crop View")

        # Add toolbox to the main layout
        self.verticalLayout_100.addWidget(self.toolBox)

        # Set up the media player
        self.create_media_player()

    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        # Connect preview button
        self.previewButton.clicked.connect(self.display_crop_preview)

        # Marker button connections
        self.mediacontrolWidget_1.startmarkerButton.clicked.connect(
            lambda: self.set_start_position(self.mediacontrolWidget_1.selectStartMarkerButton)
        )
        self.mediacontrolWidget_1.endmarkerButton.clicked.connect(
            lambda: self.set_stop_position(self.mediacontrolWidget_1.selectEndMarkerButton)
        )
        self.mediacontrolWidget_1.startmarkerButton.clicked.connect(
            lambda: self.set_start_position(self.mediacontrolWidget_2.selectStartMarkerButton)
        )
        self.mediacontrolWidget_1.endmarkerButton.clicked.connect(
            lambda: self.set_stop_position(self.mediacontrolWidget_2.selectEndMarkerButton)
        )
        self.mediacontrolWidget_2.startmarkerButton.clicked.connect(
            lambda: self.set_start_position(self.mediacontrolWidget_1.selectStartMarkerButton)
        )
        self.mediacontrolWidget_2.endmarkerButton.clicked.connect(
            lambda: self.set_stop_position(self.mediacontrolWidget_1.selectEndMarkerButton)
        )
        self.mediacontrolWidget_2.startmarkerButton.clicked.connect(
            lambda: self.set_start_position(self.mediacontrolWidget_2.selectStartMarkerButton)
        )
        self.mediacontrolWidget_2.endmarkerButton.clicked.connect(
            lambda: self.set_stop_position(self.mediacontrolWidget_2.selectEndMarkerButton)
        )

        # Media player connections
        self.volumeSlider_1.sliderMoved.connect(self.volume_slider_changed)
        self.volumeSlider_2.sliderMoved.connect(self.volume_slider_changed)
        self.timelineSlider_1.sliderMoved.connect(self.player_slider_changed)
        self.timelineSlider_2.sliderMoved.connect(self.player_slider_changed)
        self.timelineSlider_1.sliderMoved.connect(self.timelineSlider_2.setSliderPosition)
        self.timelineSlider_2.sliderMoved.connect(self.timelineSlider_1.setSliderPosition)

        # Connect media control widgets
        for control in [self.mediacontrolWidget_1, self.mediacontrolWidget_2]:
            control.cropButton.clicked.connect(self.crop_frame)
            control.videocropButton.clicked.connect(self.video_process)
            control.cancelButton.clicked.connect(self.crop_worker.terminate)
            control.cancelButton.clicked.connect(
                lambda: self.cancel_button_operation(control.cancelButton,
                                                     control.videocropButton,
                                                     control.cropButton)
            )

            control.playButton.clicked.connect(self.change_playback_state)
            control.stopButton.clicked.connect(self.stop_playback)
            control.stepbackButton.clicked.connect(self.step_back)
            control.stepfwdButton.clicked.connect(self.step_forward)
            control.fastfwdButton.clicked.connect(self.fast_forward)
            control.goto_beginingButton.clicked.connect(self.goto_beginning)
            control.goto_endButton.clicked.connect(self.goto_end)
            control.selectStartMarkerButton.clicked.connect(
                lambda: self.goto(control.selectStartMarkerButton)
            )
            control.selectEndMarkerButton.clicked.connect(
                lambda: self.goto(control.selectEndMarkerButton)
            )

        # Audio control buttons
        self.muteButton_1.clicked.connect(self.volume_mute)
        self.muteButton_2.clicked.connect(self.volume_mute)

        # Media player connections
        self.audio.mutedChanged.connect(self.change_audio_icon)
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.errorOccurred.connect(self.player_error_occurred)

        # Register button dependencies with TabStateManager
        for control in [self.mediacontrolWidget_1, self.mediacontrolWidget_2]:
            ut.register_button_dependencies(
                self.tab_state_manager,
                control.cropButton,
                {
                    self.controlWidget.widthLineEdit,
                    self.controlWidget.heightLineEdit
                }
            )

            ut.register_button_dependencies(
                self.tab_state_manager,
                control.videocropButton,
                {
                    self.controlWidget.widthLineEdit,
                    self.controlWidget.heightLineEdit
                }
            )

        # Connect input widgets for validation tracking
        self.tab_state_manager.connect_widgets(
            self.controlWidget.widthLineEdit,
            self.controlWidget.heightLineEdit,
            self.exposureCheckBox,
            self.mfaceCheckBox,
            self.tiltCheckBox,
            self.controlWidget.sensitivityDial,
            self.controlWidget.fpctDial,
            self.controlWidget.gammaDial,
            self.controlWidget.topDial,
            self.controlWidget.bottomDial,
            self.controlWidget.leftDial,
            self.controlWidget.rightDial
        )

        # Connect crop_from_path worker signals
        self.connect_crop_worker()

    def retranslateUi(self) -> None:
        """Update UI text elements"""
        super().retranslateUi()
        self.muteButton_1.setText("")
        self.muteButton_2.setText("")
        self.positionLabel_1.setText(QCoreApplication.translate("self", "00:00:00", None))
        self.positionLabel_2.setText(QCoreApplication.translate("self", "00:00:00", None))
        self.durationLabel_1.setText(QCoreApplication.translate("self", "00:00:00", None))
        self.durationLabel_2.setText(QCoreApplication.translate("self", "00:00:00", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_1),
                                 QCoreApplication.translate("self", "Video Player", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2),
                                 QCoreApplication.translate("self", "Crop View", None))

    def display_crop_preview(self) -> None:
        """Captures the current frame and displays crop_from_path preview in the imageWidget"""
        if not self.input_path:
            return None

        # Get current position
        position = self.timelineSlider_1.value()

        # Use the optimized grab_frame method for preview
        frame = self.crop_worker.grab_frame(position, self.input_path, for_preview=True)
        if frame is None:
            return None

        # Create a job with current settings
        job = self.create_job(
            FunctionType.FRAME,
            video_path=Path(self.input_path),
            destination=Path(self.destination_path or ".")
        )

        # Process the frame
        if job.multi_face_job:
            # If multi-face is enabled, show bounding boxes on all faces
            processed_image = prc.annotate_faces(frame, job, self.crop_worker.face_detection_tools)
            if processed_image is None:
                return None
            ut.display_image_on_widget(processed_image, self.imageWidget)
        else:
            # For single face, show a crop_from_path preview
            cropped_image = prc.crop_single_face(frame, job, self.crop_worker.face_detection_tools, video=True)
            if cropped_image is not None:
                # Display the cropped image
                ut.display_image_on_widget(cropped_image, self.imageWidget)
                return None

        return None

    # Media player methods
    def create_media_player(self) -> None:
        """Set up the media player"""
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.videoWidget)

    def player_error_occurred(self, error: QMediaPlayer.Error) -> None:
        """Handle media player errors"""
        match error:
            case QMediaPlayer.Error.NoError:
                pass
            case _:
                self.stop_playback()
                ut.show_error_box(
                    f'{error.name} occurred while loading the video. ',
                    'Please check the video file path and try again.'
                )

    def open_video(self) -> None:
        """Open a video file dialog with the string-based approach"""
        self.check_playback_state()
        file_name, _ = QFileDialog.getOpenFileName(
            self, 'Open Video',
            self.default_directory,
            file_manager.get_filter_string(FileCategory.VIDEO)
        )

        # Update the input path
        self.input_path = file_name

        # Also update the main window's unified address bar if this is the active tab
        main_window = self.parent().parent().parent()
        if main_window.function_tabWidget.currentIndex() == FunctionType.VIDEO:
            main_window.unified_address_bar.setText(file_name)

        # Load the video
        self.player.setSource(QUrl.fromLocalFile(file_name))
        self.reset_video_widgets()

    def open_dropped_video(self) -> None:
        """Handle video dropped onto the widget"""
        self.player.setSource(QUrl.fromLocalFile(self.input_path))
        self.reset_video_widgets()

    def reset_video_widgets(self) -> None:
        """Reset video widgets to initial state"""
        self.create_media_player()
        self.mediacontrolWidget_1.playButton.setEnabled(True)
        self.mediacontrolWidget_2.playButton.setEnabled(True)
        self.timelineSlider_1.setEnabled(True)
        self.timelineSlider_2.setEnabled(True)

    def change_audio_icon(self) -> None:
            """Update audio mute button icon based on mute state"""
            if self.audio.isMuted():
                self.muteButton_1.setIcon(QIcon(GuiIcon.MULTIMEDIA_UNMUTE))
                self.muteButton_2.setIcon(QIcon(GuiIcon.MULTIMEDIA_UNMUTE))
            else:
                self.muteButton_1.setIcon(QIcon(GuiIcon.MULTIMEDIA_MUTE))
                self.muteButton_2.setIcon(QIcon(GuiIcon.MULTIMEDIA_MUTE))

    def playback_bool(self,
                      a0: QMediaPlayer.PlaybackState = QMediaPlayer.PlaybackState.PausedState,
                      a1: QMediaPlayer.PlaybackState = QMediaPlayer.PlaybackState.StoppedState) -> \
            tuple[bool, bool]:
        """Returns a tuple of bools comparing the playback state to the Class attributes of
        PyQt6.QMediaPlayer.PlaybackState"""
        return self.player.playbackState() == a0, self.player.playbackState() == a1

    def check_playback_state(self) -> None:
        """Stops playback if in the paused state or playing state"""
        x, y = self.playback_bool(a1=QMediaPlayer.PlaybackState.PlayingState)
        if x ^ y:
            self.stop_playback()

    def change_playback_state(self) -> None:
        """Toggle between play and pause"""
        match self.player.playbackState():
            case QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
                self.speed = 0
            case _:
                self.player.play()

    def stop_playback(self) -> None:
        """Stop playback and disable timeline sliders"""
        self.timelineSlider_1.setDisabled(True)
        self.timelineSlider_2.setDisabled(True)
        # Check paused or playing
        x, y = self.playback_bool(a1=QMediaPlayer.PlaybackState.PlayingState)
        if x ^ y:
            self.player.stop()

    def fast_forward(self) -> None:
        """Speed up playback"""
        # Check paused or stopped
        x, y = self.playback_bool()
        if x ^ y:
            return
        video_speeds = np.array([1.0, 1.25, 1.5, 1.75, 2.0])
        self.reverse = 0
        self.speed += 1
        if self.speed > video_speeds.size - 1:
            self.player.setPlaybackRate(float(video_speeds[-1]))
        else:
            self.player.setPlaybackRate(float(video_speeds[self.speed]))

    def step_forward(self) -> None:
        """Step forward by 10 seconds"""
        if (new_position := self.player.position() + 10_000) >= self.player.duration():
            self.player.setPosition(self.player.duration())
        else:
            self.player.setPosition(new_position)

    def step_back(self) -> None:
        """Step backward by 10 seconds"""
        if (new_position := self.player.position() - 10_000) <= 0:
            self.player.setPosition(0)
        else:
            self.player.setPosition(new_position)

    def position_changed(self, position: int) -> None:
        """Handle position change in the media player"""
        def callback():
            for slider, label in zip((self.timelineSlider_1, self.timelineSlider_2),
                                    (self.positionLabel_1, self.positionLabel_2)):
                if slider.maximum() != self.player.duration():
                    slider.setMaximum(self.player.duration())

                slider.blockSignals(True)
                slider.setValue(position)
                slider.blockSignals(False)
                label.setText(ut.get_qtime(position).toString())

        thread = Thread(target=callback)
        thread.start()

    def duration_changed(self, duration: int) -> None:
        """Handle duration change in the media player"""
        self.timelineSlider_1.setMaximum(duration)
        self.timelineSlider_2.setMaximum(duration)
        if duration >= 0:
            q_time = ut.get_qtime(self.player.duration()).toString()
            self.durationLabel_1.setText(q_time)
            self.durationLabel_2.setText(q_time)
            self.mediacontrolWidget_1.selectEndMarkerButton.setText(q_time)
            self.mediacontrolWidget_2.selectEndMarkerButton.setText(q_time)

    def player_slider_changed(self, position: int) -> None:
        """Handle timeline slider position changes"""
        self.player.setPosition(position)

    def volume_slider_changed(self, position: int) -> None:
        """Handle volume slider changes"""
        self.audio.setVolume(position / 100.0)  # Convert to 0-1 range
        self.vol_cache = position

    def volume_mute(self) -> None:
        """Toggle audio mute state"""
        if self.audio.isMuted():
            self.audio.setMuted(False)
            self.volumeSlider_1.setValue(self.vol_cache)
            self.volumeSlider_2.setValue(self.vol_cache)
        else:
            self.audio.setMuted(True)
            self.volumeSlider_1.setValue(0)
            self.volumeSlider_2.setValue(0)

    def goto_beginning(self) -> None:
        """Go to the beginning of the video"""
        self.player.setPosition(0)

    def goto_end(self) -> None:
        """Go to the end of the video"""
        self.player.setPosition(self.player.duration())

    @staticmethod
    def set_marker_time(button: QPushButton, flag: bool, time_value: float, position: float) -> None:
        """Set time marker button text"""
        if flag:
            position = time_value
            ut.set_marker_time(button, position)

    def set_start_position(self, button: QPushButton) -> None:
        """Set the start position marker"""
        x = (time_value := self.timelineSlider_1.value() * .001) < self.stop_position
        y = self.start_position == .0 and self.stop_position == .0
        self.set_marker_time(button, x | y, time_value, self.start_position)

    def set_stop_position(self, button: QPushButton) -> None:
        """Set the stop position marker"""
        x = (time_value := self.timelineSlider_1.value() * .001) > self.start_position
        y = self.start_position == .0 and self.stop_position == .0
        self.set_marker_time(button, x | y, time_value, self.stop_position)

    def goto(self, marker_button: QPushButton) -> None:
        """Go to a marked position in the video"""
        if not marker_button.text():
            return

        position = ut.pos_from_marker(marker_button.text()) * 1000
        if position >= self.player.duration():
            self.player.setPosition(self.player.duration())
        elif position == 0:
            self.player.setPosition(0)
        else:
            self.player.setPosition(int(position))

    def connect_crop_worker(self) -> None:
        """Connect the signals from the crop_from_path worker to UI handlers"""
        # Build a list of widgets to disable during processing
        controls = []
        for control in [self.mediacontrolWidget_1, self.mediacontrolWidget_2]:
            controls.extend([
                control.cropButton, control.videocropButton, control.playButton,
                control.stopButton, control.stepbackButton, control.stepfwdButton,
                control.fastfwdButton, control.goto_beginingButton, control.goto_endButton,
                control.startmarkerButton, control.endmarkerButton,
                control.selectStartMarkerButton, control.selectEndMarkerButton
            ])

        widget_list = [
            self.controlWidget.widthLineEdit,
            self.controlWidget.heightLineEdit,
            self.controlWidget.sensitivityDial,
            self.controlWidget.fpctDial,
            self.controlWidget.gammaDial,
            self.controlWidget.topDial,
            self.controlWidget.bottomDial,
            self.controlWidget.leftDial,
            self.controlWidget.rightDial,
            *controls,
        ]

        # Disconnect any existing connections to avoid duplicate signals
        with contextlib.suppress(TypeError, RuntimeError):
            self.crop_worker.started.disconnect()
            self.crop_worker.finished.disconnect()
            self.crop_worker.progress.disconnect()

        # Video start connection - setup buttons
        self.crop_worker.started.connect(lambda: ut.disable_widget(*widget_list))
        self.crop_worker.started.connect(self.enable_cancel_buttons)

        # Video end connection - restore buttons
        self.crop_worker.finished.connect(lambda: ut.enable_widget(*widget_list))
        self.crop_worker.finished.connect(self.disable_cancel_buttons)
        self.crop_worker.finished.connect(self.reset_progress_bars)
        self.crop_worker.finished.connect(lambda: ut.show_message_box(self.destination))

        # Ensure progress signal is connected correctly
        self.crop_worker.progress.connect(self.update_progress)

        # Add direct connections for cancel buttons
        for control in [self.mediacontrolWidget_1, self.mediacontrolWidget_2]:
            control.cancelButton.clicked.connect(self.handle_cancel_click)

    def reset_progress_bars(self) -> None:
        """Reset both progress bars to zero"""
        self.progressBar.setValue(0)
        self.progressBar.repaint()
        self.progressBar_2.setValue(0)
        self.progressBar_2.repaint()
        QApplication.processEvents()

    def enable_cancel_buttons(self) -> None:
        """Enable both of the cancel buttons"""
        self.mediacontrolWidget_1.cancelButton.setEnabled(True)
        self.mediacontrolWidget_1.cancelButton.repaint()
        self.mediacontrolWidget_2.cancelButton.setEnabled(True)
        self.mediacontrolWidget_2.cancelButton.repaint()
        QApplication.processEvents()

    def disable_cancel_buttons(self) -> None:
        """Disable both of the cancel buttons"""
        self.mediacontrolWidget_1.cancelButton.setEnabled(False)
        self.mediacontrolWidget_1.cancelButton.repaint()
        self.mediacontrolWidget_2.cancelButton.setEnabled(False)
        self.mediacontrolWidget_2.cancelButton.repaint()
        QApplication.processEvents()

    def handle_cancel_click(self) -> None:
        """Handle cancel button clicks"""
        # Call the terminate method to stop the job
        self.crop_worker.terminate()

        # Re-enable control buttons and reset progress
        for control in [self.mediacontrolWidget_1, self.mediacontrolWidget_2]:
            control.cropButton.setEnabled(True)
            control.cropButton.repaint()
            control.videocropButton.setEnabled(True)
            control.videocropButton.repaint()
            control.cancelButton.setEnabled(False)
            control.cancelButton.repaint()

        # Reset progress bars
        self.progressBar.setValue(0)
        self.progressBar.repaint()
        self.progressBar_2.setValue(0)
        self.progressBar_2.repaint()

        QApplication.processEvents()

    def update_progress(self, x: int, y:int) -> None:
        """Update both progress bars based on crop_from_path worker progress"""
        if y <= 0:  # Prevent division by zero
            return

        # Calculate percentage
        percentage = min(100.0, (x / y) * 100.0)
        value = int(self.PROGRESSBAR_STEPS * percentage / 100.0)

        # Force UI updates on both progress bars
        for progress_bar in [self.progressBar, self.progressBar_2]:
            progress_bar.setValue(value)
            progress_bar.repaint()

        # Process events to ensure UI updates immediately
        QApplication.processEvents()

    @staticmethod
    def cancel_button_operation(cancel_button: QPushButton, *crop_buttons: QPushButton) -> None:
        """Handle cancel button operations"""
        cancel_button.setDisabled(True)
        for crop_button in crop_buttons:
            crop_button.setEnabled(True)

    def crop_frame(self) -> None:
        """Crop the current video frame"""
        def execute_crop():
            self.player.pause()

            # Disable crop_from_path buttons immediately
            for control in [self.mediacontrolWidget_1, self.mediacontrolWidget_2]:
                control.cropButton.setEnabled(False)
                control.cropButton.repaint()
                control.videocropButton.setEnabled(False)
                control.videocropButton.repaint()
                control.cancelButton.setEnabled(True)
                control.cancelButton.repaint()

            QApplication.processEvents()

            job = self.create_job(
                FunctionType.FRAME,
                video_path=Path(self.input_path),
                destination=Path(self.destination_path)
            )
            self.crop_worker.crop_frame(job, self.positionLabel_1, self.timelineSlider_1)

        # Check if source and destination are the same and warn if needed
        if Path(self.input_path).parent == Path(self.destination_path):
            match ut.show_warning(FunctionType.FRAME):
                case QMessageBox.StandardButton.Yes:
                    execute_crop()
                case _:
                    return
        else:
            execute_crop()

    def video_process(self) -> None:
        """Process the video between start and end markers"""
        x = self.mediacontrolWidget_1.selectStartMarkerButton.text()
        y = self.mediacontrolWidget_1.selectEndMarkerButton.text()

        def execute_crop():
            self.player.pause()

            # Disable crop_from_path buttons immediately
            for control in [self.mediacontrolWidget_1, self.mediacontrolWidget_2]:
                control.cropButton.setEnabled(False)
                control.cropButton.repaint()
                control.videocropButton.setEnabled(False)
                control.videocropButton.repaint()
                control.cancelButton.setEnabled(True)
                control.cancelButton.repaint()

            QApplication.processEvents()

            job = self.create_job(
                FunctionType.VIDEO,
                video_path=Path(self.input_path),
                destination=Path(self.destination_path),
                start_position=ut.pos_from_marker(x),
                stop_position=ut.pos_from_marker(y)
            )

            # Use Thread instead of Process to avoid pickling issues
            self.crop_worker.reset_task()
            thread = Thread(
                target=self.crop_worker.extract_frames,
                args=(job,),
                daemon=True
            )
            thread.start()

        # Check if source and destination are the same and warn if needed
        if Path(self.input_path).parent == Path(self.destination_path):
            match ut.show_warning(FunctionType.VIDEO):
                case QMessageBox.StandardButton.Yes:
                    execute_crop()
                case _:
                    return
        else:
            execute_crop()

    def run_batch_process(self, job: Job, *,
                          function: Callable[..., Any],
                          reset_worker_func: Callable[..., Any]) -> None:
        """Run a batch processing operation"""
        reset_worker_func()

        # Reset progress bars
        self.progressBar.setValue(0)
        self.progressBar_2.setValue(0)

        process = Process(target=function, daemon=True, args=(job,))
        process.run()

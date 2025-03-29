from multiprocessing import Process
from pathlib import Path
from threading import Thread
from typing import Optional, Callable, Any

import numpy as np
from PyQt6 import QtCore, QtGui, QtMultimedia, QtMultimediaWidgets, QtWidgets

from core import Job
from core import processing as prc
from core.croppers import VideoCropper
from core.enums import FunctionType
from file_types import registry
from line_edits import LineEditState, PathType
from ui import utils as ut
from .crop_widget import UiCropWidget
from .enums import GuiIcon
from .media_controls import UiMediaControlWidget


class UiVideoTabWidget(UiCropWidget):
    """Video tab widget with enhanced inheritance from base crop widget"""
    
    PROGRESSBAR_STEPS: int = 1_000  # From UiCropBatchWidget
    
    def __init__(self, crop_worker: VideoCropper, object_name: str, parent: QtWidgets.QWidget) -> None:
        """Initialize the video tab widget"""
        super().__init__(parent, object_name)
        self.crop_worker = crop_worker
        
        # Media player attributes
        self.vol_cache = 70
        self.rewind_timer = QtCore.QTimer()
        self.default_directory = registry.get_default_dir("video").as_posix()
        self.player = QtMultimedia.QMediaPlayer()
        self.audio = QtMultimedia.QAudioOutput()
        self.start_position, self.stop_position, self.step = .0, .0, 100
        self.speed = 0
        self.reverse = 0
        
        # Override the input line edit to use the correct path type
        self.inputLineEdit = self.create_str_line_edit("inputLineEdit", PathType.VIDEO)
        
        # Create additional UI elements
        self.progressBar = self.create_progress_bar("progressBar")
        self.progressBar_2 = self.create_progress_bar("progressBar_2")
        self.toolBox = QtWidgets.QToolBox(self)
        self.toolBox.setObjectName("toolBox")
        
        # Create pages for the toolbox
        self.page_1 = QtWidgets.QWidget()
        self.page_1.setObjectName("page_1")
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        
        # Set up page layouts
        self.verticalLayout_200 = ut.setup_vbox("verticalLayout_200", self.page_1)
        self.verticalLayout_300 = ut.setup_vbox("verticalLayout_300", self.page_2)
        
        # Create video-specific widgets
        self.videoWidget = QtMultimediaWidgets.QVideoWidget()
        self.videoWidget.setObjectName("videoWidget")
        self.videoWidget.setStyleSheet("background: #1f2c33")
        
        self.muteButton_1 = QtWidgets.QPushButton()
        self.muteButton_2 = QtWidgets.QPushButton()
        self.volumeSlider_1 = QtWidgets.QSlider()
        self.volumeSlider_2 = QtWidgets.QSlider()
        self.positionLabel_1 = QtWidgets.QLabel()
        self.positionLabel_2 = QtWidgets.QLabel()
        self.timelineSlider_1 = QtWidgets.QSlider()
        self.timelineSlider_2 = QtWidgets.QSlider()
        self.durationLabel_1 = QtWidgets.QLabel()
        self.durationLabel_2 = QtWidgets.QLabel()
        
        # Create duplicate checkboxes for page 2
        self.mfaceCheckBox_2 = self.create_checkbox("mfaceCheckBox_2")
        self.tiltCheckBox_2 = self.create_checkbox("tiltCheckBox_2")
        self.exposureCheckBox_2 = self.create_checkbox("exposureCheckBox_2")
        
        # Create media control widgets
        self.mediacontrolWidget_1 = None  # Will be initialized in setup_layouts
        self.mediacontrolWidget_2 = None  # Will be initialized in setup_layouts
        
        # Set up the main layout structure
        self.setup_layouts()
        
        self.connect_preview_updates() 

        # Connect signals
        self.connect_signals()
        
        # Set initial UI text
        self.retranslateUi()
        
        # Set initial toolbox page
        self.toolBox.setCurrentIndex(0)
        
        QtCore.QMetaObject.connectSlotsByName(self)

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

    def create_progress_bar(self, name: str, parent: Optional[QtWidgets.QWidget] = None) -> QtWidgets.QProgressBar:
        """Create a progress bar with consistent styling"""
        progress_bar = QtWidgets.QProgressBar() if parent is None else QtWidgets.QProgressBar(parent)
        progress_bar.setObjectName(name)
        progress_bar.setMinimumSize(QtCore.QSize(0, 12))
        progress_bar.setMaximumSize(QtCore.QSize(16_777_215, 12))
        progress_bar.setRange(0, self.PROGRESSBAR_STEPS)
        progress_bar.setValue(0)
        progress_bar.setTextVisible(False)
        return progress_bar

    def setup_layouts(self) -> None:
        """Set up the main layout structure"""
        # ---- Page 1: Video Player ----
        # Input file selection
        self.inputLineEdit.setParent(self.page_1)
        self.inputButton.setParent(self.page_1)
        icon = ut.create_button_icon(GuiIcon.CLAPPERBOARD)
        self.inputButton.setIcon(icon)
        
        input_layout = ut.setup_hbox("horizontalLayout_2")
        input_layout.addWidget(self.inputLineEdit)
        input_layout.addWidget(self.inputButton)
        input_layout.setStretch(0, 1)
        
        self.verticalLayout_200.addLayout(input_layout)
        
        # Main frame with video player
        frame_1 = self.create_main_frame("frame_1")
        frame_1.setParent(self.page_1)
        verticalLayout_9 = ut.setup_vbox("verticalLayout_9", frame_1)
        
        # Media controls layout
        mediaControlLayout = ut.setup_hbox("horizontalLayout_1")
        
        # Mute button
        self.muteButton_1.setObjectName("muteButton_1")
        self.muteButton_1.setMinimumSize(QtCore.QSize(30, 30))
        self.muteButton_1.setMaximumSize(QtCore.QSize(30, 30))
        self.muteButton_1.setBaseSize(QtCore.QSize(30, 30))
        self.muteButton_1.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_MUTE))
        mediaControlLayout.addWidget(self.muteButton_1)
        
        # Volume slider
        self.volumeSlider_1.setObjectName("volumeSlider_1")
        size_policy3 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy3.setHorizontalStretch(0)
        size_policy3.setVerticalStretch(0)
        size_policy3.setHeightForWidth(self.volumeSlider_1.sizePolicy().hasHeightForWidth())
        self.volumeSlider_1.setSizePolicy(size_policy3)
        self.volumeSlider_1.setMinimumSize(QtCore.QSize(0, 30))
        self.volumeSlider_1.setMaximumSize(QtCore.QSize(16_777_215, 30))
        self.volumeSlider_1.setMinimum(-1)
        self.volumeSlider_1.setMaximum(100)
        self.volumeSlider_1.setValue(70)
        self.volumeSlider_1.setOrientation(QtCore.Qt.Orientation.Horizontal)
        mediaControlLayout.addWidget(self.volumeSlider_1)
        
        # Position label
        self.positionLabel_1.setObjectName("positionLabel_1")
        size_policy4 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy4.setHorizontalStretch(0)
        size_policy4.setVerticalStretch(0)
        size_policy4.setHeightForWidth(self.positionLabel_1.sizePolicy().hasHeightForWidth())
        self.positionLabel_1.setSizePolicy(size_policy4)
        self.positionLabel_1.setMinimumSize(QtCore.QSize(0, 30))
        self.positionLabel_1.setMaximumSize(QtCore.QSize(16_777_215, 30))
        mediaControlLayout.addWidget(self.positionLabel_1)
        
        # Timeline slider
        self.timelineSlider_1.setObjectName("timelineSlider_1")
        size_policy5 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy5.setHorizontalStretch(1)
        size_policy5.setVerticalStretch(0)
        size_policy5.setHeightForWidth(self.timelineSlider_1.sizePolicy().hasHeightForWidth())
        self.timelineSlider_1.setSizePolicy(size_policy5)
        self.timelineSlider_1.setMinimumSize(QtCore.QSize(0, 30))
        self.timelineSlider_1.setMaximumSize(QtCore.QSize(16_777_215, 30))
        self.timelineSlider_1.setOrientation(QtCore.Qt.Orientation.Horizontal)
        mediaControlLayout.addWidget(self.timelineSlider_1)
        
        # Duration label
        self.durationLabel_1.setObjectName("durationLabel_1")
        size_policy4.setHeightForWidth(self.durationLabel_1.sizePolicy().hasHeightForWidth())
        self.durationLabel_1.setSizePolicy(size_policy4)
        self.durationLabel_1.setMinimumSize(QtCore.QSize(0, 30))
        self.durationLabel_1.setMaximumSize(QtCore.QSize(16_777_215, 30))
        mediaControlLayout.addWidget(self.durationLabel_1)
        
        # Horizontal spacer
        horizontalSpacer_1 = QtWidgets.QSpacerItem(
            20, 20, 
            QtWidgets.QSizePolicy.Policy.Fixed,
            QtWidgets.QSizePolicy.Policy.Minimum
        )
        mediaControlLayout.addItem(horizontalSpacer_1)
        
        # Checkboxes
        self.mfaceCheckBox.setParent(frame_1)
        self.tiltCheckBox.setParent(frame_1)
        self.exposureCheckBox.setParent(frame_1)
        mediaControlLayout.addWidget(self.mfaceCheckBox)
        mediaControlLayout.addWidget(self.tiltCheckBox)
        mediaControlLayout.addWidget(self.exposureCheckBox)
        
        mediaControlLayout.setStretch(0, 1)
        mediaControlLayout.setStretch(1, 3)
        
        verticalLayout_9.addLayout(mediaControlLayout)
        
        # Video widget
        self.videoWidget.setParent(frame_1)
        self.size_policy_expand_expand.setHeightForWidth(self.videoWidget.sizePolicy().hasHeightForWidth())
        self.videoWidget.setSizePolicy(self.size_policy_expand_expand)
        self.videoWidget.setMinimumSize(QtCore.QSize(200, 200))
        
        verticalLayout_9.addWidget(self.videoWidget)
        
        # Media control widget 1
        self.mediacontrolWidget_1 = UiMediaControlWidget(frame_1, self.player, self.crop_worker)
        self.mediacontrolWidget_1.setObjectName("mediacontrolWidget_1")
        verticalLayout_9.addWidget(self.mediacontrolWidget_1)
        
        # Progress bar
        self.progressBar.setParent(frame_1)
        verticalLayout_9.addWidget(self.progressBar)
        
        verticalLayout_9.setStretch(1, 1)
        
        self.verticalLayout_200.addWidget(frame_1)
        
        # Destination selection
        self.destinationLineEdit.setParent(self)
        self.destinationButton.setParent(self)
        self.setup_destination_layout(self.horizontalLayout_3)
        self.verticalLayout_200.addLayout(self.horizontalLayout_3)
        
        # Add page to toolbox
        self.toolBox.addItem(self.page_1, "Video Player")
        
        # ---- Page 2: Crop View ----
        # Main frame with crop preview
        frame_2 = self.create_main_frame("frame_2")
        frame_2.setParent(self.page_2)
        verticalLayout_10 = ut.setup_vbox("verticalLayout_10", frame_2)
        
        # Media controls layout for page 2
        mediaControlLayout2 = ut.setup_hbox("horizontalLayout_5")
        
        # Mute button
        self.muteButton_2.setObjectName("muteButton_2")
        self.muteButton_2.setMinimumSize(QtCore.QSize(30, 30))
        self.muteButton_2.setMaximumSize(QtCore.QSize(30, 30))
        self.muteButton_2.setBaseSize(QtCore.QSize(30, 30))
        self.muteButton_2.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_MUTE))
        mediaControlLayout2.addWidget(self.muteButton_2)
        
        # Volume slider
        self.volumeSlider_2.setObjectName("volumeSlider_2")
        size_policy3.setHeightForWidth(self.volumeSlider_2.sizePolicy().hasHeightForWidth())
        self.volumeSlider_2.setSizePolicy(size_policy3)
        self.volumeSlider_2.setMinimumSize(QtCore.QSize(0, 30))
        self.volumeSlider_2.setMaximumSize(QtCore.QSize(16_777_215, 30))
        self.volumeSlider_2.setMinimum(-1)
        self.volumeSlider_2.setMaximum(100)
        self.volumeSlider_2.setValue(70)
        self.volumeSlider_2.setOrientation(QtCore.Qt.Orientation.Horizontal)
        mediaControlLayout2.addWidget(self.volumeSlider_2)
        
        # Position label
        self.positionLabel_2.setObjectName("positionLabel_2")
        size_policy4.setHeightForWidth(self.positionLabel_2.sizePolicy().hasHeightForWidth())
        self.positionLabel_2.setSizePolicy(size_policy4)
        self.positionLabel_2.setMinimumSize(QtCore.QSize(0, 30))
        self.positionLabel_2.setMaximumSize(QtCore.QSize(16_777_215, 30))
        mediaControlLayout2.addWidget(self.positionLabel_2)
        
        # Timeline slider
        self.timelineSlider_2.setObjectName("timelineSlider_2")
        size_policy5.setHeightForWidth(self.timelineSlider_2.sizePolicy().hasHeightForWidth())
        self.timelineSlider_2.setSizePolicy(size_policy5)
        self.timelineSlider_2.setMinimumSize(QtCore.QSize(0, 30))
        self.timelineSlider_2.setMaximumSize(QtCore.QSize(16_777_215, 30))
        self.timelineSlider_2.setOrientation(QtCore.Qt.Orientation.Horizontal)
        mediaControlLayout2.addWidget(self.timelineSlider_2)
        
        # Duration label
        self.durationLabel_2.setObjectName("durationLabel_2")
        size_policy4.setHeightForWidth(self.durationLabel_2.sizePolicy().hasHeightForWidth())
        self.durationLabel_2.setSizePolicy(size_policy4)
        self.durationLabel_2.setMinimumSize(QtCore.QSize(0, 30))
        self.durationLabel_2.setMaximumSize(QtCore.QSize(16_777_215, 30))
        mediaControlLayout2.addWidget(self.durationLabel_2)
        
        mediaControlLayout2.setStretch(0, 1)
        mediaControlLayout2.setStretch(1, 3)
        
        verticalLayout_10.addLayout(mediaControlLayout2)
        
        # Checkbox section for page 2
        self.toggleCheckBox.setParent(frame_2)
        self.mfaceCheckBox_2.setParent(frame_2)
        self.tiltCheckBox_2.setParent(frame_2)
        self.exposureCheckBox_2.setParent(frame_2)
        
        checkboxLayout = ut.setup_hbox("horizontalLayout_4")
        checkboxLayout.addWidget(self.toggleCheckBox)
        
        # Add spacer
        horizontalSpacer_2 = QtWidgets.QSpacerItem(
            40, 20, 
            QtWidgets.QSizePolicy.Policy.Expanding,
            QtWidgets.QSizePolicy.Policy.Minimum
        )
        checkboxLayout.addItem(horizontalSpacer_2)
        
        # Add checkboxes
        checkboxLayout.addWidget(self.mfaceCheckBox_2)
        checkboxLayout.addWidget(self.tiltCheckBox_2)
        checkboxLayout.addWidget(self.exposureCheckBox_2)
        
        # Set stretch factor for spacer
        checkboxLayout.setStretch(1, 20)
        
        verticalLayout_10.addLayout(checkboxLayout)
        
        # Add preview button
        self.previewButton = QtWidgets.QPushButton("Update Preview")
        self.previewButton.setObjectName("previewButton")
        self.previewButton.setMinimumSize(QtCore.QSize(0, 40))
        self.previewButton.setMaximumSize(QtCore.QSize(16_777_215, 40))
        icon = ut.create_button_icon(GuiIcon.PICTURE)
        self.previewButton.setIcon(icon)
        self.previewButton.setIconSize(QtCore.QSize(18, 18))
        verticalLayout_10.addWidget(self.previewButton)
        
        # Image widget (preview)
        self.imageWidget.setParent(frame_2)
        verticalLayout_10.addWidget(self.imageWidget)
        
        # Media control widget 2
        self.mediacontrolWidget_2 = UiMediaControlWidget(frame_2, self.player, self.crop_worker)
        self.mediacontrolWidget_2.setObjectName("mediacontrolWidget_2")
        verticalLayout_10.addWidget(self.mediacontrolWidget_2)
        
        # Progress bar 2
        self.progressBar_2.setParent(frame_2)
        verticalLayout_10.addWidget(self.progressBar_2)
        
        # Add frame to page layout
        self.verticalLayout_300.addWidget(frame_2)
        
        # Add page to toolbox
        self.toolBox.addItem(self.page_2, "Crop View")
        
        # Add toolbox to main layout
        self.verticalLayout_100.addWidget(self.toolBox)
        
        # Set up the media player
        self.create_media_player()

    def connect_signals(self) -> None:
        """Connect widget signals to handlers"""
        # Set up checkbox synchronization between tabs
        self.tab_state_manager.synchronize_checkboxes(self.exposureCheckBox, self.exposureCheckBox_2)
        self.tab_state_manager.synchronize_checkboxes(self.tiltCheckBox, self.tiltCheckBox_2)
        self.tab_state_manager.synchronize_checkboxes(self.mfaceCheckBox, self.mfaceCheckBox_2)

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
        
        # Progress handling
        self.crop_worker.progress.connect(self.update_progress)
        
        # Media player connections
        self.volumeSlider_1.sliderMoved.connect(self.volume_slider_changed)
        self.volumeSlider_2.sliderMoved.connect(self.volume_slider_changed)
        self.timelineSlider_1.sliderMoved.connect(self.player_slider_changed)
        self.timelineSlider_2.sliderMoved.connect(self.player_slider_changed)
        self.timelineSlider_1.sliderMoved.connect(self.timelineSlider_2.setSliderPosition)
        self.timelineSlider_2.sliderMoved.connect(self.timelineSlider_1.setSliderPosition)
        
        # Button connections
        self.inputButton.clicked.connect(lambda: self.open_video())
        # self.destinationButton.clicked.connect(lambda: self.open_path(self.destinationLineEdit))
        
        # Add preview update trigger
        self.inputLineEdit.textChanged.connect(lambda: QtCore.QTimer.singleShot(1000, self.display_crop_preview))
        
        # Connect media control widgets
        for control in [self.mediacontrolWidget_1, self.mediacontrolWidget_2]:
            control.cropButton.clicked.connect(lambda: self.crop_frame())
            control.videocropButton.clicked.connect(lambda: self.video_process())
            control.cancelButton.clicked.connect(lambda: self.crop_worker.terminate())
            control.cancelButton.clicked.connect(
                lambda: self.cancel_button_operation(control.cancelButton,
                                                     control.videocropButton,
                                                     control.cropButton)
            )
            
            control.playButton.clicked.connect(lambda: self.change_playback_state())
            control.stopButton.clicked.connect(lambda: self.stop_playback())
            control.stepbackButton.clicked.connect(lambda: self.step_back())
            control.stepfwdButton.clicked.connect(lambda: self.step_forward())
            control.fastfwdButton.clicked.connect(lambda: self.fast_forward())
            control.goto_beginingButton.clicked.connect(lambda: self.goto_beginning())
            control.goto_endButton.clicked.connect(lambda: self.goto_end())
            control.selectStartMarkerButton.clicked.connect(
                lambda: self.goto(control.selectStartMarkerButton)
            )
            control.selectEndMarkerButton.clicked.connect(
                lambda: self.goto(control.selectEndMarkerButton)
            )
            
        # Audio control buttons
        self.muteButton_1.clicked.connect(lambda: self.volume_mute())
        self.muteButton_2.clicked.connect(lambda: self.volume_mute())
        
        # Media player connections
        self.audio.mutedChanged.connect(lambda: self.change_audio_icon())
        self.player.positionChanged.connect(self.position_changed)
        self.player.durationChanged.connect(self.duration_changed)
        self.player.errorOccurred.connect(self.player_error_occurred)
        
        # Register button dependencies with TabStateManager
        for control in [self.mediacontrolWidget_1, self.mediacontrolWidget_2]:
            self.tab_state_manager.register_button_dependencies(
                control.cropButton,
                {
                    self.inputLineEdit, 
                    self.destinationLineEdit, 
                    self.controlWidget.widthLineEdit,
                    self.controlWidget.heightLineEdit
                }
            )
            
            self.tab_state_manager.register_button_dependencies(
                control.videocropButton,
                {
                    self.inputLineEdit, 
                    self.destinationLineEdit, 
                    self.controlWidget.widthLineEdit,
                    self.controlWidget.heightLineEdit
                }
            )
        
        # Connect input widgets for validation tracking
        self.tab_state_manager.connect_widgets(
            self.inputLineEdit,
            self.controlWidget.widthLineEdit,
            self.controlWidget.heightLineEdit, 
            self.destinationLineEdit,
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
        
        # Connect crop worker signals
        self.connect_crop_worker()

    def retranslateUi(self) -> None:
        """Update UI text elements"""
        super().retranslateUi()
        self.inputLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", "Choose the video you want to crop", None)
        )
        self.inputButton.setText(QtCore.QCoreApplication.translate("self", "Open Video", None))
        self.destinationLineEdit.setPlaceholderText(
            QtCore.QCoreApplication.translate("self", "Choose where you want to save the cropped images", None)
        )
        self.destinationButton.setText(QtCore.QCoreApplication.translate("self", "Destination Folder", None))
        self.muteButton_1.setText("")
        self.muteButton_2.setText("")
        self.positionLabel_1.setText(QtCore.QCoreApplication.translate("self", "00:00:00", None))
        self.positionLabel_2.setText(QtCore.QCoreApplication.translate("self", "00:00:00", None))
        self.durationLabel_1.setText(QtCore.QCoreApplication.translate("self", "00:00:00", None))
        self.durationLabel_2.setText(QtCore.QCoreApplication.translate("self", "00:00:00", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_1),
                                QtCore.QCoreApplication.translate("self", "Video Player", None))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_2),
                                QtCore.QCoreApplication.translate("self", "Crop View", None))

    def display_crop_preview(self) -> None:
        """Captures the current frame and displays crop preview in the imageWidget"""
        try:
            # Only proceed if we have a valid video loaded
            if not self.inputLineEdit.text() or self.inputLineEdit.state == LineEditState.INVALID_INPUT:
                return
            
            # Get current position
            position = self.timelineSlider_1.value()
            
            # Use the optimized grab_frame method for preview
            frame = self.crop_worker.grab_frame(position, self.inputLineEdit.text(), for_preview=True)
            if frame is None:
                return
            
            # Create a job with current settings
            job = self.create_job(
                FunctionType.FRAME,
                video_path=Path(self.inputLineEdit.text()),
                destination=Path(self.destinationLineEdit.text() or ".")
            )
            
            # Process the frame
            if job.multi_face_job:
                # If multi-face is enabled, show bounding boxes on all faces
                processed_image = prc.multi_box(frame, job)
                ut.display_image_on_widget(processed_image, self.imageWidget)
            else:
                # For single face, show a crop preview
                cropped_image = prc.crop_image(frame, job, self.crop_worker.face_detection_tools)
                if cropped_image is not None:
                    # Display the cropped image
                    ut.display_image_on_widget(cropped_image, self.imageWidget)
        except Exception as e:
            print(f"Error in display_crop_preview: {e}")

    # Media player methods
    def create_media_player(self) -> None:
        """Set up the media player"""
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.videoWidget)

    def player_error_occurred(self, error: QtMultimedia.QMediaPlayer.Error) -> None:
        """Handle media player errors"""
        match error:
            case QtMultimedia.QMediaPlayer.Error.NoError:
                pass
            case _:
                self.stop_playback()
                ut.show_error_box(
                    f'{error.name} occurred while loading the video. ',
                    'Please check the video file path and try again.'
                )

    def open_video(self) -> None:
        """Open a video file dialog"""
        self.check_playback_state()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Open Video', 
            self.default_directory,
            registry.get_filter_string("video")
        )
        self.inputLineEdit.setText(file_name)
        if self.inputLineEdit.state is LineEditState.INVALID_INPUT:
            return
        self.player.setSource(QtCore.QUrl.fromLocalFile(file_name))
        self.reset_video_widgets()

    def open_dropped_video(self) -> None:
        """Handle video dropped onto the widget"""
        self.player.setSource(QtCore.QUrl.fromLocalFile(self.inputLineEdit.text()))
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
                self.muteButton_1.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_UNMUTE))
                self.muteButton_2.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_UNMUTE))
            else:
                self.muteButton_1.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_MUTE))
                self.muteButton_2.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_MUTE))

    def playback_bool(self,
                        a0: QtMultimedia.QMediaPlayer.PlaybackState = QtMultimedia.QMediaPlayer.PlaybackState.PausedState,
                        a1: QtMultimedia.QMediaPlayer.PlaybackState = QtMultimedia.QMediaPlayer.PlaybackState.StoppedState) -> \
            tuple[bool, bool]:
        """Returns a tuple of bools comparing the playback state to the Class attributes of
        PyQt6.QtMultimedia.QMediaPlayer.PlaybackState"""
        return self.player.playbackState() == a0, self.player.playbackState() == a1

    def check_playback_state(self) -> None:
        """Stops playback if in the paused state or playing state"""
        x, y = self.playback_bool(a1=QtMultimedia.QMediaPlayer.PlaybackState.PlayingState)
        if x ^ y:
            self.stop_playback()

    def change_playback_state(self) -> None:
        """Toggle between play and pause"""
        match self.player.playbackState():
            case QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
                self.player.pause()
                self.speed = 0
            case _:
                self.player.play()

    def stop_playback(self) -> None:
        """Stop playback and disable timeline sliders"""
        self.timelineSlider_1.setDisabled(True)
        self.timelineSlider_2.setDisabled(True)
        # Check paused or playing
        x, y = self.playback_bool(a1=QtMultimedia.QMediaPlayer.PlaybackState.PlayingState)
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
            qtime = ut.get_qtime(self.player.duration()).toString()
            self.durationLabel_1.setText(qtime)
            self.durationLabel_2.setText(qtime)
            self.mediacontrolWidget_1.selectEndMarkerButton.setText(qtime)
            self.mediacontrolWidget_2.selectEndMarkerButton.setText(qtime)

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
    def set_marker_time(button: QtWidgets.QPushButton, flag: bool, time_value: float, position: float) -> None:
        """Set time marker button text"""
        if flag:
            position = time_value
            ut.set_marker_time(button, position)     

    def set_start_position(self, button: QtWidgets.QPushButton) -> None:
        """Set the start position marker"""
        x = (time_value := self.timelineSlider_1.value() * .001) < self.stop_position
        y = self.start_position == .0 and self.stop_position == .0
        self.set_marker_time(button, x | y, time_value, self.start_position)

    def set_stop_position(self, button: QtWidgets.QPushButton) -> None:
        """Set the stop position marker"""
        x = (time_value := self.timelineSlider_1.value() * .001) > self.start_position
        y = self.start_position == .0 and self.stop_position == .0
        self.set_marker_time(button, x | y, time_value, self.stop_position)

    def goto(self, marker_button: QtWidgets.QPushButton) -> None:
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
        """Connect the signals from the crop worker to UI handlers"""
        # Build list of widgets to disable during processing
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
            self.inputLineEdit,
            self.destinationLineEdit,
            self.destinationButton,
            self.inputButton,
            *controls,
        ]
        # Video start connection
        self.crop_worker.started.connect(lambda: ut.disable_widget(*widget_list))
        self.crop_worker.started.connect(
            lambda: ut.enable_widget(
                self.mediacontrolWidget_1.cancelButton, 
                self.mediacontrolWidget_2.cancelButton
            )
        )

        # Video end connection
        self.crop_worker.finished.connect(lambda: ut.enable_widget(*widget_list))
        self.crop_worker.finished.connect(
            lambda: ut.disable_widget(
                self.mediacontrolWidget_1.cancelButton, 
                self.mediacontrolWidget_2.cancelButton
            )
        )
        self.crop_worker.finished.connect(lambda: ut.show_message_box(self.destination))
        self.crop_worker.progress.connect(self.update_progress)

    def update_progress(self, x: int, y:int) -> None:
        """Update the progress bars based on crop worker progress"""
        value = int(self.PROGRESSBAR_STEPS * x / y)
        self.progressBar.setValue(value)
        self.progressBar_2.setValue(value)
        QtWidgets.QApplication.processEvents()

    @staticmethod
    def cancel_button_operation(cancel_button: QtWidgets.QPushButton, *crop_buttons: QtWidgets.QPushButton) -> None:
        """Handle cancel button operations"""
        cancel_button.setDisabled(True)
        for crop_button in crop_buttons:
            crop_button.setEnabled(True)

    def crop_frame(self) -> None:
        """Crop the current video frame"""
        def execute_crop():
            self.player.pause()
            job = self.create_job(
                FunctionType.FRAME,
                video_path=Path(self.inputLineEdit.text()),
                destination=Path(self.destinationLineEdit.text())
            )
            self.crop_worker.crop_frame(job, self.positionLabel_1, self.timelineSlider_1)

        # Check if source and destination are the same and warn if needed
        if Path(self.inputLineEdit.text()).parent == Path(self.destinationLineEdit.text()):
            match ut.show_warning(FunctionType.FRAME):
                case QtWidgets.QMessageBox.StandardButton.Yes:
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
            job = self.create_job(
                FunctionType.VIDEO,
                video_path=Path(self.inputLineEdit.text()),
                destination=Path(self.destinationLineEdit.text()),
                start_position=ut.pos_from_marker(x),
                stop_position=ut.pos_from_marker(y)
            )
            self.run_batch_process(
                job, 
                function=self.crop_worker.extract_frames,
                reset_worker_func=lambda: self.crop_worker.reset_task()
            )

        # Check if source and destination are the same and warn if needed
        if Path(self.inputLineEdit.text()).parent == Path(self.destinationLineEdit.text()):
            match ut.show_warning(FunctionType.VIDEO):
                case QtWidgets.QMessageBox.StandardButton.Yes:
                    execute_crop()
                case _:
                    return
        else:
            execute_crop()

    @staticmethod
    def run_batch_process(job: Job, *,
                          function: Callable[..., Any],
                          reset_worker_func: Callable[..., Any]) -> None:
        """Run a batch processing operation"""
        reset_worker_func()
        process = Process(target=function, daemon=True, args=(job,))
        process.run()

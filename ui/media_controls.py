from PyQt6 import QtCore, QtGui, QtMultimedia, QtWidgets

from core.croppers import VideoCropper
from ui import utils as ut

from .enums import GuiIcon


class UiMediaControlWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, media_player: QtMultimedia.QMediaPlayer, crop_worker: VideoCropper):
        super().__init__(parent)
        self.media_player = media_player
        self.crop_worker = crop_worker
        self.setObjectName("MediaControlWidget")
        self.horizontalLayout = ut.setup_hbox("horizontalLayout", self)
        size_policy1 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy1.setHorizontalStretch(1)
        size_policy1.setVerticalStretch(1)

        def media_button_helper(button_name: str, icon_resource: GuiIcon) -> QtWidgets.QPushButton:
            button = ut.create_media_button(self, size_policy1, name=button_name, icon_resource=icon_resource)
            self.horizontalLayout.addWidget(button)
            return button

        self.playButton = media_button_helper("playButton", GuiIcon.MULTIMEDIA_PLAY)
        self.stopButton = media_button_helper("stopButton", GuiIcon.MULTIMEDIA_STOP)
        self.stepbackButton = media_button_helper("stepbackButton", GuiIcon.MULTIMEDIA_LEFT)
        self.stepfwdButton = media_button_helper("stepfwdButton", GuiIcon.MULTIMEDIA_RIGHT)
        self.rewindButton = media_button_helper("rewindButton", GuiIcon.MULTIMEDIA_REWIND)
        self.fastfwdButton = media_button_helper("fastfwdButton", GuiIcon.MULTIMEDIA_FASTFWD)
        self.goto_beginingButton = media_button_helper("goto_beginingButton", GuiIcon.MULTIMEDIA_BEGINNING)
        self.goto_endButton = media_button_helper("goto_endButton", GuiIcon.MULTIMEDIA_END)
        self.startmarkerButton = media_button_helper("startmarkerButton", GuiIcon.MULTIMEDIA_LEFTMARKER)
        self.endmarkerButton = media_button_helper("endmarkerButton", GuiIcon.MULTIMEDIA_RIGHTMARKER)

        self.horizontalSpacer_1 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Policy.Fixed,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_1)

        size_policy2 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy2.setHorizontalStretch(0)
        size_policy2.setVerticalStretch(0)

        self.cropButton = ut.create_media_button(self, size_policy2, name="cropButton",
                                                 icon_resource=GuiIcon.CROP)
        self.cropButton.setDisabled(True)
        self.horizontalLayout.addWidget(self.cropButton)

        self.videocropButton = ut.create_media_button(self, size_policy2, name="videocropButton",
                                                      icon_resource=GuiIcon.CLAPPERBOARD)
        self.horizontalLayout.addWidget(self.videocropButton)

        self.cancelButton = ut.create_media_button(self, size_policy2, name="cancelButton",
                                                   icon_resource=GuiIcon.CANCEL)
        self.cancelButton.setDisabled(True)
        self.horizontalLayout.addWidget(self.cancelButton)

        self.horizontalSpacer_2 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Policy.Fixed,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")

        self.label_A = ut.create_label(self, size_policy2, name="label_A", icon_resource=GuiIcon.MULTIMEDIA_LABEL_A)
        self.gridLayout.addWidget(self.label_A, 0, 0, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.selectStartMarkerButton = ut.create_marker_button(self, size_policy2, "selectStartMarkerButton")
        self.gridLayout.addWidget(self.selectStartMarkerButton, 0, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.label_B = ut.create_label(self, size_policy2, name="label_B", icon_resource=GuiIcon.MULTIMEDIA_LABEL_B)
        self.gridLayout.addWidget(self.label_B, 1, 0, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.selectEndMarkerButton = ut.create_marker_button(self, size_policy2, "selectEndMarkerButton")
        self.gridLayout.addWidget(self.selectEndMarkerButton, 1, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.gridLayout.setColumnStretch(0, 1)

        self.horizontalLayout.addLayout(self.gridLayout)

        self.playButton.setDisabled(True)
        self.stopButton.setDisabled(True)
        self.stepbackButton.setDisabled(True)
        self.stepfwdButton.setDisabled(True)
        self.fastfwdButton.setDisabled(True)
        self.rewindButton.setDisabled(True)
        self.goto_beginingButton.setDisabled(True)
        self.goto_endButton.setDisabled(True)
        self.startmarkerButton.setDisabled(True)
        self.endmarkerButton.setDisabled(True)
        self.selectEndMarkerButton.setDisabled(True)
        self.selectStartMarkerButton.setDisabled(True)
        self.videocropButton.setDisabled(True)

        self.retranslateUi()

        QtCore.QMetaObject.connectSlotsByName(self)

        self.media_player.playbackStateChanged.connect(self.player_state_changed)
        self.media_player.mediaStatusChanged.connect(self.player_status_changed)
        self.media_player.errorOccurred.connect(self.stopped_case)

        self.playButton.clicked.connect(
            lambda: ut.change_widget_state(
                True, self.stopButton, self.stepbackButton,
                self.stepfwdButton, self.fastfwdButton,
                self.goto_beginingButton, self.goto_endButton,
                self.startmarkerButton, self.endmarkerButton,
                self.selectEndMarkerButton, self.selectStartMarkerButton))

        controls = (self.cropButton, self.videocropButton, self.playButton, self.stopButton,
                    self.stepbackButton, self.stepfwdButton, self.fastfwdButton,
                    self.goto_beginingButton, self.goto_endButton, self.startmarkerButton,
                    self.endmarkerButton, self.selectStartMarkerButton, self.selectEndMarkerButton)

        # Video start connection
        self.crop_worker.started.connect(lambda: ut.disable_widget(*controls))
        self.crop_worker.started.connect(lambda: ut.enable_widget(self.cancelButton))

        # Video end connection
        self.crop_worker.finished.connect(lambda: ut.enable_widget(*controls))
        self.crop_worker.finished.connect(lambda: ut.disable_widget(self.cancelButton))


    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("self", "Form", None))
        self.playButton.setText("")
        self.stopButton.setText("")
        self.stepbackButton.setText("")
        self.stepfwdButton.setText("")
        self.rewindButton.setText("")
        self.fastfwdButton.setText("")
        self.goto_beginingButton.setText("")
        self.goto_endButton.setText("")
        self.startmarkerButton.setText("")
        self.endmarkerButton.setText("")
        self.cropButton.setText("")
        self.videocropButton.setText("")
        self.cancelButton.setText("")
        self.label_A.setText("")
        self.selectStartMarkerButton.setText(QtCore.QCoreApplication.translate("self", "00:00:00", None))
        self.label_B.setText("")
        self.selectEndMarkerButton.setText(QtCore.QCoreApplication.translate("self", "00:00:00", None))

    def playing_case(self):
        self.playButton.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_PAUSE))
        self.stopButton.setEnabled(True)
        self.stepbackButton.setEnabled(True)
        self.stepfwdButton.setEnabled(True)
        self.fastfwdButton.setEnabled(True)
        self.rewindButton.setEnabled(True)
        self.goto_beginingButton.setEnabled(True)
        self.goto_endButton.setEnabled(True)
        self.startmarkerButton.setEnabled(True)
        self.endmarkerButton.setEnabled(True)
        self.selectEndMarkerButton.setEnabled(True)
        self.selectStartMarkerButton.setEnabled(True)

    def paused_case(self):
        self.playButton.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_PLAY))
        self.stopButton.setEnabled(True)
        self.stepbackButton.setEnabled(True)
        self.stepfwdButton.setEnabled(True)
        self.fastfwdButton.setDisabled(True)
        self.rewindButton.setDisabled(True)
        self.goto_beginingButton.setEnabled(True)
        self.goto_endButton.setEnabled(True)
        self.startmarkerButton.setEnabled(True)
        self.endmarkerButton.setEnabled(True)
        self.selectEndMarkerButton.setEnabled(True)
        self.selectStartMarkerButton.setEnabled(True)

    def stopped_case(self):
        self.playButton.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_PLAY))
        self.stopButton.setDisabled(True)
        self.stepbackButton.setDisabled(True)
        self.stepfwdButton.setDisabled(True)
        self.fastfwdButton.setDisabled(True)
        self.rewindButton.setDisabled(True)
        self.goto_beginingButton.setDisabled(True)
        self.goto_endButton.setDisabled(True)
        self.startmarkerButton.setDisabled(True)
        self.endmarkerButton.setDisabled(True)
        self.selectEndMarkerButton.setDisabled(True)
        self.selectStartMarkerButton.setDisabled(True)

    def player_state_changed(self, state: QtMultimedia.QMediaPlayer.PlaybackState) -> None:
        match state:
            case QtMultimedia.QMediaPlayer.PlaybackState.PlayingState:
                self.playing_case()
            case QtMultimedia.QMediaPlayer.PlaybackState.PausedState:
                self.paused_case()
            case QtMultimedia.QMediaPlayer.PlaybackState.StoppedState:
                self.stopped_case()

    def player_status_changed(self, status: QtMultimedia.QMediaPlayer.MediaStatus) -> None:
        match status:
            case QtMultimedia.QMediaPlayer.MediaStatus.NoMedia:
                self.stopped_case()
            case QtMultimedia.QMediaPlayer.MediaStatus.LoadingMedia:
                self.player_state_changed(self.media_player.playbackState())
            case QtMultimedia.QMediaPlayer.MediaStatus.LoadedMedia:
                self.stopped_case()
            case QtMultimedia.QMediaPlayer.MediaStatus.StalledMedia:
                self.player_state_changed(self.media_player.playbackState())
            case QtMultimedia.QMediaPlayer.MediaStatus.BufferingMedia:
                self.player_state_changed(self.media_player.playbackState())
            case QtMultimedia.QMediaPlayer.MediaStatus.BufferedMedia:
                self.player_state_changed(self.media_player.playbackState())
            case QtMultimedia.QMediaPlayer.MediaStatus.EndOfMedia:
                self.stopped_case()
            case QtMultimedia.QMediaPlayer.MediaStatus.InvalidMedia:
                self.stopped_case()

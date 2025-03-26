from PyQt6 import QtCore, QtWidgets, QtGui, QtMultimedia

from core.croppers import VideoCropper
from ui import ui_utils as wf
from .enums import GuiIcon


class UiMediaControlWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget, media_player: QtMultimedia.QMediaPlayer, crop_worker: VideoCropper):
        super().__init__(parent)
        self.media_player = media_player
        self.crop_worker = crop_worker
        self.setObjectName(u"MediaControlWidget")
        self.horizontalLayout = wf.setup_hbox(u"horizontalLayout", self)
        size_policy1 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy1.setHorizontalStretch(1)
        size_policy1.setVerticalStretch(1)

        self.playButton = wf.create_media_button(self, size_policy1, name=u"playButton",
                                                 icon_resource=GuiIcon.MULTIMEDIA_PLAY)
        self.horizontalLayout.addWidget(self.playButton)

        self.stopButton = wf.create_media_button(self, size_policy1, name=u"stopButton",
                                                 icon_resource=GuiIcon.MULTIMEDIA_STOP)
        self.horizontalLayout.addWidget(self.stopButton)

        self.stepbackButton = wf.create_media_button(self, size_policy1, name=u"stepbackButton",
                                                     icon_resource=GuiIcon.MULTIMEDIA_LEFT)
        self.horizontalLayout.addWidget(self.stepbackButton)

        self.stepfwdButton = wf.create_media_button(self, size_policy1, name=u"stepfwdButton",
                                                    icon_resource=GuiIcon.MULTIMEDIA_RIGHT)
        self.horizontalLayout.addWidget(self.stepfwdButton)

        self.rewindButton = wf.create_media_button(self, size_policy1, name=u"rewindButton",
                                                   icon_resource=GuiIcon.MULTIMEDIA_REWIND)
        self.horizontalLayout.addWidget(self.rewindButton)

        self.fastfwdButton = wf.create_media_button(self, size_policy1, name=u"fastfwdButton",
                                                    icon_resource=GuiIcon.MULTIMEDIA_FASTFWD)
        self.horizontalLayout.addWidget(self.fastfwdButton)

        self.goto_beginingButton = wf.create_media_button(self, size_policy1, name=u"goto_beginingButton",
                                                          icon_resource=GuiIcon.MULTIMEDIA_BEGINING)
        self.horizontalLayout.addWidget(self.goto_beginingButton)

        self.goto_endButton = wf.create_media_button(self, size_policy1, name=u"goto_endButton",
                                                     icon_resource=GuiIcon.MULTIMEDIA_END)
        self.horizontalLayout.addWidget(self.goto_endButton)

        self.startmarkerButton = wf.create_media_button(self, size_policy1, name=u"startmarkerButton",
                                                        icon_resource=GuiIcon.MULTIMEDIA_LEFTMARKER)
        self.horizontalLayout.addWidget(self.startmarkerButton)

        self.endmarkerButton = wf.create_media_button(self, size_policy1, name=u"endmarkerButton",
                                                      icon_resource=GuiIcon.MULTIMEDIA_RIGHTMARKER)
        self.horizontalLayout.addWidget(self.endmarkerButton)

        self.horizontalSpacer_1 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Policy.Fixed,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_1)

        size_policy2 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        size_policy2.setHorizontalStretch(0)
        size_policy2.setVerticalStretch(0)

        self.cropButton = wf.create_media_button(self, size_policy2, name=u"cropButton",
                                                 icon_resource=GuiIcon.CROP)
        self.cropButton.setDisabled(True)
        self.horizontalLayout.addWidget(self.cropButton)

        self.videocropButton = wf.create_media_button(self, size_policy2, name=u"videocropButton",
                                                      icon_resource=GuiIcon.CLAPPERBOARD)
        self.horizontalLayout.addWidget(self.videocropButton)

        self.cancelButton = wf.create_media_button(self, size_policy2, name=u"cancelButton",
                                                   icon_resource=GuiIcon.CANCEL)
        self.cancelButton.setDisabled(True)
        self.horizontalLayout.addWidget(self.cancelButton)

        self.horizontalSpacer_2 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Policy.Fixed,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")

        self.label_A = wf.create_label(self, size_policy2, name=u"label_A", icon_resource=GuiIcon.MULTIMEDIA_LABEL_A)
        self.gridLayout.addWidget(self.label_A, 0, 0, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.selectStartMarkerButton = wf.create_marker_button(self, size_policy2, u"selectStartMarkerButton")
        self.gridLayout.addWidget(self.selectStartMarkerButton, 0, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.label_B = wf.create_label(self, size_policy2, name=u"label_B", icon_resource=GuiIcon.MULTIMEDIA_LABEL_B)
        self.gridLayout.addWidget(self.label_B, 1, 0, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.selectEndMarkerButton = wf.create_marker_button(self, size_policy2, u"selectEndMarkerButton")
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
        self.media_player.errorOccurred.connect(lambda: self.stopped_case())

        self.playButton.clicked.connect(
            lambda: wf.change_widget_state(
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
        self.crop_worker.started.connect(lambda: wf.disable_widget(*controls))
        self.crop_worker.started.connect(lambda: wf.enable_widget(self.cancelButton))

        # Video end connection
        self.crop_worker.finished.connect(lambda: wf.enable_widget(*controls))
        self.crop_worker.finished.connect(lambda: wf.disable_widget(self.cancelButton))


    # setupUi

    def retranslateUi(self):
        self.setWindowTitle(QtCore.QCoreApplication.translate("self", u"Form", None))
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
        self.selectStartMarkerButton.setText(QtCore.QCoreApplication.translate("self", u"00:00:00", None))
        self.label_B.setText("")
        self.selectEndMarkerButton.setText(QtCore.QCoreApplication.translate("self", u"00:00:00", None))

    def playing_case(self):
        self.playButton.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_PAUSE.value))
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
        self.playButton.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_PLAY.value))
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
        self.playButton.setIcon(QtGui.QIcon(GuiIcon.MULTIMEDIA_PLAY.value))
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

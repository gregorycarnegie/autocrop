from PyQt6 import QtCore, QtWidgets, QtGui

from core import window_functions as wf
from core.enums import GuiIcon


class UiMediaControlWidget(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget):
        super().__init__(parent)
        self.setObjectName(u"MediaControlWidget")
        self.horizontalLayout = wf.setup_hbox(u"horizontalLayout", self)
        sizePolicy1 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy1.setHorizontalStretch(1)
        sizePolicy1.setVerticalStretch(1)

        self.playButton = self.create_media_button(sizePolicy1, name=u"playButton",
                                                   icon_resource=GuiIcon.MULTIMEDIA_PLAY)
        self.horizontalLayout.addWidget(self.playButton)

        self.stopButton = self.create_media_button(sizePolicy1, name=u"stopButton",
                                                   icon_resource=GuiIcon.MULTIMEDIA_STOP)
        self.horizontalLayout.addWidget(self.stopButton)

        self.stepbackButton = self.create_media_button(sizePolicy1, name=u"stepbackButton",
                                                       icon_resource=GuiIcon.MULTIMEDIA_LEFT)
        self.horizontalLayout.addWidget(self.stepbackButton)

        self.stepfwdButton = self.create_media_button(sizePolicy1, name=u"stepfwdButton",
                                                      icon_resource=GuiIcon.MULTIMEDIA_RIGHT)
        self.horizontalLayout.addWidget(self.stepfwdButton)

        self.rewindButton = self.create_media_button(sizePolicy1, name=u"rewindButton",
                                                     icon_resource=GuiIcon.MULTIMEDIA_REWIND)
        self.horizontalLayout.addWidget(self.rewindButton)

        self.fastfwdButton = self.create_media_button(sizePolicy1, name=u"fastfwdButton",
                                                      icon_resource=GuiIcon.MULTIMEDIA_FASTFWD)
        self.horizontalLayout.addWidget(self.fastfwdButton)

        self.goto_beginingButton = self.create_media_button(sizePolicy1, name=u"goto_beginingButton",
                                                            icon_resource=GuiIcon.MULTIMEDIA_BEGINING)
        self.horizontalLayout.addWidget(self.goto_beginingButton)

        self.goto_endButton = self.create_media_button(sizePolicy1, name=u"goto_endButton",
                                                       icon_resource=GuiIcon.MULTIMEDIA_END)
        self.horizontalLayout.addWidget(self.goto_endButton)

        self.startmarkerButton = self.create_media_button(sizePolicy1, name=u"startmarkerButton",
                                                          icon_resource=GuiIcon.MULTIMEDIA_LEFTMARKER)
        self.horizontalLayout.addWidget(self.startmarkerButton)

        self.endmarkerButton = self.create_media_button(sizePolicy1, name=u"endmarkerButton",
                                                        icon_resource=GuiIcon.MULTIMEDIA_RIGHTMARKER)
        self.horizontalLayout.addWidget(self.endmarkerButton)

        self.horizontalSpacer_1 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Policy.Fixed,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_1)

        sizePolicy2 = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)

        self.cropButton = self.create_media_button(sizePolicy2, name=u"cropButton",
                                                   icon_resource=GuiIcon.CROP)
        self.cropButton.setDisabled(True)
        self.horizontalLayout.addWidget(self.cropButton)

        self.videocropButton = self.create_media_button(sizePolicy2, name=u"videocropButton",
                                                        icon_resource=GuiIcon.CLAPPERBOARD)
        self.horizontalLayout.addWidget(self.videocropButton)

        self.cancelButton = self.create_media_button(sizePolicy2, name=u"cancelButton",
                                                     icon_resource=GuiIcon.CANCEL)
        self.cancelButton.setDisabled(True)
        self.horizontalLayout.addWidget(self.cancelButton)

        self.horizontalSpacer_2 = QtWidgets.QSpacerItem(60, 20, QtWidgets.QSizePolicy.Policy.Fixed,
                                                        QtWidgets.QSizePolicy.Policy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_2)

        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")

        self.label_A = self.create_label(sizePolicy2, name=u"label_A", icon_resource=GuiIcon.MULTIMEDIA_LABEL_A)
        self.gridLayout.addWidget(self.label_A, 0, 0, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.selectStartMarkerButton = self.create_marker_button(sizePolicy2, u"selectStartMarkerButton")
        self.gridLayout.addWidget(self.selectStartMarkerButton, 0, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.label_B = self.create_label(sizePolicy2, name=u"label_B", icon_resource=GuiIcon.MULTIMEDIA_LABEL_B)
        self.gridLayout.addWidget(self.label_B, 1, 0, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.selectEndMarkerButton = self.create_marker_button(sizePolicy2, u"selectEndMarkerButton")
        self.gridLayout.addWidget(self.selectEndMarkerButton, 1, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        self.horizontalLayout.addLayout(self.gridLayout)

        self.retranslateUi()

        QtCore.QMetaObject.connectSlotsByName(self)

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

    def create_media_button(self, size_policy: QtWidgets.QSizePolicy,
                            *, name: str,
                            icon_resource: GuiIcon) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(self)
        button.setObjectName(name)
        size_policy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
        button.setSizePolicy(size_policy)
        size = QtCore.QSize(40, 40)
        button.setMinimumSize(size)
        button.setMaximumSize(size)
        button.setBaseSize(size)
        icon = QtGui.QIcon()
        icon.addFile(icon_resource.value, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        button.setIcon(icon)
        button.setIconSize(QtCore.QSize(24, 24))
        return button
    
    def create_function_button(self, size_policy: QtWidgets.QSizePolicy,
                               *, name: str,
                               icon_resource: GuiIcon) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(self)
        button.setObjectName(name)
        size_policy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
        button.setSizePolicy(size_policy)
        button.setMinimumSize(QtCore.QSize(40, 40))
        button.setMaximumSize(QtCore.QSize(16_777_215, 40))
        icon = QtGui.QIcon()
        icon.addFile(icon_resource.value, QtCore.QSize(), QtGui.QIcon.Mode.Normal, QtGui.QIcon.State.Off)
        button.setIcon(icon)
        button.setIconSize(QtCore.QSize(18, 18))
        return button

    def create_label(self, size_policy: QtWidgets.QSizePolicy,
                     *, name: str,
                     icon_resource: GuiIcon) -> QtWidgets.QLabel:
        label = QtWidgets.QLabel(self)
        label.setObjectName(name)
        size_policy.setHeightForWidth(label.sizePolicy().hasHeightForWidth())
        label.setSizePolicy(size_policy)
        size = QtCore.QSize(20, 20)
        label.setMinimumSize(size)
        label.setMaximumSize(size)
        label.setBaseSize(size)
        label.setPixmap(QtGui.QPixmap(icon_resource.value))
        label.setScaledContents(True)
        return label
    
    def create_marker_button(self, size_policy: QtWidgets.QSizePolicy,
                             name: str) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(self)
        button.setObjectName(name)
        size_policy.setHeightForWidth(button.sizePolicy().hasHeightForWidth())
        button.setSizePolicy(size_policy)
        marker_button_size = QtCore.QSize(75, 20)
        button.setMinimumSize(marker_button_size)
        button.setMaximumSize(marker_button_size)
        return button

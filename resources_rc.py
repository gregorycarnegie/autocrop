# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'formNvbzwa.ui'
##
## Created by: Qt User Interface Compiler version 6.2.3
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QMetaObject, QRect,
                            QSize, Qt)
from PySide6.QtGui import (QIcon)
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import (QFrame, QGridLayout, QHBoxLayout,
                               QLabel, QLayout, QLineEdit, QListView,
                               QMenuBar, QProgressBar, QPushButton,
                               QRadioButton, QSizePolicy, QSlider, QStatusBar,
                               QTabWidget, QVBoxLayout, QWidget, QMainWindow)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        self.ui = QUiLoader().load('resources\\forms\\form.ui')
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1343, 820)
        MainWindow.setStyleSheet(u"")
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout_7 = QVBoxLayout(self.centralwidget)
        self.verticalLayout_7.setObjectName(u"verticalLayout_7")
        self.horizontalLayout_3 = QHBoxLayout()
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_12 = QGridLayout()
        self.gridLayout_12.setObjectName(u"gridLayout_12")
        self.label_4 = QLabel(self.centralwidget)
        self.label_4.setObjectName(u"label_4")

        self.gridLayout_12.addWidget(self.label_4, 0, 0, 1, 1)

        self.lineEdit_3 = QLineEdit(self.centralwidget)
        self.lineEdit_3.setObjectName(u"lineEdit_3")
        self.lineEdit_3.setInputMethodHints(Qt.ImhDigitsOnly)

        self.gridLayout_12.addWidget(self.lineEdit_3, 1, 0, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout_12)

        self.gridLayout_13 = QGridLayout()
        self.gridLayout_13.setObjectName(u"gridLayout_13")
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setObjectName(u"label_5")

        self.gridLayout_13.addWidget(self.label_5, 0, 0, 1, 1)

        self.lineEdit_4 = QLineEdit(self.centralwidget)
        self.lineEdit_4.setObjectName(u"lineEdit_4")
        self.lineEdit_4.setInputMethodHints(Qt.ImhDigitsOnly)

        self.gridLayout_13.addWidget(self.lineEdit_4, 1, 0, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout_13)

        self.gridLayout_10 = QGridLayout()
        self.gridLayout_10.setObjectName(u"gridLayout_10")
        self.horizontalSlider_1 = QSlider(self.centralwidget)
        self.horizontalSlider_1.setObjectName(u"horizontalSlider_1")
        self.horizontalSlider_1.setMaximum(100)
        self.horizontalSlider_1.setOrientation(Qt.Horizontal)

        self.gridLayout_10.addWidget(self.horizontalSlider_1, 2, 0, 1, 1)

        self.label_10 = QLabel(self.centralwidget)
        self.label_10.setObjectName(u"label_10")

        self.gridLayout_10.addWidget(self.label_10, 1, 0, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout_10)

        self.gridLayout_14 = QGridLayout()
        self.gridLayout_14.setObjectName(u"gridLayout_14")
        self.horizontalSlider_2 = QSlider(self.centralwidget)
        self.horizontalSlider_2.setObjectName(u"horizontalSlider_2")
        self.horizontalSlider_2.setMinimum(0)
        self.horizontalSlider_2.setMaximum(2000)
        self.horizontalSlider_2.setValue(1000)
        self.horizontalSlider_2.setOrientation(Qt.Horizontal)

        self.gridLayout_14.addWidget(self.horizontalSlider_2, 1, 0, 1, 1)

        self.label_6 = QLabel(self.centralwidget)
        self.label_6.setObjectName(u"label_6")

        self.gridLayout_14.addWidget(self.label_6, 0, 0, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout_14)

        self.gridLayout_11 = QGridLayout()
        self.gridLayout_11.setObjectName(u"gridLayout_11")
        self.label_7 = QLabel(self.centralwidget)
        self.label_7.setObjectName(u"label_7")

        self.gridLayout_11.addWidget(self.label_7, 0, 0, 1, 1)

        self.horizontalSlider_3 = QSlider(self.centralwidget)
        self.horizontalSlider_3.setObjectName(u"horizontalSlider_3")
        self.horizontalSlider_3.setMaximum(100)
        self.horizontalSlider_3.setSingleStep(1)
        self.horizontalSlider_3.setValue(62)
        self.horizontalSlider_3.setOrientation(Qt.Horizontal)

        self.gridLayout_11.addWidget(self.horizontalSlider_3, 1, 0, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout_11)

        self.gridLayout_9 = QGridLayout()
        self.gridLayout_9.setObjectName(u"gridLayout_9")
        self.label_9 = QLabel(self.centralwidget)
        self.label_9.setObjectName(u"label_9")

        self.gridLayout_9.addWidget(self.label_9, 0, 0, 1, 1)

        self.horizontalSlider_4 = QSlider(self.centralwidget)
        self.horizontalSlider_4.setObjectName(u"horizontalSlider_4")
        self.horizontalSlider_4.setMaximum(100)
        self.horizontalSlider_4.setValue(50)
        self.horizontalSlider_4.setOrientation(Qt.Horizontal)

        self.gridLayout_9.addWidget(self.horizontalSlider_4, 1, 0, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout_9)

        self.gridLayout_15 = QGridLayout()
        self.gridLayout_15.setObjectName(u"gridLayout_15")
        self.pushButton_7 = QPushButton(self.centralwidget)
        self.pushButton_7.setObjectName(u"pushButton_7")

        self.gridLayout_15.addWidget(self.pushButton_7, 0, 0, 1, 1)

        self.verticalLayout_2.addLayout(self.gridLayout_15)

        self.horizontalLayout_3.addLayout(self.verticalLayout_2)

        self.verticalLayout = QVBoxLayout()
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_8 = QGridLayout()
        self.gridLayout_8.setObjectName(u"gridLayout_8")
        self.tabWidget = QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName(u"tabWidget")
        self.tab_1 = QWidget()
        self.tab_1.setObjectName(u"tab_1")
        self.verticalLayout_8 = QVBoxLayout(self.tab_1)
        self.verticalLayout_8.setObjectName(u"verticalLayout_8")
        self.verticalLayout_3 = QVBoxLayout()
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_5 = QVBoxLayout()
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4 = QHBoxLayout()
        self.horizontalLayout_4.setSpacing(0)
        self.horizontalLayout_4.setObjectName(u"horizontalLayout_4")
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_16 = QGridLayout()
        self.gridLayout_16.setObjectName(u"gridLayout_16")
        self.lineEdit_1 = QLineEdit(self.tab_1)
        self.lineEdit_1.setObjectName(u"lineEdit_1")

        self.gridLayout_16.addWidget(self.lineEdit_1, 0, 1, 1, 1)

        self.reload_pushButton_1 = QPushButton(self.tab_1)
        self.reload_pushButton_1.setObjectName(u"reload_pushButton_1")
        icon = QIcon()
        icon.addFile(u"../icons/reload.svg", QSize(), QIcon.Normal, QIcon.Off)
        self.reload_pushButton_1.setIcon(icon)

        self.gridLayout_16.addWidget(self.reload_pushButton_1, 0, 0, 1, 1)

        self.horizontalLayout_4.addLayout(self.gridLayout_16)

        self.gridLayout_17 = QGridLayout()
        self.gridLayout_17.setObjectName(u"gridLayout_17")
        self.FilePushButton = QPushButton(self.tab_1)
        self.FilePushButton.setObjectName(u"FilePushButton")
        self.FilePushButton.setStyleSheet(u"background: #b1b1b1")

        self.gridLayout_17.addWidget(self.FilePushButton, 0, 0, 1, 1)

        self.horizontalLayout_4.addLayout(self.gridLayout_17)

        self.horizontalLayout_4.setStretch(0, 10)
        self.horizontalLayout_4.setStretch(1, 1)

        self.verticalLayout_5.addLayout(self.horizontalLayout_4)

        self.gridLayout_19 = QGridLayout()
        self.gridLayout_19.setObjectName(u"gridLayout_19")
        self.gridLayout_19.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel(self.tab_1)
        self.label.setObjectName(u"label")
        sizePolicy = QSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy)
        self.label.setStyleSheet(u"background: #1f2c33")
        self.label.setFrameShape(QFrame.NoFrame)
        self.label.setScaledContents(True)
        self.label.setAlignment(Qt.AlignCenter)

        self.gridLayout_19.addWidget(self.label, 0, 0, 1, 1)

        self.verticalLayout_5.addLayout(self.gridLayout_19)

        self.gridLayout_20 = QGridLayout()
        self.gridLayout_20.setObjectName(u"gridLayout_20")
        self.gridLayout_20.setContentsMargins(0, 0, 0, 0)
        self.CropPushButton_1 = QPushButton(self.tab_1)
        self.CropPushButton_1.setObjectName(u"CropPushButton_1")
        sizePolicy1 = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.CropPushButton_1.sizePolicy().hasHeightForWidth())
        self.CropPushButton_1.setSizePolicy(sizePolicy1)
        self.CropPushButton_1.setStyleSheet(u"background: #b1b1b1")

        self.gridLayout_20.addWidget(self.CropPushButton_1, 0, 0, 1, 1)

        self.verticalLayout_5.addLayout(self.gridLayout_20)

        self.verticalLayout_5.setStretch(0, 1)
        self.verticalLayout_5.setStretch(1, 10)
        self.verticalLayout_5.setStretch(2, 1)

        self.verticalLayout_3.addLayout(self.verticalLayout_5)

        self.verticalLayout_8.addLayout(self.verticalLayout_3)

        self.tabWidget.addTab(self.tab_1, "")
        self.tab_2 = QWidget()
        self.tab_2.setObjectName(u"tab_2")
        self.verticalLayout_9 = QVBoxLayout(self.tab_2)
        self.verticalLayout_9.setObjectName(u"verticalLayout_9")
        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setSpacing(0)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_6 = QVBoxLayout()
        self.verticalLayout_6.setSpacing(0)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_5 = QHBoxLayout()
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_18 = QGridLayout()
        self.gridLayout_18.setObjectName(u"gridLayout_18")
        self.gridLayout_18.setVerticalSpacing(6)
        self.lineEdit_2 = QLineEdit(self.tab_2)
        self.lineEdit_2.setObjectName(u"lineEdit_2")

        self.gridLayout_18.addWidget(self.lineEdit_2, 1, 1, 1, 1)

        self.reload_pushButton_2 = QPushButton(self.tab_2)
        self.reload_pushButton_2.setObjectName(u"reload_pushButton_2")
        self.reload_pushButton_2.setIcon(icon)

        self.gridLayout_18.addWidget(self.reload_pushButton_2, 1, 0, 1, 1)

        self.horizontalLayout_5.addLayout(self.gridLayout_18)

        self.gridLayout_21 = QGridLayout()
        self.gridLayout_21.setObjectName(u"gridLayout_21")
        self.FolderPushButton = QPushButton(self.tab_2)
        self.FolderPushButton.setObjectName(u"FolderPushButton")
        self.FolderPushButton.setStyleSheet(u"background: #b1b1b1")

        self.gridLayout_21.addWidget(self.FolderPushButton, 0, 0, 1, 1)

        self.horizontalLayout_5.addLayout(self.gridLayout_21)

        self.horizontalLayout_5.setStretch(0, 10)
        self.horizontalLayout_5.setStretch(1, 1)

        self.verticalLayout_6.addLayout(self.horizontalLayout_5)

        self.horizontalLayout_6 = QHBoxLayout()
        self.horizontalLayout_6.setSpacing(0)
        self.horizontalLayout_6.setObjectName(u"horizontalLayout_6")
        self.horizontalLayout_6.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_22 = QGridLayout()
        self.gridLayout_22.setObjectName(u"gridLayout_22")
        self.label_2 = QLabel(self.tab_2)
        self.label_2.setObjectName(u"label_2")
        sizePolicy.setHeightForWidth(self.label_2.sizePolicy().hasHeightForWidth())
        self.label_2.setSizePolicy(sizePolicy)
        self.label_2.setStyleSheet(u"background: #1f2c33")
        self.label_2.setScaledContents(True)
        self.label_2.setAlignment(Qt.AlignCenter)

        self.gridLayout_22.addWidget(self.label_2, 0, 0, 1, 1)

        self.horizontalLayout_6.addLayout(self.gridLayout_22)

        self.gridLayout_23 = QGridLayout()
        self.gridLayout_23.setObjectName(u"gridLayout_23")
        self.listView = QListView(self.tab_2)
        self.listView.setObjectName(u"listView")

        self.gridLayout_23.addWidget(self.listView, 1, 0, 1, 1)

        self.horizontalLayout_6.addLayout(self.gridLayout_23)

        self.horizontalLayout_6.setStretch(0, 3)
        self.horizontalLayout_6.setStretch(1, 1)

        self.verticalLayout_6.addLayout(self.horizontalLayout_6)

        self.horizontalLayout_7 = QHBoxLayout()
        self.horizontalLayout_7.setObjectName(u"horizontalLayout_7")
        self.horizontalLayout_7.setContentsMargins(0, 10, 0, 10)
        self.gridLayout_25 = QGridLayout()
        self.gridLayout_25.setObjectName(u"gridLayout_25")
        self.CropPushButton_2 = QPushButton(self.tab_2)
        self.CropPushButton_2.setObjectName(u"CropPushButton_2")
        sizePolicy1.setHeightForWidth(self.CropPushButton_2.sizePolicy().hasHeightForWidth())
        self.CropPushButton_2.setSizePolicy(sizePolicy1)
        self.CropPushButton_2.setStyleSheet(u"background: #b1b1b1")

        self.gridLayout_25.addWidget(self.CropPushButton_2, 0, 0, 1, 1)

        self.horizontalLayout_7.addLayout(self.gridLayout_25)

        self.gridLayout_26 = QGridLayout()
        self.gridLayout_26.setObjectName(u"gridLayout_26")
        self.CancelPushButton = QPushButton(self.tab_2)
        self.CancelPushButton.setObjectName(u"CancelPushButton")
        sizePolicy1.setHeightForWidth(self.CancelPushButton.sizePolicy().hasHeightForWidth())
        self.CancelPushButton.setSizePolicy(sizePolicy1)
        self.CancelPushButton.setStyleSheet(u"background: #b1b1b1")

        self.gridLayout_26.addWidget(self.CancelPushButton, 0, 0, 1, 1)

        self.horizontalLayout_7.addLayout(self.gridLayout_26)

        self.verticalLayout_6.addLayout(self.horizontalLayout_7)

        self.gridLayout_28 = QGridLayout()
        self.gridLayout_28.setObjectName(u"gridLayout_28")
        self.gridLayout_28.setContentsMargins(-1, 10, -1, 10)
        self.progressBar = QProgressBar(self.tab_2)
        self.progressBar.setObjectName(u"progressBar")
        sizePolicy1.setHeightForWidth(self.progressBar.sizePolicy().hasHeightForWidth())
        self.progressBar.setSizePolicy(sizePolicy1)
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(False)

        self.gridLayout_28.addWidget(self.progressBar, 0, 0, 1, 1)

        self.verticalLayout_6.addLayout(self.gridLayout_28)

        self.verticalLayout_6.setStretch(0, 3)
        self.verticalLayout_6.setStretch(1, 30)
        self.verticalLayout_6.setStretch(2, 3)
        self.verticalLayout_6.setStretch(3, 1)

        self.verticalLayout_4.addLayout(self.verticalLayout_6)

        self.verticalLayout_9.addLayout(self.verticalLayout_4)

        self.tabWidget.addTab(self.tab_2, "")

        self.gridLayout_8.addWidget(self.tabWidget, 0, 0, 1, 1)

        self.verticalLayout.addLayout(self.gridLayout_8)

        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.lineEdit_5 = QLineEdit(self.centralwidget)
        self.lineEdit_5.setObjectName(u"lineEdit_5")

        self.horizontalLayout.addWidget(self.lineEdit_5)

        self.DestinationPushButton_1 = QPushButton(self.centralwidget)
        self.DestinationPushButton_1.setObjectName(u"DestinationPushButton_1")

        self.horizontalLayout.addWidget(self.DestinationPushButton_1)

        self.verticalLayout.addLayout(self.horizontalLayout)

        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setObjectName(u"horizontalLayout_2")
        self.horizontalLayout_2.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_6 = QGridLayout()
        self.gridLayout_6.setObjectName(u"gridLayout_6")
        self.label_3 = QLabel(self.centralwidget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setAlignment(Qt.AlignCenter)

        self.gridLayout_6.addWidget(self.label_3, 0, 0, 1, 1)

        self.horizontalLayout_2.addLayout(self.gridLayout_6)

        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.radioButton_1 = QRadioButton(self.centralwidget)
        self.radioButton_1.setObjectName(u"radioButton_1")
        self.radioButton_1.setChecked(True)

        self.gridLayout.addWidget(self.radioButton_1, 0, 0, 1, 1)

        self.horizontalLayout_2.addLayout(self.gridLayout)

        self.gridLayout_2 = QGridLayout()
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.radioButton_2 = QRadioButton(self.centralwidget)
        self.radioButton_2.setObjectName(u"radioButton_2")

        self.gridLayout_2.addWidget(self.radioButton_2, 0, 0, 1, 1)

        self.horizontalLayout_2.addLayout(self.gridLayout_2)

        self.gridLayout_4 = QGridLayout()
        self.gridLayout_4.setObjectName(u"gridLayout_4")
        self.radioButton_3 = QRadioButton(self.centralwidget)
        self.radioButton_3.setObjectName(u"radioButton_3")

        self.gridLayout_4.addWidget(self.radioButton_3, 0, 0, 1, 1)

        self.horizontalLayout_2.addLayout(self.gridLayout_4)

        self.gridLayout_3 = QGridLayout()
        self.gridLayout_3.setObjectName(u"gridLayout_3")
        self.radioButton_4 = QRadioButton(self.centralwidget)
        self.radioButton_4.setObjectName(u"radioButton_4")

        self.gridLayout_3.addWidget(self.radioButton_4, 0, 0, 1, 1)

        self.horizontalLayout_2.addLayout(self.gridLayout_3)

        self.gridLayout_7 = QGridLayout()
        self.gridLayout_7.setObjectName(u"gridLayout_7")
        self.radioButton_5 = QRadioButton(self.centralwidget)
        self.radioButton_5.setObjectName(u"radioButton_5")

        self.gridLayout_7.addWidget(self.radioButton_5, 0, 0, 1, 1)

        self.horizontalLayout_2.addLayout(self.gridLayout_7)

        self.horizontalLayout_2.setStretch(0, 4)
        self.horizontalLayout_2.setStretch(1, 1)
        self.horizontalLayout_2.setStretch(2, 1)
        self.horizontalLayout_2.setStretch(3, 1)
        self.horizontalLayout_2.setStretch(4, 1)
        self.horizontalLayout_2.setStretch(5, 1)

        self.verticalLayout.addLayout(self.horizontalLayout_2)

        self.verticalLayout.setStretch(0, 10)
        self.verticalLayout.setStretch(2, 1)

        self.horizontalLayout_3.addLayout(self.verticalLayout)

        self.horizontalLayout_3.setStretch(0, 2)
        self.horizontalLayout_3.setStretch(1, 9)

        self.verticalLayout_7.addLayout(self.horizontalLayout_3)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(MainWindow)
        self.menubar.setObjectName(u"menubar")
        self.menubar.setGeometry(QRect(0, 0, 1343, 22))
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        self.tabWidget.setCurrentIndex(0)

        QMetaObject.connectSlotsByName(MainWindow)

    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"MainWindow", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Width (px)", None))
        self.lineEdit_3.setText(QCoreApplication.translate("MainWindow", u"316", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Height (px)", None))
        self.lineEdit_4.setText(QCoreApplication.translate("MainWindow", u"476", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Padding (px)", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Gamma", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Face %", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Confidence %", None))
        self.pushButton_7.setText(QCoreApplication.translate("MainWindow", u"About Face Cropper", None))
        self.lineEdit_1.setText("")
        self.reload_pushButton_1.setText("")
        self.FilePushButton.setText(QCoreApplication.translate("MainWindow", u"Open File", None))
        self.label.setText("")
        self.CropPushButton_1.setText(QCoreApplication.translate("MainWindow", u"Crop", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_1),
                                  QCoreApplication.translate("MainWindow", u"File Crop", None))
        self.lineEdit_2.setText("")
        self.reload_pushButton_2.setText("")
        self.FolderPushButton.setText(QCoreApplication.translate("MainWindow", u"Open Folder", None))
        self.label_2.setText("")
        self.CropPushButton_2.setText(QCoreApplication.translate("MainWindow", u"Crop", None))
        self.CancelPushButton.setText(QCoreApplication.translate("MainWindow", u"Cancel", None))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2),
                                  QCoreApplication.translate("MainWindow", u"Folder Crop", None))
        self.lineEdit_5.setText("")
        self.DestinationPushButton_1.setText(QCoreApplication.translate("MainWindow", u"Destination Folder", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"Convert Image?", None))
        self.radioButton_1.setText(QCoreApplication.translate("MainWindow", u"No", None))
        self.radioButton_2.setText(QCoreApplication.translate("MainWindow", u".bmp", None))
        self.radioButton_3.setText(QCoreApplication.translate("MainWindow", u".jpg", None))
        self.radioButton_4.setText(QCoreApplication.translate("MainWindow", u".png", None))
        self.radioButton_5.setText(QCoreApplication.translate("MainWindow", u".webp", None))
    # retranslateUi

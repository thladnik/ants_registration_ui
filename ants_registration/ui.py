from __future__ import annotations

from pathlib import Path, WindowsPath
from typing import List

from pyqtgraph.Qt import QtCore, QtWidgets
from pyqtgraph.Qt.QtWidgets import *


class IntSlider(QWidget):
    updated = QtCore.Signal()

    def __init__(self, name: str, minval=0, maxval=100, parent=None):
        QWidget.__init__(self, parent)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        top_widget = QWidget()
        top_layout = QHBoxLayout()
        top_widget.setLayout(top_layout)
        main_layout.addWidget(top_widget)

        bottom_widget = QWidget()
        bottom_layout = QHBoxLayout()
        bottom_widget.setLayout(bottom_layout)
        main_layout.addWidget(bottom_widget)

        self.label = QLabel(name)
        top_layout.addWidget(self.label)

        self.slider = QSlider(parent=bottom_widget)
        self.slider.setMinimum(minval)
        self.slider.setSingleStep(1)
        self.slider.setMaximum(maxval)
        bottom_layout.addWidget(self.slider)

        self.update_btn = QPushButton('Update')
        self.update_btn.clicked.connect(lambda: self.updated.emit())
        bottom_layout.addWidget(self.update_btn)


class StackWidget(QGroupBox):
    path_changed = QtCore.Signal(str)

    def __init__(self, name: str, parent=None):
        QGroupBox.__init__(self, name, parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.setAcceptDrops(True)
        self.setLayout(QHBoxLayout())

        self.main_widget = QWidget()
        self.layout().addWidget(self.main_widget)

        main_layout = QVBoxLayout()
        self.main_widget.setLayout(main_layout)

        # Non-editable field to display the selected path
        self.file_path_display = QLineEdit()
        self.file_path_display.setReadOnly(True)
        self.path_changed.connect(self.file_path_display.setText)
        main_layout.addWidget(self.file_path_display)

        # Add file selection options
        self.file_selection = QWidget()
        file_selection_layout = QHBoxLayout()
        self.file_selection.setLayout(file_selection_layout)
        main_layout.addWidget(self.file_selection)
        self.file_button = QPushButton('Select file')
        self.file_button.clicked.connect(self.select_file)
        file_selection_layout.addWidget(self.file_button)
        file_selection_layout.addWidget(QLabel('or'))
        self.folder_button = QPushButton('Select folder')
        self.folder_button.clicked.connect(self.select_folder)
        file_selection_layout.addWidget(self.folder_button)
        file_selection_layout.addWidget(QLabel('or'))
        lbl_dragdrop = QLabel('drag and drop file/folder')
        lbl_dragdrop.setStyleSheet('font-weight:bold;')
        file_selection_layout.addWidget(lbl_dragdrop)
        file_selection_layout.addStretch()

        self.resolution = StackResolution(self)
        main_layout.addWidget(self.resolution)

        main_layout.addStretch()

        self.drop_widget = QWidget()
        self.drop_widget.hide()
        self.drop_widget.setLayout(QVBoxLayout())
        self.drop_text = QLabel('')
        self.drop_text.setStyleSheet('font-weight:bold;')
        self.drop_text.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.drop_widget.layout().addWidget(self.drop_text)
        self.layout().addWidget(self.drop_widget)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self.setFixedSize(self.size())
            self.drop_text.setText('Drop file or folder here to load...')
            self.main_widget.hide()
            self.drop_widget.show()
            event.accept()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.reset_widgets()
        event.accept()

    def dropEvent(self, event):
        self.drop_text.setText('Loading data...')

        QtWidgets.QApplication.instance().processEvents()

        for url in event.mimeData().urls():
            self.new_path(url.path())

        self.reset_widgets()

    def reset_widgets(self):
        self.main_widget.show()
        self.drop_widget.hide()

        self.setMaximumSize(9999, 9999)
        self.setMinimumSize(0, 0)

    def select_file(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select stack...', '', 'TIF Files (*.tif *.tiff *.TIF *.TIFF)')
        if not path:
            return

        self.new_path(path)

    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, 'Select folder...')
        if not path:
            return

        self.new_path(path)

    def new_path(self, path: str):

        path = Path(path)
        print(path.as_posix())

        # For drag and drop operations, there's a leading slash on Windows systems, remove it:
        if isinstance(path, WindowsPath):
            path = path.as_posix().lstrip('/')
        else:
            path = path.as_posix()

        print('>', path)

        self.path_changed.emit(path)


class StackResolution(QGroupBox):

    changed = QtCore.Signal(dict)

    def __init__(self, parent=None):
        QGroupBox.__init__(self, 'Resolution', parent)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Minimum, QtWidgets.QSizePolicy.Policy.Minimum)
        self.setMinimumSize(150, 100)
        self.setMaximumWidth(300)

        main_layout = QGridLayout()
        self.setLayout(main_layout)

        self.edit_fields = []

        # X
        self.x_label = QLabel('X')
        main_layout.addWidget(self.x_label, 0, 0)
        self.x_edit = QLineEdit()
        self.edit_fields.append(self.x_edit)
        self.x_edit.editingFinished.connect(lambda: self.resolution_updated('x'))
        main_layout.addWidget(self.x_edit, 0, 1)
        # Make X^3 button
        self.cubic_btn = QPushButton('x^3')
        self.cubic_btn.clicked.connect(self.make_cubic)
        main_layout.addWidget(self.cubic_btn, 0, 2)

        # Y
        self.y_label = QLabel('Y')
        main_layout.addWidget(self.y_label, 1, 0)
        self.y_edit = QLineEdit()
        self.edit_fields.append(self.y_edit)
        self.y_edit.editingFinished.connect(lambda: self.resolution_updated('y'))
        main_layout.addWidget(self.y_edit, 1, 1)
        # Z
        self.z_label = QLabel('Z')
        main_layout.addWidget(self.z_label, 2, 0)
        self.z_edit = QLineEdit()
        self.edit_fields.append(self.z_edit)
        self.z_edit.editingFinished.connect(lambda: self.resolution_updated('z'))
        main_layout.addWidget(self.z_edit, 2, 1)

    def set_resolution(self, *args: List[float]):
        for i, v in enumerate(args):
            self.edit_fields[i].setText(str(v))
            self.edit_fields[i].editingFinished.emit()

    def get_edit_field(self, axis: str) -> QLineEdit:
        return getattr(self, f'{axis}_edit')

    def resolution_updated(self, axis: str):
        # Get and parse value
        value = eval(self.get_edit_field(axis).text())

        # Set value to data
        print(f'Set {axis} resultion to {value}')
        self.changed.emit({axis: value})

    def make_cubic(self):
        try:
            value = eval(self.x_edit.text())
        except Exception as _:
            raise ValueError('X dimension needs to be set to apply cubic scaling')
        else:
            self.y_edit.setText(str(value))
            self.y_edit.editingFinished.emit()
            self.z_edit.setText(str(value))
            self.z_edit.editingFinished.emit()


class DynamicWidget(QWidget):

    visibility_changed = QtCore.Signal(bool)

    def __init__(self):
        QWidget.__init__(self)

    def show(self):
        if not self.isVisible():
            self.visibility_changed.emit(True)

        QWidget.show(self)

    def hide(self):
        if self.isVisible():
            self.visibility_changed.emit(False)

        QWidget.hide(self)

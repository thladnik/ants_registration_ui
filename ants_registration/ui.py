from __future__ import annotations

from typing import List

from PySide6 import QtCore
from PySide6.QtWidgets import *


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

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.file_selection = QWidget()
        file_selection_layout = QHBoxLayout()
        self.file_selection.setLayout(file_selection_layout)
        main_layout.addWidget(self.file_selection)

        # Non-editable field to display the selected file path
        self.file_path_display = QLineEdit()
        self.file_path_display.setReadOnly(True)
        self.path_changed.connect(self.file_path_display.setText)
        file_selection_layout.addWidget(self.file_path_display)

        # Button to open file dialog
        self.file_button = QPushButton('Select stack...')
        self.file_button.clicked.connect(self.select_file)
        file_selection_layout.addWidget(self.file_button)

        self.resolution = StackResolution(self)
        main_layout.addWidget(self.resolution)

        main_layout.addStretch()

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select stack...', '', 'TIF Files (*.tif *.tiff)')
        if not file_path:
            return

        self.path_changed.emit(file_path)

    def select_suite2p_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, 'Select suite2p folder...')
        if not folder_path:
            return

        self.path_changed.emit(folder_path)


class StackResolution(QWidget):

    changed = QtCore.Signal(dict)

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        self.edit_fields = []

        self.x_edit = QLineEdit()
        self.edit_fields.append(self.x_edit)
        self.x_label = QLabel('Res X [my]')
        main_layout.addWidget(self.x_label)
        self.x_edit.editingFinished.connect(lambda: self.resolution_updated('x'))
        main_layout.addWidget(self.x_edit)
        self.y_edit = QLineEdit()
        self.edit_fields.append(self.y_edit)
        self.y_label = QLabel('Res Y [my]')
        main_layout.addWidget(self.y_label)
        self.y_edit.editingFinished.connect(lambda: self.resolution_updated('y'))
        main_layout.addWidget(self.y_edit)
        self.z_edit = QLineEdit()
        self.edit_fields.append(self.z_edit)
        self.z_label = QLabel('Res Z [my]')
        main_layout.addWidget(self.z_label)
        self.z_edit.editingFinished.connect(lambda: self.resolution_updated('z'))
        main_layout.addWidget(self.z_edit)
        self.cubic_btn = QPushButton('x^3')
        self.cubic_btn.clicked.connect(self.make_cubic)
        main_layout.addWidget(self.cubic_btn)

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

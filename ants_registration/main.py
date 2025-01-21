from __future__ import annotations

import multiprocessing
import os
import pprint
import sys
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Union

import ants
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import yaml
from PySide6 import QtCore
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt

from ants_registration import widgets
from ants_registration.ui import StackWidget, DynamicWidget

# Import registration instance once in beginning
from ants_registration.registration import registration


class Window(QMainWindow):
    instance: Window

    dynamic_widgets: Dict[str, QWidget] = {}
    dynamic_buttons: Dict[str, QPushButton] = {}

    def __init__(self, *args, **kwarg):
        QMainWindow.__init__(self, *args, **kwarg)
        self.__class__.instance = self

        self.setWindowTitle('ANTs registration interface')
        self.setMinimumSize(1400, 1200)

        # Central Widget
        central_widget = QWidget()
        central_widget.setContentsMargins(0, 0, 0, 0)
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left Panel (Control Panel)
        self.control_panel = ControlPanel(parent=self)
        self.control_panel.hide()  # Disable for now
        main_layout.addWidget(self.control_panel)

        # Right panel
        self.right_layout = QVBoxLayout()

        # Info panel
        self.info_panel = QWidget()
        # self.info_panel.setContentsMargins(0, 0, 0, 0)
        info_layout = QHBoxLayout()
        info_layout.setSpacing(0)
        info_layout.setContentsMargins(0, 0, 0, 0)
        self.info_panel.setLayout(info_layout)
        self.info_panel.setMinimumSize(1, 200)
        self.info_panel.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        self.right_layout.addWidget(self.info_panel)

        # Add fixed stack widget
        self.fixed_stack = StackWidget('Reference stack')
        info_layout.addWidget(self.fixed_stack)

        # Add moving stack widget
        self.moving_stack = StackWidget('Moving stack')
        info_layout.addWidget(self.moving_stack)

        # Create dynamic widgets
        # 3D volume widget for rough alignment
        self.align_3d_widget = widgets.AlignVolumeWidget()
        # 2D widget to verify rough alignment
        self.show_alignment_2d_widget = widgets.ShowAlignment2DWidget()
        # Registration widget to configure ANTs parameters
        self.registration_widget = widgets.RegistrationWidget()
        self.registration_widget.registration_started.connect(self.registration_started)
        self.registration_widget.registration_completed.connect(self.registration_finished)
        # 3D widget to view registration result
        self.show_reg_3d_widget = widgets.ShowReg3DWidget()
        self.show_reg_2d_widget = widgets.ShowReg2DWidget()
        # Add point mapping widget
        self.map_points_widget = widgets.MapPointsWidget()
        self.fixed_stack.path_changed.connect(self.map_points_widget.fixed_stack.path_changed)
        self.moving_stack.path_changed.connect(self.map_points_widget.moving_stack.path_changed)

        # Add right layout to main layout
        main_layout.addLayout(self.right_layout)

        # Add progress bar
        self.statusBar().showMessage('Ready')
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(0)
        self.statusBar().addPermanentWidget(self.progress_bar)
        self.progress_bar.hide()

        self.file_menu = QMenu('&File', self)
        self.menuBar().addMenu(self.file_menu)

        self.file_open_ref = self.file_menu.addAction('&Open reference stack...', self.fixed_stack.select_file)
        self.file_open_ref.setShortcut('Ctrl+r')
        self.file_open_moving = self.file_menu.addAction('&Open moving stack...', self.moving_stack.select_file)
        self.file_open_moving.setShortcut('Ctrl+m')
        self.file_open_s2p = self.file_menu.addAction('&Import suite2p layer as moving stack...',
                                                      self.moving_stack.select_suite2p_folder)
        self.file_open_s2p.setShortcut('Ctrl+Shift+s')

        self.tool_menu = QMenu('&Tools')
        self.menuBar().addMenu(self.tool_menu)

        self.run_s2p_alignment = self.tool_menu.addAction('&Run s2p alignment', widgets.S2PAlignment.run)
        self.run_s2p_alignment.setShortcut('Ctrl+Shift+a')

        # Add statusbar buttons
        self.dynamic_button_bar = QWidget()
        self.dynamic_button_bar.setLayout(QHBoxLayout())
        self.statusBar().addPermanentWidget(self.dynamic_button_bar)
        self.add_dynamic_widget('Align 3D', self.align_3d_widget, selected=True)
        self.add_dynamic_widget('Show 2D alignment', self.show_alignment_2d_widget)
        self.add_dynamic_widget('Configure registration', self.registration_widget)
        self.add_dynamic_widget('Show 2D registration', self.show_reg_2d_widget)
        self.add_dynamic_widget('Show 3D registration', self.show_reg_3d_widget)
        self.add_dynamic_widget('Map points to volume', self.map_points_widget)

        # Connect signals
        # Fixed
        self.fixed_stack.path_changed.connect(registration.fixed.load_file)
        self.fixed_stack.resolution.changed.connect(registration.fixed.set_resolution)
        registration.fixed.changed.connect(self.check_existing_metadata)
        # Moving
        self.moving_stack.path_changed.connect(registration.moving.load_file)
        self.moving_stack.resolution.changed.connect(registration.moving.set_resolution)
        registration.moving.changed.connect(self.moving_stack_selected)

    def moving_stack_selected(self):
        """Set working directory to path containing the moving stack and attempt to load saved registration data
        """

        working_dir = '/'.join(registration.moving.file_path.as_posix().split('/')[:-1])
        print(f'Set working directory to {working_dir}')
        os.chdir(working_dir)

        self.check_existing_metadata()

    def check_existing_metadata(self):

        if not registration.data_complete():
            return

        meta_path = f'{registration.save_path}/metadata.yaml'
        if not os.path.exists(meta_path):
            return

        meta = yaml.safe_load(open(meta_path, 'r'))
        print(f'Found metadata: {meta}')

        self.fixed_stack.resolution.set_resolution(*meta['fixed_resolution'])
        self.moving_stack.resolution.set_resolution(*meta['moving_resolution'])
        registration.moving.translation = meta['init_translation']
        registration.moving.z_rotation = meta['init_z_rotation']

    def add_dynamic_widget(self, name: str, widget: DynamicWidget, selected: bool = False):
        # Set up widget
        widget.setVisible(selected)
        self.right_layout.addWidget(widget)
        self.dynamic_widgets[name] = widget

        # Set up button
        btn = QPushButton(name)
        btn.setCheckable(True)
        btn.setChecked(selected)
        btn.clicked.connect(lambda: self.toggle_dynamic_btn_on(name))
        self.dynamic_button_bar.layout().addWidget(btn)
        self.dynamic_buttons[name] = btn

    def toggle_dynamic_btn_on(self, name: str):
        print(f'Switch to {name}')

        for n in self.dynamic_widgets:
            self.dynamic_widgets[n].hide()
            self.dynamic_buttons[n].setChecked(False)

        self.dynamic_widgets[name].show()
        self.dynamic_buttons[name].setChecked(True)

    def registration_started(self):
        self.statusBar().showMessage('Running registration')
        self.progress_bar.show()
        self.dynamic_button_bar.setEnabled(False)
        self.fixed_stack.setEnabled(False)
        self.moving_stack.setEnabled(False)

    def registration_finished(self):
        self.statusBar().showMessage('Ready')
        self.progress_bar.hide()
        self.dynamic_button_bar.setEnabled(True)
        self.fixed_stack.setEnabled(True)
        self.moving_stack.setEnabled(True)


class ControlPanel(QGroupBox):

    def __init__(self, parent=None):
        QGroupBox.__init__(self, 'Control panel', parent=parent)

        # Left Panel (Control Panel)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.setMinimumSize(300, 1)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.MinimumExpanding)

        # Placeholder Buttons
        for i in range(5):
            button = QPushButton(f'Button {i + 1}')
            main_layout.addWidget(button)
        main_layout.addStretch()


def main():

    # Make numpy print floats nicely
    np.set_printoptions(suppress=True)

    app = QApplication(sys.argv)
    window = Window()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()

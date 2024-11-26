from __future__ import annotations

import multiprocessing
import os
import pprint
import sys
import time
import traceback
from collections import OrderedDict
from typing import Any, Dict, List, Union

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import yaml
from PySide6 import QtCore
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt

from ants_registration.stack import Stack
from ants_registration.ui import StackWidget
from registration import Registration


class Window(QMainWindow):
    instance: Window

    registration_changed = QtCore.Signal(Registration)

    dynamic_widgets: Dict[str, QWidget] = {}
    dynamic_buttons: Dict[str, QPushButton] = {}

    def __init__(self, *args, **kwarg):
        QMainWindow.__init__(self, *args, **kwarg)
        self.__class__.instance = self

        self.setWindowTitle('ANTs registration interface')
        self.setMinimumSize(1400, 1200)

        # Central Widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main Layout
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)

        # Left Panel (Control Panel)
        self.control_panel = ControlPanel(parent=self)
        main_layout.addWidget(self.control_panel)

        # Right panel
        self.right_layout = QVBoxLayout()

        # Info panel
        self.info_panel = QWidget()
        info_layout = QHBoxLayout()
        self.info_panel.setLayout(info_layout)
        self.info_panel.setMinimumSize(-1, 200)
        self.info_panel.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        self.right_layout.addWidget(self.info_panel)

        # Add fixed stack widget
        self.fixed_stack = StackWidget('Reference stack')
        info_layout.addWidget(self.fixed_stack)

        # Add moving stack widget
        self.moving_stack = StackWidget('Moving stack')
        info_layout.addWidget(self.moving_stack)

        # Create dynamic widgets
        self.align_3d_widget = VolumeAlignmentWidget()
        self.show_raw_2d_widget = Display2DWidget()
        self.registration_widget = RegistrationWidget()
        self.show_reg_3d_widget = DisplayVolumeWidget()

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

        self.file_open_ref = self.file_menu.addAction('&Open reference stack...', self.fixed_stack.select_file)
        self.file_open_ref.setShortcut('Ctrl+r')
        self.file_open_moving = self.file_menu.addAction('&Open moving stack...', self.moving_stack.select_file)
        self.file_open_moving.setShortcut('Ctrl+m')
        self.file_open_s2p = self.file_menu.addAction('&Import suite2p layer as moving stack...',
                                                      self.moving_stack.select_suite2p_folder)
        self.file_open_s2p.setShortcut('Ctrl+p')
        self.menuBar().addMenu(self.file_menu)

        # Add statusbar buttons
        self.dynamic_button_bar = QWidget()
        self.dynamic_button_bar.setLayout(QHBoxLayout())
        self.statusBar().addPermanentWidget(self.dynamic_button_bar)
        self.add_dynamic_widget('3D alignment', self.align_3d_widget, selected=True)
        self.add_dynamic_widget('Show 2D raw alignment', self.show_raw_2d_widget)
        self.add_dynamic_widget('Configure registration', self.registration_widget)
        self.add_dynamic_widget('Show 3D registration', self.show_reg_3d_widget)

        # Connect signals
        # Fixed
        self.fixed_stack.path_changed.connect(registration.fixed.load_file)
        self.fixed_stack.resolution.changed.connect(registration.fixed.set_resolution)
        registration.fixed.changed.connect(self.align_3d_widget.update_fixed_stack)
        registration.fixed.resolution_changed.connect(self.align_3d_widget.apply_fixed_stack_scale)
        registration.fixed.resolution_changed.connect(self.align_3d_widget.reset_view)
        # Moving
        self.moving_stack.path_changed.connect(registration.moving.load_file)
        self.moving_stack.resolution.changed.connect(registration.moving.set_resolution)
        registration.moving.changed.connect(self.align_3d_widget.update_moving_stack)
        registration.moving.resolution_changed.connect(self.align_3d_widget.apply_moving_stack_scale)
        registration.moving.resolution_changed.connect(self.align_3d_widget.reset_view)

    def add_dynamic_widget(self, name: str, widget: QWidget, selected: bool = False):
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
            self.dynamic_widgets[n].setVisible(False)
            self.dynamic_buttons[n].setChecked(False)

        self.dynamic_widgets[name].update_widget()

        self.dynamic_widgets[name].setVisible(True)
        self.dynamic_buttons[name].setChecked(True)


class ControlPanel(QGroupBox):

    def __init__(self, parent=None):
        QGroupBox.__init__(self, 'Control panel', parent=parent)

        # Left Panel (Control Panel)
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.setMinimumSize(400, -1)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.MinimumExpanding)

        # Placeholder Buttons
        for i in range(5):
            button = QPushButton(f'Button {i + 1}')
            main_layout.addWidget(button)
        main_layout.addStretch()


class VolumeAlignmentWidget(gl.GLViewWidget):
    translation_keys = [
        Qt.Key.Key_C,
        Qt.Key.Key_X,
        Qt.Key.Key_W,
        Qt.Key.Key_S,
        Qt.Key.Key_A,
        Qt.Key.Key_D
    ]

    translation_axes = [
        (0, 0, -1),  # up
        (0, 0, 1),  # down
        (0, -1, 0),  # front
        (0, 1, 0),  # back
        (1, 0, 0),  # left
        (-1, 0, 0)  # right
    ]

    rotation_keys = [
        Qt.Key.Key_Q,
        Qt.Key.Key_E
    ]

    rotation_directions = [
        -1,  # CCW
        1  # CW
    ]

    def __init__(self, *args, **kwargs):
        gl.GLViewWidget.__init__(self, *args, **kwargs)

        self.setBackgroundColor((30, 30, 30))
        self.setMinimumSize(1000, 1000)
        # Add volumes
        self.fixed_vol = gl.GLVolumeItem(np.zeros((1, 1, 1, 4)))
        self.addItem(self.fixed_vol)
        self.moving_vol = gl.GLVolumeItem(np.zeros((1, 1, 1, 4)))
        self.addItem(self.moving_vol)

        # Add grid
        self.grid = gl.GLGridItem()
        self.addItem(self.grid)

        # Add axes
        self.axes = gl.GLAxisItem()
        self.addItem(self.axes)

        self._rotation = 0.
        self._translation = np.array([0., 0., 0.])
        self._scale = np.array([1., 1., 1.])

    def update_fixed_stack(self):

        # Get stack data
        data_im = registration.fixed.data_rgba(color=(1., 0., 0.))

        # Add new volume item
        self.removeItem(self.fixed_vol)
        del self.fixed_vol
        self.fixed_vol = gl.GLVolumeItem(data_im, sliceDensity=1, smooth=True, glOptions='additive')
        self.addItem(self.fixed_vol)

        # Update
        self.apply_fixed_stack_scale()
        self.reset_view()

    def update_moving_stack(self):

        # Get stack data
        data_im = registration.moving.data_rgba(color=(0., 1., 0.))

        # Add new volume item
        self.removeItem(self.moving_vol)
        del self.moving_vol
        self.moving_vol = gl.GLVolumeItem(data_im, sliceDensity=1, smooth=True, glOptions='additive')
        self.addItem(self.moving_vol)

        # Update
        self.apply_moving_stack_scale()
        self.reset_view()

    def reset_view(self):

        # Get reference volume shape
        if registration.fixed.data is not None:
            volume_shape = registration.fixed.shape
            volume_scale = registration.fixed.resolution
        # Use moving volume as fallback if fixed reference hasn't been set yet
        else:
            volume_shape = registration.moving.shape
            volume_scale = registration.moving.resolution

        volume_size = volume_shape * volume_scale

        center = volume_size / 2
        self.setCameraPosition(pos=pg.Vector(*center), distance=2 * volume_size.max())

        # Set translation explicitly, bc grid.translate is applied on top of exising transforms
        tr = pg.Transform3D()
        tr.translate(*center[:2])
        self.grid.setTransform(tr)
        self.grid.setSize(*volume_size)
        self.grid.setSpacing(*[volume_size.max() // 10] * 3)

        # Update axes sizes
        self.axes.setSize(*volume_size)

    def keyPressEvent(self, ev):

        if ev.key() in self.translation_keys:
            self.apply_translation(ev)

        if ev.key() in self.rotation_keys:
            self.apply_rotation(ev)

        gl.GLViewWidget.keyPressEvent(self, ev)

    def apply_translation(self, ev):
        _axis = self.translation_axes[self.translation_keys.index(ev.key())]

        registration.moving.translation += np.array(_axis) * 1

        # print('Translation:', registration.moving.translation)

        self.update_transform()

    def apply_rotation(self, ev):
        _dir = self.rotation_directions[self.rotation_keys.index(ev.key())]

        registration.moving.z_rotation += _dir * 1

        # print('Rotation:', registration.moving.z_rotation)

        self.update_transform()

    def apply_fixed_stack_scale(self):

        self.update_transform()

    def apply_moving_stack_scale(self):

        # Reset translation and rotation
        registration.translation = np.array([0., 0., 0.])
        registration.z_rotation = 0.

        self.update_transform()

    def update_transform(self):

        # Set transforms for volumes
        if self.fixed_vol.data.shape != (1, 1, 1, 4):
            self.fixed_vol.setTransform(registration.fixed.get_pg_transform())
        if self.moving_vol.data.shape != (1, 1, 1, 4):
            self.moving_vol.setTransform(registration.moving.get_pg_transform())

    def update_widget(self):
        pass


class DisplayVolumeWidget(gl.GLViewWidget):
    translation_keys = [
        Qt.Key.Key_C,
        Qt.Key.Key_X,
        Qt.Key.Key_W,
        Qt.Key.Key_S,
        Qt.Key.Key_A,
        Qt.Key.Key_D
    ]

    translation_axes = [
        (0, 0, -1),  # up
        (0, 0, 1),  # down
        (0, -1, 0),  # front
        (0, 1, 0),  # back
        (1, 0, 0),  # left
        (-1, 0, 0)  # right
    ]

    rotation_keys = [
        Qt.Key.Key_Q,
        Qt.Key.Key_E
    ]

    rotation_directions = [
        -1,  # CCW
        1  # CW
    ]

    def __init__(self, *args, **kwargs):
        gl.GLViewWidget.__init__(self, *args, **kwargs)

        self.setBackgroundColor((30, 30, 30))
        self.setMinimumSize(1000, 1000)
        # Add volumes
        self.fixed_vol = gl.GLVolumeItem(np.zeros((1, 1, 1, 4)))
        self.addItem(self.fixed_vol)
        self.moving_vol = gl.GLVolumeItem(np.zeros((1, 1, 1, 4)))
        self.addItem(self.moving_vol)

        # Add grid
        self.grid = gl.GLGridItem()
        self.addItem(self.grid)

        # Add axes
        self.axes = gl.GLAxisItem()
        self.addItem(self.axes)

    def update_widget(self):

        if registration.result is None:
            return

        # Get stack data
        fixed_im = registration.fixed.data_rgba(color=(1., 0., 0.))

        # Add new volume item
        self.removeItem(self.fixed_vol)
        del self.fixed_vol
        self.fixed_vol = gl.GLVolumeItem(fixed_im, sliceDensity=1, smooth=True, glOptions='additive')
        self.addItem(self.fixed_vol)

        # Get stack data
        warped_stack = Stack()
        warped_stack.data = registration.result['warpedmovout'].numpy()
        warped_im = warped_stack.data_rgba(color=(0., 1., 0.))

        # Add new volume item
        self.removeItem(self.moving_vol)
        del self.moving_vol
        self.moving_vol = gl.GLVolumeItem(warped_im, sliceDensity=1, smooth=True, glOptions='additive')
        self.addItem(self.moving_vol)

        # Get reference volume shape
        if registration.fixed.data is not None:
            volume_shape = registration.fixed.shape
            volume_scale = registration.fixed.resolution
        # Use moving volume as fallback if fixed reference hasn't been set yet
        else:
            volume_shape = registration.moving.shape
            volume_scale = registration.moving.resolution

        volume_size = volume_shape * volume_scale

        center = volume_size / 2
        self.setCameraPosition(pos=pg.Vector(*center), distance=2 * volume_size.max())

        # Set translation explicitly, bc grid.translate is applied on top of exising transforms
        tr = pg.Transform3D()
        tr.translate(*center[:2])
        self.grid.setTransform(tr)
        self.grid.setSize(*volume_size)
        self.grid.setSpacing(*[volume_size.max() // 10] * 3)

        # Update axes sizes
        self.axes.setSize(*volume_size)


class Display2DWidget(QGroupBox):

    def __init__(self, parent=None):
        QGroupBox.__init__(self, '2D alignment', parent)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # self.setMinimumSize(800, 800)
        self.image_view = pg.ImageView(discreteTimeLine=True, levelMode='rgba')
        main_layout.addWidget(self.image_view)

    def update_widget(self):
        image_data = registration.init_alignment_stack()

        self.image_view.setImage(image_data, axes={'x': 0, 'y': 1, 't': 2, 'c': 3})


class Registration2DWidget(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # self.setMinimumSize(800, 800)
        self.image_view = pg.ImageView(discreteTimeLine=True, levelMode='rgba')
        main_layout.addWidget(self.image_view)

        image_data = np.concatenate(
            (registration.fixed.data[:, :, :, np.newaxis],
             registration.result['warpedmovout'].numpy()[:, :, :, np.newaxis],
             np.zeros_like(registration.fixed.data)[:, :, :, np.newaxis]),
            axis=3
        )
        self.image_view.setImage(image_data, axes={'x': 0, 'y': 1, 't': 2, 'c': 3})


class RegistrationWidget(QGroupBox):

    settings: Dict[str, Any]

    def __init__(self, parent=None):
        QGroupBox.__init__(self, 'Registration config', parent)

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        self.left_widget = QWidget()
        self.left_widget.setMaximumWidth(300)
        main_layout.addWidget(self.left_widget)
        left_layout = QVBoxLayout()
        self.left_widget.setLayout(left_layout)

        self.right_widget = QWidget()
        main_layout.addWidget(self.right_widget)
        right_layout = QVBoxLayout()
        self.right_widget.setLayout(right_layout)

        # Left side
        left_layout.addWidget(QLabel('Settings'))
        self.settings_text = QTextEdit()
        self.settings_text.textChanged.connect(self.parse_settings)
        left_layout.addWidget(self.settings_text)

        left_layout.addWidget(QLabel('Parsed settings'))
        self.settings_parsed_text = QLineEdit()
        self.settings_parsed_text.setReadOnly(True)
        left_layout.addWidget(self.settings_parsed_text)

        self.run_registration_btn = QPushButton('Run registration')
        self.run_registration_btn.setEnabled(False)
        self.run_registration_btn.clicked.connect(self.run_registration)
        left_layout.addWidget(self.run_registration_btn)

        # Right side
        right_layout.addWidget(QLabel('Log'))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_layout.addWidget(self.log_text)

        # Set registration settings
        self.settings_text.setText(yaml.safe_dump(current_reg_settings))

        # Redirect stdout
        self.thread = None
        self.worker = None
        self.timer = QtCore.QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.update_log)
        self.log_line_count = 0

        # Connect signals
        registration.fixed.changed.connect(self.check_reg_requirements)
        registration.moving.changed.connect(self.check_reg_requirements)

    def check_reg_requirements(self):
        self.run_registration_btn.setEnabled(registration.fixed.data is not None
                                             and registration.moving.data is not None)

    def parse_settings(self):

        try:
            settings = yaml.safe_load(self.settings_text.toPlainText())

        except Exception as _exc:
            traceback.print_exc()
            settings = 'Illegal format'

        else:

            # Make sure that these are tuples
            #  ANTs does not like lists for these and yaml doesn't do tuples by default
            settings['reg_iterations'] = tuple(settings['reg_iterations'])
            settings['aff_iterations'] = tuple(settings['aff_iterations'])
            settings['aff_shrink_factors'] = tuple(settings['aff_shrink_factors'])
            settings['aff_smoothing_sigmas'] = tuple(settings['aff_smoothing_sigmas'])

            # Set
            self.settings = settings

        self.settings_parsed_text.setText(str(settings))

    def run_registration(self):

        # Create and start the worker in a thread
        self.thread = QtCore.QThread()
        self.worker = RegistrationTask(self.settings)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.registration_finished)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.timer.start()

        self.run_registration_btn.setEnabled(False)
        window.statusBar().showMessage('Running registration')
        window.progress_bar.show()
        window.dynamic_button_bar.setEnabled(False)

        # Reset counter and clear log file
        self.log_line_count = 0
        with open('test.txt', 'w') as f:
            pass

        # Start
        self.thread.start()

    def update_log(self):
        with open('test.txt', 'r') as f:
            for line in f.readlines()[self.log_line_count:]:
                self.log_text.append(line.strip('\n'))
                self.log_line_count += 1

    def registration_finished(self):
        self.run_registration_btn.setEnabled(True)
        window.statusBar().showMessage('Ready')
        window.progress_bar.hide()
        window.dynamic_button_bar.setEnabled(True)

        self.timer.stop()
        self.log_text.append('Registration completed')

    def update_widget(self):
        pass


class RegistrationTask(QtCore.QObject):

    finished = QtCore.Signal()

    def __init__(self, reg_settings: Dict[str, Any], parent=None):
        QtCore.QObject.__init__(self, parent)

        self.reg_settings = reg_settings

    def run(self):

        registration.run(self.reg_settings)

        self.finished.emit()


if __name__ == '__main__':
    # Make numpy print floats nicely
    np.set_printoptions(suppress=True)

    default_reg_settings: Dict[str, Any] = dict(
        type_of_transform="Affine",
        initial_transform=None,
        outprefix="",
        mask=None,
        moving_mask=None,
        mask_all_stages=False,
        grad_step=0.2,
        flow_sigma=3,
        total_sigma=0,
        aff_metric="mattes",
        aff_sampling=32,
        aff_random_sampling_rate=0.2,
        syn_metric="mattes",
        syn_sampling=32,
        reg_iterations=(40, 20, 0),
        aff_iterations=(2100, 1200, 1200, 1200),
        aff_shrink_factors=(6, 4, 2, 1),
        aff_smoothing_sigmas=(3, 2, 1, 0),
        random_seed=None,
        multivariate_extras=None,
        restrict_transformation=None,
        smoothing_in_mm=False,
    )
    current_reg_settings = default_reg_settings.copy()

    registration = Registration()

    app = QApplication(sys.argv)
    window = Window()
    window.show()

    registration.fixed.load_file(
        'Z:/cluster/scripts/ants_registration/ants_registration/reference_data/2024-11-12_jf7_standard_stack/2024-11-12_jf7_5dpf_1p0magn_35laser_600gain_6avg_reference_tectum_top_140_to_0mu_0p88_zstep.tif')
    registration.moving.load_file(
        'Z:/cluster/scripts/ants_registration/ants_registration/reference_data/2024-11-12_jf7_standard_stack/2024-11-12_jf7_5dpf_right_hemi_1p4magn_35laser_600gain_6avg_reference_tectum_top_140_to_0mu_0p63_zstep.tif')
    sys.exit(app.exec())

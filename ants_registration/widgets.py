from __future__ import annotations

import traceback
from typing import Dict, Any, List, Union, Callable

import ants
import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl
import yaml
from PySide6 import QtCore
from PySide6.QtWidgets import *
from PySide6.QtCore import Qt

from ants_registration.ui import DynamicWidget
from ants_registration.registration import registration


class AlignVolumeWidget(DynamicWidget):

    def __init__(self):
        DynamicWidget.__init__(self)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Add controls info
        self.controls_label = QLabel("""
        Controls\n
        ---\n
        W/A -> front/back\n
        A/D -> left/right\n
        X/C -> up/down\n
        N/M -> lower/higher contrast
        """)
        main_layout.addWidget(self.controls_label)

        self.gl_widget = AlignVolumeGLWidget()
        main_layout.addWidget(self.gl_widget)

        # Connect signals
        registration.fixed.changed.connect(self.gl_widget.update_fixed_stack)
        registration.fixed.resolution_changed.connect(self.gl_widget.reset_view)

        registration.moving.changed.connect(self.gl_widget.update_moving_stack)
        registration.moving.resolution_changed.connect(self.gl_widget.reset_view)
        registration.moving.translation_changed.connect(self.gl_widget.update_transform)
        registration.moving.rotation_changed.connect(self.gl_widget.update_transform)


class AlignVolumeGLWidget(gl.GLViewWidget):
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
        # Init fixed volume
        self.fixed_vol = None
        # self.addItem(self.fixed_vol)
        # Init moving volume
        self.moving_vol = None  # gl.GLVolumeItem(self.data_moving)
        # self.addItem(self.moving_vol)

        # Add grid
        self.grid = gl.GLGridItem()
        self.addItem(self.grid)

        # Add axes
        self.axes = gl.GLAxisItem()
        self.addItem(self.axes)

        self.color_fixed = (1., 0., 0.)
        self.color_moving = (0., 1., 0.)
        self.percentile = 90
        self.alpha = 0.02

    def update_fixed_stack(self):

        # Get stack data
        data_rgba = registration.fixed.data_rgba(color=self.color_fixed,
                                                 alpha=self.alpha,
                                                 percentile=self.percentile)

        # Add new volume item
        if self.fixed_vol is not None:
            self.removeItem(self.fixed_vol)
            del self.fixed_vol
        self.fixed_vol = gl.GLVolumeItem(data_rgba, sliceDensity=1, smooth=True, glOptions='additive')
        self.addItem(self.fixed_vol)

        # Update
        self.update_transform()
        self.reset_view()

    def update_moving_stack(self):

        # Get stack data
        data_rgba = registration.moving.data_rgba(color=self.color_moving,
                                                  alpha=self.alpha,
                                                  percentile=self.percentile)

        # Add new volume item
        if self.moving_vol is not None:
            self.removeItem(self.moving_vol)
            del self.moving_vol
        self.moving_vol = gl.GLVolumeItem(data_rgba, sliceDensity=1, smooth=True, glOptions='additive')
        self.addItem(self.moving_vol)

        # Update
        self.update_transform()
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

        if ev.key() in [Qt.Key.Key_N, Qt.Key.Key_M]:
            print('Contrast')
            if ev.key() == Qt.Key.Key_N:
                self.alpha -= 0.01
            if ev.key() == Qt.Key.Key_M:
                self.alpha += 0.01

            # if self.fixed_vol is not None:
            self.update_fixed_stack()
            # if self.moving_vol is not None:
            self.update_moving_stack()

        gl.GLViewWidget.keyPressEvent(self, ev)

    def apply_translation(self, ev):
        _axis = self.translation_axes[self.translation_keys.index(ev.key())]

        registration.moving.translation = registration.moving.translation + np.array(_axis) * 1

        # print('Translation:', registration.moving.translation)

    def apply_rotation(self, ev):
        _dir = self.rotation_directions[self.rotation_keys.index(ev.key())]

        registration.moving.z_rotation = registration.moving.z_rotation + _dir * 1

    def update_transform(self):

        # Set transforms for volumes
        if self.fixed_vol is not None:
            self.fixed_vol.setTransform(registration.fixed.get_pg_transform())
        if self.moving_vol is not None:
            self.moving_vol.setTransform(registration.moving.get_pg_transform())


class ShowAlignment2DWidget(DynamicWidget):

    def __init__(self):
        DynamicWidget.__init__(self)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # self.setMinimumSize(800, 800)
        self.image_view = pg.ImageView(discreteTimeLine=True, levelMode='rgba')
        main_layout.addWidget(self.image_view)

        # Connect signals
        self.visibility_changed.connect(self.update_image)

    def update_image(self, visible: bool = True):
        if not visible:
            return

        image_data = registration.get_alignment_rgb_stack()

        self.image_view.setImage(image_data, axes={'x': 0, 'y': 1, 't': 2, 'c': 3})


class ShowReg3DWidget(DynamicWidget):

    def __init__(self):
        DynamicWidget.__init__(self)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.gl_widget = ShowReg3DGLWidget()
        main_layout.addWidget(self.gl_widget)

        # Connect signals
        self.visibility_changed.connect(self.gl_widget.update_image)
        registration.fixed.changed.connect(self.gl_widget.on_data_change)
        registration.moving.changed.connect(self.gl_widget.on_data_change)
        registration.changed.connect(self.gl_widget.on_data_change)


class ShowReg3DGLWidget(gl.GLViewWidget):

    def __init__(self, *args, **kwargs):
        gl.GLViewWidget.__init__(self, *args, **kwargs)

        self.setBackgroundColor((30, 30, 30))
        self.setMinimumSize(1000, 1000)
        # Add volumes
        self.volume = gl.GLVolumeItem(np.zeros((1, 1, 1, 4)))
        self.addItem(self.volume)

        self.re_render = True
        self.alpha = 0.02

        # Add grid
        self.grid = gl.GLGridItem()
        self.addItem(self.grid)

        # Add axes
        self.axes = gl.GLAxisItem()
        self.addItem(self.axes)

    def on_data_change(self):
        """Set re-render flag if any image data or the registration result changes"""
        self.re_render = True

    def update_image(self, visible: bool):

        if not visible:
            return

        # If re-render
        if not self.re_render:
            return

        print('Render view')

        # Get registered RGB stack
        registered_image_rgb = registration.get_registered_rgb_stack()[::-1, :, :]  # flip X for correct representation
        volume_size = np.array(registered_image_rgb.shape[:3])

        # Add alpha
        registered_image_rgba = np.concatenate([registered_image_rgb, np.zeros((*volume_size, 1))], axis=-1)
        brightness_sum = registered_image_rgb.sum(axis=3)
        selected_voxels = brightness_sum[:, :, :] > np.percentile(brightness_sum, 90, axis=(0, 1))
        registered_image_rgba[:, :, :, :3] *= 255
        registered_image_rgba[:, :, :, 3] = 255 * selected_voxels * self.alpha

        # Add new volume item
        self.removeItem(self.volume)
        del self.volume
        self.volume = gl.GLVolumeItem(registered_image_rgba, sliceDensity=1, smooth=True, glOptions='additive')
        self.addItem(self.volume)

        # Set camera and axes objects

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

        # Reset render flag
        self.re_render = False


class ShowReg2DWidget(DynamicWidget):

    def __init__(self, parent=None):
        DynamicWidget.__init__(self)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        self.image_view = pg.ImageView(discreteTimeLine=True, levelMode='rgba')
        main_layout.addWidget(self.image_view)

        self.re_render = True

        # Connect signals
        self.visibility_changed.connect(self.update_image)
        registration.fixed.changed.connect(self.on_data_change)
        registration.moving.changed.connect(self.on_data_change)
        registration.changed.connect(self.on_data_change)

    def on_data_change(self):
        """Set re-render flag if any image data or the registration result changes"""
        self.re_render = True

    def update_image(self, visible: bool):

        if not visible:
            return

        # If re-render
        if not self.re_render:
            return

        print('Render view')

        # Update image data
        image_data = registration.get_registered_rgb_stack()
        self.image_view.setImage(image_data, axes={'x': 0, 'y': 1, 't': 2, 'c': 3})

        self.re_render = False


class RegistrationWidget(QGroupBox):
    registration_started = QtCore.Signal()
    registration_completed = QtCore.Signal()

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
        self.settings_text.setText(yaml.safe_dump(registration.settings))

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

            # Set
            registration.settings.update(settings)

        self.settings_parsed_text.setText(str(settings))

    def run_registration(self):

        # Create and start the worker in a thread
        self.thread = QtCore.QThread()
        self.worker = RegistrationTask()
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.registration_finished)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.run_registration_btn.setEnabled(False)

        # Reset counter and clear log file
        self.log_line_count = 0
        with open(f'{registration.save_path}/registration.log', 'w') as f:
            pass

        # Start
        self.thread.start()
        self.timer.start()

        # Emit
        self.registration_started.emit()

    def update_log(self):
        with open(f'{registration.save_path}/registration.log', 'r') as f:
            for line in f.readlines()[self.log_line_count:]:
                self.log_text.append(line.strip('\n'))
                self.log_line_count += 1

    def registration_finished(self):
        self.run_registration_btn.setEnabled(True)

        self.timer.stop()
        self.log_text.append('Registration completed')

        self.registration_completed.emit()

    def update_widget(self):
        pass


class RegistrationTask(QtCore.QObject):
    finished = QtCore.Signal()

    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)

    def run(self):
        try:
            registration.run()
        except Exception as _exc:
            traceback.print_exc()

        self.finished.emit()


class MapPointsWidget(DynamicWidget):

    mapping_started = QtCore.Signal()
    mapping_finished = QtCore.Signal()

    def __init__(self, parent=None):
        DynamicWidget.__init__(self)

        main_layout = QVBoxLayout()
        self.setLayout(main_layout)

        # Add inner widget
        self.inner_widget = QWidget()
        inner_layout = QVBoxLayout()
        self.inner_widget.setLayout(inner_layout)
        main_layout.addWidget(self.inner_widget)

        # Add permanent selections for current moving and fixed reference
        self.moving_stack = StackSelectionWidget(-1)
        self.moving_stack.setTitle('Moving stack')
        self.moving_stack.setEnabled(False)
        inner_layout.addWidget(self.moving_stack)
        self.add_arrow()
        self.fixed_stack = StackSelectionWidget(-1)
        self.fixed_stack.setTitle('Reference stack')
        self.fixed_stack.setEnabled(False)
        inner_layout.addWidget(self.fixed_stack)

        self.additional_references: List[Union[StackSelectionWidget, None]] = []
        self.arrows: List[Union[QToolButton, None]] = []

        # Add reference button
        self.add_reference_btn = QPushButton('Add reference')
        self.add_reference_btn.clicked.connect(self.add_reference)
        main_layout.addWidget(self.add_reference_btn)

        # Run mapping button
        self.run_mapping_btn = QPushButton('Map points')
        self.run_mapping_btn.clicked.connect(self.run_mapping)
        main_layout.addWidget(self.run_mapping_btn)

        # Add stretch at bottom
        main_layout.addStretch()

        # Add attributes
        self.thread = None
        self.worker = None

    def add_arrow(self):

        arrow = QToolButton()
        arrow.setFixedSize(40, 40)
        arrow.setArrowType(Qt.ArrowType.DownArrow)
        arrow.setEnabled(False)
        self.inner_widget.layout().addWidget(arrow, alignment=Qt.AlignmentFlag.AlignHCenter)

        return arrow

    def add_reference(self):
        print('Add reference')

        # Add arrow
        arrow = self.add_arrow()
        self.arrows.append(arrow)

        # Add widget
        idx = len(self.additional_references)
        widget = StackSelectionWidget(idx, self)
        widget.delete.connect(self.remove_reference)
        self.inner_widget.layout().addWidget(widget)
        self.additional_references.append(widget)

    def remove_reference(self, idx: int):
        print(f'Remove reference {idx}')
        # Remove arrow
        arrow = self.arrows[idx]
        arrow.deleteLater()
        self.inner_widget.layout().removeWidget(arrow)
        self.arrows[idx] = None

        # Remove widget
        widget = self.additional_references[idx]
        self.inner_widget.layout().removeWidget(widget)
        widget.deleteLater()
        self.additional_references[idx] = None

    def run_mapping(self):

        # Create and start the worker in a thread
        self.thread = QtCore.QThread()
        self.worker = Task(self._run_mapping)
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.mapping_finished)

        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Start
        self.thread.start()

        self.mapping_started.emit()

    def _run_mapping(self):

        # Separate file paths into base directory and filename
        path_list = []
        for widget in [self.moving_stack, self.fixed_stack, *[w for w in self.additional_references if w is not None]]:
            path = widget.file_path.text()

            parts = path.split('/')
            file_name = parts[-1]
            path_dir = '/'.join(parts[:-1])
            path_list.append((path_dir, file_name))

        # Compile list of paths to transform files
        registration_list = []
        for mov, ref in zip(path_list[:-1], path_list[1:]):
            p = f'{mov[0]}/ants_registration/{mov[1]}/{ref[1]}'
            registration_list.append(p)
        transform_list = [f'{p}/Composite.h5' for p in registration_list]

        print('Transforms to apply:')
        print('\n'.join(transform_list))
        # Get data on suite2p moving stack (slice)
        metadata = yaml.safe_load(open(f'{registration_list[0]}/metadata.yaml', 'r'))
        src_resolution = metadata['moving_resolution']
        stats = np.load(f'{path_list[0][0]}/suite2p/plane0/stat.npy', allow_pickle=True)

        # Build coordinates on source reference frame
        roi_coords = np.array([[s['med'][0], s['med'][1], 6] for s in stats]) * np.array(src_resolution) #  + np.array(mov_slice.origin)
        roi_coords_df = pd.DataFrame(roi_coords, columns=['y', 'x', 'z'])
        # coords_1 = ants.apply_transforms_to_points(3, roi_coords_df, reg_slice_to_local['fwdtransforms'][::-1])
        # coords_2 = ants.apply_transforms_to_points(3, coords_1, reg_local_to_standard['fwdtransforms'][::-1])
        # roi_coords_transformed_df = ants.apply_transforms_to_points(3, coords_2,
        #                                                             reg_standard_to_ref['fwdtransforms'][::-1])

        # Apply transforms
        print('Apply transforms')
        # roi_coords_transformed_df = ants.apply_transforms_to_points(3, roi_coords_df, transform_list)
        roi_coords_transformed_df = roi_coords_df.copy()
        for transform in transform_list:
            roi_coords_transformed_df = ants.apply_transforms_to_points(3, roi_coords_transformed_df, [transform])

        print('Save coordinates to file')
        roi_coords_transformed_df.to_hdf(f'{registration_list[0]}/mapped_points.h5', key='coordinates')

        # Get array
        # roi_coords_transformed = roi_coords_transformed_df[['x', 'y', 'z']].values



class StackSelectionWidget(QGroupBox):

    delete = QtCore.Signal(int)
    path_changed = QtCore.Signal(str)

    def __init__(self, idx: int, parent=None):
        QGroupBox.__init__(self, f'Reference #{idx}', parent)

        self.idx = idx

        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        self.file_path = QLineEdit('')
        self.file_path.setReadOnly(True)
        main_layout.addWidget(self.file_path)

        self.file_selection_btn = QPushButton('Select...')
        # self.file_selection_btn.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum)
        main_layout.addWidget(self.file_selection_btn)

        self.delete_btn = QPushButton('Delete')
        main_layout.addWidget(self.delete_btn)

        # connect signals
        self.delete_btn.clicked.connect(lambda: self.delete.emit(self.idx))
        self.file_selection_btn.clicked.connect(self.select_file)
        self.path_changed.connect(self.file_path.setText)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, 'Select stack...', '', 'TIF Files (*.tif *.tiff)')
        if not file_path:
            return

        self.path_changed.emit(file_path)


class Task(QtCore.QObject):
    finished = QtCore.Signal()

    def __init__(self, fun: Callable, parent=None):
        QtCore.QObject.__init__(self, parent)

        self.fun = fun

    def run(self):
        try:
            self.fun()
        except Exception as _exc:
            traceback.print_exc()

        self.finished.emit()

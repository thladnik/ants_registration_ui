from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import numpy as np
import pyqtgraph as pg
import scipy
import tifffile
import yaml
from PySide6 import QtCore


class Stack(QtCore.QObject):

    changed = QtCore.Signal()
    translation_changed = QtCore.Signal()
    rotation_changed = QtCore.Signal()
    resolution_changed = QtCore.Signal()

    def __init__(self):
        QtCore.QObject.__init__(self)

        self.data = None
        self.resolution = np.array([1., 1., 1.])
        self.translation = np.array([0., 0., 0.])
        self.z_rotation: float = 0.
        self.file_path: Union[Path, None] = None
        self.metadata: Dict[str, Any] = {}

    @property
    def shape(self) -> np.ndarray:
        return np.array(self.data.shape)

    def data_rgba(self,
                  percentile: int = 90,
                  alpha: float = 0.02,
                  color: Tuple[float, float, float] = (1., 0., 0.)) -> np.ndarray:

        data_sel = self.data[:, :, :] > np.percentile(self.data, percentile, axis=(0, 1))
        data_im = np.zeros(self.data.shape + (4,), dtype=np.uint8)
        data_im[:, :, :, :3] = (np.array(color) * 255)
        data_im[data_sel, 3] = int(alpha * 255)

        return data_im[::-1, :, :]

    def set_resolution(self, res: Dict[str, float]):
        n = 0
        for i, v in enumerate(['x', 'y', 'z']):
            if res.get(v) is None:
                continue
            self.resolution[i] = res[v]
            n += 1

        if n > 0:
            self.resolution_changed.emit()

    def _load_tif(self):

        # Load stack data
        self.data = np.swapaxes(np.moveaxis(tifffile.imread(self.file_path), 0, 2), 0, 1)
        self.translation = np.array([0., 0., 0.])
        self.z_rotation = 0.

        # Load stack metadata
        self.metadata = {}

        path_parts = self.file_path.as_posix().split('/')
        dir_path, file_name = '/'.join(path_parts[:-1]), path_parts[-1]

        # TODO: Load TIF/NRRD/etc metadata

        # Load annotation metadata
        meta_files = [fn for fn in os.listdir(dir_path) if fn.endswith('metadata.yaml')]
        if len(meta_files) > 0:
            for fn in meta_files:
                _meta = yaml.safe_load(open(f'{dir_path}/{fn}', 'r'))
                self.metadata.update(_meta)

        # Update resultion based on file info
        _dim_order = ['x', 'y', 'z']
        _res_strings = {'zstack/x_res': 'x', 'zstack/y_res': 'y', 'zstack/z_res': 'z'}
        for _src, _dest in _res_strings.items():
            v = self.metadata.get(_src)

            if v is None:
                continue

            # Update resolution for dimension
            self.resolution[_dim_order.index(_dest)] = v

    def _load_suite2p(self):

        # Load layer mean image
        ops = np.load(os.path.join(self.file_path, 'plane0', 'ops.npy'), allow_pickle=True).item()
        im_slice = ops['meanImg'].T
        im_slice[:10] = 0
        im_slice[-10:] = 0

        # Create fake volume
        im_slice_stack = np.zeros(im_slice.shape + (11,))
        im_slice_stack[:, :, 6] = im_slice
        # Convolve along Z
        im_slice_stack = scipy.ndimage.gaussian_filter(im_slice_stack.astype(float), sigma=(0, 0, 2))

        self.data = (im_slice_stack - im_slice_stack.min()) / (im_slice_stack.max() - im_slice_stack.min()) * 255
        self.translation = np.array([0., 0., 0.])
        self.z_rotation = 0.

    def load_file(self, file_path: str):

        print(f'Load file {file_path}')

        self.file_path = Path(file_path)

        if any([self.file_path.as_posix().lower().endswith(ext) for ext in ['.tif', '.tiff']]):
            self._load_tif()

        elif self.file_path.as_posix().endswith('suite2p'):
            self._load_suite2p()

        self.changed.emit()

    def get_pg_transform(self) -> pg.Transform3D:

        _scale = self.resolution
        c_rot = self.shape[:3] / 2 * _scale
        _rot = np.deg2rad(self.z_rotation)
        _trans = self.translation

        T_to_orig = np.array([[1, 0, 0, -c_rot[0]],
                              [0, 1, 0, -c_rot[1]],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])

        R = np.array([[np.cos(_rot), -np.sin(_rot), 0, 0],
                      [np.sin(_rot), np.cos(_rot), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

        T_back = np.array([[1, 0, 0, c_rot[0]],
                           [0, 1, 0, c_rot[1]],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])

        T = np.array([[1, 0, 0, _trans[0]],
                      [0, 1, 0, _trans[1]],
                      [0, 0, 1, _trans[2]],
                      [0, 0, 0, 1]])

        S = np.array([[_scale[0], 0, 0, 0],
                      [0, _scale[1], 0, 0],
                      [0, 0, _scale[2], 0],
                      [0, 0, 0, 1]])

        _mat = T @ T_back @ R @ T_to_orig @ S

        return pg.Transform3D(_mat)

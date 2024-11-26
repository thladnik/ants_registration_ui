import multiprocessing
import os
import sys
from typing import Any, Dict, List, Union

import ants
import numpy as np
from PySide6 import QtCore

from stack import Stack


class Registration(QtCore.QObject):

    file_path_changed = QtCore.Signal()
    changed = QtCore.Signal()

    result: Dict[str, Any] = {}

    def __init__(self):
        QtCore.QObject.__init__(self)

        self.moving = Stack()
        self.fixed = Stack()

        self.file_path: Union[str, None] = None

    def load(self, file_path: str):
        # TODO: load reg

        if self.file_path is not None:
            self.close()

        self.file_path = file_path

    def close(self):
        # TODO: close stuff

        self.file_path = None

    def save(self, file_path: str = None):

        if self.file_path is None and file_path is not None:
            raise ValueError('No file path to save registration')

        if file_path is None:
            file_path = self.file_path

        # TODO: save registration

    def get_ants_init_transform(self) -> Union[List[ants.ANTsTransform], None]:

        # Get rotation and translation components of moving stack
        _trans = self.moving.translation
        _rot = -np.deg2rad(self.moving.z_rotation)

        # If both translation and rotation are zero, ANTs should use default initialization
        if np.isclose(_rot, 0.) and np.isclose(np.linalg.norm(_trans), 0.):
            return None

        # Get resolution and shape
        moving_scale = self.moving.resolution
        fixed_scale = self.fixed.resolution
        moving_shape = self.moving.shape
        fixed_shape = self.fixed.shape

        # Create rotation
        c_rot = np.array([moving_shape[0] / 2, moving_shape[1] / 2, 0.])
        R = np.array([[np.cos(_rot), -np.sin(_rot), 0, 0],
                      [np.sin(_rot), np.cos(_rot), 0, 0],
                      [0, 0, 1, 0]])

        tr_rot_ants = ants.create_ants_transform(transform_type='AffineTransform')
        tr_rot_ants.set_parameters(R)
        tr_rot_ants.set_fixed_parameters(c_rot * moving_scale)

        # Create translation transform
        x_offset = (fixed_scale[0] * fixed_shape[0] - moving_scale[0] * moving_shape[0])
        T = np.array([[1, 0, 0, _trans[0] - x_offset],
                      [0, 1, 0, -_trans[1]],
                      [0, 0, 1, -_trans[2]]])

        tr_trans_ants = ants.create_ants_transform(transform_type='AffineTransform')
        tr_trans_ants.set_parameters(T)

        # Combine transforms
        tr_ants = ants.compose_ants_transforms([tr_trans_ants, tr_rot_ants])

        return [tr_ants, tr_trans_ants, tr_rot_ants]

    def init_alignment_stack(self) -> np.ndarray:

        # Get Image data
        fixed_stack_ants = ants.from_numpy(self.fixed.data, spacing=(*self.fixed.resolution,))
        moving_stack_ants = ants.from_numpy(self.moving.data, spacing=(*self.moving.resolution,))

        # Get moving transform
        transforms = self.get_ants_init_transform()

        if transforms is not None:
            tr_ants = transforms[0]
        else:
            tr_ants = ants.create_ants_transform()

        # Apply
        raw_aligned_image = tr_ants.apply_to_image(moving_stack_ants, fixed_stack_ants)

        # Create image
        # TODO: use respective colors
        image_data = np.concatenate(
            (fixed_stack_ants.numpy()[:, :, :, np.newaxis],
             raw_aligned_image.numpy()[:, :, :, np.newaxis],
             np.zeros(fixed_stack_ants.shape)[:, :, :, np.newaxis]),
            axis=3
        )

        return image_data

    def run(self, settings: Dict[str, Any]):

        # Get Image data
        fixed_stack_ants = ants.from_numpy(self.fixed.data, spacing=(*self.fixed.resolution,))
        moving_stack_ants = ants.from_numpy(self.moving.data, spacing=(*self.moving.resolution,))

        # Get transforms
        transforms = self.get_ants_init_transform()

        if transforms is None:
            init_transforms = None
        else:
            trans_path = 'init_rot.mat'
            ants.write_transform(transforms[1], trans_path)
            rot_path = 'init_trans.mat'
            ants.write_transform(transforms[2], rot_path)

            init_transforms = [trans_path, rot_path]

        # Update settings
        settings['initial_transform'] = init_transforms
        settings['verbose'] = True
        settings['write_composite_transform'] = False

        # Run registration

        with ANTsLog('test.txt') as log:
            proc = multiprocessing.Process(target=ants.registration,
                                           args=(fixed_stack_ants, moving_stack_ants),
                                           kwargs=settings)
            proc.start()
            proc.join()
            # self.result = ants.registration(fixed_stack_ants, moving_stack_ants, **settings)


class ANTsLog(object):
    """
    Re-directs the ANTs verbose output to a textfile
    Method derived from Maximilian Hoffman's solution:
    https://github.com/ANTsX/ANTsPy/issues/130
    """

    def __init__(self, log_fpath):
        self.log_fpath = log_fpath
        self.redirect()

    def redirect(self):
        # get file descriptor to __stdout__ (__stdout__ used instead of stdout,
        # because JupyterLab modifies stdout)
        self.orig_stdout_fd = sys.__stdout__.fileno()
        # Duplicate file descriptor to __stdout__
        self.saved_stdout_fd = os.dup(self.orig_stdout_fd)
        # create logfile and redirect __stdout__
        # Log should be unique for each unique registration,
        # so file is always newly-created
        self.log = open(self.log_fpath, "wb")
        os.dup2(self.log.fileno(), self.orig_stdout_fd)

    def revert(self):
        """
        Put everything back as it was
        """
        # Close log
        self.log.close()
        # reset __stdout__
        os.dup2(self.saved_stdout_fd, self.orig_stdout_fd)

    def __enter__(self):
        # enter method added so clas can be used
        # with context manager
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # exit method added so clas can be used
        # with context manager
        self.revert()
